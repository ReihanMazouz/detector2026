from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from detector2026.core.models.yolov11 import YOLOv11
from detector2026.core.scripts.train_benchmark_suite import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATA_DIR,
    DEFAULT_DEVICE,
    DEFAULT_FULL_EVAL_EVERY,
    DEFAULT_MONITOR,
    DEFAULT_NUM_CLASSES,
    DEFAULT_NUM_WORKERS,
    DEFAULT_OUTPUT_DIR_PARENT,
    DEFAULT_PATIENCE,
    DEFAULT_PREPROCESSING,
    DEFAULT_REG_MAX,
    DEFAULT_RES_KEYS,
    DEFAULT_SAVE_LAST_EVERY,
    YOLO11_WIDTH_MULT,
    find_input_resolutions,
)
from detector2026.core.utils.dataset import YOLODatasetSpecificRes, load_class_index_to_name
from detector2026.core.utils.evaluate import EvalConfig, EvalRunner
from detector2026.core.utils.preprocess import preprocessing_num_channels

DEFAULT_DEVICE = 'cuda:1'
Resolution = Tuple[int, int]

DEFAULT_EPOCHS = 50
DEFAULT_LR = 1e-4
DEFAULT_MINIMUM_POSSIBLE_CANDIDATES = 7


def output_name_for_one2one(source_name: str, loss_type: str, suffix: str = "") -> str:
    return f"{source_name}_one2one_{loss_type}{suffix}"


def parse_chin_levels(value: str) -> tuple[str, ...]:
    levels = tuple(level.strip().lower() for level in value.split(",") if level.strip())
    if not levels:
        raise argparse.ArgumentTypeError("At least one transformer chin level is required.")
    invalid = [level for level in levels if level not in {"p3", "p4", "p5"}]
    if invalid:
        raise argparse.ArgumentTypeError(f"Invalid transformer chin levels: {invalid}. Use p3,p4,p5.")
    return levels


def transformer_chin_suffix(args) -> str:
    if not args.use_transformer_chin:
        return ""
    levels = "".join(args.chin_levels)
    return f"_chin_{levels}_d{args.chin_d_model}_l{args.chin_num_layers}"


def default_weights_from_source_run(source_run_dir: str | None) -> Path | None:
    if not source_run_dir:
        return None
    source_dir = Path(source_run_dir)
    for filename in ("best.pt", "last.pt"):
        candidate = source_dir / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No best.pt or last.pt found in {source_dir}")


def infer_source_name(weights_path: Path, source_run_dir: str | None) -> str:
    if source_run_dir:
        return Path(source_run_dir).name
    return weights_path.stem


def build_yolov11(
    output_dir: str,
    scale: str,
    input_channels: int,
    device: str,
    num_classes: int,
    reg_max: int,
    args=None,
) -> YOLOv11:
    chin_kwargs = {}
    if args is not None:
        chin_kwargs = {
            "use_transformer_chin": args.use_transformer_chin,
            "chin_levels": args.chin_levels,
            "chin_d_model": args.chin_d_model,
            "chin_num_heads": args.chin_num_heads,
            "chin_num_layers": args.chin_num_layers,
            "chin_ffn_ratio": args.chin_ffn_ratio,
            "chin_dropout": args.chin_dropout,
            "chin_residual_scale": args.chin_residual_scale,
        }
    return YOLOv11(
        output_dir=output_dir,
        num_classes=num_classes,
        reg_max=reg_max,
        device=device,
        input_canals=input_channels,
        width_mult=YOLO11_WIDTH_MULT[scale],
        **chin_kwargs,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load a trained YOLOv11 checkpoint, freeze the model, and train only the one2one head "
            "on the same specific-resolution dataset convention used by train_benchmark_suite.py."
        )
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Dataset root directory.")
    parser.add_argument(
        "--source-run-dir",
        default=None,
        help="Directory of the trained YOLOv11 run. best.pt is preferred, then last.pt.",
    )
    parser.add_argument("--weights", default='/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation/yolov11n_specificres_cfg512/best.pt', help="Explicit path to a trained YOLOv11 checkpoint.")
    parser.add_argument(
        "--output-dir-parent",
        default=DEFAULT_OUTPUT_DIR_PARENT,
        help="Parent directory for the one2one benchmark experiment folders.",
    )
    parser.add_argument(
        "--output-dir",
        default='/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation/one2one_head',
        help="Explicit benchmark root directory. Loss-specific folders are created inside it.",
    )
    parser.add_argument("--scale", choices=sorted(YOLO11_WIDTH_MULT.keys()), default="n")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--num-classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--reg-max", type=int, default=DEFAULT_REG_MAX)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--preprocessing", default=DEFAULT_PREPROCESSING)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--full-eval-every", type=int, default=DEFAULT_FULL_EVAL_EVERY)
    parser.add_argument("--save-last-every", type=int, default=DEFAULT_SAVE_LAST_EVERY)
    parser.add_argument("--monitor", default=DEFAULT_MONITOR)
    parser.add_argument("--res-key", default=DEFAULT_RES_KEYS[0])
    parser.add_argument(
        "--minimum-possible-candidates",
        type=int,
        default=DEFAULT_MINIMUM_POSSIBLE_CANDIDATES,
    )
    parser.add_argument(
        "--negative-to-positive-ratio",
        type=float,
        default=1.0,
        help="Hungarian one2one classification weight ratio between negatives and positives.",
    )
    parser.add_argument(
        "--use-transformer-chin",
        action="store_true",
        help="Insert a lightweight transformer chin before the one2one head.",
    )
    parser.add_argument("--chin-levels", type=parse_chin_levels, default=("p4", "p5"))
    parser.add_argument("--chin-d-model", type=int, default=128)
    parser.add_argument("--chin-num-heads", type=int, default=4)
    parser.add_argument("--chin-num-layers", type=int, default=1)
    parser.add_argument("--chin-ffn-ratio", type=float, default=2.0)
    parser.add_argument("--chin-dropout", type=float, default=0.0)
    parser.add_argument("--chin-residual-scale", type=float, default=0.0)
    parser.add_argument(
        "--losses",
        nargs="+",
        choices=("tal", "hungarian"),
        default=("tal", "hungarian"),
        help="One2one losses to benchmark, in execution order.",
    )
    parser.add_argument(
        "--no-sync-from-one2many",
        action="store_true",
        help="Do not re-copy one2many head weights before one2one training.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Run even if the output directory already exists.",
    )
    parser.add_argument(
        "--diagnose-initial",
        action="store_true",
        help="Before training, evaluate one2many+NMS, copied one2one+NMS, and copied one2one+noNMS.",
    )
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="Run the initial diagnostic and exit without training.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print resolved configuration and exit.")
    return parser.parse_args()


def cleanup_after_run():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_one2one_job(
    *,
    loss_type: str,
    output_dir: Path,
    weights_path: Path,
    args,
    input_channels: int,
    res_hw: Resolution,
):
    if output_dir.exists() and not args.overwrite:
        print(f"[SKIP] one2one {loss_type}: output_dir already exists: {output_dir}")
        return

    print(f"\n[RUN ] one2one {loss_type}")
    print(f"       output_dir = {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_yolov11(
        output_dir=str(output_dir),
        scale=args.scale,
        input_channels=input_channels,
        device=args.device,
        num_classes=args.num_classes,
        reg_max=args.reg_max,
        args=args,
    )
    try:
        model.load_weights(str(weights_path), device=model.device, eval_mode=False)
        model.train_one2one_head_only(
            minimum_possible_candidates=args.minimum_possible_candidates,
            sync_from_one2many=not args.no_sync_from_one2many,
            loss_type=loss_type,
            negative_to_positive_ratio=args.negative_to_positive_ratio,
        )

        model.fit(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            dataset="specificres",
            preprocessing=args.preprocessing,
            select_res={"res_hw": res_hw, "res_key": args.res_key},
            num_workers=args.num_workers,
            full_eval_every=args.full_eval_every,
            save_last_every=args.save_last_every,
            monitor=args.monitor,
            run_full_eval=True,
        )
    finally:
        del model
        cleanup_after_run()


def build_val_loader(args, res_hw: Resolution):
    val_dataset = YOLODatasetSpecificRes(
        data_dir=os.path.join(args.data_dir, "val/data"),
        labels_dir=os.path.join(args.data_dir, "val/labels_detect"),
        res_hw=res_hw,
        res_key=args.res_key,
        preprocessing=args.preprocessing,
    )
    num_workers = 0 if args.num_workers is None else max(0, int(args.num_workers))
    return DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available() and str(args.device).startswith("cuda"),
        collate_fn=val_dataset.collate_fn,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )


def report_checkpoint_compatibility(model: YOLOv11, weights_path: Path):
    state_dict = torch.load(weights_path, map_location=model.device)
    model_state = model.state_dict()
    compatible = [
        key for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    ]
    incompatible = [
        key for key, value in state_dict.items()
        if key in model_state and model_state[key].shape != value.shape
    ]
    unexpected = [key for key in state_dict if key not in model_state]
    missing_after_load = [key for key in model_state if key not in state_dict]
    print("\n[DIAG] Checkpoint compatibility")
    print(f"       checkpoint = {weights_path}")
    print(f"       compatible keys = {len(compatible)} / {len(model_state)} model keys")
    print(f"       incompatible shape keys = {len(incompatible)}")
    print(f"       unexpected checkpoint keys = {len(unexpected)}")
    print(f"       missing model keys before partial load = {len(missing_after_load)}")
    if incompatible[:10]:
        print("       first incompatible keys:")
        for key in incompatible[:10]:
            print(f"         - {key}: checkpoint={tuple(state_dict[key].shape)} model={tuple(model_state[key].shape)}")
    return compatible, incompatible, unexpected, missing_after_load


def summarize_eval_result(label: str, result: dict):
    values = result["extra_values"]
    print(
        f"[DIAG] {label}: "
        f"mAP50={values[0]} | "
        f"mAP50_95={values[1]} | "
        f"avgRec(low/med/high)={values[2]}/{values[3]}/{values[4]}"
    )


def run_initial_diagnostic(
    *,
    benchmark_root: Path,
    source_name: str,
    weights_path: Path,
    args,
    input_channels: int,
    res_hw: Resolution,
):
    print("\n[DIAG] Running initial one2one sanity checks")
    diag_dir = benchmark_root / f"{source_name}_one2one_initial_diagnostic"
    diag_dir.mkdir(parents=True, exist_ok=True)

    val_loader = build_val_loader(args, res_hw)
    runner = EvalRunner(
        output_dir=str(diag_dir),
        cfg=EvalConfig(iou_thresh=0.5, fa_target=0.01, img_size=res_hw),
        class_index_to_name=load_class_index_to_name(args.data_dir),
    )

    model = build_yolov11(
        output_dir=str(diag_dir),
        scale=args.scale,
        input_channels=input_channels,
        device=args.device,
        num_classes=args.num_classes,
        reg_max=args.reg_max,
        args=args,
    )
    try:
        report_checkpoint_compatibility(model, weights_path)
        missing, unexpected = model.load_weights(str(weights_path), device=model.device, eval_mode=False)
        print(f"[DIAG] load_weights missing keys after partial load = {len(missing)}")
        print(f"[DIAG] load_weights unexpected keys = {len(unexpected)}")

        model.use_one2many_head()
        model.eval()
        result = runner.run(epoch=0, model=model, val_loader=val_loader)
        summarize_eval_result("one2many + NMS", result)

        model.sync_one2one_from_one2many()
        model.use_one2one_head()
        model.eval()

        original_postprocess = model.postprocess

        def one2one_with_nms(dist_out, cls_out, feats, conf_thres=0.1, iou_thres=0.1, iou_same_box=0.9, without_nms=None, max_det=300):
            return original_postprocess(
                dist_out,
                cls_out,
                feats,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                iou_same_box=iou_same_box,
                without_nms=False,
                max_det=max_det,
            )

        model.postprocess = one2one_with_nms
        result = runner.run(epoch=1, model=model, val_loader=val_loader)
        summarize_eval_result("copied one2one + NMS", result)

        def one2one_without_nms(dist_out, cls_out, feats, conf_thres=0.1, iou_thres=0.1, iou_same_box=0.9, without_nms=None, max_det=300):
            return original_postprocess(
                dist_out,
                cls_out,
                feats,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                iou_same_box=iou_same_box,
                without_nms=True,
                max_det=max_det,
            )

        model.postprocess = one2one_without_nms
        result = runner.run(epoch=2, model=model, val_loader=val_loader)
        summarize_eval_result("copied one2one + noNMS", result)
        model.postprocess = original_postprocess
    finally:
        del model
        cleanup_after_run()


def _read_train_log(log_path: Path) -> dict[str, list[float]]:
    columns: dict[str, list[float]] = {}
    if not log_path.exists():
        return columns
    with open(log_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return columns
        columns = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in reader.fieldnames:
                value = row.get(name, "")
                try:
                    columns[name].append(float(value))
                except (TypeError, ValueError):
                    columns[name].append(float("nan"))
    return columns


def _plot_columns(
    *,
    logs: dict[str, dict[str, list[float]]],
    columns: list[str],
    ylabel: str,
    title: str,
    save_path: Path,
):
    plt.figure(figsize=(10, 6))
    has_data = False
    for loss_type, log in logs.items():
        epochs = log.get("epoch", [])
        if not epochs:
            continue
        for column in columns:
            values = log.get(column, [])
            if not values:
                continue
            plt.plot(epochs[:len(values)], values, label=f"{loss_type} {column}")
            has_data = True
    if not has_data:
        plt.close()
        return
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[PLOT] saved {save_path}")


def write_benchmark_plots(benchmark_root: Path, source_name: str, losses: Iterable[str], suffix: str = ""):
    logs = {}
    for loss_type in losses:
        log_path = benchmark_root / output_name_for_one2one(source_name, loss_type, suffix) / "train_log.csv"
        log = _read_train_log(log_path)
        if log:
            logs[loss_type] = log
        else:
            print(f"[WARN] missing or empty log for one2one {loss_type}: {log_path}")

    if not logs:
        print("[WARN] no train_log.csv found; benchmark plots were not generated.")
        return

    plot_dir = benchmark_root / f"{source_name}_one2one_benchmark_plots{suffix}"
    _plot_columns(
        logs=logs,
        columns=["train_loss", "val_loss"],
        ylabel="Loss",
        title="YOLOv11 one2one loss comparison",
        save_path=plot_dir / "loss_comparison.png",
    )
    _plot_columns(
        logs=logs,
        columns=["loss_box_train", "loss_cls_train", "loss_dfl_train"],
        ylabel="Train loss component",
        title="YOLOv11 one2one train loss components",
        save_path=plot_dir / "train_loss_components.png",
    )
    _plot_columns(
        logs=logs,
        columns=["loss_box_val", "loss_cls_val", "loss_dfl_val"],
        ylabel="Validation loss component",
        title="YOLOv11 one2one validation loss components",
        save_path=plot_dir / "val_loss_components.png",
    )
    _plot_columns(
        logs=logs,
        columns=["map50", "map50_95"],
        ylabel="mAP",
        title="YOLOv11 one2one mAP comparison",
        save_path=plot_dir / "map_comparison.png",
    )
    _plot_columns(
        logs=logs,
        columns=["avg_recall_low_snr", "avg_recall_medium_snr", "avg_recall_high_snr"],
        ylabel="Average recall",
        title="YOLOv11 one2one average recall comparison",
        save_path=plot_dir / "avg_recall_comparison.png",
    )


def main():
    args = parse_args()
    weights_path = Path(args.weights) if args.weights else default_weights_from_source_run(args.source_run_dir)
    if weights_path is None:
        raise ValueError("Provide either --weights or --source-run-dir.")
    if not weights_path.exists():
        raise FileNotFoundError(weights_path)

    input_resolutions: List[Resolution] = find_input_resolutions(args.data_dir)
    res_keys = list(DEFAULT_RES_KEYS)
    if args.res_key not in res_keys:
        raise ValueError(f"Unknown --res-key '{args.res_key}'. Expected one of {res_keys}.")
    if len(input_resolutions) != len(res_keys):
        raise ValueError(
            f"Expected {len(res_keys)} resolutions in the dataset, found {len(input_resolutions)}: {input_resolutions}"
        )

    res_index = res_keys.index(args.res_key)
    res_hw = input_resolutions[res_index]
    input_channels = preprocessing_num_channels(args.preprocessing)

    source_name = infer_source_name(weights_path, args.source_run_dir)
    benchmark_root = Path(args.output_dir) if args.output_dir else Path(args.output_dir_parent)
    losses: Iterable[str] = tuple(dict.fromkeys(args.losses))
    chin_suffix = transformer_chin_suffix(args)

    print("YOLOv11 one2one head benchmark")
    print(f"  data_dir = {args.data_dir}")
    print(f"  weights = {weights_path}")
    print(f"  output_root = {benchmark_root}")
    print(f"  scale = {args.scale}")
    print(f"  preprocessing = {args.preprocessing}")
    print(f"  input_channels = {input_channels}")
    print(f"  dataset = specificres")
    print(f"  res_key = {args.res_key}")
    print(f"  res_hw = {res_hw}")
    print(f"  epochs = {args.epochs}")
    print(f"  lr = {args.lr}")
    print(f"  losses = {list(losses)}")
    print(f"  minimum_possible_candidates = {args.minimum_possible_candidates}")
    print(f"  negative_to_positive_ratio = {args.negative_to_positive_ratio}")
    print(f"  use_transformer_chin = {args.use_transformer_chin}")
    if args.use_transformer_chin:
        print(f"  chin_levels = {args.chin_levels}")
        print(f"  chin_d_model = {args.chin_d_model}")
        print(f"  chin_num_heads = {args.chin_num_heads}")
        print(f"  chin_num_layers = {args.chin_num_layers}")
    print("\nPlanned experiments:")
    for loss_type in losses:
        output_dir = benchmark_root / output_name_for_one2one(source_name, loss_type, chin_suffix)
        status = "RUN" if args.overwrite or not output_dir.exists() else "SKIP"
        print(f"  [{status}] one2one {loss_type} -> {output_dir}")

    if args.dry_run:
        return

    if args.diagnose_initial or args.diagnostic_only:
        run_initial_diagnostic(
            benchmark_root=benchmark_root,
            source_name=source_name,
            weights_path=weights_path,
            args=args,
            input_channels=input_channels,
            res_hw=res_hw,
        )
        if args.diagnostic_only:
            return

    for loss_type in losses:
        run_one2one_job(
            loss_type=loss_type,
            output_dir=benchmark_root / output_name_for_one2one(source_name, loss_type, chin_suffix),
            weights_path=weights_path,
            args=args,
            input_channels=input_channels,
            res_hw=res_hw,
        )

    write_benchmark_plots(benchmark_root, source_name, losses, chin_suffix)


if __name__ == "__main__":
    main()
