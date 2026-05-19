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
from detector2026.core.utils.preprocess import preprocessing_num_channels

DEFAULT_DEVICE = 'cuda:1'
Resolution = Tuple[int, int]

DEFAULT_EPOCHS = 50
DEFAULT_LR = 1e-4
DEFAULT_MINIMUM_POSSIBLE_CANDIDATES = 7


def output_name_for_one2one(source_name: str, loss_type: str) -> str:
    return f"{source_name}_one2one_{loss_type}"


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
) -> YOLOv11:
    return YOLOv11(
        output_dir=output_dir,
        num_classes=num_classes,
        reg_max=reg_max,
        device=device,
        input_canals=input_channels,
        width_mult=YOLO11_WIDTH_MULT[scale],
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
    )
    try:
        model.load_weights(str(weights_path), device=model.device, eval_mode=False)
        model.train_one2one_head_only(
            minimum_possible_candidates=args.minimum_possible_candidates,
            sync_from_one2many=not args.no_sync_from_one2many,
            loss_type=loss_type,
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


def write_benchmark_plots(benchmark_root: Path, source_name: str, losses: Iterable[str]):
    logs = {}
    for loss_type in losses:
        log_path = benchmark_root / output_name_for_one2one(source_name, loss_type) / "train_log.csv"
        log = _read_train_log(log_path)
        if log:
            logs[loss_type] = log
        else:
            print(f"[WARN] missing or empty log for one2one {loss_type}: {log_path}")

    if not logs:
        print("[WARN] no train_log.csv found; benchmark plots were not generated.")
        return

    plot_dir = benchmark_root / f"{source_name}_one2one_benchmark_plots"
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
    print("\nPlanned experiments:")
    for loss_type in losses:
        output_dir = benchmark_root / output_name_for_one2one(source_name, loss_type)
        status = "RUN" if args.overwrite or not output_dir.exists() else "SKIP"
        print(f"  [{status}] one2one {loss_type} -> {output_dir}")

    if args.dry_run:
        return

    for loss_type in losses:
        run_one2one_job(
            loss_type=loss_type,
            output_dir=benchmark_root / output_name_for_one2one(source_name, loss_type),
            weights_path=weights_path,
            args=args,
            input_channels=input_channels,
            res_hw=res_hw,
        )

    write_benchmark_plots(benchmark_root, source_name, losses)


if __name__ == "__main__":
    main()
