from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from detector2026.core.models.yolov11_ablation import (
    YOLOv11RTDETR,
    YOLOv11RTDETRHead,
    YOLOv11TransformerNeck,
)
from detector2026.core.utils.dataset import YOLODatasetSpecificRes
from detector2026.core.utils.dataset import load_class_index_to_name
from detector2026.core.utils.divers import xywh2xyxy
from detector2026.core.utils.metrics import box_iou
from detector2026.core.utils.preprocess import preprocessing_num_channels


DEFAULT_DATA_DIR = "/data/RAWSIM/RMA/rf_dataset_for_real_validation"
DEFAULT_OUTPUT_ROOT = "/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation/yolov11n_ablation_complete"
DEFAULT_YOLOV11_BEST = (
    "/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation/"
    "yolov11n_specificres_cfg512/best.pt"
)


def parse_res_hw(value: str) -> tuple[int, int]:
    if "," in value:
        left, right = value.split(",", 1)
    elif "x" in value.lower():
        left, right = value.lower().split("x", 1)
    else:
        raise argparse.ArgumentTypeError("Resolution must be formatted as H,W or HxW.")
    return int(left), int(right)


def parse_args():
    parser = argparse.ArgumentParser(description="Complete YOLOv11n ablation suite.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--yolov11-best", default=DEFAULT_YOLOV11_BEST)

    parser.add_argument("--res-key", default="cfg512")
    parser.add_argument("--res-hw", type=parse_res_hw, default=(256, 256))
    parser.add_argument("--preprocessing", default="none")
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--reg-max", type=int, default=16)
    parser.add_argument("--width-mult", type=float, default=0.25)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--one2one-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--one2one-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--one2one-lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--one2one-patience", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--full-eval-every", type=int, default=5)
    parser.add_argument("--save-last-every", type=int, default=5)

    parser.add_argument("--one2one-device", default="cuda:0")
    parser.add_argument("--rtdetr-device", default="cuda:0")
    parser.add_argument("--transformer-neck-device", default="cuda:0")

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--num-decoder-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-decoder-points", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--transformer-neck-d-model", type=int, default=128)
    parser.add_argument("--transformer-neck-num-heads", type=int, default=4)
    parser.add_argument("--transformer-neck-num-layers", type=int, default=1)
    parser.add_argument("--transformer-neck-num-points", type=int, default=4)
    parser.add_argument("--transformer-neck-ffn-ratio", type=float, default=2.0)
    parser.add_argument("--transformer-neck-dropout", type=float, default=0.0)
    parser.add_argument("--transformer-neck-residual-scale", type=float, default=0.0)

    parser.add_argument("--skip-one2one-deformable", action="store_true")
    parser.add_argument("--skip-rtdetr-full", action="store_true")
    parser.add_argument("--skip-transformer-neck", action="store_true")
    parser.add_argument("--skip-transformer-neck-deformable", action="store_true")
    parser.add_argument("--skip-comparison", action="store_true")
    parser.add_argument("--num-visual-examples", type=int, default=10)
    parser.add_argument("--visual-score-threshold", type=float, default=0.05)
    parser.add_argument("--visual-top-k", type=int, default=30)
    parser.add_argument("--visual-cmap", default="viridis")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def output_exists(output_dir: Path) -> bool:
    return (output_dir / "best.pt").exists() or (output_dir / "last.pt").exists() or (output_dir / "train_log.csv").exists()


def should_skip(name: str, output_dir: Path, overwrite: bool) -> bool:
    if output_exists(output_dir) and not overwrite:
        print(f"[SKIP] {name}: output already exists at {output_dir}")
        return True
    return False


def fit_yolo_model(model, args, *, epochs, batch_size, lr, patience):
    model.fit(
        data_dir=args.data_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        dataset="specificres",
        preprocessing=args.preprocessing,
        select_res={"res_hw": args.res_hw, "res_key": args.res_key},
        num_workers=args.num_workers,
        full_eval_every=args.full_eval_every,
        save_last_every=args.save_last_every,
        monitor="val_loss",
        run_full_eval=True,
    )


def build_rtdetr_head_ablation(args, output_dir: Path, *, device: str, input_channels: int, deformable: bool):
    return YOLOv11RTDETRHead(
        output_dir=str(output_dir),
        num_classes=args.num_classes,
        reg_max=args.reg_max,
        device=device,
        input_canals=input_channels,
        width_mult=args.width_mult,
        hidden_dim=args.hidden_dim,
        num_queries=args.num_queries,
        num_decoder_layers=args.num_decoder_layers,
        num_heads=args.num_heads,
        num_decoder_points=args.num_decoder_points,
        use_deformable_attention=deformable,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )


def run_one2one_rtdetr_head(args, input_channels: int):
    name = "yolov11n_best_ft_rtdetr_head_one2one_deformable"
    output_dir = Path(args.output_root) / name
    if args.skip_one2one_deformable:
        return
    if should_skip(name, output_dir, args.overwrite):
        return

    weights_path = Path(args.yolov11_best)
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)

    print(f"\n[RUN] {name} on {args.one2one_device}")
    print(f"      loading YOLOv11 weights: {weights_path}")
    model = build_rtdetr_head_ablation(
        args,
        output_dir,
        device=args.one2one_device,
        input_channels=input_channels,
        deformable=True,
    )
    try:
        missing, unexpected = model.load_yolov11_weights(str(weights_path), device=args.one2one_device, eval_mode=False)
        print(f"      loaded compatible weights; missing={len(missing)} unexpected={len(unexpected)}")
        model.train_one2one_head_only(sync_from_one2many=True)
        model.fit(
            data_dir=args.data_dir,
            epochs=args.one2one_epochs,
            batch_size=args.one2one_batch_size,
            lr=args.one2one_lr,
            patience=args.one2one_patience,
            dataset="specificres",
            preprocessing=args.preprocessing,
            select_res={"res_hw": args.res_hw, "res_key": args.res_key},
            num_workers=args.num_workers,
            save_last_every=args.save_last_every,
            full_eval_every=args.full_eval_every,
            monitor="val_loss",
            run_full_eval=True,
        )
    finally:
        del model
        cleanup()


def run_full_rtdetr(args, input_channels: int):
    name = "yolov11n_rtdetr_yolov11_backbone_full_train"
    output_dir = Path(args.output_root) / name
    if args.skip_rtdetr_full or should_skip(name, output_dir, args.overwrite):
        return
    print(f"\n[RUN] {name} on {args.rtdetr_device}")
    model = YOLOv11RTDETR(
        output_dir=str(output_dir),
        num_classes=args.num_classes,
        reg_max=args.reg_max,
        device=args.rtdetr_device,
        input_canals=input_channels,
        width_mult=args.width_mult,
        hidden_dim=args.hidden_dim,
        num_queries=args.num_queries,
        num_decoder_layers=args.num_decoder_layers,
        num_heads=args.num_heads,
        num_decoder_points=args.num_decoder_points,
        use_deformable_attention=True,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )
    try:
        model.fit(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.one2one_batch_size,
            lr=args.one2one_lr,
            patience=args.patience,
            dataset="specificres",
            preprocessing=args.preprocessing,
            select_res={"res_hw": args.res_hw, "res_key": args.res_key},
            num_workers=args.num_workers,
            save_last_every=args.save_last_every,
            full_eval_every=args.full_eval_every,
            monitor="val_loss",
            run_full_eval=True,
        )
    finally:
        del model
        cleanup()


def run_transformer_neck(args, input_channels: int):
    name = "yolov11n_transformer_neck_full_train"
    output_dir = Path(args.output_root) / name
    if args.skip_transformer_neck or should_skip(name, output_dir, args.overwrite):
        return
    print(f"\n[RUN] {name} on {args.transformer_neck_device}")
    model = YOLOv11TransformerNeck(
        output_dir=str(output_dir),
        num_classes=args.num_classes,
        reg_max=args.reg_max,
        device=args.transformer_neck_device,
        input_canals=input_channels,
        width_mult=args.width_mult,
        transformer_d_model=args.transformer_neck_d_model,
        transformer_num_heads=args.transformer_neck_num_heads,
        transformer_num_layers=args.transformer_neck_num_layers,
        transformer_ffn_ratio=args.transformer_neck_ffn_ratio,
        transformer_dropout=args.transformer_neck_dropout,
        transformer_residual_scale=args.transformer_neck_residual_scale,
        transformer_neck_type="dense",
        transformer_num_points=args.transformer_neck_num_points,
    )
    try:
        fit_yolo_model(model, args, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience)
    finally:
        del model
        cleanup()


def run_transformer_neck_deformable(args, input_channels: int):
    name = "yolov11n_transformer_neck_deformable_full_train"
    output_dir = Path(args.output_root) / name
    if args.skip_transformer_neck_deformable or should_skip(name, output_dir, args.overwrite):
        return
    print(f"\n[RUN] {name} on {args.transformer_neck_device}")
    model = YOLOv11TransformerNeck(
        output_dir=str(output_dir),
        num_classes=args.num_classes,
        reg_max=args.reg_max,
        device=args.transformer_neck_device,
        input_canals=input_channels,
        width_mult=args.width_mult,
        transformer_d_model=args.transformer_neck_d_model,
        transformer_num_heads=args.transformer_neck_num_heads,
        transformer_num_layers=args.transformer_neck_num_layers,
        transformer_ffn_ratio=args.transformer_neck_ffn_ratio,
        transformer_dropout=args.transformer_neck_dropout,
        transformer_residual_scale=args.transformer_neck_residual_scale,
        transformer_neck_type="deformable",
        transformer_num_points=args.transformer_neck_num_points,
    )
    try:
        fit_yolo_model(model, args, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience)
    finally:
        del model
        cleanup()


def experiment_specs(args, input_channels: int):
    root = Path(args.output_root)
    return [
        {
            "name": "yolov11n_best_ft_rtdetr_head_one2one_deformable",
            "kind": "rtdetr_one2one",
            "output_dir": root / "yolov11n_best_ft_rtdetr_head_one2one_deformable",
            "build": lambda output_dir, device: build_rtdetr_head_ablation(
                args, output_dir, device=device, input_channels=input_channels, deformable=True
            ),
        },
        {
            "name": "yolov11n_transformer_neck_deformable_full_train",
            "kind": "yolo",
            "output_dir": root / "yolov11n_transformer_neck_deformable_full_train",
            "build": lambda output_dir, device: YOLOv11TransformerNeck(
                output_dir=str(output_dir),
                num_classes=args.num_classes,
                reg_max=args.reg_max,
                device=device,
                input_canals=input_channels,
                width_mult=args.width_mult,
                transformer_d_model=args.transformer_neck_d_model,
                transformer_num_heads=args.transformer_neck_num_heads,
                transformer_num_layers=args.transformer_neck_num_layers,
                transformer_ffn_ratio=args.transformer_neck_ffn_ratio,
                transformer_dropout=args.transformer_neck_dropout,
                transformer_residual_scale=args.transformer_neck_residual_scale,
                transformer_neck_type="deformable",
                transformer_num_points=args.transformer_neck_num_points,
            ),
        },
        {
            "name": "yolov11n_rtdetr_yolov11_backbone_full_train",
            "kind": "rtdetr_full",
            "output_dir": root / "yolov11n_rtdetr_yolov11_backbone_full_train",
            "build": lambda output_dir, device: YOLOv11RTDETR(
                output_dir=str(output_dir),
                num_classes=args.num_classes,
                reg_max=args.reg_max,
                device=device,
                input_canals=input_channels,
                width_mult=args.width_mult,
                hidden_dim=args.hidden_dim,
                num_queries=args.num_queries,
                num_decoder_layers=args.num_decoder_layers,
                num_heads=args.num_heads,
                num_decoder_points=args.num_decoder_points,
                use_deformable_attention=True,
                dim_feedforward=args.dim_feedforward,
                dropout=args.dropout,
            ),
        },
        {
            "name": "yolov11n_transformer_neck_full_train",
            "kind": "yolo",
            "output_dir": root / "yolov11n_transformer_neck_full_train",
            "build": lambda output_dir, device: YOLOv11TransformerNeck(
                output_dir=str(output_dir),
                num_classes=args.num_classes,
                reg_max=args.reg_max,
                device=device,
                input_canals=input_channels,
                width_mult=args.width_mult,
                transformer_d_model=args.transformer_neck_d_model,
                transformer_num_heads=args.transformer_neck_num_heads,
                transformer_num_layers=args.transformer_neck_num_layers,
                transformer_ffn_ratio=args.transformer_neck_ffn_ratio,
                transformer_dropout=args.transformer_neck_dropout,
                transformer_residual_scale=args.transformer_neck_residual_scale,
                transformer_neck_type="dense",
                transformer_num_points=args.transformer_neck_num_points,
            ),
        },
    ]


def read_log_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def best_metric_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    valid = [row for row in rows if not math.isnan(as_float(row.get("map50_95")))]
    if valid:
        return max(valid, key=lambda row: as_float(row.get("map50_95")))
    return rows[-1] if rows else None


def checkpoint_path(output_dir: Path) -> Path | None:
    for name in ("best.pt", "last.pt"):
        path = output_dir / name
        if path.exists():
            return path
    return None


def load_model_for_comparison(spec, args, device="cpu"):
    model = spec["build"](spec["output_dir"], device)
    ckpt = checkpoint_path(spec["output_dir"])
    if ckpt is None:
        return None, None
    if hasattr(model, "load_weights"):
        model.load_weights(str(ckpt), device=device, eval_mode=True)
    else:
        model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
        model.eval()
    if spec["kind"] == "rtdetr_one2one" and hasattr(model, "use_one2one_head"):
        model.use_one2one_head()
    model.to(model.device)
    model.eval()
    return model, ckpt


def compute_cost(model, args, input_channels: int):
    params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    try:
        from thop import profile

        dummy = torch.randn(1, input_channels, args.res_hw[0], args.res_hw[1], device=model.device)
        with torch.no_grad():
            macs, thop_params = profile(model, inputs=(dummy,), verbose=False)
        return {
            "params": params,
            "trainable_params": trainable_params,
            "thop_params": int(thop_params),
            "macs": int(macs),
            "flops": int(2 * macs),
            "cost_error": "",
        }
    except Exception as exc:
        return {
            "params": params,
            "trainable_params": trainable_params,
            "thop_params": "",
            "macs": "",
            "flops": "",
            "cost_error": str(exc),
        }


def save_comparison_summary(specs, args, input_channels: int, comparison_dir: Path):
    rows = []
    for spec in specs:
        log_rows = read_log_rows(spec["output_dir"] / "train_log.csv")
        best = best_metric_row(log_rows)
        last = log_rows[-1] if log_rows else {}
        model, ckpt = load_model_for_comparison(spec, args, device="cpu")
        costs = compute_cost(model, args, input_channels) if model is not None else {
            "params": "", "trainable_params": "", "thop_params": "", "macs": "", "flops": "", "cost_error": "missing checkpoint"
        }
        rows.append(
            {
                "model": spec["name"],
                "output_dir": str(spec["output_dir"]),
                "checkpoint": str(ckpt) if ckpt else "",
                "epochs_logged": len(log_rows),
                "best_epoch": best.get("epoch", "") if best else "",
                "best_map50": best.get("map50", "") if best else "",
                "best_map50_95": best.get("map50_95", "") if best else "",
                "best_avg_recall_low_snr": best.get("avg_recall_low_snr", "") if best else "",
                "best_avg_recall_medium_snr": best.get("avg_recall_medium_snr", "") if best else "",
                "best_avg_recall_high_snr": best.get("avg_recall_high_snr", "") if best else "",
                "last_epoch": last.get("epoch", ""),
                "last_train_loss": last.get("train_loss", ""),
                "last_val_loss": last.get("val_loss", ""),
                "last_map50": last.get("map50", ""),
                "last_map50_95": last.get("map50_95", ""),
                **costs,
            }
        )
        del model
        cleanup()

    path = comparison_dir / "ablation_comparison_summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[COMPARISON] summary saved to {path}")
    return rows


def plot_comparison_curves(specs, rows, comparison_dir: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable; comparison plots skipped: {exc}")
        return

    for metric in ("map50", "map50_95", "val_loss"):
        plt.figure(figsize=(10, 6))
        has_data = False
        for spec in specs:
            log_rows = read_log_rows(spec["output_dir"] / "train_log.csv")
            epochs = [as_float(row.get("epoch")) for row in log_rows]
            values = [as_float(row.get(metric)) for row in log_rows]
            valid = [(epoch, value) for epoch, value in zip(epochs, values) if not math.isnan(epoch) and not math.isnan(value)]
            if not valid:
                continue
            plt.plot([item[0] for item in valid], [item[1] for item in valid], marker="o", label=spec["name"])
            has_data = True
        if not has_data:
            plt.close()
            continue
        plt.xlabel("epoch")
        plt.ylabel(metric)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        path = comparison_dir / f"{metric}_vs_epochs_all_models.png"
        plt.savefig(path)
        plt.close()
        print(f"[COMPARISON] plot saved to {path}")

    valid_cost = [
        row for row in rows
        if not math.isnan(as_float(row.get("best_map50_95"))) and not math.isnan(as_float(row.get("flops")))
    ]
    if valid_cost:
        for x_key, x_label in (("flops", "FLOPs"), ("params", "Params")):
            plt.figure(figsize=(8, 6))
            for row in valid_cost:
                plt.scatter(as_float(row[x_key]), as_float(row["best_map50_95"]))
                plt.annotate(row["model"], (as_float(row[x_key]), as_float(row["best_map50_95"])), fontsize=7)
            plt.xlabel(x_label)
            plt.ylabel("best mAP50:95")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            path = comparison_dir / f"map50_95_vs_{x_key}.png"
            plt.savefig(path)
            plt.close()
            print(f"[COMPARISON] plot saved to {path}")


def latest_metrics_json(output_dir: Path) -> Path | None:
    metrics_dir = output_dir / "metrics"
    if not metrics_dir.exists():
        return None
    paths = sorted(metrics_dir.glob("metrics_epoch_*.json"))
    return paths[-1] if paths else None


def plot_recall_at_fixed_false_alarm(specs, comparison_dir: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable; recall plots skipped: {exc}")
        return

    plt.figure(figsize=(10, 6))
    has_data = False
    for spec in specs:
        path = latest_metrics_json(spec["output_dir"])
        if path is None:
            continue
        with path.open("r") as handle:
            metrics = json.load(handle)
        recall_snr = metrics.get("recall_snr", {}).get("global", {})
        snr_bins = recall_snr.get("snr_bins")
        recall = recall_snr.get("recall")
        if not snr_bins or not recall:
            continue
        xs = [(float(snr_bins[i]) + float(snr_bins[i + 1])) / 2.0 for i in range(len(recall))]
        plt.plot(xs, [float(v) for v in recall], marker="o", label=spec["name"])
        has_data = True
    if not has_data:
        plt.close()
        print("[WARN] No recall_snr/global curves found in metrics JSON; fixed false-alarm recall plot skipped.")
        return
    plt.xlabel("SNR")
    plt.ylabel("Recall at fixed false alarm")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    path = comparison_dir / "recall_at_fixed_false_alarm_vs_snr.png"
    plt.savefig(path)
    plt.close()
    print(f"[COMPARISON] plot saved to {path}")


def _class_name(mapping: dict[int, str], label: int) -> str:
    return mapping.get(int(label), str(int(label)))


def _gt_from_targets(targets: torch.Tensor, image_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows = targets[targets[:, 0].long() == image_index]
    if rows.numel() == 0:
        return torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.long)
    boxes = xywh2xyxy(rows[:, 2:6]).clamp(0.0, 1.0)
    labels = rows[:, 1].long()
    return boxes, labels


def _detections_to_normalized(
    detection: torch.Tensor,
    *,
    input_hw: tuple[int, int],
    score_threshold: float,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if detection is None or detection.numel() == 0:
        return torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long)

    det = detection.detach().cpu()
    det = det[det[:, 4] >= float(score_threshold)]
    if det.shape[0] > int(top_k):
        det = det[det[:, 4].argsort(descending=True)[: int(top_k)]]
    if det.numel() == 0:
        return torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long)

    height, width = float(input_hw[0]), float(input_hw[1])
    boxes = torch.stack(
        [
            det[:, 0] / width,
            det[:, 1] / height,
            det[:, 2] / width,
            det[:, 3] / height,
        ],
        dim=1,
    ).clamp(0.0, 1.0)
    return boxes, det[:, 4], det[:, 5].long()


def _draw_visual_panel(
    ax,
    *,
    image: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    class_names: dict[int, str],
    title: str,
    cmap: str,
):
    import matplotlib.patches as patches

    image = image.detach().cpu()
    height, width = image.shape[-2:]
    image_2d = image[0] if image.ndim == 3 else image.squeeze(0)
    ax.imshow(image_2d.numpy(), aspect="auto", cmap=cmap)
    ax.set_title(title, fontsize=9)
    ax.axis("off")

    for gt_index, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
        x1, y1, x2, y2 = box.tolist()
        ax.add_patch(
            patches.Rectangle(
                (x1 * width, y1 * height),
                (x2 - x1) * width,
                (y2 - y1) * height,
                linewidth=2,
                edgecolor="lime",
                facecolor="none",
            )
        )
        ax.text(
            x1 * width,
            y1 * height,
            f"GT {gt_index}: {_class_name(class_names, int(label))}",
            color="black",
            fontsize=7,
            bbox=dict(facecolor="lime", alpha=0.75, pad=1),
        )

    pairwise_iou = (
        box_iou(pred_boxes, gt_boxes)
        if pred_boxes.numel() and gt_boxes.numel()
        else torch.zeros((len(pred_boxes), len(gt_boxes)))
    )
    best_iou = pairwise_iou.max(dim=1).values if pairwise_iou.numel() else torch.zeros((len(pred_boxes),))
    best_gt = pairwise_iou.argmax(dim=1) if pairwise_iou.numel() else torch.full((len(pred_boxes),), -1, dtype=torch.long)

    for pred_index, (box, score, label) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
        x1, y1, x2, y2 = box.tolist()
        color = "cyan" if best_iou[pred_index] >= 0.5 else "red"
        ax.add_patch(
            patches.Rectangle(
                (x1 * width, y1 * height),
                (x2 - x1) * width,
                (y2 - y1) * height,
                linewidth=1.2,
                edgecolor=color,
                facecolor="none",
                linestyle="--",
            )
        )
        gt_suffix = f", gt={int(best_gt[pred_index])}" if int(best_gt[pred_index]) >= 0 else ""
        ax.text(
            x1 * width,
            max(0.0, y1 * height - 4.0),
            f"P{pred_index}: {_class_name(class_names, int(label))} {float(score):.2f}, IoU={float(best_iou[pred_index]):.2f}{gt_suffix}",
            color="white",
            fontsize=6,
            bbox=dict(facecolor=color, alpha=0.75, pad=1),
        )


def save_visual_examples(specs, args, input_channels: int, comparison_dir: Path):
    import matplotlib.pyplot as plt

    visual_dir = comparison_dir / "visual_examples"
    visual_dir.mkdir(parents=True, exist_ok=True)
    class_names = load_class_index_to_name(args.data_dir)
    dataset = YOLODatasetSpecificRes(
        data_dir=os.path.join(args.data_dir, "val/data"),
        labels_dir=os.path.join(args.data_dir, "val/labels_detect"),
        res_hw=args.res_hw,
        res_key=args.res_key,
        preprocessing=args.preprocessing,
    )
    loader = DataLoader(dataset, batch_size=min(args.num_visual_examples, max(1, len(dataset))), shuffle=False, collate_fn=YOLODatasetSpecificRes.collate_fn)
    try:
        imgs, targets, _ = next(iter(loader))
    except StopIteration:
        print("[WARN] Empty validation dataset; visual examples skipped.")
        return

    models = {}
    for spec in specs:
        model, ckpt = load_model_for_comparison(spec, args, device="cpu")
        if model is None:
            continue
        models[spec["name"]] = (model, ckpt)

    if not models:
        print("[WARN] No checkpoints found; visual examples skipped.")
        return

    batch_predictions = {}
    with torch.no_grad():
        for name, (model, _) in models.items():
            imgs_device = imgs.to(model.device, dtype=torch.float32)
            outputs = model(imgs_device)
            if isinstance(outputs, dict):
                detections = model.postprocess_for_metrics(
                    outputs,
                    conf_threshold=args.visual_score_threshold,
                    max_det=args.visual_top_k,
                )
            else:
                dist_out, cls_out = outputs
                detections = model.postprocess(
                    dist_out,
                    cls_out,
                    dist_out,
                    conf_thres=args.visual_score_threshold,
                    max_det=args.visual_top_k,
                )
            batch_predictions[name] = detections

    summaries = []
    model_names = list(models.keys())
    num_samples = min(args.num_visual_examples, imgs.shape[0])
    for batch_index in range(num_samples):
        gt_boxes, gt_labels = _gt_from_targets(targets, batch_index)
        fig, axes = plt.subplots(
            1,
            len(model_names),
            figsize=(max(5 * len(model_names), 8), 5),
            sharex=True,
            sharey=True,
        )
        if len(model_names) == 1:
            axes = [axes]

        sample_summary = {
            "sample_index": int(batch_index),
            "n_gt": int(len(gt_boxes)),
            "models": {},
        }
        for ax, name in zip(axes, model_names):
            boxes, scores, labels = _detections_to_normalized(
                batch_predictions[name][batch_index],
                input_hw=args.res_hw,
                score_threshold=args.visual_score_threshold,
                top_k=args.visual_top_k,
            )
            _draw_visual_panel(
                ax,
                image=imgs[batch_index],
                pred_boxes=boxes,
                pred_scores=scores,
                pred_labels=labels,
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
                class_names=class_names,
                title=f"{name} | n={len(boxes)}",
                cmap=args.visual_cmap,
            )
            sample_summary["models"][name] = {
                "checkpoint": str(models[name][1]),
                "n_pred": int(len(boxes)),
                "scores": [float(value) for value in scores.tolist()],
                "labels": [int(value) for value in labels.tolist()],
            }

        output_path = visual_dir / f"sample_{batch_index:05d}.png"
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        summaries.append(sample_summary)

    (visual_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"[COMPARISON] saved {len(summaries)} validation spectrum visualizations to {visual_dir}")
    for model, _ in models.values():
        del model
    cleanup()


def run_final_comparison(args, input_channels: int):
    comparison_dir = Path(args.output_root) / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    specs = experiment_specs(args, input_channels)

    print("\n[COMPARISON] building final ablation comparison")
    rows = save_comparison_summary(specs, args, input_channels, comparison_dir)
    plot_comparison_curves(specs, rows, comparison_dir)
    plot_recall_at_fixed_false_alarm(specs, comparison_dir)
    save_visual_examples(specs, args, input_channels, comparison_dir)


def main():
    args = parse_args()
    input_channels = preprocessing_num_channels(args.preprocessing)

    print("Complete YOLOv11n ablation suite")
    print(f"  data_dir = {args.data_dir}")
    print(f"  output_root = {args.output_root}")
    print(f"  yolov11_best = {args.yolov11_best}")
    print(f"  res_key = {args.res_key}")
    print(f"  res_hw = {args.res_hw}")
    print(f"  preprocessing = {args.preprocessing}")
    print(f"  input_channels = {input_channels}")
    print(f"  full_eval_every = {args.full_eval_every}")
    print(f"  early_stopping_monitor = val_loss")
    print(f"  patience = {args.patience}")
    print(f"  one2one_patience = {args.one2one_patience}")
    print(f"  rtdetr_hidden_dim = {args.hidden_dim}")
    print(f"  rtdetr_num_queries = {args.num_queries}")
    print(f"  rtdetr_num_decoder_layers = {args.num_decoder_layers}")
    print("  experiments:")
    print(f"    - YOLOv11 best -> RTDETR one2one deformable head on {args.one2one_device}")
    print(f"    - transformer neck deformable full training on {args.transformer_neck_device}")
    print(f"    - RTDETR with YOLOv11 backbone and RTDETR hybrid neck on {args.rtdetr_device}")
    print(f"    - transformer neck dense full training on {args.transformer_neck_device}")

    if args.dry_run:
        return

    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    run_one2one_rtdetr_head(args, input_channels)
    run_transformer_neck_deformable(args, input_channels)
    run_full_rtdetr(args, input_channels)
    run_transformer_neck(args, input_channels)
    if not args.skip_comparison:
        run_final_comparison(args, input_channels)


if __name__ == "__main__":
    main()
