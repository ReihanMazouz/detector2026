from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from detector2026.core.models.mr_yolo import MR_YOLO  # noqa: E402
from detector2026.core.models.mr_yolo_ablation import (  # noqa: E402
    MRYOLOBranchCrossAttentionAblation,
    MRYOLOInputCrossAttentionAblation,
)
from detector2026.core.models.yolov11 import YOLOv11  # noqa: E402
from detector2026.core.models.yolov11_ablation import (  # noqa: E402
    YOLOv11DATBackbone,
    YOLOv11NoNeck,
    YOLOv11P3Direct,
    YOLOv11P3RTDETR,
    YOLOv11RTDETR,
    YOLOv11RTDETRHead,
    YOLOv11SwinBackbone,
    YOLOv11TransformerNeck,
)
from detector2026.core.scripts.train_benchmark_suite import (  # noqa: E402
    DEFAULT_RES_KEYS,
    find_input_resolutions,
)
from detector2026.core.utils.analysing_results import dataset_analysis_with_metrics  # noqa: E402
from detector2026.core.utils.dataset import (  # noqa: E402
    YOLODatasetFusedMultiRes,
    YOLODatasetSpecificRes,
    load_class_index_to_name,
)
from detector2026.core.utils.preprocess import preprocessing_num_channels  # noqa: E402


DEFAULT_DATA_DIR = "/data/RAWSIM/RMA/rf_dataset_for_real_validation"
DEFAULT_OUTPUT_DIR = (
    "/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation/"
    "ablation_fine_comparison"
)
DEFAULT_TRAINING_ROOT = "/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation"
DEFAULT_YOLO_ABLATION_ROOT = (
    "/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation/"
    "yolov11n_ablation"
)
DEFAULT_P3_ROOT = (
    "/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation/"
    "yolov11n_p3_rtdetr_ablation"
)
DEFAULT_MR_ABLATION_ROOT = (
    "/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation/"
    "mr_yolo_ablation"
)


@dataclass(frozen=True)
class EvalSpec:
    name: str
    family: str
    checkpoint: Path
    dataset_mode: str
    builder: Callable[[str], torch.nn.Module]


def parse_hw(value: str) -> tuple[int, int]:
    if "," in value:
        left, right = value.split(",", 1)
    elif "x" in value.lower():
        left, right = value.lower().split("x", 1)
    else:
        raise argparse.ArgumentTypeError("Expected H,W or HxW.")
    return int(left), int(right)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all YOLO/MR-YOLO transformer ablations on common fine metrics."
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--training-root", default=DEFAULT_TRAINING_ROOT)
    parser.add_argument("--yolo-ablation-root", default=DEFAULT_YOLO_ABLATION_ROOT)
    parser.add_argument(
        "--yolo-ablation-complete-root",
        default=(
            "/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation/"
            "yolov11n_ablation_complete"
        ),
        help="Optional fallback root used for ablations only present in the complete run folder.",
    )
    parser.add_argument("--p3-root", default=DEFAULT_P3_ROOT)
    parser.add_argument("--mr-ablation-root", default=DEFAULT_MR_ABLATION_ROOT)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--preprocessing", default="none")
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--reg-max", type=int, default=16)
    parser.add_argument("--width-mult", type=float, default=0.25)
    parser.add_argument("--res-key", default="cfg512")
    parser.add_argument("--res-hw", type=parse_hw, default=(256, 256))
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--false-alarm-target", type=float, default=0.01)
    parser.add_argument("--low-snr", type=parse_hw, default=(-20, 0), help="SNR band H,W style, e.g. -20,0.")
    parser.add_argument("--medium-snr", type=parse_hw, default=(0, 10))
    parser.add_argument("--high-snr", type=parse_hw, default=(10, 20))
    parser.add_argument(
        "--size-thresholds",
        type=float,
        nargs=2,
        default=(0.03, 0.08),
        metavar=("SMALL_MAX", "MEDIUM_MAX"),
        help="Relative min(width,height) thresholds for small/medium/large GT boxes.",
    )
    parser.add_argument("--include-missing", action="store_true", help="Keep missing checkpoints as NaN rows.")
    parser.add_argument("--max-models", type=int, default=0, help="Debug limit. 0 evaluates all specs.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return str(obj)


def avg_recall_between(snr_bins: Sequence[float], recall: Sequence[float], left: float, right: float) -> float:
    bins = np.asarray(snr_bins, dtype=float)
    rec = np.asarray(recall, dtype=float)
    if len(bins) != len(rec) + 1:
        return float("nan")

    left = max(float(left), float(bins[0]))
    right = min(float(right), float(bins[-1]))
    if right <= left:
        return float("nan")

    area = 0.0
    width_total = 0.0
    for idx, value in enumerate(rec):
        bin_left = float(bins[idx])
        bin_right = float(bins[idx + 1])
        overlap_left = max(left, bin_left)
        overlap_right = min(right, bin_right)
        width = max(0.0, overlap_right - overlap_left)
        if width > 0 and not math.isnan(float(value)):
            area += float(value) * width
            width_total += width
    return float(area / width_total) if width_total > 0 else float("nan")


def recall_for_size_band(stats: dict[str, Any], conf_thresh: float, left: float, right: float | None) -> float:
    tp = 0
    fn = 0
    for rec in stats.get("tp", []):
        if float(rec.get("score", 0.0)) < conf_thresh:
            continue
        wh = rec.get("gt_wh")
        if not wh:
            continue
        size = float(min(wh))
        if size >= left and (right is None or size < right):
            tp += 1
    for rec in stats.get("fn", []):
        wh = rec.get("gt_wh")
        if not wh:
            continue
        size = float(min(wh))
        if size >= left and (right is None or size < right):
            fn += 1
    denom = tp + fn
    return float(tp / denom) if denom else float("nan")


def summarize_metrics(
    full_metrics: dict[str, Any],
    stats: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, float]:
    map_stats = full_metrics.get("map_stats", {})
    recall_snr = full_metrics.get("recall_snr", {}).get("global", {})
    snr_bins = recall_snr.get("snr_bins")
    recall_curve = recall_snr.get("recall")
    conf_thresh = float(full_metrics.get("conf_thresh", 0.0))
    small_max, medium_max = map(float, args.size_thresholds)

    return {
        "map50": float(map_stats.get("mAP50", float("nan"))),
        "map50_95": float(map_stats.get("mAP50:95", float("nan"))),
        "conf_thresh": conf_thresh,
        "recall_low_snr": avg_recall_between(snr_bins, recall_curve, *args.low_snr)
        if snr_bins is not None else float("nan"),
        "recall_medium_snr": avg_recall_between(snr_bins, recall_curve, *args.medium_snr)
        if snr_bins is not None else float("nan"),
        "recall_high_snr": avg_recall_between(snr_bins, recall_curve, *args.high_snr)
        if snr_bins is not None else float("nan"),
        "recall_small": recall_for_size_band(stats, conf_thresh, 0.0, small_max),
        "recall_medium": recall_for_size_band(stats, conf_thresh, small_max, medium_max),
        "recall_large": recall_for_size_band(stats, conf_thresh, medium_max, None),
    }


def make_specific_loader(args: argparse.Namespace) -> DataLoader:
    dataset = YOLODatasetSpecificRes(
        data_dir=str(Path(args.data_dir) / args.split / "data"),
        labels_dir=str(Path(args.data_dir) / args.split / "labels_detect"),
        res_hw=tuple(args.res_hw),
        res_key=args.res_key,
        preprocessing=args.preprocessing,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=dataset.collate_fn,
    )


def make_fused_loader(args: argparse.Namespace) -> DataLoader:
    dataset = YOLODatasetFusedMultiRes(
        data_dir=str(Path(args.data_dir) / args.split / "data"),
        labels_dir=str(Path(args.data_dir) / args.split / "labels_detect"),
        res_keys=DEFAULT_RES_KEYS,
        preprocessing=args.preprocessing,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=dataset.collate_fn,
    )


def load_checkpoint(model: torch.nn.Module, checkpoint: Path, device: str) -> None:
    if hasattr(model, "load_weights"):
        model.load_weights(str(checkpoint), device=device, eval_mode=True)
    else:
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()


def build_specs(args: argparse.Namespace) -> list[EvalSpec]:
    input_channels = preprocessing_num_channels(args.preprocessing)
    training_root = Path(args.training_root)
    yolo_root = Path(args.yolo_ablation_root)
    yolo_complete_root = Path(args.yolo_ablation_complete_root)
    p3_root = Path(args.p3_root)
    mr_root = Path(args.mr_ablation_root)
    input_resolutions = find_input_resolutions(args.data_dir, split=args.split)

    yolo_common = dict(
        num_classes=args.num_classes,
        reg_max=args.reg_max,
        device=args.device,
        input_canals=input_channels,
        width_mult=args.width_mult,
        input_hw=tuple(args.res_hw),
    )
    rtdetr_common = dict(
        hidden_dim=128,
        num_queries=100,
        num_decoder_layers=6,
        num_heads=8,
        dim_feedforward=1024,
        dropout=0.0,
        matcher_num_threads=8,
    )
    mr_common = dict(
        input_resolutions=input_resolutions,
        num_classes=args.num_classes,
        reg_max=args.reg_max,
        device=args.device,
        in_ch=input_channels,
        width_mult=args.width_mult,
    )

    def ckpt(root: Path, name: str) -> Path:
        return root / name / "best.pt"

    def first_existing_ckpt(*candidates: Path) -> Path:
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return candidates[0]

    return [
        EvalSpec(
            "YOLOv11_baseline",
            "yolo",
            ckpt(training_root, f"yolov11n_specificres_{args.res_key}"),
            "specificres",
            lambda output_dir: YOLOv11(output_dir=output_dir, **yolo_common),
        ),
        EvalSpec(
            "YOLOv11_No_Neck",
            "yolo",
            first_existing_ckpt(
                ckpt(yolo_root, "yolov11n_no_neck_full_train"),
                ckpt(yolo_complete_root, "yolov11n_no_neck_full_train"),
            ),
            "specificres",
            lambda output_dir: YOLOv11NoNeck(output_dir=output_dir, **yolo_common),
        ),
        EvalSpec(
            "YOLOv11RTDETRHead",
            "one2one",
            first_existing_ckpt(
                ckpt(yolo_root, "yolov11n_best_ft_rtdeter_head_one2one_deformable"),
                ckpt(yolo_root, "yolov11n_best_ft_rtdetr_head_one2one_deformable"),
                ckpt(yolo_complete_root, "yolov11n_best_ft_rtdetr_head_one2one_deformable"),
            ),
            "specificres",
            lambda output_dir: YOLOv11RTDETRHead(
                output_dir=output_dir,
                num_decoder_points=4,
                use_deformable_attention=True,
                **yolo_common,
                **rtdetr_common,
            ),
        ),
        EvalSpec(
            "YOLOv11RTDETR",
            "one2one",
            first_existing_ckpt(
                ckpt(yolo_root, "yolov11n_rtder_yolov11_backbone_full_train"),
                ckpt(yolo_root, "yolov11n_rtdetr_yolov11_backbone_full_train"),
                ckpt(yolo_complete_root, "yolov11n_rtdetr_yolov11_backbone_full_train"),
            ),
            "specificres",
            lambda output_dir: YOLOv11RTDETR(
                output_dir=output_dir,
                num_decoder_points=4,
                use_deformable_attention=True,
                **yolo_common,
                **rtdetr_common,
            ),
        ),
        EvalSpec(
            "YOLOv11TransformerNeck",
            "yolo",
            first_existing_ckpt(
                ckpt(yolo_root, "yolov11n_transformer_neck_full_train"),
                ckpt(yolo_complete_root, "yolov11n_transformer_neck_full_train"),
            ),
            "specificres",
            lambda output_dir: YOLOv11TransformerNeck(
                output_dir=output_dir,
                transformer_d_model=128,
                transformer_num_heads=4,
                transformer_num_layers=1,
                transformer_ffn_ratio=2.0,
                transformer_dropout=0.0,
                transformer_residual_scale=0.0,
                transformer_neck_type="dense",
                transformer_num_points=4,
                **yolo_common,
            ),
        ),
        EvalSpec(
            "YOLOv11_Deformable_Neck",
            "yolo",
            first_existing_ckpt(
                ckpt(yolo_root, "yolov11n_transformer_neck_deformable_full_train"),
                ckpt(yolo_complete_root, "yolov11n_transformer_neck_deformable_full_train"),
            ),
            "specificres",
            lambda output_dir: YOLOv11TransformerNeck(
                output_dir=output_dir,
                transformer_d_model=128,
                transformer_num_heads=4,
                transformer_num_layers=1,
                transformer_ffn_ratio=2.0,
                transformer_dropout=0.0,
                transformer_residual_scale=0.0,
                transformer_neck_type="deformable",
                transformer_num_points=4,
                **yolo_common,
            ),
        ),
        EvalSpec(
            "YOLOv11P3Direct",
            "yolo",
            ckpt(p3_root, "exp1_yolov11n_p3_direct_tal_topk10"),
            "specificres",
            lambda output_dir: YOLOv11P3Direct(output_dir=output_dir, tal_topk=10, **yolo_common),
        ),
        EvalSpec(
            "YOLOv11P3RTDETR_frozen",
            "one2one",
            ckpt(p3_root, "exp2_1_yolov11n_p3_rtdetr_frozen_backbone"),
            "specificres",
            lambda output_dir: YOLOv11P3RTDETR(
                output_dir=output_dir,
                tal_topk=10,
                num_decoder_points=16,
                freeze_backbone=True,
                **yolo_common,
                **rtdetr_common,
            ),
        ),
        EvalSpec(
            "YOLOv11P3RTDETR_full",
            "one2one",
            ckpt(p3_root, "exp2_2_yolov11n_p3_rtdetr_full_train"),
            "specificres",
            lambda output_dir: YOLOv11P3RTDETR(
                output_dir=output_dir,
                tal_topk=10,
                num_decoder_points=16,
                freeze_backbone=False,
                **yolo_common,
                **rtdetr_common,
            ),
        ),
        EvalSpec(
            "YOLOv11SwinBackbone",
            "yolo",
            first_existing_ckpt(
                ckpt(yolo_root, "yolov11n_swin_backbone_full_train"),
                ckpt(yolo_complete_root, "yolov11n_swin_backbone_full_train"),
            ),
            "specificres",
            lambda output_dir: YOLOv11SwinBackbone(
                output_dir=output_dir,
                swin_depths=(2, 2, 4, 2),
                swin_num_heads=(2, 4, 8, 8),
                swin_window_size=8,
                **yolo_common,
            ),
        ),
        EvalSpec(
            "YOLOv11DATBackbone",
            "yolo",
            first_existing_ckpt(
                ckpt(yolo_root, "yolov11n_dat_backbone_full_train"),
                ckpt(yolo_complete_root, "yolov11n_dat_backbone_full_train"),
            ),
            "specificres",
            lambda output_dir: YOLOv11DATBackbone(
                output_dir=output_dir,
                dat_depths=(2, 2, 4, 2),
                dat_num_heads=(2, 4, 8, 8),
                dat_window_size=8,
                dat_num_points=7,
                **yolo_common,
            ),
        ),
        EvalSpec(
            "MR_YOLO_baseline",
            "mr_yolo",
            ckpt(training_root, "mr_yolo_n_fused_cfg512_cfg256_cfg128_cfg1024_cfg2048"),
            "fused",
            lambda output_dir: MR_YOLO(
                output_dir=output_dir,
                backbone_mode="TFSep_pyramid",
                outfusion_channels_mult=1,
                **mr_common,
            ),
        ),
        EvalSpec(
            "MRYOLOBranchCrossAttention_deformable",
            "mr_yolo",
            ckpt(mr_root, "mr_yolo_branch_cross_attention_deformable"),
            "fused",
            lambda output_dir: MRYOLOBranchCrossAttentionAblation(
                output_dir=output_dir,
                fusion_mode="deformable",
                center_resolution_index=0,
                fusion_d_model=128,
                fusion_num_heads=4,
                fusion_num_layers=1,
                fusion_num_points=4,
                outfusion_channels_mult=2,
                **mr_common,
            ),
        ),
        EvalSpec(
            "MRYOLOInputCrossAttention_deformable",
            "mr_yolo",
            ckpt(mr_root, "mr_yolo_input_cross_attention_deformable"),
            "fused",
            lambda output_dir: MRYOLOInputCrossAttentionAblation(
                output_dir=output_dir,
                fusion_mode="deformable",
                encoder_channels=16,
                center_resolution_index=0,
                fusion_d_model=128,
                fusion_num_heads=4,
                fusion_num_layers=1,
                fusion_num_points=4,
                **mr_common,
            ),
        ),
        EvalSpec(
            "MRYOLOBranchCrossAttention_global",
            "mr_yolo",
            ckpt(mr_root, "mr_yolo_branch_cross_attention_global"),
            "fused",
            lambda output_dir: MRYOLOBranchCrossAttentionAblation(
                output_dir=output_dir,
                fusion_mode="global",
                center_resolution_index=0,
                fusion_d_model=128,
                fusion_num_heads=4,
                fusion_num_layers=1,
                outfusion_channels_mult=2,
                **mr_common,
            ),
        ),
        EvalSpec(
            "MRYOLOInputCrossAttention_global",
            "mr_yolo",
            ckpt(mr_root, "mr_yolo_input_cross_attention_global"),
            "fused",
            lambda output_dir: MRYOLOInputCrossAttentionAblation(
                output_dir=output_dir,
                fusion_mode="global",
                encoder_channels=16,
                center_resolution_index=0,
                fusion_d_model=128,
                fusion_num_heads=4,
                fusion_num_layers=1,
                **mr_common,
            ),
        ),
    ]


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_one(
    spec: EvalSpec,
    loader: DataLoader,
    args: argparse.Namespace,
    output_dir: Path,
    class_index_to_name: dict[int, str] | None,
) -> dict[str, Any]:
    model_output_dir = output_dir / "model_outputs" / spec.name
    model = spec.builder(str(model_output_dir))
    try:
        load_checkpoint(model, spec.checkpoint, args.device)
        model.eval()
        metrics_json = output_dir / "json" / f"{spec.name}.json"
        stats_json = output_dir / "stats" / f"{spec.name}.json"
        full_metrics = dataset_analysis_with_metrics(
            model=model,
            val_loader=loader,
            iou_thresh=args.iou_thresh,
            fa=args.false_alarm_target,
            img_size=tuple(args.res_hw),
            to_save=str(metrics_json),
            to_plot=False,
            stats_path=stats_json,
            class_index_to_name=class_index_to_name,
        )
        with open(stats_json, "r", encoding="utf-8") as handle:
            stats = json.load(handle)
        row = summarize_metrics(full_metrics, stats, args)
        row.update(
            {
                "model": spec.name,
                "family": spec.family,
                "checkpoint": str(spec.checkpoint),
                "metrics_json": str(metrics_json),
                "stats_json": str(stats_json),
                "status": "ok",
            }
        )
        return row
    finally:
        del model
        cleanup()


def write_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "family",
        "status",
        "map50",
        "map50_95",
        "recall_low_snr",
        "recall_medium_snr",
        "recall_high_snr",
        "recall_small",
        "recall_medium",
        "recall_large",
        "conf_thresh",
        "checkpoint",
        "metrics_json",
        "stats_json",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def plot_bars(rows: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"[WARN] matplotlib indisponible, plots non générés: {exc}")
        return

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return
    names = [row["model"] for row in ok_rows]
    metrics = [
        ("map50", "mAP@50"),
        ("map50_95", "mAP@50:95"),
        ("recall_low_snr", "Recall low SNR"),
        ("recall_medium_snr", "Recall medium SNR"),
        ("recall_high_snr", "Recall high SNR"),
        ("recall_small", "Recall small objects"),
        ("recall_medium", "Recall medium objects"),
        ("recall_large", "Recall large objects"),
    ]
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for key, title in metrics:
        values = [float(row.get(key, float("nan"))) for row in ok_rows]
        fig_height = max(4.0, 0.35 * len(names))
        fig, ax = plt.subplots(figsize=(10, fig_height))
        y = np.arange(len(names))
        ax.barh(y, values, color="#4C78A8")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel(title)
        ax.set_title(title)
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{key}.png", dpi=200)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = build_specs(args)
    if args.max_models > 0:
        specs = specs[: args.max_models]

    print(f"[INFO] {len(specs)} expériences configurées")
    for spec in specs:
        state = "OK" if spec.checkpoint.is_file() else "MISSING"
        print(f"  - {spec.name}: {state} {spec.checkpoint}")

    if args.dry_run:
        return

    specific_loader = make_specific_loader(args)
    fused_loader = make_fused_loader(args)
    loaders = {"specificres": specific_loader, "fused": fused_loader}
    class_index_to_name = load_class_index_to_name(Path(args.data_dir))

    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, 1):
        print(f"\n[{index}/{len(specs)}] {spec.name}")
        if not spec.checkpoint.is_file():
            message = f"checkpoint missing: {spec.checkpoint}"
            print(f"[SKIP] {message}")
            if args.include_missing:
                rows.append(
                    {
                        "model": spec.name,
                        "family": spec.family,
                        "status": "missing_checkpoint",
                        "checkpoint": str(spec.checkpoint),
                    }
                )
            continue
        try:
            row = evaluate_one(spec, loaders[spec.dataset_mode], args, output_dir, class_index_to_name)
            rows.append(row)
            print(
                "[OK] "
                f"mAP50={row['map50']:.4f} "
                f"mAP50:95={row['map50_95']:.4f} "
                f"R_low={row['recall_low_snr']:.4f} "
                f"R_small={row['recall_small']:.4f}"
            )
        except Exception as exc:
            print(f"[ERROR] {spec.name}: {exc}")
            rows.append(
                {
                    "model": spec.name,
                    "family": spec.family,
                    "status": f"error: {exc}",
                    "checkpoint": str(spec.checkpoint),
                }
            )
            cleanup()

    csv_path = output_dir / "ablation_fine_metrics.csv"
    write_csv(rows, csv_path)
    plot_bars(rows, output_dir)
    summary_path = output_dir / "ablation_fine_metrics.json"
    summary_path.write_text(json.dumps(rows, indent=2, default=json_default), encoding="utf-8")

    print(f"\n[DONE] CSV: {csv_path}")
    print(f"[DONE] JSON: {summary_path}")
    print(f"[DONE] Plots: {output_dir / 'plots'}")


if __name__ == "__main__":
    main()
