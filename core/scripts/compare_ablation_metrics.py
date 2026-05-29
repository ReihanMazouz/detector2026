from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from detector2026.core.models.mr_yolo import MR_YOLO  # noqa: E402
from detector2026.core.models.mr_yolo_ablation import (  # noqa: E402
    MRPatchBackboneYOLOOne2ManyHead,
    MRYOLOBranchCrossAttentionAblation,
    MRViTPatchDetector,
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
DEFAULT_ONE2ONE_ROOT = (
    "/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation/"
    "one2one_head"
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
    parser.add_argument("--one2one-root", default=DEFAULT_ONE2ONE_ROOT)
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
    parser.add_argument(
        "--save-full-stats",
        action="store_true",
        help="Persist raw TP/FP/FN stats. Disabled by default because files can be hundreds of MB per model.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-evaluate models already marked status=ok.")
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


def box_characterization_stats(stats: dict[str, Any], conf_thresh: float) -> dict[str, float]:
    ious = []
    center_errors = []
    wh_errors = []
    area_ratios = []
    for rec in stats.get("tp", []):
        if float(rec.get("score", 0.0)) < conf_thresh:
            continue
        pred_box = rec.get("pred_box")
        gt_box = rec.get("gt_box")
        if not pred_box or not gt_box:
            continue
        px1, py1, px2, py2 = map(float, pred_box)
        gx1, gy1, gx2, gy2 = map(float, gt_box)
        pw = max(px2 - px1, 0.0)
        ph = max(py2 - py1, 0.0)
        gw = max(gx2 - gx1, 1e-12)
        gh = max(gy2 - gy1, 1e-12)
        pcx = 0.5 * (px1 + px2)
        pcy = 0.5 * (py1 + py2)
        gcx = 0.5 * (gx1 + gx2)
        gcy = 0.5 * (gy1 + gy2)
        ious.append(float(rec.get("max_iou", float("nan"))))
        center_errors.append(float(((pcx - gcx) ** 2 + (pcy - gcy) ** 2) ** 0.5))
        wh_errors.append(float(0.5 * (abs(pw - gw) / gw + abs(ph - gh) / gh)))
        area_ratios.append(float((pw * ph) / max(gw * gh, 1e-12)))

    def mean(values: list[float]) -> float:
        arr = np.asarray([value for value in values if not math.isnan(value)], dtype=float)
        return float(arr.mean()) if arr.size else float("nan")

    def median(values: list[float]) -> float:
        arr = np.asarray([value for value in values if not math.isnan(value)], dtype=float)
        return float(np.median(arr)) if arr.size else float("nan")

    return {
        "box_iou_mean": mean(ious),
        "box_iou_median": median(ious),
        "box_center_error_mean": mean(center_errors),
        "box_wh_relative_error_mean": mean(wh_errors),
        "box_area_ratio_mean": mean(area_ratios),
    }


def redundancy_stats(stats: dict[str, Any], conf_thresh: float) -> dict[str, float]:
    tp_count = sum(1 for rec in stats.get("tp", []) if float(rec.get("score", 0.0)) >= conf_thresh)
    gt_count = len(stats.get("tp", [])) + len(stats.get("fn", []))
    redundant = [
        rec for rec in stats.get("redundant", [])
        if float(rec.get("score", 0.0)) >= conf_thresh
    ]
    redundant_count = len(redundant)
    redundant_ious = [
        float(rec.get("max_iou", float("nan")))
        for rec in redundant
        if not math.isnan(float(rec.get("max_iou", float("nan"))))
    ]
    return {
        "redundant_boxes": float(redundant_count),
        "redundant_boxes_per_gt": float(redundant_count / gt_count) if gt_count else float("nan"),
        "redundant_boxes_per_tp": float(redundant_count / tp_count) if tp_count else float("nan"),
        "redundant_iou_mean": float(np.mean(redundant_ious)) if redundant_ious else float("nan"),
    }


def summarize_metrics(
    full_metrics: dict[str, Any],
    stats: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, float]:
    map_stats = full_metrics.get("map_stats", {})
    model_info = full_metrics.get("model_info", {})
    params = model_info.get("params", float("nan"))
    macs = model_info.get("flops", float("nan"))
    recall_snr = full_metrics.get("recall_snr", {}).get("global", {})
    snr_bins = recall_snr.get("snr_bins")
    recall_curve = recall_snr.get("recall")
    conf_thresh = float(full_metrics.get("conf_thresh", 0.0))
    small_max, medium_max = map(float, args.size_thresholds)
    box_stats = box_characterization_stats(stats, conf_thresh)
    duplicate_stats = redundancy_stats(stats, conf_thresh)

    summary = {
        "map50": float(map_stats.get("mAP50", float("nan"))),
        "map50_95": float(map_stats.get("mAP50:95", float("nan"))),
        "params": float(params) if params is not None else float("nan"),
        "macs": float(macs) if macs is not None else float("nan"),
        "flops": float(2 * macs) if macs is not None else float("nan"),
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
    summary.update(box_stats)
    summary.update(duplicate_stats)
    return summary


def sanitize_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def compact_curve_payload(full_metrics: dict[str, Any], summary: dict[str, float]) -> dict[str, Any]:
    f1_stats = full_metrics.get("f1_stats", {})
    recall_snr = full_metrics.get("recall_snr", {}).get("global", {})
    map_stats = full_metrics.get("map_stats", {})
    model_info = full_metrics.get("model_info", {})
    operating_point = full_metrics.get("operating_point", {})
    return {
        "summary": summary,
        "map_stats": map_stats,
        "model_info": model_info,
        "operating_point": operating_point,
        "confidence_threshold": full_metrics.get("conf_thresh"),
        "precision_recall": {
            "threshold": f1_stats.get("thr", []),
            "recall": f1_stats.get("recall", []),
            "precision": f1_stats.get("precision", []),
            "f1": f1_stats.get("f1", []),
        },
        "recall_snr": {
            "snr_bins": recall_snr.get("snr_bins", []),
            "recall": recall_snr.get("recall", []),
        },
    }


def save_compact_curve_payload(
    full_metrics: dict[str, Any],
    summary: dict[str, float],
    path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = compact_curve_payload(full_metrics, summary)
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    return path


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
        max_dim=10**9,
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


def load_checkpoint(model: torch.nn.Module, checkpoint: Path, device: str) -> dict[str, Any]:
    if hasattr(model, "load_weights"):
        missing, unexpected = model.load_weights(str(checkpoint), device=device, eval_mode=True)
        return {
            "method": "load_weights",
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
        }
    else:
        state = torch.load(checkpoint, map_location=device)
        incompatible = model.load_state_dict(state)
        model.to(device)
        model.eval()
        return {
            "method": "load_state_dict",
            "missing_keys": len(incompatible.missing_keys),
            "unexpected_keys": len(incompatible.unexpected_keys),
        }


def configure_eval_head(model: torch.nn.Module, spec: EvalSpec) -> None:
    if spec.family in {"one2one", "one2one_yolo"} and hasattr(model, "use_one2one_head"):
        model.use_one2one_head()


def eval_img_size_for_model(model: torch.nn.Module, spec: EvalSpec, args: argparse.Namespace) -> tuple[int, int]:
    if spec.dataset_mode == "fused" and hasattr(model, "input_resolutions"):
        resolutions = list(getattr(model, "input_resolutions"))
        return tuple(int(max(values)) for values in zip(*resolutions))
    return tuple(args.res_hw)


def build_specs(args: argparse.Namespace) -> list[EvalSpec]:
    input_channels = preprocessing_num_channels(args.preprocessing)
    training_root = Path(args.training_root)
    yolo_root = Path(args.yolo_ablation_root)
    yolo_complete_root = Path(args.yolo_ablation_complete_root)
    p3_root = Path(args.p3_root)
    one2one_root = Path(args.one2one_root)
    mr_root = Path(args.mr_ablation_root)
    input_resolutions = find_input_resolutions(args.data_dir, split=args.split)
    if len(input_resolutions) != len(DEFAULT_RES_KEYS):
        raise ValueError(
            f"Expected {len(DEFAULT_RES_KEYS)} resolutions for MR models, "
            f"found {len(input_resolutions)}: {input_resolutions}"
        )

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

    def ckpt_or_file(root: Path, name: str) -> Path:
        direct = root / name
        return direct if direct.is_file() else direct / "best.pt"

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
                ckpt(yolo_complete_root, "yolov11n_no_neck_direct_p3p4p5"),
                ckpt(yolo_root, "yolov11n_no_neck_full_train"),
                ckpt(yolo_complete_root, "yolov11n_no_neck_full_train"),
            ),
            "specificres",
            lambda output_dir: YOLOv11NoNeck(output_dir=output_dir, **yolo_common),
        ),
        EvalSpec(
            "YOLOv11_One2One_TAL",
            "one2one_yolo",
            first_existing_ckpt(
                ckpt_or_file(one2one_root, "best_one2one_benchmark_tal"),
                ckpt(one2one_root, "yolov11n_specificres_cfg512_one2one_tal"),
            ),
            "specificres",
            lambda output_dir: YOLOv11(output_dir=output_dir, **yolo_common),
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
            "MRPatchBackboneYOLO_P3_One2Many",
            "mr_yolo",
            first_existing_ckpt(
                ckpt(mr_root, "mr_patch_backbone_yolo_one2one_head"),
                ckpt(mr_root, "mr_patch_backbone_yolo_one2many_head"),
            ),
            "fused",
            lambda output_dir: MRPatchBackboneYOLOOne2ManyHead(
                input_resolutions=input_resolutions,
                output_dir=output_dir,
                num_classes=args.num_classes,
                reg_max=args.reg_max,
                device=args.device,
                in_ch=input_channels,
                d_model=128,
                patch_size=8,
                num_encoder_layers=3,
                num_heads=4,
                num_intra_points=8,
                num_inter_neighbors=8,
                dim_feedforward=512,
                dropout=0.0,
                p3_hw=(32, 32),
                stride=32,
            ),
        ),
        EvalSpec(
            "MRViTPatchDetector_RTDETR",
            "one2one",
            ckpt(mr_root, "mr_vit_patch_detector_rtdetr"),
            "fused",
            lambda output_dir: MRViTPatchDetector(
                input_resolutions=input_resolutions,
                output_dir=output_dir,
                num_classes=args.num_classes,
                device=args.device,
                in_ch=input_channels,
                d_model=256,
                num_encoder_layers=6,
                num_decoder_layers=6,
                num_queries=100,
                patch_grid_hw=(32, 32),
                num_heads=8,
                num_encoder_points=16,
                num_decoder_points=16,
                dim_feedforward=1024,
                dropout=0.0,
                matcher_num_threads=8,
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
        load_info = load_checkpoint(model, spec.checkpoint, args.device)
        configure_eval_head(model, spec)
        model.eval()
        eval_img_size = eval_img_size_for_model(model, spec, args)
        metrics_json = output_dir / "json" / f"{spec.name}.json"
        compact_stats_json = output_dir / "stats_compact" / f"{spec.name}.json"
        full_stats_json = output_dir / "stats_full" / f"{spec.name}.json"
        if args.save_full_stats:
            stats_path = full_stats_json
            full_metrics = dataset_analysis_with_metrics(
                model=model,
                val_loader=loader,
                iou_thresh=args.iou_thresh,
                fa=args.false_alarm_target,
                img_size=eval_img_size,
                to_save=str(metrics_json),
                to_plot=False,
                stats_path=stats_path,
                class_index_to_name=class_index_to_name,
            )
            with open(stats_path, "r", encoding="utf-8") as handle:
                stats = json.load(handle)
        else:
            with tempfile.TemporaryDirectory(prefix="ablation_stats_") as tmp_dir:
                stats_path = Path(tmp_dir) / f"{spec.name}.json"
                full_metrics = dataset_analysis_with_metrics(
                    model=model,
                    val_loader=loader,
                    iou_thresh=args.iou_thresh,
                    fa=args.false_alarm_target,
                    img_size=eval_img_size,
                    to_save=str(metrics_json),
                    to_plot=False,
                    stats_path=stats_path,
                    class_index_to_name=class_index_to_name,
                )
                with open(stats_path, "r", encoding="utf-8") as handle:
                    stats = json.load(handle)
        row = summarize_metrics(full_metrics, stats, args)
        save_compact_curve_payload(full_metrics, row, compact_stats_json)
        row.update(
            {
                "model": spec.name,
                "family": spec.family,
                "checkpoint": str(spec.checkpoint),
                "load_method": load_info["method"],
                "load_missing_keys": load_info["missing_keys"],
                "load_unexpected_keys": load_info["unexpected_keys"],
                "metrics_json": str(metrics_json),
                "compact_stats_json": str(compact_stats_json),
                "full_stats_json": str(full_stats_json) if args.save_full_stats else "",
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
        "params",
        "macs",
        "flops",
        "recall_low_snr",
        "recall_medium_snr",
        "recall_high_snr",
        "recall_small",
        "recall_medium",
        "recall_large",
        "box_iou_mean",
        "box_iou_median",
        "box_center_error_mean",
        "box_wh_relative_error_mean",
        "box_area_ratio_mean",
        "redundant_boxes",
        "redundant_boxes_per_gt",
        "redundant_boxes_per_tp",
        "redundant_iou_mean",
        "conf_thresh",
        "checkpoint",
        "load_method",
        "load_missing_keys",
        "load_unexpected_keys",
        "metrics_json",
        "compact_stats_json",
        "full_stats_json",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_existing_rows(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.is_file():
        return []
    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_is_reusable(row: dict[str, Any]) -> bool:
    if row.get("status") != "ok":
        return False
    compact_path = row.get("compact_stats_json", "")
    metrics_path = row.get("metrics_json", "")
    if compact_path and not Path(compact_path).is_file():
        return False
    if metrics_path and not Path(metrics_path).is_file():
        return False
    return True


def merge_rows_by_model(existing_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = {row.get("model", ""): row for row in existing_rows if row.get("model")}
    for row in new_rows:
        if row.get("model"):
            merged[row["model"]] = row
    return list(merged.values())


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
        ("box_iou_mean", "Mean TP IoU"),
        ("box_center_error_mean", "Mean center error"),
        ("box_wh_relative_error_mean", "Mean relative size error"),
        ("redundant_boxes_per_gt", "Redundant boxes per GT"),
        ("redundant_boxes_per_tp", "Redundant boxes per TP"),
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
        if key not in {"box_center_error_mean", "box_wh_relative_error_mean", "redundant_boxes_per_gt", "redundant_boxes_per_tp"}:
            ax.set_xlim(0.0, 1.0)
        ax.set_xlabel(title)
        ax.set_title(title)
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{key}.png", dpi=200)
        plt.close(fig)


def _load_compact_payload(row: dict[str, Any]) -> dict[str, Any] | None:
    path = row.get("compact_stats_json")
    if not path:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _snr_centers(snr_bins: Sequence[float]) -> np.ndarray:
    bins = np.asarray(snr_bins, dtype=float)
    if len(bins) < 2:
        return np.asarray([], dtype=float)
    return 0.5 * (bins[:-1] + bins[1:])


def plot_curves(rows: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"[WARN] matplotlib indisponible, courbes non générées: {exc}")
        return

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return

    curves_dir = output_dir / "plots" / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    aggregate_snr = []
    aggregate_pr = []

    for row in ok_rows:
        payload = _load_compact_payload(row)
        if payload is None:
            continue
        model_name = row["model"]
        stem = sanitize_filename(model_name)
        pr = payload.get("precision_recall", {})
        recall = np.asarray(pr.get("recall", []), dtype=float)
        precision = np.asarray(pr.get("precision", []), dtype=float)
        f1 = np.asarray(pr.get("f1", []), dtype=float)
        thresholds = np.asarray(pr.get("threshold", []), dtype=float)
        snr = payload.get("recall_snr", {})
        snr_bins = snr.get("snr_bins", [])
        snr_recall = np.asarray(snr.get("recall", []), dtype=float)
        snr_centers = _snr_centers(snr_bins)

        if len(recall) and len(precision):
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(recall, precision, linewidth=2.0)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(f"Precision-Recall - {model_name}")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, linestyle="--", alpha=0.35)
            fig.tight_layout()
            fig.savefig(curves_dir / f"{stem}_precision_recall.png", dpi=200)
            plt.close(fig)
            aggregate_pr.append((model_name, recall, precision))

        if len(thresholds) and len(recall) and len(precision):
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(thresholds, recall, label="Recall", linewidth=2.0)
            ax.plot(thresholds, precision, label="Precision", linewidth=2.0)
            ax.set_xlabel("Confidence threshold")
            ax.set_ylabel("Score")
            ax.set_title(f"Precision/Recall vs threshold - {model_name}")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend()
            fig.tight_layout()
            fig.savefig(curves_dir / f"{stem}_precision_recall_threshold.png", dpi=200)
            plt.close(fig)

        if len(thresholds) and len(f1):
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(thresholds, f1, color="#2CA02C", linewidth=2.0)
            ax.set_xlabel("Confidence threshold")
            ax.set_ylabel("F1-score")
            ax.set_title(f"F1-score vs threshold - {model_name}")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, linestyle="--", alpha=0.35)
            fig.tight_layout()
            fig.savefig(curves_dir / f"{stem}_f1_threshold.png", dpi=200)
            plt.close(fig)

        if len(snr_centers) and len(snr_recall):
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(snr_centers, snr_recall, marker="o", linewidth=2.0)
            ax.set_xlabel("SNR (dB)")
            ax.set_ylabel("Recall")
            ax.set_title(f"Recall vs SNR - {model_name}")
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, linestyle="--", alpha=0.35)
            fig.tight_layout()
            fig.savefig(curves_dir / f"{stem}_recall_snr.png", dpi=200)
            plt.close(fig)
            aggregate_snr.append((model_name, snr_centers, snr_recall))

    if aggregate_snr:
        fig_height = max(5.0, min(12.0, 0.28 * len(aggregate_snr) + 4.0))
        fig, ax = plt.subplots(figsize=(10, fig_height))
        for model_name, x, y in aggregate_snr:
            ax.plot(x, y, marker="o", linewidth=1.6, label=model_name)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Recall")
        ax.set_title("Recall vs SNR")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=7, ncol=1, loc="center left", bbox_to_anchor=(1.0, 0.5))
        fig.tight_layout()
        fig.savefig(curves_dir / "compare_recall_snr.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    if aggregate_pr:
        fig_height = max(5.0, min(12.0, 0.28 * len(aggregate_pr) + 4.0))
        fig, ax = plt.subplots(figsize=(10, fig_height))
        for model_name, x, y in aggregate_pr:
            ax.plot(x, y, linewidth=1.6, label=model_name)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=7, ncol=1, loc="center left", bbox_to_anchor=(1.0, 0.5))
        fig.tight_layout()
        fig.savefig(curves_dir / "compare_precision_recall.png", dpi=200, bbox_inches="tight")
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

    csv_path = output_dir / "ablation_fine_metrics.csv"
    existing_rows = read_existing_rows(csv_path)
    reusable_rows_by_model = {
        row["model"]: row
        for row in existing_rows
        if row.get("model") and row_is_reusable(row)
    }
    if reusable_rows_by_model and not args.overwrite:
        print(f"[INFO] reprise: {len(reusable_rows_by_model)} modèles déjà évalués seront conservés")

    specific_loader = make_specific_loader(args)
    fused_loader = make_fused_loader(args)
    loaders = {"specificres": specific_loader, "fused": fused_loader}
    class_index_to_name = load_class_index_to_name(Path(args.data_dir))

    new_rows: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, 1):
        print(f"\n[{index}/{len(specs)}] {spec.name}")
        if not args.overwrite and spec.name in reusable_rows_by_model:
            print(f"[SKIP] {spec.name}: déjà évalué avec status=ok")
            continue
        if not spec.checkpoint.is_file():
            message = f"checkpoint missing: {spec.checkpoint}"
            print(f"[SKIP] {message}")
            if args.include_missing:
                new_rows.append(
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
            new_rows.append(row)
            print(
                "[OK] "
                f"mAP50={row['map50']:.4f} "
                f"mAP50:95={row['map50_95']:.4f} "
                f"R_low={row['recall_low_snr']:.4f} "
                f"R_small={row['recall_small']:.4f}"
            )
        except Exception as exc:
            print(f"[ERROR] {spec.name}: {exc}")
            new_rows.append(
                {
                    "model": spec.name,
                    "family": spec.family,
                    "status": f"error: {exc}",
                    "checkpoint": str(spec.checkpoint),
                }
            )
            cleanup()

    rows = merge_rows_by_model([] if args.overwrite else existing_rows, new_rows)
    write_csv(rows, csv_path)
    plot_bars(rows, output_dir)
    plot_curves(rows, output_dir)
    summary_path = output_dir / "ablation_fine_metrics.json"
    summary_path.write_text(json.dumps(rows, indent=2, default=json_default), encoding="utf-8")

    print(f"\n[DONE] CSV: {csv_path}")
    print(f"[DONE] JSON: {summary_path}")
    print(f"[DONE] Plots: {output_dir / 'plots'}")


if __name__ == "__main__":
    main()
