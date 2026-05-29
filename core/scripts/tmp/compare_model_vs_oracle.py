from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from detector2026.core.models.yolov11 import YOLOv11
from detector2026.core.utils.analysing_results import analyse_results, stats_analysis_with_metrics
from detector2026.core.utils.dataset._common import load_label_items
from detector2026.core.utils.fusion_uni_res import oracle_or_post_nms
from detector2026.core.utils.preprocess import build_preprocessor


# =====================================================================
# PARAMETRES EN DUR
# =====================================================================

CHECKPOINT_PATH = Path(
    "/data/RAWSIM/RMA/Thesis_work/yolo_perso/training_folder/rf_dataset_thesis/yolov11n_specificres_cfg512/best.pt"
)
DATASET_PATH = Path(
    "/data/RAWSIM/RMA/rf_dataset_thesis"
)
SPLIT = "val"
DEVICE = "cuda:0" 
OUTPUT_JSON = CHECKPOINT_PATH.parent / "compare_model_vs_oracle.json"

NUM_CLASSES = 20
WIDTH_MULT = 0.25
REG_MAX = 16
PREPROCESSING = "none"
RES_KEY = "cfg512"
RES_HW = (256, 256)
ANISOTROPIC = False
P3_SIZE = (64, 64)

SAMPLE_LIMIT = 100
BASE_POSTPROCESS_CONF = 0.05
POSTPROCESS_IOU = 0.1
SAME_BOX_IOU = 0.9
EVAL_IOU = 0.5
FALSE_ALARM_TARGET = 0.01

PSNR_KEYS = ["cfg128", "cfg256", "cfg512", "cfg1024", "cfg2048"]
CLASS_INDEX_TO_NAME = {
    0: "no_mod",
    1: "LFM",
    2: "NLFM",
    3: "QFM",
    4: "FMCW_TRI",
    5: "barker_biphasique",
    6: "random_biphasique",
    7: "FSK",
    8: "P1",
    9: "P2",
    10: "P3",
    11: "P4",
    12: "frank",
    13: "T1",
    14: "T2",
    15: "T3",
    16: "T4",
    17: "OFDM",
    18: "FHSS",
    19: "DSSS",
}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _build_model() -> YOLOv11:
    model = YOLOv11(
        output_dir=str(CHECKPOINT_PATH.parent),
        num_classes=NUM_CLASSES,
        reg_max=REG_MAX,
        device=DEVICE,
        input_canals=1,
        width_mult=WIDTH_MULT,
        anisotropic=ANISOTROPIC,
        p3_size=P3_SIZE,
        input_hw=RES_HW,
    )
    model.load_weights(str(CHECKPOINT_PATH), device=DEVICE, eval_mode=True)
    model.eval()
    return model


def _load_gt(label_path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[float]]]:
    items = load_label_items(label_path)

    gt_boxes = []
    gt_labels = []
    gt_snrs = []
    gt_psnrs = []

    for item in items:
        xc = float(item["xc"])
        yc = float(item["yc"])
        w = float(item["w"])
        h = float(item["h"])

        gt_boxes.append([xc - w / 2.0, yc - h / 2.0, xc + w / 2.0, yc + h / 2.0])
        gt_labels.append(int(item["class"]))
        gt_snrs.append(float(item.get("snr", -1.0)))

        psnr = item.get("psnr")
        if isinstance(psnr, dict):
            gt_psnrs.append([float(psnr.get(key, -1.0)) for key in PSNR_KEYS])
        else:
            gt_psnrs.append([])

    gt_boxes_tensor = torch.tensor(gt_boxes, dtype=torch.float32) if gt_boxes else torch.zeros((0, 4), dtype=torch.float32)
    gt_labels_tensor = torch.tensor(gt_labels, dtype=torch.long) if gt_labels else torch.zeros((0,), dtype=torch.long)
    gt_snrs_tensor = torch.tensor(gt_snrs, dtype=torch.float32) if gt_snrs else torch.zeros((0,), dtype=torch.float32)

    return gt_boxes_tensor, gt_labels_tensor, gt_snrs_tensor, gt_psnrs


def _load_raw_tensors(sample_path: Path) -> List[torch.Tensor]:
    raw = torch.load(sample_path, map_location="cpu")
    if isinstance(raw, list):
        return raw
    return [raw]


def _pick_tensor_for_resolution(raw_tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    target_h, target_w = RES_HW
    for tensor in raw_tensors:
        if tensor.ndim == 2 and tuple(tensor.shape) == (target_h, target_w):
            return tensor
        if tensor.ndim == 3 and tuple(tensor.shape[-2:]) == (target_h, target_w):
            return tensor
    raise ValueError(f"Aucun tenseur de taille {RES_HW} trouve dans le sample.")


def _run_model_on_sample(model: YOLOv11, sample_path: Path) -> torch.Tensor:
    raw_tensors = _load_raw_tensors(sample_path)
    raw_tensor = _pick_tensor_for_resolution(raw_tensors)

    preprocess = build_preprocessor(PREPROCESSING)
    image = preprocess(raw_tensor, cfg_key=RES_KEY).unsqueeze(0).to(model.device, dtype=torch.float32)

    with torch.no_grad():
        dist_out, cls_out = model(image)
        processed = model.postprocess(
            dist_out,
            cls_out,
            dist_out,
            conf_thres=BASE_POSTPROCESS_CONF,
            iou_thres=POSTPROCESS_IOU,
            iou_same_box=SAME_BOX_IOU,
        )

    detections = processed[0] if processed else None
    if detections is None or len(detections) == 0:
        return torch.zeros((0, 6), dtype=torch.float32)

    detections = detections.detach().cpu().to(torch.float32)
    h, w = RES_HW
    return torch.stack(
        [
            (detections[:, 0] / float(w)).clamp(0.0, 1.0),
            (detections[:, 1] / float(h)).clamp(0.0, 1.0),
            (detections[:, 2] / float(w)).clamp(0.0, 1.0),
            (detections[:, 3] / float(h)).clamp(0.0, 1.0),
            detections[:, 4],
            detections[:, 5],
        ],
        dim=1,
    )


def _stats_from_predictions(
    predictions: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    gt_snrs: torch.Tensor,
    gt_psnrs: List[List[float]],
) -> Dict[str, List[Dict[str, Any]]]:
    if len(predictions) > 0:
        pred_boxes = predictions[:, :4]
        pred_scores = predictions[:, 4]
        pred_labels = predictions[:, 5].long()
    else:
        pred_boxes = torch.zeros((0, 4), dtype=torch.float32)
        pred_scores = torch.zeros((0,), dtype=torch.float32)
        pred_labels = torch.zeros((0,), dtype=torch.long)

    return analyse_results(
        pred_boxes=pred_boxes,
        pred_scores=pred_scores,
        pred_labels=pred_labels,
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        gt_snrs=gt_snrs,
        iou_thresh=EVAL_IOU,
        gt_psnrs=gt_psnrs,
        psnr_keys=PSNR_KEYS,
    )


def _count_stats(stats: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:
    return {
        "tp": len(stats["tp"]),
        "fp": len(stats["fp"]),
        "fn": len(stats["fn"]),
    }


def _merge_stats(aggregate: Dict[str, List[Dict[str, Any]]], sample_stats: Dict[str, List[Dict[str, Any]]]) -> None:
    aggregate["tp"].extend(sample_stats["tp"])
    aggregate["fp"].extend(sample_stats["fp"])
    aggregate["fn"].extend(sample_stats["fn"])


def _metric_summary(metrics: Dict[str, Any]) -> Dict[str, float]:
    map_stats = metrics.get("map_stats", {})
    recall_snr = metrics.get("recall_snr", {}).get("global", {})
    snr_bins = recall_snr.get("snr_bins", [])
    recall = recall_snr.get("recall", [])

    def _avg_recall_between(a: float, b: float) -> float:
        if len(snr_bins) == 0 or len(recall) == 0:
            return float("nan")
        snr_bins_np = np.asarray(snr_bins, dtype=float)
        recall_np = np.asarray(recall, dtype=float)
        left = max(a, float(snr_bins_np[0]))
        right = min(b, float(snr_bins_np[-1]))
        if right <= left:
            return float("nan")

        area = 0.0
        for k in range(len(recall_np)):
            bin_left = snr_bins_np[k]
            bin_right = snr_bins_np[k + 1]
            overlap_left = max(bin_left, left)
            overlap_right = min(bin_right, right)
            width = max(0.0, overlap_right - overlap_left)
            if width > 0:
                area += float(recall_np[k]) * width
        return float(area / max(right - left, 1e-12))

    return {
        "mAP50": float(map_stats.get("mAP50", float("nan"))),
        "mAP50:95": float(map_stats.get("mAP50:95", float("nan"))),
        "avg_recall_low_snr": _avg_recall_between(-10.0, 19.0),
        "avg_recall_medium_snr": _avg_recall_between(0.0, 19.0),
        "avg_recall_high_snr": _avg_recall_between(10.0, 19.0),
    }


def main() -> None:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint introuvable: '{CHECKPOINT_PATH}'")
    if not DATASET_PATH.is_dir():
        raise FileNotFoundError(f"Dataset introuvable: '{DATASET_PATH}'")

    sample_paths = sorted((DATASET_PATH / SPLIT / "data").glob("*.pt"))[:SAMPLE_LIMIT]
    labels_dir = DATASET_PATH / SPLIT / "labels_detect"
    if not sample_paths:
        raise RuntimeError(f"Aucun sample trouve dans '{DATASET_PATH / SPLIT / 'data'}'.")

    print("[1/4] Chargement du modele")
    model = _build_model()

    print("[2/4] Comparaison modele seul vs oracle")
    model_stats = {"tp": [], "fp": [], "fn": []}
    oracle_stats = {"tp": [], "fp": [], "fn": []}
    per_sample = []

    for sample_path in sample_paths:
        label_path = labels_dir / f"{sample_path.stem}.json"
        gt_boxes, gt_labels, gt_snrs, gt_psnrs = _load_gt(label_path)

        model_predictions = _run_model_on_sample(model, sample_path)
        oracle_output = oracle_or_post_nms(
            prediction_sets=[model_predictions],
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            iou_thresh=EVAL_IOU,
            false_alarm_iou_thresh=EVAL_IOU,
        )
        oracle_predictions = oracle_output["oracle_predictions"].detach().cpu().to(torch.float32)
        model_sample_stats = _stats_from_predictions(model_predictions, gt_boxes, gt_labels, gt_snrs, gt_psnrs)
        oracle_sample_stats = _stats_from_predictions(oracle_predictions, gt_boxes, gt_labels, gt_snrs, gt_psnrs)

        _merge_stats(model_stats, model_sample_stats)
        _merge_stats(oracle_stats, oracle_sample_stats)

        model_counts = _count_stats(model_sample_stats)
        oracle_counts = _count_stats(oracle_sample_stats)

        per_sample.append(
            {
                "sample_id": sample_path.stem,
                "num_gt": int(len(gt_boxes)),
                "num_model_predictions": int(len(model_predictions)),
                "num_oracle_predictions": int(len(oracle_predictions)),
                "model_counts": model_counts,
                "oracle_counts": oracle_counts,
                "counts_equal": model_counts == oracle_counts,
            }
        )

        print(
            f"  - {sample_path.stem}: "
            f"model={model_counts} | oracle={oracle_counts}"
        )

    print("[3/4] Calcul des metriques agregees")
    model_metrics = stats_analysis_with_metrics(
        model_stats,
        fa=FALSE_ALARM_TARGET,
        to_plot=False,
        class_index_to_name=CLASS_INDEX_TO_NAME,
    )
    oracle_metrics = stats_analysis_with_metrics(
        oracle_stats,
        fa=FALSE_ALARM_TARGET,
        to_plot=False,
        class_index_to_name=CLASS_INDEX_TO_NAME,
    )

    print("[4/4] Sauvegarde du resultat")
    payload = {
        "checkpoint": str(CHECKPOINT_PATH),
        "dataset_path": str(DATASET_PATH),
        "split": SPLIT,
        "device": DEVICE,
        "sample_limit": SAMPLE_LIMIT,
        "eval_iou": EVAL_IOU,
        "false_alarm_target": FALSE_ALARM_TARGET,
        "model_config": {
            "model_id": "yolov11",
            "num_classes": NUM_CLASSES,
            "width_mult": WIDTH_MULT,
            "reg_max": REG_MAX,
            "preprocessing": PREPROCESSING,
            "res_key": RES_KEY,
            "res_hw": RES_HW,
            "anisotropic": ANISOTROPIC,
            "p3_size": P3_SIZE,
        },
        "aggregated_counts": {
            "model": _count_stats(model_stats),
            "oracle": _count_stats(oracle_stats),
        },
        "model_metrics": model_metrics,
        "oracle_metrics": oracle_metrics,
        "per_sample": per_sample,
    }

    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")

    print("\n[Comparaison terminee]")
    print(f"Resultat JSON : {OUTPUT_JSON}")
    print(f"Model counts  : {_count_stats(model_stats)}")
    print(f"Oracle counts : {_count_stats(oracle_stats)}")
    print("\n[Metriques]")
    model_summary = _metric_summary(model_metrics)
    oracle_summary = _metric_summary(oracle_metrics)
    for key in ("mAP50", "mAP50:95", "avg_recall_low_snr", "avg_recall_medium_snr", "avg_recall_high_snr"):
        print(
            f"{key:22s} "
            f"model={model_summary[key]:.6f} | oracle={oracle_summary[key]:.6f}"
        )


if __name__ == "__main__":
    main()
