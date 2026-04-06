from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from detector2026.core.models.yolov11 import YOLOv11
from detector2026.core.utils.analysing_results import (
    analyse_results,
    confusion_matrix_snr,
    map_from_stats,
    precision_recall_stats,
    recall_per_max_psnr_bin,
    recall_per_size_bin,
    recall_per_snr_bin,
)
from detector2026.core.utils.dataset._common import load_label_items
from detector2026.core.utils.fusion_uni_res import nms_fusion_post_nms
from detector2026.core.utils.preprocess import build_preprocessor


# =====================================================================
# PARAMETRES A EDITER
# =====================================================================

DATASET_PATH = Path("/Users/tailleesarah/Documents/thèse/icml/ICML2026DataSimulator/examples/output/rf_dataset_v2")
SPLIT = "val"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 20
WIDTH_MULT = 0.25  # YOLOv11n
REG_MAX = 16
PREPROCESSING = "none"

BASE_POSTPROCESS_CONF = 0.05
POSTPROCESS_IOU = 0.1
SAME_BOX_IOU = 0.9
FUSION_NMS_IOU = 0.5
EVAL_IOU = 0.5
FUSION_CLASS_AGNOSTIC = False
FALSE_ALARM_TARGET = 0.01

# Deux options:
# - "keep_model_thresholds": on garde la fusion NMS telle qu'elle sort des modeles
#   individuels deja regles a 1% de fausse alarme.
# - "fusion_1pct_fa": on re-seuille ensuite la fusion NMS elle-meme pour viser 1%
#   de fausse alarme.
FINAL_CONF_MODE = "keep_model_thresholds"

OUTPUT_DIR = Path("/Users/tailleesarah/Documents/thèse/icml/detector2026/runs/nms_eval_simple")

MODEL_SPECS = [
    {
        "label": "cfg512",
        "checkpoint": "/Users/tailleesarah/Documents/thèse/icml/detector2026/runs/examples_of_training/yolov11n_specificres/best.pt",
        "res_key": "cfg512",
        "res_hw": (256, 256),
    },
    # {
    #     "label": "cfg256",
    #     "checkpoint": "/Users/.../yolov11n_cfg256/best.pt",
    #     "res_key": "cfg256",
    #     "res_hw": (128, 128),
    # },
]

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

PSNR_KEYS = ["cfg128", "cfg256", "cfg512", "cfg1024", "cfg2048"]


def _json_default(obj: Any) -> Any:
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
    return obj


def _build_model(spec: Dict[str, Any]) -> YOLOv11:
    checkpoint_path = Path(spec["checkpoint"])
    config_path = checkpoint_path.parent / "config.json"
    anisotropic = False
    p3_size = (64, 64)
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            model_config = dict(config.get("model_config", {}))
            anisotropic = bool(model_config.get("anisotropic", False))
            p3_size = tuple(model_config.get("p3_size", [64, 64]))
        except (OSError, json.JSONDecodeError):
            pass

    return YOLOv11(
        output_dir=str(OUTPUT_DIR),
        num_classes=NUM_CLASSES,
        width_mult=WIDTH_MULT,
        reg_max=REG_MAX,
        input_canals=1,
        device=DEVICE,
        anisotropic=anisotropic,
        p3_size=p3_size,
        input_hw=tuple(spec["res_hw"]),
    )


def _safe_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_metrics_json(spec: Dict[str, Any]) -> Path:
    checkpoint_path = Path(spec["checkpoint"])
    run_dir = checkpoint_path.parent
    train_log_path = run_dir / "train_log.csv"
    if not train_log_path.is_file():
        raise FileNotFoundError(f"train_log.csv introuvable dans '{run_dir}'.")

    with train_log_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise RuntimeError(f"train_log.csv est vide dans '{run_dir}'.")

    best_row = None
    best_score = None
    for row in rows:
        score = _safe_float(row.get("map50_95"))
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_row = row

    if best_row is None:
        raise RuntimeError(f"Aucune valeur map50_95 exploitable dans '{train_log_path}'.")

    epoch_value = _safe_float(best_row.get("epoch"))
    if epoch_value is None:
        raise RuntimeError(f"Impossible de lire l'epoch associee au best.pt dans '{train_log_path}'.")

    metrics_path = run_dir / "metrics" / f"metrics_epoch_{int(epoch_value):03d}.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"Le metrics JSON de l'epoch best.pt est introuvable: '{metrics_path}'."
        )
    return metrics_path


def _find_threshold_for_one_percent_fa(spec: Dict[str, Any]) -> float:
    metrics_json = _resolve_metrics_json(spec)
    payload = json.loads(metrics_json.read_text(encoding="utf-8"))
    pr = payload["f1_stats"]

    threshold = 0.0
    for thr, precision in zip(pr["thr"], pr["precision"]):
        if (1.0 - precision) <= FALSE_ALARM_TARGET:
            threshold = float(thr)
            break
    return threshold


def _load_gt(label_path: Path):
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

    gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32) if gt_boxes else torch.zeros((0, 4), dtype=torch.float32)
    gt_labels = torch.tensor(gt_labels, dtype=torch.long) if gt_labels else torch.zeros((0,), dtype=torch.long)
    gt_snrs = torch.tensor(gt_snrs, dtype=torch.float32) if gt_snrs else torch.zeros((0,), dtype=torch.float32)
    return gt_boxes, gt_labels, gt_snrs, gt_psnrs


def _load_raw_tensors(sample_path: Path) -> List[torch.Tensor]:
    raw = torch.load(sample_path, map_location="cpu")
    return raw if isinstance(raw, list) else [raw]


def _pick_tensor_for_resolution(raw_tensors: Sequence[torch.Tensor], res_hw: Tuple[int, int]) -> torch.Tensor:
    target_h, target_w = res_hw
    for tensor in raw_tensors:
        if tensor.ndim == 2 and tuple(tensor.shape) == (target_h, target_w):
            return tensor
        if tensor.ndim == 3 and tuple(tensor.shape[-2:]) == (target_h, target_w):
            return tensor
    raise ValueError(f"Aucun tenseur de taille {res_hw} trouve dans le sample.")


def _run_one_model_on_one_sample(
    model: YOLOv11,
    spec: Dict[str, Any],
    conf_thresh: float,
    sample_path: Path,
) -> torch.Tensor:
    raw_tensors = _load_raw_tensors(sample_path)
    raw_tensor = _pick_tensor_for_resolution(raw_tensors, tuple(spec["res_hw"]))

    preprocess = build_preprocessor(PREPROCESSING)
    image = preprocess(raw_tensor, cfg_key=spec["res_key"]).unsqueeze(0).to(model.device)

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
    detections = detections[detections[:, 4] >= float(conf_thresh)]
    if len(detections) == 0:
        return torch.zeros((0, 6), dtype=torch.float32)

    h, w = spec["res_hw"]
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


def _avg_recall_between(snr_bins: Sequence[float], recall: Sequence[float], a: float, b: float) -> float:
    snr_bins = np.asarray(snr_bins, dtype=float)
    recall = np.asarray(recall, dtype=float)
    left = max(a, float(snr_bins[0]))
    right = min(b, float(snr_bins[-1]))
    if right <= left:
        return float("nan")

    area = 0.0
    for idx, value in enumerate(recall):
        bin_left = snr_bins[idx]
        bin_right = snr_bins[idx + 1]
        overlap_left = max(bin_left, left)
        overlap_right = min(bin_right, right)
        width = max(0.0, overlap_right - overlap_left)
        if width > 0:
            area += float(value) * width
    return float(area / max(right - left, 1e-12))


def _compute_full_metrics(stats: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    f1_stats = precision_recall_stats(
        stats,
        to_plot=False,
        with_classes=True,
        class_index_to_name=CLASS_INDEX_TO_NAME,
    )

    fusion_conf_thresh = 0.0
    for thr, precision in zip(f1_stats["thr"], f1_stats["precision"]):
        if (1.0 - precision) <= FALSE_ALARM_TARGET:
            fusion_conf_thresh = float(thr)
            break

    if FINAL_CONF_MODE == "keep_model_thresholds":
        applied_conf_thresh = 0.0
    elif FINAL_CONF_MODE == "fusion_1pct_fa":
        applied_conf_thresh = fusion_conf_thresh
    else:
        raise ValueError(
            f"FINAL_CONF_MODE='{FINAL_CONF_MODE}' invalide. "
            "Choisis 'keep_model_thresholds' ou 'fusion_1pct_fa'."
        )

    recall_snr = recall_per_snr_bin(
        stats,
        snr_bins=range(-20, 20),
        to_plot=False,
        with_classes=True,
        class_index_to_name=CLASS_INDEX_TO_NAME,
        conf_thresh=applied_conf_thresh,
    )
    recall_size = recall_per_size_bin(
        stats,
        conf_thresh=applied_conf_thresh,
        with_classes=True,
        class_index_to_name=CLASS_INDEX_TO_NAME,
        snr_min=10,
    )
    recall_max_psnr = recall_per_max_psnr_bin(
        stats,
        psnr_bins=list(range(0, 31)),
        conf_thresh=applied_conf_thresh,
        with_classes=True,
        class_index_to_name=CLASS_INDEX_TO_NAME,
    )
    map_stats = map_from_stats(
        stats,
        iou_thresholds=np.linspace(0, 1, 21),
        to_plot=False,
        class_index_to_name=CLASS_INDEX_TO_NAME,
        conf_thresh=applied_conf_thresh,
        with_classes=True,
    )
    conf_matrix_high = confusion_matrix_snr(
        stats,
        snr_min=10,
        snr_max=20,
        class_index_to_name=CLASS_INDEX_TO_NAME,
        to_plot=False,
        conf_thresh=applied_conf_thresh,
        normalize="row",
    )
    conf_matrix_medium = confusion_matrix_snr(
        stats,
        snr_min=0,
        snr_max=10,
        class_index_to_name=CLASS_INDEX_TO_NAME,
        to_plot=False,
        conf_thresh=applied_conf_thresh,
        normalize="row",
    )
    conf_matrix_low = confusion_matrix_snr(
        stats,
        snr_min=-20,
        snr_max=0,
        class_index_to_name=CLASS_INDEX_TO_NAME,
        to_plot=False,
        conf_thresh=applied_conf_thresh,
        normalize="row",
    )

    summary = {
        "map50": map_stats["mAP50"],
        "map50_95": map_stats["mAP50:95"],
        "avg_recall_low_snr": _avg_recall_between(recall_snr["global"]["snr_bins"], recall_snr["global"]["recall"], -10.0, 19.0),
        "avg_recall_medium_snr": _avg_recall_between(recall_snr["global"]["snr_bins"], recall_snr["global"]["recall"], 0.0, 19.0),
        "avg_recall_high_snr": _avg_recall_between(recall_snr["global"]["snr_bins"], recall_snr["global"]["recall"], 10.0, 19.0),
        "final_conf_mode": FINAL_CONF_MODE,
        "fusion_1pct_conf_thresh": fusion_conf_thresh,
        "applied_conf_thresh": applied_conf_thresh,
    }

    return {
        "summary": summary,
        "f1_stats": f1_stats,
        "recall_snr": recall_snr,
        "recall_size_minsnr10db": recall_size,
        "recall_max_psnr": recall_max_psnr,
        "map_stats": map_stats,
        "conf_matrix_high_snr": conf_matrix_high,
        "conf_matrix_medium_snr": conf_matrix_medium,
        "conf_matrix_low_snr": conf_matrix_low,
    }


def main() -> None:
    if len(MODEL_SPECS) == 0:
        raise RuntimeError("MODEL_SPECS est vide. Ajoute au moins un modele.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = OUTPUT_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Chargement des modeles")
    models = []
    thresholds = {}
    resolved_specs = []

    for spec in MODEL_SPECS:
        checkpoint = Path(spec["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint introuvable: '{checkpoint}'.")

        model = _build_model(spec)
        model.load_weights(str(checkpoint), device=DEVICE, eval_mode=True)
        model.eval()
        models.append(model)

        resolved_spec = dict(spec)
        resolved_spec["checkpoint"] = str(checkpoint)
        resolved_specs.append(resolved_spec)

    print("[2/4] Recherche des seuils par modele pour 1% de fausse alarme")
    for spec in resolved_specs:
        threshold = _find_threshold_for_one_percent_fa(spec)
        thresholds[spec["label"]] = threshold
        print(f"  - {spec['label']}: conf_thresh = {threshold:.4f}")

    print("[3/4] Evaluation fusion NMS")
    sample_paths = sorted((DATASET_PATH / SPLIT / "data").glob("*.pt"))
    labels_dir = DATASET_PATH / SPLIT / "labels_detect"
    fusion_stats = {"tp": [], "fp": [], "fn": []}

    for sample_path in tqdm(sample_paths, desc="NMS fusion eval", unit="sample"):
        label_path = labels_dir / f"{sample_path.stem}.json"
        gt_boxes, gt_labels, gt_snrs, gt_psnrs = _load_gt(label_path)

        prediction_sets = []
        for model, spec in zip(models, resolved_specs):
            prediction_sets.append(
                _run_one_model_on_one_sample(
                    model=model,
                    spec=spec,
                    conf_thresh=thresholds[spec["label"]],
                    sample_path=sample_path,
                )
            )

        fusion_output = nms_fusion_post_nms(
            prediction_sets=prediction_sets,
            iou_thresh=FUSION_NMS_IOU,
            agnostic=FUSION_CLASS_AGNOSTIC,
        )

        fused_predictions = fusion_output["fused_predictions"]
        if len(fused_predictions) > 0:
            pred_boxes = fused_predictions[:, :4]
            pred_scores = fused_predictions[:, 4]
            pred_labels = fused_predictions[:, 5].long()
        else:
            pred_boxes = torch.zeros((0, 4), dtype=torch.float32)
            pred_scores = torch.zeros((0,), dtype=torch.float32)
            pred_labels = torch.zeros((0,), dtype=torch.long)

        sample_stats = analyse_results(
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
        fusion_stats["tp"].extend(sample_stats["tp"])
        fusion_stats["fp"].extend(sample_stats["fp"])
        fusion_stats["fn"].extend(sample_stats["fn"])

    print("[4/4] Calcul des metriques finales")
    metrics = _compute_full_metrics(fusion_stats)

    payload = {
        "dataset_path": str(DATASET_PATH),
        "split": SPLIT,
        "device": DEVICE,
        "false_alarm_target": FALSE_ALARM_TARGET,
        "fusion_nms_iou": FUSION_NMS_IOU,
        "eval_iou": EVAL_IOU,
        "fusion_class_agnostic": FUSION_CLASS_AGNOSTIC,
        "model_specs": resolved_specs,
        "model_thresholds_for_1pct_fa": thresholds,
        "fusion_metrics": metrics,
    }

    output_json = run_dir / "nms_fusion_eval.json"
    output_json.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")

    print("\n[Evaluation terminee]")
    print(f"Resultat JSON : {output_json}")
    print(f"mAP50         : {metrics['summary']['map50']:.6f}")
    print(f"mAP50:95      : {metrics['summary']['map50_95']:.6f}")
    print(f"Recall low    : {metrics['summary']['avg_recall_low_snr']:.6f}")
    print(f"Recall medium : {metrics['summary']['avg_recall_medium_snr']:.6f}")
    print(f"Recall high   : {metrics['summary']['avg_recall_high_snr']:.6f}")


if __name__ == "__main__":
    main()
