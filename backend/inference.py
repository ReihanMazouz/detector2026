from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from fastapi import HTTPException

from .config import PROJECT_ROOT
from .datasets import _load_class_index_to_name, _sample_cfg_labels, sample_preview_payload

WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from detector2026.core.models.mr_yolo import MR_YOLO
from detector2026.core.models.tf_attn_yolo import TF_Attn_Yolo
from detector2026.core.models.yolov8 import YOLOv8
from detector2026.core.models.yolov11 import YOLOv11
from detector2026.core.utils.analysing_results import analyse_results


def _resolve_checkpoint_path(path_value: str) -> Path:
    checkpoint_path = Path(path_value).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = (PROJECT_ROOT / checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise HTTPException(status_code=404, detail=f"Checkpoint introuvable: '{path_value}'.")
    return checkpoint_path


def _load_run_config(run_dir: Path) -> Dict[str, Any] | None:
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        return None
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _infer_num_classes_from_checkpoint(checkpoint_path: Path) -> int | None:
    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        return None

    final_candidates: List[int] = []
    fallback_candidates: List[int] = []
    for key, value in state_dict.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            continue
        if "detect.cv_clsobj" in key and key.endswith(".weight") and value.ndim == 4:
            out_channels = int(value.shape[0])
            fallback_candidates.append(out_channels)
            if re.search(r"detect\.cv_clsobj\.\d+\.2\.weight$", key):
                final_candidates.append(out_channels)

    if final_candidates:
        return final_candidates[0]
    if fallback_candidates:
        return min(fallback_candidates)
    return None


def _read_model_summary(run_dir: Path) -> str:
    summary_path = run_dir / "model_summary.txt"
    if not summary_path.is_file():
        raise HTTPException(status_code=400, detail="model_summary.txt introuvable dans le dossier du run.")
    return summary_path.read_text(encoding="utf-8", errors="ignore")


def _summary_model_name(summary_text: str) -> str:
    match = re.search(r"^# Model:\s*(.+)$", summary_text, flags=re.MULTILINE)
    if not match:
        raise HTTPException(status_code=400, detail="Impossible d'identifier le type de modele depuis model_summary.txt.")
    return match.group(1).strip()


def _infer_width_mult(summary_text: str, default: float = 0.25) -> float:
    match = re.search(r"Conv2d\(\d+,\s*(\d+),\s*kernel_size=", summary_text)
    if not match:
        return default
    out_channels = int(match.group(1))
    if out_channels <= 0:
        return default
    return max(out_channels / 64.0, 0.125)


def _find_sample_data_path(dataset_path: Path, split: str, sample_id: str) -> Path:
    data_path = dataset_path / split / "data" / f"{sample_id}.pt"
    if not data_path.is_file():
        raise HTTPException(status_code=404, detail=f"Sample tensor introuvable pour '{sample_id}'.")
    return data_path


def _find_sample_label_path(dataset_path: Path, split: str, sample_id: str) -> Path:
    label_path = dataset_path / split / "labels_detect" / f"{sample_id}.json"
    if not label_path.is_file():
        raise HTTPException(status_code=404, detail=f"Label introuvable pour '{sample_id}'.")
    return label_path


def _load_spectra(data_path: Path) -> List[torch.Tensor]:
    try:
        raw = torch.load(data_path, map_location="cpu")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Impossible de charger le sample '{data_path.name}': {exc}") from exc

    if isinstance(raw, list):
        tensors = raw
    else:
        tensors = [raw]

    prepared: List[torch.Tensor] = []
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise HTTPException(status_code=400, detail=f"Contenu invalide dans '{data_path.name}'.")
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4:
            raise HTTPException(status_code=400, detail=f"Shape non supportee pour l'inference: {list(tensor.shape)}")
        prepared.append(tensor.to(torch.float32))
    return prepared


def _build_model_from_config(config: Dict[str, Any], checkpoint_path: Path, num_classes: int, input_resolutions: List[Tuple[int, int]]):
    model_id = str(config.get("model_id", ""))
    model_config = dict(config.get("model_config", {}))
    output_dir = str(checkpoint_path.parent)
    device = "cpu"
    reg_max = int(model_config.get("reg_max", 16))
    width_mult = float(model_config.get("width_mult", 0.5))
    anisotropic = bool(model_config.get("anisotropic", False))
    p3_size = tuple(model_config.get("p3_size", [64, 64]))
    input_hw = input_resolutions[0] if input_resolutions else None

    if model_id == "mr_yolo":
        return MR_YOLO(
            input_resolutions=input_resolutions,
            num_classes=num_classes,
            device=device,
            reg_max=reg_max,
            output_dir=output_dir,
            width_mult=width_mult,
            backbone_mode=str(model_config.get("backbone_mode", "TFSep_pyramid")),
            outfusion_channels_mult=int(model_config.get("outfusion_channels_mult", 1)),
        )
    if model_id == "yolov8":
        return YOLOv8(
            num_classes=num_classes,
            device=device,
            reg_max=reg_max,
            output_dir=output_dir,
            width_mult=width_mult,
            anisotropic=anisotropic,
            p3_size=p3_size,
            input_hw=input_hw,
        )
    if model_id == "yolov11":
        return YOLOv11(
            num_classes=num_classes,
            device=device,
            reg_max=reg_max,
            output_dir=output_dir,
            width_mult=width_mult,
            anisotropic=anisotropic,
            p3_size=p3_size,
            input_hw=input_hw,
        )
    if model_id == "tf_attn_yolo":
        return TF_Attn_Yolo(
            num_classes=num_classes,
            device=device,
            reg_max=reg_max,
            output_dir=output_dir,
            width_mult=width_mult,
            anisotropic=anisotropic,
            p3_size=p3_size,
            input_hw=input_hw,
        )
    raise HTTPException(status_code=400, detail=f"model_id non supporte pour l'inference: '{model_id}'.")


def _build_model_from_summary(summary_text: str, checkpoint_path: Path, num_classes: int, input_resolutions: List[Tuple[int, int]]):
    model_name = _summary_model_name(summary_text)
    width_mult = _infer_width_mult(summary_text)
    output_dir = str(checkpoint_path.parent)
    device = "cpu"
    reg_max = 16

    if model_name == "TF_Attn_Yolo":
        return TF_Attn_Yolo(
            output_dir=output_dir,
            num_classes=num_classes,
            reg_max=reg_max,
            device=device,
            width_mult=width_mult,
        )
    if model_name == "YOLOv11":
        return YOLOv11(
            output_dir=output_dir,
            num_classes=num_classes,
            reg_max=reg_max,
            device=device,
            width_mult=width_mult,
        )
    if model_name == "YOLOv8":
        return YOLOv8(
            output_dir=output_dir,
            num_classes=num_classes,
            reg_max=reg_max,
            device=device,
            width_mult=width_mult,
        )
    if model_name == "MR_YOLO":
        return MR_YOLO(
            input_resolutions=input_resolutions,
            output_dir=output_dir,
            num_classes=num_classes,
            reg_max=reg_max,
            device=device,
            width_mult=width_mult if width_mult else 0.5,
            backbone_mode="TFSep_pyramid",
            outfusion_channels_mult=1,
        )
    raise HTTPException(status_code=400, detail=f"Modele non supporte depuis model_summary.txt: '{model_name}'.")


def _build_model(checkpoint_path: Path, dataset_path: Path, split: str, sample_id: str, num_classes: int):
    run_dir = checkpoint_path.parent
    data_path = _find_sample_data_path(dataset_path, split, sample_id)
    sample_tensors = _load_spectra(data_path)
    input_resolutions = [(int(tensor.shape[-2]), int(tensor.shape[-1])) for tensor in sample_tensors]

    config = _load_run_config(run_dir)
    if config is not None:
        model = _build_model_from_config(config, checkpoint_path, num_classes, input_resolutions)
    else:
        summary_text = _read_model_summary(run_dir)
        model = _build_model_from_summary(summary_text, checkpoint_path, num_classes, input_resolutions)

    model.load_weights(str(checkpoint_path), device="cpu", eval_mode=True)
    return model, sample_tensors


def _config_specific_res_hw(config: Dict[str, Any] | None) -> Tuple[int, int] | None:
    if not config:
        return None
    data_loading = dict(config.get("data_loading", {}))
    if str(data_loading.get("dataset_mode", "")).lower() == "specificres":
        select_res = dict(data_loading.get("select_res", {}))
        res_hw = select_res.get("res_hw")
    else:
        training = dict(config.get("training", {}))
        if str(training.get("dataset_mode", "")).lower() != "specificres":
            return None
        model_config = dict(config.get("model_config", {}))
        res_hw = model_config.get("res_hw")
    if not isinstance(res_hw, (list, tuple)) or len(res_hw) != 2:
        return None
    return int(res_hw[0]), int(res_hw[1])


def _find_cfg_index_for_res_hw(sample_tensors: List[torch.Tensor], res_hw: Tuple[int, int]) -> int:
    target_h, target_w = int(res_hw[0]), int(res_hw[1])
    for idx, tensor in enumerate(sample_tensors):
        if tuple(tensor.shape[-2:]) == (target_h, target_w):
            return idx
    raise HTTPException(
        status_code=400,
        detail=f"Aucun tenseur de taille {res_hw} dans le sample pour ce run specificres.",
    )


def _infer_specific_res_key(config: Dict[str, Any] | None, run_dir: Path) -> str | None:
    if config:
        data_loading = dict(config.get("data_loading", {}))
        select_res = dict(data_loading.get("select_res", {}))
        res_key = select_res.get("res_key")
        if isinstance(res_key, str) and res_key:
            return res_key

        training = dict(config.get("training", {}))
        if str(training.get("dataset_mode", "")).lower() == "specificres":
            model_config = dict(config.get("model_config", {}))
            fallback_res_key = model_config.get("res_key")
            if isinstance(fallback_res_key, str) and fallback_res_key:
                return fallback_res_key

    run_name = run_dir.name
    match = re.search(r"(?:specificres_|cfg)(\d+)", run_name)
    if match:
        return f"cfg{match.group(1)}"
    return None


def _find_cfg_index_for_res_key(label_path: Path, spectra_len: int, res_key: str) -> int:
    try:
        labels = json.loads(label_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"Impossible de lire les labels pour resoudre '{res_key}': {exc}") from exc

    cfg_labels = _sample_cfg_labels(labels if isinstance(labels, list) else [], spectra_len)
    if res_key not in cfg_labels:
        raise HTTPException(
            status_code=400,
            detail=f"Le sample ne contient pas de mapping cfg vers '{res_key}'. Valeurs disponibles: {cfg_labels}",
        )
    return cfg_labels.index(res_key)


def _prediction_boxes(processed_output: List[torch.Tensor], class_index_to_name: Dict[int, str], width: int, height: int) -> List[Dict[str, Any]]:
    detections = processed_output[0] if processed_output else None
    if detections is None or len(detections) == 0:
        return []

    predictions: List[Dict[str, Any]] = []
    for det in detections.detach().cpu():
        x1, y1, x2, y2, score, class_id = det.tolist()
        class_id = int(class_id)
        x1 = max(0.0, min(float(x1), float(width)))
        x2 = max(0.0, min(float(x2), float(width)))
        y1 = max(0.0, min(float(y1), float(height)))
        y2 = max(0.0, min(float(y2), float(height)))
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        predictions.append(
            {
                "class_id": class_id,
                "class_name": class_index_to_name.get(class_id, f"Class {class_id}"),
                "confidence": float(score),
                "xc": ((x1 + x2) / 2.0) / max(width, 1),
                "yc": ((y1 + y2) / 2.0) / max(height, 1),
                "w": w / max(width, 1),
                "h": h / max(height, 1),
            }
        )
    return predictions


def _xyxy_to_xywh_payload(box_xyxy: List[float], class_id: int, class_name: str, **extra: Any) -> Dict[str, Any]:
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    payload = {
        "class_id": int(class_id),
        "class_name": class_name,
        "xc": (x1 + x2) / 2.0,
        "yc": (y1 + y2) / 2.0,
        "w": width,
        "h": height,
    }
    payload.update(extra)
    return payload


def _analysis_payload(
    *,
    preview_boxes: List[Dict[str, Any]],
    processed_output: List[torch.Tensor],
    class_index_to_name: Dict[int, str],
    image_width: int,
    image_height: int,
    iou_thres: float,
) -> Dict[str, Any]:
    detections = processed_output[0] if processed_output else None

    if detections is not None and len(detections):
        pred_boxes = torch.stack(
            [
                detections[:, 0] / float(image_width),
                detections[:, 1] / float(image_height),
                detections[:, 2] / float(image_width),
                detections[:, 3] / float(image_height),
            ],
            dim=1,
        ).cpu()
        pred_scores = detections[:, 4].cpu()
        pred_labels = detections[:, 5].long().cpu()
    else:
        pred_boxes = torch.zeros((0, 4), dtype=torch.float32)
        pred_scores = torch.zeros((0,), dtype=torch.float32)
        pred_labels = torch.zeros((0,), dtype=torch.long)

    if preview_boxes:
        gt_boxes = []
        gt_labels = []
        gt_snrs = []
        for box in preview_boxes:
            gt_boxes.append([
                float(box["xc"]) - float(box["w"]) / 2.0,
                float(box["yc"]) - float(box["h"]) / 2.0,
                float(box["xc"]) + float(box["w"]) / 2.0,
                float(box["yc"]) + float(box["h"]) / 2.0,
            ])
            gt_labels.append(int(box["class_id"]))
            gt_snrs.append(float(box["snr"]) if box.get("snr") is not None else 0.0)
        gt_boxes_tensor = torch.tensor(gt_boxes, dtype=torch.float32)
        gt_labels_tensor = torch.tensor(gt_labels, dtype=torch.long)
        gt_snrs_tensor = torch.tensor(gt_snrs, dtype=torch.float32)
    else:
        gt_boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        gt_labels_tensor = torch.zeros((0,), dtype=torch.long)
        gt_snrs_tensor = torch.zeros((0,), dtype=torch.float32)

    analysis_raw = analyse_results(
        pred_boxes=pred_boxes,
        pred_scores=pred_scores,
        pred_labels=pred_labels,
        gt_boxes=gt_boxes_tensor,
        gt_labels=gt_labels_tensor,
        gt_snrs=gt_snrs_tensor,
        iou_thresh=iou_thres,
    )

    tp_items = [
        _xyxy_to_xywh_payload(
            item["pred_box"],
            class_id=int(item["label"]),
            class_name=class_index_to_name.get(int(item["label"]), f"Class {int(item['label'])}"),
            confidence=float(item["score"]),
            iou=float(item["max_iou"]),
        )
        for item in analysis_raw["tp"]
    ]
    fp_items = [
        _xyxy_to_xywh_payload(
            item["pred_box"],
            class_id=int(item["label"]),
            class_name=class_index_to_name.get(int(item["label"]), f"Class {int(item['label'])}"),
            confidence=float(item["score"]),
        )
        for item in analysis_raw["fp"]
    ]
    fn_items = [
        _xyxy_to_xywh_payload(
            item["gt_box"],
            class_id=int(item["label"]),
            class_name=class_index_to_name.get(int(item["label"]), f"Class {int(item['label'])}"),
            snr=float(item["snr"]) if item.get("snr") is not None else None,
        )
        for item in analysis_raw["fn"]
    ]

    tp_count = len(tp_items)
    fp_count = len(fp_items)
    fn_count = len(fn_items)
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) else 0.0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) else 0.0
    avg_conf = sum(item["confidence"] for item in tp_items + fp_items) / (tp_count + fp_count) if (tp_count + fp_count) else 0.0
    avg_iou = sum(item["iou"] for item in tp_items) / tp_count if tp_count else 0.0

    return {
        "tp": tp_items,
        "fp": fp_items,
        "fn": fn_items,
        "summary": {
            "tp_count": tp_count,
            "fp_count": fp_count,
            "fn_count": fn_count,
            "precision": precision,
            "recall": recall,
            "average_confidence": avg_conf,
            "average_iou": avg_iou,
        },
    }


def _candidate_rank_key(candidate: Dict[str, Any]) -> Tuple[float, float, float, float]:
    summary = dict(candidate.get("analysis", {}).get("summary", {}))
    return (
        float(summary.get("tp_count", 0)),
        -float(summary.get("fn_count", 0)),
        -float(summary.get("fp_count", 0)),
        float(summary.get("average_confidence", 0.0)),
    )


def artifacts_preview_payload(
    dataset_path: Path,
    *,
    split: str,
    sample_id: str,
    cfg_index: int,
    checkpoint_path: str,
    conf_thres: float = 0.1,
    iou_thres: float = 0.1,
) -> Dict[str, Any]:
    resolved_checkpoint = _resolve_checkpoint_path(checkpoint_path)
    run_config = _load_run_config(resolved_checkpoint.parent)
    data_path = _find_sample_data_path(dataset_path, split, sample_id)
    label_path = _find_sample_label_path(dataset_path, split, sample_id)
    sample_tensors = _load_spectra(data_path)

    effective_cfg_index = cfg_index
    specific_res_hw = _config_specific_res_hw(run_config)
    if specific_res_hw is not None:
        effective_cfg_index = _find_cfg_index_for_res_hw(sample_tensors, specific_res_hw)
    else:
        specific_res_key = _infer_specific_res_key(run_config, resolved_checkpoint.parent)
        if specific_res_key is not None:
            effective_cfg_index = _find_cfg_index_for_res_key(label_path, len(sample_tensors), specific_res_key)

    class_index_to_name = _load_class_index_to_name(dataset_path)
    config_num_classes = None
    if run_config is not None:
        model_config = dict(run_config.get("model_config", {}))
        if model_config.get("num_classes") is not None:
            config_num_classes = int(model_config["num_classes"])
    checkpoint_num_classes = _infer_num_classes_from_checkpoint(resolved_checkpoint)
    num_classes = int(config_num_classes or checkpoint_num_classes or max(len(class_index_to_name), 1))

    model, sample_tensors = _build_model(resolved_checkpoint, dataset_path, split, sample_id, num_classes)

    candidate_payloads: List[Dict[str, Any]] = []
    if isinstance(model, MR_YOLO):
        preview = sample_preview_payload(dataset_path, split=split, sample_id=sample_id, cfg_index=effective_cfg_index)
        processed_output, _, _ = model.predict(
            sample_tensors,
            conf_threshold=conf_thres,
            iou_thres=iou_thres,
        )
        predictions = _prediction_boxes(
            processed_output,
            class_index_to_name,
            width=int(preview["image"]["width"]),
            height=int(preview["image"]["height"]),
        )
        analysis = _analysis_payload(
            preview_boxes=preview["boxes"],
            processed_output=processed_output,
            class_index_to_name=class_index_to_name,
            image_width=int(preview["image"]["width"]),
            image_height=int(preview["image"]["height"]),
            iou_thres=iou_thres,
        )
    else:
        auto_select_legacy_cfg = run_config is None and len(sample_tensors) > 1
        candidate_indices = range(len(sample_tensors)) if auto_select_legacy_cfg else [effective_cfg_index]
        for candidate_idx in candidate_indices:
            if candidate_idx < 0 or candidate_idx >= len(sample_tensors):
                continue
            candidate_preview = sample_preview_payload(dataset_path, split=split, sample_id=sample_id, cfg_index=candidate_idx)
            candidate_output, _, _ = model.predict(
                sample_tensors[candidate_idx],
                conf_threshold=conf_thres,
                iou_thres=iou_thres,
            )
            candidate_predictions = _prediction_boxes(
                candidate_output,
                class_index_to_name,
                width=int(candidate_preview["image"]["width"]),
                height=int(candidate_preview["image"]["height"]),
            )
            candidate_analysis = _analysis_payload(
                preview_boxes=candidate_preview["boxes"],
                processed_output=candidate_output,
                class_index_to_name=class_index_to_name,
                image_width=int(candidate_preview["image"]["width"]),
                image_height=int(candidate_preview["image"]["height"]),
                iou_thres=iou_thres,
            )
            candidate_payloads.append(
                {
                    "cfg_index": int(candidate_idx),
                    "preview": candidate_preview,
                    "processed_output": candidate_output,
                    "predictions": candidate_predictions,
                    "analysis": candidate_analysis,
                }
            )

        if not candidate_payloads:
            raise HTTPException(status_code=400, detail="Aucune entree candidate valide pour l'inference.")

        best_candidate = max(candidate_payloads, key=_candidate_rank_key)
        effective_cfg_index = int(best_candidate["cfg_index"])
        preview = best_candidate["preview"]
        processed_output = best_candidate["processed_output"]
        predictions = best_candidate["predictions"]
        analysis = best_candidate["analysis"]

    preview["checkpoint_path"] = str(resolved_checkpoint)
    preview["requested_cfg_index"] = int(cfg_index)
    preview["cfg_index"] = int(effective_cfg_index)
    preview["predictions"] = predictions
    preview["prediction_count"] = len(predictions)
    preview["analysis"] = analysis
    preview["inference"] = {
        "conf_thres": conf_thres,
        "iou_thres": iou_thres,
        "device": str(model.device),
        "model_name": model.__class__.__name__,
        "effective_cfg_index": int(effective_cfg_index),
    }
    if candidate_payloads:
        preview["inference"]["cfg_candidates"] = [
            {
                "cfg_index": int(candidate["cfg_index"]),
                "prediction_count": int(len(candidate["predictions"])),
                "summary": dict(candidate["analysis"]["summary"]),
            }
            for candidate in candidate_payloads
        ]
        preview["inference"]["auto_selected_cfg"] = True
    return preview
