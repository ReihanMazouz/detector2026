from __future__ import annotations

import base64
import io
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from fastapi import HTTPException
from matplotlib import cm
from PIL import Image

from .config import PROJECT_ROOT


def _dataset_diagnostics(dataset_path: Path) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "exists": dataset_path.exists(),
        "is_dir": dataset_path.is_dir(),
        "splits": {},
    }

    if not dataset_path.exists() or not dataset_path.is_dir():
        return diagnostics

    for split_name in ("train", "val"):
        split_dir = dataset_path / split_name
        diagnostics["splits"][split_name] = {
            "exists": split_dir.is_dir(),
            "data_dir": (split_dir / "data").is_dir(),
            "labels_detect_dir": (split_dir / "labels_detect").is_dir(),
            "data_count": len(list((split_dir / "data").glob("*.pt"))) if (split_dir / "data").is_dir() else 0,
            "label_count": len(list((split_dir / "labels_detect").glob("*.json"))) if (split_dir / "labels_detect").is_dir() else 0,
        }
    return diagnostics


def resolve_dataset_path(path_value: str) -> Path:
    dataset_path = Path(path_value)
    if not dataset_path.is_absolute():
        dataset_path = (PROJECT_ROOT / dataset_path).resolve()
    return dataset_path


def _describe_series(values: List[float]) -> Dict[str, float]:
    if not values:
      return {}

    sorted_values = sorted(values)
    return {
        "count": len(sorted_values),
        "min": float(sorted_values[0]),
        "max": float(sorted_values[-1]),
        "mean": float(statistics.fmean(sorted_values)),
        "median": float(statistics.median(sorted_values)),
        "p90": float(sorted_values[min(len(sorted_values) - 1, int(0.9 * (len(sorted_values) - 1)))]),
    }


def _build_histogram(
    values: List[float],
    *,
    bins: int,
    lower: float,
    upper: float,
    value_key: str = "center",
) -> List[Dict[str, float]]:
    if bins <= 0:
        return []

    counts = [0 for _ in range(bins)]
    width = (upper - lower) / bins if upper > lower else 1

    for raw_value in values:
        value = min(max(raw_value, lower), upper)
        if value == upper:
            index = bins - 1
        else:
            index = int((value - lower) / width)
        counts[index] += 1

    histogram = []
    for index, count in enumerate(counts):
        start = lower + index * width
        end = start + width
        histogram.append(
            {
                "label": f"{start:.2f} - {end:.2f}",
                value_key: float(start + width / 2),
                "count": count,
            }
        )
    return histogram


def _build_auto_histogram(values: List[float], *, bins: int) -> List[Dict[str, float]]:
    if not values:
        return []

    lower = float(min(values))
    upper = float(max(values))
    if lower == upper:
        lower -= 0.5
        upper += 0.5
    return _build_histogram(values, bins=bins, lower=lower, upper=upper)


def _files_size(paths: List[Path]) -> int:
    return sum(path.stat().st_size for path in paths if path.is_file())


def _load_class_index_to_name(dataset_path: Path) -> Dict[int, str]:
    class_map_path = dataset_path / "class_index_to_name.json"
    if not class_map_path.is_file():
        return {}

    with class_map_path.open("r", encoding="utf-8") as handle:
        raw_mapping = json.load(handle)

    mapping: Dict[int, str] = {}
    for key, value in raw_mapping.items():
        try:
            mapping[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return mapping


def training_dataset_info(dataset_path: Path) -> Dict[str, Any]:
    diagnostics = _dataset_diagnostics(dataset_path)
    errors: List[str] = []

    if not dataset_path.exists():
        errors.append("Pas de dataset a ce chemin.")
    elif not dataset_path.is_dir():
        errors.append("Le chemin fourni n'est pas un dossier.")

    split_names = [name for name in ("train", "val") if (dataset_path / name).is_dir()] if dataset_path.is_dir() else []
    if dataset_path.is_dir() and not split_names:
        errors.append("Le dossier dataset ne contient pas de splits 'train' ou 'val'.")

    for split_name in ("train", "val"):
        split_diag = diagnostics.get("splits", {}).get(split_name, {})
        if split_diag.get("exists") and not split_diag.get("data_dir"):
            errors.append(f"Le split '{split_name}' ne contient pas de dossier 'data'.")
        if split_diag.get("exists") and not split_diag.get("labels_detect_dir"):
            errors.append(f"Le split '{split_name}' ne contient pas de dossier 'labels_detect'.")

    class_index_to_name = _load_class_index_to_name(dataset_path) if dataset_path.is_dir() else {}
    if dataset_path.is_dir() and not class_index_to_name:
        errors.append("Le fichier 'class_index_to_name.json' est absent ou vide.")

    num_classes = len(class_index_to_name)
    return {
        "dataset_path": str(dataset_path),
        "exists": dataset_path.exists(),
        "is_dir": dataset_path.is_dir(),
        "valid": len(errors) == 0,
        "errors": errors,
        "available_splits": split_names,
        "class_index_to_name": class_index_to_name,
        "num_classes": num_classes,
        "diagnostics": diagnostics,
    }


def _collect_split_stats(split_dir: Path, *, class_index_to_name: Dict[int, str]) -> Dict[str, Any]:
    data_files = sorted((split_dir / "data").glob("*.pt"))
    label_files = sorted((split_dir / "labels_detect").glob("*.json"))
    pulse_files = sorted((split_dir / "pulses").glob("*.npy"))
    emitter_files = sorted((split_dir / "emitters").glob("*.npy"))
    segment_files = sorted((split_dir / "labels_segment").glob("*.pt"))
    storage_breakdown = {
        "data": _files_size(data_files),
        "labels_detect": _files_size(label_files),
        "pulses": _files_size(pulse_files),
        "emitters": _files_size(emitter_files),
        "labels_segment": _files_size(segment_files),
    }

    class_counts: Counter = Counter()
    widths: List[float] = []
    heights: List[float] = []
    areas: List[float] = []
    aspect_ratios: List[float] = []
    snrs: List[float] = []
    boxes_per_sample: List[int] = []
    psnr_by_cfg: Dict[str, List[float]] = {}

    for label_path in label_files:
        with label_path.open("r", encoding="utf-8") as handle:
            labels = json.load(handle)

        if not isinstance(labels, list):
            continue

        boxes_per_sample.append(len(labels))
        for box in labels:
            if not isinstance(box, dict):
                continue

            class_id = int(box.get("class", -1))
            class_counts[class_id] += 1

            width = float(box.get("w", 0.0))
            height = float(box.get("h", 0.0))
            area = width * height

            widths.append(width)
            heights.append(height)
            areas.append(area)
            if height > 0:
                aspect_ratios.append(width / height)

            snr = box.get("snr")
            if isinstance(snr, (int, float)):
                snrs.append(float(snr))

            psnr = box.get("psnr", {})
            if isinstance(psnr, dict):
                for cfg_name, cfg_value in psnr.items():
                    if isinstance(cfg_value, (int, float)):
                        psnr_by_cfg.setdefault(str(cfg_name), []).append(float(cfg_value))

    total_boxes = sum(class_counts.values())
    class_distribution = []
    for class_id, count in class_counts.most_common():
        class_distribution.append(
            {
                "class_id": class_id,
                "class_name": class_index_to_name.get(class_id, f"Class {class_id}"),
                "count": count,
                "ratio": (count / total_boxes) if total_boxes else 0.0,
            }
        )

    psnr_summary = []
    for cfg_name in sorted(psnr_by_cfg):
        values = psnr_by_cfg[cfg_name]
        psnr_summary.append(
            {
                "cfg": cfg_name,
                "count": len(values),
                "mean": float(statistics.fmean(values)),
                "max": float(max(values)),
                "min": float(min(values)),
            }
        )

    return {
        "sample_count": len(label_files),
        "data_files": len(data_files),
        "label_files": len(label_files),
        "pulse_files": len(pulse_files),
        "emitter_files": len(emitter_files),
        "segment_files": len(segment_files),
        "box_count": total_boxes,
        "avg_boxes_per_sample": (total_boxes / len(label_files)) if label_files else 0.0,
        "class_distribution": class_distribution,
        "box_metrics": {
            "width": _describe_series(widths),
            "height": _describe_series(heights),
            "area": _describe_series(areas),
            "aspect_ratio": _describe_series(aspect_ratios),
            "snr": _describe_series(snrs),
            "boxes_per_sample": _describe_series([float(value) for value in boxes_per_sample]),
        },
        "histograms": {
            "width": _build_histogram(widths, bins=10, lower=0.0, upper=1.0),
            "height": _build_histogram(heights, bins=10, lower=0.0, upper=1.0),
            "area": _build_histogram(areas, bins=10, lower=0.0, upper=1.0),
            "snr": _build_auto_histogram(snrs, bins=12),
            "boxes_per_sample": _build_histogram(
                [float(value) for value in boxes_per_sample],
                bins=8,
                lower=0.0,
                upper=max(8.0, float(max(boxes_per_sample))) if boxes_per_sample else 8.0,
            ),
        },
        "psnr_by_cfg": psnr_summary,
        "storage": {
            "total_bytes": sum(storage_breakdown.values()),
            "by_folder": [
                {"folder": folder_name, "bytes": folder_size}
                for folder_name, folder_size in storage_breakdown.items()
            ],
        },
    }


def dataset_stats(dataset_path: Path, *, split: str) -> Dict[str, Any]:
    if not dataset_path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "message": "Dataset path not found.",
                "diagnostics": _dataset_diagnostics(dataset_path),
            },
        )
    if not dataset_path.is_dir():
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Dataset path is not a directory.",
                "diagnostics": _dataset_diagnostics(dataset_path),
            },
        )

    split_names = [name for name in ("train", "val") if (dataset_path / name).is_dir()]
    if not split_names:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Dataset directory does not contain 'train' or 'val' splits.",
                "diagnostics": _dataset_diagnostics(dataset_path),
            },
        )
    if split not in split_names:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Split '{split}' is not available.",
                "available_splits": split_names,
                "diagnostics": _dataset_diagnostics(dataset_path),
            },
        )

    class_index_to_name = _load_class_index_to_name(dataset_path)
    split_stats = _collect_split_stats(dataset_path / split, class_index_to_name=class_index_to_name)

    return {
        "dataset_path": str(dataset_path),
        "split_names": split_names,
        "split": split,
        "totals": {
            "sample_count": split_stats["sample_count"],
            "box_count": split_stats["box_count"],
            "class_count": len(split_stats["class_distribution"]),
            "avg_boxes_per_sample": split_stats["avg_boxes_per_sample"],
            "total_bytes": split_stats["storage"]["total_bytes"],
        },
        "split_stats": split_stats,
        "class_distribution": split_stats["class_distribution"],
        "psnr_overview": split_stats["psnr_by_cfg"],
        "class_index_to_name": class_index_to_name,
    }


def _split_dir(dataset_path: Path, split: str) -> Path:
    split_dir = dataset_path / split
    if not split_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Unknown split '{split}' for dataset.",
                "diagnostics": _dataset_diagnostics(dataset_path),
            },
        )
    return split_dir


def _sample_cfg_labels(labels: List[Dict[str, Any]], spectra_len: int) -> List[str]:
    if labels:
        psnr = labels[0].get("psnr")
        if isinstance(psnr, dict) and psnr:
            keys = list(psnr.keys())
            if len(keys) == spectra_len:
                return keys
    return [f"cfg{index}" for index in range(spectra_len)]


def sample_preview_payload(
    dataset_path: Path,
    *,
    split: str,
    sample_id: str,
    cfg_index: int,
) -> Dict[str, Any]:
    class_index_to_name = _load_class_index_to_name(dataset_path)
    split_root = _split_dir(dataset_path, split)
    label_path = split_root / "labels_detect" / f"{sample_id}.json"
    data_path = split_root / "data" / f"{sample_id}.pt"

    if not label_path.is_file():
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Label file not found for sample '{sample_id}'.",
                "sample_id": sample_id,
                "label_path": str(label_path),
            },
        )
    if not data_path.is_file():
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Data file not found for sample '{sample_id}'.",
                "sample_id": sample_id,
                "data_path": str(data_path),
            },
        )

    try:
        with label_path.open("r", encoding="utf-8") as handle:
            labels = json.load(handle)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Unable to parse label file.",
                "label_path": str(label_path),
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        ) from exc

    try:
        spectra = torch.load(data_path, map_location="cpu")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Unable to load tensor sample file.",
                "data_path": str(data_path),
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        ) from exc

    if not isinstance(spectra, list) or not spectra:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Sample '{sample_id}' does not contain spectrogram tensors.",
                "data_path": str(data_path),
                "payload_type": type(spectra).__name__,
            },
        )
    if cfg_index < 0 or cfg_index >= len(spectra):
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"cfg_index must be between 0 and {len(spectra) - 1}.",
                "cfg_index": cfg_index,
            },
        )

    tensor = spectra[cfg_index]
    if not isinstance(tensor, torch.Tensor):
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Invalid tensor payload for sample '{sample_id}'.",
                "tensor_type": type(tensor).__name__,
            },
        )

    # Avoid relying on PyTorch's direct NumPy bridge, which can fail depending
    # on the local torch/numpy build combination.
    array = np.asarray(tensor.detach().cpu().to(torch.float32).tolist(), dtype=np.float32)
    if array.ndim == 3:
        array = array[0]
    if array.ndim != 2:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Expected 2D spectrum for sample '{sample_id}'.",
                "shape": list(array.shape),
            },
        )

    vmax = float(array.max()) if array.size else 1.0
    normalized = np.zeros_like(array, dtype=np.float32) if vmax <= 0 else (array / vmax).astype(np.float32)
    rgba = cm.get_cmap("cividis")(normalized)
    image = Image.fromarray((rgba[:, :, :3] * 255).astype(np.uint8))

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    cfg_labels = _sample_cfg_labels(labels if isinstance(labels, list) else [], len(spectra))
    boxes = []
    classes_present: Counter = Counter()
    for box in labels if isinstance(labels, list) else []:
        if not isinstance(box, dict):
            continue
        class_id = int(box.get("class", -1))
        class_name = class_index_to_name.get(class_id, f"Class {class_id}")
        boxes.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "xc": float(box.get("xc", 0)),
                "yc": float(box.get("yc", 0)),
                "w": float(box.get("w", 0)),
                "h": float(box.get("h", 0)),
                "snr": float(box.get("snr", 0)) if isinstance(box.get("snr"), (int, float)) else None,
                "psnr": {
                    key: float(value)
                    for key, value in box.get("psnr", {}).items()
                    if isinstance(value, (int, float))
                } if isinstance(box.get("psnr"), dict) else {},
            }
        )
        classes_present[class_name] += 1

    return {
        "dataset_path": str(dataset_path),
        "split": split,
        "sample_id": sample_id,
        "cfg_index": cfg_index,
        "cfg_labels": cfg_labels,
        "image": {
            "data_url": image_data_url,
            "width": int(image.width),
            "height": int(image.height),
        },
        "boxes": boxes,
        "class_names": list(classes_present.keys()),
        "box_count": len(boxes),
    }


def list_dataset_examples(
    dataset_path: Path, *, split: str, offset: int = 0, limit: int = 200
) -> Dict[str, Any]:
    class_index_to_name = _load_class_index_to_name(dataset_path)
    split_root = _split_dir(dataset_path, split)
    label_dir = split_root / "labels_detect"
    data_dir = split_root / "data"

    samples = []
    for label_path in sorted(label_dir.glob("*.json")):
        sample_id = label_path.stem
        data_path = data_dir / f"{sample_id}.pt"
        if not data_path.is_file():
            continue

        try:
            with label_path.open("r", encoding="utf-8") as handle:
                labels = json.load(handle)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Unable to parse one of the label files.",
                    "label_path": str(label_path),
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            ) from exc

        class_names = []
        if isinstance(labels, list):
            class_ids = sorted({int(item.get("class", -1)) for item in labels if isinstance(item, dict)})
            class_names = [class_index_to_name.get(class_id, f"Class {class_id}") for class_id in class_ids]
        else:
            labels = []

        samples.append(
            {
                "sample_id": sample_id,
                "box_count": len(labels),
                "class_names": class_names,
            }
        )

    total = len(samples)
    start = max(0, int(offset))
    page_size = max(1, min(int(limit), 1000))
    end = min(total, start + page_size)
    page_samples = samples[start:end]

    return {
        "dataset_path": str(dataset_path),
        "split": split,
        "total": total,
        "offset": start,
        "limit": page_size,
        "has_more": end < total,
        "samples": page_samples,
    }
