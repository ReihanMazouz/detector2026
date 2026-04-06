from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi import HTTPException

from .config import RUNS_ROOT


METRIC_KEYS = [
    "train_loss",
    "val_loss",
    "loss_box_train",
    "loss_cls_train",
    "loss_dfl_train",
    "loss_box_val",
    "loss_cls_val",
    "loss_dfl_val",
    "map50",
    "map50_95",
    "avg_recall_low_snr",
    "avg_recall_medium_snr",
    "avg_recall_high_snr",
]

EVAL_ARTIFACTS = {
    "oracle_eval.json": "oracle_metrics",
    "nms_fusion_eval.json": "fusion_metrics",
}


def _safe_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _candidate_run_dirs() -> List[Path]:
    roots = [RUNS_ROOT / "examples_of_training", RUNS_ROOT]
    seen: set[Path] = set()
    run_dirs: List[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if not child.is_dir() or child in seen:
                continue
            if (child / "train_log.csv").is_file() or _find_eval_artifact(child) is not None:
                seen.add(child)
                run_dirs.append(child)
    return sorted(run_dirs, key=lambda path: path.name.lower())


def _find_eval_artifact(run_dir: Path) -> tuple[Path, str] | None:
    for filename, metrics_key in EVAL_ARTIFACTS.items():
        candidate = run_dir / filename
        if candidate.is_file():
            return candidate, metrics_key
    return None


def _load_eval_artifact(run_dir: Path) -> Dict[str, Any] | None:
    artifact = _find_eval_artifact(run_dir)
    if artifact is None:
        return None

    artifact_path, metrics_key = artifact
    try:
        raw_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid evaluation artifact in '{artifact_path.name}': {exc}") from exc

    metrics_payload = raw_payload.get(metrics_key)
    if not isinstance(metrics_payload, dict):
        raise HTTPException(status_code=400, detail=f"Missing '{metrics_key}' in '{artifact_path.name}'.")

    summary = metrics_payload.get("summary")
    if not isinstance(summary, dict):
        raise HTTPException(status_code=400, detail=f"Missing summary in '{artifact_path.name}'.")

    return {
        "artifact_path": artifact_path,
        "metrics_key": metrics_key,
        "raw_payload": raw_payload,
        "metrics_payload": metrics_payload,
        "summary": summary,
    }


def _read_train_log_rows(run_dir: Path) -> List[Dict[str, str]]:
    csv_path = run_dir / "train_log.csv"
    if not csv_path.is_file():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _resolve_run_dir(run_path: str) -> Path:
    path = Path(run_path).expanduser()
    if not path.is_absolute():
        path = (RUNS_ROOT / path).resolve()
    if not path.is_dir():
        raise HTTPException(status_code=404, detail=f"Unknown run path '{run_path}'.")
    return path


def _epoch_rows(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    epoch_rows: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        item: Dict[str, Any] = {"epoch": int(float(row.get("epoch", index + 1)))}
        for key in METRIC_KEYS:
            item[key] = _safe_float(row.get(key))
        item["metrics_json_path"] = row.get("metrics_json_path")
        epoch_rows.append(item)
    return epoch_rows


def _best_epoch(epoch_rows: List[Dict[str, Any]], metric_key: str, mode: str) -> Dict[str, Any] | None:
    valid_rows = [row for row in epoch_rows if row.get(metric_key) is not None]
    if not valid_rows:
        return None
    if mode == "min":
        return min(valid_rows, key=lambda row: row[metric_key])
    return max(valid_rows, key=lambda row: row[metric_key])


def _snapshot_from_row(row: Dict[str, Any] | None, metric_key: str, label: str) -> Dict[str, Any] | None:
    if row is None:
        return None
    return {
        "label": label,
        "epoch": row.get("epoch"),
        "metric_key": metric_key,
        "metric_value": row.get(metric_key),
        "metrics": {key: row.get(key) for key in METRIC_KEYS},
        "metrics_json_path": row.get("metrics_json_path"),
    }


def _metrics_json_payload(run_dir: Path, row: Dict[str, Any] | None) -> Dict[str, Any]:
    if row is None:
        return {}

    metrics_json_path = row.get("metrics_json_path")
    candidate_paths: List[Path] = []

    if metrics_json_path:
        raw_path = Path(str(metrics_json_path))
        if raw_path.is_absolute():
            candidate_paths.append(raw_path)
        else:
            candidate_paths.append((run_dir / raw_path).resolve())

    epoch = row.get("epoch")
    if epoch is not None:
        candidate_paths.append(run_dir / "metrics" / f"metrics_epoch_{int(epoch):03d}.json")

    seen: set[Path] = set()
    for path in candidate_paths:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

    return {}


def _extract_model_info(run_dir: Path, row: Dict[str, Any] | None) -> Dict[str, Any]:
    payload = _metrics_json_payload(run_dir, row)
    model_info = payload.get("model_info")
    if not isinstance(model_info, dict):
        return {"params": None, "flops": None}
    return {
        "params": _safe_float(model_info.get("params")),
        "flops": _safe_float(model_info.get("flops")),
    }


def _run_summary(run_dir: Path) -> Dict[str, Any]:
    raw_rows = _read_train_log_rows(run_dir)
    if not raw_rows:
        eval_artifact = _load_eval_artifact(run_dir)
        if eval_artifact is None:
            raise HTTPException(status_code=404, detail=f"No train_log.csv found for run '{run_dir.name}'.")

        summary = eval_artifact["summary"]
        artifact_path = eval_artifact["artifact_path"]
        metric_snapshot = {
            "train_loss": None,
            "val_loss": None,
            "map50": _safe_float(summary.get("map50")),
            "map50_95": _safe_float(summary.get("map50_95")),
            "avg_recall_low_snr": _safe_float(summary.get("avg_recall_low_snr")),
            "avg_recall_medium_snr": _safe_float(summary.get("avg_recall_medium_snr")),
            "avg_recall_high_snr": _safe_float(summary.get("avg_recall_high_snr")),
        }
        eval_label = "oracle eval" if eval_artifact["metrics_key"] == "oracle_metrics" else "nms fusion eval"
        snapshot_base = {
            "epoch": 1,
            "metrics_json_path": str(artifact_path),
            "metrics": metric_snapshot,
        }

        return {
            "run_name": run_dir.name,
            "path": str(run_dir),
            "source": "examples" if "examples_of_training" in str(run_dir) else "runs",
            "epochs_completed": 1,
            "final_epoch": 1,
            "final_metrics": metric_snapshot,
            "final_model_info": {"params": None, "flops": None},
            "best_snapshots": {
                "checkpoint": {
                    **snapshot_base,
                    "label": eval_label,
                    "metric_key": "map50_95",
                    "metric_value": metric_snapshot["map50_95"],
                    "model_info": {"params": None, "flops": None},
                },
                "val_loss": None,
                "map50": {
                    **snapshot_base,
                    "label": "best mAP50",
                    "metric_key": "map50",
                    "metric_value": metric_snapshot["map50"],
                },
                "avg_recall_low_snr": {
                    **snapshot_base,
                    "label": "best recall low SNR",
                    "metric_key": "avg_recall_low_snr",
                    "metric_value": metric_snapshot["avg_recall_low_snr"],
                },
            },
            "checkpoint_policy": {
                "best_checkpoint_metric": "map50_95",
                "best_checkpoint_mode": "max",
                "early_stopping_metric": None,
                "early_stopping_mode": None,
            },
            "artifacts": {
                "best_pt": False,
                "last_pt": False,
                "loss_curves": False,
                "map_curves": False,
                "avg_recall_curves": False,
                "model_summary": False,
                "evaluation_artifact": artifact_path.name,
            },
        }

    epoch_rows = _epoch_rows(raw_rows)
    last_row = epoch_rows[-1]
    best_checkpoint_row = _best_epoch(epoch_rows, "map50_95", "max")
    best_val_loss_row = _best_epoch(epoch_rows, "val_loss", "min")
    best_map50_row = _best_epoch(epoch_rows, "map50", "max")
    best_recall_row = _best_epoch(epoch_rows, "avg_recall_low_snr", "max")
    best_checkpoint_model_info = _extract_model_info(run_dir, best_checkpoint_row)
    final_model_info = _extract_model_info(run_dir, last_row)

    return {
        "run_name": run_dir.name,
        "path": str(run_dir),
        "source": "examples" if "examples_of_training" in str(run_dir) else "runs",
        "epochs_completed": len(epoch_rows),
        "final_epoch": int(last_row.get("epoch", len(epoch_rows))),
        "final_metrics": {
            "train_loss": last_row.get("train_loss"),
            "val_loss": last_row.get("val_loss"),
            "map50": last_row.get("map50"),
            "map50_95": last_row.get("map50_95"),
            "avg_recall_low_snr": last_row.get("avg_recall_low_snr"),
            "avg_recall_medium_snr": last_row.get("avg_recall_medium_snr"),
            "avg_recall_high_snr": last_row.get("avg_recall_high_snr"),
        },
        "final_model_info": final_model_info,
        "best_snapshots": {
            "checkpoint": {
                **(_snapshot_from_row(best_checkpoint_row, "map50_95", "best.pt (best mAP50:95)") or {}),
                "model_info": best_checkpoint_model_info,
            } if best_checkpoint_row else None,
            "val_loss": _snapshot_from_row(best_val_loss_row, "val_loss", "best val_loss"),
            "map50": _snapshot_from_row(best_map50_row, "map50", "best mAP50"),
            "avg_recall_low_snr": _snapshot_from_row(best_recall_row, "avg_recall_low_snr", "best recall low SNR"),
        },
        "checkpoint_policy": {
            "best_checkpoint_metric": "map50_95",
            "best_checkpoint_mode": "max",
            "early_stopping_metric": "val_loss",
            "early_stopping_mode": "min",
        },
        "artifacts": {
            "best_pt": (run_dir / "best.pt").is_file(),
            "last_pt": (run_dir / "last.pt").is_file(),
            "loss_curves": (run_dir / "loss_curves.png").is_file(),
            "map_curves": (run_dir / "map_curves.png").is_file(),
            "avg_recall_curves": (run_dir / "avg_recall_curves.png").is_file(),
            "model_summary": (run_dir / "model_summary.txt").is_file(),
        },
    }


def list_evaluation_runs() -> Dict[str, Any]:
    return {"runs": [_run_summary(run_dir) for run_dir in _candidate_run_dirs()]}


def evaluation_run_details(run_path: str) -> Dict[str, Any]:
    path = _resolve_run_dir(run_path)

    raw_rows = _read_train_log_rows(path)
    if not raw_rows:
        eval_artifact = _load_eval_artifact(path)
        if eval_artifact is None:
            raise HTTPException(status_code=404, detail=f"No train_log.csv found for run '{path.name}'.")
        summary = _run_summary(path)
        return {
            "summary": summary,
            "epoch_rows": [
                {
                    "epoch": 1,
                    "train_loss": None,
                    "val_loss": None,
                    "map50": summary["final_metrics"].get("map50"),
                    "map50_95": summary["final_metrics"].get("map50_95"),
                    "avg_recall_low_snr": summary["final_metrics"].get("avg_recall_low_snr"),
                    "avg_recall_medium_snr": summary["final_metrics"].get("avg_recall_medium_snr"),
                    "avg_recall_high_snr": summary["final_metrics"].get("avg_recall_high_snr"),
                    "metrics_json_path": str(eval_artifact["artifact_path"]),
                }
            ],
            "available_metrics": METRIC_KEYS,
        }

    return {
        "summary": _run_summary(path),
        "epoch_rows": _epoch_rows(raw_rows),
        "available_metrics": METRIC_KEYS,
    }


def _load_metrics_payload(run_dir: Path, epoch: int) -> tuple[Path, Dict[str, Any]]:
    eval_artifact = _load_eval_artifact(run_dir)
    if eval_artifact is not None:
        return eval_artifact["artifact_path"], eval_artifact["metrics_payload"]

    direct = run_dir / "metrics" / f"metrics_epoch_{epoch:03d}.json"
    if direct.is_file():
        return direct, json.loads(direct.read_text(encoding="utf-8"))

    matches = sorted((run_dir / "metrics").glob(f"metrics_epoch_{epoch:03d}.json"))
    if matches:
        metrics_path = matches[0]
        return metrics_path, json.loads(metrics_path.read_text(encoding="utf-8"))

    raise HTTPException(status_code=404, detail=f"No metrics JSON found for epoch {epoch} in '{run_dir}'.")


def _normalize_numeric_list(values: Any, *, field_name: str) -> List[float]:
    if not isinstance(values, list):
        raise HTTPException(status_code=404, detail=f"Invalid '{field_name}' series format in metrics JSON.")
    return [float(value) for value in values]


def _class_option_label(class_key: str) -> str:
    try:
        return f"Class {int(class_key)}"
    except (TypeError, ValueError):
        return str(class_key)


def _sorted_class_keys(per_class: Dict[str, Any]) -> List[str]:
    def _sort_key(key: str) -> tuple[int, int | str]:
        try:
            return (0, int(key))
        except (TypeError, ValueError):
            return (1, str(key))

    return sorted(per_class.keys(), key=_sort_key)


def evaluation_recall_snr(run_path: str, epoch: int) -> Dict[str, Any]:
    run_dir = _resolve_run_dir(run_path)
    metrics_path, payload = _load_metrics_payload(run_dir, epoch)
    recall_payload = payload.get("recall_snr", {})
    global_recall = recall_payload.get("global", {})
    snr_bins = _normalize_numeric_list(global_recall.get("snr_bins"), field_name="recall_snr.global.snr_bins")
    recall = _normalize_numeric_list(global_recall.get("recall"), field_name="recall_snr.global.recall")
    per_class_raw = recall_payload.get("per_class", {})

    if not isinstance(per_class_raw, dict):
        per_class_raw = {}

    per_class: Dict[str, Dict[str, List[float]]] = {}
    class_options: List[Dict[str, str]] = []
    for class_key in _sorted_class_keys(per_class_raw):
        class_payload = per_class_raw.get(class_key)
        if not isinstance(class_payload, dict):
            continue
        class_options.append({"key": str(class_key), "label": _class_option_label(str(class_key))})
        per_class[str(class_key)] = {
            "snr_bins": _normalize_numeric_list(
                class_payload.get("snr_bins", snr_bins),
                field_name=f"recall_snr.per_class.{class_key}.snr_bins",
            ),
            "recall": _normalize_numeric_list(
                class_payload.get("recall"),
                field_name=f"recall_snr.per_class.{class_key}.recall",
            ),
        }

    return {
        "run_path": str(run_dir),
        "epoch": epoch,
        "metrics_json_path": str(metrics_path),
        "global": {
            "snr_bins": snr_bins,
            "recall": recall,
        },
        "per_class": per_class,
        "class_options": class_options,
    }


def evaluation_f1_stats(run_path: str, epoch: int) -> Dict[str, Any]:
    run_dir = _resolve_run_dir(run_path)
    metrics_path, payload = _load_metrics_payload(run_dir, epoch)
    f1_payload = payload.get("f1_stats", {})

    if not isinstance(f1_payload, dict):
        raise HTTPException(status_code=404, detail=f"No f1_stats found in '{metrics_path.name}'.")

    global_stats = {
        "thr": _normalize_numeric_list(f1_payload.get("thr"), field_name="f1_stats.thr"),
        "recall": _normalize_numeric_list(f1_payload.get("recall"), field_name="f1_stats.recall"),
        "precision": _normalize_numeric_list(f1_payload.get("precision"), field_name="f1_stats.precision"),
        "f1": _normalize_numeric_list(f1_payload.get("f1"), field_name="f1_stats.f1"),
    }

    per_class_raw = f1_payload.get("per_class", {})
    if not isinstance(per_class_raw, dict):
        per_class_raw = {}

    per_class: Dict[str, Dict[str, List[float]]] = {}
    class_options: List[Dict[str, str]] = []
    for class_key in _sorted_class_keys(per_class_raw):
        class_payload = per_class_raw.get(class_key)
        if not isinstance(class_payload, dict):
            continue
        class_options.append({"key": str(class_key), "label": _class_option_label(str(class_key))})
        per_class[str(class_key)] = {
            "thr": _normalize_numeric_list(
                class_payload.get("thr", global_stats["thr"]),
                field_name=f"f1_stats.per_class.{class_key}.thr",
            ),
            "recall": _normalize_numeric_list(
                class_payload.get("recall"),
                field_name=f"f1_stats.per_class.{class_key}.recall",
            ),
            "precision": _normalize_numeric_list(
                class_payload.get("precision"),
                field_name=f"f1_stats.per_class.{class_key}.precision",
            ),
            "f1": _normalize_numeric_list(
                class_payload.get("f1"),
                field_name=f"f1_stats.per_class.{class_key}.f1",
            ),
        }

    return {
        "run_path": str(run_dir),
        "epoch": epoch,
        "metrics_json_path": str(metrics_path),
        "global": global_stats,
        "per_class": per_class,
        "class_options": class_options,
    }


def _extract_confusion_matrix(payload: Dict[str, Any], key: str) -> List[List[float]]:
    matrix = payload.get(key)
    if not isinstance(matrix, list) or not matrix:
        raise HTTPException(status_code=404, detail=f"No '{key}' found in metrics JSON.")

    normalized: List[List[float]] = []
    expected_width: int | None = None
    for row in matrix:
        if not isinstance(row, list) or not row:
            raise HTTPException(status_code=404, detail=f"Invalid '{key}' matrix format in metrics JSON.")
        if expected_width is None:
            expected_width = len(row)
        if len(row) != expected_width:
            raise HTTPException(status_code=404, detail=f"Non rectangular '{key}' matrix in metrics JSON.")
        normalized.append([float(value) for value in row])

    if len(normalized) != expected_width:
        raise HTTPException(status_code=404, detail=f"Non square '{key}' matrix in metrics JSON.")

    return normalized


def _default_confusion_labels(size: int) -> List[str]:
    if size <= 0:
        return []
    if size == 1:
        return ["bg"]
    return [f"c{i}" for i in range(size - 1)] + ["bg"]


def evaluation_confusion_matrices(run_path: str, epoch: int) -> Dict[str, Any]:
    run_dir = _resolve_run_dir(run_path)
    metrics_path, payload = _load_metrics_payload(run_dir, epoch)

    low_snr = _extract_confusion_matrix(payload, "conf_matrix_low_snr")
    medium_snr = _extract_confusion_matrix(payload, "conf_matrix_medium_snr")
    high_snr = _extract_confusion_matrix(payload, "conf_matrix_high_snr")
    class_labels = _default_confusion_labels(len(low_snr))

    return {
        "run_path": str(run_dir),
        "epoch": epoch,
        "metrics_json_path": str(metrics_path),
        "class_labels": class_labels,
        "matrices": {
            "low_snr": low_snr,
            "medium_snr": medium_snr,
            "high_snr": high_snr,
        },
    }
