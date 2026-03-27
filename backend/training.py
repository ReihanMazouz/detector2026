from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from csv import DictReader
from pathlib import Path
from typing import Any, Dict, List

import torch

from .config import DEFAULT_RUNS_ROOT, PROJECT_ROOT, RUNS_ROOT
from .datasets import _load_class_index_to_name, resolve_dataset_path

RUNS: List[Dict[str, Any]] = []
PROCESSES: Dict[str, subprocess.Popen[Any]] = {}


def _available_devices() -> List[str]:
    if torch.cuda.is_available():
        return [f"cuda:{index}" for index in range(torch.cuda.device_count())] + ["cpu"]
    return ["cpu"]


def _slugify_run_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._")
    return cleaned or "training-run"


def _run_output_dir(run_name: str, run_id: str) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    dirname = f"{timestamp}-{_slugify_run_name(run_name)}-{run_id[:8]}"
    return RUNS_ROOT / dirname


def _resolve_runs_root(value: str | None) -> Path:
    raw = str(value or DEFAULT_RUNS_ROOT).strip()
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    return candidate


def _infer_num_classes(dataset_path_value: str, fallback: int = 20) -> int:
    try:
        dataset_path = resolve_dataset_path(dataset_path_value)
        class_map = _load_class_index_to_name(dataset_path)
    except Exception:
        return fallback
    return len(class_map) or fallback


def _run_output_dir_from_root(runs_root: Path, run_name: str, run_id: str) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    dirname = f"{timestamp}-{_slugify_run_name(run_name)}-{run_id[:8]}"
    return runs_root / dirname


def _persist_run_state(run: Dict[str, Any]) -> None:
    output_dir = Path(run["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    config_payload = {
        "run_id": run["run_id"],
        "run_name": run["run_name"],
        "model_id": run["model_id"],
        "dataset_path": run["dataset_path"],
        "output_root": run["output_root"],
        "output_dir": run["output_dir"],
        "training": run["training"],
        "model_config": run["model_config"],
        "created_at": run["created_at"],
    }
    status_payload = {
        "run_id": run["run_id"],
        "status": run["status"],
        "progress": run["progress"],
        "updated_at": run["updated_at"],
        "output_dir": run["output_dir"],
        "logs": run["logs"],
        "error_message": run.get("error_message"),
    }

    (output_dir / "config.json").write_text(json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "status.json").write_text(json.dumps(status_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "backend_events.log").write_text("\n".join(run["logs"]) + "\n", encoding="utf-8")


def _read_latest_metrics(run: Dict[str, Any]) -> Dict[str, Any] | None:
    csv_path = Path(run["output_dir"]) / "train_log.csv"
    if not csv_path.exists():
        return None

    try:
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = DictReader(handle)
            rows = list(reader)
    except OSError:
        return None

    if not rows:
        return None

    last_row = rows[-1]
    epochs_total = max(1, int(run.get("training", {}).get("epochs", 1)))
    try:
        epoch_index = int(float(last_row.get("epoch", 0)))
    except (TypeError, ValueError):
        epoch_index = 0

    return {
        "epoch": epoch_index,
        "progress": min(99, max(run.get("progress", 2), int((epoch_index / epochs_total) * 100))),
        "map50": last_row.get("map50"),
        "map50_95": last_row.get("map50_95"),
        "metrics_json_path": last_row.get("metrics_json_path"),
        "train_log_path": str(csv_path),
    }


def _read_log_tail(log_path: Path, max_lines: int = 30) -> str | None:
    if not log_path.exists():
        return None
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    if not lines:
        return None
    return "\n".join(lines[-max_lines:])


def _read_log_lines(log_path: Path) -> List[str]:
    if not log_path.exists():
        return []
    try:
        return log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []


def _parse_live_status(log_path: Path, *, epochs_total: int) -> Dict[str, Any] | None:
    lines = _read_log_lines(log_path)
    if not lines:
        return None

    epoch_header_pattern = re.compile(r"Epoch\s+(\d+)/(\d+)")
    progress_pattern = re.compile(
        r"Epoch\s+(?P<epoch>\d+)\s+(?P<icon>[^\s]+)\s+(?P<phase>Training|Validation):\s+"
        r"(?P<pct>\d+)%\|.*?\|\s*(?P<current>\d+)/(?P<total>\d+)"
        r"(?:\s*\[(?P<elapsed>[^\]]+)\])?"
        r"(?:.*?,\s*(?P<metric_name>loss|val_loss)=(?P<metric_value>[-+0-9.eE]+))?"
    )

    current_epoch = None
    discovered_total = epochs_total
    phase = None
    batch_current = None
    batch_total = None
    batch_percent = None
    metric_name = None
    metric_value = None
    elapsed = None

    for line in lines:
        header_match = epoch_header_pattern.search(line)
        if header_match:
            current_epoch = int(header_match.group(1))
            discovered_total = int(header_match.group(2))

        progress_match = progress_pattern.search(line)
        if progress_match:
            current_epoch = int(progress_match.group("epoch"))
            phase_name = progress_match.group("phase")
            phase = "train" if phase_name == "Training" else "val"
            batch_current = int(progress_match.group("current"))
            batch_total = int(progress_match.group("total"))
            batch_percent = int(progress_match.group("pct"))
            metric_name = progress_match.group("metric_name")
            metric_value = progress_match.group("metric_value")
            elapsed = progress_match.group("elapsed")

    if current_epoch is None:
        return None

    epoch_total = max(1, discovered_total or epochs_total or 1)
    epoch_progress = batch_percent
    if epoch_progress is None and batch_current is not None and batch_total:
        epoch_progress = int((batch_current / max(1, batch_total)) * 100)

    overall_progress = min(
        99,
        max(
            0,
            int((((current_epoch - 1) + ((epoch_progress or 0) / 100.0)) / epoch_total) * 100),
        ),
    )

    return {
        "epoch": current_epoch,
        "epochs_total": epoch_total,
        "phase": phase,
        "batch_current": batch_current,
        "batch_total": batch_total,
        "epoch_progress_percent": epoch_progress,
        "overall_progress_percent": overall_progress,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "elapsed": elapsed,
    }


def _extract_error_message(log_text: str | None) -> str | None:
    if not log_text:
        return None

    lines = log_text.splitlines()
    traceback_index = None
    for index, line in enumerate(lines):
        if line.startswith("Traceback (most recent call last):"):
            traceback_index = index
            break

    if traceback_index is not None:
        return "\n".join(lines[traceback_index:])

    error_markers = (
        "AssertionError:",
        "AttributeError:",
        "RuntimeError:",
        "ValueError:",
        "TypeError:",
        "KeyError:",
        "FileNotFoundError:",
        "ImportError:",
        "ModuleNotFoundError:",
    )
    for line in reversed(lines):
        if any(marker in line for marker in error_markers):
            return line.strip()

    return None

MODEL_LIBRARY: Dict[str, Dict[str, Any]] = {
    "mr_yolo": {
        "id": "mr_yolo",
        "label": "MR_YOLO",
        "family": "multi_res",
        "tagline": "Detecteur multi-resolution concu pour fusionner plusieurs spectres STFT dans un seul modele.",
        "description": "Modele multi-resolution inspire du script train_multi_res.py.",
        "status": "stable",
        "complexity": {
            "design_score": "avance",
            "expected_latency": "moyenne",
            "memory_profile": "elevee",
            "best_for": "datasets fused avec plusieurs resolutions simultanees",
        },
        "dataset_modes": ["fused", "unires"],
        "default_config": {
            "width_mult": 0.5,
            "reg_max": 16,
            "backbone_mode": "TFSep_pyramid",
            "outfusion_channels_mult": 1,
            "device": "cuda:0",
        },
        "options": {
            "backbone_modes": ["F", "pyramid", "TFSep_pyramid", "TFSep_pyramid_up"],
            "devices": ["cuda:0", "cpu"],
        },
        "layers": [
            {
                "id": "inputs",
                "name": "Entrees multi-STFT",
                "module": "forward(inputs: List[Tensor])",
                "role": "inputs",
                "summary": "Recoit une liste de tenseurs, un par resolution.",
                "source": "models/mr_yolo.py",
                "editable_params": [
                    {"key": "input_resolutions", "label": "Input resolutions", "type": "text", "default": "auto"},
                ],
            },
            {
                "id": "backbone",
                "name": "Backbone MR",
                "module": "self.backbone = BackboneClass(...)",
                "role": "backbone",
                "summary": "Backbone multi-resolution choisi via BACKBONE_REGISTRY.",
                "source": "models/mr_yolo.py",
                "editable_params": [
                    {"key": "backbone_mode", "label": "Backbone mode", "type": "select", "options": ["F", "pyramid", "TFSep_pyramid", "TFSep_pyramid_up"], "default": "TFSep_pyramid"},
                    {"key": "width_mult", "label": "Width multiplier", "type": "number", "default": 0.5, "step": 0.05},
                    {"key": "outfusion_channels_mult", "label": "Outfusion channels", "type": "number", "default": 1, "step": 1},
                ],
            },
            {
                "id": "neck_p4",
                "name": "Neck P4 refinement",
                "module": "self.head_c3_1",
                "role": "neck",
                "summary": "Fusion P5 upsample + P4 puis raffinement Conv + TFSepBlock.",
                "source": "models/mr_yolo.py",
                "editable_params": [
                    {"key": "p4_block_mode", "label": "Block mode", "type": "select", "options": ["parallel"], "default": "parallel"},
                    {"key": "p4_residual", "label": "Residual", "type": "select", "options": ["true", "false"], "default": "true"},
                ],
            },
            {
                "id": "neck_p3",
                "name": "Neck P3 refinement",
                "module": "self.head_c3_2",
                "role": "neck",
                "summary": "Fusion P4 upsample + P3 pour la branche small.",
                "source": "models/mr_yolo.py",
                "editable_params": [
                    {"key": "p3_block_mode", "label": "Block mode", "type": "select", "options": ["parallel"], "default": "parallel"},
                    {"key": "p3_residual", "label": "Residual", "type": "select", "options": ["true", "false"], "default": "true"},
                ],
            },
            {
                "id": "detect",
                "name": "Detect head",
                "module": "self.detect = Detect(...)",
                "role": "head",
                "summary": "Tete de detection P3/P4/P5 avec DFL.",
                "source": "models/mr_yolo.py",
                "editable_params": [
                    {"key": "reg_max", "label": "Reg max", "type": "select", "options": [8, 16, 24], "default": 16},
                    {"key": "num_classes", "label": "Num classes", "type": "number", "default": 20, "step": 1},
                ],
            },
        ],
        "architecture_blocks": [
            {
                "name": "Entrees multi-STFT",
                "kind": "inputs",
                "summary": "Une liste de tenseurs est fournie, un par resolution.",
                "detail": "Le modele verifie la coherence des dimensions avant le passage backbone.",
                "accent": "signal",
            },
            {
                "name": "Backbone MR",
                "kind": "backbone",
                "summary": "Le backbone est choisi via BACKBONE_REGISTRY selon backbone_mode.",
                "detail": "Les variantes F, pyramid et TFSep_pyramid gerent la fusion des resolutions avant le neck.",
                "accent": "accent",
            },
            {
                "name": "Neck FPN/PAN",
                "kind": "neck",
                "summary": "Fusion top-down puis bottom-up sur trois echelles P3/P4/P5.",
                "detail": "Les blocs Conv et TFSepBlock raffinent les cartes avant detection.",
                "accent": "muted",
            },
            {
                "name": "Detect head",
                "kind": "head",
                "summary": "Tete de detection a trois niveaux avec DFL et strides derives du backbone.",
                "detail": "Les sorties dist_out et cls_out sont utilisees par YOLODetectionLoss.",
                "accent": "accent",
            },
        ],
        "construction_steps": [
            "Decouverte des resolutions d'entree a partir d'un exemple .pt dans train_multi_res.py.",
            "Initialisation de MR_YOLO avec num_classes, width_mult, reg_max et backbone_mode.",
            "Creation dynamique du backbone puis deduction automatique des strides.",
            "Assemblage du neck FPN/PAN et de la tete Detect.",
            "Entrainement via model.fit(data_dir=..., dataset='fused', batch_size=64, epochs=300, patience=30).",
        ],
        "editable_parameters": [
            {
                "key": "width_mult",
                "label": "Width multiplier",
                "type": "range",
                "min": 0.25,
                "max": 1.0,
                "step": 0.05,
                "default": 0.5,
                "group": "capacite",
                "impact": "Ajuste le nombre de canaux dans les blocs principaux.",
            },
            {
                "key": "reg_max",
                "label": "Reg max",
                "type": "select",
                "options": [8, 16, 24],
                "default": 16,
                "group": "detection",
                "impact": "Controle la finesse de la regression distribuee dans Detect.",
            },
            {
                "key": "backbone_mode",
                "label": "Backbone mode",
                "type": "select",
                "options": ["F", "pyramid", "TFSep_pyramid", "TFSep_pyramid_up"],
                "default": "TFSep_pyramid",
                "group": "architecture",
                "impact": "Change la maniere dont les branches multi-resolutions sont fusionnees.",
            },
            {
                "key": "outfusion_channels_mult",
                "label": "Outfusion channels",
                "type": "range",
                "min": 1,
                "max": 4,
                "step": 1,
                "default": 1,
                "group": "architecture",
                "impact": "Ajuste la largeur des canaux apres fusion backbone.",
            },
            {
                "key": "device",
                "label": "Device",
                "type": "select",
                "options": ["cuda:0", "cpu"],
                "default": "cuda:0",
                "group": "runtime",
                "impact": "Choisit la cible de calcul au lancement.",
            },
        ],
        "training_recipe": {
            "entrypoint": "test/train_multi_res.py",
            "model_class": "models/mr_yolo.py::MR_YOLO",
            "fit_signature": "model.fit(data_dir, batch_size, dataset='fused', epochs, patience)",
            "notes": [
                "Le script detecte d'abord automatiquement les resolutions presentes dans le dataset.",
                "Le dataset attendu contient des fichiers .pt multi-resolutions dans split/data.",
                "Le mode fused est la configuration nominale pour le training multi-resolution.",
            ],
        },
        "source_map": [
            {"label": "Classe principale", "path": "models/mr_yolo.py", "kind": "python"},
            {"label": "Script de training", "path": "test/train_multi_res.py", "kind": "python"},
            {"label": "Base d'entrainement", "path": "models/base.py", "kind": "python"},
            {"label": "Tete Detect", "path": "models/Head/detect.py", "kind": "python"},
        ],
    },
    "yolov8": {
        "id": "yolov8",
        "label": "YOLOv8",
        "family": "uni_res",
        "tagline": "Implementation from-scratch proche du YAML Ultralytics, adaptee au spectre mono-resolution.",
        "description": "Modele mono-resolution inspire du script train_unires.py.",
        "status": "stable",
        "complexity": {
            "design_score": "equilibre",
            "expected_latency": "faible",
            "memory_profile": "moyenne",
            "best_for": "experiences mono-resolution type cfg256/cfg512",
        },
        "dataset_modes": ["specificres", "singleres"],
        "default_config": {
            "width_mult": 0.5,
            "reg_max": 16,
            "device": "cuda:0",
            "res_key": "cfg512",
            "res_hw": [256, 256],
        },
        "options": {
            "devices": ["cuda:0", "cpu"],
            "res_keys": ["cfg128", "cfg256", "cfg512", "cfg1024", "cfg2048"],
        },
        "layers": [
            {
                "id": "stem",
                "name": "Stem conv",
                "module": "self.b0 / self.b1",
                "role": "stem",
                "summary": "Projection initiale de l'entree vers les premiers canaux.",
                "source": "models/yolov8.py",
                "editable_params": [
                    {"key": "in_ch", "label": "Input channels", "type": "number", "default": 1, "step": 1},
                    {"key": "width_mult", "label": "Width multiplier", "type": "number", "default": 0.5, "step": 0.05},
                ],
            },
            {
                "id": "backbone_c2f",
                "name": "Backbone C2f",
                "module": "self.b2 ... self.b9",
                "role": "backbone",
                "summary": "Empilement Conv/C2f/SPPF jusqu'aux cartes P3/P4/P5.",
                "source": "models/yolov8.py",
                "editable_params": [
                    {"key": "width_mult", "label": "Width multiplier", "type": "number", "default": 0.5, "step": 0.05},
                    {"key": "depth_mult", "label": "Depth multiplier", "type": "number", "default": 0.33, "step": 0.01},
                ],
            },
            {
                "id": "pan_small",
                "name": "PAN small branch",
                "module": "self.h0 / self.h1",
                "role": "neck",
                "summary": "Fusion top-down pour reconstruire la branche P3.",
                "source": "models/yolov8.py",
                "editable_params": [
                    {"key": "small_shortcut", "label": "Shortcut", "type": "select", "options": ["true", "false"], "default": "true"},
                ],
            },
            {
                "id": "pan_medium_large",
                "name": "PAN medium-large branch",
                "module": "self.down3 / self.h2 / self.down4 / self.h3",
                "role": "neck",
                "summary": "Reconstruction bottom-up des branches medium et large.",
                "source": "models/yolov8.py",
                "editable_params": [
                    {"key": "medium_shortcut", "label": "Medium shortcut", "type": "select", "options": ["true", "false"], "default": "true"},
                    {"key": "large_shortcut", "label": "Large shortcut", "type": "select", "options": ["true", "false"], "default": "true"},
                ],
            },
            {
                "id": "detect",
                "name": "Detect head",
                "module": "self.detect = Detect(...)",
                "role": "head",
                "summary": "Sortie detection small/medium/large.",
                "source": "models/yolov8.py",
                "editable_params": [
                    {"key": "reg_max", "label": "Reg max", "type": "select", "options": [8, 16, 24], "default": 16},
                    {"key": "strides", "label": "Strides", "type": "text", "default": "[8, 16, 32]"},
                ],
            },
        ],
        "architecture_blocks": [
            {
                "name": "Stem conv",
                "kind": "inputs",
                "summary": "Une entree spectrale unique est projetee dans le backbone.",
                "detail": "Le stem et les premiers blocs C2f suivent une structure YOLOv8 classique.",
                "accent": "signal",
            },
            {
                "name": "Backbone C2f",
                "kind": "backbone",
                "summary": "Empilement de Conv, C2f et SPPF jusqu'aux cartes P3/P4/P5.",
                "detail": "width_mult et depth_mult ajustent les canaux et le nombre de repetitions.",
                "accent": "accent",
            },
            {
                "name": "PAN head",
                "kind": "neck",
                "summary": "Chemin FPN/PAN classique pour reconstruire les cartes small, medium et large.",
                "detail": "Les blocs h0 a h3 fusionnent les niveaux par concat puis C2f.",
                "accent": "muted",
            },
            {
                "name": "Detect + loss",
                "kind": "head",
                "summary": "La tete Detect applique la regression distribuee et la classification.",
                "detail": "YOLODetectionLoss exploite les strides [8, 16, 32].",
                "accent": "accent",
            },
        ],
        "construction_steps": [
            "Initialisation de YOLOv8 avec width_mult, depth_mult, reg_max et device.",
            "Choix d'une resolution de travail via select_res dans train_unires.py.",
            "Construction du backbone C2f puis du PAN head a trois echelles.",
            "Entrainement via model.fit(..., dataset='specificres', select_res={'res_hw': (256,256), 'res_key': 'cfg512'}).",
        ],
        "editable_parameters": [
            {
                "key": "width_mult",
                "label": "Width multiplier",
                "type": "range",
                "min": 0.25,
                "max": 1.25,
                "step": 0.05,
                "default": 0.5,
                "group": "capacite",
                "impact": "Redimensionne les canaux internes du modele.",
            },
            {
                "key": "reg_max",
                "label": "Reg max",
                "type": "select",
                "options": [8, 16, 24],
                "default": 16,
                "group": "detection",
                "impact": "Modifie la resolution de la regression distribuee.",
            },
            {
                "key": "res_key",
                "label": "Resolution key",
                "type": "select",
                "options": ["cfg128", "cfg256", "cfg512", "cfg1024", "cfg2048"],
                "default": "cfg512",
                "group": "dataset",
                "impact": "Selectionne le sous-ensemble de resolution cible dans le dataset.",
            },
            {
                "key": "res_hw",
                "label": "Resolution HxW",
                "type": "tuple",
                "default": [256, 256],
                "group": "dataset",
                "impact": "Definit explicitement la taille d'entree envoyee au modele.",
            },
            {
                "key": "device",
                "label": "Device",
                "type": "select",
                "options": ["cuda:0", "cpu"],
                "default": "cuda:0",
                "group": "runtime",
                "impact": "Choisit la cible de calcul au lancement.",
            },
        ],
        "training_recipe": {
            "entrypoint": "test/train_unires.py",
            "model_class": "models/yolov8.py::YOLOv8",
            "fit_signature": "model.fit(data_dir, batch_size, dataset='specificres', epochs, patience, select_res)",
            "notes": [
                "Le mode specificres vise un sous-ensemble de resolution au sein du dataset.",
                "Le couple res_key / res_hw pilote la resolution effectivement chargee.",
                "Le script de reference utilise batch_size=64 et epochs=300.",
            ],
        },
        "source_map": [
            {"label": "Classe principale", "path": "models/yolov8.py", "kind": "python"},
            {"label": "Script de training", "path": "test/train_unires.py", "kind": "python"},
            {"label": "Base d'entrainement", "path": "models/base.py", "kind": "python"},
            {"label": "Tete Detect", "path": "models/Head/detect.py", "kind": "python"},
        ],
    },
    "yolov11": {
        "id": "yolov11",
        "label": "YOLOv11",
        "family": "uni_res",
        "tagline": "Variante experimentale avec C3k2, SPPF et C2PSA pour tester une tete plus expressive.",
        "description": "Variante experimentale disponible dans le codebase detector2026.",
        "status": "experimental",
        "complexity": {
            "design_score": "avance",
            "expected_latency": "moyenne",
            "memory_profile": "moyenne",
            "best_for": "ablation et experimentation sur backbone/head",
        },
        "dataset_modes": ["specificres", "singleres"],
        "default_config": {
            "width_mult": 0.5,
            "reg_max": 16,
            "device": "cuda:0",
        },
        "options": {
            "devices": ["cuda:0", "cpu"],
        },
        "layers": [
            {
                "id": "stem",
                "name": "Stem and early convs",
                "module": "conv1 / conv2 / c3_1",
                "role": "stem",
                "summary": "Premiers etages de projection et d'extraction.",
                "source": "models/yolov11.py",
                "editable_params": [
                    {"key": "input_canals", "label": "Input channels", "type": "number", "default": 1, "step": 1},
                    {"key": "width_mult", "label": "Width multiplier", "type": "number", "default": 0.5, "step": 0.05},
                ],
            },
            {
                "id": "backbone",
                "name": "Backbone C3k2",
                "module": "conv3 ... c3_4",
                "role": "backbone",
                "summary": "Cascade de Conv et C3k2 pour produire f3, f4 et f5.",
                "source": "models/yolov11.py",
                "editable_params": [
                    {"key": "width_mult", "label": "Width multiplier", "type": "number", "default": 0.5, "step": 0.05},
                    {"key": "c3_shortcut", "label": "Shortcut mode", "type": "select", "options": ["mixed"], "default": "mixed"},
                ],
            },
            {
                "id": "context",
                "name": "SPPF + C2PSA",
                "module": "self.sppf / self.attn",
                "role": "context",
                "summary": "Contexte global puis attention sur la derniere carte.",
                "source": "models/yolov11.py",
                "editable_params": [
                    {"key": "attn_repeats", "label": "Attention repeats", "type": "number", "default": 2, "step": 1},
                ],
            },
            {
                "id": "head",
                "name": "FPN/PAN head",
                "module": "head_c3_1 ... head_c3_4",
                "role": "neck",
                "summary": "Fusion top-down et bottom-up sur trois echelles.",
                "source": "models/yolov11.py",
                "editable_params": [
                    {"key": "head_shortcut_large", "label": "Large head shortcut", "type": "select", "options": ["true", "false"], "default": "true"},
                ],
            },
            {
                "id": "detect",
                "name": "Detect head",
                "module": "self.detect = Detect(...)",
                "role": "head",
                "summary": "Branche finale de detection avec reg_max configurable.",
                "source": "models/yolov11.py",
                "editable_params": [
                    {"key": "reg_max", "label": "Reg max", "type": "select", "options": [8, 16, 24], "default": 16},
                    {"key": "strides", "label": "Strides", "type": "text", "default": "[8, 16, 32]"},
                ],
            },
        ],
        "architecture_blocks": [
            {
                "name": "Backbone progressif",
                "kind": "backbone",
                "summary": "Conv + C3k2 en cascade pour construire f2, f3, f4 et f5.",
                "detail": "Les etages successifs reduisent la resolution tout en augmentant la capacite.",
                "accent": "signal",
            },
            {
                "name": "SPPF + C2PSA",
                "kind": "attention",
                "summary": "Ajout de contexte global puis bloc d'attention sur le dernier niveau.",
                "detail": "C2PSA sert a enrichir la representation avant le head.",
                "accent": "accent",
            },
            {
                "name": "Head FPN/PAN",
                "kind": "neck",
                "summary": "Fusion small/medium/large par upsample, concat et C3k2.",
                "detail": "La structure reste proche des familles YOLO recentes.",
                "accent": "muted",
            },
            {
                "name": "Detect",
                "kind": "head",
                "summary": "Sorties de detection sur trois echelles avec reg_max configurable.",
                "detail": "Le modele est configure pour des strides [8, 16, 32].",
                "accent": "accent",
            },
        ],
        "construction_steps": [
            "Definition manuelle des canaux c1 a c5 via width_mult.",
            "Empilement backbone puis insertion de SPPF et C2PSA.",
            "Assemblage du head small/medium/large avec Conv et C3k2.",
            "Initialisation de Detect puis entrainement via BaseModel.fit.",
        ],
        "editable_parameters": [
            {
                "key": "width_mult",
                "label": "Width multiplier",
                "type": "range",
                "min": 0.25,
                "max": 1.0,
                "step": 0.05,
                "default": 0.5,
                "group": "capacite",
                "impact": "Dimensionne les canaux c1 a c5.",
            },
            {
                "key": "reg_max",
                "label": "Reg max",
                "type": "select",
                "options": [8, 16, 24],
                "default": 16,
                "group": "detection",
                "impact": "Controle la precision de la branche bbox.",
            },
            {
                "key": "device",
                "label": "Device",
                "type": "select",
                "options": ["cuda:0", "cpu"],
                "default": "cuda:0",
                "group": "runtime",
                "impact": "Choisit la cible de calcul au lancement.",
            },
        ],
        "training_recipe": {
            "entrypoint": "models/yolov11.py",
            "model_class": "models/yolov11.py::YOLOv11",
            "fit_signature": "BaseModel.fit(data_dir, batch_size, dataset, epochs, patience, ...)",
            "notes": [
                "Le modele est present dans le codebase, mais pas encore expose par un script de training dedie.",
                "La structure est utile pour les experimentations architecturelles.",
                "Il faut stabiliser le flux backend avant un lancement temps reel natif.",
            ],
        },
        "source_map": [
            {"label": "Classe principale", "path": "models/yolov11.py", "kind": "python"},
            {"label": "Base d'entrainement", "path": "models/base.py", "kind": "python"},
            {"label": "Blocs", "path": "nn/blocks.py", "kind": "python"},
            {"label": "Tete Detect", "path": "models/Head/detect.py", "kind": "python"},
        ],
    },
}


def _training_model_summary(model: Dict[str, Any]) -> Dict[str, Any]:
    devices = _available_devices()
    default_config = dict(model["default_config"])
    options = dict(model["options"])
    options["devices"] = devices
    default_config["device"] = default_config.get("device") if default_config.get("device") in devices else devices[0]
    return {
        "id": model["id"],
        "label": model["label"],
        "family": model["family"],
        "tagline": model["tagline"],
        "description": model["description"],
        "status": model["status"],
        "complexity": model["complexity"],
        "dataset_modes": model["dataset_modes"],
        "default_config": default_config,
        "options": options,
    }


def training_models_payload() -> Dict[str, Any]:
    return {"models": [_training_model_summary(model) for model in MODEL_LIBRARY.values()]}


def _system_resources() -> Dict[str, Any]:
    cpu_count = os.cpu_count() or 1
    try:
        load1, load5, load15 = os.getloadavg()
        cpu_percent = min(100.0, round((load1 / max(cpu_count, 1)) * 100, 1))
    except OSError:
        load1, load5, load15 = (0.0, 0.0, 0.0)
        cpu_percent = 0.0

    gpus = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            used_bytes = total_bytes - free_bytes
            utilization = 0.0 if total_bytes <= 0 else round((used_bytes / total_bytes) * 100, 1)
            gpus.append(
                {
                    "name": props.name,
                    "device": f"cuda:{index}",
                    "memory_total_gb": round(total_bytes / (1024 ** 3), 2),
                    "memory_used_gb": round(used_bytes / (1024 ** 3), 2),
                    "memory_utilization_percent": utilization,
                }
            )
    else:
        gpus.append(
            {
                "name": "CPU only",
                "device": "cpu",
                "memory_total_gb": 0.0,
                "memory_used_gb": 0.0,
                "memory_utilization_percent": 0.0,
            }
        )

    return {
        "cpu": {
            "logical_cores": cpu_count,
            "loadavg": [round(load1, 2), round(load5, 2), round(load15, 2)],
            "utilization_percent": cpu_percent,
        },
        "gpus": gpus,
    }


def system_status_payload() -> Dict[str, Any]:
    return {"resources": _system_resources()}


def start_training_run(payload: Dict[str, Any]) -> Dict[str, Any]:
    now = time.time()
    run_id = uuid.uuid4().hex
    run_name = str(payload.get("run_name", "training-run"))
    dataset_path_value = str(payload.get("dataset_path", ""))
    model_config = dict(payload.get("model_config", {}))
    model_config["num_classes"] = int(model_config.get("num_classes") or _infer_num_classes(dataset_path_value))
    runs_root_input = str(payload.get("output_root", DEFAULT_RUNS_ROOT))
    runs_root = _resolve_runs_root(runs_root_input)
    output_dir = _run_output_dir_from_root(runs_root, run_name, run_id)
    run = {
        "run_id": run_id,
        "status": "running",
        "created_at": now,
        "updated_at": now,
        "progress": 2,
        "model_id": str(payload.get("model_id", "unknown")),
        "run_name": run_name,
        "dataset_path": dataset_path_value,
        "output_root": str(runs_root),
        "output_dir": str(output_dir),
        "training": dict(payload.get("training", {})),
        "model_config": model_config,
        "logs": [
            "Initialisation du run d'entrainement.",
            "Validation de la configuration.",
            "Allocation de la ressource de calcul.",
            f"Racine des runs: {runs_root}",
            f"Dossier de sortie: {output_dir}",
            f"Nombre de classes detecte: {model_config['num_classes']}",
        ],
    }
    _persist_run_state(run)
    worker_path = PROJECT_ROOT / "backend" / "train_worker.py"
    session_log_path = Path(run["output_dir"]) / "session.log"
    with session_log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            [sys.executable, str(worker_path), str(Path(run["output_dir"]) / "config.json")],
            cwd=str(PROJECT_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    run["pid"] = process.pid
    PROCESSES[run_id] = process
    RUNS.insert(0, run)
    _persist_run_state(run)
    return run


def cancel_training_run(run_id: str) -> Dict[str, Any]:
    for run in RUNS:
        if run["run_id"] != run_id:
            continue
        if run["status"] in {"completed", "failed", "canceled"}:
            return dict(run)
        process = PROCESSES.get(run_id)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        PROCESSES.pop(run_id, None)
        run["status"] = "canceled"
        run["updated_at"] = time.time()
        run["progress"] = min(run.get("progress", 0), 99)
        run["logs"] = list(run.get("logs", [])) + ["Arret demande depuis l'interface utilisateur."]
        _persist_run_state(run)
        return dict(run)
    raise KeyError(run_id)


def list_training_runs() -> Dict[str, Any]:
    now = time.time()
    runs = []
    for run in RUNS:
        process = PROCESSES.get(run["run_id"])
        latest_metrics = _read_latest_metrics(run)
        log_path = Path(run["output_dir"]) / "session.log"
        live_status = _parse_live_status(log_path, epochs_total=max(1, int(run.get("training", {}).get("epochs", 1))))
        if latest_metrics:
            run["latest_metrics"] = latest_metrics
            run["progress"] = latest_metrics["progress"]
        elif live_status:
            run["live_status"] = live_status
            run["progress"] = max(run.get("progress", 2), int(live_status["overall_progress_percent"]))
        else:
            run.pop("live_status", None)

        if run["status"] in {"running", "finishing"} and process is not None:
            return_code = process.poll()
            if return_code is None:
                run["status"] = "finishing" if run.get("progress", 0) >= 96 else "running"
                run["updated_at"] = now
                _persist_run_state(run)
            else:
                PROCESSES.pop(run["run_id"], None)
                run["updated_at"] = now
                if return_code == 0:
                    run["status"] = "completed"
                    run["progress"] = 100
                    run["logs"] = list(run.get("logs", [])) + ["Entrainement termine avec succes."]
                else:
                    log_tail = _read_log_tail(log_path)
                    run["status"] = "failed"
                    run["error_message"] = _extract_error_message(log_tail) or (
                        f"Le process d'entrainement s'est termine avec le code {return_code} sans traceback explicite."
                    )
                    run["logs"] = list(run.get("logs", [])) + [f"Le process d'entrainement s'est termine avec le code {return_code}."]
                _persist_run_state(run)
        elif run["status"] == "canceled":
            run["updated_at"] = run.get("updated_at", now)
            _persist_run_state(run)
        run["session_log_path"] = str(log_path)
        run["log_tail"] = _read_log_tail(log_path)
        runs.append(dict(run))
    return {"runs": runs, "resources": _system_resources()}
