from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from detector2026.core.models.yolov11 import YOLOv11
from detector2026.core.utils.analysing_results import dataset_analysis_with_metrics
from detector2026.core.utils.dataset import YOLODatasetSpecificRes, load_class_index_to_name
from detector2026.core.utils.preprocess import preprocessing_num_channels


# =====================================================================
# PARAMETRES A EDITER
# =====================================================================

CHECKPOINT_PATH = Path(
    "/Users/tailleesarah/Documents/thèse/icml/detector2026/runs/examples_of_training/yolov11n_specificres_cfg512/best.pt"
)
DATASET_PATH = Path(
    "/Users/tailleesarah/Documents/thèse/icml/ICML2026DataSimulator/examples/output/rf_thesis_dataset"
)
SPLIT = "val"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
OUTPUT_JSON = CHECKPOINT_PATH.parent / "evaluate_model_metrics.json"

NUM_CLASSES = 20
WIDTH_MULT = 0.25  # YOLOv11n
REG_MAX = 16
DATASET_MODE = "specificres"
PREPROCESSING = "none"
RES_KEY = "cfg512"
RES_HW = (256, 256)
ANISOTROPIC = False
P3_SIZE = (64, 64)

BATCH_SIZE = 1
IOU_THRESH = 0.5
FALSE_ALARM_TARGET = 0.01


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


def main() -> None:
    if DATASET_MODE != "specificres":
        raise ValueError("Ce script simplifie est prevu pour DATASET_MODE='specificres'.")
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint introuvable: '{CHECKPOINT_PATH}'")
    if not DATASET_PATH.is_dir():
        raise FileNotFoundError(f"Dataset introuvable: '{DATASET_PATH}'")

    input_channels = preprocessing_num_channels(PREPROCESSING)

    print("[1/4] Construction du modele")
    model = YOLOv11(
        output_dir=str(CHECKPOINT_PATH.parent),
        num_classes=NUM_CLASSES,
        reg_max=REG_MAX,
        device=DEVICE,
        input_canals=input_channels,
        width_mult=WIDTH_MULT,
        anisotropic=ANISOTROPIC,
        p3_size=P3_SIZE,
        input_hw=RES_HW,
    )

    print("[2/4] Chargement des poids")
    model.load_weights(str(CHECKPOINT_PATH), device=DEVICE, eval_mode=True)
    model.eval()

    print("[3/4] Construction du dataloader")
    eval_dataset = YOLODatasetSpecificRes(
        data_dir=str(DATASET_PATH / SPLIT / "data"),
        labels_dir=str(DATASET_PATH / SPLIT / "labels_detect"),
        res_hw=RES_HW,
        res_key=RES_KEY,
        preprocessing=PREPROCESSING,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        collate_fn=eval_dataset.collate_fn,
    )

    print("[4/4] Evaluation")
    full_metrics = dataset_analysis_with_metrics(
        model=model,
        val_loader=eval_loader,
        iou_thresh=IOU_THRESH,
        fa=FALSE_ALARM_TARGET,
        img_size=RES_HW,
        to_save=False,
        to_plot=False,
        class_index_to_name=load_class_index_to_name(DATASET_PATH),
    )

    payload = {
        "checkpoint": str(CHECKPOINT_PATH),
        "dataset_path": str(DATASET_PATH),
        "split": SPLIT,
        "device": DEVICE,
        "batch_size": BATCH_SIZE,
        "iou_thresh": IOU_THRESH,
        "false_alarm_target": FALSE_ALARM_TARGET,
        "model_config": {
            "model_id": "yolov11",
            "num_classes": NUM_CLASSES,
            "width_mult": WIDTH_MULT,
            "reg_max": REG_MAX,
            "dataset_mode": DATASET_MODE,
            "preprocessing": PREPROCESSING,
            "res_key": RES_KEY,
            "res_hw": RES_HW,
            "anisotropic": ANISOTROPIC,
            "p3_size": P3_SIZE,
        },
        "metrics": full_metrics,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")

    map_stats = full_metrics.get("map_stats", {})
    print("\n[Evaluation terminee]")
    print(f"Resultat JSON : {OUTPUT_JSON}")
    print(f"mAP50         : {map_stats.get('mAP50')}")
    print(f"mAP50:95      : {map_stats.get('mAP50:95')}")


if __name__ == "__main__":
    main()
