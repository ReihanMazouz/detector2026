#!/usr/bin/env python3
"""
Unified evaluation script for detector2026 clean models.

Loads a checkpoint, runs inference on a dataset split, computes detection
metrics (mAP50, mAP50:95, precision, recall by SNR bin, etc.) and writes
the results to a JSON file.

Supported models
----------------
  yolov8  yolov11  yolov12  tf_attn  mr_yolo  detr

Examples
--------
  # Evaluate YOLOv11n on the val split
  python evaluate.py --model yolov11 --scale n \\
      --checkpoint runs/yolov11_n_specificres_cfg512/best.pt \\
      --data-dir /data/rf_dataset \\
      --res-key cfg512 --res-hw 256 256

  # Evaluate MR-YOLO nano
  python evaluate.py --model mr_yolo --scale n \\
      --checkpoint runs/mr_yolo_n_fused_cfg512_cfg256/best.pt \\
      --data-dir /data/rf_dataset \\
      --res-keys cfg512 cfg256

  # Evaluate on the test split, save JSON elsewhere
  python evaluate.py --model yolov11 --scale n \\
      --checkpoint runs/yolov11_n_specificres_cfg512/best.pt \\
      --data-dir /data/rf_dataset --res-key cfg512 --res-hw 256 256 \\
      --split test --output-json results/yolov11n_test.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from detector2026.clean.models import DETR, MR_YOLO, TF_Attn_Yolo, YOLOv8, YOLOv11, YOLOv12
from detector2026.clean.utils.analysing_results import dataset_analysis_with_metrics
from detector2026.clean.utils.dataset import (
    YOLODatasetFusedMultiRes,
    YOLODatasetSpecificRes,
    load_class_index_to_name,
)
from detector2026.clean.utils.preprocess import preprocessing_num_channels

Resolution = Tuple[int, int]

SCALE_WIDTH: Dict[str, float] = {"n": 0.25, "s": 0.50, "m": 0.75, "l": 1.00}
SCALE_DEPTH: Dict[str, float] = {"n": 0.33, "s": 0.33, "m": 0.67, "l": 1.00}
DEFAULT_RES_KEYS: List[str] = ["cfg512", "cfg256", "cfg128", "cfg1024", "cfg2048"]
MR_BACKBONE_MODE = "TFSep_pyramid"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_pt_resolutions(data_dir: str, split: str = "val") -> List[Resolution]:
    images_dir = Path(data_dir) / split / "data"
    pt_file = next(images_dir.glob("*.pt"), None)
    if pt_file is None:
        raise FileNotFoundError(f"No .pt file found in {images_dir}")
    specs = torch.load(pt_file, map_location="cpu")
    if not isinstance(specs, list):
        raise ValueError(f"Expected a list of tensors in {pt_file}")
    resolutions: List[Resolution] = []
    for i, spec in enumerate(specs):
        h = int(spec.shape[-2]) if spec.ndim >= 2 else 0
        w = int(spec.shape[-1]) if spec.ndim >= 1 else 0
        resolutions.append((h, w))
    return resolutions


def detect_mr_resolutions(data_dir: str, res_keys: List[str], split: str) -> Dict[str, Resolution]:
    all_res = _read_pt_resolutions(data_dir, split)
    if len(all_res) != len(res_keys):
        raise ValueError(
            f"Dataset has {len(all_res)} resolution tensors but "
            f"{len(res_keys)} res_keys provided: {res_keys}."
        )
    return dict(zip(res_keys, all_res))


def detect_single_res_hw(data_dir: str, res_key: str, split: str) -> Resolution:
    all_res = _read_pt_resolutions(data_dir, split)
    if len(all_res) == 1:
        return all_res[0]
    try:
        idx = DEFAULT_RES_KEYS.index(res_key)
    except ValueError:
        raise ValueError(
            f"Cannot auto-detect resolution for res_key='{res_key}'. "
            "Pass --res-hw explicitly."
        )
    if idx >= len(all_res):
        raise ValueError(
            f"res_key='{res_key}' maps to index {idx} but dataset only has "
            f"{len(all_res)} resolution tensors. Pass --res-hw explicitly."
        )
    return all_res[idx]


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


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(
    model_id: str,
    scale: str,
    output_dir: str,
    num_classes: int,
    reg_max: int,
    device: str,
    input_channels: int,
    input_resolutions: Optional[Dict[str, Resolution]] = None,
    detr_kwargs: Optional[dict] = None,
) -> torch.nn.Module:
    w = SCALE_WIDTH[scale]
    d = SCALE_DEPTH[scale]

    if model_id == "yolov11":
        return YOLOv11(
            output_dir=output_dir,
            num_classes=num_classes,
            reg_max=reg_max,
            device=device,
            input_canals=input_channels,
            width_mult=w,
        )
    if model_id == "yolov8":
        return YOLOv8(
            output_dir=output_dir,
            num_classes=num_classes,
            reg_max=reg_max,
            device=device,
            in_ch=input_channels,
            width_mult=w,
            depth_mult=d,
        )
    if model_id == "yolov12":
        return YOLOv12(
            output_dir=output_dir,
            num_classes=num_classes,
            reg_max=reg_max,
            device=device,
            in_ch=input_channels,
            width_mult=w,
            depth_mult=d,
        )
    if model_id == "tf_attn":
        return TF_Attn_Yolo(
            output_dir=output_dir,
            num_classes=num_classes,
            reg_max=reg_max,
            device=device,
            input_canals=input_channels,
            width_mult=w,
        )
    if model_id == "mr_yolo":
        if input_resolutions is None:
            raise ValueError("mr_yolo requires input_resolutions.")
        return MR_YOLO(
            output_dir=output_dir,
            num_classes=num_classes,
            reg_max=reg_max,
            device=device,
            in_ch=input_channels,
            width_mult=w,
            input_resolutions=list(input_resolutions.values()),
            backbone_mode=MR_BACKBONE_MODE,
        )
    if model_id == "detr":
        kw = detr_kwargs or {}
        return DETR(
            output_dir=output_dir,
            num_classes=num_classes,
            device=device,
            input_channels=input_channels,
            width_mult=w,
            hidden_dim=kw.get("hidden_dim", 256),
            num_queries=kw.get("num_queries", 100),
            num_encoder_layers=kw.get("encoder_layers", 2),
            num_decoder_layers=kw.get("decoder_layers", 3),
            nheads=kw.get("nheads", 8),
            dim_feedforward=kw.get("dim_feedforward", 1024),
            dropout=kw.get("dropout", 0.0),
        )
    raise ValueError(f"Unknown model: '{model_id}'")


# ── Dataset factory ───────────────────────────────────────────────────────────

def build_loader(
    model_id: str,
    data_dir: str,
    split: str,
    preprocessing: str,
    res_key: str,
    res_hw: Optional[Resolution],
    res_keys: Optional[List[str]],
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    pin_memory = torch.cuda.is_available()

    if model_id == "mr_yolo":
        dataset = YOLODatasetFusedMultiRes(
            data_dir=os.path.join(data_dir, split, "data"),
            labels_dir=os.path.join(data_dir, split, "labels_detect"),
            res_keys=tuple(res_keys) if res_keys else None,
            preprocessing=preprocessing,
        )
    else:
        dataset = YOLODatasetSpecificRes(
            data_dir=os.path.join(data_dir, split, "data"),
            labels_dir=os.path.join(data_dir, split, "labels_detect"),
            res_hw=res_hw,
            res_key=res_key,
            preprocessing=preprocessing,
        )

    loader_kwargs: dict = {
        "batch_size": batch_size,
        "shuffle": False,
        "pin_memory": pin_memory,
        "collate_fn": dataset.collate_fn,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = True

    return DataLoader(dataset, **loader_kwargs)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a detector2026 model checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = parser.add_argument_group("Model")
    g.add_argument(
        "--model", required=True,
        choices=["yolov8", "yolov11", "yolov12", "tf_attn", "mr_yolo", "detr"],
    )
    g.add_argument("--scale", default="n", choices=["n", "s", "m", "l"])
    g.add_argument("--num-classes", type=int, default=20)
    g.add_argument("--reg-max", type=int, default=16)
    g.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file.")

    g = parser.add_argument_group("Data")
    g.add_argument("--data-dir", required=True)
    g.add_argument("--split", default="val", choices=["train", "val", "test"])
    g.add_argument("--preprocessing", default="none")
    g.add_argument("--res-key", default="cfg512")
    g.add_argument("--res-hw", type=int, nargs=2, metavar=("H", "W"), default=None)
    g.add_argument("--res-keys", nargs="+", default=None)

    g = parser.add_argument_group("Evaluation")
    g.add_argument("--iou-thresh", type=float, default=0.5)
    g.add_argument("--false-alarm-target", type=float, default=0.01)
    g.add_argument("--batch-size", type=int, default=16)
    g.add_argument("--num-workers", type=int, default=4)
    g.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    g = parser.add_argument_group("Output")
    g.add_argument(
        "--output-json", default=None,
        help="Path for the JSON results file. Defaults to <checkpoint_dir>/eval_<split>.json.",
    )
    g.add_argument("--no-plots", action="store_true", help="Skip generating plot images.")
    g.add_argument("--no-save", action="store_true", help="Do not save per-image predictions.")

    g = parser.add_argument_group("DETR-specific")
    g.add_argument("--detr-hidden-dim", type=int, default=256)
    g.add_argument("--detr-num-queries", type=int, default=100)
    g.add_argument("--detr-encoder-layers", type=int, default=2)
    g.add_argument("--detr-decoder-layers", type=int, default=3)
    g.add_argument("--detr-nheads", type=int, default=8)
    g.add_argument("--detr-dim-feedforward", type=int, default=1024)
    g.add_argument("--detr-dropout", type=float, default=0.0)

    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    input_channels = preprocessing_num_channels(args.preprocessing)

    # -- Resolve resolutions --
    input_resolutions: Optional[Dict[str, Resolution]] = None
    res_hw: Optional[Resolution] = None

    if args.model == "mr_yolo":
        if not args.res_keys:
            raise ValueError("--res-keys is required for mr_yolo.")
        input_resolutions = detect_mr_resolutions(str(data_dir), args.res_keys, args.split)
        print("Detected input resolutions:")
        for key, hw in input_resolutions.items():
            print(f"  {key}: {hw}")
    else:
        if args.res_hw is not None:
            res_hw = tuple(args.res_hw)
        else:
            res_hw = detect_single_res_hw(str(data_dir), args.res_key, args.split)
            print(f"Auto-detected resolution for '{args.res_key}': {res_hw}")

    # -- Build model --
    print("\n[1/4] Building model...")
    detr_kwargs = {
        "hidden_dim": args.detr_hidden_dim,
        "num_queries": args.detr_num_queries,
        "encoder_layers": args.detr_encoder_layers,
        "decoder_layers": args.detr_decoder_layers,
        "nheads": args.detr_nheads,
        "dim_feedforward": args.detr_dim_feedforward,
        "dropout": args.detr_dropout,
    }
    output_dir = str(checkpoint_path.parent)
    model = build_model(
        model_id=args.model,
        scale=args.scale,
        output_dir=output_dir,
        num_classes=args.num_classes,
        reg_max=args.reg_max,
        device=args.device,
        input_channels=input_channels,
        input_resolutions=input_resolutions,
        detr_kwargs=detr_kwargs,
    )

    # -- Load weights --
    print("[2/4] Loading weights...")
    missing, unexpected = model.load_weights(str(checkpoint_path), device=args.device, eval_mode=True)
    if missing:
        print(f"  [warning] Missing keys  : {len(missing)}")
    if unexpected:
        print(f"  [warning] Unexpected keys: {len(unexpected)}")
    model.eval()

    # -- Build dataloader --
    print("[3/4] Building dataloader...")
    resolved_workers = max(0, args.num_workers)
    loader = build_loader(
        model_id=args.model,
        data_dir=str(data_dir),
        split=args.split,
        preprocessing=args.preprocessing,
        res_key=args.res_key,
        res_hw=res_hw,
        res_keys=args.res_keys,
        batch_size=args.batch_size,
        num_workers=resolved_workers,
    )
    print(f"  Split={args.split} | {len(loader.dataset)} samples | batch={args.batch_size}")

    # -- Evaluate --
    print("[4/4] Evaluating...")
    img_size = res_hw if res_hw is not None else max(input_resolutions.values(), key=lambda hw: hw[0] * hw[1])
    full_metrics = dataset_analysis_with_metrics(
        model=model,
        val_loader=loader,
        iou_thresh=args.iou_thresh,
        fa=args.false_alarm_target,
        img_size=img_size,
        to_save=not args.no_save,
        to_plot=not args.no_plots,
        class_index_to_name=load_class_index_to_name(str(data_dir)),
    )

    # -- Save results --
    output_json = Path(args.output_json) if args.output_json else (
        checkpoint_path.parent / f"eval_{args.split}.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)

    model_config: dict = {
        "model_id": args.model,
        "scale": args.scale,
        "num_classes": args.num_classes,
        "reg_max": args.reg_max,
        "preprocessing": args.preprocessing,
    }
    if args.model == "mr_yolo":
        model_config["res_keys"] = args.res_keys
        model_config["input_resolutions"] = {k: list(v) for k, v in input_resolutions.items()}
    else:
        model_config["res_key"] = args.res_key
        model_config["res_hw"] = list(res_hw)

    payload = {
        "checkpoint": str(checkpoint_path),
        "dataset_path": str(data_dir),
        "split": args.split,
        "device": args.device,
        "batch_size": args.batch_size,
        "iou_thresh": args.iou_thresh,
        "false_alarm_target": args.false_alarm_target,
        "model_config": model_config,
        "metrics": full_metrics,
    }

    output_json.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )

    map_stats = full_metrics.get("map_stats", {})
    print("\n[Evaluation complete]")
    print(f"  JSON results : {output_json}")
    print(f"  mAP50        : {map_stats.get('mAP50')}")
    print(f"  mAP50:95     : {map_stats.get('mAP50:95')}")


if __name__ == "__main__":
    main()
