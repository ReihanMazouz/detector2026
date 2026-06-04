#!/usr/bin/env python3
"""
Unified training script for detector2026 clean models.

Supported models
----------------
  yolov8    YOLOv8  (single-resolution, one-to-many YOLO head)
  yolov11   YOLOv11 (single-resolution, one-to-many YOLO head)
  yolov12   YOLOv12-turbo (single-resolution, area-attention blocks)
  tf_attn   TF-Attn-YOLO  (YOLOv11 backbone + transformer attention)
  mr_yolo   MR-YOLO (multi-resolution branches, fused at P3/P4/P5)
  detr      DETR    (transformer encoder-decoder, Hungarian matching)

Scale options (width / depth multipliers)
-----------------------------------------
  n  nano   (width=0.25, depth=0.33)
  s  small  (width=0.50, depth=0.33)
  m  medium (width=0.75, depth=0.67)
  l  large  (width=1.00, depth=1.00)  -- YOLO family only

Examples
--------
  # Single-resolution YOLOv11 nano at 256×256
  python train.py --model yolov11 --scale n \\
      --data-dir /data/rf_dataset --res-key cfg512 --res-hw 256 256

  # Multi-resolution MR-YOLO nano
  python train.py --model mr_yolo --scale n \\
      --data-dir /data/rf_dataset \\
      --res-keys cfg512 cfg256 cfg128 cfg1024 cfg2048

  # DETR small at 256×256
  python train.py --model detr --scale s \\
      --data-dir /data/rf_dataset --res-key cfg512 --res-hw 256 256

  # Dry run (print config without training)
  python train.py --model yolov11 --scale n --dry-run \\
      --data-dir /data/rf_dataset --res-key cfg512 --res-hw 256 256
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from detector2026.clean.models import DETR, MR_YOLO, TF_Attn_Yolo, YOLOv8, YOLOv11, YOLOv12
from detector2026.clean.utils.preprocess import preprocessing_num_channels

Resolution = Tuple[int, int]

# ── Scale tables ─────────────────────────────────────────────────────────────

SCALE_WIDTH: Dict[str, float] = {"n": 0.25, "s": 0.50, "m": 0.75, "l": 1.00}
SCALE_DEPTH: Dict[str, float] = {"n": 0.33, "s": 0.33, "m": 0.67, "l": 1.00}

# Default ordered resolution keys (must match the order of tensors in .pt files)
DEFAULT_RES_KEYS: List[str] = ["cfg512", "cfg256", "cfg128", "cfg1024", "cfg2048"]

MR_BACKBONE_MODE = "TFSep_pyramid"
MR_OUTFUSION_CHANNELS_MULT = 1


# ── Dataset helpers ───────────────────────────────────────────────────────────

def _read_pt_resolutions(data_dir: str, split: str = "train") -> List[Resolution]:
    images_dir = Path(data_dir) / split / "data"
    pt_file = next(images_dir.glob("*.pt"), None)
    if pt_file is None:
        raise FileNotFoundError(f"No .pt file found in {images_dir}")
    specs = torch.load(pt_file, map_location="cpu")
    if not isinstance(specs, list):
        raise ValueError(f"Expected a list of tensors in {pt_file}, got {type(specs)}")
    resolutions: List[Resolution] = []
    for i, spec in enumerate(specs):
        if not torch.is_tensor(spec):
            raise ValueError(f"Element {i} in {pt_file} is not a tensor")
        h, w = (spec.shape[-2], spec.shape[-1]) if spec.ndim >= 2 else (int(spec.shape[0]), 1)
        resolutions.append((int(h), int(w)))
    return resolutions


def detect_mr_resolutions(data_dir: str, res_keys: List[str]) -> Dict[str, Resolution]:
    all_resolutions = _read_pt_resolutions(data_dir)
    if len(all_resolutions) != len(res_keys):
        raise ValueError(
            f"Dataset has {len(all_resolutions)} resolution tensors but "
            f"{len(res_keys)} res_keys provided: {res_keys}. "
            "Ensure --res-keys matches the number of tensors in each .pt file."
        )
    return dict(zip(res_keys, all_resolutions))


def detect_single_res_hw(data_dir: str, res_key: str) -> Resolution:
    all_resolutions = _read_pt_resolutions(data_dir)
    if len(all_resolutions) == 1:
        return all_resolutions[0]
    try:
        idx = DEFAULT_RES_KEYS.index(res_key)
    except ValueError:
        raise ValueError(
            f"Cannot auto-detect resolution for res_key='{res_key}'. "
            "Either pass --res-hw explicitly or use a standard res_key "
            f"from: {DEFAULT_RES_KEYS}."
        )
    if idx >= len(all_resolutions):
        raise ValueError(
            f"res_key='{res_key}' maps to index {idx} but the dataset only has "
            f"{len(all_resolutions)} resolution tensors. Pass --res-hw explicitly."
        )
    return all_resolutions[idx]


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
            outfusion_channels_mult=MR_OUTFUSION_CHANNELS_MULT,
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
            aux_loss_weight=kw.get("aux_loss_weight", 1.0),
            eos_coef=kw.get("eos_coef", 0.1),
        )

    raise ValueError(f"Unknown model: '{model_id}'")


# ── Output directory ──────────────────────────────────────────────────────────

def auto_output_dir_name(model_id: str, scale: str, res_key: str, res_keys: List[str]) -> str:
    if model_id == "mr_yolo":
        return f"mr_yolo_{scale}_fused_{'_'.join(res_keys)}"
    return f"{model_id}_{scale}_specificres_{res_key}"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a detector2026 model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Model --
    g = parser.add_argument_group("Model")
    g.add_argument(
        "--model", required=True,
        choices=["yolov8", "yolov11", "yolov12", "tf_attn", "mr_yolo", "detr"],
    )
    g.add_argument(
        "--scale", default="n", choices=["n", "s", "m", "l"],
        help="n=nano, s=small, m=medium, l=large (YOLO family only).",
    )
    g.add_argument("--num-classes", type=int, default=20)
    g.add_argument("--reg-max", type=int, default=16)

    # -- Data --
    g = parser.add_argument_group("Data")
    g.add_argument("--data-dir", required=True, help="Dataset root directory.")
    g.add_argument("--preprocessing", default="none")
    g.add_argument(
        "--res-key", default="cfg512",
        help="Resolution key for single-resolution models.",
    )
    g.add_argument(
        "--res-hw", type=int, nargs=2, metavar=("H", "W"), default=None,
        help="Input size (H W). Auto-detected from .pt files if omitted.",
    )
    g.add_argument(
        "--res-keys", nargs="+", default=None,
        help="Ordered resolution keys for mr_yolo (e.g. cfg512 cfg256 cfg1024).",
    )

    # -- Output --
    g = parser.add_argument_group("Output")
    g.add_argument(
        "--output-dir-parent", default="runs",
        help="Parent directory that will contain all experiment folders.",
    )
    g.add_argument(
        "--output-dir-name", default=None,
        help="Experiment folder name. Auto-generated if omitted.",
    )
    g.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    # -- Training --
    g = parser.add_argument_group("Training")
    g.add_argument("--epochs", type=int, default=100)
    g.add_argument("--batch-size", type=int, default=32)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--patience", type=int, default=10)
    g.add_argument("--num-workers", type=int, default=None)
    g.add_argument("--full-eval-every", type=int, default=5)
    g.add_argument("--save-last-every", type=int, default=5)
    g.add_argument(
        "--monitor", default="val_loss",
        choices=["val_loss", "map50", "map50_95"],
    )

    # -- DETR-specific --
    g = parser.add_argument_group("DETR-specific (ignored for other models)")
    g.add_argument("--detr-hidden-dim", type=int, default=256)
    g.add_argument("--detr-num-queries", type=int, default=100)
    g.add_argument("--detr-encoder-layers", type=int, default=2)
    g.add_argument("--detr-decoder-layers", type=int, default=3)
    g.add_argument("--detr-nheads", type=int, default=8)
    g.add_argument("--detr-dim-feedforward", type=int, default=1024)
    g.add_argument("--detr-dropout", type=float, default=0.0)
    g.add_argument("--detr-aux-loss-weight", type=float, default=1.0)
    g.add_argument("--detr-eos-coef", type=float, default=0.1)

    parser.add_argument("--dry-run", action="store_true", help="Print config and exit.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.model == "mr_yolo":
        if not args.res_keys:
            raise ValueError("--res-keys is required for mr_yolo.")
        if args.scale == "l":
            raise ValueError("Scale 'l' is not supported for mr_yolo (max scale: m).")
    else:
        if not args.res_key:
            raise ValueError(f"--res-key is required for {args.model}.")
    if args.model == "detr" and args.lr > 5e-4:
        print(f"[warning] DETR typically trains with lr ≤ 1e-4. You used lr={args.lr}.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    validate_args(args)

    input_channels = preprocessing_num_channels(args.preprocessing)

    # -- Resolve resolutions --
    input_resolutions: Optional[Dict[str, Resolution]] = None
    select_res: Optional[dict] = None
    dataset_mode: str

    if args.model == "mr_yolo":
        input_resolutions = detect_mr_resolutions(args.data_dir, args.res_keys)
        dataset_mode = "fused"
        select_res = {"res_keys": list(input_resolutions.keys())}
        print("Detected input resolutions:")
        for key, hw in input_resolutions.items():
            print(f"  {key}: {hw}")
    else:
        if args.res_hw is not None:
            res_hw: Resolution = tuple(args.res_hw)
        else:
            res_hw = detect_single_res_hw(args.data_dir, args.res_key)
            print(f"Auto-detected resolution for '{args.res_key}': {res_hw}")
        dataset_mode = "specificres"
        select_res = {"res_hw": res_hw, "res_key": args.res_key}

    # -- Build output path --
    dir_name = args.output_dir_name or auto_output_dir_name(
        args.model, args.scale, args.res_key, args.res_keys or []
    )
    output_dir = str(Path(args.output_dir_parent) / dir_name)

    # -- Print summary --
    print(f"\nModel      : {args.model} ({args.scale})")
    print(f"Device     : {args.device}")
    print(f"Output     : {output_dir}")
    print(f"Channels   : {input_channels}  (preprocessing={args.preprocessing})")
    print(f"Dataset    : {dataset_mode}  select_res={select_res}")
    print(f"Epochs     : {args.epochs}  batch={args.batch_size}  lr={args.lr}")
    print(f"Monitor    : {args.monitor}  patience={args.patience}")

    if args.dry_run:
        print("\n[dry-run] Exiting without training.")
        return

    if Path(output_dir).exists():
        print(f"\n[SKIP] Output directory already exists: {output_dir}")
        return

    # -- Build model --
    detr_kwargs = {
        "hidden_dim": args.detr_hidden_dim,
        "num_queries": args.detr_num_queries,
        "encoder_layers": args.detr_encoder_layers,
        "decoder_layers": args.detr_decoder_layers,
        "nheads": args.detr_nheads,
        "dim_feedforward": args.detr_dim_feedforward,
        "dropout": args.detr_dropout,
        "aux_loss_weight": args.detr_aux_loss_weight,
        "eos_coef": args.detr_eos_coef,
    }
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

    # -- Train --
    try:
        model.fit(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            dataset=dataset_mode,
            preprocessing=args.preprocessing,
            select_res=select_res,
            num_workers=args.num_workers,
            full_eval_every=args.full_eval_every,
            save_last_every=args.save_last_every,
            monitor=args.monitor,
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
