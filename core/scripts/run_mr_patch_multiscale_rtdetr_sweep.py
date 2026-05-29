"""Sweep training script for MRPatchMultiScaleRTDETRHead with enc4.

Trains multiple decoder configurations sequentially. Each run is saved in its
own sub-directory under --output-root. All configurations use 4 encoder layers
(enc4) — the 1-layer upgrade cost (~6.9 G MACs) is paid once; decoder layers
are cheap (~140 M MACs each).

Configurations (label, dec_layers, dec_pts, dec_heads):
  enc4_dec2_pts8_h8   — minimal decoder (reference)
  enc4_dec3_pts8_h8
  enc4_dec4_pts8_h8   — max capacity at pts8
  enc4_dec4_pts16_h8  — more sampling points
  enc4_dec4_pts8_h4   — fewer heads (cheaper self-attn)

Usage:
    python run_mr_patch_multiscale_rtdetr_sweep.py [--dry-run] [--overwrite]
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from detector2026.core.models.mr_yolo_ablation import MRPatchMultiScaleRTDETRHead  # noqa: E402
from detector2026.core.scripts.train_benchmark_suite import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_DEVICE,
    DEFAULT_NUM_CLASSES,
    DEFAULT_PREPROCESSING,
    DEFAULT_REG_MAX,
    DEFAULT_RES_KEYS,
    find_input_resolutions,
)
from detector2026.core.utils.preprocess import preprocessing_num_channels  # noqa: E402


DEFAULT_OUTPUT_ROOT = (
    "/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation/mr_yolo_ablation"
)
DEFAULT_BACKBONE_CHECKPOINT = (
    "/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation/"
    "mr_yolo_ablation/mr_patch_backbone_yolo_one2many_head/best.pt"
)

# (tag, num_encoder_layers, num_decoder_layers, num_decoder_points, num_heads_decoder)
SWEEP_CONFIGS: List[Tuple[str, int, int, int, int]] = [
    ("enc4_dec2_pts8_h8",  4, 2,  8, 8),
    ("enc4_dec3_pts8_h8",  4, 3,  8, 8),
    ("enc4_dec4_pts8_h8",  4, 4,  8, 8),
    ("enc4_dec4_pts16_h8", 4, 4, 16, 8),
    ("enc4_dec4_pts8_h4",  4, 4,  8, 4),
]


@dataclass
class SweepConfig:
    tag: str
    num_encoder_layers: int
    num_decoder_layers: int
    num_decoder_points: int
    num_heads_decoder: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep MRPatchMultiScaleRTDETRHead enc4 configs.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--backbone-checkpoint",
        default=DEFAULT_BACKBONE_CHECKPOINT,
        help="Optional: initialise backbone from a pre-trained MRPatchBackbone* checkpoint.",
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--num-classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--reg-max", type=int, default=DEFAULT_REG_MAX)
    parser.add_argument("--preprocessing", default=DEFAULT_PREPROCESSING)
    # Training schedule
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--full-eval-every", type=int, default=5)
    parser.add_argument("--save-last-every", type=int, default=5)
    # Fixed backbone architecture
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--num-heads-backbone", type=int, default=4)
    parser.add_argument("--num-intra-points", type=int, default=8)
    parser.add_argument("--num-inter-neighbors", type=int, default=8)
    parser.add_argument("--dim-feedforward-backbone", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)
    # Fixed RT-DETR head settings
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--dim-feedforward-decoder", type=int, default=1024)
    parser.add_argument("--matcher-num-threads", type=int, default=8)
    # Control
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Subset of config tags to run (e.g. enc4_dec2_pts8_h8). Runs all if omitted.",
    )
    return parser.parse_args()


def output_is_complete(output_dir: Path) -> bool:
    return (output_dir / "best.pt").exists() or (output_dir / "last.pt").exists()


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_config(
    cfg: SweepConfig,
    args: argparse.Namespace,
    input_resolutions: list,
    input_channels: int,
) -> None:
    output_dir = Path(args.output_root) / f"mr_patch_multiscale_rtdetr_{cfg.tag}"
    backbone_ckpt = Path(args.backbone_checkpoint)

    num_levels = len(input_resolutions)
    patch_shapes = [(h // args.patch_size, w // args.patch_size) for h, w in input_resolutions]
    total_tokens = sum(h * w for h, w in patch_shapes)

    print(f"\n{'='*70}")
    print(f"  Config: {cfg.tag}")
    print(f"{'='*70}")
    print(f"  output_dir     = {output_dir}")
    print(f"  backbone_ckpt  = {backbone_ckpt}  exists={backbone_ckpt.is_file()}")
    print(f"  num_levels={num_levels}  total_tokens={total_tokens}")
    print(f"  backbone: enc_layers={cfg.num_encoder_layers}  d_model={args.d_model}  patch_size={args.patch_size}")
    print(f"  decoder:  layers={cfg.num_decoder_layers}  pts={cfg.num_decoder_points}  heads={cfg.num_heads_decoder}")

    if args.dry_run:
        print("  [DRY RUN] skipping training")
        return

    if output_is_complete(output_dir) and not args.overwrite:
        print(f"  [SKIP] checkpoint already present in {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    model = MRPatchMultiScaleRTDETRHead(
        input_resolutions=list(input_resolutions),
        output_dir=str(output_dir),
        num_classes=args.num_classes,
        reg_max=args.reg_max,
        device=args.device,
        in_ch=input_channels,
        d_model=args.d_model,
        patch_size=args.patch_size,
        num_encoder_layers=cfg.num_encoder_layers,
        num_heads_backbone=args.num_heads_backbone,
        num_intra_points=args.num_intra_points,
        num_inter_neighbors=args.num_inter_neighbors,
        dim_feedforward_backbone=args.dim_feedforward_backbone,
        dropout=args.dropout,
        hidden_dim=args.hidden_dim,
        num_queries=args.num_queries,
        num_decoder_layers=cfg.num_decoder_layers,
        num_heads_decoder=cfg.num_heads_decoder,
        num_decoder_points=cfg.num_decoder_points,
        dim_feedforward_decoder=args.dim_feedforward_decoder,
        matcher_num_threads=args.matcher_num_threads,
    )
    try:
        if backbone_ckpt.is_file():
            missing, unexpected = model.load_backbone_weights(str(backbone_ckpt), device=args.device)
            print(f"  [OK] backbone weights loaded — missing={len(missing)}  unexpected={len(unexpected)}")
        else:
            print("  [INFO] no backbone checkpoint — training from scratch")

        model.fit(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            preprocessing=args.preprocessing,
            res_keys=tuple(DEFAULT_RES_KEYS),
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            full_eval_every=args.full_eval_every,
            save_last_every=args.save_last_every,
            monitor="val_loss",
        )
    finally:
        del model
        cleanup()

    print(f"  [DONE] {output_dir}")


def main() -> None:
    args = parse_args()
    input_resolutions = find_input_resolutions(args.data_dir)
    if len(input_resolutions) != len(DEFAULT_RES_KEYS):
        raise ValueError(
            f"Expected {len(DEFAULT_RES_KEYS)} resolutions, found {len(input_resolutions)}."
        )
    input_channels = preprocessing_num_channels(args.preprocessing)

    configs = [SweepConfig(*c) for c in SWEEP_CONFIGS]
    if args.configs:
        selected = set(args.configs)
        configs = [c for c in configs if c.tag in selected]
        if not configs:
            raise ValueError(f"No matching configs for: {args.configs}")

    print(f"MRPatchMultiScaleRTDETRHead enc4 sweep — {len(configs)} configuration(s)")
    for cfg in configs:
        train_config(cfg, args, input_resolutions, input_channels)

    print("\n[ALL DONE]")


if __name__ == "__main__":
    main()
