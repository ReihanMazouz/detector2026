from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from detector2026.core.models.mr_yolo_ablation import (  # noqa: E402
    MRYOLOPatchSpatialBranchCrossAttentionRTDETRHead,
)
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


DEFAULT_OUTPUT_DIR_PARENT = (
    "/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation/mr_yolo_ablation"
)
DEFAULT_RUN_NAME = "mr_yolo_patch_p2p4p5_branch_cross_attention_rtdetr_head"
DEFAULT_BACKBONE_CHECKPOINT = (
    "/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation/"
    "mr_yolo_ablation/mr_yolo_patch_p2p4p5_branch_cross_attention/best.pt"
)


def parse_hw(value: str) -> tuple[int, int]:
    if "," in value:
        left, right = value.split(",", 1)
    elif "x" in value.lower():
        left, right = value.lower().split("x", 1)
    else:
        raise argparse.ArgumentTypeError("Expected H,W or HxW.")
    return int(left), int(right)


def output_is_complete(output_dir: Path) -> bool:
    return (output_dir / "best.pt").exists() or (output_dir / "last.pt").exists()


def cleanup_after_run() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an RT-DETR one-to-one head on top of the frozen "
            "PatchSpatial+BranchCrossAttention backbone."
        )
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir-parent", default=DEFAULT_OUTPUT_DIR_PARENT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--backbone-checkpoint", default=DEFAULT_BACKBONE_CHECKPOINT)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--num-classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--reg-max", type=int, default=DEFAULT_REG_MAX)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--preprocessing", default=DEFAULT_PREPROCESSING)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--full-eval-every", type=int, default=5)
    parser.add_argument("--save-last-every", type=int, default=5)
    parser.add_argument("--width-mult", type=float, default=0.25)
    parser.add_argument("--outfusion-channels-mult", type=int, default=2)
    parser.add_argument("--constant-backbone-ch", type=int, default=0)
    parser.add_argument("--patch-size", type=parse_hw, default=(8, 8))
    parser.add_argument("--patch-d-model", type=int, default=128)
    parser.add_argument("--patch-num-heads", type=int, default=4)
    parser.add_argument("--patch-num-layers", type=int, default=1)
    parser.add_argument("--patch-num-points", type=int, default=16)
    parser.add_argument("--patch-ffn-ratio", type=float, default=2.0)
    parser.add_argument("--patch-dropout", type=float, default=0.0)
    parser.add_argument("--patch-alpha-bound", type=float, default=1.0)
    parser.add_argument("--fusion-mode", choices=("deformable", "global"), default="deformable")
    parser.add_argument("--center-resolution-index", type=int, default=None)
    parser.add_argument("--fusion-d-model", type=int, default=128)
    parser.add_argument("--fusion-num-heads", type=int, default=4)
    parser.add_argument("--fusion-num-layers", type=int, default=1)
    parser.add_argument("--fusion-num-points", type=int, default=4)
    parser.add_argument("--fusion-ffn-ratio", type=float, default=2.0)
    parser.add_argument("--fusion-dropout", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--num-decoder-layers", type=int, default=6)
    parser.add_argument("--num-heads-decoder", type=int, default=8)
    parser.add_argument("--num-decoder-points", type=int, default=8)
    parser.add_argument("--dim-feedforward-decoder", type=int, default=1024)
    parser.add_argument("--matcher-num-threads", type=int, default=8)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_resolutions = find_input_resolutions(args.data_dir)
    if len(input_resolutions) != len(DEFAULT_RES_KEYS):
        raise ValueError(
            "Mismatch between dataset resolutions and DEFAULT_RES_KEYS: "
            f"{len(input_resolutions)} resolutions for {len(DEFAULT_RES_KEYS)} keys."
        )

    input_channels = preprocessing_num_channels(args.preprocessing)
    output_dir = Path(args.output_dir_parent) / args.run_name
    backbone_checkpoint = Path(args.backbone_checkpoint)

    print("MRYOLOPatchSpatialBranchCrossAttentionRTDETRHead")
    print(f"  output_dir = {output_dir}")
    print(f"  backbone_checkpoint = {backbone_checkpoint} exists={backbone_checkpoint.is_file()}")
    print(f"  device = {args.device}")
    print(f"  preprocessing = {args.preprocessing}")
    print(f"  input_channels = {input_channels}")
    print("  backbone = PatchSpatialBranchCrossAttentionBackbone frozen")
    print("  one2many Detect head = replaced by RT-DETR one2one head")
    print(f"  RT-DETR decoder_layers = {args.num_decoder_layers}")
    print(f"  RT-DETR decoder_points = {args.num_decoder_points}")
    print(f"  use_amp = {not args.no_amp}")

    if args.dry_run:
        return
    if output_is_complete(output_dir) and not args.overwrite:
        print(f"[SKIP] checkpoint deja present dans {output_dir}")
        return
    if not backbone_checkpoint.is_file():
        raise FileNotFoundError(f"Backbone checkpoint not found: {backbone_checkpoint}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model = MRYOLOPatchSpatialBranchCrossAttentionRTDETRHead(
        input_resolutions=list(input_resolutions),
        output_dir=str(output_dir),
        num_classes=args.num_classes,
        reg_max=args.reg_max,
        device=args.device,
        in_ch=input_channels,
        width_mult=args.width_mult,
        outfusion_channels_mult=args.outfusion_channels_mult,
        constant_backbone_ch=args.constant_backbone_ch,
        patch_size=tuple(args.patch_size),
        patch_d_model=args.patch_d_model,
        patch_num_heads=args.patch_num_heads,
        patch_num_layers=args.patch_num_layers,
        patch_num_points=args.patch_num_points,
        patch_ffn_ratio=args.patch_ffn_ratio,
        patch_dropout=args.patch_dropout,
        patch_alpha_bound=None if args.patch_alpha_bound < 0 else args.patch_alpha_bound,
        fusion_mode=args.fusion_mode,
        center_resolution_index=args.center_resolution_index,
        fusion_d_model=args.fusion_d_model,
        fusion_num_heads=args.fusion_num_heads,
        fusion_num_layers=args.fusion_num_layers,
        fusion_num_points=args.fusion_num_points,
        fusion_ffn_ratio=args.fusion_ffn_ratio,
        fusion_dropout=args.fusion_dropout,
        hidden_dim=args.hidden_dim,
        num_queries=args.num_queries,
        num_decoder_layers=args.num_decoder_layers,
        num_heads_decoder=args.num_heads_decoder,
        num_decoder_points=args.num_decoder_points,
        dim_feedforward_decoder=args.dim_feedforward_decoder,
        matcher_num_threads=args.matcher_num_threads,
        debug=args.debug,
    )
    try:
        missing, unexpected = model.load_frozen_backbone_weights(str(backbone_checkpoint), device=args.device)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in model.backbone.parameters())
        print(
            f"[OK] frozen backbone loaded: missing={len(missing)} unexpected={len(unexpected)} "
            f"frozen_params={frozen_params} trainable_params={trainable_params}"
        )
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
            use_amp=not args.no_amp,
        )
    finally:
        del model
        cleanup_after_run()


if __name__ == "__main__":
    main()
