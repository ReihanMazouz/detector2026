from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from detector2026.core.models.mr_yolo_ablation import (  # noqa: E402
    MRYOLOPatchSpatialBranchCrossAttentionAblation,
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
DEFAULT_RUN_NAME = "mr_yolo_patch_p2p4p5_branch_cross_attention"
DEFAULT_EPOCHS = 300
DEFAULT_PATIENCE = 100
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-3
DEFAULT_NUM_WORKERS = None
DEFAULT_FULL_EVAL_EVERY = 5
DEFAULT_SAVE_LAST_EVERY = 5
DEFAULT_MONITOR = "val_loss"
DEFAULT_WIDTH_MULT = 0.25
DEFAULT_PATCH_SIZE = (8, 8)
DEFAULT_PATCH_D_MODEL = 128
DEFAULT_PATCH_NUM_HEADS = 4
DEFAULT_PATCH_NUM_LAYERS = 1
DEFAULT_PATCH_NUM_POINTS = 16
DEFAULT_PATCH_FFN_RATIO = 2.0
DEFAULT_PATCH_DROPOUT = 0.0
DEFAULT_FUSION_MODE = "deformable"
DEFAULT_FUSION_D_MODEL = 128
DEFAULT_FUSION_NUM_HEADS = 4
DEFAULT_FUSION_NUM_LAYERS = 1
DEFAULT_FUSION_NUM_POINTS = 4
DEFAULT_FUSION_FFN_RATIO = 2.0
DEFAULT_FUSION_DROPOUT = 0.0


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
            "Run MR-YOLO ablation with patch spatial attention on P2/P4/P5, "
            "P3 branch cross-attention fusion, and no C2PSA."
        )
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir-parent", default=DEFAULT_OUTPUT_DIR_PARENT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--num-classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--reg-max", type=int, default=DEFAULT_REG_MAX)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--preprocessing", default=DEFAULT_PREPROCESSING)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--full-eval-every", type=int, default=DEFAULT_FULL_EVAL_EVERY)
    parser.add_argument("--save-last-every", type=int, default=DEFAULT_SAVE_LAST_EVERY)
    parser.add_argument("--monitor", default=DEFAULT_MONITOR)
    parser.add_argument("--width-mult", type=float, default=DEFAULT_WIDTH_MULT)
    parser.add_argument("--outfusion-channels-mult", type=int, default=2)
    parser.add_argument("--constant-backbone-ch", type=int, default=0)
    parser.add_argument("--patch-size", type=parse_hw, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--patch-d-model", type=int, default=DEFAULT_PATCH_D_MODEL)
    parser.add_argument("--patch-num-heads", type=int, default=DEFAULT_PATCH_NUM_HEADS)
    parser.add_argument("--patch-num-layers", type=int, default=DEFAULT_PATCH_NUM_LAYERS)
    parser.add_argument("--patch-num-points", type=int, default=DEFAULT_PATCH_NUM_POINTS)
    parser.add_argument("--patch-ffn-ratio", type=float, default=DEFAULT_PATCH_FFN_RATIO)
    parser.add_argument("--patch-dropout", type=float, default=DEFAULT_PATCH_DROPOUT)
    parser.add_argument(
        "--patch-alpha-bound",
        type=float,
        default=1.0,
        help="Bound patch gate alpha as bound*tanh(alpha). Use a negative value to disable.",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="Clip gradient norm before optimizer.step. Use a negative value to disable clipping.",
    )
    parser.add_argument("--fusion-mode", choices=("deformable", "global"), default=DEFAULT_FUSION_MODE)
    parser.add_argument("--center-resolution-index", type=int, default=None)
    parser.add_argument("--fusion-d-model", type=int, default=DEFAULT_FUSION_D_MODEL)
    parser.add_argument("--fusion-num-heads", type=int, default=DEFAULT_FUSION_NUM_HEADS)
    parser.add_argument("--fusion-num-layers", type=int, default=DEFAULT_FUSION_NUM_LAYERS)
    parser.add_argument("--fusion-num-points", type=int, default=DEFAULT_FUSION_NUM_POINTS)
    parser.add_argument("--fusion-ffn-ratio", type=float, default=DEFAULT_FUSION_FFN_RATIO)
    parser.add_argument("--fusion-dropout", type=float, default=DEFAULT_FUSION_DROPOUT)
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Optional checkpoint path used to initialize model weights before training.",
    )
    parser.add_argument(
        "--resume-from-best",
        action="store_true",
        help="Initialize from <output_dir>/best.pt before continuing training.",
    )
    parser.add_argument("--debug", action="store_true", help="Raise as soon as model activations become NaN/Inf.")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision.")
    parser.add_argument("--overwrite", action="store_true", help="Run even if best.pt/last.pt already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print configuration without training.")
    args = parser.parse_args()
    if args.resume_from_best and args.resume_checkpoint is not None:
        raise ValueError("Use either --resume-from-best or --resume-checkpoint, not both.")
    return args


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
    resume_checkpoint = Path(args.resume_checkpoint) if args.resume_checkpoint else None
    if args.resume_from_best:
        resume_checkpoint = output_dir / "best.pt"

    print("MRYOLOPatchSpatialBranchCrossAttentionAblation")
    print(f"  output_dir = {output_dir}")
    print(f"  device = {args.device}")
    print(f"  preprocessing = {args.preprocessing}")
    print(f"  input_channels = {input_channels}")
    print(f"  width_mult = {args.width_mult}")
    print(f"  use_amp = {not args.no_amp}")
    print(f"  debug = {args.debug}")
    print(f"  grad_clip_norm = {None if args.grad_clip_norm < 0 else args.grad_clip_norm}")
    print(f"  resume_checkpoint = {resume_checkpoint}")
    print("  patch spatial attention:")
    print(f"    locations = P2, P4, P5")
    print(f"    patch_size = {tuple(args.patch_size)}")
    print(f"    d_model = {args.patch_d_model}")
    print(f"    num_heads = {args.patch_num_heads}")
    print(f"    num_layers = {args.patch_num_layers}")
    print(f"    num_points = {args.patch_num_points}")
    print(f"    alpha_bound = {None if args.patch_alpha_bound < 0 else args.patch_alpha_bound}")
    print("  branch cross-attention fusion:")
    print(f"    location = P3")
    print(f"    mode = {args.fusion_mode}")
    print(f"    d_model = {args.fusion_d_model}")
    print(f"    num_heads = {args.fusion_num_heads}")
    print(f"    num_layers = {args.fusion_num_layers}")
    print(f"    num_points = {args.fusion_num_points}")
    print("  C2PSA = disabled")
    print("  resolutions:")
    for index, (res_key, res_hw) in enumerate(zip(DEFAULT_RES_KEYS, input_resolutions)):
        print(f"    {index}. {res_key}: {res_hw}")

    if args.dry_run:
        return
    if output_is_complete(output_dir) and not args.overwrite and resume_checkpoint is None:
        print(f"[SKIP] checkpoint deja present dans {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    model = MRYOLOPatchSpatialBranchCrossAttentionAblation(
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
        debug=args.debug,
    )
    model._grad_clip_norm = None if args.grad_clip_norm < 0 else args.grad_clip_norm
    model._check_finite_after_step = bool(args.debug)
    try:
        if resume_checkpoint is not None:
            if not resume_checkpoint.is_file():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")
            state_dict = torch.load(resume_checkpoint, map_location=args.device)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(
                f"[OK] resumed weights from {resume_checkpoint} "
                f"(missing={len(missing)} unexpected={len(unexpected)})"
            )

        model.fit(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            dataset="fused",
            preprocessing=args.preprocessing,
            select_res={"res_keys": tuple(DEFAULT_RES_KEYS)},
            num_workers=args.num_workers,
            full_eval_every=args.full_eval_every,
            save_last_every=args.save_last_every,
            monitor=args.monitor,
            use_amp=not args.no_amp,
        )
    finally:
        del model
        cleanup_after_run()


if __name__ == "__main__":
    main()
