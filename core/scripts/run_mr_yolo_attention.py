from __future__ import annotations

import argparse
import gc
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from detector2026.core.models.mr_yolo_ablation import (  # noqa: E402
    MRYOLOBranchCrossAttentionAblation,
    MRYOLOInputCrossAttentionAblation,
    MRYOLOPatchSpatialAttentionAblation,
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
DEFAULT_EPOCHS = 300
DEFAULT_PATIENCE = 100
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-3
DEFAULT_NUM_WORKERS = None
DEFAULT_FULL_EVAL_EVERY = 5
DEFAULT_SAVE_LAST_EVERY = 5
DEFAULT_MONITOR = "val_loss"
DEFAULT_WIDTH_MULT = 0.25
DEFAULT_FUSION_D_MODEL = 128
DEFAULT_FUSION_NUM_HEADS = 4
DEFAULT_FUSION_NUM_LAYERS = 1
DEFAULT_FUSION_NUM_POINTS = 4
DEFAULT_FUSION_FFN_RATIO = 2.0
DEFAULT_FUSION_DROPOUT = 0.0
DEFAULT_INPUT_ENCODER_CHANNELS = 16
DEFAULT_PATCH_D_MODEL = 128
DEFAULT_PATCH_NUM_HEADS = 4
DEFAULT_PATCH_NUM_LAYERS = 1
DEFAULT_PATCH_NUM_POINTS = 16
DEFAULT_PATCH_FFN_RATIO = 2.0
DEFAULT_PATCH_DROPOUT = 0.0
DEFAULT_PATCH_LATENT_GRID = (16, 16)


@dataclass(frozen=True)
class TrainingJob:
    label: str
    output_dir_name: str
    model_builder: Callable[[str], torch.nn.Module]


def output_is_complete(output_dir: Path) -> bool:
    return (output_dir / "best.pt").exists() or (output_dir / "last.pt").exists()


def cleanup_after_run() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_jobs(args, input_resolutions: Sequence[tuple[int, int]], input_channels: int) -> list[TrainingJob]:
    common_kwargs = dict(
        input_resolutions=list(input_resolutions),
        num_classes=args.num_classes,
        reg_max=args.reg_max,
        device=args.device,
        in_ch=input_channels,
        width_mult=args.width_mult,
        fusion_mode="deformable",
        center_resolution_index=args.center_resolution_index,
        fusion_d_model=args.fusion_d_model,
        fusion_num_heads=args.fusion_num_heads,
        fusion_num_layers=args.fusion_num_layers,
        fusion_num_points=args.fusion_num_points,
        fusion_ffn_ratio=args.fusion_ffn_ratio,
        fusion_dropout=args.fusion_dropout,
    )

    return [
        TrainingJob(
            label="MRYOLOBranchCrossAttentionAblation, deformable",
            output_dir_name="mr_yolo_branch_cross_attention_deformable",
            model_builder=lambda output_dir: MRYOLOBranchCrossAttentionAblation(
                output_dir=output_dir,
                outfusion_channels_mult=args.outfusion_channels_mult,
                constant_backbone_ch=args.constant_backbone_ch,
                **common_kwargs,
            ),
        ),
        TrainingJob(
            label="MRYOLOInputCrossAttentionAblation, deformable",
            output_dir_name="mr_yolo_input_cross_attention_deformable",
            model_builder=lambda output_dir: MRYOLOInputCrossAttentionAblation(
                output_dir=output_dir,
                encoder_channels=args.input_encoder_channels,
                **common_kwargs,
            ),
        ),
        TrainingJob(
            label="MRYOLOPatchSpatialAttentionAblation, x2",
            output_dir_name="mr_yolo_patch_spatial_attention_x2",
            model_builder=lambda output_dir: MRYOLOPatchSpatialAttentionAblation(
                input_resolutions=list(input_resolutions),
                output_dir=output_dir,
                num_classes=args.num_classes,
                reg_max=args.reg_max,
                device=args.device,
                in_ch=input_channels,
                width_mult=args.width_mult,
                outfusion_channels_mult=args.outfusion_channels_mult,
                constant_backbone_ch=args.constant_backbone_ch,
                patch_latent_grid_hw=tuple(args.patch_latent_grid),
                patch_d_model=args.patch_d_model,
                patch_num_heads=args.patch_num_heads,
                patch_num_layers=args.patch_num_layers,
                patch_num_points=args.patch_num_points,
                patch_ffn_ratio=args.patch_ffn_ratio,
                patch_dropout=args.patch_dropout,
            ),
        ),
    ]


def run_job(job: TrainingJob, args, output_dir_parent: Path) -> None:
    output_dir = output_dir_parent / job.output_dir_name
    if output_dir.exists() and output_is_complete(output_dir) and not args.overwrite:
        print(f"[SKIP] {job.label}: checkpoint deja present dans {output_dir}")
        return

    print(f"[RUN ] {job.label}")
    print(f"       output_dir = {output_dir}")
    model = job.model_builder(str(output_dir))
    try:
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
        )
    finally:
        del model
        cleanup_after_run()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run MR-YOLO attention ablations: BranchCrossAttention, "
            "InputCrossAttention and PatchSpatialAttention."
        )
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir-parent", default=DEFAULT_OUTPUT_DIR_PARENT)
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
    parser.add_argument("--center-resolution-index", type=int, default=0)
    parser.add_argument("--fusion-d-model", type=int, default=DEFAULT_FUSION_D_MODEL)
    parser.add_argument("--fusion-num-heads", type=int, default=DEFAULT_FUSION_NUM_HEADS)
    parser.add_argument("--fusion-num-layers", type=int, default=DEFAULT_FUSION_NUM_LAYERS)
    parser.add_argument("--fusion-num-points", type=int, default=DEFAULT_FUSION_NUM_POINTS)
    parser.add_argument("--fusion-ffn-ratio", type=float, default=DEFAULT_FUSION_FFN_RATIO)
    parser.add_argument("--fusion-dropout", type=float, default=DEFAULT_FUSION_DROPOUT)
    parser.add_argument("--input-encoder-channels", type=int, default=DEFAULT_INPUT_ENCODER_CHANNELS)
    parser.add_argument("--patch-d-model", type=int, default=DEFAULT_PATCH_D_MODEL)
    parser.add_argument("--patch-num-heads", type=int, default=DEFAULT_PATCH_NUM_HEADS)
    parser.add_argument("--patch-num-layers", type=int, default=DEFAULT_PATCH_NUM_LAYERS)
    parser.add_argument("--patch-num-points", type=int, default=DEFAULT_PATCH_NUM_POINTS)
    parser.add_argument("--patch-ffn-ratio", type=float, default=DEFAULT_PATCH_FFN_RATIO)
    parser.add_argument("--patch-dropout", type=float, default=DEFAULT_PATCH_DROPOUT)
    parser.add_argument(
        "--patch-latent-grid",
        type=lambda value: tuple(int(part) for part in value.lower().replace("x", ",").split(",")),
        default=DEFAULT_PATCH_LATENT_GRID,
        help="Latent token grid for patch spatial attention, formatted as H,W or HxW.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Run even if best.pt/last.pt already exists.")
    parser.add_argument("--dry-run", action="store_true", help="List the planned jobs without training.")
    args = parser.parse_args()
    if len(args.patch_latent_grid) != 2:
        raise ValueError("--patch-latent-grid must contain exactly two integers.")
    return args


def main() -> None:
    args = parse_args()
    output_dir_parent = Path(args.output_dir_parent)
    output_dir_parent.mkdir(parents=True, exist_ok=True)

    input_resolutions = find_input_resolutions(args.data_dir)
    if len(input_resolutions) != len(DEFAULT_RES_KEYS):
        raise ValueError(
            "Mismatch between dataset resolutions and DEFAULT_RES_KEYS: "
            f"{len(input_resolutions)} resolutions for {len(DEFAULT_RES_KEYS)} keys."
        )
    if not 0 <= int(args.center_resolution_index) < len(input_resolutions):
        raise ValueError(
            f"--center-resolution-index must be in [0, {len(input_resolutions) - 1}], "
            f"got {args.center_resolution_index}."
        )

    input_channels = preprocessing_num_channels(args.preprocessing)
    jobs = build_jobs(args, input_resolutions, input_channels)

    print("Resolutions utilisees:")
    for index, (res_key, res_hw) in enumerate(zip(DEFAULT_RES_KEYS, input_resolutions)):
        marker = " (centre)" if index == args.center_resolution_index else ""
        print(f"  {index}. {res_key}: {res_hw}{marker}")
    print(f"Preprocessing = {args.preprocessing}")
    print(f"Input channels = {input_channels}")
    print(f"Device = {args.device}")
    print(f"Output parent = {output_dir_parent}")
    print("Jobs planifies:")
    for job in jobs:
        output_dir = output_dir_parent / job.output_dir_name
        status = "skip" if output_dir.exists() and output_is_complete(output_dir) and not args.overwrite else "run"
        print(f"  - [{status}] {job.label} -> {output_dir}")

    if args.dry_run:
        return

    for job in jobs:
        run_job(job, args, output_dir_parent)


if __name__ == "__main__":
    main()
