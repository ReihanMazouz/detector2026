from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from detector2026.core.models.mr_yolo_ablation import MRPatchBackboneYOLOOne2ManyHead  # noqa: E402
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
DEFAULT_RUN_NAME = "mr_patch_backbone_yolo_one2many_head"
DEFAULT_EPOCHS = 300
DEFAULT_PATIENCE = 100
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-3
DEFAULT_NUM_WORKERS = None
DEFAULT_FULL_EVAL_EVERY = 5
DEFAULT_SAVE_LAST_EVERY = 5
DEFAULT_MONITOR = "val_loss"
DEFAULT_D_MODEL = 128
DEFAULT_PATCH_SIZE = 8
DEFAULT_ENCODER_LAYERS = 3
DEFAULT_NUM_HEADS = 4
DEFAULT_INTRA_POINTS = 8
DEFAULT_INTER_NEIGHBORS = 8
DEFAULT_FFN_DIM = 512
DEFAULT_DROPOUT = 0.0
DEFAULT_P3_HW = (32, 32)
DEFAULT_STRIDE = 32


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
        description="Run the MR isotropic patch backbone with single-scale YOLO one-to-many head."
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

    parser.add_argument("--d-model", type=int, default=DEFAULT_D_MODEL)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--num-encoder-layers", type=int, default=DEFAULT_ENCODER_LAYERS)
    parser.add_argument("--num-heads", type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument("--num-intra-points", type=int, default=DEFAULT_INTRA_POINTS)
    parser.add_argument("--num-inter-neighbors", type=int, default=DEFAULT_INTER_NEIGHBORS)
    parser.add_argument("--dim-feedforward", type=int, default=DEFAULT_FFN_DIM)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--p3-hw", type=parse_hw, default=DEFAULT_P3_HW)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)

    parser.add_argument("--overwrite", action="store_true", help="Run even if best.pt/last.pt already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print configuration without training.")
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

    print("MRPatchBackboneYOLOOne2ManyHead ablation")
    print(f"  output_dir = {output_dir}")
    print(f"  device = {args.device}")
    print(f"  preprocessing = {args.preprocessing}")
    print(f"  input_channels = {input_channels}")
    print(f"  d_model = {args.d_model}")
    print(f"  patch_size = {args.patch_size}")
    print(f"  encoder_layers = {args.num_encoder_layers}")
    print(f"  num_heads = {args.num_heads}")
    print(f"  intra_points = {args.num_intra_points}")
    print(f"  inter_neighbors = {args.num_inter_neighbors}")
    print(f"  p3_hw = {tuple(args.p3_hw)}")
    print(f"  yolo_stride = {args.stride}")
    print("  resolutions:")
    for index, (res_key, res_hw) in enumerate(zip(DEFAULT_RES_KEYS, input_resolutions)):
        patch_h = args.patch_size if res_hw[0] % args.patch_size == 0 else None
        patch_w = args.patch_size if res_hw[1] % args.patch_size == 0 else None
        grid_h = res_hw[0] // args.patch_size if patch_h is not None else None
        grid_w = res_hw[1] // args.patch_size if patch_w is not None else None
        print(f"    {index}. {res_key}: {res_hw}, patch=({patch_h},{patch_w}), grid=({grid_h},{grid_w})")

    if args.dry_run:
        return
    if output_is_complete(output_dir) and not args.overwrite:
        print(f"[SKIP] checkpoint deja present dans {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    model = MRPatchBackboneYOLOOne2ManyHead(
        input_resolutions=list(input_resolutions),
        output_dir=str(output_dir),
        num_classes=args.num_classes,
        reg_max=args.reg_max,
        device=args.device,
        in_ch=input_channels,
        d_model=args.d_model,
        patch_size=args.patch_size,
        num_encoder_layers=args.num_encoder_layers,
        num_heads=args.num_heads,
        num_intra_points=args.num_intra_points,
        num_inter_neighbors=args.num_inter_neighbors,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        p3_hw=tuple(args.p3_hw),
        stride=args.stride,
    )
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


if __name__ == "__main__":
    main()
