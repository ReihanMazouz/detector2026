from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from detector2026.core.models.mr_yolo_ablation import MRViTPatchDetector  # noqa: E402
from detector2026.core.scripts.train_benchmark_suite import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_DEVICE,
    DEFAULT_NUM_CLASSES,
    DEFAULT_PREPROCESSING,
    DEFAULT_RES_KEYS,
    find_input_resolutions,
)
from detector2026.core.utils.preprocess import preprocessing_num_channels  # noqa: E402


DEFAULT_OUTPUT_DIR_PARENT = (
    "/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation/mr_yolo_ablation"
)
DEFAULT_RUN_NAME = "mr_vit_patch_detector_rtdetr"
DEFAULT_EPOCHS = 300
DEFAULT_PATIENCE = 100
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 1e-4
DEFAULT_NUM_WORKERS = None
DEFAULT_FULL_EVAL_EVERY = 5
DEFAULT_SAVE_LAST_EVERY = 5
DEFAULT_MONITOR = "val_loss"
DEFAULT_D_MODEL = 256
DEFAULT_ENCODER_LAYERS = 6
DEFAULT_DECODER_LAYERS = 6
DEFAULT_QUERIES = 100
DEFAULT_PATCH_GRID = (32, 32)
DEFAULT_NUM_HEADS = 8
DEFAULT_ENCODER_POINTS = 16
DEFAULT_DECODER_POINTS = 16
DEFAULT_FFN_DIM = 1024
DEFAULT_DROPOUT = 0.0
DEFAULT_MATCHER_THREADS = 1


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
        description="Run the MRViTPatchDetector RT-DETR-like multi-resolution patch ablation."
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir-parent", default=DEFAULT_OUTPUT_DIR_PARENT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--num-classes", type=int, default=DEFAULT_NUM_CLASSES)
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
    parser.add_argument("--num-encoder-layers", type=int, default=DEFAULT_ENCODER_LAYERS)
    parser.add_argument("--num-decoder-layers", type=int, default=DEFAULT_DECODER_LAYERS)
    parser.add_argument("--num-queries", type=int, default=DEFAULT_QUERIES)
    parser.add_argument("--patch-grid", type=parse_hw, default=DEFAULT_PATCH_GRID)
    parser.add_argument("--num-heads", type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument("--num-encoder-points", type=int, default=DEFAULT_ENCODER_POINTS)
    parser.add_argument("--num-decoder-points", type=int, default=DEFAULT_DECODER_POINTS)
    parser.add_argument("--dim-feedforward", type=int, default=DEFAULT_FFN_DIM)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--matcher-num-threads", type=int, default=DEFAULT_MATCHER_THREADS)

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

    print("MRViTPatchDetector ablation")
    print(f"  output_dir = {output_dir}")
    print(f"  device = {args.device}")
    print(f"  preprocessing = {args.preprocessing}")
    print(f"  input_channels = {input_channels}")
    print(f"  patch_grid = {tuple(args.patch_grid)}")
    print(f"  d_model = {args.d_model}")
    print(f"  encoder_layers = {args.num_encoder_layers}")
    print(f"  decoder_layers = {args.num_decoder_layers}")
    print(f"  num_queries = {args.num_queries}")
    print("  resolutions:")
    for index, (res_key, res_hw) in enumerate(zip(DEFAULT_RES_KEYS, input_resolutions)):
        patch_h = res_hw[0] // args.patch_grid[0] if res_hw[0] % args.patch_grid[0] == 0 else None
        patch_w = res_hw[1] // args.patch_grid[1] if res_hw[1] % args.patch_grid[1] == 0 else None
        print(f"    {index}. {res_key}: {res_hw}, patch=({patch_h},{patch_w})")

    if args.dry_run:
        return
    if output_is_complete(output_dir) and not args.overwrite:
        print(f"[SKIP] checkpoint deja present dans {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    model = MRViTPatchDetector(
        input_resolutions=list(input_resolutions),
        output_dir=str(output_dir),
        num_classes=args.num_classes,
        device=args.device,
        in_ch=input_channels,
        d_model=args.d_model,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        num_queries=args.num_queries,
        patch_grid_hw=tuple(args.patch_grid),
        num_heads=args.num_heads,
        num_encoder_points=args.num_encoder_points,
        num_decoder_points=args.num_decoder_points,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        matcher_num_threads=args.matcher_num_threads,
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
