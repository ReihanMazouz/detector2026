from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from detector2026.core.models.mr_yolo import MR_YOLO  # noqa: E402
from detector2026.core.scripts.train_benchmark_suite import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_DEVICE,
    DEFAULT_NUM_CLASSES,
    DEFAULT_PREPROCESSING,
    DEFAULT_REG_MAX,
    DEFAULT_RES_KEYS,
    MR_BACKBONE_MODE,
    MR_OUTFUSION_CHANNELS_MULT,
    MR_WIDTH_MULT,
    find_input_resolutions,
)
from detector2026.core.utils.preprocess import preprocessing_num_channels  # noqa: E402


DEFAULT_OUTPUT_ROOT = (
    "/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation/baselines"
)
DEFAULT_RUN_NAME = "v100_mr_yolovn_baseline"


def output_is_complete(output_dir: Path) -> bool:
    return (output_dir / "best.pt").exists() or (output_dir / "last.pt").exists()


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MR-YOLOvn fused multires baseline, intended for V100.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--preprocessing", default=DEFAULT_PREPROCESSING)
    parser.add_argument("--num-classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--reg-max", type=int, default=DEFAULT_REG_MAX)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--full-eval-every", type=int, default=5)
    parser.add_argument("--save-last-every", type=int, default=5)
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
    output_dir = Path(args.output_root) / args.run_name

    print("MR-YOLOvn baseline on V100")
    print(f"  output_dir = {output_dir}")
    print(f"  data_dir = {args.data_dir}")
    print(f"  device = {args.device}")
    print(f"  preprocessing = {args.preprocessing}")
    print(f"  input_channels = {input_channels}")
    print(f"  width_mult = {MR_WIDTH_MULT['n']}")
    print(f"  backbone_mode = {MR_BACKBONE_MODE}")
    print(f"  outfusion_channels_mult = {MR_OUTFUSION_CHANNELS_MULT}")
    print("  resolutions:")
    for res_key, res_hw in zip(DEFAULT_RES_KEYS, input_resolutions):
        print(f"    {res_key}: {res_hw}")

    if args.dry_run:
        return
    if output_is_complete(output_dir) and not args.overwrite:
        print(f"[SKIP] checkpoint deja present dans {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    model = MR_YOLO(
        input_resolutions=list(input_resolutions),
        output_dir=str(output_dir),
        num_classes=args.num_classes,
        reg_max=args.reg_max,
        device=args.device,
        in_ch=input_channels,
        width_mult=MR_WIDTH_MULT["n"],
        backbone_mode=MR_BACKBONE_MODE,
        outfusion_channels_mult=MR_OUTFUSION_CHANNELS_MULT,
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
            monitor="val_loss",
            run_full_eval=True,
        )
    finally:
        del model
        cleanup()


if __name__ == "__main__":
    main()
