from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from detector2026.core.models.yolov11 import YOLOv11  # noqa: E402
from detector2026.core.scripts.train_benchmark_suite import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_DEVICE,
    DEFAULT_NUM_CLASSES,
    DEFAULT_PREPROCESSING,
    DEFAULT_REG_MAX,
    YOLO11_WIDTH_MULT,
)
from detector2026.core.utils.preprocess import preprocessing_num_channels  # noqa: E402


DEFAULT_OUTPUT_ROOT = (
    "/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation/baselines"
)
DEFAULT_RUN_NAME = "h100_yolov11n_baseline_cfg512"


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


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv11n unires baseline, intended for H100.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--res-key", default="cfg512")
    parser.add_argument("--res-hw", type=parse_hw, default=(256, 256))
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
    input_channels = preprocessing_num_channels(args.preprocessing)
    output_dir = Path(args.output_root) / args.run_name

    print("YOLOv11n baseline on H100")
    print(f"  output_dir = {output_dir}")
    print(f"  data_dir = {args.data_dir}")
    print(f"  device = {args.device}")
    print(f"  res = {args.res_key} {args.res_hw}")
    print(f"  preprocessing = {args.preprocessing}")
    print(f"  input_channels = {input_channels}")
    print(f"  width_mult = {YOLO11_WIDTH_MULT['n']}")

    if args.dry_run:
        return
    if output_is_complete(output_dir) and not args.overwrite:
        print(f"[SKIP] checkpoint deja present dans {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLOv11(
        output_dir=str(output_dir),
        num_classes=args.num_classes,
        reg_max=args.reg_max,
        device=args.device,
        input_canals=input_channels,
        width_mult=YOLO11_WIDTH_MULT["n"],
        input_hw=args.res_hw,
    )
    try:
        model.fit(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            dataset="specificres",
            preprocessing=args.preprocessing,
            select_res={"res_hw": args.res_hw, "res_key": args.res_key},
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
