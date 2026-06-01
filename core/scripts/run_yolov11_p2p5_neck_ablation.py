from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from detector2026.core.models.yolov11_ablation import YOLOv11P2P5Neck  # noqa: E402
from detector2026.core.utils.preprocess import preprocessing_num_channels  # noqa: E402


DEFAULT_DATA_DIR = "/data/RAWSIM/RMA/rf_dataset_for_real_validation"
DEFAULT_OUTPUT_ROOT = (
    "/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation/yolov11_p2p5_neck_ablation"
)


def parse_res_hw(value: str) -> tuple[int, int]:
    if "," in value:
        left, right = value.split(",", 1)
    elif "x" in value.lower():
        left, right = value.lower().split("x", 1)
    else:
        raise argparse.ArgumentTypeError("Resolution must be formatted as H,W or HxW.")
    return int(left), int(right)


def output_is_complete(output_dir: Path) -> bool:
    return (output_dir / "best.pt").exists() or (output_dir / "last.pt").exists()


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv11 with a larger P2-P5 neck and four Detect heads."
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="yolov11n_p2p5_neck")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--res-key", default="cfg512")
    parser.add_argument("--res-hw", type=parse_res_hw, default=(256, 256))
    parser.add_argument("--preprocessing", default="none")
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--reg-max", type=int, default=16)
    parser.add_argument("--width-mult", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--full-eval-every", type=int, default=5)
    parser.add_argument("--save-last-every", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_channels = preprocessing_num_channels(args.preprocessing)
    output_dir = Path(args.output_root) / args.run_name

    print("YOLOv11 P2-P5 neck ablation")
    print(f"  output_dir = {output_dir}")
    print(f"  data_dir = {args.data_dir}")
    print(f"  device = {args.device}")
    print(f"  res = {args.res_key} {args.res_hw}")
    print(f"  preprocessing = {args.preprocessing}")
    print(f"  input_channels = {input_channels}")
    print("  detect levels = P2, P3, P4, P5")
    print("  strides = [4, 8, 16, 32]")

    if args.dry_run:
        return
    if output_is_complete(output_dir) and not args.overwrite:
        print(f"[SKIP] checkpoint deja present dans {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLOv11P2P5Neck(
        output_dir=str(output_dir),
        num_classes=args.num_classes,
        reg_max=args.reg_max,
        device=args.device,
        input_canals=input_channels,
        width_mult=args.width_mult,
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
