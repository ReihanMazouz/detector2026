from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from detector2026.core.models.yolov11_ablation import YOLOv11NoNeckScaleDeformableDecoder
from detector2026.core.utils.preprocess import preprocessing_num_channels


DEFAULT_DATA_DIR = "/data/RAWSIM/RMA/rf_dataset_for_real_validation"
DEFAULT_OUTPUT_ROOT = (
    "/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation/"
    "yolov11n_ablation"
)
DEFAULT_YOLOV11_BEST = (
    "/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation/"
    "yolov11n_specificres_cfg512/best.pt"
)
DEFAULT_RUN_NAME = "yolov11n_no_neck_scale_deformable_decoder_head_only"


def parse_res_hw(value: str) -> tuple[int, int]:
    if "," in value:
        left, right = value.split(",", 1)
    elif "x" in value.lower():
        left, right = value.lower().split("x", 1)
    else:
        raise argparse.ArgumentTypeError("Resolution must be formatted as H,W or HxW.")
    return int(left), int(right)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv11n no-neck scale-specialized deformable decoder ablation."
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--yolov11-best", default=DEFAULT_YOLOV11_BEST)

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--res-key", default="cfg512")
    parser.add_argument("--res-hw", type=parse_res_hw, default=(256, 256))
    parser.add_argument("--preprocessing", default="none")
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--reg-max", type=int, default=16)
    parser.add_argument("--width-mult", type=float, default=0.25)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--full-eval-every", type=int, default=5)
    parser.add_argument("--save-last-every", type=int, default=5)

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--query-counts", type=int, nargs=3, default=(64, 32, 16))
    parser.add_argument("--num-decoder-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-decoder-points", type=int, default=16)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--matcher-num-threads", type=int, default=8)

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def output_exists(output_dir: Path) -> bool:
    return (
        (output_dir / "best.pt").exists()
        or (output_dir / "last.pt").exists()
        or (output_dir / "train_log.csv").exists()
    )


def main() -> None:
    args = parse_args()
    input_channels = preprocessing_num_channels(args.preprocessing)
    output_dir = Path(args.output_root) / args.run_name
    weights_path = Path(args.yolov11_best)

    print("YOLOv11n no-neck scale deformable decoder ablation")
    print(f"  data_dir = {args.data_dir}")
    print(f"  output_dir = {output_dir}")
    print(f"  yolov11_best = {weights_path}")
    print(f"  device = {args.device}")
    print(f"  res_key = {args.res_key}")
    print(f"  res_hw = {args.res_hw}")
    print(f"  preprocessing = {args.preprocessing}")
    print(f"  input_channels = {input_channels}")
    print(f"  query_counts P3/P4/P5 = {tuple(args.query_counts)}")
    print(f"  decoder_layers = {args.num_decoder_layers}")
    print(f"  decoder_points = {args.num_decoder_points}")
    print("  training = decoder heads only; YOLOv11 no-neck backbone frozen")

    if args.dry_run:
        return
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    if output_exists(output_dir) and not args.overwrite:
        print(f"[SKIP] output already exists at {output_dir}; use --overwrite to rerun.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLOv11NoNeckScaleDeformableDecoder(
        output_dir=str(output_dir),
        num_classes=args.num_classes,
        reg_max=args.reg_max,
        device=args.device,
        input_canals=input_channels,
        width_mult=args.width_mult,
        input_hw=tuple(args.res_hw),
        hidden_dim=args.hidden_dim,
        query_counts=tuple(args.query_counts),
        num_decoder_layers=args.num_decoder_layers,
        num_heads=args.num_heads,
        num_decoder_points=args.num_decoder_points,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        matcher_num_threads=args.matcher_num_threads,
        freeze_backbone=True,
    )
    try:
        missing, unexpected = model.load_yolov11_weights(str(weights_path), device=args.device, eval_mode=False)
        print(f"[LOAD] compatible YOLOv11 weights loaded; missing={len(missing)} unexpected={len(unexpected)}")
        model.set_decoder_only_training()
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
            prefetch_factor=args.prefetch_factor,
            save_last_every=args.save_last_every,
            full_eval_every=args.full_eval_every,
            monitor="val_loss",
            run_full_eval=True,
        )
    finally:
        del model
        cleanup()


if __name__ == "__main__":
    main()
