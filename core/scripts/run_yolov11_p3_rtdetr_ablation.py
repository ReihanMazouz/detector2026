from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from detector2026.core.models.yolov11_ablation import YOLOv11P3Direct, YOLOv11P3RTDETR  # noqa: E402
from detector2026.core.utils.preprocess import preprocessing_num_channels  # noqa: E402


DEFAULT_DATA_DIR = "/data/RAWSIM/RMA/rf_dataset_for_real_validation"
DEFAULT_OUTPUT_ROOT = "/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation/yolov11n_p3_rtdetr_ablation"


def parse_res_hw(value: str) -> tuple[int, int]:
    if "," in value:
        left, right = value.split(",", 1)
    elif "x" in value.lower():
        left, right = value.lower().split("x", 1)
    else:
        raise argparse.ArgumentTypeError("Resolution must be formatted as H,W or HxW.")
    return int(left), int(right)


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv11n P3-only vs P3 RT-DETR head ablation.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--res-key", default="cfg512")
    parser.add_argument("--res-hw", type=parse_res_hw, default=(256, 256))
    parser.add_argument("--preprocessing", default="none")
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--reg-max", type=int, default=16)
    parser.add_argument("--width-mult", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--rtdetr-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rtdetr-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rtdetr-lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--rtdetr-patience", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--full-eval-every", type=int, default=5)
    parser.add_argument("--save-last-every", type=int, default=5)
    parser.add_argument("--tal-topk", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--num-decoder-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-decoder-points", type=int, default=16)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--matcher-num-threads", type=int, default=8)
    parser.add_argument("--skip-exp1", action="store_true")
    parser.add_argument("--skip-exp21", action="store_true")
    parser.add_argument("--skip-exp22", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def output_exists(output_dir: Path) -> bool:
    return (output_dir / "best.pt").exists() or (output_dir / "last.pt").exists() or (output_dir / "train_log.csv").exists()


def should_skip(name: str, output_dir: Path, overwrite: bool) -> bool:
    if output_exists(output_dir) and not overwrite:
        print(f"[SKIP] {name}: output already exists at {output_dir}")
        return True
    return False


def fit_kwargs(args, *, epochs, batch_size, lr, patience, include_prefetch: bool):
    kwargs = dict(
        data_dir=args.data_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        dataset="specificres",
        preprocessing=args.preprocessing,
        select_res={"res_hw": args.res_hw, "res_key": args.res_key},
        num_workers=args.num_workers,
        full_eval_every=args.full_eval_every,
        save_last_every=args.save_last_every,
        monitor="val_loss",
        run_full_eval=True,
    )
    if include_prefetch:
        kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs


def build_p3_direct(args, output_dir: Path, input_channels: int):
    return YOLOv11P3Direct(
        output_dir=str(output_dir),
        num_classes=args.num_classes,
        reg_max=args.reg_max,
        device=args.device,
        input_canals=input_channels,
        width_mult=args.width_mult,
        input_hw=args.res_hw,
        tal_topk=args.tal_topk,
    )


def build_p3_rtdetr(args, output_dir: Path, input_channels: int, *, freeze_backbone: bool):
    return YOLOv11P3RTDETR(
        output_dir=str(output_dir),
        num_classes=args.num_classes,
        reg_max=args.reg_max,
        device=args.device,
        input_canals=input_channels,
        width_mult=args.width_mult,
        input_hw=args.res_hw,
        hidden_dim=args.hidden_dim,
        num_queries=args.num_queries,
        num_decoder_layers=args.num_decoder_layers,
        num_heads=args.num_heads,
        num_decoder_points=args.num_decoder_points,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        matcher_num_threads=args.matcher_num_threads,
        freeze_backbone=freeze_backbone,
    )


def run_exp1(args, input_channels: int) -> Path:
    name = "exp1_yolov11n_p3_direct_tal_topk10"
    output_dir = Path(args.output_root) / name
    if args.skip_exp1 or should_skip(name, output_dir, args.overwrite):
        return output_dir / "best.pt"
    print(f"\n[RUN] {name}")
    model = build_p3_direct(args, output_dir, input_channels)
    try:
        model.fit(**fit_kwargs(
            args,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            include_prefetch=False,
        ))
    finally:
        del model
        cleanup()
    return output_dir / "best.pt"


def run_rtdetr(args, input_channels: int, *, exp1_best: Path, freeze_backbone: bool):
    suffix = "frozen_backbone" if freeze_backbone else "full_train"
    name = f"exp2_{'1' if freeze_backbone else '2'}_yolov11n_p3_rtdetr_{suffix}"
    output_dir = Path(args.output_root) / name
    if freeze_backbone and args.skip_exp21:
        return
    if not freeze_backbone and args.skip_exp22:
        return
    if should_skip(name, output_dir, args.overwrite):
        return
    if not exp1_best.is_file():
        raise FileNotFoundError(f"Missing experiment 1 checkpoint: {exp1_best}")

    print(f"\n[RUN] {name}")
    print(f"      init from {exp1_best}")
    model = build_p3_rtdetr(args, output_dir, input_channels, freeze_backbone=freeze_backbone)
    try:
        missing, unexpected = model.load_p3_yolo_weights(str(exp1_best), device=args.device, eval_mode=False)
        print(f"      loaded compatible weights; missing={len(missing)} unexpected={len(unexpected)}")
        if freeze_backbone:
            model.set_head_only_training()
        else:
            model.set_full_training()
        model.fit(**fit_kwargs(
            args,
            epochs=args.rtdetr_epochs,
            batch_size=args.rtdetr_batch_size,
            lr=args.rtdetr_lr,
            patience=args.rtdetr_patience,
            include_prefetch=True,
        ))
    finally:
        del model
        cleanup()


def main():
    args = parse_args()
    input_channels = preprocessing_num_channels(args.preprocessing)
    exp1_best = Path(args.output_root) / "exp1_yolov11n_p3_direct_tal_topk10" / "best.pt"

    print("YOLOv11n P3 RT-DETR ablation")
    print(f"  data_dir = {args.data_dir}")
    print(f"  output_root = {args.output_root}")
    print(f"  device = {args.device}")
    print(f"  res = {args.res_key} {args.res_hw}")
    print(f"  input_channels = {input_channels}")
    print("  experiments:")
    print("    1. P3-only YOLOv11n + Detect + TAL topk=10")
    print("    2.1 P3-only RT-DETR head, initialized from exp1 best.pt, frozen backbone")
    print("    2.2 P3-only RT-DETR head, initialized from exp1 best.pt, full training")

    if args.dry_run:
        return

    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    if not args.skip_exp1:
        exp1_best = run_exp1(args, input_channels)
    run_rtdetr(args, input_channels, exp1_best=exp1_best, freeze_backbone=True)
    run_rtdetr(args, input_channels, exp1_best=exp1_best, freeze_backbone=False)


if __name__ == "__main__":
    main()
