from __future__ import annotations

import argparse
import gc
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from detector2026.core.models.yolov11_ablation import YOLOv11P3Direct, YOLOv11P3RTDETR  # noqa: E402
from detector2026.core.utils.preprocess import preprocessing_num_channels  # noqa: E402


DEFAULT_DATA_DIR = "/data/RAWSIM/RMA/rf_dataset_for_real_validation"
DEFAULT_OUTPUT_ROOT = (
    "/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation/"
    "yolov11n_p3_rtdetr_head_ablation"
)
DEFAULT_EXP1_CHECKPOINT = (
    "/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation/"
    "yolov11n_p3_rtdetr_ablation/exp1_yolov11n_p3_direct_tal_topk10/best.pt"
)

# (num_decoder_layers, num_decoder_points, num_heads)
HEAD_CONFIGS: list[tuple[int, int, int]] = [
    # 8-head sweep (original)
    (2, 4, 8),
    (3, 4, 8),
    (3, 8, 8),
    (3, 16, 8),
    (4, 4, 8),
    # 4-head sweep
    (2, 4, 4),
    (3, 4, 4),
    (3, 8, 4),
]


@dataclass(frozen=True)
class HeadConfig:
    num_decoder_layers: int
    num_decoder_points: int
    num_heads: int

    @property
    def tag(self) -> str:
        return f"{self.num_decoder_layers}layers_{self.num_decoder_points}pts_{self.num_heads}h"


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
        description=(
            "Ablation on the RT-DETR head capacity for P3-only detection: "
            "varies num_decoder_layers and num_decoder_points."
        )
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--exp1-checkpoint",
        default=DEFAULT_EXP1_CHECKPOINT,
        help="Path to a pre-trained P3-direct best.pt checkpoint used to initialise all RT-DETR runs.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--res-key", default="cfg512")
    parser.add_argument("--res-hw", type=parse_res_hw, default=(256, 256))
    parser.add_argument("--preprocessing", default="none")
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--reg-max", type=int, default=16)
    parser.add_argument("--width-mult", type=float, default=0.25)
    # Training schedule
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--full-eval-every", type=int, default=5)
    parser.add_argument("--save-last-every", type=int, default=5)
    # Fixed RT-DETR head parameters (not ablated)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--tal-topk", type=int, default=10)
    parser.add_argument("--matcher-num-threads", type=int, default=8)
    # Control
    parser.add_argument("--overwrite", action="store_true", help="Re-run experiments that already have a checkpoint.")
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


def should_skip(name: str, output_dir: Path, overwrite: bool) -> bool:
    if output_exists(output_dir) and not overwrite:
        print(f"[SKIP] {name}: output already exists at {output_dir}")
        return True
    return False


def fit_kwargs(args: argparse.Namespace) -> dict:
    return dict(
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
        full_eval_every=args.full_eval_every,
        save_last_every=args.save_last_every,
        monitor="val_loss",
        run_full_eval=True,
    )


def run_head_config(
    args: argparse.Namespace,
    cfg: HeadConfig,
    input_channels: int,
    exp1_checkpoint: Path,
) -> None:
    name = f"p3_rtdetr_{cfg.tag}"
    output_dir = Path(args.output_root) / name
    if should_skip(name, output_dir, args.overwrite):
        return
    if not exp1_checkpoint.is_file():
        raise FileNotFoundError(
            f"P3-direct checkpoint not found: {exp1_checkpoint}\n"
            "Run the P3-direct experiment first or pass --exp1-checkpoint."
        )

    print(f"\n[RUN] {name}")
    print(f"      layers={cfg.num_decoder_layers}  points={cfg.num_decoder_points}  heads={cfg.num_heads}")
    print(f"      init from {exp1_checkpoint}")

    model = YOLOv11P3RTDETR(
        output_dir=str(output_dir),
        num_classes=args.num_classes,
        reg_max=args.reg_max,
        device=args.device,
        input_canals=input_channels,
        width_mult=args.width_mult,
        input_hw=args.res_hw,
        tal_topk=args.tal_topk,
        hidden_dim=args.hidden_dim,
        num_queries=args.num_queries,
        num_decoder_layers=cfg.num_decoder_layers,
        num_heads=cfg.num_heads,
        num_decoder_points=cfg.num_decoder_points,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        matcher_num_threads=args.matcher_num_threads,
        freeze_backbone=False,
    )
    try:
        missing, unexpected = model.load_p3_yolo_weights(
            str(exp1_checkpoint), device=args.device, eval_mode=False
        )
        print(f"      loaded compatible weights; missing={len(missing)} unexpected={len(unexpected)}")
        model.set_full_training()
        model.fit(**fit_kwargs(args))
    finally:
        del model
        cleanup()


def main() -> None:
    args = parse_args()
    input_channels = preprocessing_num_channels(args.preprocessing)
    exp1_checkpoint = Path(args.exp1_checkpoint)
    configs = [HeadConfig(layers, pts, heads) for layers, pts, heads in HEAD_CONFIGS]

    print("YOLOv11n P3 RT-DETR head capacity ablation")
    print(f"  data_dir        = {args.data_dir}")
    print(f"  output_root     = {args.output_root}")
    print(f"  device          = {args.device}")
    print(f"  res             = {args.res_key} {args.res_hw}")
    print(f"  input_channels  = {input_channels}")
    print(f"  exp1_checkpoint = {exp1_checkpoint}  exists={exp1_checkpoint.is_file()}")
    print(f"  epochs={args.epochs}  patience={args.patience}  lr={args.lr}")
    print(f"  hidden_dim={args.hidden_dim}  num_queries={args.num_queries}  num_heads={args.num_heads}")
    print(f"  dim_feedforward={args.dim_feedforward}  dropout={args.dropout}")
    print()
    print("  experiments:")
    for i, cfg in enumerate(configs, 1):
        print(f"    {i}. {cfg.tag}  (layers={cfg.num_decoder_layers} pts={cfg.num_decoder_points} heads={cfg.num_heads})")

    if args.dry_run:
        return

    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    for cfg in configs:
        run_head_config(args, cfg, input_channels, exp1_checkpoint)

    print("\n[DONE] all head ablation experiments finished.")
    print(f"[DONE] outputs: {args.output_root}")


if __name__ == "__main__":
    main()
