from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from detector2026.core.models import DETR
from detector2026.core.scripts.train_benchmark_suite import find_input_resolutions
from detector2026.core.utils.preprocess import preprocessing_num_channels


DEFAULT_DATA_DIR = "/data/RAWSIM/RMA/rf_dataset_for_real_validation"
DEFAULT_OUTPUT_DIR_PARENT = "/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation"
DEFAULT_RES_KEYS = ["cfg512", "cfg256", "cfg128", "cfg1024", "cfg2048"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate DETR on the benchmark dataset.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir-parent", default=DEFAULT_OUTPUT_DIR_PARENT)
    parser.add_argument("--output-dir-name", default="detr_n_specificres_cfg512")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--res-key", default="cfg512", choices=DEFAULT_RES_KEYS)
    parser.add_argument("--preprocessing", default="none")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--full-eval-every", type=int, default=5)
    parser.add_argument("--save-last-every", type=int, default=5)
    parser.add_argument("--monitor", default="val_loss", choices=["val_loss", "map50", "map50_95"])
    parser.add_argument("--width-mult", type=float, default=0.50)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--aux-loss-weight", type=float, default=1.0)
    parser.add_argument("--eos-coef", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    input_resolutions = find_input_resolutions(args.data_dir)
    if len(input_resolutions) != len(DEFAULT_RES_KEYS):
        raise ValueError(f"Expected {len(DEFAULT_RES_KEYS)} resolutions, found {len(input_resolutions)}: {input_resolutions}")
    res_key_to_hw = dict(zip(DEFAULT_RES_KEYS, input_resolutions))
    input_hw = res_key_to_hw[args.res_key]
    input_channels = preprocessing_num_channels(args.preprocessing)
    output_dir = Path(args.output_dir_parent) / args.output_dir_name

    print(f"DETR output_dir = {output_dir}")
    print(f"Resolution = {args.res_key}: {input_hw}")
    print(f"Preprocessing = {args.preprocessing}")
    print(f"Input channels = {input_channels}")
    print(f"Device = {args.device if torch.cuda.is_available() or not args.device.startswith('cuda') else 'cpu'}")
    print(f"Full eval every = {args.full_eval_every}")
    print(f"Monitor = {args.monitor}")
    if args.dry_run:
        return

    model = DETR(
        output_dir=str(output_dir),
        num_classes=args.num_classes,
        device=args.device,
        input_channels=input_channels,
        width_mult=args.width_mult,
        hidden_dim=args.hidden_dim,
        num_queries=args.num_queries,
        num_encoder_layers=args.encoder_layers,
        num_decoder_layers=args.decoder_layers,
        nheads=args.nheads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        input_hw=input_hw,
        aux_loss_weight=args.aux_loss_weight,
        eos_coef=args.eos_coef,
    )
    model.fit(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        dataset="specificres",
        preprocessing=args.preprocessing,
        select_res={"res_hw": input_hw, "res_key": args.res_key},
        num_workers=args.num_workers,
        full_eval_every=args.full_eval_every,
        save_last_every=args.save_last_every,
        monitor=args.monitor,
    )


if __name__ == "__main__":
    main()
