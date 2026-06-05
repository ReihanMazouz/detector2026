from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from detector2026.core.models.yolov11_ablation import YOLOv11RTDETRHead
from detector2026.core.scripts.train_benchmark_suite import (
    DEFAULT_DATA_DIR,
    DEFAULT_DEVICE,
    DEFAULT_NUM_CLASSES,
    DEFAULT_OUTPUT_DIR_PARENT,
    DEFAULT_PREPROCESSING,
    DEFAULT_REG_MAX,
    DEFAULT_RES_KEYS,
    YOLO11_WIDTH_MULT,
    find_input_resolutions,
)
from detector2026.core.utils.dataset import YOLODatasetSpecificRes, load_class_index_to_name
from detector2026.core.utils.detr_loss import targets_from_yolo_tensor
from detector2026.core.utils.evaluate import EvalConfig, EvalRunner, TrainingPlots
from detector2026.core.utils.preprocess import preprocessing_num_channels
from detector2026.core.utils.rtdetr_loss import RTDETRROCCalibrationLoss


DEFAULT_DEVICE = "cuda:0"
DEFAULT_RUN_NAME = "yolov11n_rtdetr_head_roc_calibration"
DEFAULT_WEIGHTS = (
    "/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation/yolov11n_ablation/"
    "yolov11n_best_ft_rtdetr_head_one2one_deformable/best.pt"
)


def parse_hw(value: str) -> tuple[int, int]:
    if "," in value:
        left, right = value.split(",", 1)
    elif "x" in value.lower():
        left, right = value.lower().split("x", 1)
    else:
        raise argparse.ArgumentTypeError("Expected H,W or HxW.")
    return int(left), int(right)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a trained YOLOv11 + RT-DETR one-to-one head and calibrate only the "
            "RT-DETR classification heads with a false-alarm-target ROC loss."
        )
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-dir-parent", default=f"{DEFAULT_OUTPUT_DIR_PARENT}/yolov11n_ablation")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--num-classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--reg-max", type=int, default=DEFAULT_REG_MAX)
    parser.add_argument("--scale", choices=sorted(YOLO11_WIDTH_MULT.keys()), default="n")
    parser.add_argument("--res-key", default=DEFAULT_RES_KEYS[0])
    parser.add_argument("--res-hw", type=parse_hw, default=None)
    parser.add_argument("--preprocessing", default=DEFAULT_PREPROCESSING)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--save-last-every", type=int, default=1)
    parser.add_argument("--full-eval-every", type=int, default=5)
    parser.add_argument("--no-amp", action="store_true")

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--num-decoder-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-decoder-points", type=int, default=8)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--matcher-num-threads", type=int, default=8)

    parser.add_argument("--pfa-target", type=float, default=0.01)
    parser.add_argument("--roc-margin", type=float, default=0.02)
    parser.add_argument("--roc-alpha", type=float, default=20.0)
    parser.add_argument("--roc-weight", type=float, default=1.0)
    parser.add_argument("--no-aux-cls-loss", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def output_is_complete(output_dir: Path) -> bool:
    return (output_dir / "best.pt").is_file() or (output_dir / "last.pt").is_file()


def build_loaders(args: argparse.Namespace, input_hw: tuple[int, int]):
    dataset_kwargs = dict(
        res_hw=input_hw,
        res_key=args.res_key,
        preprocessing=args.preprocessing,
    )
    train_dataset = YOLODatasetSpecificRes(
        data_dir=os.path.join(args.data_dir, "train/data"),
        labels_dir=os.path.join(args.data_dir, "train/labels_detect"),
        **dataset_kwargs,
    )
    val_dataset = YOLODatasetSpecificRes(
        data_dir=os.path.join(args.data_dir, "val/data"),
        labels_dir=os.path.join(args.data_dir, "val/labels_detect"),
        **dataset_kwargs,
    )
    loader_kwargs = dict(
        batch_size=args.batch_size,
        pin_memory=torch.cuda.is_available() and str(args.device).startswith("cuda"),
        collate_fn=YOLODatasetSpecificRes.collate_fn,
        num_workers=max(0, int(args.num_workers)),
        persistent_workers=int(args.num_workers) > 0,
    )
    if int(args.num_workers) > 0:
        loader_kwargs["prefetch_factor"] = max(2, int(args.prefetch_factor))
    return (
        DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        DataLoader(val_dataset, shuffle=False, **loader_kwargs),
    )


def build_model(args: argparse.Namespace, output_dir: Path, input_hw: tuple[int, int], input_channels: int):
    model = YOLOv11RTDETRHead(
        output_dir=str(output_dir),
        num_classes=args.num_classes,
        reg_max=args.reg_max,
        device=args.device,
        input_canals=input_channels,
        width_mult=YOLO11_WIDTH_MULT[args.scale],
        input_hw=input_hw,
        hidden_dim=args.hidden_dim,
        num_queries=args.num_queries,
        num_decoder_layers=args.num_decoder_layers,
        num_heads=args.num_heads,
        num_decoder_points=args.num_decoder_points,
        use_deformable_attention=True,
        dim_feedforward=args.dim_feedforward,
        dropout=0.0,
        matcher_num_threads=args.matcher_num_threads,
    )
    model.load_weights(str(args.weights), device=args.device, eval_mode=True)
    model.use_one2one_head()
    model.input_hw = input_hw
    model._last_image_hw = input_hw
    return model


def freeze_for_score_calibration(model: YOLOv11RTDETRHead) -> int:
    for param in model.parameters():
        param.requires_grad = False

    trainable = 0
    score_modules = [model.detect_one2one.enc_score_head, model.detect_one2one.dec_score_head]
    for module in score_modules:
        for param in module.parameters():
            param.requires_grad = True
            trainable += param.numel()
    return trainable


def set_frozen_modules_eval(model: YOLOv11RTDETRHead) -> None:
    model._set_frozen_parts_eval()
    for name, module in model.detect_one2one.named_children():
        if name not in {"enc_score_head", "dec_score_head"}:
            module.eval()
    model.detect_one2one.enc_score_head.train()
    model.detect_one2one.dec_score_head.train()


def run_epoch(model, criterion, loader, optimizer, scaler, train: bool, desc: str):
    if train:
        model.train()
        set_frozen_modules_eval(model)
    else:
        model.train()
        set_frozen_modules_eval(model)

    total_loss = 0.0
    parts_sum = {
        "loss_cls": 0.0,
        "loss_roc": 0.0,
        "roc_tau": 0.0,
        "roc_num_pos": 0.0,
        "roc_num_neg": 0.0,
    }
    amp_enabled = scaler.is_enabled() if scaler is not None else False
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for imgs, targets, _ in tqdm(loader, desc=desc, unit="batch"):
            imgs = imgs.to(model.device, non_blocking=torch.cuda.is_available())
            targets = targets.to(model.device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(imgs)
                target_list = targets_from_yolo_tensor(targets, outputs["pred_logits"].shape[0], outputs["pred_logits"].device)
                loss, parts = criterion(outputs, target_list)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += float(loss.detach().item())
            for key in parts_sum:
                if key in parts:
                    parts_sum[key] += float(parts[key])

    denom = max(1, len(loader))
    return total_loss / denom, {key: value / denom for key, value in parts_sum.items()}


def main() -> int:
    args = parse_args()
    input_resolutions = find_input_resolutions(args.data_dir, split="train")
    input_hw = tuple(args.res_hw) if args.res_hw is not None else tuple(input_resolutions[0])
    input_channels = preprocessing_num_channels(args.preprocessing)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_dir_parent) / args.run_name
    weights_path = Path(args.weights)

    print("YOLOv11 RT-DETR ROC calibration")
    print(f"  weights = {weights_path}")
    print(f"  output_dir = {output_dir}")
    print(f"  input_hw = {input_hw} ({args.res_key})")
    print(f"  pfa_target = {args.pfa_target}")
    print(f"  roc_margin = {args.roc_margin}")
    print(f"  roc_alpha = {args.roc_alpha}")
    print(f"  roc_weight = {args.roc_weight}")
    if args.dry_run:
        return 0
    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if output_is_complete(output_dir) and not args.overwrite:
        print(f"[SKIP] checkpoint already present in {output_dir}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = build_loaders(args, input_hw)
    model = build_model(args, output_dir, input_hw, input_channels)
    criterion = RTDETRROCCalibrationLoss(
        num_classes=args.num_classes,
        pfa_target=args.pfa_target,
        margin=args.roc_margin,
        roc_alpha=args.roc_alpha,
        roc_weight=args.roc_weight,
        aux_cls_loss=not args.no_aux_cls_loss,
        matcher_num_threads=args.matcher_num_threads,
    )
    criterion.to(model.device)
    trainable = freeze_for_score_calibration(model)
    print(f"  trainable score-head params = {trainable:,}")

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp and str(args.device).startswith("cuda"))
    eval_runner = EvalRunner(
        output_dir=str(output_dir),
        cfg=EvalConfig(iou_thresh=0.5, fa_target=args.pfa_target, img_size=input_hw),
        class_index_to_name=load_class_index_to_name(args.data_dir),
    )
    extra_headers = eval_runner.extra_headers()
    log_path = output_dir / "train_log.csv"
    with log_path.open("w", newline="") as handle:
        csv.writer(handle).writerow(
            [
                "epoch",
                "train_loss",
                "val_loss",
                "loss_cls_val",
                "loss_roc_val",
                "roc_tau_val",
                "roc_num_pos_val",
                "roc_num_neg_val",
                *extra_headers,
            ]
        )

    best_val = float("inf")
    bad_epochs = 0
    try:
        for epoch in range(1, int(args.epochs) + 1):
            start = time.perf_counter()
            train_loss, train_parts = run_epoch(
                model,
                criterion,
                train_loader,
                optimizer,
                scaler,
                train=True,
                desc=f"Epoch {epoch} ROC train",
            )
            val_loss, val_parts = run_epoch(
                model,
                criterion,
                val_loader,
                None,
                scaler,
                train=False,
                desc=f"Epoch {epoch} ROC val",
            )
            should_eval = (epoch % max(1, int(args.full_eval_every)) == 0) or epoch == int(args.epochs)
            if should_eval:
                model.eval()
                eval_result = eval_runner.run(epoch=epoch, model=model, val_loader=val_loader)
                extra_values = eval_result["extra_values"]
            else:
                extra_values = [None, None, *([float("nan")] * 7), None]

            with log_path.open("a", newline="") as handle:
                csv.writer(handle).writerow(
                    [
                        epoch,
                        train_loss,
                        val_loss,
                        val_parts["loss_cls"],
                        val_parts["loss_roc"],
                        val_parts["roc_tau"],
                        val_parts["roc_num_pos"],
                        val_parts["roc_num_neg"],
                        *extra_values,
                    ]
                )

            if epoch % max(1, int(args.save_last_every)) == 0 or epoch == int(args.epochs):
                torch.save(model.state_dict(), output_dir / "last.pt")
            if val_loss < best_val:
                best_val = val_loss
                bad_epochs = 0
                torch.save(model.state_dict(), output_dir / "best.pt")
            else:
                bad_epochs += 1

            print(
                f"ROC epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f} "
                f"cls={val_parts['loss_cls']:.4f} roc={val_parts['loss_roc']:.4f} "
                f"tau={val_parts['roc_tau']:.4f} time={time.perf_counter() - start:.1f}s"
            )
            if should_eval:
                TrainingPlots.plot_losses(str(log_path), save_path=str(output_dir / "loss_curves.png"))
                TrainingPlots.plot_maps(str(log_path), save_path=str(output_dir / "map_curves.png"))
                TrainingPlots.plot_avg_recalls(str(log_path), save_path=str(output_dir / "avg_recall_curves.png"))
                TrainingPlots.plot_size_recalls(str(log_path), save_path=str(output_dir / "recall_size_curves.png"))
                TrainingPlots.plot_box_iou(str(log_path), save_path=str(output_dir / "box_iou_curves.png"))
            if bad_epochs >= int(args.patience):
                print(f"Early stopping on val_loss after {bad_epochs} epochs without improvement.")
                break
    finally:
        del model
        cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
