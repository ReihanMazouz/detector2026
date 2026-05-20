from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from detector2026.core.models.yolov11 import YOLOv11
from detector2026.core.scripts.train_benchmark_suite import DEFAULT_RES_KEYS, YOLO11_WIDTH_MULT, find_input_resolutions
from detector2026.core.utils.dataset import YOLODatasetSpecificRes, load_class_index_to_name
from detector2026.core.utils.divers import xywh2xyxy
from detector2026.core.utils.evaluate import EvalConfig, EvalRunner
from detector2026.core.utils.metrics import box_iou
from detector2026.core.utils.preprocess import preprocessing_num_channels


DEFAULT_DATA_DIR = "/data/RAWSIM/RMA/rf_dataset_for_real_validation"
DEFAULT_OUTPUT_DIR = "/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation/one2one_head/visualizations"
DEFAULT_ONE2MANY_CHECKPOINT = "/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation/yolov11n_specificres_cfg512/best.pt"
DEFAULT_ONE2ONE_TAL_CHECKPOINT = "/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation/one2one_head/yolov11n_specificres_cfg512_one2one_tal/best.pt"
DEFAULT_ONE2ONE_HUNGARIAN_CHECKPOINT = "/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation/one2one_head/yolov11n_specificres_cfg512_one2one_hungarian/best.pt"
DEFAULT_DEVICE = "cuda:1"


def parse_chin_levels(value: str) -> tuple[str, ...]:
    levels = tuple(level.strip().lower() for level in value.split(",") if level.strip())
    if not levels:
        raise argparse.ArgumentTypeError("At least one transformer chin level is required.")
    invalid = [level for level in levels if level not in {"p3", "p4", "p5"}]
    if invalid:
        raise argparse.ArgumentTypeError(f"Invalid transformer chin levels: {invalid}. Use p3,p4,p5.")
    return levels


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate YOLOv11 one2many+NMS, one2one TAL, and one2one Hungarian on the full dataset, "
            "then visualize the same samples for all three models."
        )
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--one2many-checkpoint", default=DEFAULT_ONE2MANY_CHECKPOINT, help="Baseline YOLOv11 checkpoint, usually best.pt.")
    parser.add_argument("--one2one-tal-checkpoint", default=DEFAULT_ONE2ONE_TAL_CHECKPOINT, help="Fine-tuned one2one TAL checkpoint.")
    parser.add_argument("--one2one-hungarian-checkpoint", default=DEFAULT_ONE2ONE_HUNGARIAN_CHECKPOINT, help="Fine-tuned one2one Hungarian checkpoint.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--reg-max", type=int, default=16)
    parser.add_argument("--scale", choices=sorted(YOLO11_WIDTH_MULT.keys()), default="n")
    parser.add_argument("--res-key", default="cfg512", choices=DEFAULT_RES_KEYS)
    parser.add_argument("--preprocessing", default="none")
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--eval-iou-thresh", type=float, default=0.5)
    parser.add_argument("--eval-fa-target", type=float, default=0.01)
    parser.add_argument(
        "--use-transformer-chin",
        action="store_true",
        help="Build one2one models with the transformer chin before loading their checkpoints.",
    )
    parser.add_argument("--chin-levels", type=parse_chin_levels, default=("p4", "p5"))
    parser.add_argument("--chin-d-model", type=int, default=128)
    parser.add_argument("--chin-num-heads", type=int, default=4)
    parser.add_argument("--chin-num-layers", type=int, default=1)
    parser.add_argument("--chin-ffn-ratio", type=float, default=2.0)
    parser.add_argument("--chin-dropout", type=float, default=0.0)
    parser.add_argument("--chin-residual-scale", type=float, default=0.0)
    return parser.parse_args()


def _class_name(mapping: dict[int, str], label: int) -> str:
    return mapping.get(int(label), str(int(label)))


def _gt_from_targets(targets: torch.Tensor, image_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows = targets[targets[:, 0].long() == image_index]
    boxes = xywh2xyxy(rows[:, 2:6]).clamp(0.0, 1.0)
    labels = rows[:, 1].long()
    return boxes, labels


def _detections_to_normalized(
    detection: torch.Tensor,
    *,
    input_hw: tuple[int, int],
    score_threshold: float,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if detection is None or detection.numel() == 0:
        return torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long)

    det = detection.detach().cpu()
    keep = det[:, 4] >= score_threshold
    det = det[keep]
    if det.shape[0] > top_k:
        det = det[det[:, 4].argsort(descending=True)[:top_k]]
    height, width = float(input_hw[0]), float(input_hw[1])
    boxes = torch.stack(
        [
            det[:, 0] / width,
            det[:, 1] / height,
            det[:, 2] / width,
            det[:, 3] / height,
        ],
        dim=1,
    ).clamp(0.0, 1.0)
    return boxes, det[:, 4], det[:, 5].long()


def _draw_panel(
    ax,
    *,
    image: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    class_names: dict[int, str],
    title: str,
    cmap: str,
):
    image = image.detach().cpu()
    height, width = image.shape[-2:]
    image_2d = image[0] if image.ndim == 3 else image.squeeze(0)
    ax.imshow(image_2d.numpy(), aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.axis("off")

    for gt_index, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
        x1, y1, x2, y2 = box.tolist()
        ax.add_patch(
            patches.Rectangle(
                (x1 * width, y1 * height),
                (x2 - x1) * width,
                (y2 - y1) * height,
                linewidth=2,
                edgecolor="lime",
                facecolor="none",
            )
        )
        ax.text(
            x1 * width,
            y1 * height,
            f"GT {gt_index}: {_class_name(class_names, int(label))}",
            color="black",
            fontsize=7,
            bbox=dict(facecolor="lime", alpha=0.75, pad=1),
        )

    pairwise_iou = box_iou(pred_boxes, gt_boxes) if pred_boxes.numel() and gt_boxes.numel() else torch.zeros((len(pred_boxes), len(gt_boxes)))
    best_iou = pairwise_iou.max(dim=1).values if pairwise_iou.numel() else torch.zeros((len(pred_boxes),))
    best_gt = pairwise_iou.argmax(dim=1) if pairwise_iou.numel() else torch.full((len(pred_boxes),), -1, dtype=torch.long)

    for pred_index, (box, score, label) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
        x1, y1, x2, y2 = box.tolist()
        color = "cyan" if best_iou[pred_index] >= 0.5 else "red"
        ax.add_patch(
            patches.Rectangle(
                (x1 * width, y1 * height),
                (x2 - x1) * width,
                (y2 - y1) * height,
                linewidth=1.2,
                edgecolor=color,
                facecolor="none",
                linestyle="--",
            )
        )
        gt_suffix = f", gt={int(best_gt[pred_index])}" if int(best_gt[pred_index]) >= 0 else ""
        ax.text(
            x1 * width,
            max(0.0, y1 * height - 4.0),
            f"P{pred_index}: {_class_name(class_names, int(label))} {float(score):.2f}, IoU={float(best_iou[pred_index]):.2f}{gt_suffix}",
            color="white",
            fontsize=6,
            bbox=dict(facecolor=color, alpha=0.75, pad=1),
        )


def _build_model(args, output_dir: Path, input_channels: int) -> YOLOv11:
    return YOLOv11(
        output_dir=str(output_dir),
        num_classes=args.num_classes,
        reg_max=args.reg_max,
        device=args.device,
        input_canals=input_channels,
        width_mult=YOLO11_WIDTH_MULT[args.scale],
        use_transformer_chin=args.use_transformer_chin,
        chin_levels=args.chin_levels,
        chin_d_model=args.chin_d_model,
        chin_num_heads=args.chin_num_heads,
        chin_num_layers=args.chin_num_layers,
        chin_ffn_ratio=args.chin_ffn_ratio,
        chin_dropout=args.chin_dropout,
        chin_residual_scale=args.chin_residual_scale,
    )


def _load_models(args, output_dir: Path, input_channels: int) -> dict[str, tuple[YOLOv11, bool]]:
    specs = {
        "one2many_nms": (args.one2many_checkpoint, "one2many", False),
        "one2one_tal": (args.one2one_tal_checkpoint, "one2one", True),
        "one2one_hungarian": (args.one2one_hungarian_checkpoint, "one2one", True),
    }
    models = {}
    for name, (checkpoint, head, without_nms) in specs.items():
        model = _build_model(args, output_dir / name, input_channels)
        model.load_weights(checkpoint, device=args.device, eval_mode=True)
        if head == "one2one":
            model.use_one2one_head()
        else:
            model.use_one2many_head()
        model.eval()
        models[name] = (model, without_nms)
    return models


def _build_loader(args, dataset, *, subset_indices=None):
    if subset_indices is not None:
        dataset = torch.utils.data.Subset(dataset, list(subset_indices))
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=YOLODatasetSpecificRes.collate_fn,
        pin_memory=torch.cuda.is_available() and str(args.device).startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )


def _summarize_eval_result(name: str, result: dict[str, Any]) -> dict[str, Any]:
    full_metrics = result["full_metrics"]
    values = result["extra_values"]
    operating_point = full_metrics.get("operating_point", {})
    model_info = full_metrics.get("model_info", {})
    return {
        "model": name,
        "map50": values[0],
        "map50_95": values[1],
        "avg_recall_low_snr": values[2],
        "avg_recall_medium_snr": values[3],
        "avg_recall_high_snr": values[4],
        "conf_thresh": full_metrics.get("conf_thresh"),
        "tp_at_conf_thresh": operating_point.get("tp_at_conf_thresh"),
        "fp_at_conf_thresh": operating_point.get("fp_at_conf_thresh"),
        "fn_total": operating_point.get("fn_total"),
        "tp_raw": operating_point.get("tp_raw"),
        "fp_raw": operating_point.get("fp_raw"),
        "params": model_info.get("params"),
        "flops": model_info.get("flops"),
        "metrics_json_path": values[5],
    }


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]):
    fieldnames = [
        "model",
        "map50",
        "map50_95",
        "avg_recall_low_snr",
        "avg_recall_medium_snr",
        "avg_recall_high_snr",
        "conf_thresh",
        "tp_at_conf_thresh",
        "fp_at_conf_thresh",
        "fn_total",
        "tp_raw",
        "fp_raw",
        "params",
        "flops",
        "metrics_json_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def _evaluate_models(args, output_dir: Path, models: dict[str, tuple[YOLOv11, bool]], eval_loader, class_names, input_hw):
    metrics_rows = []
    for name, (model, _) in models.items():
        print(f"\n[EVAL] {name}")
        runner = EvalRunner(
            output_dir=str(output_dir / "metrics" / name),
            cfg=EvalConfig(iou_thresh=args.eval_iou_thresh, fa_target=args.eval_fa_target, img_size=input_hw),
            class_index_to_name=class_names,
        )
        result = runner.run(epoch=0, model=model, val_loader=eval_loader)
        summary = _summarize_eval_result(name, result)
        metrics_rows.append(summary)
        print(
            f"       map50={summary['map50']} | map50_95={summary['map50_95']} | "
            f"conf_thresh={summary['conf_thresh']} | "
            f"TP/FP/FN={summary['tp_at_conf_thresh']}/{summary['fp_at_conf_thresh']}/{summary['fn_total']}"
        )

    _write_metrics_csv(output_dir / "metrics_summary.csv", metrics_rows)
    (output_dir / "metrics_summary.json").write_text(json.dumps(metrics_rows, indent=2, default=_json_default), encoding="utf-8")
    return metrics_rows


def _visualize_samples(args, output_dir: Path, models: dict[str, tuple[YOLOv11, bool]], loader, class_names, input_hw):
    summaries: list[dict[str, Any]] = []
    seen = 0

    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(next(iter(models.values()))[0].device)
            batch_predictions = {}
            for name, (model, without_nms) in models.items():
                outputs = model(images)
                dist_out, cls_out = outputs
                detections = model.postprocess(
                    dist_out,
                    cls_out,
                    dist_out,
                    conf_thres=args.score_threshold,
                    without_nms=without_nms,
                )
                batch_predictions[name] = detections

            for batch_index in range(images.shape[0]):
                global_index = args.start_index + seen
                gt_boxes, gt_labels = _gt_from_targets(targets, batch_index)
                fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
                sample_summary: dict[str, Any] = {
                    "sample_index": int(global_index),
                    "n_gt": int(len(gt_boxes)),
                    "models": {},
                }
                for ax, name in zip(axes, ("one2many_nms", "one2one_tal", "one2one_hungarian")):
                    boxes, scores, labels = _detections_to_normalized(
                        batch_predictions[name][batch_index],
                        input_hw=input_hw,
                        score_threshold=args.score_threshold,
                        top_k=args.top_k,
                    )
                    _draw_panel(
                        ax,
                        image=images[batch_index],
                        pred_boxes=boxes,
                        pred_scores=scores,
                        pred_labels=labels,
                        gt_boxes=gt_boxes,
                        gt_labels=gt_labels,
                        class_names=class_names,
                        title=f"{name} | n={len(boxes)}",
                        cmap=args.cmap,
                    )
                    sample_summary["models"][name] = {
                        "n_pred": int(len(boxes)),
                        "scores": [float(value) for value in scores.tolist()],
                        "labels": [int(value) for value in labels.tolist()],
                    }

                output_path = output_dir / "samples" / f"sample_{global_index:05d}.png"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fig.tight_layout()
                fig.savefig(output_path, dpi=180)
                plt.close(fig)
                summaries.append(sample_summary)
                seen += 1

    return summaries


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_resolutions = find_input_resolutions(args.data_dir, split=args.split)
    if len(input_resolutions) != len(DEFAULT_RES_KEYS):
        raise ValueError(f"Expected {len(DEFAULT_RES_KEYS)} resolutions, found {len(input_resolutions)}: {input_resolutions}")
    res_key_to_hw = dict(zip(DEFAULT_RES_KEYS, input_resolutions))
    input_hw = res_key_to_hw[args.res_key]
    input_channels = preprocessing_num_channels(args.preprocessing)
    class_names = load_class_index_to_name(args.data_dir)

    dataset = YOLODatasetSpecificRes(
        data_dir=os.path.join(args.data_dir, args.split, "data"),
        labels_dir=os.path.join(args.data_dir, args.split, "labels_detect"),
        res_hw=input_hw,
        res_key=args.res_key,
        preprocessing=args.preprocessing,
    )
    eval_loader = _build_loader(args, dataset)
    subset_indices = range(args.start_index, min(len(dataset), args.start_index + args.num_samples))
    visual_loader = _build_loader(args, dataset, subset_indices=subset_indices)

    models = _load_models(args, output_dir, input_channels)
    metrics_rows = _evaluate_models(args, output_dir, models, eval_loader, class_names, input_hw)
    summaries = _visualize_samples(args, output_dir, models, visual_loader, class_names, input_hw)

    payload = {
        "data_dir": str(args.data_dir),
        "split": args.split,
        "res_key": args.res_key,
        "input_hw": list(input_hw),
        "dataset_size": len(dataset),
        "eval_iou_thresh": float(args.eval_iou_thresh),
        "eval_fa_target": float(args.eval_fa_target),
        "score_threshold": float(args.score_threshold),
        "top_k": int(args.top_k),
        "checkpoints": {
            "one2many_nms": args.one2many_checkpoint,
            "one2one_tal": args.one2one_tal_checkpoint,
            "one2one_hungarian": args.one2one_hungarian_checkpoint,
        },
        "metrics": metrics_rows,
        "samples": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    print(f"Saved full-dataset metrics to {output_dir / 'metrics_summary.csv'}")
    print(f"Saved {len(summaries)} sample visualizations to {output_dir / 'samples'}")
    print(f"Saved summary to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
