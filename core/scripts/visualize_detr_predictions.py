from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from detector2026.core.models import DETR
from detector2026.core.scripts.train_benchmark_suite import find_input_resolutions
from detector2026.core.utils.dataset import YOLODatasetSpecificRes, load_class_index_to_name
from detector2026.core.utils.divers import xywh2xyxy
from detector2026.core.utils.metrics import box_iou
from detector2026.core.utils.preprocess import preprocessing_num_channels


DEFAULT_RES_KEYS = ["cfg512", "cfg256", "cfg128", "cfg1024", "cfg2048"]
DEFAULT_DATA_DIR = "/data/RAWSIM/RMA/rf_dataset_for_real_validation"
DEFAULT_OUTPUT_DIR_PARENT = "/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation/detr_nightly_sweep/baseline"
CHEKPOINT_PATH = "/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation/detr_nightly_sweep/baseline/best.pt"

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize DETR predictions against ground truth on dataset samples.")
    parser.add_argument("--data-dir",default=DEFAULT_DATA_DIR,)
    parser.add_argument("--checkpoint", default=CHEKPOINT_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR_PARENT)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--res-key", default="cfg512", choices=DEFAULT_RES_KEYS)
    parser.add_argument("--preprocessing", default="none")
    parser.add_argument("--width-mult", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cmap", default="viridis")
    return parser.parse_args()


def _class_name(mapping: dict[int, str], label: int) -> str:
    return mapping.get(int(label), str(int(label)))


def _gt_from_targets(targets: torch.Tensor, image_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows = targets[targets[:, 0].long() == image_index]
    boxes = xywh2xyxy(rows[:, 2:6]).clamp(0.0, 1.0)
    labels = rows[:, 1].long()
    return boxes, labels


def _draw_sample(
    *,
    image: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    class_names: dict[int, str],
    output_path: Path,
    cmap: str,
    sample_title: str,
) -> dict[str, Any]:
    image = image.detach().cpu()
    pred_boxes = pred_boxes.detach().cpu()
    pred_scores = pred_scores.detach().cpu()
    pred_labels = pred_labels.detach().cpu()
    gt_boxes = gt_boxes.detach().cpu()
    gt_labels = gt_labels.detach().cpu()
    height, width = image.shape[-2:]

    fig, ax = plt.subplots(figsize=(10, 6))
    arr = image.squeeze(0).numpy()
    ax.imshow(arr, aspect="auto", cmap=cmap)
    ax.set_title(sample_title)
    ax.axis("off")

    for gt_index, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
        x1, y1, x2, y2 = box.tolist()
        rect = patches.Rectangle(
            (x1 * width, y1 * height),
            (x2 - x1) * width,
            (y2 - y1) * height,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1 * width,
            y1 * height,
            f"GT {gt_index}: {_class_name(class_names, int(label))}",
            color="black",
            fontsize=9,
            bbox=dict(facecolor="lime", alpha=0.75, pad=2),
        )

    pairwise_iou = box_iou(pred_boxes, gt_boxes) if pred_boxes.numel() and gt_boxes.numel() else torch.zeros((len(pred_boxes), len(gt_boxes)))
    best_iou = pairwise_iou.max(dim=1).values if pairwise_iou.numel() else torch.zeros((len(pred_boxes),))
    best_gt = pairwise_iou.argmax(dim=1) if pairwise_iou.numel() else torch.full((len(pred_boxes),), -1, dtype=torch.long)

    for pred_index, (box, score, label) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
        x1, y1, x2, y2 = box.tolist()
        color = "red" if best_iou[pred_index] < 0.5 else "cyan"
        rect = patches.Rectangle(
            (x1 * width, y1 * height),
            (x2 - x1) * width,
            (y2 - y1) * height,
            linewidth=1.5,
            edgecolor=color,
            facecolor="none",
            linestyle="--",
        )
        ax.add_patch(rect)
        gt_suffix = f", gt={int(best_gt[pred_index])}" if int(best_gt[pred_index]) >= 0 else ""
        ax.text(
            x1 * width,
            max(0.0, y1 * height - 4.0),
            f"P{pred_index}: {_class_name(class_names, int(label))} {float(score):.2f}, IoU={float(best_iou[pred_index]):.2f}{gt_suffix}",
            color="white",
            fontsize=8,
            bbox=dict(facecolor=color, alpha=0.75, pad=2),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return {
        "n_gt": int(len(gt_boxes)),
        "n_pred": int(len(pred_boxes)),
        "max_iou_per_pred": [float(value) for value in best_iou.tolist()],
        "scores": [float(value) for value in pred_scores.tolist()],
        "labels": [int(value) for value in pred_labels.tolist()],
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    input_resolutions = find_input_resolutions(args.data_dir, split=args.split)
    if len(input_resolutions) != len(DEFAULT_RES_KEYS):
        raise ValueError(f"Expected {len(DEFAULT_RES_KEYS)} resolutions, found {len(input_resolutions)}: {input_resolutions}")
    res_key_to_hw = dict(zip(DEFAULT_RES_KEYS, input_resolutions))
    input_hw = res_key_to_hw[args.res_key]
    input_channels = preprocessing_num_channels(args.preprocessing)
    class_names = load_class_index_to_name(args.data_dir)

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
    )
    model.load_weights(args.checkpoint, device=args.device, eval_mode=True)

    dataset = YOLODatasetSpecificRes(
        data_dir=os.path.join(args.data_dir, args.split, "data"),
        labels_dir=os.path.join(args.data_dir, args.split, "labels_detect"),
        res_hw=input_hw,
        res_key=args.res_key,
        preprocessing=args.preprocessing,
    )
    subset_indices = range(args.start_index, min(len(dataset), args.start_index + args.num_samples))
    subset = torch.utils.data.Subset(dataset, list(subset_indices))
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=YOLODatasetSpecificRes.collate_fn,
    )

    summaries = []
    seen = 0
    model.eval()
    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(model.device)
            outputs = model(images)
            detections = model.postprocess(
                outputs,
                score_threshold=args.score_threshold,
                top_k=args.top_k,
                absolute_boxes=False,
            )
            for batch_index, detection in enumerate(detections):
                global_index = args.start_index + seen
                gt_boxes, gt_labels = _gt_from_targets(targets, batch_index)
                summary = _draw_sample(
                    image=images[batch_index],
                    pred_boxes=detection["boxes"],
                    pred_scores=detection["scores"],
                    pred_labels=detection["labels"],
                    gt_boxes=gt_boxes,
                    gt_labels=gt_labels,
                    class_names=class_names,
                    output_path=output_dir / "samples" / f"sample_{global_index:05d}.png",
                    cmap=args.cmap,
                    sample_title=f"sample {global_index} | threshold={args.score_threshold:g} | top_k={args.top_k}",
                )
                summary["sample_index"] = int(global_index)
                summaries.append(summary)
                seen += 1

    payload = {
        "checkpoint": str(args.checkpoint),
        "data_dir": str(args.data_dir),
        "split": args.split,
        "res_key": args.res_key,
        "input_hw": list(input_hw),
        "score_threshold": float(args.score_threshold),
        "top_k": int(args.top_k),
        "samples": summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {len(summaries)} sample visualizations to {output_dir / 'samples'}")
    print(f"Saved summary to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
