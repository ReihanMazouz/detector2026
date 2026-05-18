from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment

from .divers import xywh2xyxy
from .metrics import bbox_iou


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)
    return bbox_iou(boxes1[:, None, :], boxes2[None, :, :], xywh=False, GIoU=True).squeeze(-1)


class HungarianMatcher:
    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        self.cost_class = float(cost_class)
        self.cost_bbox = float(cost_bbox)
        self.cost_giou = float(cost_giou)
        if self.cost_class == 0.0 and self.cost_bbox == 0.0 and self.cost_giou == 0.0:
            raise ValueError("At least one matching cost must be non-zero.")

    @torch.no_grad()
    def __call__(self, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]]):
        logits = outputs["pred_logits"]
        boxes = outputs["pred_boxes"]

        indices = []
        for batch_index in range(logits.shape[0]):
            tgt_labels = targets[batch_index]["labels"]
            tgt_boxes = targets[batch_index]["boxes"]
            if tgt_boxes.numel() == 0:
                empty = torch.empty(0, dtype=torch.int64, device=logits.device)
                indices.append((empty, empty))
                continue

            out_prob = logits[batch_index].softmax(-1)
            cost_class = -out_prob[:, tgt_labels]
            cost_bbox = torch.cdist(boxes[batch_index], tgt_boxes, p=1)
            cost_giou = -generalized_box_iou(
                xywh2xyxy(boxes[batch_index]),
                xywh2xyxy(tgt_boxes),
            )
            cost = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
            row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
            indices.append(
                (
                    torch.as_tensor(row_ind, dtype=torch.int64, device=logits.device),
                    torch.as_tensor(col_ind, dtype=torch.int64, device=logits.device),
                )
            )
        return indices
