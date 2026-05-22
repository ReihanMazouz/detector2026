from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment

from .divers import xywh2xyxy
from .metrics import bbox_iou


def _sanitize_xywh(boxes: torch.Tensor) -> torch.Tensor:
    boxes = torch.nan_to_num(boxes, nan=0.0, posinf=1.0, neginf=0.0)
    xy = boxes[..., :2].clamp(0.0, 1.0)
    wh = boxes[..., 2:].clamp(1e-6, 1.0)
    return torch.cat((xy, wh), dim=-1)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)
    iou = bbox_iou(boxes1[:, None, :], boxes2[None, :, :], xywh=False, GIoU=True).squeeze(-1)
    return torch.nan_to_num(iou, nan=-1.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)


class HungarianMatcher:
    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        use_focal_loss: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        self.cost_class = float(cost_class)
        self.cost_bbox = float(cost_bbox)
        self.cost_giou = float(cost_giou)
        self.use_focal_loss = bool(use_focal_loss)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        if self.cost_class == 0.0 and self.cost_bbox == 0.0 and self.cost_giou == 0.0:
            raise ValueError("At least one matching cost must be non-zero.")

    @torch.no_grad()
    def __call__(self, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]]):
        logits = torch.nan_to_num(outputs["pred_logits"], nan=0.0, posinf=50.0, neginf=-50.0)
        boxes = _sanitize_xywh(outputs["pred_boxes"])

        indices = []
        for batch_index in range(logits.shape[0]):
            tgt_labels = targets[batch_index]["labels"]
            tgt_boxes = _sanitize_xywh(targets[batch_index]["boxes"])
            if tgt_boxes.numel() == 0:
                empty = torch.empty(0, dtype=torch.int64, device=logits.device)
                indices.append((empty, empty))
                continue

            if self.use_focal_loss:
                pred_scores = logits[batch_index].sigmoid().clamp(1e-8, 1.0 - 1e-8)
                pred_scores = pred_scores[:, tgt_labels]
                neg_cost = (1.0 - self.focal_alpha) * pred_scores.pow(self.focal_gamma) * (-(1.0 - pred_scores).log())
                pos_cost = self.focal_alpha * (1.0 - pred_scores).pow(self.focal_gamma) * (-pred_scores.log())
                cost_class = pos_cost - neg_cost
            else:
                out_prob = logits[batch_index].softmax(-1)
                cost_class = -out_prob[:, tgt_labels]
            cost_bbox = torch.cdist(boxes[batch_index], tgt_boxes, p=1)
            cost_giou = -generalized_box_iou(
                xywh2xyxy(boxes[batch_index]),
                xywh2xyxy(tgt_boxes),
            )
            cost = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
            cost = torch.nan_to_num(cost, nan=1e6, posinf=1e6, neginf=-1e6)
            row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
            indices.append(
                (
                    torch.as_tensor(row_ind, dtype=torch.int64, device=logits.device),
                    torch.as_tensor(col_ind, dtype=torch.int64, device=logits.device),
                )
            )
        return indices
