from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .detr_matcher import HungarianMatcher
from .divers import xywh2xyxy
from .metrics import bbox_iou


def targets_from_yolo_tensor(targets: torch.Tensor, batch_size: int, device: torch.device) -> list[dict[str, torch.Tensor]]:
    targets = targets.to(device)
    packed = []
    for batch_index in range(batch_size):
        rows = targets[targets[:, 0].long() == batch_index]
        packed.append(
            {
                "labels": rows[:, 1].long(),
                "boxes": rows[:, 2:6].to(dtype=torch.float32).clamp(0.0, 1.0),
            }
        )
    return packed


class DETRLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher | None = None,
        eos_coef: float = 0.1,
        lambda_cls: float = 1.0,
        lambda_bbox: float = 5.0,
        lambda_giou: float = 2.0,
        aux_loss: bool = True,
        aux_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.matcher = matcher or HungarianMatcher()
        self.lambda_cls = float(lambda_cls)
        self.lambda_bbox = float(lambda_bbox)
        self.lambda_giou = float(lambda_giou)
        self.aux_loss = bool(aux_loss)
        self.aux_loss_weight = float(aux_loss_weight)

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = float(eos_coef)
        self.register_buffer("empty_weight", empty_weight)

    @staticmethod
    def _get_src_permutation_idx(indices):
        if not indices:
            empty = torch.empty(0, dtype=torch.int64)
            return empty, empty
        batch_idx = torch.cat([torch.full_like(src, index) for index, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for src, _ in indices])
        return batch_idx, src_idx

    def _loss_single(self, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]]):
        indices = self.matcher(outputs, targets)
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        idx = self._get_src_permutation_idx(indices)

        target_classes = torch.full(
            pred_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=pred_logits.device,
        )
        if idx[0].numel() > 0:
            target_classes_o = torch.cat([target["labels"][j] for target, (_, j) in zip(targets, indices)])
            target_classes[idx] = target_classes_o

        loss_cls = F.cross_entropy(pred_logits.transpose(1, 2), target_classes, self.empty_weight)
        num_boxes = max(float(sum(len(target["labels"]) for target in targets)), 1.0)
        if idx[0].numel() == 0:
            loss_bbox = pred_boxes.sum() * 0.0
            loss_giou = pred_boxes.sum() * 0.0
        else:
            src_boxes = pred_boxes[idx]
            target_boxes = torch.cat([target["boxes"][j] for target, (_, j) in zip(targets, indices)], dim=0)
            loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="sum") / num_boxes
            giou = bbox_iou(
                xywh2xyxy(src_boxes),
                xywh2xyxy(target_boxes),
                xywh=False,
                GIoU=True,
            )
            loss_giou = (1.0 - giou.diag()).sum() / num_boxes

        total = self.lambda_cls * loss_cls + self.lambda_bbox * loss_bbox + self.lambda_giou * loss_giou
        return total, {
            "loss_cls": loss_cls,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
        }

    def forward(self, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]]):
        total, raw_parts = self._loss_single(outputs, targets)
        parts = {
            "loss_cls": raw_parts["loss_cls"],
            "loss_bbox": raw_parts["loss_bbox"],
            "loss_giou": raw_parts["loss_giou"],
        }

        aux_outputs = outputs.get("aux_outputs", [])
        if self.aux_loss:
            for layer_index, aux_output in enumerate(aux_outputs):
                aux_total, aux_parts = self._loss_single(aux_output, targets)
                total = total + self.aux_loss_weight * aux_total
                for key, value in aux_parts.items():
                    parts[f"{key}_aux_{layer_index}"] = value

        detached_parts = {key: float(value.detach().item()) for key, value in parts.items()}
        return total, detached_parts
