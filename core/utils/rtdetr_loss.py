from __future__ import annotations

import torch
import torch.nn.functional as F

from .detr_loss import DETRLoss, targets_from_yolo_tensor
from .rtdetr_matcher import HungarianMatcher


class RTDETRLoss(DETRLoss):
    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher | None = None,
        eos_coef: float = 0.1,
        lambda_cls: float = 1.0,
        lambda_bbox: float = 5.0,
        lambda_giou: float = 2.0,
        cls_loss_type: str = "varifocal",
        vfl_alpha: float = 0.75,
        vfl_gamma: float = 2.0,
        aux_loss: bool = True,
        matcher_num_threads: int = 1,
    ):
        super().__init__(
            num_classes=num_classes,
            matcher=matcher
            or HungarianMatcher(
                use_focal_loss=str(cls_loss_type).lower() == "varifocal",
                num_threads=matcher_num_threads,
            ),
            eos_coef=eos_coef,
            lambda_cls=lambda_cls,
            lambda_bbox=lambda_bbox,
            lambda_giou=lambda_giou,
            aux_loss=aux_loss,
            cls_loss_type=cls_loss_type,
            vfl_alpha=vfl_alpha,
            vfl_gamma=vfl_gamma,
        )


class RTDETRROCCalibrationLoss(RTDETRLoss):
    """Classification-only RT-DETR calibration loss with a target false-alarm operating point.

    The Hungarian matching and standard classification loss are kept, but localization
    losses are omitted. The additional ROC term estimates a detached threshold from
    unmatched query scores and pushes matched query scores above that threshold.
    """

    def __init__(
        self,
        num_classes: int,
        pfa_target: float = 0.01,
        margin: float = 0.0,
        roc_alpha: float = 20.0,
        roc_weight: float = 1.0,
        aux_cls_loss: bool = True,
        matcher: HungarianMatcher | None = None,
        cls_loss_type: str = "varifocal",
        matcher_num_threads: int = 1,
        **kwargs,
    ):
        super().__init__(
            num_classes=num_classes,
            matcher=matcher,
            cls_loss_type=cls_loss_type,
            matcher_num_threads=matcher_num_threads,
            aux_loss=aux_cls_loss,
            lambda_bbox=0.0,
            lambda_giou=0.0,
            **kwargs,
        )
        if not 0.0 < float(pfa_target) < 1.0:
            raise ValueError("pfa_target must be in (0, 1).")
        self.pfa_target = float(pfa_target)
        self.margin = float(margin)
        self.roc_alpha = float(roc_alpha)
        self.roc_weight = float(roc_weight)

    def _scores_for_logits(self, pred_logits: torch.Tensor) -> torch.Tensor:
        class_logits = torch.nan_to_num(pred_logits[..., : self.num_classes], nan=0.0, posinf=50.0, neginf=-50.0)
        if self.cls_loss_type == "varifocal":
            return class_logits.sigmoid()
        return class_logits.softmax(-1)[..., : self.num_classes]

    def _roc_loss(
        self,
        pred_logits: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        indices,
        idx,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        scores = self._scores_for_logits(pred_logits)
        batch_size, num_queries, _ = scores.shape
        query_device = scores.device
        pos_scores = []
        neg_scores = []

        for batch_index in range(batch_size):
            matched_queries, matched_targets = indices[batch_index]
            matched_queries = matched_queries.to(device=query_device)
            matched_targets = matched_targets.to(device=query_device)
            is_positive = torch.zeros(num_queries, dtype=torch.bool, device=query_device)
            if matched_queries.numel() > 0:
                is_positive[matched_queries] = True
                labels = targets[batch_index]["labels"].to(device=query_device)[matched_targets]
                pos_scores.append(scores[batch_index, matched_queries, labels])
            if (~is_positive).any():
                neg_scores.append(scores[batch_index, ~is_positive].max(dim=-1).values)

        if not pos_scores:
            zero = pred_logits.sum() * 0.0
            return zero, {"roc_tau": 0.0, "roc_num_pos": 0.0, "roc_num_neg": 0.0}

        positive_scores = torch.cat(pos_scores)
        if neg_scores:
            negative_scores = torch.cat(neg_scores)
            quantile = 1.0 - self.pfa_target
            tau = torch.quantile(negative_scores.detach().float(), quantile).to(dtype=positive_scores.dtype)
        else:
            negative_scores = positive_scores.new_empty(0)
            tau = positive_scores.detach().new_tensor(0.0)

        roc = F.softplus(self.roc_alpha * (tau + self.margin - positive_scores)).mean()
        return roc, {
            "roc_tau": float(tau.detach().item()),
            "roc_num_pos": float(positive_scores.numel()),
            "roc_num_neg": float(negative_scores.numel()),
        }

    def _loss_cls_roc_single(self, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]]):
        indices = self.matcher(outputs, targets)
        pred_logits = torch.nan_to_num(outputs["pred_logits"], nan=0.0, posinf=50.0, neginf=-50.0)
        pred_boxes = outputs["pred_boxes"].detach()
        idx = self._get_src_permutation_idx(indices)
        loss_cls = self._loss_classification(pred_logits, pred_boxes, targets, indices, idx)
        loss_roc, roc_parts = self._roc_loss(pred_logits, targets, indices, idx)
        total = loss_cls + self.roc_weight * loss_roc
        return total, {"loss_cls": loss_cls, "loss_roc": loss_roc, **roc_parts}

    def forward(self, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]]):
        total, raw_parts = self._loss_cls_roc_single(outputs, targets)
        parts = dict(raw_parts)

        aux_outputs = outputs.get("aux_outputs", [])
        if self.aux_loss:
            for layer_index, aux_output in enumerate(aux_outputs):
                aux_total, aux_parts = self._loss_cls_roc_single(aux_output, targets)
                total = total + self.aux_loss_weight * aux_total
                for key, value in aux_parts.items():
                    parts[f"{key}_aux_{layer_index}"] = value

        detached_parts = {
            key: float(value.detach().item()) if torch.is_tensor(value) else float(value)
            for key, value in parts.items()
        }
        return total, detached_parts
