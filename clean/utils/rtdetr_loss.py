from __future__ import annotations

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
