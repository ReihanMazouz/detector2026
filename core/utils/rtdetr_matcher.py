from __future__ import annotations

from .detr_matcher import HungarianMatcher as _BaseHungarianMatcher, generalized_box_iou


class HungarianMatcher(_BaseHungarianMatcher):
    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        use_focal_loss: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__(
            cost_class=cost_class,
            cost_bbox=cost_bbox,
            cost_giou=cost_giou,
            use_focal_loss=use_focal_loss,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )
