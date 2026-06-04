from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from ..post_process import nms_pure


def _coerce_prediction_tensor(
    predictions: Optional[torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if predictions is None:
        return torch.zeros((0, 6), device=device, dtype=dtype)

    if not torch.is_tensor(predictions):
        predictions = torch.as_tensor(predictions, device=device, dtype=dtype)
    else:
        predictions = predictions.to(device=device, dtype=dtype)

    if predictions.ndim != 2:
        raise ValueError(
            "Each prediction set must be a 2D tensor shaped [N, >=6] with "
            "[x1, y1, x2, y2, score, class, ...]."
        )
    if predictions.shape[1] < 6:
        raise ValueError(
            "Each prediction row must contain at least 6 values: "
            "[x1, y1, x2, y2, score, class]."
        )

    return predictions[:, :6].detach().clone()


def _empty_output(*, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros((0, 6), device=device, dtype=dtype)


def _gather_rows(tensor: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    if len(indices) == 0:
        return _empty_output(device=tensor.device, dtype=tensor.dtype)
    return tensor[torch.as_tensor(indices, device=tensor.device, dtype=torch.long)]


def nms_fusion_post_nms(
    prediction_sets: Sequence[Optional[torch.Tensor]],
    *,
    iou_thresh: float = 0.5,
    agnostic: bool = False,
    max_det: int = 300,
    max_wh: float = 7680.0,
) -> Dict[str, Any]:
    """
    Fuse already post-NMS single-resolution predictions with a single global NMS.

    Args:
        prediction_sets:
            Sequence of tensors shaped [N_r, >=6], one tensor per model.
            Rows are expected as [x1, y1, x2, y2, score, class, ...].
        iou_thresh:
            IoU threshold applied during the global fusion NMS.
        agnostic:
            If True, NMS is class-agnostic. Otherwise boxes are offset by class.
        max_det:
            Maximum number of fused detections kept after NMS.
        max_wh:
            Class offset magnitude when `agnostic=False`.
    """
    if not 0.0 <= iou_thresh <= 1.0:
        raise ValueError("iou_thresh must be between 0 and 1.")

    device = torch.device("cpu")
    dtype = torch.float32

    normalized_sets: List[torch.Tensor] = []
    all_sources: List[Dict[str, int]] = []
    for resolution_index, preds in enumerate(prediction_sets):
        preds_tensor = _coerce_prediction_tensor(preds, device=device, dtype=dtype)
        normalized_sets.append(preds_tensor)
        base_index = len(all_sources)
        all_sources.extend(
            {
                "global_index": base_index + local_index,
                "resolution_index": resolution_index,
                "prediction_index": local_index,
            }
            for local_index in range(len(preds_tensor))
        )

    if normalized_sets:
        all_predictions = torch.cat(normalized_sets, dim=0)
    else:
        all_predictions = _empty_output(device=device, dtype=dtype)

    if len(all_predictions) == 0:
        return {
            "all_predictions": all_predictions,
            "all_sources": all_sources,
            "kept_indices": [],
            "fused_predictions": all_predictions,
            "fused_sources": [],
            "iou_thresh": float(iou_thresh),
            "agnostic": bool(agnostic),
        }

    boxes = all_predictions[:, :4]
    scores = all_predictions[:, 4]
    classes = all_predictions[:, 5:6]
    boxes_for_nms = boxes if agnostic else boxes + classes * float(max_wh)

    keep = nms_pure(boxes_for_nms, scores, iou_threshold=iou_thresh)
    keep = keep[:max_det]
    kept_indices = keep.detach().cpu().tolist()

    return {
        "all_predictions": all_predictions,
        "all_sources": all_sources,
        "kept_indices": kept_indices,
        "fused_predictions": _gather_rows(all_predictions, kept_indices),
        "fused_sources": [all_sources[idx] for idx in kept_indices],
        "iou_thresh": float(iou_thresh),
        "agnostic": bool(agnostic),
    }
