from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from ..metrics import box_iou


def _coerce_prediction_tensor(
    predictions: Optional[torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Normalize a post-NMS prediction tensor to shape [N, 6]."""
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


def _unique_keep_order(indices: Sequence[int]) -> List[int]:
    seen = set()
    ordered: List[int] = []
    for idx in indices:
        if idx in seen:
            continue
        seen.add(idx)
        ordered.append(idx)
    return ordered


def _box_area(boxes: torch.Tensor) -> torch.Tensor:
    widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    return widths * heights


def _maximum_bipartite_matching(
    adjacency: Sequence[Sequence[int]],
    candidate_scores: Sequence[Dict[int, float]],
) -> Dict[int, int]:
    """
    Compute a maximum-cardinality GT->prediction assignment.

    Tie-breaking is heuristic: GTs with fewer candidates are processed first,
    and each GT tries higher-IoU predictions first.
    """
    ordered_gts = sorted(
        range(len(adjacency)),
        key=lambda gt_idx: (
            len(adjacency[gt_idx]),
            -max(candidate_scores[gt_idx].values(), default=-1.0),
            gt_idx,
        ),
    )

    pred_to_gt: Dict[int, int] = {}

    def _try_assign(gt_idx: int, visited_preds: set[int]) -> bool:
        ordered_preds = sorted(
            adjacency[gt_idx],
            key=lambda pred_idx: (-candidate_scores[gt_idx].get(pred_idx, 0.0), pred_idx),
        )
        for pred_idx in ordered_preds:
            if pred_idx in visited_preds:
                continue
            visited_preds.add(pred_idx)
            current_gt = pred_to_gt.get(pred_idx)
            if current_gt is None or _try_assign(current_gt, visited_preds):
                pred_to_gt[pred_idx] = gt_idx
                return True
        return False

    for gt_idx in ordered_gts:
        _try_assign(gt_idx, set())

    return {gt_idx: pred_idx for pred_idx, gt_idx in pred_to_gt.items()}


def _suppress_redundant_false_alarms(
    predictions: torch.Tensor,
    indices: Sequence[int],
    iou_thresh: float,
) -> List[int]:
    """
    Remove overlapping false alarms using the rule from the paper pseudocode:
    if two false alarms overlap by at least ``iou_thresh``, discard the smaller box.
    """
    kept_indices = list(indices)
    if len(kept_indices) < 2:
        return kept_indices

    local_boxes = predictions[torch.as_tensor(kept_indices, device=predictions.device, dtype=torch.long), :4]
    ious = box_iou(local_boxes, local_boxes)
    areas = _box_area(local_boxes)

    removed = torch.zeros(len(kept_indices), device=predictions.device, dtype=torch.bool)
    for i in range(len(kept_indices)):
        if removed[i]:
            continue
        for j in range(len(kept_indices)):
            if i == j or removed[i]:
                continue
            if ious[i, j] >= iou_thresh and areas[i] < areas[j]:
                removed[i] = True

    return [idx for idx, is_removed in zip(kept_indices, removed.tolist()) if not is_removed]


def oracle_or_post_nms(
    prediction_sets: Sequence[Optional[torch.Tensor]],
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    *,
    iou_thresh: float = 0.5,
    false_alarm_iou_thresh: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Oracle-OR fusion on already post-NMS single-resolution predictions.

    Args:
        prediction_sets:
            Sequence of tensors shaped [N_r, >=6], one tensor per single-resolution
            model. Rows are expected as [x1, y1, x2, y2, score, class, ...].
        gt_boxes:
            Ground-truth boxes shaped [N_gt, 4] in xyxy format.
        gt_labels:
            Ground-truth class indices shaped [N_gt].
        iou_thresh:
            IoU threshold used to associate predictions with a GT instance.
        false_alarm_iou_thresh:
            IoU threshold for the redundancy filtering applied to raw false alarms.
            Defaults to ``iou_thresh``.

    Returns:
        Dictionary containing the fused oracle predictions and traceability metadata.
        The main outputs are:
            - ``oracle_predictions``: selected oracle detections + retained false alarms
            - ``oracle_false_alarms``: retained false alarms after redundancy filtering
            - ``selected_predictions``: one best candidate per covered GT
    """
    if gt_boxes.ndim != 2 or gt_boxes.shape[1] != 4:
        raise ValueError("gt_boxes must be a tensor shaped [N_gt, 4] in xyxy format.")
    if gt_labels.ndim != 1:
        raise ValueError("gt_labels must be a 1D tensor shaped [N_gt].")
    if gt_boxes.shape[0] != gt_labels.shape[0]:
        raise ValueError("gt_boxes and gt_labels must have the same number of rows.")

    false_alarm_iou_thresh = iou_thresh if false_alarm_iou_thresh is None else false_alarm_iou_thresh

    device = gt_boxes.device
    dtype = gt_boxes.dtype if gt_boxes.numel() else torch.float32
    gt_boxes = gt_boxes.to(device=device, dtype=dtype)
    gt_labels = gt_labels.to(device=device)

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

    matched_indices = set()
    intersecting_indices = set()
    per_gt_candidates: List[Dict[str, Any]] = []

    for gt_index, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
        if len(all_predictions) == 0:
            break

        ious = box_iou(all_predictions[:, :4], gt_box.unsqueeze(0)).squeeze(1)
        candidate_indices = (ious >= iou_thresh).nonzero(as_tuple=True)[0].tolist()

        if not candidate_indices:
            per_gt_candidates.append(
                {
                    "gt_index": gt_index,
                    "gt_label": int(gt_label.item()),
                    "candidate_indices": [],
                    "class_filtered_indices": [],
                    "retained_indices": [],
                    "candidate_ious": {},
                }
            )
            continue

        intersecting_indices.update(candidate_indices)

        cls_candidate_indices = [
            idx for idx in candidate_indices
            if int(all_predictions[idx, 5].item()) == int(gt_label.item())
        ]
        retained_indices = cls_candidate_indices if cls_candidate_indices else candidate_indices

        matched_indices.update(retained_indices)
        per_gt_candidates.append(
            {
                "gt_index": gt_index,
                "gt_label": int(gt_label.item()),
                "candidate_indices": candidate_indices,
                "class_filtered_indices": cls_candidate_indices,
                "retained_indices": retained_indices,
                "candidate_ious": {idx: float(ious[idx].item()) for idx in retained_indices},
            }
        )

    gt_adjacency = [entry["retained_indices"] for entry in per_gt_candidates]
    gt_candidate_scores = [entry["candidate_ious"] for entry in per_gt_candidates]
    gt_to_pred = _maximum_bipartite_matching(gt_adjacency, gt_candidate_scores)

    selected_indices = []
    per_gt_selection: List[Dict[str, Any]] = []
    for entry in per_gt_candidates:
        selected_index = gt_to_pred.get(entry["gt_index"])
        if selected_index is not None:
            selected_indices.append(selected_index)
        per_gt_selection.append(
            {
                **entry,
                "selected_index": selected_index,
                "selected_iou": (
                    float(entry["candidate_ious"][selected_index])
                    if selected_index is not None
                    else None
                ),
            }
        )

    # False alarms are only predictions that do not intersect any ground truth.
    raw_false_alarm_indices = [idx for idx in range(len(all_predictions)) if idx not in intersecting_indices]
    oracle_false_alarm_indices = _suppress_redundant_false_alarms(
        all_predictions,
        raw_false_alarm_indices,
        false_alarm_iou_thresh,
    )

    oracle_indices = _unique_keep_order(selected_indices + oracle_false_alarm_indices)

    def _subset_sources(indices: Sequence[int]) -> List[Dict[str, int]]:
        return [all_sources[idx] for idx in indices]

    return {
        "all_predictions": all_predictions,
        "all_sources": all_sources,
        "selected_indices": selected_indices,
        "selected_predictions": _gather_rows(all_predictions, selected_indices),
        "selected_sources": _subset_sources(selected_indices),
        "matched_indices": sorted(matched_indices),
        "matched_predictions": _gather_rows(all_predictions, sorted(matched_indices)),
        "matched_sources": _subset_sources(sorted(matched_indices)),
        "intersecting_indices": sorted(intersecting_indices),
        "intersecting_predictions": _gather_rows(all_predictions, sorted(intersecting_indices)),
        "intersecting_sources": _subset_sources(sorted(intersecting_indices)),
        "raw_false_alarm_indices": raw_false_alarm_indices,
        "raw_false_alarms": _gather_rows(all_predictions, raw_false_alarm_indices),
        "raw_false_alarm_sources": _subset_sources(raw_false_alarm_indices),
        "oracle_false_alarm_indices": oracle_false_alarm_indices,
        "oracle_false_alarms": _gather_rows(all_predictions, oracle_false_alarm_indices),
        "oracle_false_alarm_sources": _subset_sources(oracle_false_alarm_indices),
        "oracle_indices": oracle_indices,
        "oracle_predictions": _gather_rows(all_predictions, oracle_indices),
        "oracle_sources": _subset_sources(oracle_indices),
        "per_gt_selection": per_gt_selection,
        "iou_thresh": float(iou_thresh),
        "false_alarm_iou_thresh": float(false_alarm_iou_thresh),
    }
