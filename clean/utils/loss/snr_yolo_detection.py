import torch
import torch.nn as nn

from ..divers import xywh2xyxy
from .bbox import BboxLoss
from .common import image_size_from_feats, stride_tensor_xyxy


class SNRYOLODetectionLoss:
    def __init__(
        self,
        reg_max=16,
        lambda_box=7.5,
        lambda_dfl=1.5,
        lambda_cls=0.5,
        num_classes=80,
        strides=[8, 16, 32],
        tal_topk=10,
        minimum_possible_candidates=None,
        snr_min=-2,
        device="cpu",
    ):
        self.lambda_box = lambda_box
        self.lambda_dfl = lambda_dfl
        self.lambda_cls = lambda_cls
        self.nc = num_classes
        self.strides = strides
        self.tal_topk = tal_topk
        self.device = device
        self.reg_max = reg_max

        self.no = num_classes + reg_max * 4
        self.use_dfl = reg_max > 1
        self.snr_min = snr_min

        from ..tal import TaskAlignedAssigner

        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            minimum_possible_candidates=minimum_possible_candidates,
        )
        self.bbox_loss = BboxLoss(self.reg_max).to(device)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0].long()
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out[..., :5], out[..., 5:]

    def bbox_decode(self, anchor_points, pred_dist):
        from ..tal import dist2bbox

        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, pred_distri, pred_scores, batch, feats):
        from ..tal import make_anchors

        loss = torch.zeros(3, device=self.device)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = image_size_from_feats(feats[0], self.strides[0], self.device, dtype)
        anchor_points, stride_tensor = make_anchors(feats, self.strides, 0.5)
        stride_tensor_boxes = stride_tensor_xyxy(stride_tensor)

        targets = torch.cat((
            batch["batch_idx"].view(-1, 1),
            batch["cls"].view(-1, 1),
            batch["bboxes"],
            batch["snr"],
        ), dim=1)

        targets, snrs = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, matched_gt_inds = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor_boxes).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        snrs_flat = snrs.squeeze(-1)
        assigned_snrs = torch.gather(snrs_flat, dim=1, index=matched_gt_inds.clamp(min=0))
        assigned_snrs = (assigned_snrs - self.snr_min).clamp(min=0.0)
        assigned_snrs = assigned_snrs.unsqueeze(-1)

        loss_cls = self.bce(pred_scores, target_scores.to(dtype))
        loss[1] = (loss_cls * assigned_snrs).sum() / (target_scores_sum.sum() + 1e-6)

        debug_data = []
        if fg_mask.sum():
            target_bboxes /= stride_tensor_boxes
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.lambda_box
        loss[1] *= self.lambda_cls
        loss[2] *= self.lambda_dfl

        for b in range(batch_size):
            fg_inds = fg_mask[b]
            debug_data.append({
                "gt_boxes": gt_bboxes[b].detach().cpu(),
                "task_selected_pred_boxes_abs": (pred_bboxes[b][fg_inds] * stride_tensor_boxes[fg_inds]).detach().cpu(),
                "task_selected_anchor_points_abs": (anchor_points[fg_inds] * stride_tensor[fg_inds]).detach().cpu(),
                "pred_bboxes_abs": (pred_bboxes[b] * stride_tensor_boxes).detach().cpu(),
            })

        return loss.sum() * batch_size, [
            loss[0].item(),
            loss[1].item(),
            loss[2].item(),
        ], debug_data
