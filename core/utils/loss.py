import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from .metrics import bbox_iou
from .tal import make_anchors, bbox2dist, dist2bbox, TaskAlignedAssigner
from .divers import xywh2xyxy, concat_levels
from ..models.anisotropic_utils import stride_hw_to_xy


def _stride_tensor_xyxy(stride_tensor):
    return torch.cat([stride_tensor, stride_tensor], dim=1)


def _image_size_from_feats(feat, stride_hw, device, dtype):
    feat_h, feat_w = feat.shape[2:]
    stride_x, stride_y = stride_hw_to_xy(stride_hw)
    return torch.tensor([feat_h * stride_y, feat_w * stride_x], device=device, dtype=dtype)

class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL)."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module with regularization maximum."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """Return sum of left and right DFL losses from https://ieeexplore.ieee.org/document/9792391."""
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)
    

class BboxLoss(nn.Module):
    """Criterion class for computing training losses for bounding boxes."""

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """Compute IoU and DFL losses for bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class YOLODetectionLoss:
    def __init__(self,
                 reg_max=16,
                 lambda_box=7.5,
                 lambda_dfl=1.5,
                 lambda_cls=0.5,
                 num_classes=80,
                 strides=[8, 16, 32],
                 tal_topk=10,
                 minimum_possible_candidates=None,
                 device='cpu'):

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
        """Preprocess targets by converting to tensor format and scaling coordinates."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, pred_distri, pred_scores, batch, feats):
        loss = torch.zeros(3, device=self.device)
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = _image_size_from_feats(feats[0], self.strides[0], self.device, dtype)
        anchor_points, stride_tensor = make_anchors(feats, self.strides, 0.5)
        stride_tensor_boxes = _stride_tensor_xyxy(stride_tensor)

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        H, W = imgsz
        scale_tensor = torch.tensor([W, H, W, H], device=self.device)

        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=scale_tensor)
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor_boxes).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

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
            fg_inds = fg_inds.bool()
            debug_data.append({
                "gt_boxes": gt_bboxes[b].detach().cpu(),
                "task_selected_pred_boxes_abs": (pred_bboxes[b][fg_inds] * stride_tensor_boxes[fg_inds]).detach().cpu(),
                "task_selected_anchor_points_abs": (anchor_points[fg_inds] * stride_tensor[fg_inds]).detach().cpu(),
                "pred_bboxes_abs": (pred_bboxes[b] * stride_tensor_boxes).detach().cpu()
            })

        return loss.sum() * batch_size, [
            loss[0].item(),
            loss[1].item(),
            loss[2].item()
        ], debug_data


class YOLOOne2OneHungarianLoss:
    def __init__(self,
                 reg_max=16,
                 lambda_box=7.5,
                 lambda_dfl=1.5,
                 lambda_cls=0.5,
                 cost_class=2.0,
                 cost_bbox=5.0,
                 cost_giou=2.0,
                 negative_to_positive_ratio=1.0,
                 num_classes=80,
                 strides=[8, 16, 32],
                 device='cpu'):

        self.lambda_box = lambda_box
        self.lambda_dfl = lambda_dfl
        self.lambda_cls = lambda_cls
        self.cost_class = float(cost_class)
        self.cost_bbox = float(cost_bbox)
        self.cost_giou = float(cost_giou)
        self.negative_to_positive_ratio = float(negative_to_positive_ratio)
        self.nc = num_classes
        self.strides = strides
        self.device = device
        self.reg_max = reg_max
        self.use_dfl = reg_max > 1
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    @torch.no_grad()
    def match(self, pred_scores, pred_bboxes_norm, gt_labels, gt_bboxes_norm):
        indices = []
        for b in range(pred_scores.shape[0]):
            valid = gt_bboxes_norm[b].sum(-1).gt(0.0)
            tgt_boxes = gt_bboxes_norm[b][valid]
            tgt_labels = gt_labels[b][valid].squeeze(-1).long()
            if tgt_boxes.numel() == 0:
                empty = torch.empty(0, dtype=torch.long, device=pred_scores.device)
                indices.append((empty, empty))
                continue

            scores = pred_scores[b].sigmoid()
            cost_class = -scores[:, tgt_labels]
            cost_bbox = torch.cdist(pred_bboxes_norm[b], tgt_boxes, p=1)
            cost_giou = -bbox_iou(
                pred_bboxes_norm[b].unsqueeze(1),
                tgt_boxes.unsqueeze(0),
                xywh=False,
                GIoU=True,
            ).squeeze(-1)
            cost = (
                self.cost_class * cost_class
                + self.cost_bbox * cost_bbox
                + self.cost_giou * cost_giou
            )
            row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
            indices.append((
                torch.as_tensor(row_ind, dtype=torch.long, device=pred_scores.device),
                torch.as_tensor(col_ind, dtype=torch.long, device=pred_scores.device),
            ))
        return indices

    def __call__(self, pred_distri, pred_scores, batch, feats):
        dtype = pred_scores.dtype
        batch_size, num_anchors = pred_scores.shape[:2]
        imgsz = _image_size_from_feats(feats[0], self.strides[0], self.device, dtype)
        anchor_points, stride_tensor = make_anchors(feats, self.strides, 0.5)
        stride_tensor_boxes = _stride_tensor_xyxy(stride_tensor)

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        H, W = imgsz
        scale_tensor = torch.tensor([W, H, W, H], device=self.device, dtype=dtype)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=scale_tensor)
        gt_labels, gt_bboxes_abs = targets.split((1, 4), 2)
        gt_bboxes_norm = gt_bboxes_abs / scale_tensor

        pred_bboxes_grid = self.bbox_decode(anchor_points, pred_distri)
        pred_bboxes_abs = pred_bboxes_grid * stride_tensor_boxes
        pred_bboxes_norm = pred_bboxes_abs / scale_tensor

        indices = self.match(pred_scores.detach(), pred_bboxes_norm.detach(), gt_labels, gt_bboxes_norm)

        cls_targets = torch.zeros_like(pred_scores)
        fg_mask = torch.zeros((batch_size, num_anchors), dtype=torch.bool, device=pred_scores.device)

        matched_pred_batches = []
        matched_pred_indices = []
        matched_target_boxes_norm = []
        matched_target_boxes_grid = []
        matched_target_labels = []
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            valid = gt_bboxes_abs[b].sum(-1).gt(0.0)
            labels_b = gt_labels[b][valid].squeeze(-1).long()[tgt_idx]
            boxes_abs_b = gt_bboxes_abs[b][valid][tgt_idx]
            boxes_norm_b = boxes_abs_b / scale_tensor
            boxes_grid_b = boxes_abs_b / stride_tensor_boxes[src_idx]

            cls_targets[b, src_idx, labels_b] = 1.0
            fg_mask[b, src_idx] = True

            matched_pred_batches.append(torch.full_like(src_idx, b))
            matched_pred_indices.append(src_idx)
            matched_target_boxes_norm.append(boxes_norm_b)
            matched_target_boxes_grid.append(boxes_grid_b)
            matched_target_labels.append(labels_b)

        num_pos_logits = cls_targets.sum().clamp(min=1.0)
        num_neg_logits = (cls_targets.numel() - cls_targets.sum()).clamp(min=1.0)
        neg_weight = self.negative_to_positive_ratio * num_pos_logits / num_neg_logits
        cls_weights = torch.where(
            cls_targets > 0,
            torch.ones_like(cls_targets),
            torch.full_like(cls_targets, neg_weight),
        )
        loss_cls = (self.bce(pred_scores, cls_targets.to(dtype)) * cls_weights).sum() / cls_weights.sum().clamp(min=1.0)
        loss_box = pred_bboxes_norm.sum() * 0.0
        loss_dfl = pred_bboxes_norm.sum() * 0.0
        if matched_pred_indices:
            batch_idx = torch.cat(matched_pred_batches)
            src_idx = torch.cat(matched_pred_indices)
            target_boxes_norm = torch.cat(matched_target_boxes_norm, dim=0)
            target_boxes_grid = torch.cat(matched_target_boxes_grid, dim=0)
            src_boxes_norm = pred_bboxes_norm[batch_idx, src_idx]

            iou = bbox_iou(src_boxes_norm, target_boxes_norm, xywh=False, CIoU=True).squeeze(-1)
            loss_box = (1.0 - iou).mean()

            if self.dfl_loss:
                target_ltrb = bbox2dist(anchor_points[src_idx], target_boxes_grid, self.reg_max - 1)
                loss_dfl = self.dfl_loss(
                    pred_distri[batch_idx, src_idx].view(-1, self.reg_max),
                    target_ltrb,
                ).mean()

        loss = torch.stack((
            loss_box * self.lambda_box,
            loss_cls * self.lambda_cls,
            loss_dfl * self.lambda_dfl,
        ))

        debug_data = []
        for b in range(batch_size):
            fg_inds = fg_mask[b]
            debug_data.append({
                "gt_boxes": gt_bboxes_abs[b].detach().cpu(),
                "task_selected_pred_boxes_abs": pred_bboxes_abs[b][fg_inds].detach().cpu(),
                "task_selected_anchor_points_abs": (anchor_points[fg_inds] * stride_tensor[fg_inds]).detach().cpu(),
                "pred_bboxes_abs": pred_bboxes_abs[b].detach().cpu()
            })

        return loss.sum() * batch_size, [
            loss[0].item(),
            loss[1].item(),
            loss[2].item()
        ], debug_data


class SNRYOLODetectionLoss:
    def __init__(self,
                 reg_max=16,
                 lambda_box=7.5,
                 lambda_dfl=1.5,
                 lambda_cls=0.5,
                 num_classes=80,
                 strides=[8, 16, 32],
                 tal_topk=10,
                 minimum_possible_candidates=None,
                 snr_min=-2,
                 device='cpu'):

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
        nl, ne = targets.shape  # expected: ne = 7
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0].long()  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))  # rescale boxes
        return out[..., :5], out[..., 5:]  # return targets, snr


    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, pred_distri, pred_scores, batch, feats):
        loss = torch.zeros(3, device=self.device)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = _image_size_from_feats(feats[0], self.strides[0], self.device, dtype)
        anchor_points, stride_tensor = make_anchors(feats, self.strides, 0.5)
        stride_tensor_boxes = _stride_tensor_xyxy(stride_tensor)

        targets = torch.cat((batch["batch_idx"].view(-1, 1),
                            batch["cls"].view(-1, 1),
                            batch["bboxes"],
                            batch["snr"]), dim=1)  # shape: (N, 7)

        targets, snrs = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        snr_weight = (snrs - self.snr_min).clamp(min=0.0)  # shape: (B, N, 1)

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

        # snrs: shape (B, N, 1)
        snrs_flat = snrs.squeeze(-1)  # (B, N)

        assigned_snrs = torch.gather(snrs_flat, dim=1, index=matched_gt_inds.clamp(min=0))  # (B, A)
        assigned_snrs = (assigned_snrs - self.snr_min).clamp(min=0.0)
        assigned_snrs = assigned_snrs.unsqueeze(-1)  # (B, A, 1)


        loss_cls = self.bce(pred_scores, target_scores.to(dtype))  # (B, A, C)
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
                "pred_bboxes_abs": (pred_bboxes[b] * stride_tensor_boxes).detach().cpu()
            })

        return loss.sum() * batch_size, [
            loss[0].item(),
            loss[1].item(),
            loss[2].item()
        ], debug_data
