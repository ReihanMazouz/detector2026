import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from ..divers import xywh2xyxy
from ..metrics import bbox_iou
from .common import image_size_from_feats, stride_tensor_xyxy
from .dfl import DFLoss


class YOLOOne2OneHungarianLoss:
    def __init__(
        self,
        reg_max=16,
        lambda_box=7.5,
        lambda_dfl=1.5,
        lambda_cls=0.5,
        cost_class=2.0,
        cost_bbox=5.0,
        cost_giou=2.0,
        negative_to_positive_ratio=1.0,
        cls_loss_type="varifocal",
        vfl_alpha=0.75,
        vfl_gamma=2.0,
        iou_class_target=True,
        num_classes=80,
        strides=[8, 16, 32],
        device="cpu",
    ):
        self.lambda_box = lambda_box
        self.lambda_dfl = lambda_dfl
        self.lambda_cls = lambda_cls
        self.cost_class = float(cost_class)
        self.cost_bbox = float(cost_bbox)
        self.cost_giou = float(cost_giou)
        self.negative_to_positive_ratio = float(negative_to_positive_ratio)
        self.cls_loss_type = str(cls_loss_type).lower()
        self.vfl_alpha = float(vfl_alpha)
        self.vfl_gamma = float(vfl_gamma)
        self.iou_class_target = bool(iou_class_target)
        self.nc = num_classes
        self.strides = strides
        self.device = device
        self.reg_max = reg_max
        self.use_dfl = reg_max > 1
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=device)
        if self.cls_loss_type not in {"bce", "varifocal"}:
            raise ValueError("cls_loss_type must be 'bce' or 'varifocal'.")

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
        from ..tal import dist2bbox

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
        from ..tal import bbox2dist, make_anchors

        dtype = pred_scores.dtype
        batch_size, num_anchors = pred_scores.shape[:2]
        imgsz = image_size_from_feats(feats[0], self.strides[0], self.device, dtype)
        anchor_points, stride_tensor = make_anchors(feats, self.strides, 0.5)
        stride_tensor_boxes = stride_tensor_xyxy(stride_tensor)

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

            if self.iou_class_target:
                target_quality = bbox_iou(
                    pred_bboxes_norm[b, src_idx].detach(),
                    boxes_norm_b,
                    xywh=False,
                ).squeeze(-1).clamp_(0.0, 1.0)
            else:
                target_quality = torch.ones_like(labels_b, dtype=dtype)
            cls_targets[b, src_idx, labels_b] = target_quality.to(cls_targets.dtype)
            fg_mask[b, src_idx] = True

            matched_pred_batches.append(torch.full_like(src_idx, b))
            matched_pred_indices.append(src_idx)
            matched_target_boxes_norm.append(boxes_norm_b)
            matched_target_boxes_grid.append(boxes_grid_b)
            matched_target_labels.append(labels_b)

        num_pos_logits = fg_mask.sum().to(dtype).clamp(min=1.0)
        if self.cls_loss_type == "varifocal":
            pred_prob = pred_scores.detach().sigmoid()
            cls_weights = torch.where(
                cls_targets > 0,
                cls_targets,
                self.vfl_alpha * pred_prob.pow(self.vfl_gamma) * self.negative_to_positive_ratio,
            )
            loss_cls = (self.bce(pred_scores, cls_targets.to(dtype)) * cls_weights).sum() / num_pos_logits
        else:
            num_target_score = cls_targets.sum().clamp(min=1.0)
            num_neg_logits = (cls_targets.numel() - (cls_targets > 0).sum()).to(dtype).clamp(min=1.0)
            neg_weight = self.negative_to_positive_ratio * num_target_score / num_neg_logits
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
                "pred_bboxes_abs": pred_bboxes_abs[b].detach().cpu(),
            })

        return loss.sum() * batch_size, [
            loss[0].item(),
            loss[1].item(),
            loss[2].item(),
        ], debug_data
