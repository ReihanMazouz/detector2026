import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..nn.convs import Conv, DWConv
from ..nn.blocks import C3k2, SPPF, C2PSA, DFL
from .base import BaseModel
from ..utils.loss import YOLODetectionLoss, YOLOOne2OneHungarianLoss
from ..utils.tal import make_anchors, dist2bbox
from .Head.detect import Detect, One2OneDetect
from .anisotropic_utils import build_anisotropic_standard_plan


class TransformerChin(nn.Module):
    """Light transformer block between the YOLO neck and the detection head."""

    LEVEL_TO_INDEX = {"p3": 0, "p4": 1, "p5": 2}

    def __init__(
        self,
        in_channels,
        levels=("p4", "p5"),
        d_model=128,
        num_heads=4,
        num_layers=1,
        ffn_ratio=2.0,
        dropout=0.0,
        residual_scale=0.0,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.levels = tuple(str(level).lower() for level in levels)
        self.level_indices = []
        for level in self.levels:
            if level not in self.LEVEL_TO_INDEX:
                raise ValueError("TransformerChin levels must be chosen from 'p3', 'p4', 'p5'.")
            self.level_indices.append(self.LEVEL_TO_INDEX[level])

        selected_channels = [in_channels[index] for index in self.level_indices]
        self.proj_in = nn.ModuleList(nn.Conv2d(channels, d_model, kernel_size=1) for channels in selected_channels)
        self.proj_out = nn.ModuleList(nn.Conv2d(d_model, channels, kernel_size=1) for channels in selected_channels)
        self.pos_proj = nn.Linear(2, d_model)
        self.level_embed = nn.Parameter(torch.zeros(len(self.level_indices), d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=max(d_model, int(d_model * ffn_ratio)),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def _position_tokens(self, height, width, level_index, device, dtype):
        y = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack((xx, yy), dim=-1).reshape(1, height * width, 2)
        pos = self.pos_proj(coords)
        return pos + self.level_embed[level_index].to(device=device, dtype=dtype).view(1, 1, -1)

    def forward(self, features):
        outputs = list(features)
        tokens = []
        shapes = []

        for local_index, feature_index in enumerate(self.level_indices):
            feat = features[feature_index]
            projected = self.proj_in[local_index](feat)
            batch_size, channels, height, width = projected.shape
            token = projected.flatten(2).transpose(1, 2)
            token = token + self._position_tokens(height, width, local_index, token.device, token.dtype)
            tokens.append(token)
            shapes.append((batch_size, channels, height, width))

        encoded = self.encoder(torch.cat(tokens, dim=1))
        start = 0
        for local_index, feature_index in enumerate(self.level_indices):
            batch_size, channels, height, width = shapes[local_index]
            end = start + height * width
            encoded_level = encoded[:, start:end].transpose(1, 2).reshape(batch_size, channels, height, width)
            delta = self.proj_out[local_index](encoded_level)
            outputs[feature_index] = features[feature_index] + self.residual_scale * delta
            start = end

        return tuple(outputs)


class YOLOv11(BaseModel):
    def __init__(
        self,
        output_dir,
        num_classes=80,
        strides=None,
        reg_max=16,
        device="cuda:0",
        input_canals=1,
        width_mult=0.25,
        debug=False,
        anisotropic=False,
        p3_size=(64, 64),
        input_hw=None,
        use_transformer_chin=False,
        chin_levels=("p4", "p5"),
        chin_d_model=128,
        chin_num_heads=4,
        chin_num_layers=1,
        chin_ffn_ratio=2.0,
        chin_dropout=0.0,
        chin_residual_scale=0.0,
    ):
        super().__init__(device=device, output_dir=output_dir)
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.debug = debug
        self.anisotropic = bool(anisotropic)
        self.p3_size = tuple(p3_size)
        self.input_hw = tuple(input_hw) if input_hw is not None else None
        self.use_transformer_chin = bool(use_transformer_chin)

        if strides is None:
            strides = [8, 16, 32]

        if self.anisotropic:
            if self.input_hw is None:
                raise ValueError("input_hw must be provided when anisotropic=True.")
            anisotropic_plan = build_anisotropic_standard_plan(self.input_hw, self.p3_size)
            self.pre_backbone_resize_hw = anisotropic_plan["pre_resize_hw"]
            stage_strides = anisotropic_plan["stage_strides"]
            self.strides = anisotropic_plan["detect_strides"]
        else:
            self.pre_backbone_resize_hw = None
            stage_strides = [(2, 2), (2, 2), (2, 2)]
            self.strides = strides

        # scaled channel counts
        c1 = int(64 * width_mult)
        c2 = int(128 * width_mult)
        c3 = int(256 * width_mult)
        c4 = int(512 * width_mult)
        c5 = int(1024 * width_mult)

        # ---------------- Backbone ----------------
        self.conv1 = Conv(input_canals, c1, k=3, s=stage_strides[0])             # P1/2
        self.conv2 = Conv(c1, c2, k=3, s=stage_strides[1])            # P2/4
        self.c3_1 = C3k2(c2, c3, shortcut=False)
        self.conv3 = Conv(c3, c3, k=3, s=stage_strides[2])            # P3/8
        self.c3_2 = C3k2(c3, c3, shortcut=False)
        self.conv4 = Conv(c3, c4, k=3, s=2)            # P4/16
        self.c3_3 = C3k2(c4, c4, shortcut=True)
        self.conv5 = Conv(c4, c5, k=3, s=2)            # P5/32
        self.c3_4 = C3k2(c5, c5, shortcut=True)
        self.sppf = SPPF(c5, c5)
        self.attn = C2PSA(c1=c5, c2=c5, n=2, e=0.5)

        # ---------------- Head (FPN) ----------------
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.head_c3_1 = C3k2(c5 + c4, c4, shortcut=False)  # small P4
        self.head_c3_2 = C3k2(c4 + c3, c3, shortcut=False)  # small P3
        self.down_p3   = Conv(c3, c3, k=3, s=2)
        self.head_c3_3 = C3k2(c3 + c4, c4, shortcut=False)  # medium P4
        self.down_p4   = Conv(c4, c4, k=3, s=2)
        self.head_c3_4 = C3k2(c4 + c5, c5, shortcut=True)   # large P5

        # ---------------- Detect ----------------
        # Using custom Detect with separated branches and DFL
        self.detect = Detect(
            in_channels=[c3, c4, c5],
            num_classes=self.num_classes,
            reg_max=self.reg_max, 
            strides=self.strides
        )

        self.detect.bias_init(image_size=self.input_hw if self.input_hw is not None else 1024)
        self.detect_one2one = One2OneDetect(self.detect)
        self.transformer_chin = (
            TransformerChin(
                in_channels=[c3, c4, c5],
                levels=chin_levels,
                d_model=chin_d_model,
                num_heads=chin_num_heads,
                num_layers=chin_num_layers,
                ffn_ratio=chin_ffn_ratio,
                dropout=chin_dropout,
                residual_scale=chin_residual_scale,
            )
            if self.use_transformer_chin
            else None
        )
        self.active_head = "one2many"

        # Loss
        self.criterion = YOLODetectionLoss(num_classes=num_classes, strides = self.strides, reg_max=self.reg_max, device=self.device,)
        self.criterion_one2many = self.criterion
        self.criterion_one2one = YOLODetectionLoss(
            num_classes=num_classes,
            strides=self.strides,
            reg_max=self.reg_max,
            tal_topk=1,
            minimum_possible_candidates=7,
            device=self.device,
        )
        # self.criterion = SNRYOLODetectionLoss(num_classes=num_classes, strides = self.strides, reg_max=self.reg_max, device=self.device,)

        self.to(self.device)

    def _prepare_input(self, x):
        if self.pre_backbone_resize_hw is None:
            return x
        target_hw = tuple(self.pre_backbone_resize_hw)
        if tuple(x.shape[-2:]) == target_hw:
            return x
        return F.interpolate(x, size=target_hw, mode="nearest")

    def forward_features(self, x):
        x = self._prepare_input(x)
        # Backbone
        x = self.conv1(x)
        self.debug_shape("conv1", x)

        x = self.conv2(x)
        self.debug_shape("conv2", x)

        f2 = self.c3_1(x)
        self.debug_shape("c3_1 (f2)", f2)

        x = self.conv3(f2)
        self.debug_shape("conv3", x)

        f3 = self.c3_2(x)
        self.debug_shape("c3_2 (f3)", f3)

        x = self.conv4(f3)
        self.debug_shape("conv4", x)

        f4 = self.c3_3(x)
        self.debug_shape("c3_3 (f4)", f4)

        x = self.conv5(f4)
        self.debug_shape("conv5", x)

        x = self.c3_4(x)
        x = self.sppf(x)
        f5 = self.attn(x)
        self.debug_shape("attn (f5)", f5)

        # Head: small
        p5_up = self.upsample(f5)
        self.debug_shape("p5_up", p5_up)

        p4_feat = torch.cat([p5_up, f4], dim=1)
        self.debug_shape("p4_feat", p4_feat)

        p4_out = self.head_c3_1(p4_feat)
        self.debug_shape("p4_out", p4_out)

        p4_up = self.upsample(p4_out)
        self.debug_shape("p4_up", p4_up)

        p3_feat = torch.cat([p4_up, f3], dim=1)
        self.debug_shape("p3_feat", p3_feat)

        p3_out = self.head_c3_2(p3_feat)
        self.debug_shape("p3_out", p3_out)

        # Head: medium
        p3_down = self.down_p3(p3_out)
        self.debug_shape("p3_down", p3_down)

        pm_feat = torch.cat([p3_down, p4_out], dim=1)
        self.debug_shape("pm_feat", pm_feat)

        p4_out2 = self.head_c3_3(pm_feat)
        self.debug_shape("p4_out2", p4_out2)

        # Head: large
        p4_down = self.down_p4(p4_out2)
        self.debug_shape("p4_down", p4_down)

        pl_feat = torch.cat([p4_down, f5], dim=1)
        self.debug_shape("pl_feat", pl_feat)

        p5_out = self.head_c3_4(pl_feat)
        self.debug_shape("p5_out", p5_out)

        return p3_out, p4_out2, p5_out

    def forward(self, x, head=None):
        p3_out, p4_out2, p5_out = self.forward_features(x)
        head = self.active_head if head is None else head

        if head == "one2one":
            if self.transformer_chin is not None:
                p3_out, p4_out2, p5_out = self.transformer_chin((p3_out, p4_out2, p5_out))
            return self.detect_one2one(p3_out, p4_out2, p5_out)
        if head != "one2many":
            raise ValueError("head must be 'one2many' or 'one2one'.")

        return self.detect(p3_out, p4_out2, p5_out)

    def training_forward(self, imgs):
        return self(imgs, head=self.active_head)

    def get_training_criterion(self):
        if self.active_head == "one2one":
            return self.criterion_one2one
        return self.criterion

    def train_one2one_head_only(self, minimum_possible_candidates=7, sync_from_one2many=True, loss_type="tal",
                                negative_to_positive_ratio=1.0):
        """Freeze the existing model and train only the copied one2one detect head."""
        if sync_from_one2many:
            self.sync_one2one_from_one2many()
        self.active_head = "one2one"
        loss_type = str(loss_type).lower()
        if loss_type == "tal":
            self.criterion_one2one = YOLODetectionLoss(
                num_classes=self.num_classes,
                strides=self.strides,
                reg_max=self.reg_max,
                tal_topk=1,
                minimum_possible_candidates=minimum_possible_candidates,
                device=self.device,
            )
        elif loss_type == "hungarian":
            self.criterion_one2one = YOLOOne2OneHungarianLoss(
                num_classes=self.num_classes,
                strides=self.strides,
                reg_max=self.reg_max,
                negative_to_positive_ratio=negative_to_positive_ratio,
                device=self.device,
            )
        else:
            raise ValueError("loss_type must be 'tal' or 'hungarian'.")
        self.criterion = self.criterion_one2one
        for param in self.parameters():
            param.requires_grad = False
        if self.transformer_chin is not None:
            for param in self.transformer_chin.parameters():
                param.requires_grad = True
        for param in self.detect_one2one.parameters():
            param.requires_grad = True
        self.detect.eval()
        if self.transformer_chin is not None:
            self.transformer_chin.train()
        self.detect_one2one.train()

    def use_one2many_head(self):
        self.active_head = "one2many"
        self.criterion = self.criterion_one2many

    def use_one2one_head(self):
        self.active_head = "one2one"

    def train(self, mode=True):
        super().train(mode)
        if self.active_head == "one2one":
            self._set_frozen_parts_eval()
            if self.transformer_chin is not None:
                self.transformer_chin.train(mode)
            self.detect_one2one.train(mode)
        return self

    def postprocess(self, dist_out, cls_out, feats, conf_thres=0.1, iou_thres=0.1, iou_same_box=0.9, without_nms=None, max_det=300):
        without_nms = self.active_head == "one2one" if without_nms is None else bool(without_nms)
        if not without_nms:
            return super().postprocess(
                dist_out,
                cls_out,
                feats,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                iou_same_box=iou_same_box,
            )

        pred_dist = torch.cat([x.flatten(2) for x in dist_out], dim=2).permute(0, 2, 1)
        pred_cls = torch.cat([x.flatten(2) for x in cls_out], dim=2).permute(0, 2, 1)
        batch_size, num_anchors, _ = pred_dist.shape

        anchor_points, stride_tensor = make_anchors(feats, self.strides)
        anchor_points = anchor_points.to(pred_dist.device)
        stride_tensor = stride_tensor.to(pred_dist.device)
        stride_tensor_boxes = torch.cat([stride_tensor, stride_tensor], dim=1)

        if self.reg_max > 1:
            proj = torch.arange(self.reg_max, dtype=pred_dist.dtype, device=pred_dist.device)
            pred_ltrb = pred_dist.view(batch_size, num_anchors, 4, self.reg_max).softmax(3).matmul(proj)
        else:
            pred_ltrb = pred_dist

        pred_bboxes = dist2bbox(pred_ltrb, anchor_points, xywh=False) * stride_tensor_boxes
        cls_scores = pred_cls.sigmoid()
        scores, labels = cls_scores.max(dim=-1)

        results = []
        for boxes_i, scores_i, labels_i in zip(pred_bboxes, scores, labels):
            keep = scores_i > conf_thres
            detections = torch.cat(
                [boxes_i, scores_i.unsqueeze(1), labels_i.to(dtype=boxes_i.dtype).unsqueeze(1)],
                dim=1,
            )[keep]
            if detections.shape[0] > max_det:
                detections = detections[detections[:, 4].argsort(descending=True)[:max_det]]
            results.append(detections)
        return results

    def _set_frozen_parts_eval(self):
        for name, module in self.named_children():
            if name not in {"detect_one2one", "transformer_chin"}:
                module.eval()

    def sync_one2one_from_one2many(self):
        self.detect_one2one.cv_dist.load_state_dict(self.detect.cv_dist.state_dict())
        self.detect_one2one.cv_clsobj.load_state_dict(self.detect.cv_clsobj.state_dict())
        self.detect_one2one.dfl.load_state_dict(self.detect.dfl.state_dict())
    
    def debug_shape(self, name, tensor):
        if self.debug:
            print(f"[DEBUG] {name:<20} shape = {tuple(tensor.shape)}")
