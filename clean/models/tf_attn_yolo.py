import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..nn.convs import Conv, DWConv
from ..nn.blocks import C3k2, SPPF, C2PSA, DFL, TFSepBlock  # TFSepBlock ajouté
from .base import BaseModel
from ..utils.loss import YOLODetectionLoss, SNRYOLODetectionLoss
from .Head.detect import Detect
from .anisotropic_utils import build_anisotropic_standard_plan


class TF_Attn_Yolo(BaseModel):
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
    ):
        super().__init__(device=device, output_dir=output_dir)
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.debug = debug
        self.anisotropic = bool(anisotropic)
        self.p3_size = tuple(p3_size)
        self.input_hw = tuple(input_hw) if input_hw is not None else None

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
        self.conv2 = Conv(c1, c2, k=3, s=stage_strides[1])                        # P2/4

        # Remplacement C3k2(c2 -> c3) par: adapt 1x1 (c2->c3) + TFSepBlock(ch=c3)
        self.c3_1_in = Conv(c2, c3, k=1, s=1)
        self.c3_1 = TFSepBlock(ch=c3, n=1, residual=True, mode="parallel")

        self.conv3 = Conv(c3, c3, k=3, s=stage_strides[2])                        # P3/8
        # Remplacement C3k2(c3 -> c3)
        self.c3_2 = TFSepBlock(ch=c3, n=1, residual=True, mode="parallel")

        self.conv4 = Conv(c3, c4, k=3, s=2)                        # P4/16
        # Remplacement C3k2(c4 -> c4)
        self.c3_3 = TFSepBlock(ch=c4, n=1, residual=True, mode="parallel")

        self.conv5 = Conv(c4, c5, k=3, s=2)                        # P5/32
        # Remplacement C3k2(c5 -> c5)
        self.c3_4 = TFSepBlock(ch=c5, n=1, residual=True, mode="parallel")

        self.sppf = SPPF(c5, c5)
        self.attn = C2PSA(c1=c5, c2=c5, n=2, e=0.5)

        # ---------------- Head (FPN) ----------------
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # head_c3_1: C3k2(c5 + c4 -> c4)  ==> adapt + TF-Sep(c4)
        self.head_c3_1_in = Conv(c5 + c4, c4, k=1, s=1)
        self.head_c3_1 = TFSepBlock(ch=c4, n=1, residual=True, mode="parallel")

        # head_c3_2: C3k2(c4 + c3 -> c3)  ==> adapt + TF-Sep(c3)
        self.head_c3_2_in = Conv(c4 + c3, c3, k=1, s=1)
        self.head_c3_2 = TFSepBlock(ch=c3, n=1, residual=True, mode="parallel")

        self.down_p3   = Conv(c3, c3, k=3, s=2)

        # head_c3_3: C3k2(c3 + c4 -> c4)  ==> adapt + TF-Sep(c4)
        self.head_c3_3_in = Conv(c3 + c4, c4, k=1, s=1)
        self.head_c3_3 = TFSepBlock(ch=c4, n=1, residual=True, mode="parallel")

        self.down_p4   = Conv(c4, c4, k=3, s=2)

        # head_c3_4: C3k2(c4 + c5 -> c5)  ==> adapt + TF-Sep(c5)
        self.head_c3_4_in = Conv(c4 + c5, c5, k=1, s=1)
        self.head_c3_4 = TFSepBlock(ch=c5, n=1, residual=True, mode="parallel")

        # ---------------- Detect ----------------
        self.detect = Detect(
            in_channels=[c3, c4, c5],
            num_classes=self.num_classes,
            reg_max=self.reg_max,
            strides=self.strides
        )
        self.detect.bias_init(image_size=self.input_hw if self.input_hw is not None else 1024)

        # Loss
        self.criterion = YOLODetectionLoss(
            num_classes=num_classes,
            strides=self.strides,
            reg_max=self.reg_max,
            device=self.device,
        )
        # self.criterion = SNRYOLODetectionLoss(...)

        self.to(self.device)

    def _prepare_input(self, x):
        if self.pre_backbone_resize_hw is None:
            return x
        target_hw = tuple(self.pre_backbone_resize_hw)
        if tuple(x.shape[-2:]) == target_hw:
            return x
        return F.interpolate(x, size=target_hw, mode="nearest")

    def forward(self, x):
        x = self._prepare_input(x)
        # Backbone
        x = self.conv1(x)
        self.debug_shape("conv1", x)

        x = self.conv2(x)
        self.debug_shape("conv2", x)

        x = self.c3_1_in(x)
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

        p4_feat = self.head_c3_1_in(p4_feat)
        p4_out = self.head_c3_1(p4_feat)
        self.debug_shape("p4_out", p4_out)

        p4_up = self.upsample(p4_out)
        self.debug_shape("p4_up", p4_up)

        p3_feat = torch.cat([p4_up, f3], dim=1)
        self.debug_shape("p3_feat", p3_feat)

        p3_feat = self.head_c3_2_in(p3_feat)
        p3_out = self.head_c3_2(p3_feat)
        self.debug_shape("p3_out", p3_out)

        # Head: medium
        p3_down = self.down_p3(p3_out)
        self.debug_shape("p3_down", p3_down)

        pm_feat = torch.cat([p3_down, p4_out], dim=1)
        self.debug_shape("pm_feat", pm_feat)

        pm_feat = self.head_c3_3_in(pm_feat)
        p4_out2 = self.head_c3_3(pm_feat)
        self.debug_shape("p4_out2", p4_out2)

        # Head: large
        p4_down = self.down_p4(p4_out2)
        self.debug_shape("p4_down", p4_down)

        pl_feat = torch.cat([p4_down, f5], dim=1)
        self.debug_shape("pl_feat", pl_feat)

        pl_feat = self.head_c3_4_in(pl_feat)
        p5_out = self.head_c3_4(pl_feat)
        self.debug_shape("p5_out", p5_out)

        # Detect
        outputs = self.detect(p3_out, p4_out2, p5_out)
        return outputs

    def debug_shape(self, name, tensor):
        if self.debug:
            print(f"[DEBUG] {name:<20} shape = {tuple(tensor.shape)}")
