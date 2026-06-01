from __future__ import annotations

import torch
import torch.nn as nn

from ...nn.blocks import C3k2
from ...nn.convs import Conv
from ...utils.loss import YOLODetectionLoss
from ..Head.detect import Detect, One2OneDetect
from ..anisotropic_utils import stride_hw_to_xy
from ..yolov11 import YOLOv11


def _mul_stride(a, b):
    ax, ay = stride_hw_to_xy(a)
    bx, by = stride_hw_to_xy(b)
    return (ay * by, ax * bx)


class YOLOv11P2P5Neck(YOLOv11):
    """YOLOv11 with a larger P2-P5 neck and four detection heads.

    The backbone exposes P2, P3, P4 and P5. The neck performs a top-down path
    down to P2, then a bottom-up PAN path back to P5, returning P2', P3',
    P4' and P5'. This adds a stride-4 detection level for small objects.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        width_mult = kwargs.get("width_mult", 0.25)
        input_hw = kwargs.get("input_hw", None)
        anisotropic = bool(kwargs.get("anisotropic", False))
        strides = kwargs.get("strides", None)

        c2 = int(128 * width_mult)
        c3 = int(256 * width_mult)
        c4 = int(512 * width_mult)
        c5 = int(1024 * width_mult)

        # Top-down FPN: P5 -> P4 -> P3 -> P2.
        self.p5_to_p4 = C3k2(c5 + c4, c4, shortcut=False)
        self.p4_to_p3 = C3k2(c4 + c3, c3, shortcut=False)
        self.p3_to_p2 = C3k2(c3 + c3, c2, shortcut=False)

        # Bottom-up PAN: P2 -> P3 -> P4 -> P5.
        self.down_p2 = Conv(c2, c3, k=3, s=2)
        self.p2_to_p3 = C3k2(c3 + c3, c3, shortcut=False)
        self.down_p3_p2p5 = Conv(c3, c4, k=3, s=2)
        self.p3_to_p4 = C3k2(c4 + c4, c4, shortcut=False)
        self.down_p4_p2p5 = Conv(c4, c5, k=3, s=2)
        self.p4_to_p5 = C3k2(c5 + c5, c5, shortcut=True)

        if anisotropic:
            stage_strides = self._resolve_stage_strides_for_p2(input_hw)
            p2_stride = _mul_stride(stage_strides[0], stage_strides[1])
            p3_stride = _mul_stride(p2_stride, stage_strides[2])
            p4_stride = _mul_stride(p3_stride, (2, 2))
            p5_stride = _mul_stride(p4_stride, (2, 2))
            self.strides = [p2_stride, p3_stride, p4_stride, p5_stride]
        else:
            self.strides = strides if strides is not None else [4, 8, 16, 32]

        self.detect = Detect(
            in_channels=[c2, c3, c4, c5],
            num_classes=self.num_classes,
            reg_max=self.reg_max,
            strides=self.strides,
        )
        self.detect.bias_init(image_size=self.input_hw if self.input_hw is not None else 1024)
        self.detect_one2one = One2OneDetect(self.detect)
        self.criterion = YOLODetectionLoss(
            num_classes=self.num_classes,
            strides=self.strides,
            reg_max=self.reg_max,
            device=self.device,
        )
        self.criterion_one2many = self.criterion
        self.criterion_one2one = YOLODetectionLoss(
            num_classes=self.num_classes,
            strides=self.strides,
            reg_max=self.reg_max,
            tal_topk=1,
            minimum_possible_candidates=7,
            device=self.device,
        )
        self.to(self.device)

    @staticmethod
    def _resolve_stage_strides_for_p2(input_hw):
        from ..anisotropic_utils import build_anisotropic_standard_plan

        if input_hw is None:
            raise ValueError("input_hw must be provided when anisotropic=True.")
        plan = build_anisotropic_standard_plan(tuple(input_hw), (64, 64))
        return plan["stage_strides"]

    def forward_features(self, x):
        x = self._prepare_input(x)

        x = self.conv1(x)
        self.debug_shape("conv1", x)

        x = self.conv2(x)
        self.debug_shape("conv2", x)

        f2 = self.c3_1(x)
        self.debug_shape("c3_1 (f2/P2)", f2)

        x = self.conv3(f2)
        self.debug_shape("conv3", x)

        f3 = self.c3_2(x)
        self.debug_shape("c3_2 (f3/P3)", f3)

        x = self.conv4(f3)
        self.debug_shape("conv4", x)

        f4 = self.c3_3(x)
        self.debug_shape("c3_3 (f4/P4)", f4)

        x = self.conv5(f4)
        self.debug_shape("conv5", x)

        x = self.c3_4(x)
        x = self.sppf(x)
        f5 = self.attn(x)
        self.debug_shape("attn (f5/P5)", f5)

        p4_td = self.p5_to_p4(torch.cat([self.upsample(f5), f4], dim=1))
        self.debug_shape("p4_td", p4_td)

        p3_td = self.p4_to_p3(torch.cat([self.upsample(p4_td), f3], dim=1))
        self.debug_shape("p3_td", p3_td)

        p2_out = self.p3_to_p2(torch.cat([self.upsample(p3_td), f2], dim=1))
        self.debug_shape("p2_out", p2_out)

        p3_out = self.p2_to_p3(torch.cat([self.down_p2(p2_out), p3_td], dim=1))
        self.debug_shape("p3_out", p3_out)

        p4_out = self.p3_to_p4(torch.cat([self.down_p3_p2p5(p3_out), p4_td], dim=1))
        self.debug_shape("p4_out", p4_out)

        p5_out = self.p4_to_p5(torch.cat([self.down_p4_p2p5(p4_out), f5], dim=1))
        self.debug_shape("p5_out", p5_out)

        return p2_out, p3_out, p4_out, p5_out

    def forward(self, x, head=None):
        p2_out, p3_out, p4_out, p5_out = self.forward_features(x)
        head = self.active_head if head is None else head

        if head == "one2one":
            return self.detect_one2one(p2_out, p3_out, p4_out, p5_out)
        if head != "one2many":
            raise ValueError("head must be 'one2many' or 'one2one'.")
        return self.detect(p2_out, p3_out, p4_out, p5_out)
