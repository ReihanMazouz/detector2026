# yolo_perso/models/YOLOv8.py
import torch, torch.nn as nn
import torch.nn.functional as F
from ..nn.convs import Conv
from ..nn.blocks import C2f, SPPF
from ..utils.loss import YOLODetectionLoss
from .Head.detect import Detect
from .base import BaseModel
import math
from .anisotropic_utils import build_anisotropic_standard_plan

class YOLOv8(BaseModel):
    """
    Implémentation 'from-scratch' fidèle au YAML Ultralytics YOLOv8 (P3-P5).
    Les couples (depth_mult, width_mult) viennent du fichier .yaml :
        n : (0.33, 0.25)   s : (0.33, 0.50)
        m : (0.67, 0.75)   l : (1.00, 1.00)   x : (1.00, 1.25)
    """
    # --------------------------------------------------------------------- #
    def __init__(self,
                 output_dir: str,
                 num_classes: int = 80,
                 width_mult: float = 0.25,
                 depth_mult: float = 0.33,
                 in_ch: int = 1,
                 reg_max: int = 16,
                 strides=None,
                 device: str = "cuda:0",
                 debug: bool = False,
                 anisotropic: bool = False,
                 p3_size=(64, 64),
                 input_hw=None):
        super().__init__(device=device, output_dir=output_dir)
        self.nc, self.debug = num_classes, debug
        self.wm, self.dm = width_mult, depth_mult
        self.num_classes = num_classes
        self.reg_max = reg_max
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

        # ---------------- helper pour canaux / répétitions ---------------- #
        def c(x):  # applique width_mult et limite à 1024 (comme YAML)
            return min(int(math.ceil(x * self.wm)), 1024)

        def r(x):  # applique depth_mult (>=1)
            return max(int(round(x * self.dm)), 1)

        # -------------- canaux de base (64-1024) -------------- #
        c1, c2, c3, c4, c5 = map(c, (64, 128, 256, 512, 1024))

        # --------------------------- Backbone ---------------------------- #
        self.b0 = Conv(in_ch, c1, 3, stage_strides[0])                   # P1/2
        self.b1 = Conv(c1,  c2, 3, stage_strides[1])                    # P2/4
        self.b2 = C2f(c2,  c2, n=r(3), shortcut=True)    # 3× C2f

        self.b3 = Conv(c2,  c3, 3, stage_strides[2])                    # P3/8
        self.b4 = C2f(c3,  c3, n=r(6), shortcut=True)

        self.b5 = Conv(c3,  c4, 3, 2)                    # P4/16
        self.b6 = C2f(c4,  c4, n=r(6), shortcut=True)

        self.b7 = Conv(c4,  c5, 3, 2)                    # P5/32
        self.b8 = C2f(c5,  c5, n=r(3), shortcut=True)
        self.b9 = SPPF(c5, c5, k=5)                      # contexte global

        # ------------------------------ FPN / PAN ------------------------- #
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        # P5↑ + P4
        self.h0 = C2f(c5 + c4, c4, n=r(3), shortcut=True)   # P4/16
        # P4↑ + P3
        self.h1 = C2f(c4 + c3, c3, n=r(3), shortcut=True)   # P3/8 (small)

        # P3↓ + P4
        self.down3 = Conv(c3, c3, 3, 2)
        self.h2    = C2f(c3 + c4, c4, n=r(3), shortcut=True)  # P4/16 (medium)

        # P4↓ + P5
        self.down4 = Conv(c4, c4, 3, 2)
        self.h3    = C2f(c4 + c5, c5, n=r(3), shortcut=True)  # P5/32 (large)

        # ------------------------------ Detect --------------------------- #
        self.detect = Detect(
            in_channels=[c3, c4, c5],
            num_classes=self.nc,
            reg_max=reg_max,
            strides=self.strides)
        self.detect.bias_init(image_size=self.input_hw if self.input_hw is not None else 1024)

        self.criterion = YOLODetectionLoss(
            num_classes=num_classes,
            strides=self.strides,
            reg_max=reg_max,
            device=self.device)

        self.to(self.device)

    def _prepare_input(self, x):
        if self.pre_backbone_resize_hw is None:
            return x
        target_hw = tuple(self.pre_backbone_resize_hw)
        if tuple(x.shape[-2:]) == target_hw:
            return x
        return F.interpolate(x, size=target_hw, mode="nearest")

    # --------------------------------------------------------------------- #
    def forward(self, x):
        x = self._prepare_input(x)
        # Backbone
        x = self.b0(x)
        x = self.b1(x)
        p2 = self.b2(x)         # P2/4
        x  = self.b3(p2)
        p3 = self.b4(x)         # P3/8
        x  = self.b5(p3)
        p4 = self.b6(x)         # P4/16
        x  = self.b7(p4)
        p5 = self.b9(self.b8(x))  # P5/32 after SPPF

        # ----------- PAN head -----------
        p5_up = self.up(p5)
        p4_in = torch.cat([p5_up, p4], 1)
        p4_out = self.h0(p4_in)

        p4_up = self.up(p4_out)
        p3_in = torch.cat([p4_up, p3], 1)
        p3_out = self.h1(p3_in)      # small

        p3_down = self.down3(p3_out)
        p4b_in  = torch.cat([p3_down, p4_out], 1)
        p4b_out = self.h2(p4b_in)    # medium

        p4_down = self.down4(p4b_out)
        p5_in   = torch.cat([p4_down, p5], 1)
        p5_out  = self.h3(p5_in)     # large

        # Detect
        return self.detect(p3_out, p4b_out, p5_out)
