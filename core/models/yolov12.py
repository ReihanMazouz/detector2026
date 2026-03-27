import torch
import torch.nn as nn
import math

from ..nn.convs import Conv
from ..nn.blocks import C3k2, A2C2f
from .base import BaseModel
from ..utils.loss import YOLODetectionLoss
from .Head.detect import Detect

class YOLOv12(BaseModel):
    """
    YOLOv12-turbo – backbone P3/8-P5/32 et head PAN-A2C2f.
    """
    # --------------------------- méta-paramètres ----------------------------
    strides = [8, 16, 32]

    def __init__(self,
                 output_dir: str,
                 num_classes: int = 80,
                 device: str = "cuda:0",
                 in_ch: int = 1,
                 reg_max: int = 16,
                 width_mult: float = 0.25,
                 depth_mult: float = 0.5,
                 debug: bool = False):
        super().__init__(device=device, output_dir=output_dir)
        self.nc, self.debug = num_classes, debug

        self.width_mult = width_mult
        self.depth_mult = depth_mult

        # ---------- calc. canaux (clamp à max=1024) ----------
        def c(c_):                    # helper qui applique width_mult
            return min(int(c_ * self.width_mult), 1024)

        c1, c2, c3, c4, c5 = map(c, (64, 128, 256, 512, 1024))

        # --------------------------- Backbone -------------------------------
        # P1/2  ↘
        self.conv1 = Conv(in_ch,  c1, k=3, s=2)                   # g=1
        # P2/4  ↘ (groups = 2)
        self.conv2 = Conv(c1,     c2, k=3, s=2, g=2)              # g=2
        # C3k2 ×2
        self.c3_1 = C3k2(c2, c3, n=self._rep(2), shortcut=False)

        # P3/8  ↘ (groups = 4)
        self.conv3 = Conv(c3,     c3, k=3, s=2, g=4)
        self.c3_2 = C3k2(c3, c3, n=self._rep(2), shortcut=False)

        # P4/16 ↘
        self.conv4 = Conv(c3,     c4, k=3, s=2)
        # ⚠️ nouveau bloc Area-Attention  (4 rép. → depth-scaled)
        self.a2c2f4 = A2C2f(c4, c4, n=self._rep(4), shortcut=True, e=0.5)

        # P5/32 ↘
        self.conv5 = Conv(c4,     c5, k=3, s=2)
        self.a2c2f5 = A2C2f(c5, c5, n=self._rep(4), shortcut=True, e=0.5)

        # --------------------------- Head (PAN-A2C2f) -----------------------
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        # P5 ↑ concat P4
        self.head_p4 = A2C2f(c5 + c4, c4, n=self._rep(2), shortcut=False)
        # P4 ↑ concat P3
        self.head_p3 = A2C2f(c4 + c3, c3, n=self._rep(2), shortcut=False)

        # P3 ↓ concat (p4_out)
        self.down_p3  = Conv(c3, c3, k=3, s=2)
        self.head_p4b = A2C2f(c3 + c4, c4, n=self._rep(2), shortcut=False)

        # P4 ↓ concat backbone P5
        self.down_p4  = Conv(c4, c4, k=3, s=2)
        self.head_p5  = C3k2(c4 + c5, c5, n=self._rep(2), shortcut=True)

        # --------------------------- Detect -------------------------------
        self.detect = Detect(in_channels=[c3, c4, c5],
                             num_classes=self.nc,
                             reg_max=reg_max,
                             strides=self.strides)
        self.detect.bias_init(image_size=1024)
        self.criterion = YOLODetectionLoss(self.nc, self.strides,
                                           reg_max, self.device)
        self.to(self.device)

    # --------------------------- Helpers -----------------------------------
    def _rep(self, n):
        """Applique depth_mult en conservant au moins 1 répétition."""
        return max(int(n * self.depth_mult), 1)

    # --------------------------- Forward -----------------------------------
    def forward(self, x):
        # Backbone
        x  = self.conv1(x)
        x  = self.conv2(x)
        f3 = self.c3_1(x)            # P2/4

        x  = self.conv3(f3)
        f4 = self.c3_2(x)            # P3/8

        x  = self.conv4(f4)
        f4a = self.a2c2f4(x)         # P4/16

        x  = self.conv5(f4a)
        f5 = self.a2c2f5(x)          # P5/32
        # ------------------------------------------------------------------
        p5_up = self.up(f5)
        p4_in = torch.cat([p5_up, f4a], 1)
        p4    = self.head_p4(p4_in)

        p4_up = self.up(p4)
        p3_in = torch.cat([p4_up, f4], 1)
        p3    = self.head_p3(p3_in)

        p3_down = self.down_p3(p3)
        p4b_in  = torch.cat([p3_down, p4], 1)
        p4b     = self.head_p4b(p4b_in)

        p4_down = self.down_p4(p4b)
        p5_in   = torch.cat([p4_down, f5], 1)
        p5_out  = self.head_p5(p5_in)

        # Detect
        return self.detect(p3, p4b, p5_out)
