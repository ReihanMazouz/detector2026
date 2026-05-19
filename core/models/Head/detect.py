
import torch
import torch.nn as nn
import math
import copy

from ...nn.convs import Conv, DWConv
from ...nn.blocks import DFL
from ..anisotropic_utils import stride_hw_to_xy


class Detect(nn.Module):
    """
    Custom YOLO Detect head with separated bbox and cls/objectness branches and DFL decoding:
    - in_channels: list of channel dims for P3, P4, P5 feature maps
    - num_classes: number of object classes (0 → only objectness)
    - anchors_per_level: number of anchors per spatial location
    - reg_max: number of bins for bbox distribution
    """
    def __init__(self, in_channels, strides, num_classes=80, reg_max=16):
        super().__init__()
        self.nc = num_classes                   # number of classes
        self.reg_max = reg_max                  # distribution bins per bbox side
        self.nl = len(in_channels)              # number of detection layers
        self.legacy=True
        self.strides = strides

        # Determine channels for classification branch:
        c2, c3 = max((16, in_channels[0] // 4, self.reg_max * 4)), max(in_channels[0], min(self.nc, 100))  # channels

        # Branch for bbox distribution regression (4 sides * reg_max bins)
        self.cv_dist = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in in_channels
        )

        self.cv_clsobj = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in in_channels)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in in_channels
            )
        )

        # Distribution Focal Loss decoder for bbox
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, *features):
        # features: P3, P4, P5
        dist_outputs = [conv(f) for conv, f in zip(self.cv_dist, features)]
        clsobj_outputs = [conv(f) for conv, f in zip(self.cv_clsobj, features)]
        return dist_outputs, clsobj_outputs
    
    def bias_init(self, image_size=640):
        """
        Initialise les biais des têtes bbox (DFL) et classification pour faciliter la convergence.
        """
        if isinstance(image_size, int):
            image_h = float(image_size)
            image_w = float(image_size)
        else:
            image_h = float(image_size[0])
            image_w = float(image_size[1])

        for a, b, s in zip(self.cv_dist, self.cv_clsobj, self.strides):
            stride_x, stride_y = stride_hw_to_xy(s)
            # Initialisation des biais de la branche bbox (DFL)
            if hasattr(a[-1], "bias"):
                a[-1].bias.data[:] = 1.0

            # Initialisation des biais de la branche classification
            if hasattr(b[-1], "bias"):
                nx = max(image_w / stride_x, 1.0)
                ny = max(image_h / stride_y, 1.0)
                b[-1].bias.data[:self.nc] = math.log(5 / self.nc / (nx * ny))

    # def bias_init(self, image_size=(640, 640)):
    #     """
    #     Robust bias initialization for non-square images.
    #     image_size: (H, W)
    #     """
    #     if isinstance(image_size, int):
    #         H, W = image_size, image_size
    #     else:
    #         H, W = image_size

    #     for reg_branch, cls_branch, s in zip(self.cv_dist, self.cv_clsobj, self.strides):

    #         # ---------------------------------------------------------
    #         # 1. BBOX / DFL branch
    #         # ---------------------------------------------------------
    #         if hasattr(reg_branch[-1], "bias") and reg_branch[-1].bias is not None:
    #             reg_branch[-1].bias.data.fill_(1.0)

    #         # ---------------------------------------------------------
    #         # 2. Classification branch
    #         # ---------------------------------------------------------
    #         if hasattr(cls_branch[-1], "bias") and cls_branch[-1].bias is not None:

    #             # number of grid cells at this detection scale
    #             ny = H / s
    #             nx = W / s
    #             N = max(ny * nx, 1)

    #             # small initial probability per cell:
    #             #  p = 5 / N     -> standard YOLO convention
    #             p = 5.0 / N
    #             p = min(max(p, 1e-9), 1 - 1e-9)  # clamp for safety

    #             # logit(p)
    #             bias_value = math.log(p / (1.0 - p))

    #             # only first `nc` biases correspond to class logits
    #             cls_branch[-1].bias.data[:self.nc].fill_(bias_value)


class One2OneDetect(nn.Module):
    """Detect head initialized as a copy of an existing one2many Detect head."""

    def __init__(self, one2many_head: Detect):
        super().__init__()
        self.nc = one2many_head.nc
        self.reg_max = one2many_head.reg_max
        self.nl = one2many_head.nl
        self.legacy = one2many_head.legacy
        self.strides = one2many_head.strides
        self.cv_dist = copy.deepcopy(one2many_head.cv_dist)
        self.cv_clsobj = copy.deepcopy(one2many_head.cv_clsobj)
        self.dfl = copy.deepcopy(one2many_head.dfl)

    def forward(self, *features):
        dist_outputs = [conv(f) for conv, f in zip(self.cv_dist, features)]
        clsobj_outputs = [conv(f) for conv, f in zip(self.cv_clsobj, features)]
        return dist_outputs, clsobj_outputs
