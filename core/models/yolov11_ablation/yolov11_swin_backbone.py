from __future__ import annotations

import torch

from ..Backbones.SwinBackbone import SwinBackbone
from ..yolov11 import YOLOv11


class YOLOv11SwinBackbone(YOLOv11):
    """YOLOv11 ablation replacing the convolutional backbone with a Swin backbone."""

    def __init__(
        self,
        *args,
        swin_embed_dim: int | None = None,
        swin_depths=(2, 2, 4, 2),
        swin_num_heads=(2, 4, 8, 8),
        swin_window_size: int = 8,
        swin_mlp_ratio: float = 4.0,
        swin_drop_rate: float = 0.0,
        swin_attn_drop_rate: float = 0.0,
        swin_drop_path_rate: float = 0.05,
        **kwargs,
    ):
        input_canals = kwargs.get("input_canals", 1)
        width_mult = kwargs.get("width_mult", 0.25)
        if len(args) >= 6:
            input_canals = args[5]
        if len(args) >= 7:
            width_mult = args[6]

        super().__init__(*args, **kwargs)

        for name in ("conv1", "conv2", "c3_1", "conv3", "c3_2", "conv4", "c3_3", "conv5", "c3_4", "sppf", "attn"):
            if hasattr(self, name):
                delattr(self, name)

        if swin_embed_dim is None:
            swin_embed_dim = max(16, int(256 * float(width_mult)))
        self.swin_backbone = SwinBackbone(
            in_chans=input_canals,
            embed_dim=int(swin_embed_dim),
            depths=tuple(int(v) for v in swin_depths),
            num_heads=tuple(int(v) for v in swin_num_heads),
            window_size=int(swin_window_size),
            mlp_ratio=float(swin_mlp_ratio),
            drop_rate=float(swin_drop_rate),
            attn_drop_rate=float(swin_attn_drop_rate),
            drop_path_rate=float(swin_drop_path_rate),
        )
        self.to(self.device)

    def forward_features(self, x):
        x = self._prepare_input(x)
        f3, f4, f5 = self.swin_backbone(x)
        self.debug_shape("swin f3", f3)
        self.debug_shape("swin f4", f4)
        self.debug_shape("swin f5", f5)

        p5_up = self.upsample(f5)
        self.debug_shape("p5_up", p5_up)

        p4_feat = self._crop_and_cat(p5_up, f4)
        self.debug_shape("p4_feat", p4_feat)

        p4_out = self.head_c3_1(p4_feat)
        self.debug_shape("p4_out", p4_out)

        p4_up = self.upsample(p4_out)
        self.debug_shape("p4_up", p4_up)

        p3_feat = self._crop_and_cat(p4_up, f3)
        self.debug_shape("p3_feat", p3_feat)

        p3_out = self.head_c3_2(p3_feat)
        self.debug_shape("p3_out", p3_out)

        p3_down = self.down_p3(p3_out)
        self.debug_shape("p3_down", p3_down)

        pm_feat = self._crop_and_cat(p3_down, p4_out)
        self.debug_shape("pm_feat", pm_feat)

        p4_out2 = self.head_c3_3(pm_feat)
        self.debug_shape("p4_out2", p4_out2)

        p4_down = self.down_p4(p4_out2)
        self.debug_shape("p4_down", p4_down)

        pl_feat = self._crop_and_cat(p4_down, f5)
        self.debug_shape("pl_feat", pl_feat)

        p5_out = self.head_c3_4(pl_feat)
        self.debug_shape("p5_out", p5_out)

        return p3_out, p4_out2, p5_out

    @staticmethod
    def _crop_and_cat(a, b):
        height = min(a.shape[-2], b.shape[-2])
        width = min(a.shape[-1], b.shape[-1])
        return torch.cat([a[..., :height, :width], b[..., :height, :width]], dim=1)
