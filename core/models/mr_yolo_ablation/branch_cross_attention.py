from typing import List, Tuple, Union

import torch
import torch.nn as nn

from ...nn.blocks import C2PSA, SPPF, TFSepBlock
from ...nn.convs import Conv
from ...utils.loss import YOLODetectionLoss
from ..Backbones.TF_BranchBackbone import BranchBackbone
from ..Head.detect import Detect
from ..base import BaseModel
from .fusion import InterResolutionCrossAttentionFusion


def _make_even_channels(value: int, minimum: int = 2) -> int:
    value = max(int(value), minimum)
    return value if value % 2 == 0 else value + 1


class BranchCrossAttentionBackbone(nn.Module):
    """MR branch backbone whose P3 fusion is inter-resolution cross-attention."""

    def __init__(
        self,
        input_resolutions: List[Tuple[int, int]],
        width_mult: float = 0.25,
        in_ch: int = 1,
        out_channels_mult: int = 2,
        constant_ch: Union[bool, int] = False,
        fusion_mode: str = "deformable",
        center_resolution_index: int | None = None,
        fusion_d_model: int = 128,
        fusion_num_heads: int = 4,
        fusion_num_layers: int = 1,
        fusion_num_points: int = 4,
        fusion_ffn_ratio: float = 2.0,
        fusion_dropout: float = 0.0,
    ):
        super().__init__()
        if not input_resolutions:
            raise ValueError("input_resolutions must contain at least one resolution.")

        self.input_resolutions = list(input_resolutions)
        self.last_forward_features = []
        self.branches = nn.ModuleList(
            [
                BranchBackbone(
                    res,
                    target_hw=(64, 64),
                    width_mult=width_mult,
                    cmax=256,
                    in_ch=in_ch,
                    constant_ch=constant_ch,
                )
                for res in input_resolutions
            ]
        )

        branch_out_channels = [branch.out_channels()[-1] for branch in self.branches]
        base_ch = branch_out_channels[0]

        if constant_ch:
            c3 = _make_even_channels(int(constant_ch * width_mult))
            c4 = c3
            c5 = c3
        else:
            c3 = _make_even_channels(min(int(1024 * width_mult), base_ch * out_channels_mult))
            c4 = _make_even_channels(min(int(1024 * width_mult), c3 * 2))
            c5 = _make_even_channels(min(int(1024 * width_mult), c4 * 2))
        self.out_channels = (c3, c4, c5)

        self.strides = []
        for branch in self.branches:
            s_h, s_w = branch.strides[-1]
            self.strides.append(
                [
                    (s_h, s_w),
                    (s_h * 2, s_w * 2),
                    (s_h * 4, s_w * 4),
                ]
            )

        self.fuse_p3 = InterResolutionCrossAttentionFusion(
            input_channels=branch_out_channels,
            out_channels=c3,
            d_model=fusion_d_model,
            num_heads=fusion_num_heads,
            num_layers=fusion_num_layers,
            num_points=fusion_num_points,
            ffn_ratio=fusion_ffn_ratio,
            dropout=fusion_dropout,
            fusion_mode=fusion_mode,
            center_resolution_index=center_resolution_index,
        )
        self.c3_p3 = TFSepBlock(c3, n=1, residual=True, mode="parallel")

        self.conv_p4 = Conv(c3, c4, k=3, s=2)
        self.c3_p4 = TFSepBlock(c4, n=1, residual=True, mode="parallel")

        self.conv_p5 = Conv(c4, c5, k=3, s=2)
        self.c3_p5 = TFSepBlock(c5, n=1, residual=True, mode="parallel")
        self.sppf = SPPF(c5, c5, k=5)
        self.psa = C2PSA(c5, c5, n=min(2, c5 // 1024), e=0.5)

    def forward(self, inputs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p3s = [branch(x)[-1] for branch, x in zip(self.branches, inputs)]
        self.last_forward_features = [tuple(p3s), None, None]

        p3 = self.fuse_p3(p3s)
        p3 = self.c3_p3(p3)

        p4 = self.c3_p4(self.conv_p4(p3))
        p5 = self.conv_p5(p4)
        p5 = self.c3_p5(p5)
        p5 = self.sppf(p5)
        p5 = self.psa(p5)

        return p3, p4, p5


class MRYOLOBranchCrossAttentionAblation(BaseModel):
    """
    MR-YOLO ablation with BranchBackbone per resolution and cross-attention fusion.
    """

    def __init__(
        self,
        input_resolutions: List[Tuple[int, int]],
        output_dir: str,
        num_classes: int = 80,
        reg_max: int = 16,
        device: str = "cuda:0",
        in_ch: int = 1,
        width_mult: float = 0.25,
        outfusion_channels_mult: int = 2,
        constant_backbone_ch: Union[bool, int] = False,
        fusion_mode: str = "deformable",
        center_resolution_index: int | None = None,
        fusion_d_model: int = 128,
        fusion_num_heads: int = 4,
        fusion_num_layers: int = 1,
        fusion_num_points: int = 4,
        fusion_ffn_ratio: float = 2.0,
        fusion_dropout: float = 0.0,
        debug: bool = False,
    ):
        super().__init__(device=device, output_dir=output_dir)
        self.debug = debug
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.input_resolutions = list(input_resolutions)
        self.in_ch = int(in_ch)

        self.last_forward_features_before_fusion = []
        self.last_forward_features_after_fusion = []

        self.backbone = BranchCrossAttentionBackbone(
            input_resolutions=input_resolutions,
            width_mult=width_mult,
            in_ch=in_ch,
            out_channels_mult=outfusion_channels_mult,
            constant_ch=constant_backbone_ch,
            fusion_mode=fusion_mode,
            center_resolution_index=center_resolution_index,
            fusion_d_model=fusion_d_model,
            fusion_num_heads=fusion_num_heads,
            fusion_num_layers=fusion_num_layers,
            fusion_num_points=fusion_num_points,
            fusion_ffn_ratio=fusion_ffn_ratio,
            fusion_dropout=fusion_dropout,
        )
        c3, c4, c5 = self.backbone.out_channels

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.head_c3_1 = nn.Sequential(
            Conv(c5 + c4, c4, k=1, s=1),
            TFSepBlock(ch=c4, n=1, residual=True, mode="parallel"),
        )
        self.head_c3_2 = nn.Sequential(
            Conv(c4 + c3, c3, k=1, s=1),
            TFSepBlock(ch=c3, n=1, residual=True, mode="parallel"),
        )
        self.down_p3 = Conv(c3, c3, 3, 2)
        self.head_c3_3 = nn.Sequential(
            Conv(c3 + c4, c4, k=1, s=1),
            TFSepBlock(ch=c4, n=1, residual=True, mode="parallel"),
        )
        self.down_p4 = Conv(c4, c4, 3, 2)
        self.head_c3_4 = nn.Sequential(
            Conv(c4 + c5, c5, k=1, s=1),
            TFSepBlock(ch=c5, n=1, residual=True, mode="parallel"),
        )

        raw_strides = self.backbone.strides
        self.strides = [
            max(max(h, w) for (h, w) in (branch[j] for branch in raw_strides))
            for j in range(3)
        ]

        self.detect = Detect(
            in_channels=[c3, c4, c5],
            strides=self.strides,
            num_classes=num_classes,
            reg_max=reg_max,
        )
        max_input_dim = max(max(height, width) for height, width in input_resolutions)
        self.detect.bias_init(image_size=max_input_dim)

        self.criterion = YOLODetectionLoss(
            num_classes=num_classes,
            strides=self.detect.strides,
            reg_max=reg_max,
            device=device,
        )

        self.to(device)

    def _validate_inputs(self, inputs: List[torch.Tensor]) -> None:
        if len(inputs) != len(self.input_resolutions):
            raise ValueError(
                f"Expected {len(self.input_resolutions)} inputs, got {len(inputs)}."
            )
        for index, (x, (height, width)) in enumerate(zip(inputs, self.input_resolutions)):
            if x.dim() != 4:
                raise ValueError(f"Input #{index} must be 4D, got {x.dim()}D.")
            if x.shape[1] != self.in_ch:
                raise ValueError(
                    f"Input #{index} has {x.shape[1]} channels, expected {self.in_ch}."
                )
            if tuple(x.shape[-2:]) != (height, width):
                raise ValueError(
                    f"Input #{index} has shape {tuple(x.shape[-2:])}, "
                    f"expected {(height, width)}."
                )

    def forward(self, inputs: List[torch.Tensor]):
        self._validate_inputs(inputs)

        p3, p4, p5 = self.backbone(inputs)
        self._dbg("after backbone p3", p3)
        self._dbg("after backbone p4", p4)
        self._dbg("after backbone p5", p5)

        self.last_forward_features_before_fusion = self.backbone.last_forward_features
        self.last_forward_features_after_fusion = [p3, p4, p5]

        p5_up = self.upsample(p5)
        p4_out = self.head_c3_1(torch.cat([p5_up, p4], dim=1))

        p4_up = self.upsample(p4_out)
        p3_out = self.head_c3_2(torch.cat([p4_up, p3], dim=1))

        p3_d = self.down_p3(p3_out)
        p4_out2 = self.head_c3_3(torch.cat([p3_d, p4_out], dim=1))

        p4_d = self.down_p4(p4_out2)
        p5_out = self.head_c3_4(torch.cat([p4_d, p5], dim=1))

        self._dbg("after Neck p3_out", p3_out)
        self._dbg("after Neck p4_out2", p4_out2)
        self._dbg("after Neck p5_out", p5_out)

        return self.detect(p3_out, p4_out2, p5_out)

    def _dbg(self, name: str, tensor: torch.Tensor) -> None:
        if self.debug:
            print(f"[DBG] {name:<25}: {tuple(tensor.shape)}")
