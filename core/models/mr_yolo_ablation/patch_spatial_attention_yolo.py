from __future__ import annotations

from typing import List, Tuple, Union

import torch
import torch.nn as nn

from ...nn.blocks import C2PSA, SPPF, TFSepBlock
from ...nn.convs import Conv
from ...utils.loss import YOLODetectionLoss
from ..Backbones.TF_BranchBackbone import BranchBackbone
from ..Head.detect import Detect
from ..base import BaseModel
from .branch_cross_attention import _make_even_channels
from .patch_spatial_attention import MRPatchSpatialAttentionBlock


def _patch_sizes_for(resolutions: list[tuple[int, int]], grid_hw: tuple[int, int]) -> list[tuple[int, int]]:
    grid_h, grid_w = grid_hw
    patch_sizes = []
    for height, width in resolutions:
        if height % grid_h != 0 or width % grid_w != 0:
            raise ValueError(f"Resolution {(height, width)} is not divisible by latent grid {grid_hw}.")
        patch_sizes.append((height // grid_h, width // grid_w))
    return patch_sizes


class PatchSpatialAttentionBackbone(nn.Module):
    """MR-YOLO branch backbone with two latent spatial-attention insertions."""

    def __init__(
        self,
        input_resolutions: List[Tuple[int, int]],
        width_mult: float = 0.25,
        in_ch: int = 1,
        out_channels_mult: int = 2,
        constant_ch: Union[bool, int] = False,
        latent_grid_hw: tuple[int, int] = (16, 16),
        patch_d_model: int = 128,
        patch_num_heads: int = 4,
        patch_num_layers: int = 1,
        patch_num_points: int = 16,
        patch_ffn_ratio: float = 2.0,
        patch_dropout: float = 0.0,
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
        self.stage_indices = self._resolve_stage_indices()
        p1_channels, p1_resolutions, p2_channels, p2_resolutions, p3_channels = self._infer_stage_metadata()

        self.attn_p1 = MRPatchSpatialAttentionBlock(
            input_channels=p1_channels,
            input_resolutions=p1_resolutions,
            patch_sizes=_patch_sizes_for(p1_resolutions, latent_grid_hw),
            d_model=patch_d_model,
            num_heads=patch_num_heads,
            num_layers=patch_num_layers,
            num_points=patch_num_points,
            mlp_ratio=patch_ffn_ratio,
            dropout=patch_dropout,
        )
        self.attn_p2 = MRPatchSpatialAttentionBlock(
            input_channels=p2_channels,
            input_resolutions=p2_resolutions,
            patch_sizes=_patch_sizes_for(p2_resolutions, latent_grid_hw),
            d_model=patch_d_model,
            num_heads=patch_num_heads,
            num_layers=patch_num_layers,
            num_points=patch_num_points,
            mlp_ratio=patch_ffn_ratio,
            dropout=patch_dropout,
        )

        base_ch = p3_channels[0]
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

        self.fuse_p3 = nn.Sequential(
            Conv(sum(p3_channels), c3, k=1, s=1),
            TFSepBlock(ch=c3, n=1, residual=True, mode="parallel"),
        )
        self.conv_p4 = Conv(c3, c4, k=3, s=2)
        self.c3_p4 = TFSepBlock(ch=c4, n=1, residual=True, mode="parallel")
        self.conv_p5 = Conv(c4, c5, k=3, s=2)
        self.c3_p5 = TFSepBlock(ch=c5, n=1, residual=True, mode="parallel")
        self.sppf = SPPF(c5, c5, k=5)
        self.psa = C2PSA(c5, c5, n=min(2, c5 // 1024), e=0.5)

    def _resolve_stage_indices(self) -> list[tuple[int, int, int]]:
        indices = []
        for branch in self.branches:
            selected = list(branch.out_indices[-3:])
            if not selected:
                raise ValueError("BranchBackbone has no selected feature indices.")
            p1_idx = selected[0]
            p2_idx = selected[1] if len(selected) >= 2 else selected[0]
            p3_idx = selected[-1]
            indices.append((p1_idx, p2_idx, p3_idx))
        return indices

    def _infer_stage_metadata(self):
        was_training = [branch.training for branch in self.branches]
        for branch in self.branches:
            branch.eval()
        with torch.no_grad():
            dummy_inputs = [
                torch.zeros(1, 1, height, width)
                for height, width in self.input_resolutions
            ]
            outputs = [branch(x) for branch, x in zip(self.branches, dummy_inputs)]
        for branch, training in zip(self.branches, was_training):
            branch.train(training)

        p1 = [out[0] for out in outputs]
        p2 = [out[1] if len(out) >= 2 else out[0] for out in outputs]
        p3 = [out[-1] for out in outputs]
        return (
            [feat.shape[1] for feat in p1],
            [tuple(feat.shape[-2:]) for feat in p1],
            [feat.shape[1] for feat in p2],
            [tuple(feat.shape[-2:]) for feat in p2],
            [feat.shape[1] for feat in p3],
        )

    def _run_to(
        self,
        states: list[torch.Tensor],
        positions: list[int],
        targets: list[int],
    ) -> tuple[list[torch.Tensor], list[int]]:
        next_states = []
        next_positions = []
        for state, position, target, branch in zip(states, positions, targets, self.branches):
            out = state
            for layer_index in range(position + 1, target + 1):
                out = branch.model[layer_index](out)
            next_states.append(out)
            next_positions.append(target)
        return next_states, next_positions

    def forward(self, inputs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states = list(inputs)
        positions = [-1] * len(states)

        p1_targets = [indices[0] for indices in self.stage_indices]
        states, positions = self._run_to(states, positions, p1_targets)
        p1s = self.attn_p1(states)
        states = p1s

        p2_targets = [indices[1] for indices in self.stage_indices]
        states, positions = self._run_to(states, positions, p2_targets)
        p2s = self.attn_p2(states)
        states = p2s

        p3_targets = [indices[2] for indices in self.stage_indices]
        p3s, positions = self._run_to(states, positions, p3_targets)
        self.last_forward_features = [tuple(p1s), tuple(p2s), tuple(p3s)]

        p3 = self.fuse_p3(torch.cat(p3s, dim=1))
        p4 = self.c3_p4(self.conv_p4(p3))
        p5 = self.conv_p5(p4)
        p5 = self.c3_p5(p5)
        p5 = self.sppf(p5)
        p5 = self.psa(p5)
        return p3, p4, p5


class MRYOLOPatchSpatialAttentionAblation(BaseModel):
    """MR-YOLO with latent patch spatial-attention blocks inserted in the branches."""

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
        patch_latent_grid_hw: tuple[int, int] = (16, 16),
        patch_d_model: int = 128,
        patch_num_heads: int = 4,
        patch_num_layers: int = 1,
        patch_num_points: int = 16,
        patch_ffn_ratio: float = 2.0,
        patch_dropout: float = 0.0,
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

        self.backbone = PatchSpatialAttentionBackbone(
            input_resolutions=input_resolutions,
            width_mult=width_mult,
            in_ch=in_ch,
            out_channels_mult=outfusion_channels_mult,
            constant_ch=constant_backbone_ch,
            latent_grid_hw=patch_latent_grid_hw,
            patch_d_model=patch_d_model,
            patch_num_heads=patch_num_heads,
            patch_num_layers=patch_num_layers,
            patch_num_points=patch_num_points,
            patch_ffn_ratio=patch_ffn_ratio,
            patch_dropout=patch_dropout,
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
            raise ValueError(f"Expected {len(self.input_resolutions)} inputs, got {len(inputs)}.")
        for index, (x, (height, width)) in enumerate(zip(inputs, self.input_resolutions)):
            if x.dim() != 4:
                raise ValueError(f"Input #{index} must be 4D, got {x.dim()}D.")
            if x.shape[1] != self.in_ch:
                raise ValueError(f"Input #{index} has {x.shape[1]} channels, expected {self.in_ch}.")
            if tuple(x.shape[-2:]) != (height, width):
                raise ValueError(
                    f"Input #{index} has shape {tuple(x.shape[-2:])}, expected {(height, width)}."
                )

    def forward(self, inputs: List[torch.Tensor]):
        self._validate_inputs(inputs)
        p3, p4, p5 = self.backbone(inputs)
        self.last_forward_features_before_fusion = self.backbone.last_forward_features
        self.last_forward_features_after_fusion = [p3, p4, p5]

        p4_out = self.head_c3_1(torch.cat([self.upsample(p5), p4], dim=1))
        p3_out = self.head_c3_2(torch.cat([self.upsample(p4_out), p3], dim=1))
        p4_out2 = self.head_c3_3(torch.cat([self.down_p3(p3_out), p4_out], dim=1))
        p5_out = self.head_c3_4(torch.cat([self.down_p4(p4_out2), p5], dim=1))
        return self.detect(p3_out, p4_out2, p5_out)
