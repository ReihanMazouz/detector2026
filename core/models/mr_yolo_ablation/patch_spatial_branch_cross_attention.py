from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...nn.blocks import SPPF, TFSepBlock
from ...nn.convs import Conv
from ...utils.dataset import YOLODatasetFusedMultiRes, load_class_index_to_name
from ...utils.detr_loss import targets_from_yolo_tensor
from ...utils.evaluate import EvalConfig, EvalRunner, TrainingPlots
from ...utils.loss import YOLODetectionLoss
from ...utils.rtdetr_loss import RTDETRLoss
from ..Backbones.TF_BranchBackbone import BranchBackbone
from ..Head.detect import Detect
from ..Head.rtdetr import RTDETRHead
from ..base import BaseModel, _resolve_num_workers, _supports_cuda
from .branch_cross_attention import _make_even_channels
from .fusion import InterResolutionCrossAttentionFusion
from .patch_spatial_attention import MRPatchSpatialAttentionBlock


def _fixed_patch_sizes(
    resolutions: Sequence[Tuple[int, int]],
    patch_size: Tuple[int, int],
) -> list[Tuple[int, int]]:
    patch_h, patch_w = patch_size
    patch_sizes = []
    for resolution in resolutions:
        height, width = resolution
        if height % patch_h != 0 or width % patch_w != 0:
            raise ValueError(f"Patch size {patch_size} does not divide resolution {resolution}.")
        patch_sizes.append((patch_h, patch_w))
    return patch_sizes


class PatchSpatialBranchCrossAttentionBackbone(nn.Module):
    """MR branch backbone with P2 patch gates and P3 cross-attention fusion.

    Patch spatial blocks are applied on each P2 branch feature with a fixed
    patch size, then P3 branch features are fused by inter-resolution
    cross-attention. After fusion, single-map patch spatial blocks are applied
    on P4 and P5. C2PSA is intentionally omitted.
    """

    def __init__(
        self,
        input_resolutions: List[Tuple[int, int]],
        width_mult: float = 0.25,
        in_ch: int = 1,
        out_channels_mult: int = 2,
        constant_ch: Union[bool, int] = False,
        patch_size: Tuple[int, int] = (8, 8),
        patch_d_model: int = 128,
        patch_num_heads: int = 4,
        patch_num_layers: int = 1,
        patch_num_points: int = 16,
        patch_ffn_ratio: float = 2.0,
        patch_dropout: float = 0.0,
        patch_alpha_bound: float | None = 1.0,
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
        super().__init__()
        if not input_resolutions:
            raise ValueError("input_resolutions must contain at least one resolution.")

        self.input_resolutions = list(input_resolutions)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.debug = bool(debug)
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
        p2_channels, p2_resolutions, p3_channels, p3_resolutions = self._infer_stage_metadata()

        self.attn_p2 = nn.ModuleList(
            MRPatchSpatialAttentionBlock(
                input_channels=[channels],
                input_resolutions=[resolution],
                patch_sizes=_fixed_patch_sizes([resolution], self.patch_size),
                d_model=patch_d_model,
                num_heads=patch_num_heads,
                num_layers=patch_num_layers,
                num_points=patch_num_points,
                mlp_ratio=patch_ffn_ratio,
                dropout=patch_dropout,
                alpha_bound=patch_alpha_bound,
            )
            for channels, resolution in zip(p2_channels, p2_resolutions)
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

        self.fuse_p3 = InterResolutionCrossAttentionFusion(
            input_channels=p3_channels,
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
        self.c3_p4 = TFSepBlock(ch=c4, n=1, residual=True, mode="parallel")
        self.conv_p5 = Conv(c4, c5, k=3, s=2)
        self.c3_p5 = TFSepBlock(ch=c5, n=1, residual=True, mode="parallel")
        self.sppf = SPPF(c5, c5, k=5)

        p4_resolution, p5_resolution = self._infer_neck_resolutions(
            p3_resolution=p3_resolutions[
                len(p3_resolutions) // 2 if center_resolution_index is None else center_resolution_index
            ],
            c3=c3,
        )
        self.attn_p4 = MRPatchSpatialAttentionBlock(
            input_channels=[c4],
            input_resolutions=[p4_resolution],
            patch_sizes=_fixed_patch_sizes([p4_resolution], self.patch_size),
            d_model=patch_d_model,
            num_heads=patch_num_heads,
            num_layers=patch_num_layers,
            num_points=patch_num_points,
            mlp_ratio=patch_ffn_ratio,
            dropout=patch_dropout,
            alpha_bound=patch_alpha_bound,
        )
        self.attn_p5 = MRPatchSpatialAttentionBlock(
            input_channels=[c5],
            input_resolutions=[p5_resolution],
            patch_sizes=_fixed_patch_sizes([p5_resolution], self.patch_size),
            d_model=patch_d_model,
            num_heads=patch_num_heads,
            num_layers=patch_num_layers,
            num_points=patch_num_points,
            mlp_ratio=patch_ffn_ratio,
            dropout=patch_dropout,
            alpha_bound=patch_alpha_bound,
        )

    def _check_finite(self, name: str, tensor: torch.Tensor) -> None:
        if not self.debug:
            return
        if not torch.isfinite(tensor).all():
            detached = tensor.detach()
            finite = detached[torch.isfinite(detached)]
            if finite.numel():
                stats = (
                    f"min={finite.min().item():.6g} "
                    f"max={finite.max().item():.6g} "
                    f"mean={finite.mean().item():.6g}"
                )
            else:
                stats = "all values are non-finite"
            raise RuntimeError(f"Non-finite tensor after {name}: shape={tuple(tensor.shape)} {stats}")

    def _check_finite_sequence(self, name: str, tensors: Sequence[torch.Tensor]) -> None:
        for index, tensor in enumerate(tensors):
            self._check_finite(f"{name}[{index}]", tensor)

    def _resolve_stage_indices(self) -> list[tuple[int, int]]:
        indices = []
        for branch in self.branches:
            selected = list(branch.out_indices[-3:])
            if not selected:
                raise ValueError("BranchBackbone has no selected feature indices.")
            p2_idx = selected[1] if len(selected) >= 2 else selected[0]
            p3_idx = selected[-1]
            indices.append((p2_idx, p3_idx))
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

        p2 = [out[1] if len(out) >= 2 else out[0] for out in outputs]
        p3 = [out[-1] for out in outputs]
        return (
            [feat.shape[1] for feat in p2],
            [tuple(feat.shape[-2:]) for feat in p2],
            [feat.shape[1] for feat in p3],
            [tuple(feat.shape[-2:]) for feat in p3],
        )

    def _infer_neck_resolutions(self, p3_resolution: Tuple[int, int], c3: int):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            x = torch.zeros(1, c3, *p3_resolution)
            p4 = self.c3_p4(self.conv_p4(x))
            p5 = self.conv_p5(p4)
            p5 = self.c3_p5(p5)
            p5 = self.sppf(p5)
        self.train(was_training)
        return tuple(p4.shape[-2:]), tuple(p5.shape[-2:])

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

        p2_targets = [indices[0] for indices in self.stage_indices]
        states, positions = self._run_to(states, positions, p2_targets)
        self._check_finite_sequence("raw_p2", states)
        p2s = [attn([state])[0] for attn, state in zip(self.attn_p2, states)]
        self._check_finite_sequence("attn_p2", p2s)
        states = p2s

        p3_targets = [indices[1] for indices in self.stage_indices]
        p3s, positions = self._run_to(states, positions, p3_targets)
        self._check_finite_sequence("raw_p3", p3s)
        self.last_forward_features = [None, tuple(p2s), tuple(p3s)]

        p3 = self.c3_p3(self.fuse_p3(p3s))
        self._check_finite("fused_p3", p3)
        p4 = self.c3_p4(self.conv_p4(p3))
        self._check_finite("raw_p4", p4)
        p4 = self.attn_p4([p4])[0]
        self._check_finite("attn_p4", p4)
        p5 = self.conv_p5(p4)
        p5 = self.c3_p5(p5)
        p5 = self.sppf(p5)
        self._check_finite("raw_p5", p5)
        p5 = self.attn_p5([p5])[0]
        self._check_finite("attn_p5", p5)
        return p3, p4, p5


class MRYOLOPatchSpatialBranchCrossAttentionAblation(BaseModel):
    """MR-YOLO with P2/P4/P5 patch spatial attention and P3 cross-attention fusion."""

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
        patch_size: Tuple[int, int] = (8, 8),
        patch_d_model: int = 128,
        patch_num_heads: int = 4,
        patch_num_layers: int = 1,
        patch_num_points: int = 16,
        patch_ffn_ratio: float = 2.0,
        patch_dropout: float = 0.0,
        patch_alpha_bound: float | None = 1.0,
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
        self.num_classes = int(num_classes)
        self.reg_max = int(reg_max)
        self.input_resolutions = list(input_resolutions)
        self.in_ch = int(in_ch)
        self.last_forward_features_before_fusion = []
        self.last_forward_features_after_fusion = []

        self.backbone = PatchSpatialBranchCrossAttentionBackbone(
            input_resolutions=input_resolutions,
            width_mult=width_mult,
            in_ch=in_ch,
            out_channels_mult=outfusion_channels_mult,
            constant_ch=constant_backbone_ch,
            patch_size=patch_size,
            patch_d_model=patch_d_model,
            patch_num_heads=patch_num_heads,
            patch_num_layers=patch_num_layers,
            patch_num_points=patch_num_points,
            patch_ffn_ratio=patch_ffn_ratio,
            patch_dropout=patch_dropout,
            patch_alpha_bound=patch_alpha_bound,
            fusion_mode=fusion_mode,
            center_resolution_index=center_resolution_index,
            fusion_d_model=fusion_d_model,
            fusion_num_heads=fusion_num_heads,
            fusion_num_layers=fusion_num_layers,
            fusion_num_points=fusion_num_points,
            fusion_ffn_ratio=fusion_ffn_ratio,
            fusion_dropout=fusion_dropout,
            debug=debug,
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
        outputs = self.detect(p3_out, p4_out2, p5_out)
        if self.debug:
            dist_out, clsobj_out = outputs
            for index, tensor in enumerate(dist_out):
                self.backbone._check_finite(f"detect_dist[{index}]", tensor)
            for index, tensor in enumerate(clsobj_out):
                self.backbone._check_finite(f"detect_clsobj[{index}]", tensor)
        return outputs


class MRYOLOPatchSpatialBranchCrossAttentionRTDETRHead(BaseModel):
    """Frozen patch-spatial/cross-attention backbone with an RT-DETR one-to-one head."""

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
        patch_size: Tuple[int, int] = (8, 8),
        patch_d_model: int = 128,
        patch_num_heads: int = 4,
        patch_num_layers: int = 1,
        patch_num_points: int = 16,
        patch_ffn_ratio: float = 2.0,
        patch_dropout: float = 0.0,
        patch_alpha_bound: float | None = 1.0,
        fusion_mode: str = "deformable",
        center_resolution_index: int | None = None,
        fusion_d_model: int = 128,
        fusion_num_heads: int = 4,
        fusion_num_layers: int = 1,
        fusion_num_points: int = 4,
        fusion_ffn_ratio: float = 2.0,
        fusion_dropout: float = 0.0,
        hidden_dim: int = 128,
        num_queries: int = 100,
        num_decoder_layers: int = 6,
        num_heads_decoder: int = 8,
        num_decoder_points: int = 8,
        dim_feedforward_decoder: int = 1024,
        matcher_num_threads: int = 8,
        debug: bool = False,
    ):
        super().__init__(device=device, output_dir=output_dir)
        self.debug = bool(debug)
        self.num_classes = int(num_classes)
        self.reg_max = int(reg_max)
        self.input_resolutions = list(input_resolutions)
        self.in_ch = int(in_ch)
        self._image_hw: Tuple[int, int] = (
            max(r[0] for r in self.input_resolutions),
            max(r[1] for r in self.input_resolutions),
        )

        self.backbone = PatchSpatialBranchCrossAttentionBackbone(
            input_resolutions=input_resolutions,
            width_mult=width_mult,
            in_ch=in_ch,
            out_channels_mult=outfusion_channels_mult,
            constant_ch=constant_backbone_ch,
            patch_size=patch_size,
            patch_d_model=patch_d_model,
            patch_num_heads=patch_num_heads,
            patch_num_layers=patch_num_layers,
            patch_num_points=patch_num_points,
            patch_ffn_ratio=patch_ffn_ratio,
            patch_dropout=patch_dropout,
            patch_alpha_bound=patch_alpha_bound,
            fusion_mode=fusion_mode,
            center_resolution_index=center_resolution_index,
            fusion_d_model=fusion_d_model,
            fusion_num_heads=fusion_num_heads,
            fusion_num_layers=fusion_num_layers,
            fusion_num_points=fusion_num_points,
            fusion_ffn_ratio=fusion_ffn_ratio,
            fusion_dropout=fusion_dropout,
            debug=debug,
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

        self.detect_one2one = RTDETRHead(
            in_channels=[c3, c4, c5],
            strides=self.strides,
            num_classes=self.num_classes,
            reg_max=self.reg_max,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads_decoder,
            num_decoder_points=num_decoder_points,
            use_deformable_attention=True,
            dim_feedforward=dim_feedforward_decoder,
            dropout=0.0,
            learnt_init_query=False,
        )
        self.detect_one2one.bias_init(image_size=max(self._image_hw))
        self.criterion = RTDETRLoss(
            num_classes=self.num_classes,
            matcher_num_threads=matcher_num_threads,
        )
        self.to(self.device)

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

    def _set_frozen_features_eval(self) -> None:
        self.backbone.eval()
        self.upsample.eval()
        self.head_c3_1.eval()
        self.head_c3_2.eval()
        self.down_p3.eval()
        self.head_c3_3.eval()
        self.down_p4.eval()
        self.head_c3_4.eval()

    def freeze_feature_extractor(self) -> int:
        frozen = 0
        feature_modules = [
            self.backbone,
            self.upsample,
            self.head_c3_1,
            self.head_c3_2,
            self.down_p3,
            self.head_c3_3,
            self.down_p4,
            self.head_c3_4,
        ]
        for module in feature_modules:
            for param in module.parameters():
                param.requires_grad = False
                frozen += param.numel()
        for param in self.detect_one2one.parameters():
            param.requires_grad = True
        self._freeze_feature_extractor = True
        self._set_frozen_features_eval()
        return frozen

    def freeze_backbone(self) -> int:
        return self.freeze_feature_extractor()

    def load_frozen_backbone_weights(self, weights_path: str, device: str = "cpu") -> Tuple[list, list]:
        state_dict = torch.load(weights_path, map_location=device)
        model_state = self.state_dict()
        compatible = {}
        for key, value in state_dict.items():
            if key.startswith("detect."):
                continue
            if key in model_state and model_state[key].shape == value.shape:
                compatible[key] = value
        missing, unexpected = self.load_state_dict(compatible, strict=False)
        self.freeze_feature_extractor()
        return missing, unexpected

    def load_frozen_feature_weights(self, weights_path: str, device: str = "cpu") -> Tuple[list, list]:
        return self.load_frozen_backbone_weights(weights_path, device=device)

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "_freeze_feature_extractor", False):
            self._set_frozen_features_eval()
            self.detect_one2one.train(mode)
        return self

    def _forward_features(self, inputs: List[torch.Tensor]):
        p3, p4, p5 = self.backbone(inputs)
        p4_out = self.head_c3_1(torch.cat([self.upsample(p5), p4], dim=1))
        p3_out = self.head_c3_2(torch.cat([self.upsample(p4_out), p3], dim=1))
        p4_out2 = self.head_c3_3(torch.cat([self.down_p3(p3_out), p4_out], dim=1))
        p5_out = self.head_c3_4(torch.cat([self.down_p4(p4_out2), p5], dim=1))
        return p3_out, p4_out2, p5_out

    def forward(self, inputs: List[torch.Tensor]):
        self._validate_inputs(inputs)
        p3, p4, p5 = self._forward_features(inputs)
        return self.detect_one2one(p3, p4, p5, image_size=self._image_hw)

    def loss_from_batch(self, outputs, targets):
        batch_size = outputs["pred_logits"].shape[0]
        target_list = targets_from_yolo_tensor(targets, batch_size, outputs["pred_logits"].device)
        return self.criterion(outputs, target_list)

    def postprocess(self, outputs, conf_thres: float = 0.1, max_det: int = 300, **_):
        logits = outputs["pred_logits"]
        boxes = outputs["pred_boxes"]
        probs = logits[..., : self.num_classes].sigmoid()
        scores, labels = probs.max(dim=-1)
        image_h, image_w = float(self._image_hw[0]), float(self._image_hw[1])
        results = []
        for boxes_i, scores_i, labels_i in zip(boxes, scores, labels):
            keep = scores_i >= float(conf_thres)
            if not keep.any():
                results.append(torch.zeros((0, 6), device=logits.device, dtype=logits.dtype))
                continue
            sel = boxes_i[keep]
            xc, yc, w, h = sel.unbind(-1)
            x1 = (xc - 0.5 * w).clamp(0.0, 1.0) * image_w
            y1 = (yc - 0.5 * h).clamp(0.0, 1.0) * image_h
            x2 = (xc + 0.5 * w).clamp(0.0, 1.0) * image_w
            y2 = (yc + 0.5 * h).clamp(0.0, 1.0) * image_h
            dets = torch.stack(
                (x1, y1, x2, y2, scores_i[keep], labels_i[keep].to(logits.dtype)), dim=1
            )
            if dets.shape[0] > max_det:
                dets = dets[dets[:, 4].argsort(descending=True)[:max_det]]
            results.append(dets)
        return results

    def postprocess_for_metrics(self, outputs, conf_threshold: float = 0.1, max_det: int = 300, **kwargs):
        return self.postprocess(outputs, conf_thres=conf_threshold, max_det=max_det, **kwargs)

    def fit(
        self,
        data_dir: str,
        epochs: int = 300,
        batch_size: int = 64,
        lr: float = 1e-4,
        patience: int = 10,
        preprocessing: str = "none",
        preprocessing_kwargs=None,
        res_keys: tuple = (),
        num_workers=None,
        prefetch_factor: int = 4,
        monitor: str = "val_loss",
        save_last_every: int = 5,
        full_eval_every: int = 5,
        run_full_eval: bool = True,
        use_amp: bool = True,
        **_,
    ):
        if monitor != "val_loss":
            raise ValueError("MRYOLOPatchSpatialBranchCrossAttentionRTDETRHead only supports monitor='val_loss'.")

        ds_kw = dict(
            res_keys=res_keys,
            preprocessing=preprocessing,
            preprocessing_kwargs=preprocessing_kwargs,
        )
        train_ds = YOLODatasetFusedMultiRes(
            data_dir=os.path.join(data_dir, "train/data"),
            labels_dir=os.path.join(data_dir, "train/labels_detect"),
            **ds_kw,
        )
        val_ds = YOLODatasetFusedMultiRes(
            data_dir=os.path.join(data_dir, "val/data"),
            labels_dir=os.path.join(data_dir, "val/labels_detect"),
            **ds_kw,
        )

        pin = _supports_cuda(self.device)
        nw = _resolve_num_workers(num_workers)
        dl_kw = dict(
            batch_size=batch_size,
            pin_memory=pin,
            collate_fn=YOLODatasetFusedMultiRes.collate_fn,
            num_workers=nw,
            persistent_workers=bool(nw > 0),
        )
        if nw > 0:
            dl_kw["prefetch_factor"] = max(2, int(prefetch_factor))
        train_loader = DataLoader(train_ds, shuffle=True, **dl_kw)
        val_loader = DataLoader(val_ds, shuffle=False, **dl_kw)

        optimizer = torch.optim.AdamW(
            (p for p in self.parameters() if p.requires_grad), lr=lr, weight_decay=1e-4
        )
        scaler = torch.cuda.amp.GradScaler(
            enabled=bool(use_amp) and str(self.device).startswith("cuda")
        )

        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "train_log.csv"

        eval_runner = None
        extra_headers: list = []
        if run_full_eval:
            eval_runner = EvalRunner(
                output_dir=str(output_dir),
                cfg=EvalConfig(iou_thresh=0.5, fa_target=0.01, img_size=self._image_hw),
                class_index_to_name=load_class_index_to_name(data_dir),
            )
            extra_headers = eval_runner.extra_headers()

        with log_path.open("w", newline="") as fh:
            csv.writer(fh).writerow(
                ["epoch", "train_loss", "val_loss", "loss_cls_val", "loss_bbox_val", "loss_giou_val",
                 *extra_headers]
            )

        best_val = float("inf")
        bad_epochs = 0
        for epoch in range(1, int(epochs) + 1):
            t0 = time.perf_counter()
            train_loss, _ = self._run_epoch(train_loader, optimizer, scaler, train=True,
                                            desc=f"Epoch {epoch} train")
            val_loss, val_parts = self._run_epoch(val_loader, None, scaler, train=False,
                                                  desc=f"Epoch {epoch} val")

            should_eval = run_full_eval and (
                epoch % max(1, int(full_eval_every)) == 0 or epoch == int(epochs)
            )
            extra_values: list = []
            if run_full_eval:
                if should_eval:
                    extra_values = eval_runner.run(epoch=epoch, model=self, val_loader=val_loader)["extra_values"]
                else:
                    extra_values = [None, None, *([float("nan")] * 7), None]

            with log_path.open("a", newline="") as fh:
                csv.writer(fh).writerow([
                    epoch, train_loss, val_loss,
                    val_parts.get("loss_cls", 0.0),
                    val_parts.get("loss_bbox", 0.0),
                    val_parts.get("loss_giou", 0.0),
                    *extra_values,
                ])

            if epoch % max(1, int(save_last_every)) == 0 or epoch == int(epochs):
                torch.save(self.state_dict(), output_dir / "last.pt")
            if val_loss < best_val:
                best_val = val_loss
                bad_epochs = 0
                torch.save(self.state_dict(), output_dir / "best.pt")
            else:
                bad_epochs += 1

            print(
                f"MR-YOLO PatchSpatial CrossAttn RTDETR epoch {epoch}: "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"time={time.perf_counter() - t0:.1f}s"
            )
            if run_full_eval and should_eval:
                TrainingPlots.plot_losses(str(log_path), save_path=str(output_dir / "loss_curves.png"))
                TrainingPlots.plot_maps(str(log_path), save_path=str(output_dir / "map_curves.png"))
                TrainingPlots.plot_avg_recalls(str(log_path), save_path=str(output_dir / "avg_recall_curves.png"))
                TrainingPlots.plot_size_recalls(str(log_path), save_path=str(output_dir / "recall_size_curves.png"))
                TrainingPlots.plot_box_iou(str(log_path), save_path=str(output_dir / "box_iou_curves.png"))

            if bad_epochs >= int(patience):
                print(f"Early stopping on val_loss after {bad_epochs} epochs without improvement.")
                break

    def _run_epoch(self, loader: DataLoader, optimizer, scaler, train: bool, desc: str):
        self.train(train)
        if getattr(self, "_freeze_feature_extractor", False):
            self._set_frozen_features_eval()
        total_loss = 0.0
        parts_sum = {"loss_cls": 0.0, "loss_bbox": 0.0, "loss_giou": 0.0}
        amp_enabled = scaler.is_enabled() if scaler is not None else False
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for imgs, targets, _ in tqdm(loader, desc=desc, unit="batch"):
                imgs = imgs if isinstance(imgs, (list, tuple)) else [imgs]
                imgs = [img.to(self.device, non_blocking=_supports_cuda(self.device)) for img in imgs]
                targets = targets.to(self.device)
                if optimizer is not None:
                    optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    outputs = self(imgs)
                    loss, parts = self.loss_from_batch(outputs, targets)
                if train:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                total_loss += float(loss.detach().item())
                for key in parts_sum:
                    if key in parts:
                        parts_sum[key] += float(parts[key])
        num_batches = max(1, len(loader))
        return total_loss / num_batches, {key: value / num_batches for key, value in parts_sum.items()}
