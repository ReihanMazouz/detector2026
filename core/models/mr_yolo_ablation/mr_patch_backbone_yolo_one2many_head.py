"""Multi-resolution isotropic patch backbone with a single-scale YOLO one-to-many head."""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...nn.convs import Conv
from ...utils.loss import YOLODetectionLoss
from ..Head.detect import Detect
from ..Head.rtdetr import MSDeformAttn
from ..base import BaseModel
from .mr_vit_patch_detector import _sinusoidal_2d


def _normalized_patch_boxes(
    shape: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    h, w = shape
    y0 = torch.arange(h, device=device, dtype=dtype) / h
    x0 = torch.arange(w, device=device, dtype=dtype) / w
    yy0, xx0 = torch.meshgrid(y0, x0, indexing="ij")
    yy1 = yy0 + (1.0 / h)
    xx1 = xx0 + (1.0 / w)
    return torch.stack([xx0, yy0, xx1, yy1], dim=-1).reshape(-1, 4)


def _normalized_patch_centers(
    shape: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    boxes = _normalized_patch_boxes(shape, device, dtype)
    return 0.5 * (boxes[:, :2] + boxes[:, 2:])


class RestrictedInterResolutionAttention(nn.Module):
    """Attention only over patches whose normalized time-frequency surfaces overlap."""

    def __init__(self, d_model: int, num_heads: int, num_neighbors: int, dropout: float):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads
        self.num_neighbors = int(num_neighbors)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self._index_cache: dict[tuple[Tuple[int, int], Tuple[int, int], int, str], torch.Tensor] = {}

    def _overlap_indices(
        self,
        target_shape: Tuple[int, int],
        source_shape: Tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        key = (tuple(target_shape), tuple(source_shape), self.num_neighbors, str(device))
        cached = self._index_cache.get(key)
        if cached is not None:
            return cached

        target_boxes = _normalized_patch_boxes(target_shape, device, torch.float32)
        source_boxes = _normalized_patch_boxes(source_shape, device, torch.float32)
        target_centers = 0.5 * (target_boxes[:, :2] + target_boxes[:, 2:])
        source_centers = 0.5 * (source_boxes[:, :2] + source_boxes[:, 2:])

        left_top = torch.maximum(target_boxes[:, None, :2], source_boxes[None, :, :2])
        right_bottom = torch.minimum(target_boxes[:, None, 2:], source_boxes[None, :, 2:])
        inter_wh = (right_bottom - left_top).clamp_min(0.0)
        inter_area = inter_wh[..., 0] * inter_wh[..., 1]

        source_area = (source_boxes[:, 2] - source_boxes[:, 0]) * (source_boxes[:, 3] - source_boxes[:, 1])
        overlap_score = inter_area / source_area.clamp_min(1e-12)
        distances = torch.cdist(target_centers, source_centers, p=2)

        k = min(self.num_neighbors, source_boxes.shape[0])
        ranked_score = overlap_score.masked_fill(overlap_score <= 0.0, -1.0)
        if k < source_boxes.shape[0]:
            indices = torch.topk(ranked_score, k=k, dim=1, largest=True).indices
            no_overlap = ranked_score.max(dim=1).values <= 0.0
            if no_overlap.any():
                nearest = torch.topk(distances[no_overlap], k=k, dim=1, largest=False).indices
                indices[no_overlap] = nearest
        else:
            overlap_order = torch.argsort(ranked_score, dim=1, descending=True)
            nearest_order = torch.argsort(distances, dim=1)
            no_overlap = ranked_score.max(dim=1).values <= 0.0
            indices = overlap_order
            indices[no_overlap] = nearest_order[no_overlap]

        self._index_cache[key] = indices
        return indices

    def forward(
        self,
        target: torch.Tensor,
        sources: List[torch.Tensor],
        target_shape: Tuple[int, int],
        source_shapes: List[Tuple[int, int]],
    ) -> torch.Tensor:
        batch_size, num_target, _ = target.shape
        query = self.q_proj(target).view(batch_size, num_target, self.num_heads, self.head_dim)
        updates = []

        for source, source_shape in zip(sources, source_shapes):
            neighbor_idx = self._overlap_indices(target_shape, source_shape, target.device)
            neighbors = source[:, neighbor_idx, :]  # [B, N_target, M, D]
            key = self.k_proj(neighbors).view(
                batch_size,
                num_target,
                neighbor_idx.shape[1],
                self.num_heads,
                self.head_dim,
            )
            value = self.v_proj(neighbors).view(
                batch_size,
                num_target,
                neighbor_idx.shape[1],
                self.num_heads,
                self.head_dim,
            )
            logits = (query.unsqueeze(2) * key).sum(dim=-1) * (self.head_dim ** -0.5)
            weights = self.drop(logits.softmax(dim=2))
            update = (weights.unsqueeze(-1) * value).sum(dim=2)
            updates.append(update.reshape(batch_size, num_target, self.d_model))

        if not updates:
            return torch.zeros_like(target)
        return self.out(torch.stack(updates, dim=0).sum(dim=0))


class IsotropicRestrictedPatchEncoderLayer(nn.Module):
    """Intra-resolution deformable attention, restricted inter-resolution attention, FFN."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_intra_points: int,
        num_inter_neighbors: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.intra_attn = MSDeformAttn(d_model, num_levels=1, num_heads=num_heads, num_points=num_intra_points)
        self.inter_attn = RestrictedInterResolutionAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_neighbors=num_inter_neighbors,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _reference_points(
        shape: Tuple[int, int],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        centers = _normalized_patch_centers(shape, device, dtype)
        return centers.view(1, -1, 1, 2).expand(batch_size, -1, -1, -1)

    def forward(
        self,
        tokens_by_res: List[torch.Tensor],
        spatial_shapes: List[Tuple[int, int]],
    ) -> List[torch.Tensor]:
        intra_tokens = []
        for tokens, shape in zip(tokens_by_res, spatial_shapes):
            refs = self._reference_points(shape, tokens.shape[0], tokens.device, tokens.dtype)
            update = self.intra_attn(tokens, refs, tokens, [shape])
            intra_tokens.append(self.norm1(tokens + self.drop(update)))

        inter_tokens = []
        for index, tokens in enumerate(intra_tokens):
            sources = [src for src_index, src in enumerate(intra_tokens) if src_index != index]
            source_shapes = [
                shape for shape_index, shape in enumerate(spatial_shapes) if shape_index != index
            ]
            update = self.inter_attn(tokens, sources, spatial_shapes[index], source_shapes)
            inter_tokens.append(self.norm2(tokens + self.drop(update)))

        output = []
        for tokens in inter_tokens:
            update = self.linear2(self.drop(F.relu(self.linear1(tokens))))
            output.append(self.norm3(tokens + self.drop(update)))
        return output


class IsotropicRestrictedPatchBackbone(nn.Module):
    """Backbone V2: isotropic 8x8 patches and restricted multi-resolution attention."""

    def __init__(
        self,
        input_resolutions: List[Tuple[int, int]],
        in_ch: int = 1,
        d_model: int = 128,
        patch_size: int = 8,
        num_layers: int = 3,
        num_heads: int = 4,
        num_intra_points: int = 8,
        num_inter_neighbors: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        p3_hw: Tuple[int, int] = (32, 32),
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4 for sinusoidal 2D position encoding.")

        self.input_resolutions = [tuple(res) for res in input_resolutions]
        self.in_ch = int(in_ch)
        self.d_model = int(d_model)
        self.patch_size = int(patch_size)
        self.p3_hw = tuple(int(v) for v in p3_hw)
        self.patch_shapes = self._patch_shapes(self.input_resolutions, self.patch_size)

        self.conv_stems = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_ch, 16, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.GELU(),
            )
            for _ in self.input_resolutions
        )
        self.patch_embeds = nn.ModuleList(
            nn.Conv2d(16, d_model, kernel_size=self.patch_size, stride=self.patch_size)
            for _ in self.input_resolutions
        )
        self.res_embed = nn.Parameter(torch.zeros(len(self.input_resolutions), 1, d_model))
        nn.init.trunc_normal_(self.res_embed, std=0.02)

        self.encoder = nn.ModuleList(
            IsotropicRestrictedPatchEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                num_intra_points=num_intra_points,
                num_inter_neighbors=num_inter_neighbors,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )

        fusion_channels = d_model * len(self.input_resolutions)
        self.p3_proj = nn.Sequential(
            Conv(fusion_channels, d_model, k=1, s=1),
            Conv(d_model, d_model, k=3, s=1),
        )
        self.out_channels = (d_model,)

    @staticmethod
    def _patch_shapes(
        input_resolutions: List[Tuple[int, int]],
        patch_size: int,
    ) -> List[Tuple[int, int]]:
        shapes = []
        for h, w in input_resolutions:
            if h % patch_size != 0 or w % patch_size != 0:
                raise ValueError(f"Resolution {(h, w)} is not divisible by patch_size={patch_size}.")
            shapes.append((h // patch_size, w // patch_size))
        return shapes

    def _tokenize(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        tokens_by_res = []
        for index, (x, stem, embed, shape) in enumerate(
            zip(inputs, self.conv_stems, self.patch_embeds, self.patch_shapes)
        ):
            x = embed(stem(x))
            if tuple(x.shape[-2:]) != tuple(shape):
                raise RuntimeError(f"Patch map #{index} has shape {tuple(x.shape[-2:])}, expected {shape}.")
            tokens = x.flatten(2).transpose(1, 2)
            pos = _sinusoidal_2d(shape[0], shape[1], self.d_model, x.device, x.dtype)
            tokens_by_res.append(tokens + pos + self.res_embed[index].to(dtype=tokens.dtype))
        return tokens_by_res

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        tokens_by_res = self._tokenize(inputs)
        for layer in self.encoder:
            tokens_by_res = layer(tokens_by_res, self.patch_shapes)

        maps = []
        for tokens, shape in zip(tokens_by_res, self.patch_shapes):
            feature = tokens.transpose(1, 2).reshape(tokens.shape[0], self.d_model, shape[0], shape[1])
            feature = F.interpolate(feature, size=self.p3_hw, mode="bilinear", align_corners=False)
            maps.append(feature)

        fused = torch.cat(maps, dim=1)
        return self.p3_proj(fused)


class MRPatchBackboneYOLOOne2ManyHead(BaseModel):
    """MR-ViT V2 backbone followed by a single-scale YOLO one-to-many Detect head."""

    def __init__(
        self,
        input_resolutions: List[Tuple[int, int]],
        output_dir: str,
        num_classes: int = 20,
        reg_max: int = 16,
        device: str = "cuda:0",
        in_ch: int = 1,
        d_model: int = 128,
        patch_size: int = 8,
        num_encoder_layers: int = 3,
        num_heads: int = 4,
        num_intra_points: int = 8,
        num_inter_neighbors: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        p3_hw: Tuple[int, int] = (32, 32),
        stride: int = 32,
    ):
        super().__init__(device=device, output_dir=output_dir)
        self.input_resolutions = [tuple(res) for res in input_resolutions]
        self.num_classes = int(num_classes)
        self.reg_max = int(reg_max)
        self.in_ch = int(in_ch)
        self.strides = [int(stride)]

        self.backbone = IsotropicRestrictedPatchBackbone(
            input_resolutions=self.input_resolutions,
            in_ch=in_ch,
            d_model=d_model,
            patch_size=patch_size,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            num_intra_points=num_intra_points,
            num_inter_neighbors=num_inter_neighbors,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            p3_hw=p3_hw,
        )
        self.detect = Detect(
            in_channels=list(self.backbone.out_channels),
            strides=self.strides,
            num_classes=self.num_classes,
            reg_max=self.reg_max,
        )
        max_input_dim = max(max(h, w) for h, w in self.input_resolutions)
        self.detect.bias_init(image_size=max_input_dim)
        self.criterion = YOLODetectionLoss(
            num_classes=self.num_classes,
            strides=self.strides,
            reg_max=self.reg_max,
            device=self.device,
        )
        self.to(self.device)

    def _validate_inputs(self, inputs: List[torch.Tensor]) -> None:
        if len(inputs) != len(self.input_resolutions):
            raise ValueError(f"Expected {len(self.input_resolutions)} inputs, got {len(inputs)}.")
        for index, (x, resolution) in enumerate(zip(inputs, self.input_resolutions)):
            if x.dim() != 4:
                raise ValueError(f"Input #{index} must be 4D, got {x.dim()}D.")
            if x.shape[1] != self.in_ch:
                raise ValueError(f"Input #{index} has {x.shape[1]} channels, expected {self.in_ch}.")
            if tuple(x.shape[-2:]) != tuple(resolution):
                raise ValueError(
                    f"Input #{index} has shape {tuple(x.shape[-2:])}, expected {tuple(resolution)}."
                )

    def forward(self, inputs: List[torch.Tensor]):
        self._validate_inputs(inputs)
        p3 = self.backbone(inputs)
        return self.detect(p3)
