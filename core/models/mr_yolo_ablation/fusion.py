from typing import List, Sequence

import torch
import torch.nn as nn

from ..Head.rtdetr import MSDeformAttn


class _CrossAttentionBlock(nn.Module):
    """Cross-attention block without intra-resolution self-attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm_query = nn.LayerNorm(d_model)
        self.norm_memory = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)

        hidden_dim = max(d_model, int(d_model * ffn_ratio))
        self.ffn = None
        if ffn_ratio > 0:
            self.ffn = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(
            self.norm_query(query),
            self.norm_memory(memory),
            self.norm_memory(memory),
            need_weights=False,
        )
        query = query + self.attn_dropout(attn_out)
        if self.ffn is not None:
            query = query + self.ffn(query)
        return query


class _DeformableAttentionBlock(nn.Module):
    """Multi-resolution deformable attention block without resizing feature maps."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_points: int = 4,
        num_levels: int = 3,
        ffn_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm_query = nn.LayerNorm(d_model)
        self.norm_memory = nn.LayerNorm(d_model)
        self.attn = MSDeformAttn(
            hidden_dim=d_model,
            num_levels=num_levels,
            num_heads=num_heads,
            num_points=num_points,
        )
        self.attn_dropout = nn.Dropout(dropout)

        hidden_dim = max(d_model, int(d_model * ffn_ratio))
        self.ffn = None
        if ffn_ratio > 0:
            self.ffn = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        memory: torch.Tensor,
        value_shapes: Sequence[tuple[int, int]],
    ) -> torch.Tensor:
        attn_out = self.attn(
            self.norm_query(query),
            reference_points,
            self.norm_memory(memory),
            value_shapes,
        )
        query = query + self.attn_dropout(attn_out)
        if self.ffn is not None:
            query = query + self.ffn(query)
        return query


class InterResolutionCrossAttentionFusion(nn.Module):
    """
    Fuse multi-resolution feature maps by cross-attention.

    The target queries come from the central resolution. Keys and values come
    from all resolutions. No self-attention is applied inside a resolution.
    """

    VALID_MODES = {"global", "deformable"}

    def __init__(
        self,
        input_channels: Sequence[int],
        out_channels: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 1,
        ffn_ratio: float = 2.0,
        dropout: float = 0.0,
        fusion_mode: str = "deformable",
        num_points: int = 4,
        center_resolution_index: int | None = None,
    ):
        super().__init__()
        if not input_channels:
            raise ValueError("input_channels must contain at least one resolution.")
        if fusion_mode not in self.VALID_MODES:
            raise ValueError(f"fusion_mode must be one of {sorted(self.VALID_MODES)}.")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")

        self.input_channels = tuple(int(ch) for ch in input_channels)
        self.num_resolutions = len(self.input_channels)
        self.out_channels = int(out_channels)
        self.d_model = int(d_model)
        self.fusion_mode = fusion_mode
        self.num_points = int(num_points)
        if center_resolution_index is None:
            center_resolution_index = self.num_resolutions // 2
        if not 0 <= int(center_resolution_index) < self.num_resolutions:
            raise ValueError(
                f"center_resolution_index must be in [0, {self.num_resolutions - 1}], "
                f"got {center_resolution_index}."
            )
        self.center_resolution_index = int(center_resolution_index)

        self.input_projections = nn.ModuleList(
            nn.Conv2d(channels, d_model, kernel_size=1)
            for channels in self.input_channels
        )
        self.position_proj = nn.Linear(2, d_model)
        self.resolution_embed = nn.Parameter(torch.zeros(self.num_resolutions, d_model))
        if self.fusion_mode == "global":
            self.blocks = nn.ModuleList(
                _CrossAttentionBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            )
            self.deformable_blocks = nn.ModuleList()
        else:
            self.blocks = nn.ModuleList()
            self.deformable_blocks = nn.ModuleList(
                _DeformableAttentionBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    num_points=self.num_points,
                    num_levels=self.num_resolutions,
                    ffn_ratio=ffn_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            )
        self.output_projection = nn.Conv2d(d_model, out_channels, kernel_size=1)

    def _coords(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        y = torch.linspace(0.5 / height, 1.0 - 0.5 / height, height, device=device, dtype=dtype)
        x = torch.linspace(0.5 / width, 1.0 - 0.5 / width, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), dim=-1).reshape(1, height * width, 2)

    def _position_tokens(
        self,
        height: int,
        width: int,
        resolution_index: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        y = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack((xx, yy), dim=-1).reshape(1, height * width, 2)
        pos = self.position_proj(coords)
        res = self.resolution_embed[resolution_index].to(device=device, dtype=dtype).view(1, 1, -1)
        return pos + res

    def _tokens_from_feature(self, feature: torch.Tensor, resolution_index: int) -> torch.Tensor:
        batch_size, _, height, width = feature.shape
        tokens = feature.flatten(2).transpose(1, 2)
        return tokens + self._position_tokens(
            height=height,
            width=width,
            resolution_index=resolution_index,
            device=feature.device,
            dtype=feature.dtype,
        ).expand(batch_size, -1, -1)

    def _project_inputs(self, features: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        if len(features) != self.num_resolutions:
            raise ValueError(
                f"Expected {self.num_resolutions} feature maps, got {len(features)}."
            )

        projected = []
        batch_size = None
        for index, (feature, projection) in enumerate(zip(features, self.input_projections)):
            if feature.dim() != 4:
                raise ValueError(f"Feature #{index} must be 4D, got {feature.dim()}D.")
            if feature.shape[1] != self.input_channels[index]:
                raise ValueError(
                    f"Feature #{index} has {feature.shape[1]} channels, "
                    f"expected {self.input_channels[index]}."
                )
            if batch_size is None:
                batch_size = int(feature.shape[0])
            elif int(feature.shape[0]) != batch_size:
                raise ValueError("All feature maps must have the same batch size.")
            projected.append(projection(feature))
        return projected

    def _global_fusion(self, projected: Sequence[torch.Tensor]) -> torch.Tensor:
        target = projected[self.center_resolution_index]
        query = self._tokens_from_feature(target, self.center_resolution_index)
        memory = torch.cat(
            [
                self._tokens_from_feature(feature, resolution_index)
                for resolution_index, feature in enumerate(projected)
            ],
            dim=1,
        )
        for block in self.blocks:
            query = block(query, memory)
        return query

    def _deformable_fusion(self, projected: Sequence[torch.Tensor]) -> torch.Tensor:
        target = projected[self.center_resolution_index]
        batch_size, _, target_h, target_w = target.shape
        query = self._tokens_from_feature(target, self.center_resolution_index)

        memory = torch.cat(
            [
                self._tokens_from_feature(feature, resolution_index)
                for resolution_index, feature in enumerate(projected)
            ],
            dim=1,
        )
        value_shapes = [tuple(feature.shape[-2:]) for feature in projected]
        reference_points = self._coords(target_h, target_w, target.device, target.dtype)
        reference_points = reference_points.expand(batch_size, -1, -1).unsqueeze(2)
        reference_points = reference_points.repeat(1, 1, self.num_resolutions, 1)

        for block in self.deformable_blocks:
            query = block(query, reference_points, memory, value_shapes)
        return query

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        projected = self._project_inputs(features)
        target = projected[self.center_resolution_index]
        batch_size, _, target_h, target_w = target.shape

        if self.fusion_mode == "global":
            fused_tokens = self._global_fusion(projected)
        else:
            fused_tokens = self._deformable_fusion(projected)

        fused = fused_tokens.transpose(1, 2).reshape(batch_size, self.d_model, target_h, target_w)
        return self.output_projection(fused).contiguous()
