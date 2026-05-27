from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..Head.rtdetr import MSDeformAttn


def _reference_points(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    y = torch.linspace(0.5 / height, 1.0 - 0.5 / height, height, device=device, dtype=dtype)
    x = torch.linspace(0.5 / width, 1.0 - 0.5 / width, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(1, height * width, 1, 2)


class _MLP(nn.Module):
    def __init__(self, dim: int, ratio: float, dropout: float):
        super().__init__()
        hidden = max(dim, int(dim * ratio))
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _IntraDeformableBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_points: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MSDeformAttn(
            hidden_dim=d_model,
            num_levels=1,
            num_heads=num_heads,
            num_points=num_points,
        )
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = _MLP(d_model, mlp_ratio, dropout)

    def forward(self, tokens: torch.Tensor, grid_hw: Tuple[int, int]) -> torch.Tensor:
        reference = _reference_points(*grid_hw, device=tokens.device, dtype=tokens.dtype)
        reference = reference.expand(tokens.shape[0], -1, -1, -1)
        x = self.norm1(tokens)
        tokens = tokens + self.drop(self.attn(x, reference, x, [grid_hw]))
        return tokens + self.mlp(self.norm2(tokens))


class _InterResolutionAlignedBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = _MLP(d_model, mlp_ratio, dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # [B, R, N, D] -> [B*N, R, D]
        batch, resolutions, num_tokens, dim = tokens.shape
        x = tokens.permute(0, 2, 1, 3).reshape(batch * num_tokens, resolutions, dim)
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.drop(y)
        x = x + self.mlp(self.norm2(x))
        return x.reshape(batch, num_tokens, resolutions, dim).permute(0, 2, 1, 3)


class MRPatchSpatialAttentionBlock(nn.Module):
    """
    Multi-resolution patch attention that outputs pixel-wise spatial gates.

    Tokens are used only to compute an attention map. The input feature maps are
    modulated residually and keep their original shapes.
    """

    def __init__(
        self,
        input_channels: Sequence[int],
        input_resolutions: Sequence[Tuple[int, int]],
        patch_sizes: Sequence[Tuple[int, int]],
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 1,
        num_points: int = 16,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if not (len(input_channels) == len(input_resolutions) == len(patch_sizes)):
            raise ValueError("input_channels, input_resolutions and patch_sizes must have the same length.")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")

        self.input_channels = tuple(int(ch) for ch in input_channels)
        self.input_resolutions = [tuple(map(int, res)) for res in input_resolutions]
        self.patch_sizes = [tuple(map(int, patch)) for patch in patch_sizes]
        self.num_resolutions = len(self.input_channels)
        self.d_model = int(d_model)

        grids = []
        for resolution, patch in zip(self.input_resolutions, self.patch_sizes):
            height, width = resolution
            patch_h, patch_w = patch
            if height % patch_h != 0 or width % patch_w != 0:
                raise ValueError(f"Patch size {patch} does not divide resolution {resolution}.")
            grids.append((height // patch_h, width // patch_w))
        if len(set(grids)) != 1:
            raise ValueError(f"All patch sizes must produce the same latent grid, got {grids}.")
        self.latent_grid = grids[0]
        self.num_tokens = self.latent_grid[0] * self.latent_grid[1]

        self.patch_embeds = nn.ModuleList(
            nn.Conv2d(ch, d_model, kernel_size=patch, stride=patch)
            for ch, patch in zip(self.input_channels, self.patch_sizes)
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, self.num_tokens, d_model))
        self.res_embed = nn.Parameter(torch.zeros(1, self.num_resolutions, 1, d_model))
        self.intra_blocks = nn.ModuleList(
            _IntraDeformableBlock(d_model, num_heads, num_points, mlp_ratio, dropout)
            for _ in range(num_layers)
        )
        self.inter_blocks = nn.ModuleList(
            _InterResolutionAlignedBlock(d_model, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.spatial_heads = nn.ModuleList(nn.Linear(d_model, 1) for _ in self.input_channels)
        self.alpha = nn.Parameter(torch.zeros(self.num_resolutions))

    def _validate_inputs(self, features: Sequence[torch.Tensor]) -> None:
        if len(features) != self.num_resolutions:
            raise ValueError(f"Expected {self.num_resolutions} feature maps, got {len(features)}.")
        for index, (feature, channels, resolution) in enumerate(
            zip(features, self.input_channels, self.input_resolutions)
        ):
            if feature.dim() != 4:
                raise ValueError(f"Feature #{index} must be 4D, got {feature.dim()}D.")
            if feature.shape[1] != channels:
                raise ValueError(f"Feature #{index} has {feature.shape[1]} channels, expected {channels}.")
            if tuple(feature.shape[-2:]) != resolution:
                raise ValueError(f"Feature #{index} has shape {tuple(feature.shape[-2:])}, expected {resolution}.")

    def forward(self, features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        self._validate_inputs(features)
        tokens = []
        for patch_embed, feature in zip(self.patch_embeds, features):
            tokens.append(patch_embed(feature).flatten(2).transpose(1, 2))
        x = torch.stack(tokens, dim=1)
        x = x + self.pos_embed.to(device=x.device, dtype=x.dtype) + self.res_embed.to(device=x.device, dtype=x.dtype)

        batch, resolutions, num_tokens, dim = x.shape
        for intra, inter in zip(self.intra_blocks, self.inter_blocks):
            flat = x.reshape(batch * resolutions, num_tokens, dim)
            flat = intra(flat, self.latent_grid)
            x = flat.reshape(batch, resolutions, num_tokens, dim)
            x = inter(x)
        x = self.output_norm(x)

        outputs = []
        height, width = self.latent_grid
        for index, (feature, head) in enumerate(zip(features, self.spatial_heads)):
            attn = head(x[:, index]).transpose(1, 2).reshape(batch, 1, height, width)
            attn = F.interpolate(attn, size=feature.shape[-2:], mode="bilinear", align_corners=False)
            gate = 2.0 * torch.sigmoid(attn) - 1.0
            outputs.append(feature + self.alpha[index].to(dtype=feature.dtype) * gate * feature)
        return outputs
