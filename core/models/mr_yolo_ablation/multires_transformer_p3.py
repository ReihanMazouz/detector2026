from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

from ..Head.rtdetr import MSDeformAttn


def _make_2d_reference_points(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    y = torch.linspace(0.5 / height, 1.0 - 0.5 / height, height, device=device, dtype=dtype)
    x = torch.linspace(0.5 / width, 1.0 - 0.5 / width, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(1, height * width, 1, 2)


class _MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, dropout: float):
        super().__init__()
        hidden_dim = max(dim, int(dim * mlp_ratio))
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _IntraResolutionDeformableBlock(nn.Module):
    """Deformable self-attention inside one resolution grid."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_points: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MSDeformAttn(
            hidden_dim=dim,
            num_levels=1,
            num_heads=num_heads,
            num_points=num_points,
        )
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, tokens: torch.Tensor, grid_hw: Tuple[int, int]) -> torch.Tensor:
        height, width = grid_hw
        reference = _make_2d_reference_points(height, width, tokens.device, tokens.dtype)
        reference = reference.expand(tokens.shape[0], -1, -1, -1)
        x = self.norm1(tokens)
        attended = self.attn(x, reference, x, [grid_hw])
        tokens = tokens + self.drop(attended)
        return tokens + self.mlp(self.norm2(tokens))


class _InterResolutionAlignedBlock(nn.Module):
    """Attention only between physically corresponding tokens across resolutions."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, R, N, C]
        batch_size, num_resolutions, num_tokens, dim = tokens.shape
        x = tokens.permute(0, 2, 1, 3).reshape(batch_size * num_tokens, num_resolutions, dim)
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.drop(y)
        x = x + self.mlp(self.norm2(x))
        return x.reshape(batch_size, num_tokens, num_resolutions, dim).permute(0, 2, 1, 3)


class _PatchMerging(nn.Module):
    """Token-only 2x2 patch merging, Swin-style."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * in_dim)
        self.reduction = nn.Linear(4 * in_dim, out_dim, bias=False)

    def forward(self, tokens: torch.Tensor, grid_hw: Tuple[int, int]) -> tuple[torch.Tensor, Tuple[int, int]]:
        batch_size, num_tokens, channels = tokens.shape
        height, width = grid_hw
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError(f"PatchMerging requires an even grid, got {(height, width)}.")
        if num_tokens != height * width:
            raise ValueError("Token count does not match grid size.")

        x = tokens.reshape(batch_size, height, width, channels)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1).reshape(batch_size, -1, 4 * channels)
        return self.reduction(self.norm(x)), (height // 2, width // 2)


class _MRDSTStage(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_blocks: int,
        num_points: int,
        mlp_ratio: float,
        dropout: float,
    ):
        super().__init__()
        self.intra_blocks = nn.ModuleList(
            _IntraResolutionDeformableBlock(
                dim=dim,
                num_heads=num_heads,
                num_points=num_points,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        )
        self.inter_blocks = nn.ModuleList(
            _InterResolutionAlignedBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        )

    def forward(self, tokens: torch.Tensor, grid_hw: Tuple[int, int]) -> torch.Tensor:
        # tokens: [B, R, N, C]
        batch_size, num_resolutions, num_tokens, channels = tokens.shape
        for intra, inter in zip(self.intra_blocks, self.inter_blocks):
            flattened = tokens.reshape(batch_size * num_resolutions, num_tokens, channels)
            flattened = intra(flattened, grid_hw)
            tokens = flattened.reshape(batch_size, num_resolutions, num_tokens, channels)
            tokens = inter(tokens)
        return tokens


class MRDeformableSpectralTransformerP3(nn.Module):
    """
    Transformer multi-resolution backbone up to a fused P3 token map.

    Inputs are spectrograms at different resolutions. Each input is patchified
    with an anisotropic patch size so all resolutions share the same token grid.
    Stages alternate deformable intra-resolution attention and aligned
    inter-resolution attention, then token-only patch merging.
    """

    def __init__(
        self,
        input_resolutions: Sequence[Tuple[int, int]],
        patch_sizes: Sequence[Tuple[int, int]],
        in_ch: int = 1,
        dims: Sequence[int] = (96, 192, 384),
        num_heads: Sequence[int] = (3, 6, 12),
        depths: Sequence[int] = (1, 1, 1),
        num_points: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        fusion: str = "attention_pool",
    ):
        super().__init__()
        if len(input_resolutions) != len(patch_sizes):
            raise ValueError("input_resolutions and patch_sizes must have the same length.")
        if len(dims) != 3 or len(num_heads) != 3 or len(depths) != 3:
            raise ValueError("dims, num_heads and depths must describe exactly P1, P2 and P3.")
        if fusion not in {"attention_pool", "mean"}:
            raise ValueError("fusion must be 'attention_pool' or 'mean'.")

        self.input_resolutions = [tuple(map(int, res)) for res in input_resolutions]
        self.patch_sizes = [tuple(map(int, patch)) for patch in patch_sizes]
        self.num_resolutions = len(self.input_resolutions)
        self.in_ch = int(in_ch)
        self.dims = tuple(int(dim) for dim in dims)
        self.out_channels = self.dims[-1]
        self.fusion = fusion

        grids = []
        for resolution, patch in zip(self.input_resolutions, self.patch_sizes):
            height, width = resolution
            patch_h, patch_w = patch
            if height % patch_h != 0 or width % patch_w != 0:
                raise ValueError(f"Patch size {patch} does not divide resolution {resolution}.")
            grids.append((height // patch_h, width // patch_w))
        if len(set(grids)) != 1:
            raise ValueError(f"All anisotropic patch sizes must produce the same grid, got {grids}.")
        self.p1_grid = grids[0]
        if self.p1_grid[0] % 4 != 0 or self.p1_grid[1] % 4 != 0:
            raise ValueError(f"P1 grid must be divisible by 4 to build P2 and P3, got {self.p1_grid}.")
        self.p2_grid = (self.p1_grid[0] // 2, self.p1_grid[1] // 2)
        self.p3_grid = (self.p2_grid[0] // 2, self.p2_grid[1] // 2)

        self.patch_embeds = nn.ModuleList(
            nn.Conv2d(in_ch, self.dims[0], kernel_size=patch, stride=patch)
            for patch in self.patch_sizes
        )
        self.pos_embed = nn.ParameterList(
            nn.Parameter(torch.zeros(1, grid[0] * grid[1], dim))
            for grid, dim in zip((self.p1_grid, self.p2_grid, self.p3_grid), self.dims)
        )
        self.res_embed = nn.ParameterList(
            nn.Parameter(torch.zeros(self.num_resolutions, dim))
            for dim in self.dims
        )
        self.stages = nn.ModuleList(
            _MRDSTStage(
                dim=self.dims[index],
                num_heads=int(num_heads[index]),
                num_blocks=int(depths[index]),
                num_points=num_points,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for index in range(3)
        )
        self.merges = nn.ModuleList(
            [
                _PatchMerging(self.dims[0], self.dims[1]),
                _PatchMerging(self.dims[1], self.dims[2]),
            ]
        )
        if self.fusion == "attention_pool":
            self.fusion_query = nn.Parameter(torch.zeros(1, 1, self.dims[-1]))
            self.fusion_norm = nn.LayerNorm(self.dims[-1])
            self.fusion_attn = nn.MultiheadAttention(
                self.dims[-1],
                int(num_heads[-1]),
                dropout=dropout,
                batch_first=True,
            )
        else:
            self.fusion_query = None
            self.fusion_norm = None
            self.fusion_attn = None
        self.output_norm = nn.LayerNorm(self.dims[-1])

    def _add_level_embeddings(self, tokens: torch.Tensor, level: int) -> torch.Tensor:
        pos = self.pos_embed[level].to(device=tokens.device, dtype=tokens.dtype).unsqueeze(1)
        res = self.res_embed[level].to(device=tokens.device, dtype=tokens.dtype).view(1, self.num_resolutions, 1, -1)
        return tokens + pos + res

    def _validate_inputs(self, inputs: Sequence[torch.Tensor]) -> None:
        if len(inputs) != self.num_resolutions:
            raise ValueError(f"Expected {self.num_resolutions} inputs, got {len(inputs)}.")
        for index, (x, resolution) in enumerate(zip(inputs, self.input_resolutions)):
            if x.dim() != 4:
                raise ValueError(f"Input #{index} must be 4D, got {x.dim()}D.")
            if x.shape[1] != self.in_ch:
                raise ValueError(f"Input #{index} has {x.shape[1]} channels, expected {self.in_ch}.")
            if tuple(x.shape[-2:]) != resolution:
                raise ValueError(f"Input #{index} has shape {tuple(x.shape[-2:])}, expected {resolution}.")

    def _fuse_p3(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, R, N, C]
        if self.fusion == "mean":
            return tokens.mean(dim=1)

        batch_size, num_resolutions, num_tokens, channels = tokens.shape
        memory = tokens.permute(0, 2, 1, 3).reshape(batch_size * num_tokens, num_resolutions, channels)
        query = self.fusion_query.to(device=tokens.device, dtype=tokens.dtype).expand(batch_size * num_tokens, -1, -1)
        memory = self.fusion_norm(memory)
        fused, _ = self.fusion_attn(query, memory, memory, need_weights=False)
        return fused.reshape(batch_size, num_tokens, channels)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        self._validate_inputs(inputs)

        per_resolution = []
        for patch_embed, x in zip(self.patch_embeds, inputs):
            z = patch_embed(x).flatten(2).transpose(1, 2)
            per_resolution.append(z)
        tokens = torch.stack(per_resolution, dim=1)
        grid_hw = self.p1_grid

        for level, stage in enumerate(self.stages):
            tokens = self._add_level_embeddings(tokens, level)
            tokens = stage(tokens, grid_hw)
            if level < 2:
                _, num_resolutions, _, _ = tokens.shape
                merged = []
                next_grid = None
                for resolution_index in range(num_resolutions):
                    z, next_grid = self.merges[level](tokens[:, resolution_index], grid_hw)
                    merged.append(z)
                tokens = torch.stack(merged, dim=1)
                grid_hw = next_grid

        p3_tokens = self.output_norm(self._fuse_p3(tokens))
        batch_size, num_tokens, channels = p3_tokens.shape
        height, width = grid_hw
        if num_tokens != height * width:
            raise ValueError("P3 token count does not match P3 grid.")
        return p3_tokens.transpose(1, 2).reshape(batch_size, channels, height, width)
