from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .SwinBackbone import (
    DropPath,
    PatchEmbed,
    PatchMerging,
    WindowAttention,
    _window_partition,
    _window_reverse,
)


class DeformableAttention(nn.Module):
    """DAT-style deformable self-attention (Xia et al., CVPR 2022).

    All HW queries attend on r² content-adaptive sampled key/value tokens.
    Offsets are bounded by tanh to prevent divergence in early training.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        num_points: int = 7,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        offset_scale: float = 0.5,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads.")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.num_points = int(num_points)
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.offset_scale = float(offset_scale)

        # Lightweight offset network: pool to r×r then predict 2D offsets
        self.offset_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(num_points),
            nn.Conv2d(dim, dim, kernel_size=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, 2, kernel_size=1),
        )
        nn.init.zeros_(self.offset_net[-1].weight)
        nn.init.zeros_(self.offset_net[-1].bias)

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B×H×W×C (BHWC layout, same as Swin)
        B, H, W, C = x.shape
        r = self.num_points

        x_nchw = x.permute(0, 3, 1, 2).contiguous()  # B×C×H×W

        # Reference grid: r×r uniform points in [-1, 1]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, r, device=x.device),
            torch.linspace(-1.0, 1.0, r, device=x.device),
            indexing="ij",
        )
        ref_grid = torch.stack([grid_x, grid_y], dim=-1)  # r×r×2
        ref_grid = ref_grid.unsqueeze(0).expand(B, -1, -1, -1)  # B×r×r×2

        # Predict and bound offsets
        offsets = self.offset_net(x_nchw)               # B×2×r×r
        offsets = offsets.permute(0, 2, 3, 1)           # B×r×r×2
        offsets = torch.tanh(offsets) * self.offset_scale

        # Deformed sampling positions in [-1, 1]
        deformed = (ref_grid + offsets).clamp(-1.0, 1.0)  # B×r×r×2

        # Bilinear sample at deformed positions
        x_sampled = F.grid_sample(
            x_nchw, deformed, mode="bilinear", padding_mode="zeros", align_corners=True
        )  # B×C×r×r
        x_sampled = x_sampled.permute(0, 2, 3, 1).reshape(B, r * r, C)  # B×r²×C

        # Queries from all tokens; K/V from deformed positions
        x_flat = x.reshape(B, H * W, C)
        q = self.q(x_flat).reshape(B, H * W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        kv = self.kv(x_sampled).reshape(B, r * r, 2, self.num_heads, C // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # B×heads×r²×head_dim

        attn = (q * self.scale) @ k.transpose(-2, -1)  # B×heads×HW×r²
        attn = self.attn_drop(attn.softmax(dim=-1))

        out = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        out = self.proj_drop(self.proj(out))
        return out.reshape(B, H, W, C)


class DATBlock(nn.Module):
    """Single DAT block: either local window attention or deformable attention.

    Blocks alternate: even index → local window, odd index → deformable.
    This replicates the interleaving strategy from the original DAT paper.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        use_deformable: bool = False,
        num_points: int = 7,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        offset_scale: float = 0.5,
    ):
        super().__init__()
        self.use_deformable = bool(use_deformable)
        self.window_size = int(window_size)
        self.norm1 = nn.LayerNorm(dim)
        if use_deformable:
            self.attn: nn.Module = DeformableAttention(
                dim=dim,
                num_heads=num_heads,
                num_points=num_points,
                attn_drop=attn_drop,
                proj_drop=drop,
                offset_scale=offset_scale,
            )
        else:
            self.attn = WindowAttention(
                dim=dim,
                window_size=window_size,
                num_heads=num_heads,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)

        if self.use_deformable:
            x = self.attn(x)
        else:
            pad_h = (self.window_size - H % self.window_size) % self.window_size
            pad_w = (self.window_size - W % self.window_size) % self.window_size
            if pad_h or pad_w:
                x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            hp, wp = x.shape[1], x.shape[2]
            windows = _window_partition(x, self.window_size)
            attn_windows = self.attn(windows)
            x = _window_reverse(attn_windows, self.window_size, hp, wp, B)
            if pad_h or pad_w:
                x = x[:, :H, :W, :].contiguous()

        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class DATStage(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        num_points: int,
        mlp_ratio: float,
        drop: float,
        attn_drop: float,
        drop_path: Sequence[float],
        downsample: bool = True,
        offset_scale: float = 0.5,
    ):
        super().__init__()
        self.downsample = PatchMerging(dim) if downsample else None
        block_dim = dim * 2 if downsample else dim
        self.blocks = nn.ModuleList(
            DATBlock(
                dim=block_dim,
                num_heads=num_heads,
                window_size=window_size,
                use_deformable=(i % 2 == 1),  # alternate local / deformable
                num_points=num_points,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i],
                offset_scale=offset_scale,
            )
            for i in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample is not None:
            x = self.downsample(x)
        for block in self.blocks:
            x = block(x)
        return x


class DATBackbone(nn.Module):
    """Hierarchical backbone with alternating local-window and deformable attention.

    Mirrors the SwinBackbone API exactly (same __init__ signature + out_channels),
    with an additional `num_points` parameter controlling the r×r deformable grid.
    Returns (P3, P4, P5) feature maps in NCHW format.
    """

    def __init__(
        self,
        in_chans: int = 1,
        embed_dim: int = 64,
        depths: Sequence[int] = (2, 2, 4, 2),
        num_heads: Sequence[int] = (2, 4, 8, 8),
        window_size: int = 8,
        num_points: int = 7,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.05,
        offset_scale: float = 0.5,
    ):
        super().__init__()
        if len(depths) not in {3, 4} or len(num_heads) not in {3, 4}:
            raise ValueError("depths and num_heads must contain 3 or 4 values.")
        depths = tuple(depths[:3])
        num_heads = tuple(num_heads[:3])
        self.embed_dim = int(embed_dim)
        self.out_channels = (embed_dim, embed_dim * 2, embed_dim * 4)

        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dim, patch_size=4)

        total_depth = sum(depths)
        dpr = torch.linspace(0, drop_path_rate, total_depth).tolist()
        offset = 0

        self.stage1 = DATStage(
            embed_dim, depths[0], num_heads[0], window_size, num_points,
            mlp_ratio, drop_rate, attn_drop_rate, dpr[offset:offset + depths[0]],
            downsample=False, offset_scale=offset_scale,
        )
        offset += depths[0]
        self.stage2 = DATStage(
            embed_dim, depths[1], num_heads[1], window_size, num_points,
            mlp_ratio, drop_rate, attn_drop_rate, dpr[offset:offset + depths[1]],
            downsample=True, offset_scale=offset_scale,
        )
        offset += depths[1]
        self.stage3 = DATStage(
            embed_dim * 2, depths[2], num_heads[2], window_size, num_points,
            mlp_ratio, drop_rate, attn_drop_rate, dpr[offset:offset + depths[2]],
            downsample=True, offset_scale=offset_scale,
        )

    @staticmethod
    def _nchw(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.patch_embed(x)
        p3 = self.stage1(x)
        p4 = self.stage2(p3)
        p5 = self.stage3(p4)
        return self._nchw(p3), self._nchw(p4), self._nchw(p5)
