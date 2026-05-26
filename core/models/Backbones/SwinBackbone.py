from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_2tuple(value):
    if isinstance(value, tuple):
        return value
    return (value, value)


def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, c)


def _window_reverse(windows: torch.Tensor, window_size: int, height: int, width: int, batch_size: int) -> torch.Tensor:
    x = windows.view(batch_size, height // window_size, width // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, height, width, -1)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep_prob) * mask.floor()


class PatchEmbed(nn.Module):
    def __init__(self, in_chans: int = 1, embed_dim: int = 32, patch_size: int = 4):
        super().__init__()
        self.patch_size = _to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_h, patch_w = self.patch_size
        pad_h = (patch_h - x.shape[-2] % patch_h) % patch_h
        pad_w = (patch_w - x.shape[-1] % patch_w) % patch_w
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return self.norm(x)


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int = 8, num_heads: int = 4, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads.")
        self.dim = int(dim)
        self.window_size = int(window_size)
        self.num_heads = int(num_heads)
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        relative_size = (2 * window_size - 1) * (2 * window_size - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(relative_size, num_heads))

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1), persistent=False)

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b_windows, n, c = x.shape
        qkv = self.qkv(x).reshape(b_windows, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)]
        bias = bias.reshape(n, n, -1).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(b_windows // num_windows, num_windows, self.num_heads, n, n)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(b_windows, n, c)
        return self.proj_drop(self.proj(x))


class SwinBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
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

    def _attention_mask(self, height: int, width: int, device: torch.device) -> torch.Tensor | None:
        if self.shift_size == 0:
            return None
        img_mask = torch.zeros((1, height, width, 1), device=device)
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        count = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = count
                count += 1
        mask_windows = _window_partition(img_mask, self.window_size).squeeze(-1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x.shape
        shortcut = x
        x = self.norm1(x)

        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        hp, wp = x.shape[1], x.shape[2]

        shift = self.shift_size if min(hp, wp) > self.window_size else 0
        shifted = torch.roll(x, shifts=(-shift, -shift), dims=(1, 2)) if shift > 0 else x
        windows = _window_partition(shifted, self.window_size)
        attn_windows = self.attn(windows, self._attention_mask(hp, wp, x.device) if shift > 0 else None)
        shifted = _window_reverse(attn_windows, self.window_size, hp, wp, b)
        x = torch.roll(shifted, shifts=(shift, shift), dims=(1, 2)) if shift > 0 else shifted

        if pad_h or pad_w:
            x = x[:, :h, :w, :].contiguous()
        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchMerging(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim * 4)
        self.reduction = nn.Linear(dim * 4, dim * 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] % 2 or x.shape[2] % 2:
            x = F.pad(x, (0, 0, 0, x.shape[2] % 2, 0, x.shape[1] % 2))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        return self.reduction(self.norm(x))


class SwinStage(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        drop: float,
        attn_drop: float,
        drop_path: Sequence[float],
        downsample: bool = True,
    ):
        super().__init__()
        self.downsample = PatchMerging(dim) if downsample else None
        block_dim = dim * 2 if downsample else dim
        self.blocks = nn.ModuleList(
            SwinBlock(
                dim=block_dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if index % 2 == 0 else window_size // 2,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[index],
            )
            for index in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample is not None:
            x = self.downsample(x)
        for block in self.blocks:
            x = block(x)
        return x


class SwinBackbone(nn.Module):
    """Compact hierarchical Swin backbone returning P3, P4 and P5 feature maps."""

    def __init__(
        self,
        in_chans: int = 1,
        embed_dim: int = 64,
        depths: Sequence[int] = (2, 2, 4, 2),
        num_heads: Sequence[int] = (2, 4, 8, 8),
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.05,
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

        self.stage1 = SwinStage(embed_dim, depths[0], num_heads[0], window_size, mlp_ratio, drop_rate, attn_drop_rate, dpr[offset:offset + depths[0]], downsample=False)
        offset += depths[0]
        self.stage2 = SwinStage(embed_dim, depths[1], num_heads[1], window_size, mlp_ratio, drop_rate, attn_drop_rate, dpr[offset:offset + depths[1]], downsample=True)
        offset += depths[1]
        self.stage3 = SwinStage(embed_dim * 2, depths[2], num_heads[2], window_size, mlp_ratio, drop_rate, attn_drop_rate, dpr[offset:offset + depths[2]], downsample=True)

    @staticmethod
    def _nchw(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.patch_embed(x)
        p3 = self.stage1(x)
        p4 = self.stage2(p3)
        p5 = self.stage3(p4)
        return self._nchw(p3), self._nchw(p4), self._nchw(p5)
