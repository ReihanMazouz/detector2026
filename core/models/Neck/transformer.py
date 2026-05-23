import torch
import torch.nn as nn
import torch.nn.functional as F

from ..Head.rtdetr import MSDeformAttn


class TransformerPyramidNeck(nn.Module):
    """Transformer neck that mixes P3, P4 and P5 tokens and returns feature maps at the same scales."""

    def __init__(
        self,
        in_channels,
        d_model=128,
        num_heads=4,
        num_layers=1,
        ffn_ratio=2.0,
        dropout=0.0,
        residual_scale=0.0,
    ):
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError("TransformerPyramidNeck expects in_channels for P3, P4 and P5.")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.proj_in = nn.ModuleList(nn.Conv2d(channels, d_model, kernel_size=1) for channels in in_channels)
        self.proj_out = nn.ModuleList(nn.Conv2d(d_model, channels, kernel_size=1) for channels in in_channels)
        self.pos_proj = nn.Linear(2, d_model)
        self.level_embed = nn.Parameter(torch.zeros(3, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=max(d_model, int(d_model * ffn_ratio)),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def _position_tokens(self, height, width, level_index, device, dtype):
        y = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack((xx, yy), dim=-1).reshape(1, height * width, 2)
        pos = self.pos_proj(coords)
        return pos + self.level_embed[level_index].to(device=device, dtype=dtype).view(1, 1, -1)

    def forward(self, p3, p4, p5):
        features = (p3, p4, p5)
        tokens = []
        shapes = []

        for level_index, feat in enumerate(features):
            projected = self.proj_in[level_index](feat)
            batch_size, channels, height, width = projected.shape
            token = projected.flatten(2).transpose(1, 2)
            token = token + self._position_tokens(height, width, level_index, token.device, token.dtype)
            tokens.append(token)
            shapes.append((batch_size, channels, height, width))

        encoded = self.encoder(torch.cat(tokens, dim=1))

        outputs = []
        start = 0
        for level_index, feat in enumerate(features):
            batch_size, channels, height, width = shapes[level_index]
            end = start + height * width
            encoded_level = encoded[:, start:end].transpose(1, 2).reshape(batch_size, channels, height, width)
            delta = self.proj_out[level_index](encoded_level)
            outputs.append(feat + self.residual_scale * delta)
            start = end

        return tuple(outputs)


class DeformablePyramidNeckLayer(nn.Module):
    """Sparse multi-scale attention layer for P3/P4/P5 tokens."""

    def __init__(self, d_model=128, num_heads=4, num_points=4, ffn_ratio=2.0, dropout=0.0):
        super().__init__()
        self.attn = MSDeformAttn(
            hidden_dim=d_model,
            num_levels=3,
            num_heads=num_heads,
            num_points=num_points,
        )
        ffn_dim = max(d_model, int(d_model * ffn_ratio))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, reference_points, value_shapes):
        attn_out = self.attn(tokens, reference_points, tokens, value_shapes)
        tokens = self.norm1(tokens + self.dropout(attn_out))
        ffn_out = self.linear2(self.dropout(F.gelu(self.linear1(tokens))))
        return self.norm2(tokens + self.dropout(ffn_out))


class DeformablePyramidNeck(nn.Module):
    """Deformable neck that mixes P3, P4 and P5 with sparse multi-scale attention."""

    def __init__(
        self,
        in_channels,
        d_model=128,
        num_heads=4,
        num_layers=1,
        num_points=4,
        ffn_ratio=2.0,
        dropout=0.0,
        residual_scale=0.0,
    ):
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError("DeformablePyramidNeck expects in_channels for P3, P4 and P5.")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.proj_in = nn.ModuleList(nn.Conv2d(channels, d_model, kernel_size=1) for channels in in_channels)
        self.proj_out = nn.ModuleList(nn.Conv2d(d_model, channels, kernel_size=1) for channels in in_channels)
        self.pos_proj = nn.Linear(2, d_model)
        self.level_embed = nn.Parameter(torch.zeros(3, d_model))
        self.layers = nn.ModuleList(
            DeformablePyramidNeckLayer(
                d_model=d_model,
                num_heads=num_heads,
                num_points=num_points,
                ffn_ratio=ffn_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def _coords(self, height, width, device, dtype):
        y = torch.linspace(0.5 / height, 1.0 - 0.5 / height, height, device=device, dtype=dtype)
        x = torch.linspace(0.5 / width, 1.0 - 0.5 / width, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), dim=-1).reshape(1, height * width, 2)

    def _position_tokens(self, coords, level_index):
        pos = self.pos_proj(coords)
        return pos + self.level_embed[level_index].to(device=coords.device, dtype=coords.dtype).view(1, 1, -1)

    def forward(self, p3, p4, p5):
        features = (p3, p4, p5)
        tokens = []
        reference_points = []
        shapes = []
        value_shapes = []

        for level_index, feat in enumerate(features):
            projected = self.proj_in[level_index](feat)
            batch_size, channels, height, width = projected.shape
            coords = self._coords(height, width, projected.device, projected.dtype)
            token = projected.flatten(2).transpose(1, 2)
            token = token + self._position_tokens(coords, level_index)
            tokens.append(token)
            reference_points.append(coords.expand(batch_size, -1, -1))
            shapes.append((batch_size, channels, height, width))
            value_shapes.append((height, width))

        encoded = torch.cat(tokens, dim=1)
        reference_points = torch.cat(reference_points, dim=1).unsqueeze(2).repeat(1, 1, 3, 1)
        for layer in self.layers:
            encoded = layer(encoded, reference_points, value_shapes)

        outputs = []
        start = 0
        for level_index, feat in enumerate(features):
            batch_size, channels, height, width = shapes[level_index]
            end = start + height * width
            encoded_level = encoded[:, start:end].transpose(1, 2).reshape(batch_size, channels, height, width)
            delta = self.proj_out[level_index](encoded_level)
            outputs.append(feat + self.residual_scale * delta)
            start = end

        return tuple(outputs)
