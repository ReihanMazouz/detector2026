import torch
import torch.nn as nn


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
