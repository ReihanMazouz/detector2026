import torch
import torch.nn as nn
import torch.nn.functional as F

from ...nn.convs import Conv
from ...nn.blocks import C3k2


class RTDETRHybridEncoderNeck(nn.Module):
    """RT-DETR-style neck: project P3/P4/P5, attend only on P5, then fuse with FPN/PAN."""

    def __init__(
        self,
        in_channels,
        hidden_dim=256,
        num_heads=8,
        num_encoder_layers=1,
        ffn_ratio=4.0,
        dropout=0.0,
        depth_mult=1.0,
    ):
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError("RTDETRHybridEncoderNeck expects in_channels for P3, P4 and P5.")

        self.out_channels = [hidden_dim, hidden_dim, hidden_dim]
        self.input_proj = nn.ModuleList(
            nn.Sequential(nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False), nn.BatchNorm2d(hidden_dim))
            for channels in in_channels
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=max(hidden_dim, int(hidden_dim * ffn_ratio)),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.p5_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.p5_pos_proj = nn.Linear(2, hidden_dim)

        num_blocks = max(1, round(3 * depth_mult))
        self.lateral_p5 = Conv(hidden_dim, hidden_dim, k=1, s=1)
        self.fpn_p4 = C3k2(hidden_dim * 2, hidden_dim, n=num_blocks, shortcut=False)
        self.lateral_p4 = Conv(hidden_dim, hidden_dim, k=1, s=1)
        self.fpn_p3 = C3k2(hidden_dim * 2, hidden_dim, n=num_blocks, shortcut=False)

        self.down_p3 = Conv(hidden_dim, hidden_dim, k=3, s=2)
        self.pan_p4 = C3k2(hidden_dim * 2, hidden_dim, n=num_blocks, shortcut=False)
        self.down_p4 = Conv(hidden_dim, hidden_dim, k=3, s=2)
        self.pan_p5 = C3k2(hidden_dim * 2, hidden_dim, n=num_blocks, shortcut=True)

    def _position_tokens(self, height, width, device, dtype):
        y = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack((xx, yy), dim=-1).reshape(1, height * width, 2)
        return self.p5_pos_proj(coords)

    def _encode_p5(self, p5):
        batch_size, channels, height, width = p5.shape
        tokens = p5.flatten(2).transpose(1, 2)
        tokens = tokens + self._position_tokens(height, width, tokens.device, tokens.dtype)
        encoded = self.p5_encoder(tokens)
        return encoded.transpose(1, 2).reshape(batch_size, channels, height, width).contiguous()

    def forward(self, p3, p4, p5):
        p3, p4, p5 = [projection(feature) for projection, feature in zip(self.input_proj, (p3, p4, p5))]
        p5 = self._encode_p5(p5)

        p5_lateral = self.lateral_p5(p5)
        p4_inner = self.fpn_p4(torch.cat([F.interpolate(p5_lateral, scale_factor=2.0, mode="nearest"), p4], dim=1))
        p4_lateral = self.lateral_p4(p4_inner)
        p3_out = self.fpn_p3(torch.cat([F.interpolate(p4_lateral, scale_factor=2.0, mode="nearest"), p3], dim=1))

        p4_out = self.pan_p4(torch.cat([self.down_p3(p3_out), p4_inner], dim=1))
        p5_out = self.pan_p5(torch.cat([self.down_p4(p4_out), p5_lateral], dim=1))
        return p3_out, p4_out, p5_out
