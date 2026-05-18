from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nn.blocks import C3k2, SPPF
from ..nn.convs import Conv
from ..utils.detr_loss import DETRLoss
from ..utils.divers import xywh2xyxy
from .base import BaseModel


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        layers = []
        for layer_index in range(num_layers):
            in_dim = input_dim if layer_index == 0 else hidden_dim
            out_dim = output_dim if layer_index == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer_index, layer in enumerate(self.layers):
            x = layer(x)
            if layer_index < len(self.layers) - 1:
                x = F.relu(x, inplace=True)
        return x


class DETRDecoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, nheads: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, nheads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, nheads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    @staticmethod
    def _with_pos_embed(tensor: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        *,
        query_pos: torch.Tensor | None = None,
        memory_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = k = self._with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, need_weights=False)[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        q = self._with_pos_embed(tgt, query_pos)
        k = self._with_pos_embed(memory, memory_pos)
        tgt2 = self.cross_attn(q, k, value=memory, need_weights=False)[0]
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        return self.norm3(tgt + tgt2)


class DETRDecoder(nn.Module):
    def __init__(self, decoder_layer: DETRDecoderLayer, num_layers: int, norm: nn.Module | None = None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.norm = norm

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        *,
        query_pos: torch.Tensor | None = None,
        memory_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(output, memory, query_pos=query_pos, memory_pos=memory_pos)
            intermediate.append(self.norm(output) if self.norm is not None else output)
        return torch.stack(intermediate, dim=0)


class DETR(BaseModel):
    def __init__(
        self,
        output_dir: str,
        num_classes: int = 80,
        device: str = "cuda:0",
        input_channels: int = 1,
        width_mult: float = 0.50,
        hidden_dim: int = 256,
        num_queries: int = 100,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 3,
        nheads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        input_hw: tuple[int, int] | None = None,
        aux_loss: bool = True,
        aux_loss_weight: float = 1.0,
    ):
        super().__init__(device=device, output_dir=output_dir)
        if hidden_dim % 4 != 0:
            raise ValueError("hidden_dim must be divisible by 4.")
        if hidden_dim % nheads != 0:
            raise ValueError("hidden_dim must be divisible by nheads.")
        if num_classes < 1:
            raise ValueError("num_classes must be at least 1.")
        if num_queries < 1:
            raise ValueError("num_queries must be at least 1.")
        if num_encoder_layers < 1:
            raise ValueError("num_encoder_layers must be at least 1.")
        if num_decoder_layers < 1:
            raise ValueError("num_decoder_layers must be at least 1.")
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.num_queries = int(num_queries)
        self.input_hw = tuple(input_hw) if input_hw is not None else None
        self.aux_loss = bool(aux_loss)

        c1 = max(16, int(64 * width_mult))
        c2 = max(32, int(128 * width_mult))
        c3 = max(64, int(256 * width_mult))
        c4 = max(128, int(512 * width_mult))
        self.backbone = nn.Sequential(
            Conv(input_channels, c1, k=3, s=2),
            Conv(c1, c2, k=3, s=2),
            C3k2(c2, c2, shortcut=False),
            Conv(c2, c3, k=3, s=2),
            C3k2(c3, c3, shortcut=False),
            Conv(c3, c4, k=3, s=2),
            SPPF(c4, c4),
        )
        self.input_proj = nn.Conv2d(c4, hidden_dim, kernel_size=1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = DETRDecoderLayer(hidden_dim, nheads, dim_feedforward, dropout)
        self.decoder = DETRDecoder(decoder_layer, num_layers=num_decoder_layers, norm=nn.LayerNorm(hidden_dim))
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.class_embed = nn.Linear(hidden_dim, self.num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.criterion = DETRLoss(
            num_classes=self.num_classes,
            aux_loss=self.aux_loss,
            aux_loss_weight=aux_loss_weight,
        )
        self.to(self.device)

    def _positional_encoding_2d(self, height: int, width: int, *, device, dtype) -> torch.Tensor:
        y_embed, x_embed = torch.meshgrid(
            torch.arange(1, height + 1, device=device, dtype=dtype),
            torch.arange(1, width + 1, device=device, dtype=dtype),
            indexing="ij",
        )
        eps = torch.finfo(dtype).eps
        scale = 2.0 * torch.pi
        y_embed = y_embed / (y_embed[-1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, -1:] + eps) * scale
        num_pos_feats = self.hidden_dim // 2
        dim_t = torch.arange(num_pos_feats, device=device, dtype=dtype)
        dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / max(num_pos_feats, 1))

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        pos = torch.cat((pos_y, pos_x), dim=-1)
        return pos.reshape(1, height * width, self.hidden_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        features = self.input_proj(self.backbone(x))
        batch_size, _, height, width = features.shape
        src = features.flatten(2).transpose(1, 2)
        memory_pos = self._positional_encoding_2d(height, width, device=src.device, dtype=src.dtype)
        memory = self.encoder(src + memory_pos)
        query_pos = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        tgt = torch.zeros_like(query_pos)
        hs = self.decoder(tgt, memory, query_pos=query_pos, memory_pos=memory_pos)

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
        }
        if self.aux_loss and hs.shape[0] > 1:
            out["aux_outputs"] = [
                {"pred_logits": logits, "pred_boxes": boxes}
                for logits, boxes in zip(outputs_class[:-1], outputs_coord[:-1])
            ]
        return out

    def postprocess(
        self,
        outputs: dict[str, torch.Tensor],
        *,
        score_threshold: float = 0.0,
        top_k: int | None = None,
        absolute_boxes: bool = False,
    ) -> list[dict[str, torch.Tensor]]:
        logits = outputs["pred_logits"]
        boxes = outputs["pred_boxes"]
        probs = logits.softmax(-1)
        scores, labels = probs[..., :-1].max(dim=-1)

        processed = []
        for batch_scores, batch_labels, batch_boxes in zip(scores, labels, boxes):
            keep = batch_scores >= float(score_threshold)
            selected_scores = batch_scores[keep]
            selected_labels = batch_labels[keep]
            selected_boxes = batch_boxes[keep]

            if top_k is not None and selected_scores.numel() > int(top_k):
                selected_scores, indices = selected_scores.topk(int(top_k))
                selected_labels = selected_labels[indices]
                selected_boxes = selected_boxes[indices]

            output_boxes = xywh2xyxy(selected_boxes).clamp(0.0, 1.0)
            if absolute_boxes:
                if self.input_hw is None:
                    raise ValueError("absolute_boxes=True requires input_hw to be set.")
                image_h, image_w = self.input_hw
                scale = output_boxes.new_tensor([image_w, image_h, image_w, image_h])
                output_boxes = output_boxes * scale

            processed.append(
                {
                    "scores": selected_scores,
                    "labels": selected_labels,
                    "boxes": output_boxes,
                }
            )
        return processed
