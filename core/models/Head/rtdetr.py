import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...nn.blocks import DFL
from ...nn.convs import Conv
from ...utils.stride import stride_hw_to_xy
from ...utils.tal import dist2bbox, make_anchors


def inverse_sigmoid(x, eps=1e-6):
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x / (1.0 - x))


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        for layer_idx in range(num_layers):
            in_dim = input_dim if layer_idx == 0 else hidden_dim
            out_dim = output_dim if layer_idx == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.layers) - 1:
                x = torch.relu(x)
        return x


class MultiScaleDeformableAttention(nn.Module):
    # Local PyTorch implementation based on grid_sample. It matches the multi-scale
    # deformable attention idea, but not the optimized CUDA kernel used by official RT-DETR.
    def __init__(self, hidden_dim, num_heads=8, num_levels=3, num_points=4, dropout=0.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_levels = int(num_levels)
        self.num_points = int(num_points)
        self.head_dim = self.hidden_dim // self.num_heads

        self.sampling_offsets = nn.Linear(hidden_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(hidden_dim, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        angles = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid = torch.stack([angles.cos(), angles.sin()], dim=-1)
        grid = grid / grid.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-6)
        grid = grid.view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        for point_idx in range(self.num_points):
            grid[:, :, point_idx, :] *= point_idx + 1
        self.sampling_offsets.bias.data = grid.reshape(-1)
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(self, query, reference_boxes, multi_scale_features):
        batch_size, num_queries, _ = query.shape
        values = []
        for feature in multi_scale_features:
            height, width = feature.shape[2:]
            value = feature.flatten(2).transpose(1, 2)
            value = self.value_proj(value).transpose(1, 2).reshape(batch_size, self.hidden_dim, height, width)
            values.append(value)
        offsets = self.sampling_offsets(query).view(
            batch_size, num_queries, self.num_heads, self.num_levels, self.num_points, 2
        )
        weights = self.attention_weights(query).view(
            batch_size, num_queries, self.num_heads, self.num_levels * self.num_points
        )
        weights = weights.softmax(-1).view(batch_size, num_queries, self.num_heads, self.num_levels, self.num_points)

        reference_xy = reference_boxes[..., :2].view(batch_size, num_queries, 1, 1, 1, 2)
        reference_wh = reference_boxes[..., 2:].view(batch_size, num_queries, 1, 1, 1, 2).clamp(min=1e-4)
        sampling_locations = reference_xy + offsets / float(self.num_points) * reference_wh * 0.5

        output = query.new_zeros(batch_size, num_queries, self.num_heads, self.head_dim)
        for level_idx, value in enumerate(values):
            height, width = value.shape[2:]
            value = value.view(batch_size, self.num_heads, self.head_dim, height, width)
            value = value.reshape(batch_size * self.num_heads, self.head_dim, height, width)

            grid = sampling_locations[:, :, :, level_idx]
            grid = grid.permute(0, 2, 1, 3, 4).reshape(batch_size * self.num_heads, num_queries, self.num_points, 2)
            grid = grid * 2.0 - 1.0

            sampled = F.grid_sample(value, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            sampled = sampled.view(batch_size, self.num_heads, self.head_dim, num_queries, self.num_points)
            sampled = sampled.permute(0, 3, 1, 4, 2)
            output = output + (sampled * weights[:, :, :, level_idx].unsqueeze(-1)).sum(dim=3)

        output = output.reshape(batch_size, num_queries, self.hidden_dim)
        return self.dropout(self.output_proj(output))


class RTDETRDecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, num_levels=3, num_points=4, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = MultiScaleDeformableAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            dropout=dropout,
        )
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, target, query_pos, reference_boxes, multi_scale_features):
        query = key = target + query_pos
        attended, _ = self.self_attn(query, key, target)
        target = self.norm1(target + self.dropout1(attended))

        deformable = self.cross_attn(target + query_pos, reference_boxes, multi_scale_features)
        target = self.norm2(target + self.dropout2(deformable))

        ffn = self.linear2(F.gelu(self.linear1(target)))
        target = self.norm3(target + self.dropout3(ffn))
        return target


class DenseTransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def forward(self, target, query_pos, reference_boxes, multi_scale_features):
        memory = torch.cat([feature.flatten(2).transpose(1, 2) for feature in multi_scale_features], dim=1)
        _ = reference_boxes
        return self.layer(target + query_pos, memory)


class RTDETRHead(nn.Module):
    """RT-DETR-like one-to-one head for YOLO pyramid features."""

    def __init__(
        self,
        in_channels,
        strides,
        num_classes=80,
        reg_max=16,
        hidden_dim=256,
        num_queries=300,
        num_decoder_layers=3,
        num_heads=8,
        dim_feedforward=1024,
        dropout=0.0,
        num_decoder_points=4,
        use_deformable_attention=True,
        learnt_init_query=False,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")

        self.nc = int(num_classes)
        self.reg_max = int(reg_max)
        self.nl = len(in_channels)
        self.strides = strides
        self.hidden_dim = int(hidden_dim)
        self.num_queries = int(num_queries)
        self.num_decoder_layers = int(num_decoder_layers)
        self.num_decoder_points = int(num_decoder_points)
        self.use_deformable_attention = bool(use_deformable_attention)
        self.learnt_init_query = bool(learnt_init_query)

        self.input_proj = nn.ModuleList(
            nn.Sequential(nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False), nn.BatchNorm2d(hidden_dim))
            for channels in in_channels
        )

        c2 = max((16, in_channels[0] // 4, self.reg_max * 4))
        self.cv_dist = nn.ModuleList(
            nn.Sequential(Conv(channels, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1))
            for channels in in_channels
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        self.enc_output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.enc_score_head = nn.Linear(hidden_dim, self.nc + 1)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        if self.learnt_init_query:
            self.tgt_embed = nn.Embedding(self.num_queries, hidden_dim)

        decoder_layer_cls = RTDETRDecoderLayer if self.use_deformable_attention else DenseTransformerDecoderLayer
        self.decoder_layers = nn.ModuleList(
            (
                decoder_layer_cls(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    num_levels=self.nl,
                    num_points=self.num_decoder_points,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                if self.use_deformable_attention
                else decoder_layer_cls(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
            )
            for _ in range(self.num_decoder_layers)
        )
        self.dec_score_head = nn.ModuleList(nn.Linear(hidden_dim, self.nc + 1) for _ in range(self.num_decoder_layers))
        self.dec_bbox_head = nn.ModuleList(MLP(hidden_dim, hidden_dim, 4, num_layers=3) for _ in range(self.num_decoder_layers))

        self._reset_parameters()

    def _reset_parameters(self):
        prior_prob = 0.01
        bias_value = -math.log((1.0 - prior_prob) / prior_prob)
        nn.init.constant_(self.enc_score_head.bias, bias_value)
        for cls_head, box_head in zip(self.dec_score_head, self.dec_bbox_head):
            nn.init.constant_(cls_head.bias, bias_value)
            nn.init.constant_(box_head.layers[-1].weight, 0.0)
            nn.init.constant_(box_head.layers[-1].bias, 0.0)

    def bias_init(self, image_size=640):
        for branch, stride in zip(self.cv_dist, self.strides):
            if hasattr(branch[-1], "bias"):
                branch[-1].bias.data[:] = 1.0
        _ = image_size

    def _image_size_from_features(self, features, dtype, device):
        feat_h, feat_w = features[0].shape[2:]
        stride_x, stride_y = stride_hw_to_xy(self.strides[0])
        return torch.tensor([feat_h * stride_y, feat_w * stride_x], device=device, dtype=dtype)

    def _project_features(self, features):
        return [projection(feature) for projection, feature in zip(self.input_proj, features)]

    def _get_encoder_input(self, projected_features):
        tokens = []
        shapes = []
        for feature in projected_features:
            height, width = feature.shape[2:]
            tokens.append(feature.flatten(2).transpose(1, 2))
            shapes.append((height, width))
        return torch.cat(tokens, dim=1), shapes

    def _generate_reference_boxes(self, features, image_size):
        dist_outputs = [branch(feature) for branch, feature in zip(self.cv_dist, features)]
        pred_dist = torch.cat([output.flatten(2) for output in dist_outputs], dim=2)
        batch_size, _, num_anchors = pred_dist.shape

        anchor_points, stride_tensor = make_anchors(list(features), self.strides, 0.5)
        anchor_points = anchor_points.to(device=pred_dist.device, dtype=pred_dist.dtype)
        stride_tensor = stride_tensor.to(device=pred_dist.device, dtype=pred_dist.dtype)

        decoded = dist2bbox(self.dfl(pred_dist).transpose(1, 2), anchor_points, xywh=True)
        stride_tensor_boxes = torch.cat([stride_tensor, stride_tensor], dim=1).unsqueeze(0)
        boxes_abs = decoded * stride_tensor_boxes
        scale = torch.tensor(
            [image_size[1], image_size[0], image_size[1], image_size[0]],
            device=pred_dist.device,
            dtype=pred_dist.dtype,
        ).view(1, 1, 4)
        boxes = (boxes_abs / scale).clamp(0.0, 1.0)
        return boxes.reshape(batch_size, num_anchors, 4)

    def _get_decoder_input(self, memory, reference_boxes):
        batch_size, num_tokens, _ = memory.shape
        features = self.enc_output(memory)
        enc_logits = self.enc_score_head(features)
        object_scores = enc_logits[..., : self.nc].max(dim=-1).values
        num_queries = min(self.num_queries, num_tokens)

        topk_indices = torch.topk(object_scores, num_queries, dim=1).indices
        gather_features = topk_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        gather_boxes = topk_indices.unsqueeze(-1).expand(-1, -1, 4)
        top_features = features.gather(1, gather_features)
        top_boxes = reference_boxes.gather(1, gather_boxes).detach()
        enc_top_logits = enc_logits.gather(1, topk_indices.unsqueeze(-1).expand(-1, -1, self.nc + 1))

        if self.learnt_init_query:
            query = self.tgt_embed.weight[:num_queries].unsqueeze(0).expand(batch_size, -1, -1)
        else:
            query = top_features.detach() if self.training else top_features

        return query, top_boxes, enc_top_logits

    def forward(self, *features, image_size=None):
        if len(features) == 1 and isinstance(features[0], (list, tuple)):
            features = tuple(features[0])
        if len(features) != self.nl:
            raise ValueError(f"RTDETRHead expects {self.nl} feature maps, got {len(features)}.")

        dtype = features[0].dtype
        device = features[0].device
        if image_size is None:
            image_size = self._image_size_from_features(features, dtype=dtype, device=device)
        else:
            image_size = torch.as_tensor(image_size, device=device, dtype=dtype)

        projected_features = self._project_features(features)
        memory, _ = self._get_encoder_input(projected_features)
        reference_boxes = self._generate_reference_boxes(features, image_size)
        query, reference_boxes, enc_logits = self._get_decoder_input(memory, reference_boxes)
        query_pos = self.query_pos_head(reference_boxes)

        outputs = []
        target = query
        reference_logits = inverse_sigmoid(reference_boxes)
        for layer, cls_head, box_head in zip(self.decoder_layers, self.dec_score_head, self.dec_bbox_head):
            target = layer(target, query_pos, reference_boxes, projected_features)
            pred_logits = cls_head(target)
            pred_boxes = (reference_logits + box_head(target)).sigmoid()
            outputs.append({"pred_logits": pred_logits, "pred_boxes": pred_boxes})

        final_output = outputs[-1]
        final_output["aux_outputs"] = [{"pred_logits": enc_logits, "pred_boxes": reference_boxes}, *outputs[:-1]]
        return final_output
