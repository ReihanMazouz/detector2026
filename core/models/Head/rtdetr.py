import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from ...nn.blocks import DFL
from ...nn.convs import Conv
from ...utils.stride import stride_hw_to_xy
from ...utils.tal import dist2bbox, make_anchors


def inverse_sigmoid(x, eps=1e-6):
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x / (1.0 - x))


def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape
    split_sizes = [int(height * width) for height, width in value_spatial_shapes]
    value_list = value.split(split_sizes, dim=1)
    sampling_grids = 2.0 * sampling_locations - 1.0
    sampling_value_list = []
    for level, (height, width) in enumerate(value_spatial_shapes):
        height = int(height)
        width = int(width)
        value_l = value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, height, width)
        sampling_grid_l = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampling_value_l = F.grid_sample(
            value_l,
            sampling_grid_l,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l)

    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()


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


class MSDeformAttn(nn.Module):
    """Ultralytics/RT-DETR multi-scale deformable attention."""

    def __init__(self, hidden_dim=256, num_levels=3, num_heads=8, num_points=4):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_levels = int(num_levels)
        self.num_points = int(num_points)

        self.sampling_offsets = nn.Linear(hidden_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(hidden_dim, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        angles = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([angles.cos(), angles.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for point_idx in range(self.num_points):
            grid_init[:, :, point_idx, :] *= point_idx + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(self, query, reference_boxes, value, value_shapes, value_mask=None):
        batch_size, num_queries = query.shape[:2]
        num_values = value.shape[1]
        value_shapes_tensor = torch.as_tensor(value_shapes, dtype=torch.long, device=query.device)
        if int((value_shapes_tensor[:, 0] * value_shapes_tensor[:, 1]).sum()) != num_values:
            raise ValueError("value_shapes do not match flattened feature length.")

        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], 0.0)
        value = value.view(batch_size, num_values, self.num_heads, self.hidden_dim // self.num_heads)
        sampling_offsets = self.sampling_offsets(query).view(
            batch_size, num_queries, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            batch_size, num_queries, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            batch_size, num_queries, self.num_heads, self.num_levels, self.num_points
        )

        if reference_boxes.shape[-1] == 2:
            offset_normalizer = value_shapes_tensor.to(dtype=query.dtype).flip(-1)
            sampling_locations = reference_boxes[:, :, None, :, None, :] + (
                sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_boxes.shape[-1] == 4:
            sampling_locations = reference_boxes[:, :, None, :, None, :2] + (
                sampling_offsets / self.num_points * reference_boxes[:, :, None, :, None, 2:].clamp(min=1e-4) * 0.5
            )
        else:
            raise ValueError(f"Last dim of reference_boxes must be 2 or 4, got {reference_boxes.shape[-1]}.")

        output = multi_scale_deformable_attn_pytorch(
            value,
            value_shapes_tensor,
            sampling_locations,
            attention_weights,
        )
        return self.output_proj(output)


class RTDETRDecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, num_levels=3, num_points=4, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.cross_attn = MSDeformAttn(hidden_dim=hidden_dim, num_levels=num_levels, num_heads=num_heads, num_points=num_points)
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, target):
        target2 = self.linear2(self.dropout3(F.relu(self.linear1(target))))
        target = target + self.dropout4(target2)
        return self.norm3(target)

    def forward(self, target, reference_boxes, memory, shapes, padding_mask=None, attn_mask=None, query_pos=None):
        query = key = self.with_pos_embed(target, query_pos)
        attended = self.self_attn(
            query.transpose(0, 1),
            key.transpose(0, 1),
            target.transpose(0, 1),
            attn_mask=attn_mask,
        )[0].transpose(0, 1)
        target = self.norm1(target + self.dropout1(attended))

        deformable = self.cross_attn(
            self.with_pos_embed(target, query_pos),
            reference_boxes.unsqueeze(2),
            memory,
            shapes,
            padding_mask,
        )
        target = self.norm2(target + self.dropout2(deformable))
        return self.forward_ffn(target)


class DeformableTransformerDecoder(nn.Module):
    """Ultralytics/RT-DETR deformable decoder with iterative box refinement."""

    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        self.eval_idx = int(eval_idx) if eval_idx >= 0 else int(num_layers) + int(eval_idx)

    def forward(
        self,
        target,
        reference_boxes,
        memory,
        shapes,
        bbox_head,
        score_head,
        pos_mlp,
        attn_mask=None,
        padding_mask=None,
    ):
        output = target
        dec_bboxes = []
        dec_logits = []
        last_refined_bbox = None
        for layer_index, layer in enumerate(self.layers):
            output = layer(
                output,
                reference_boxes,
                memory,
                shapes,
                padding_mask=padding_mask,
                attn_mask=attn_mask,
                query_pos=pos_mlp(reference_boxes),
            )
            bbox_delta = bbox_head[layer_index](output)
            refined_bbox = torch.sigmoid(bbox_delta + inverse_sigmoid(reference_boxes))

            if self.training:
                dec_logits.append(score_head[layer_index](output))
                if layer_index == 0:
                    dec_bboxes.append(refined_bbox)
                else:
                    dec_bboxes.append(torch.sigmoid(bbox_delta + inverse_sigmoid(last_refined_bbox)))
            elif layer_index == self.eval_idx:
                dec_logits.append(score_head[layer_index](output))
                dec_bboxes.append(refined_bbox)
                break

            last_refined_bbox = refined_bbox
            reference_boxes = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(dec_bboxes), torch.stack(dec_logits)


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
        self.enc_score_head = nn.Linear(hidden_dim, self.nc)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        if self.learnt_init_query:
            self.tgt_embed = nn.Embedding(self.num_queries, hidden_dim)

        if self.use_deformable_attention:
            decoder_layer = RTDETRDecoderLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_levels=self.nl,
                num_points=self.num_decoder_points,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            self.decoder = DeformableTransformerDecoder(hidden_dim, decoder_layer, self.num_decoder_layers)
        else:
            self.decoder_layers = nn.ModuleList(
                DenseTransformerDecoderLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(self.num_decoder_layers)
            )
        self.dec_score_head = nn.ModuleList(nn.Linear(hidden_dim, self.nc) for _ in range(self.num_decoder_layers))
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
        enc_top_logits = enc_logits.gather(1, topk_indices.unsqueeze(-1).expand(-1, -1, self.nc))

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
        memory, shapes = self._get_encoder_input(projected_features)
        reference_boxes = self._generate_reference_boxes(features, image_size)
        query, reference_boxes, enc_logits = self._get_decoder_input(memory, reference_boxes)

        if self.use_deformable_attention:
            dec_boxes, dec_logits = self.decoder(
                query,
                reference_boxes,
                memory,
                shapes,
                self.dec_bbox_head,
                self.dec_score_head,
                self.query_pos_head,
            )
            outputs = [
                {"pred_logits": layer_logits, "pred_boxes": layer_boxes}
                for layer_boxes, layer_logits in zip(dec_boxes.unbind(0), dec_logits.unbind(0))
            ]
        else:
            outputs = []
            target = query
            reference_logits = inverse_sigmoid(reference_boxes)
            query_pos = self.query_pos_head(reference_boxes)
            for layer, cls_head, box_head in zip(self.decoder_layers, self.dec_score_head, self.dec_bbox_head):
                target = layer(target, query_pos, reference_boxes, projected_features)
                pred_logits = cls_head(target)
                pred_boxes = (reference_logits + box_head(target)).sigmoid()
                outputs.append({"pred_logits": pred_logits, "pred_boxes": pred_boxes})

        final_output = outputs[-1]
        final_output["aux_outputs"] = [{"pred_logits": enc_logits, "pred_boxes": reference_boxes}, *outputs[:-1]]
        return final_output
