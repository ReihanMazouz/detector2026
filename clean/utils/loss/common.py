import torch

from ..stride import stride_hw_to_xy


def stride_tensor_xyxy(stride_tensor):
    return torch.cat([stride_tensor, stride_tensor], dim=1)


def image_size_from_feats(feat, stride_hw, device, dtype):
    feat_h, feat_w = feat.shape[2:]
    stride_x, stride_y = stride_hw_to_xy(stride_hw)
    return torch.tensor([feat_h * stride_y, feat_w * stride_x], device=device, dtype=dtype)
