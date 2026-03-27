from __future__ import annotations

import torch


def ensure_single_spectrum_2d(tensor: torch.Tensor) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor, got {type(tensor)}")

    if tensor.ndim == 2:
        return tensor

    if tensor.ndim == 3 and tensor.shape[0] == 1:
        return tensor[0]

    raise ValueError(
        "Expected a 2D spectrum tensor or a 3D tensor with a singleton channel dimension, "
        f"got shape {tuple(tensor.shape)}"
    )


def ensure_chw_float(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(0).to(torch.float32)
    if tensor.ndim == 3:
        return tensor.to(torch.float32)
    raise ValueError(f"Expected a 2D or 3D tensor, got shape {tuple(tensor.shape)}")


def minmax_scale(values: torch.Tensor) -> torch.Tensor:
    values = values.to(torch.float32)
    min_value = values.min()
    max_value = values.max()
    if max_value <= min_value:
        return torch.zeros_like(values, dtype=torch.float32)
    return (values - min_value) / (max_value - min_value)
