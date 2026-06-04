from collections.abc import Iterable


def stride_hw_to_xy(stride_hw):
    if isinstance(stride_hw, (int, float)):
        stride = float(stride_hw)
        return stride, stride

    items = list(stride_hw) if isinstance(stride_hw, Iterable) else None
    if items is None or len(items) != 2:
        raise ValueError("stride must contain exactly two values (H, W).")

    stride_h = float(items[0])
    stride_w = float(items[1])
    if stride_h <= 0 or stride_w <= 0:
        raise ValueError(f"stride must contain strictly positive values, got {items}.")
    return stride_w, stride_h
