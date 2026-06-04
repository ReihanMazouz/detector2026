import math
from typing import Iterable, List, Tuple


HW = Tuple[int, int]
StrideHW = Tuple[float, float]
ConvStrideHW = Tuple[int, int]


def _as_hw(value: Iterable[int | float], name: str) -> HW:
    items = list(value)
    if len(items) != 2:
        raise ValueError(f"{name} must contain exactly two values (H, W).")
    height = int(items[0])
    width = int(items[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} must contain strictly positive values, got {items}.")
    return height, width


def _as_hw_float(value: Iterable[int | float], name: str) -> StrideHW:
    items = list(value)
    if len(items) != 2:
        raise ValueError(f"{name} must contain exactly two values (H, W).")
    height = float(items[0])
    width = float(items[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} must contain strictly positive values, got {items}.")
    return height, width


def _closest_stage_factor(start: int, target: int) -> int:
    ratio = max(float(start) / float(target), 1e-9)
    candidates = (1, 2, 4, 8)
    return min(candidates, key=lambda factor: abs(math.log2(ratio) - math.log2(factor)))


def _factor_to_stage_axis_strides(factor: int) -> List[int]:
    if factor == 1:
        return [1, 1, 1]
    if factor == 2:
        return [2, 1, 1]
    if factor == 4:
        return [2, 2, 1]
    if factor == 8:
        return [2, 2, 2]
    raise ValueError(f"Unsupported stage factor '{factor}'. Expected one of 1, 2, 4, 8.")


def build_anisotropic_standard_plan(input_hw: Iterable[int | float], p3_size: Iterable[int | float]) -> dict:
    input_h, input_w = _as_hw(input_hw, "input_hw")
    p3_h, p3_w = _as_hw(p3_size, "p3_size")

    if p3_h % 4 != 0 or p3_w % 4 != 0:
        raise ValueError(f"p3_size must be divisible by 4 on both axes, got {(p3_h, p3_w)}.")

    stage_factor_h = _closest_stage_factor(input_h, p3_h)
    stage_factor_w = _closest_stage_factor(input_w, p3_w)

    pre_resize_hw = (p3_h * stage_factor_h, p3_w * stage_factor_w)
    stride_h = _factor_to_stage_axis_strides(stage_factor_h)
    stride_w = _factor_to_stage_axis_strides(stage_factor_w)
    stage_strides: List[ConvStrideHW] = list(zip(stride_h, stride_w))

    p3_stride = (input_h / float(p3_h), input_w / float(p3_w))
    detect_strides: List[StrideHW] = [
        p3_stride,
        (p3_stride[0] * 2.0, p3_stride[1] * 2.0),
        (p3_stride[0] * 4.0, p3_stride[1] * 4.0),
    ]

    return {
        "input_hw": (input_h, input_w),
        "p3_size": (p3_h, p3_w),
        "pre_resize_hw": pre_resize_hw,
        "stage_strides": stage_strides,
        "detect_strides": detect_strides,
    }


def stride_hw_to_xy(stride_hw: int | float | Iterable[int | float]) -> Tuple[float, float]:
    if isinstance(stride_hw, (int, float)):
        stride = float(stride_hw)
        return stride, stride
    stride_h, stride_w = _as_hw_float(stride_hw, "stride")
    return float(stride_w), float(stride_h)


def stride_hw_to_xyxy(stride_hw: int | float | Iterable[int | float]) -> Tuple[float, float, float, float]:
    stride_x, stride_y = stride_hw_to_xy(stride_hw)
    return stride_x, stride_y, stride_x, stride_y
