from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def _metric_stub(*args, **kwargs):
    raise RuntimeError("Optional training/evaluation dependency unavailable in this environment")


try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is required to run this script. Activate the detector2026 environment first."
    ) from exc

try:
    from thop import profile
except ModuleNotFoundError as exc:
    raise SystemExit(
        "thop is required to compute model cost. Install dependencies from detector2026 first."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))


if "torchinfo" not in sys.modules:
    torchinfo_stub = types.ModuleType("torchinfo")
    torchinfo_stub.summary = lambda *args, **kwargs: "torchinfo summary unavailable in this environment"
    sys.modules["torchinfo"] = torchinfo_stub

if "sklearn" not in sys.modules:
    sklearn_stub = types.ModuleType("sklearn")
    sklearn_metrics_stub = types.ModuleType("sklearn.metrics")
    sklearn_metrics_stub.precision_score = _metric_stub
    sklearn_metrics_stub.recall_score = _metric_stub
    sklearn_metrics_stub.average_precision_score = _metric_stub

    class _ConfusionMatrixDisplayStub:
        def __init__(self, *args, **kwargs):
            pass

        def plot(self, *args, **kwargs):
            return self

    sklearn_metrics_stub.ConfusionMatrixDisplay = _ConfusionMatrixDisplayStub
    sklearn_stub.metrics = sklearn_metrics_stub
    sys.modules["sklearn"] = sklearn_stub
    sys.modules["sklearn.metrics"] = sklearn_metrics_stub

if "detector2026.core.utils.dataset" not in sys.modules:
    dataset_stub = types.ModuleType("detector2026.core.utils.dataset")

    class _DatasetStub:
        pass

    dataset_stub.YOLODatasetFusedMultiRes = _DatasetStub
    dataset_stub.YOLODatasetSpecificRes = _DatasetStub
    dataset_stub.YOLODatasetSingleRes = _DatasetStub
    sys.modules["detector2026.core.utils.dataset"] = dataset_stub

if "detector2026.core.utils.display_outputs" not in sys.modules:
    display_stub = types.ModuleType("detector2026.core.utils.display_outputs")
    display_stub.plot_batch_with_boxes = lambda *args, **kwargs: None
    display_stub.plot_batch_matched_boxes = lambda *args, **kwargs: None
    display_stub.plot_predicted_boxes_batch = lambda *args, **kwargs: None
    sys.modules["detector2026.core.utils.display_outputs"] = display_stub

if "detector2026.core.utils.training_functions" not in sys.modules:
    training_functions_stub = types.ModuleType("detector2026.core.utils.training_functions")
    training_functions_stub.should_stop_early_from_csv = lambda *args, **kwargs: False
    sys.modules["detector2026.core.utils.training_functions"] = training_functions_stub

if "detector2026.core.utils.metrics" not in sys.modules:
    metrics_stub = types.ModuleType("detector2026.core.utils.metrics")
    metrics_stub.match_boxes_iou = _metric_stub
    metrics_stub.box_iou = _metric_stub
    metrics_stub.bbox_iou = _metric_stub

    class _ConfusionMatrixStub:
        def __init__(self, *args, **kwargs):
            pass

    metrics_stub.ConfusionMatrix = _ConfusionMatrixStub
    sys.modules["detector2026.core.utils.metrics"] = metrics_stub

if "detector2026.core.utils.evaluate" not in sys.modules:
    evaluate_stub = types.ModuleType("detector2026.core.utils.evaluate")

    class _EvalStub:
        def __init__(self, *args, **kwargs):
            pass

    evaluate_stub.EvalRunner = _EvalStub
    evaluate_stub.EvalConfig = _EvalStub
    evaluate_stub.MetricsLogger = _EvalStub
    evaluate_stub.TrainingPlots = _EvalStub
    sys.modules["detector2026.core.utils.evaluate"] = evaluate_stub


from detector2026.core.models.mr_yolo import MR_YOLO
from detector2026.core.models.tf_attn_yolo import TF_Attn_Yolo
from detector2026.core.nn.blocks import Attention, DFL, PCSA, SCSA


DEFAULT_MR_RESOLUTIONS = [(64, 1024), (128, 512), (256, 256), (512, 128), (1024, 64)]
DEFAULT_TF_HW = (256, 256)
DEFAULT_WIDTH_MULT = 0.25


def _human(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}G"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return str(value)


def _zero_ops(module: nn.Module, inputs: Tuple[Any, ...], outputs: Any) -> None:
    module.total_ops += torch.DoubleTensor([0.0])


def _thop_profile(model: nn.Module, inputs: Tuple[Any, ...]) -> Tuple[int, int]:
    macs, params = profile(
        model,
        inputs=inputs,
        custom_ops={DFL: _zero_ops, SCSA: _zero_ops},
        verbose=False,
    )
    return int(macs), int(params)


def _extra_attention_macs(module: Attention, x: torch.Tensor) -> int:
    batch_size, _, height, width = x.shape
    num_tokens = height * width
    qk = batch_size * module.num_heads * num_tokens * num_tokens * module.key_dim
    av = batch_size * module.num_heads * num_tokens * num_tokens * module.head_dim
    softmax_and_scale = 4 * batch_size * module.num_heads * num_tokens * num_tokens
    return int(qk + av + softmax_and_scale)


def _extra_pcsa_macs(module: PCSA, x: torch.Tensor) -> int:
    batch_size, channels, height, width = x.shape
    pooled_h = height // module.pool_kernel
    pooled_w = width // module.pool_kernel
    num_tokens = pooled_h * pooled_w
    if num_tokens <= 0:
        return 0
    qk = batch_size * channels * channels * num_tokens
    av = batch_size * channels * channels * num_tokens
    softmax_and_scale = 4 * batch_size * channels * channels
    return int(qk + av + softmax_and_scale)


def _missing_macs(model: nn.Module, inputs: Tuple[Any, ...]) -> int:
    extra_macs = 0
    hooks = []

    def hook_fn(module: nn.Module, module_inputs: Tuple[Any, ...], outputs: Any) -> None:
        nonlocal extra_macs
        x = module_inputs[0]
        if isinstance(module, Attention):
            extra_macs += _extra_attention_macs(module, x)
        elif isinstance(module, PCSA):
            extra_macs += _extra_pcsa_macs(module, x)

    for module in model.modules():
        if isinstance(module, (Attention, PCSA)):
            hooks.append(module.register_forward_hook(hook_fn))

    try:
        with torch.inference_mode():
            model(*inputs)
    finally:
        for hook in hooks:
            hook.remove()

    return int(extra_macs)


def _build_mr_model(device: str) -> MR_YOLO:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        model = MR_YOLO(
            input_resolutions=DEFAULT_MR_RESOLUTIONS,
            output_dir="/tmp/mr_yolo_profile",
            num_classes=20,
            reg_max=16,
            device=device,
            in_ch=1,
            width_mult=DEFAULT_WIDTH_MULT,
            backbone_mode="TFSep_pyramid",
            outfusion_channels_mult=1,
        )
    model.eval()
    return model


def _build_tf_model(device: str) -> TF_Attn_Yolo:
    model = TF_Attn_Yolo(
        output_dir="/tmp/tf_attn_yolo_profile",
        num_classes=20,
        reg_max=16,
        device=device,
        input_canals=1,
        width_mult=DEFAULT_WIDTH_MULT,
    )
    model.eval()
    return model


def _summarize(model_name: str, model: nn.Module, inputs: Tuple[Any, ...]) -> None:
    thop_macs_raw, thop_params_raw = _thop_profile(model, inputs)
    extra_macs = _missing_macs(model, inputs)
    params_manual = sum(param.numel() for param in model.parameters())
    macs_corrected = thop_macs_raw + extra_macs
    flops_corrected = 2 * macs_corrected

    print(f"\n[{model_name}]")
    print(f"thop params raw      : {thop_params_raw} ({_human(thop_params_raw)})")
    print(f"manual params        : {params_manual} ({_human(params_manual)})")
    print(f"thop macs raw        : {thop_macs_raw} ({_human(thop_macs_raw)})")
    print(f"extra macs added     : {extra_macs} ({_human(extra_macs)})")
    print(f"corrected macs       : {macs_corrected} ({_human(macs_corrected)})")
    print(f"corrected flops      : {flops_corrected} ({_human(flops_corrected)})")

    if thop_params_raw != params_manual:
        print("param check          : thop raw is incomplete on this model")
    else:
        print("param check          : ok")

    if extra_macs > 0:
        print("mac check            : plain thop undercounts unsupported matrix attention ops")
    else:
        print("mac check            : ok")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate thop on MR_YOLO and TF_Attn_Yolo.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", choices=["mr", "tf", "all"], default="all")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.model in {"mr", "all"}:
        mr_model = _build_mr_model(args.device)
        mr_inputs = ([torch.randn(1, 1, h, w) for h, w in DEFAULT_MR_RESOLUTIONS],)
        _summarize("MR_YOLO", mr_model, mr_inputs)

    if args.model in {"tf", "all"}:
        tf_model = _build_tf_model(args.device)
        tf_inputs = (torch.randn(1, 1, DEFAULT_TF_HW[0], DEFAULT_TF_HW[1]),)
        _summarize("TF_Attn_Yolo", tf_model, tf_inputs)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
