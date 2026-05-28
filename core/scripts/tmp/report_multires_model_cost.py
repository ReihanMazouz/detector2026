from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import types
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def _metric_stub(*args, **kwargs):
    raise RuntimeError("Optional training/evaluation dependency unavailable in this environment")


try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is required to run this script. Activate the detector2026 environment first."
    ) from exc

try:
    from thop import profile
except ModuleNotFoundError as exc:
    raise SystemExit(
        "thop is required to compute FLOPs/params. Install dependencies from detector2026 first."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(REPO_ROOT))


if "torchinfo" not in sys.modules:
    torchinfo_stub = types.ModuleType("torchinfo")

    def _summary_stub(*args, **kwargs):
        return "torchinfo summary unavailable in this environment"

    torchinfo_stub.summary = _summary_stub
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
    dataset_stub.load_class_index_to_name = lambda *args, **kwargs: {}
    sys.modules["detector2026.core.utils.dataset"] = dataset_stub

if "detector2026.core.utils.display_outputs" not in sys.modules:
    display_stub = types.ModuleType("detector2026.core.utils.display_outputs")

    def _noop(*args, **kwargs):
        return None

    display_stub.plot_batch_with_boxes = _noop
    display_stub.plot_batch_matched_boxes = _noop
    display_stub.plot_predicted_boxes_batch = _noop
    sys.modules["detector2026.core.utils.display_outputs"] = display_stub

if "detector2026.core.utils.training_functions" not in sys.modules:
    training_functions_stub = types.ModuleType("detector2026.core.utils.training_functions")
    training_functions_stub.should_stop_early_from_csv = lambda *args, **kwargs: False
    sys.modules["detector2026.core.utils.training_functions"] = training_functions_stub

if "detector2026.core.utils.metrics" not in sys.modules:
    metrics_stub = types.ModuleType("detector2026.core.utils.metrics")
    metrics_stub.match_boxes_iou = _metric_stub

    class _ConfusionMatrixStub:
        def __init__(self, *args, **kwargs):
            pass

    metrics_stub.ConfusionMatrix = _ConfusionMatrixStub
    metrics_stub.box_iou = _metric_stub
    metrics_stub.bbox_iou = _metric_stub
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


from detector2026.core.models.Head.rtdetr import MSDeformAttn, RTDETRDecoderLayer
from detector2026.core.models.mr_yolo import MR_YOLO
from detector2026.core.models.mr_yolo_ablation import MRViTPatchDetector


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_JSON = SCRIPT_DIR / "multires_model_cost.json"
OUTPUT_MD = SCRIPT_DIR / "multires_model_cost.md"
ARTIFACT_DIR = SCRIPT_DIR / "_model_cost_artifacts"

NUM_CLASSES = 20
WIDTH_MULT = 0.25
REG_MAX = 16
IN_CH = 1
BACKBONE_MODE = "TFSep_pyramid"
OUTFUSION_CHANNELS_MULT = 1
MR_VIT_PATCH_GRID = (32, 32)
MR_VIT_D_MODEL = 256
MR_VIT_ENCODER_LAYERS = 6
MR_VIT_DECODER_LAYERS = 6
MR_VIT_QUERIES = 100
MR_VIT_HEADS = 8
MR_VIT_ENCODER_POINTS = 16
MR_VIT_DECODER_POINTS = 16
MR_VIT_FFN_DIM = 1024
MR_VIT_DROPOUT = 0.0


def _benchmark_configs() -> Dict[str, List[Tuple[int, int]]]:
    full_7 = [
        (32, 2048),
        (64, 1024),
        (128, 512),
        (256, 256),
        (512, 128),
        (1024, 64),
        (2048, 32),
    ]
    return {
        "mr_1": [(256, 256)],
        "mr_2": [(128, 512), (512, 128)],
        "mr_3": [(128, 512), (256, 256), (512, 128)],
        "mr_5": [(64, 1024), (128, 512), (256, 256), (512, 128), (1024, 64)],
        "mr_7": full_7,
        "mr_2_far_gap_2_pow_5": [(64, 1024), (1024, 64)],
    }


def _resolution_label(resolutions: Sequence[Tuple[int, int]]) -> str:
    return ", ".join(f"{h}x{w}" for h, w in resolutions)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _build_inputs(
    resolutions: Sequence[Tuple[int, int]],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> List[torch.Tensor]:
    return [
        torch.randn(batch_size, IN_CH, h, w, device=device, dtype=dtype)
        for h, w in resolutions
    ]


def _build_model(
    resolutions: Sequence[Tuple[int, int]],
    device: torch.device,
    output_dir: Path,
) -> MR_YOLO:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        model = MR_YOLO(
            input_resolutions=list(resolutions),
            output_dir=str(output_dir),
            num_classes=NUM_CLASSES,
            reg_max=REG_MAX,
            device=str(device),
            in_ch=IN_CH,
            width_mult=WIDTH_MULT,
            backbone_mode=BACKBONE_MODE,
            outfusion_channels_mult=OUTFUSION_CHANNELS_MULT,
        )
    model.eval()
    return model


def _build_mr_vit_patch_detector(
    resolutions: Sequence[Tuple[int, int]],
    device: torch.device,
    output_dir: Path,
) -> MRViTPatchDetector:
    model = MRViTPatchDetector(
        input_resolutions=list(resolutions),
        output_dir=str(output_dir),
        num_classes=NUM_CLASSES,
        device=str(device),
        in_ch=IN_CH,
        d_model=MR_VIT_D_MODEL,
        num_encoder_layers=MR_VIT_ENCODER_LAYERS,
        num_decoder_layers=MR_VIT_DECODER_LAYERS,
        num_queries=MR_VIT_QUERIES,
        patch_grid_hw=MR_VIT_PATCH_GRID,
        num_heads=MR_VIT_HEADS,
        num_encoder_points=MR_VIT_ENCODER_POINTS,
        num_decoder_points=MR_VIT_DECODER_POINTS,
        dim_feedforward=MR_VIT_FFN_DIM,
        dropout=MR_VIT_DROPOUT,
    )
    model.eval()
    return model


def _human_number(value: float) -> str:
    abs_value = abs(float(value))
    if abs_value >= 1e9:
        return f"{value / 1e9:.2f}G"
    if abs_value >= 1e6:
        return f"{value / 1e6:.2f}M"
    if abs_value >= 1e3:
        return f"{value / 1e3:.2f}K"
    return f"{value:.0f}"


def _extra_mha_macs(module: torch.nn.MultiheadAttention, query: torch.Tensor, key: torch.Tensor) -> int:
    if query.dim() != 3 or key.dim() != 3:
        return 0
    if module.batch_first:
        batch_size, q_len, embed_dim = query.shape
        _, k_len, _ = key.shape
    else:
        q_len, batch_size, embed_dim = query.shape
        k_len = key.shape[0]
    num_heads = module.num_heads
    head_dim = embed_dim // num_heads
    qkv_proj = batch_size * (q_len + 2 * k_len) * embed_dim * embed_dim
    out_proj = batch_size * q_len * embed_dim * embed_dim
    qk = batch_size * num_heads * q_len * k_len * head_dim
    av = batch_size * num_heads * q_len * k_len * head_dim
    softmax_and_scale = 4 * batch_size * num_heads * q_len * k_len
    return int(qkv_proj + out_proj + qk + av + softmax_and_scale)


def _extra_rtdetr_decoder_layer_macs(module: RTDETRDecoderLayer, target: torch.Tensor) -> int:
    return _extra_mha_macs(module.self_attn, target.transpose(0, 1), target.transpose(0, 1))


def _extra_ms_deform_attn_macs(module: MSDeformAttn, query: torch.Tensor) -> int:
    if query.dim() != 3:
        return 0
    batch_size, num_queries, embed_dim = query.shape
    head_dim = embed_dim // module.num_heads
    sample_points = batch_size * num_queries * module.num_heads * module.num_levels * module.num_points
    return int(sample_points * head_dim)


def _missing_attention_macs(model: torch.nn.Module, inputs: Tuple[object, ...]) -> int:
    extra_macs = 0
    hooks = []

    def hook_fn(module: torch.nn.Module, module_inputs: Tuple[object, ...], outputs: object) -> None:
        nonlocal extra_macs
        x = module_inputs[0]
        if isinstance(module, RTDETRDecoderLayer):
            extra_macs += _extra_rtdetr_decoder_layer_macs(module, x)
        elif isinstance(module, MSDeformAttn):
            extra_macs += _extra_ms_deform_attn_macs(module, x)

    for module in model.modules():
        if isinstance(module, (RTDETRDecoderLayer, MSDeformAttn)):
            hooks.append(module.register_forward_hook(hook_fn))

    try:
        with torch.inference_mode():
            model(*inputs)
    finally:
        for hook in hooks:
            hook.remove()
    return int(extra_macs)


def _activation_bytes(model: torch.nn.Module, inputs: Tuple[object, ...]) -> int:
    total = 0
    hooks = []

    def add_tensor(tensor: torch.Tensor) -> None:
        nonlocal total
        total += tensor.numel() * tensor.element_size()

    def visit(value: object) -> None:
        if torch.is_tensor(value):
            add_tensor(value)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    def hook_fn(_module: torch.nn.Module, _module_inputs: Tuple[object, ...], outputs: object) -> None:
        visit(outputs)

    for module in model.modules():
        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(hook_fn))

    try:
        with torch.inference_mode():
            model(*inputs)
    finally:
        for hook in hooks:
            hook.remove()
    return int(total)


def _profile_config(
    label: str,
    resolutions: Sequence[Tuple[int, int]],
    device: torch.device,
    batch_size: int,
) -> Dict[str, object]:
    run_dir = ARTIFACT_DIR / label
    run_dir.mkdir(parents=True, exist_ok=True)

    model = _build_model(resolutions, device=device, output_dir=run_dir)
    inputs = _build_inputs(resolutions, batch_size=batch_size, device=device)

    with torch.inference_mode():
        macs, params = profile(model, inputs=(inputs,), verbose=False)
    activation_bytes = _activation_bytes(model, (inputs,))

    return {
        "label": label,
        "num_resolutions": len(resolutions),
        "resolutions": [list(hw) for hw in resolutions],
        "resolution_label": _resolution_label(resolutions),
        "batch_size": batch_size,
        "params": int(params),
        "macs": int(macs),
        "flops": int(2 * macs),
        "extra_attention_macs": 0,
        "activation_bytes": activation_bytes,
        "params_human": _human_number(params),
        "macs_human": _human_number(macs),
        "flops_human": _human_number(2 * macs),
        "activation_human": _human_number(activation_bytes),
    }


def _profile_mr_vit_patch_detector(
    label: str,
    resolutions: Sequence[Tuple[int, int]],
    device: torch.device,
    batch_size: int,
) -> Dict[str, object]:
    run_dir = ARTIFACT_DIR / label
    run_dir.mkdir(parents=True, exist_ok=True)

    model = _build_mr_vit_patch_detector(resolutions, device=device, output_dir=run_dir)
    inputs = _build_inputs(resolutions, batch_size=batch_size, device=device)
    profile_inputs = (inputs,)

    with torch.inference_mode():
        thop_macs, thop_params = profile(model, inputs=profile_inputs, verbose=False)
    extra_macs = _missing_attention_macs(model, profile_inputs)
    params = sum(param.numel() for param in model.parameters())
    macs = int(thop_macs) + int(extra_macs)
    activation_bytes = _activation_bytes(model, profile_inputs)

    return {
        "label": label,
        "num_resolutions": len(resolutions),
        "resolutions": [list(hw) for hw in resolutions],
        "resolution_label": _resolution_label(resolutions),
        "batch_size": batch_size,
        "params": int(params),
        "thop_params": int(thop_params),
        "thop_macs": int(thop_macs),
        "extra_attention_macs": int(extra_macs),
        "macs": macs,
        "flops": int(2 * macs),
        "activation_bytes": activation_bytes,
        "params_human": _human_number(params),
        "macs_human": _human_number(macs),
        "flops_human": _human_number(2 * macs),
        "activation_human": _human_number(activation_bytes),
        "details": (
            f"patch_grid={MR_VIT_PATCH_GRID[0]}x{MR_VIT_PATCH_GRID[1]}, "
            f"d={MR_VIT_D_MODEL}, enc={MR_VIT_ENCODER_LAYERS}, "
            f"dec={MR_VIT_DECODER_LAYERS}, queries={MR_VIT_QUERIES}, "
            f"heads={MR_VIT_HEADS}, points={MR_VIT_ENCODER_POINTS}/{MR_VIT_DECODER_POINTS}"
        ),
    }


def _to_markdown(rows: Sequence[Dict[str, object]]) -> str:
    lines = [
        "| Config | Resolutions | Params | MACs corrigees | FLOPs corrigees | MACs attention ajoutes | Activation memoire |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['label']}` | `{row['resolution_label']}` | "
            f"`{row['params_human']}` ({row['params']}) | "
            f"`{row['macs_human']}` ({row['macs']}) | "
            f"`{row['flops_human']}` ({row['flops']}) | "
            f"`{_human_number(row.get('extra_attention_macs', 0))}` ({row.get('extra_attention_macs', 0)}) | "
            f"`{row['activation_human']}` ({row['activation_bytes']}) |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report parameter counts and FLOPs for the multi-resolution configurations."
    )
    parser.add_argument("--device", default="auto", help="Device to use: auto, cpu, cuda:0, ...")
    parser.add_argument("--batch-size", type=int, default=1, help="Dummy batch size for profiling.")
    parser.add_argument(
        "--model",
        choices=["mr_yolo", "mr_vit_patch_detector", "all"],
        default="all",
        help="Model family to profile.",
    )
    parser.add_argument("--json-out", type=Path, default=OUTPUT_JSON, help="Output JSON path.")
    parser.add_argument("--md-out", type=Path, default=OUTPUT_MD, help="Output Markdown path.")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    configs = _benchmark_configs()

    rows: List[Dict[str, object]] = []
    if args.model in {"mr_yolo", "all"}:
        for label, resolutions in configs.items():
            print(f"[INFO] Profiling {label}: {_resolution_label(resolutions)} on {device}")
            row = _profile_config(
                label=label,
                resolutions=resolutions,
                device=device,
                batch_size=args.batch_size,
            )
            rows.append(row)
            print(
                f"[OK] {label}: params={row['params_human']} ({row['params']}), "
                f"macs={row['macs_human']} ({row['macs']}), "
                f"flops={row['flops_human']} ({row['flops']})"
            )
    if args.model in {"mr_vit_patch_detector", "all"}:
        label = "mr_vit_patch_detector"
        resolutions = configs["mr_5"]
        print(f"[INFO] Profiling {label}: {_resolution_label(resolutions)} on {device}")
        row = _profile_mr_vit_patch_detector(
            label=label,
            resolutions=resolutions,
            device=device,
            batch_size=args.batch_size,
        )
        rows.append(row)
        print(
            f"[OK] {label}: params={row['params_human']} ({row['params']}), "
            f"macs={row['macs_human']} ({row['macs']}), "
            f"flops={row['flops_human']} ({row['flops']}), "
            f"extra={_human_number(row['extra_attention_macs'])}"
        )

    payload = {
        "device": str(device),
        "batch_size": args.batch_size,
        "model_family": args.model,
        "model_kwargs": {
            "num_classes": NUM_CLASSES,
            "width_mult": WIDTH_MULT,
            "reg_max": REG_MAX,
            "in_ch": IN_CH,
            "backbone_mode": BACKBONE_MODE,
            "outfusion_channels_mult": OUTFUSION_CHANNELS_MULT,
            "mr_vit_patch_detector": {
                "patch_grid_hw": MR_VIT_PATCH_GRID,
                "d_model": MR_VIT_D_MODEL,
                "num_encoder_layers": MR_VIT_ENCODER_LAYERS,
                "num_decoder_layers": MR_VIT_DECODER_LAYERS,
                "num_queries": MR_VIT_QUERIES,
                "num_heads": MR_VIT_HEADS,
                "num_encoder_points": MR_VIT_ENCODER_POINTS,
                "num_decoder_points": MR_VIT_DECODER_POINTS,
                "dim_feedforward": MR_VIT_FFN_DIM,
                "dropout": MR_VIT_DROPOUT,
            },
        },
        "results": rows,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(_to_markdown(rows), encoding="utf-8")

    print(f"[DONE] JSON written to {args.json_out}")
    print(f"[DONE] Markdown written to {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
