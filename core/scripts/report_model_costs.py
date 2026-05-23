from __future__ import annotations

import argparse
import csv
import contextlib
import io
import json
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
    dataset_stub.load_class_index_to_name = lambda *args, **kwargs: {}
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
from detector2026.core.models.yolov11 import YOLOv11
from detector2026.core.models.Head.rtdetr import MSDeformAttn, RTDETRDecoderLayer
from detector2026.core.models.Neck.transformer import TransformerPyramidNeck
from detector2026.core.models.yolov11_ablation.yolov11_rtdetr import YOLOv11RTDETR
from detector2026.core.models.yolov11_ablation.yolov11_rtdetr_head import YOLOv11RTDETRHead
from detector2026.core.models.yolov11_ablation.yolov11_transformer_neck import YOLOv11TransformerNeck
from detector2026.core.nn.blocks import Attention, DFL, PCSA, SCSA


DEFAULT_MR_RESOLUTIONS = [(64, 1024), (128, 512), (256, 256), (512, 128), (1024, 64)]
DEFAULT_TF_HW = (256, 256)
DEFAULT_WIDTH_MULT = 0.25
DEFAULT_NUM_CLASSES = 20
DEFAULT_REG_MAX = 16
DEFAULT_RTDETR_HIDDEN_DIM = 128
DEFAULT_RTDETR_NUM_QUERIES = 100
DEFAULT_RTDETR_DECODER_LAYERS = 6
DEFAULT_RTDETR_NUM_HEADS = 8
DEFAULT_RTDETR_DECODER_POINTS = 4
DEFAULT_RTDETR_FFN_DIM = 1024
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "tmp"


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


def _extra_mha_macs(module: nn.MultiheadAttention, query: torch.Tensor, key: torch.Tensor) -> int:
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


def _extra_transformer_neck_encoder_macs(module: TransformerPyramidNeck, features: Tuple[torch.Tensor, ...]) -> int:
    if len(features) != 3:
        return 0
    first_layer = module.encoder.layers[0]
    embed_dim = first_layer.self_attn.embed_dim
    num_heads = first_layer.self_attn.num_heads
    head_dim = embed_dim // num_heads
    ffn_dim = first_layer.linear1.out_features

    batch_size = features[0].shape[0]
    num_tokens = sum(feat.shape[-2] * feat.shape[-1] for feat in features)
    qkv_proj = batch_size * 3 * num_tokens * embed_dim * embed_dim
    out_proj = batch_size * num_tokens * embed_dim * embed_dim
    qk = batch_size * num_heads * num_tokens * num_tokens * head_dim
    av = batch_size * num_heads * num_tokens * num_tokens * head_dim
    softmax_and_scale = 4 * batch_size * num_heads * num_tokens * num_tokens
    ffn = batch_size * num_tokens * (embed_dim * ffn_dim + ffn_dim * embed_dim)
    per_layer = qkv_proj + out_proj + qk + av + softmax_and_scale + ffn
    return int(per_layer * module.encoder.num_layers)


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
        elif isinstance(module, RTDETRDecoderLayer):
            extra_macs += _extra_rtdetr_decoder_layer_macs(module, x)
        elif isinstance(module, MSDeformAttn):
            extra_macs += _extra_ms_deform_attn_macs(module, x)
        elif isinstance(module, TransformerPyramidNeck):
            extra_macs += _extra_transformer_neck_encoder_macs(module, module_inputs)

    for module in model.modules():
        if isinstance(module, (Attention, PCSA, RTDETRDecoderLayer, MSDeformAttn, TransformerPyramidNeck)):
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
            num_classes=DEFAULT_NUM_CLASSES,
            reg_max=DEFAULT_REG_MAX,
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
        num_classes=DEFAULT_NUM_CLASSES,
        reg_max=DEFAULT_REG_MAX,
        device=device,
        input_canals=1,
        width_mult=DEFAULT_WIDTH_MULT,
    )
    model.eval()
    return model


def _build_yolov11_model(device: str) -> YOLOv11:
    model = YOLOv11(
        output_dir="/tmp/yolov11_profile",
        num_classes=DEFAULT_NUM_CLASSES,
        reg_max=DEFAULT_REG_MAX,
        device=device,
        input_canals=1,
        width_mult=DEFAULT_WIDTH_MULT,
        input_hw=DEFAULT_TF_HW,
    )
    model.eval()
    return model


def _rtdetr_head_kwargs() -> Dict[str, Any]:
    return {
        "hidden_dim": DEFAULT_RTDETR_HIDDEN_DIM,
        "num_queries": DEFAULT_RTDETR_NUM_QUERIES,
        "num_decoder_layers": DEFAULT_RTDETR_DECODER_LAYERS,
        "num_heads": DEFAULT_RTDETR_NUM_HEADS,
        "num_decoder_points": DEFAULT_RTDETR_DECODER_POINTS,
        "use_deformable_attention": True,
        "dim_feedforward": DEFAULT_RTDETR_FFN_DIM,
    }


def _build_yolov11_rtdetr_head_model(device: str) -> YOLOv11RTDETRHead:
    model = YOLOv11RTDETRHead(
        output_dir="/tmp/yolov11_rtdetr_head_profile",
        num_classes=DEFAULT_NUM_CLASSES,
        reg_max=DEFAULT_REG_MAX,
        device=device,
        input_canals=1,
        width_mult=DEFAULT_WIDTH_MULT,
        input_hw=DEFAULT_TF_HW,
        **_rtdetr_head_kwargs(),
    )
    model.use_one2one_head()
    model.eval()
    return model


def _build_yolov11_rtdetr_full_model(device: str) -> YOLOv11RTDETR:
    model = YOLOv11RTDETR(
        output_dir="/tmp/yolov11_rtdetr_full_profile",
        num_classes=DEFAULT_NUM_CLASSES,
        reg_max=DEFAULT_REG_MAX,
        device=device,
        input_canals=1,
        width_mult=DEFAULT_WIDTH_MULT,
        input_hw=DEFAULT_TF_HW,
        **_rtdetr_head_kwargs(),
    )
    model.eval()
    return model


def _build_yolov11_transformer_neck_model(device: str) -> YOLOv11TransformerNeck:
    model = YOLOv11TransformerNeck(
        output_dir="/tmp/yolov11_transformer_neck_profile",
        num_classes=DEFAULT_NUM_CLASSES,
        reg_max=DEFAULT_REG_MAX,
        device=device,
        input_canals=1,
        width_mult=DEFAULT_WIDTH_MULT,
        input_hw=DEFAULT_TF_HW,
    )
    model.eval()
    return model


def _build_yolov11_deformable_neck_model(device: str) -> YOLOv11TransformerNeck:
    model = YOLOv11TransformerNeck(
        output_dir="/tmp/yolov11_deformable_neck_profile",
        num_classes=DEFAULT_NUM_CLASSES,
        reg_max=DEFAULT_REG_MAX,
        device=device,
        input_canals=1,
        width_mult=DEFAULT_WIDTH_MULT,
        input_hw=DEFAULT_TF_HW,
        transformer_d_model=128,
        transformer_num_heads=4,
        transformer_num_layers=1,
        transformer_num_points=4,
        transformer_ffn_ratio=2.0,
        transformer_dropout=0.0,
        transformer_residual_scale=0.0,
        transformer_neck_type="deformable",
    )
    model.eval()
    return model


def _summarize(model_name: str, model: nn.Module, inputs: Tuple[Any, ...]) -> Dict[str, Any]:
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

    return {
        "model": model_name,
        "params": params_manual,
        "params_human": _human(params_manual),
        "thop_params": thop_params_raw,
        "thop_macs": thop_macs_raw,
        "extra_attention_macs": extra_macs,
        "macs": macs_corrected,
        "macs_human": _human(macs_corrected),
        "flops": flops_corrected,
        "flops_human": _human(flops_corrected),
        "input": _describe_inputs(inputs),
    }


def _describe_inputs(inputs: Tuple[Any, ...]) -> str:
    first = inputs[0]
    if isinstance(first, list):
        return ", ".join(f"1x1x{tensor.shape[-2]}x{tensor.shape[-1]}" for tensor in first)
    if torch.is_tensor(first):
        return "x".join(str(dim) for dim in first.shape)
    return str(type(first).__name__)


def _write_reports(rows: Sequence[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "model_costs.csv"
    json_path = output_dir / "model_costs.json"
    md_path = output_dir / "model_costs.md"

    fieldnames = [
        "model",
        "input",
        "params",
        "params_human",
        "thop_macs",
        "extra_attention_macs",
        "macs",
        "macs_human",
        "flops",
        "flops_human",
    ]
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})

    with json_path.open("w") as json_file:
        json.dump(list(rows), json_file, indent=2)

    with md_path.open("w") as md_file:
        md_file.write("| Model | Input | Params | MACs corrigees | FLOPs corrigees | MACs attention ajoutes |\n")
        md_file.write("| --- | --- | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            md_file.write(
                f"| {row['model']} | {row['input']} | {row['params_human']} | "
                f"{row['macs_human']} | {row['flops_human']} | {_human(row['extra_attention_macs'])} |\n"
            )

    print(f"\nReports written to:")
    print(f"- {csv_path}")
    print(f"- {json_path}")
    print(f"- {md_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report corrected model costs for detector2026 models.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--model",
        choices=[
            "mr",
            "tf",
            "yolov11",
            "yolov11-rtdetr-head",
            "yolov11-rtdetr-full",
            "yolov11-transformer-neck",
            "yolov11-deformable-neck",
            "yolov11-ablation",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: List[Dict[str, Any]] = []
    single_image_inputs = (torch.randn(1, 1, DEFAULT_TF_HW[0], DEFAULT_TF_HW[1]),)

    if args.model in {"mr", "all"}:
        mr_model = _build_mr_model(args.device)
        mr_inputs = ([torch.randn(1, 1, h, w) for h, w in DEFAULT_MR_RESOLUTIONS],)
        rows.append(_summarize("MR_YOLO", mr_model, mr_inputs))

    if args.model in {"tf", "all"}:
        tf_model = _build_tf_model(args.device)
        rows.append(_summarize("TF_Attn_Yolo", tf_model, single_image_inputs))

    if args.model in {"yolov11", "yolov11-ablation", "all"}:
        yolov11_model = _build_yolov11_model(args.device)
        rows.append(_summarize("YOLOv11", yolov11_model, single_image_inputs))

    if args.model in {"yolov11-rtdetr-head", "yolov11-ablation", "all"}:
        rtdetr_head_model = _build_yolov11_rtdetr_head_model(args.device)
        rows.append(_summarize("YOLOv11_RTDETR_Head", rtdetr_head_model, single_image_inputs))

    if args.model in {"yolov11-rtdetr-full", "yolov11-ablation", "all"}:
        rtdetr_full_model = _build_yolov11_rtdetr_full_model(args.device)
        rows.append(_summarize("YOLOv11_RTDETR_Full", rtdetr_full_model, single_image_inputs))

    if args.model in {"yolov11-transformer-neck", "yolov11-ablation", "all"}:
        transformer_neck_model = _build_yolov11_transformer_neck_model(args.device)
        rows.append(_summarize("YOLOv11_Transformer_Neck", transformer_neck_model, single_image_inputs))

    if args.model in {"yolov11-deformable-neck", "yolov11-ablation", "all"}:
        deformable_neck_model = _build_yolov11_deformable_neck_model(args.device)
        rows.append(_summarize("YOLOv11_Deformable_Neck", deformable_neck_model, single_image_inputs))

    _write_reports(rows, args.output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
