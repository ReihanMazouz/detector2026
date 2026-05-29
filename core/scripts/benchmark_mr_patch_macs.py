"""Compare MACs and parameters between MRPatchBackboneYOLOOne2ManyHead and
MRPatchMultiResRTDETRHead.

Usage:
    python benchmark_mr_patch_macs.py [--device cpu] [--batch-size 1]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import types
from typing import Any, Dict, List, Tuple

# ── minimal stubs so model imports don't pull in training/dataset deps ────────
def _metric_stub(*args, **kwargs):
    raise RuntimeError("stub")

for _mod_name, _attrs in [
    ("detector2026.core.utils.dataset", {
        "YOLODatasetFusedMultiRes": type("_S", (), {}),
        "YOLODatasetSpecificRes": type("_S", (), {}),
        "YOLODatasetSingleRes": type("_S", (), {}),
        "load_class_index_to_name": lambda *a, **k: {},
    }),
    ("detector2026.core.utils.display_outputs", {
        "plot_batch_with_boxes": lambda *a, **k: None,
        "plot_batch_matched_boxes": lambda *a, **k: None,
        "plot_predicted_boxes_batch": lambda *a, **k: None,
    }),
    ("detector2026.core.utils.training_functions", {
        "should_stop_early_from_csv": lambda *a, **k: False,
    }),
    ("detector2026.core.utils.metrics", {
        "match_boxes_iou": _metric_stub,
        "box_iou": _metric_stub,
        "bbox_iou": _metric_stub,
        "ConfusionMatrix": type("_CM", (), {"__init__": lambda s, *a, **k: None}),
    }),
    ("detector2026.core.utils.evaluate", {
        "EvalRunner": type("_E", (), {"__init__": lambda s, *a, **k: None}),
        "EvalConfig": type("_E", (), {"__init__": lambda s, *a, **k: None}),
        "MetricsLogger": type("_E", (), {"__init__": lambda s, *a, **k: None}),
        "TrainingPlots": type("_E", (), {}),
    }),
]:
    if _mod_name not in sys.modules:
        stub = types.ModuleType(_mod_name)
        for k, v in _attrs.items():
            setattr(stub, k, v)
        sys.modules[_mod_name] = stub

try:
    import torch
    import torch.nn as nn
    from thop import profile as thop_profile
except ModuleNotFoundError as exc:
    raise SystemExit("PyTorch and thop are required. Activate the detector2026 environment.") from exc

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from detector2026.core.models.mr_yolo_ablation import (  # noqa: E402
    MRPatchBackboneYOLOOne2ManyHead,
    MRPatchMultiResRTDETRHead,
)
from detector2026.core.models.Head.rtdetr import MSDeformAttn, RTDETRDecoderLayer  # noqa: E402
from detector2026.core.models.mr_yolo_ablation.mr_patch_backbone_yolo_one2many_head import (  # noqa: E402
    RestrictedInterResolutionAttention,
)
from detector2026.core.nn.blocks import DFL, SCSA  # noqa: E402

# Default architecture (must match the trained models)
DEFAULT_RESOLUTIONS: List[Tuple[int, int]] = [
    (64, 1024), (128, 512), (256, 256), (512, 128), (1024, 64)
]
DEFAULT_D_MODEL = 128
DEFAULT_PATCH_SIZE = 8
DEFAULT_ENCODER_LAYERS = 3
DEFAULT_NUM_HEADS_BACKBONE = 4
DEFAULT_NUM_INTRA_POINTS = 8
DEFAULT_NUM_INTER_NEIGHBORS = 8
DEFAULT_DIM_FFN_BACKBONE = 512
DEFAULT_P3_HW = (32, 32)
DEFAULT_STRIDE = 32
DEFAULT_NUM_CLASSES = 20
DEFAULT_REG_MAX = 16

DEFAULT_HIDDEN_DIM = 128
DEFAULT_NUM_QUERIES = 100
DEFAULT_NUM_DECODER_LAYERS = 2
DEFAULT_NUM_HEADS_DECODER = 8
DEFAULT_NUM_DECODER_POINTS = 8
DEFAULT_DIM_FFN_DECODER = 1024


# ── MACs helpers ──────────────────────────────────────────────────────────────

def _human(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3f} G"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f} M"
    if value >= 1_000:
        return f"{value / 1_000:.1f} K"
    return str(value)


def _zero_ops(module: nn.Module, inputs: Tuple[Any, ...], outputs: Any) -> None:
    module.total_ops += torch.DoubleTensor([0.0])


def _macs_rtdetr_decoder_layer(module: RTDETRDecoderLayer, target: torch.Tensor) -> int:
    """Self-attention (MHA) inside the RT-DETR decoder layer."""
    t = target.transpose(0, 1) if not getattr(module.self_attn, "batch_first", False) else target
    if t.dim() != 3:
        return 0
    batch_size, q_len, embed_dim = t.shape
    num_heads = module.self_attn.num_heads
    head_dim = embed_dim // num_heads
    qkv_proj = batch_size * 3 * q_len * embed_dim * embed_dim
    out_proj = batch_size * q_len * embed_dim * embed_dim
    qk = batch_size * num_heads * q_len * q_len * head_dim
    av = qk
    return int(qkv_proj + out_proj + qk + av)


def _macs_ms_deform_attn(module: MSDeformAttn, query: torch.Tensor) -> int:
    """Sampling-based cross-attention — only the weighted sum part is missed by thop."""
    if query.dim() != 3:
        return 0
    batch_size, num_queries, embed_dim = query.shape
    head_dim = embed_dim // module.num_heads
    return int(batch_size * num_queries * module.num_heads * module.num_levels * module.num_points * head_dim)


def _macs_restricted_inter_res_attn(
    module: RestrictedInterResolutionAttention,
    target: torch.Tensor,
    sources: List[torch.Tensor],
) -> int:
    """QK dot-product and AV weighted sum inside RestrictedInterResolutionAttention.
    thop already counts the Q/K/V/out linear projections.
    """
    if target.dim() != 3:
        return 0
    batch_size, num_target, _ = target.shape
    # per source: 2 × B × N_t × M × d_model  (QK reduce + AV weighted sum)
    per_source = 2 * batch_size * num_target * module.num_neighbors * module.d_model
    return int(per_source * len(sources))


def _missing_macs(model: nn.Module, inputs: Tuple[Any, ...]) -> int:
    extra = 0
    hooks = []

    def _hook(module: nn.Module, module_inputs: Tuple[Any, ...], _outputs: Any) -> None:
        nonlocal extra
        if isinstance(module, RTDETRDecoderLayer):
            extra += _macs_rtdetr_decoder_layer(module, module_inputs[0])
        elif isinstance(module, MSDeformAttn):
            extra += _macs_ms_deform_attn(module, module_inputs[0])
        elif isinstance(module, RestrictedInterResolutionAttention):
            target = module_inputs[0]
            sources = module_inputs[1] if len(module_inputs) > 1 else []
            extra += _macs_restricted_inter_res_attn(module, target, sources)

    for m in model.modules():
        if isinstance(m, (RTDETRDecoderLayer, MSDeformAttn, RestrictedInterResolutionAttention)):
            hooks.append(m.register_forward_hook(_hook))
    try:
        with torch.inference_mode():
            model(list(inputs))
    finally:
        for h in hooks:
            h.remove()

    return extra


class _ListForwardWrapper(nn.Module):
    """Wraps a model whose forward(inputs: List[Tensor]) so thop can call it
    with positional arguments (*tensors) instead of a single list."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
    def forward(self, *args):
        return self.model(list(args))


def _profile(model_name: str, model: nn.Module, inputs: Tuple[Any, ...]) -> Dict[str, Any]:
    model.eval()
    wrapped = _ListForwardWrapper(model)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        thop_macs, thop_params = thop_profile(
            wrapped, inputs=inputs,
            custom_ops={DFL: _zero_ops, SCSA: _zero_ops},
            verbose=False,
        )
    extra = _missing_macs(model, inputs)  # hooks directly on the unwrapped model
    params = sum(p.numel() for p in model.parameters())
    macs_total = int(thop_macs) + extra

    print(f"\n{'─'*60}")
    print(f"  {model_name}")
    print(f"{'─'*60}")
    print(f"  Params              : {_human(params)} ({params:,})")
    print(f"  thop MACs (raw)     : {_human(int(thop_macs))}")
    print(f"  missing MACs added  : {_human(extra)}")
    print(f"  Total MACs          : {_human(macs_total)}")
    print(f"  FLOPs (×2)          : {_human(2 * macs_total)}")

    return {"name": model_name, "params": params, "macs": macs_total, "macs_thop": int(thop_macs), "macs_extra": extra}


# ── model builders ────────────────────────────────────────────────────────────

def _common_kwargs(input_resolutions, in_ch):
    return dict(
        input_resolutions=input_resolutions,
        output_dir="/tmp/_profile",
        num_classes=DEFAULT_NUM_CLASSES,
        reg_max=DEFAULT_REG_MAX,
        in_ch=in_ch,
        d_model=DEFAULT_D_MODEL,
        patch_size=DEFAULT_PATCH_SIZE,
        num_intra_points=DEFAULT_NUM_INTRA_POINTS,
        num_inter_neighbors=DEFAULT_NUM_INTER_NEIGHBORS,
        dropout=0.0,
    )


def build_yolo_one2many(resolutions, device, in_ch):
    with contextlib.redirect_stdout(io.StringIO()):
        model = MRPatchBackboneYOLOOne2ManyHead(
            **_common_kwargs(resolutions, in_ch),
            num_encoder_layers=DEFAULT_ENCODER_LAYERS,
            num_heads=DEFAULT_NUM_HEADS_BACKBONE,
            dim_feedforward=DEFAULT_DIM_FFN_BACKBONE,
            p3_hw=DEFAULT_P3_HW,
            stride=DEFAULT_STRIDE,
            device=device,
        )
    return model.eval()


def build_multires_rtdetr(
    resolutions,
    device,
    in_ch,
    num_encoder_layers: int = DEFAULT_ENCODER_LAYERS,
    num_decoder_layers: int = DEFAULT_NUM_DECODER_LAYERS,
    num_decoder_points: int = DEFAULT_NUM_DECODER_POINTS,
    num_heads_decoder: int = DEFAULT_NUM_HEADS_DECODER,
):
    with contextlib.redirect_stdout(io.StringIO()):
        model = MRPatchMultiResRTDETRHead(
            **_common_kwargs(resolutions, in_ch),
            num_encoder_layers=num_encoder_layers,
            num_heads_backbone=DEFAULT_NUM_HEADS_BACKBONE,
            dim_feedforward_backbone=DEFAULT_DIM_FFN_BACKBONE,
            device=device,
            hidden_dim=DEFAULT_HIDDEN_DIM,
            num_queries=DEFAULT_NUM_QUERIES,
            num_decoder_layers=num_decoder_layers,
            num_heads_decoder=num_heads_decoder,
            num_decoder_points=num_decoder_points,
            dim_feedforward_decoder=DEFAULT_DIM_FFN_DECODER,
        )
    return model.eval()


# ── sweep configurations ──────────────────────────────────────────────────────

# (label, enc_layers, dec_layers, dec_points, dec_heads)
SWEEP_CONFIGS = [
    # reference: original YOLO one2many with fused 32×32  (built separately)
    # MultiRes RT-DETR configs
    ("MR-RTDETR  enc3 dec2 pts8  h8",  3, 2,  8, 8),
    ("MR-RTDETR  enc4 dec2 pts8  h8",  4, 2,  8, 8),
    ("MR-RTDETR  enc4 dec3 pts8  h8",  4, 3,  8, 8),
    ("MR-RTDETR  enc4 dec4 pts8  h8",  4, 4,  8, 8),
    ("MR-RTDETR  enc4 dec2 pts16 h8",  4, 2, 16, 8),
    ("MR-RTDETR  enc4 dec4 pts16 h8",  4, 4, 16, 8),
    ("MR-RTDETR  enc4 dec4 pts8  h4",  4, 4,  8, 4),
]


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Sweep MACs across MR patch backbone configs.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--in-ch", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    resolutions = DEFAULT_RESOLUTIONS
    batch = args.batch_size
    in_ch = args.in_ch

    print(f"Input resolutions : {resolutions}")
    print(f"Patch size        : {DEFAULT_PATCH_SIZE}")
    patch_shapes = [(h // DEFAULT_PATCH_SIZE, w // DEFAULT_PATCH_SIZE) for h, w in resolutions]
    token_counts = [h * w for h, w in patch_shapes]
    print(f"Patch grids       : {patch_shapes}")
    print(f"Tokens per res    : {token_counts}  total={sum(token_counts)}")
    print(f"Batch size        : {batch}  |  device: {args.device}\n")

    inputs = tuple(
        torch.zeros(batch, in_ch, h, w, device=args.device)
        for h, w in resolutions
    )

    results = []

    # reference YOLO one2many
    model_ref = build_yolo_one2many(resolutions, args.device, in_ch)
    r_ref = _profile("YOLO-one2many  enc3 fused-32×32  (reference)", model_ref, inputs)
    del model_ref
    results.append(r_ref)

    # sweep
    for label, enc, dec, pts, heads in SWEEP_CONFIGS:
        model = build_multires_rtdetr(resolutions, args.device, in_ch,
                                        num_encoder_layers=enc,
                                        num_decoder_layers=dec,
                                        num_decoder_points=pts,
                                        num_heads_decoder=heads)
        r = _profile(label, model, inputs)
        del model
        results.append(r)

    # ── summary table ─────────────────────────────────────────────────────────
    ref_macs = results[0]["macs"]
    ref_params = results[0]["params"]

    col_name  = 46
    col_num   = 10
    sep = f"  {'─'*col_name}  {'─'*col_num}  {'─'*col_num}  {'─'*8}  {'─'*8}"

    print(f"\n{'═'*90}")
    print("  SWEEP SUMMARY")
    print(f"{'═'*90}")
    print(f"  {'Configuration':<{col_name}}  {'Params':>{col_num}}  {'MACs':>{col_num}}  {'Δ MACs':>8}  {'Δ Params':>8}")
    print(sep)
    for r in results:
        delta_macs   = f"+{(r['macs']   - ref_macs)   / 1e6:.0f} M"   if r["macs"]   != ref_macs   else "—"
        delta_params = f"+{(r['params'] - ref_params) / 1e6:.2f} M" if r["params"] != ref_params else "—"
        print(f"  {r['name']:<{col_name}}  {_human(r['params']):>{col_num}}  {_human(r['macs']):>{col_num}}  {delta_macs:>8}  {delta_params:>8}")

    print(f"\n  Note: Δ is relative to the YOLO reference.")
    print(f"  Encoder cost  ≈ {_human(ref_macs)} (dominates in all configs)")


if __name__ == "__main__":
    main()
