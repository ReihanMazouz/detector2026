from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import statistics
import sys
import time
import types
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def _metric_stub(*args, **kwargs):
    raise RuntimeError("Optional training/evaluation dependency unavailable in this environment")

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is required to run this benchmark. Activate the detector2026 environment first."
    ) from exc

try:
    if hasattr(torch.backends, "nnpack") and hasattr(torch.backends.nnpack, "enabled"):
        torch.backends.nnpack.enabled = False
except Exception:
    pass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

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

from detector2026.core.models.mr_yolo import MR_YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_JSON = SCRIPT_DIR / "multires_latency_benchmark.json"
OUTPUT_MD = SCRIPT_DIR / "multires_latency_benchmark.md"

NUM_CLASSES = 20
WIDTH_MULT = 0.5
REG_MAX = 16
IN_CH = 1
BACKBONE_MODE = "TFSep_pyramid"
OUTFUSION_CHANNELS_MULT = 1

POSTPROCESS_CONF = 0.05
POSTPROCESS_IOU = 0.1
POSTPROCESS_SAME_BOX_IOU = 0.9


def _resolution_label(resolutions: Sequence[Tuple[int, int]]) -> str:
    return ", ".join(f"{h}x{w}" for h, w in resolutions)


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


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _build_inputs(
    resolutions: Sequence[Tuple[int, int]],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> List[torch.Tensor]:
    inputs = []
    for h, w in resolutions:
        inputs.append(torch.randn(batch_size, IN_CH, h, w, device=device, dtype=dtype))
    return inputs


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


def _time_callable(
    fn,
    device: torch.device,
    warmup: int,
    iters: int,
) -> List[float]:
    samples_ms: List[float] = []
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        _sync(device)

        for _ in range(iters):
            _sync(device)
            start = time.perf_counter()
            fn()
            _sync(device)
            end = time.perf_counter()
            samples_ms.append((end - start) * 1000.0)
    return samples_ms


def _summarize(samples_ms: Sequence[float]) -> Dict[str, float]:
    return {
        "mean_ms": float(statistics.fmean(samples_ms)),
        "std_ms": float(statistics.pstdev(samples_ms)) if len(samples_ms) > 1 else 0.0,
        "min_ms": float(min(samples_ms)),
        "max_ms": float(max(samples_ms)),
    }


def _benchmark_one(
    label: str,
    resolutions: Sequence[Tuple[int, int]],
    device: torch.device,
    batch_size: int,
    warmup: int,
    iters: int,
) -> Dict[str, object]:
    run_dir = SCRIPT_DIR / "_latency_benchmark_artifacts" / label
    run_dir.mkdir(parents=True, exist_ok=True)

    model = _build_model(resolutions, device=device, output_dir=run_dir)
    inputs = _build_inputs(resolutions, batch_size=batch_size, device=device)

    with torch.inference_mode():
        dist_out, cls_out = model(inputs)
        model.postprocess(
            dist_out,
            cls_out,
            dist_out,
            conf_thres=POSTPROCESS_CONF,
            iou_thres=POSTPROCESS_IOU,
            iou_same_box=POSTPROCESS_SAME_BOX_IOU,
        )
    _sync(device)

    forward_samples = _time_callable(lambda: model(inputs), device=device, warmup=warmup, iters=iters)

    def _forward_postprocess():
        dist_out, cls_out = model(inputs)
        model.postprocess(
            dist_out,
            cls_out,
            dist_out,
            conf_thres=POSTPROCESS_CONF,
            iou_thres=POSTPROCESS_IOU,
            iou_same_box=POSTPROCESS_SAME_BOX_IOU,
        )

    end_to_end_samples = _time_callable(_forward_postprocess, device=device, warmup=warmup, iters=iters)

    num_input_pixels = int(sum(h * w for h, w in resolutions))
    result = {
        "label": label,
        "num_resolutions": len(resolutions),
        "resolutions": [list(hw) for hw in resolutions],
        "resolution_label": _resolution_label(resolutions),
        "total_input_pixels": num_input_pixels,
        "forward_ms": _summarize(forward_samples),
        "forward_plus_postprocess_ms": _summarize(end_to_end_samples),
    }

    del model
    del inputs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _relative(value: float, reference: float) -> float:
    if reference <= 0:
        return float("nan")
    return value / reference


def _build_markdown(results: Sequence[Dict[str, object]], metadata: Dict[str, object]) -> str:
    rows = []
    rows.append("| Resolutions | Temps de traitement (ms) |")
    rows.append("| --- | ---: |")

    for item in results:
        e2e = item["forward_plus_postprocess_ms"]
        rows.append(
            "| {resolution_label} | {e_mean:.3f} +- {e_std:.3f} |".format(
                resolution_label=item["resolution_label"],
                e_mean=float(e2e["mean_ms"]),
                e_std=float(e2e["std_ms"]),
            )
        )

    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MR_YOLO latency as a function of branch count.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, ...")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    device = _resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    configs = _benchmark_configs()
    ordered_labels = ["mr_1", "mr_2", "mr_3", "mr_5", "mr_7", "mr_2_far_gap_2_pow_5"]

    results: List[Dict[str, object]] = []
    for label in ordered_labels:
        results.append(
            _benchmark_one(
                label=label,
                resolutions=configs[label],
                device=device,
                batch_size=args.batch_size,
                warmup=args.warmup,
                iters=args.iters,
            )
        )

    metadata = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "iters": args.iters,
        "torch_version": torch.__version__,
        "num_threads": torch.get_num_threads(),
    }
    payload = {"metadata": metadata, "results": results}
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    markdown = _build_markdown(results, metadata)
    OUTPUT_MD.write_text(markdown, encoding="utf-8")
    print(markdown)
    print("")
    print(f"JSON saved to: {OUTPUT_JSON}")
    print(f"Markdown table saved to: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
