from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PARENT = REPO_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from detector2026.core.nn.blocks import Attention, DFL, PCSA, SCSA
from detector2026.core.scripts.train_benchmark_suite import (
    DEFAULT_DATA_DIR,
    DEFAULT_NUM_CLASSES,
    DEFAULT_OUTPUT_DIR_PARENT,
    DEFAULT_PREPROCESSING,
    DEFAULT_REG_MAX,
    DEFAULT_RES_KEYS,
    build_jobs,
    find_input_resolutions,
)
from detector2026.core.utils.analysing_results import dataset_analysis_with_metrics
from detector2026.core.utils.dataset import YOLODatasetSpecificRes, load_class_index_to_name
from detector2026.core.utils.preprocess import preprocessing_num_channels

try:
    from thop import profile
except ModuleNotFoundError as exc:
    raise SystemExit("thop is required. Activate the detector2026 environment first.") from exc


DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "benchmark_exports" / "rf_dataset_for_real_validation"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _zero_ops(module: torch.nn.Module, inputs: tuple[Any, ...], outputs: Any) -> None:
    module.total_ops += torch.DoubleTensor([0.0])


def _attention_extra_macs(module: Attention, x: torch.Tensor) -> int:
    batch_size, _, height, width = x.shape
    num_tokens = height * width
    qk = batch_size * module.num_heads * num_tokens * num_tokens * module.key_dim
    av = batch_size * module.num_heads * num_tokens * num_tokens * module.head_dim
    softmax_and_scale = 4 * batch_size * module.num_heads * num_tokens * num_tokens
    return int(qk + av + softmax_and_scale)


def _pcsa_extra_macs(module: PCSA, x: torch.Tensor) -> int:
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


def _missing_attention_macs(model: torch.nn.Module, inputs: tuple[Any, ...]) -> int:
    extra_macs = 0
    hooks = []

    def hook_fn(module: torch.nn.Module, module_inputs: tuple[Any, ...], outputs: Any) -> None:
        nonlocal extra_macs
        x = module_inputs[0]
        if isinstance(module, Attention):
            extra_macs += _attention_extra_macs(module, x)
        elif isinstance(module, PCSA):
            extra_macs += _pcsa_extra_macs(module, x)

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


def _to_device(imgs: Any, device: torch.device) -> Any:
    if isinstance(imgs, list):
        return [img.to(device, dtype=torch.float32, non_blocking=device.type == "cuda") for img in imgs]
    return imgs.to(device, dtype=torch.float32, non_blocking=device.type == "cuda")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _mean_latency_seconds(
    model: torch.nn.Module,
    batch: Any,
    device: torch.device,
    warmup: int,
    iters: int,
) -> float:
    model.eval()
    batch = _to_device(batch, device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(batch)
        _sync(device)
        start = time.perf_counter()
        for _ in range(iters):
            model(batch)
        _sync(device)
    return (time.perf_counter() - start) / max(iters, 1)


def _profile_model(model: torch.nn.Module, sample_imgs: Any, device: torch.device) -> dict[str, int]:
    if isinstance(sample_imgs, list):
        dummy = [torch.randn((1, img.shape[1], img.shape[2], img.shape[3]), device=device) for img in sample_imgs]
    else:
        dummy = torch.randn((1, sample_imgs.shape[1], sample_imgs.shape[2], sample_imgs.shape[3]), device=device)

    raw_macs, thop_params = profile(
        model,
        inputs=(dummy,),
        custom_ops={DFL: _zero_ops, SCSA: _zero_ops},
        verbose=False,
    )
    extra_macs = _missing_attention_macs(model, (dummy,))
    params = sum(param.numel() for param in model.parameters())
    macs = int(raw_macs) + int(extra_macs)
    return {
        "params": int(params),
        "thop_params": int(thop_params),
        "flops": int(2 * macs),
    }


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _load_or_compute_metrics(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    metrics_path: Path,
    reuse: bool,
    iou_thresh: float,
    fa_target: float,
    img_size: tuple[int, int],
    class_index_to_name: dict[int, str] | None,
) -> dict[str, Any]:
    if metrics_path.is_file():
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    if not reuse:
        return {}

    metrics = dataset_analysis_with_metrics(
        model=model,
        val_loader=loader,
        iou_thresh=iou_thresh,
        fa=fa_target,
        img_size=img_size,
        to_save=False,
        to_plot=False,
        class_index_to_name=class_index_to_name,
    )
    _save_json(metrics_path, metrics)
    return metrics


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _metric_path_from_row(run_dir: Path, row: dict[str, str]) -> Path | None:
    raw_path = str(row.get("metrics_json_path") or "").strip()
    if not raw_path or raw_path.lower() in {"none", "nan"}:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = run_dir / path
    return path if path.is_file() else None


def _existing_metrics_paths(run_dir: Path) -> list[Path]:
    paths: list[Path] = []
    rows = _read_csv_rows(run_dir / "train_log.csv")
    for row in rows:
        path = _metric_path_from_row(run_dir, row)
        if path is not None:
            paths.append(path)

    metrics_dir = run_dir / "metrics"
    if metrics_dir.is_dir():
        paths.extend(sorted(metrics_dir.glob("metrics_epoch_*.json")))

    unique_paths: list[Path] = []
    seen = set()
    for path in paths:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(path)
    return unique_paths


def _metric_value(metrics: dict[str, Any], key: str) -> float | None:
    map_stats = metrics.get("map_stats") or {}
    aliases = {
        "map50": "mAP50",
        "map50_95": "mAP50:95",
    }
    value = map_stats.get(aliases.get(key, key))
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_best_existing_metrics(
    run_dir: Path,
    selection: str,
) -> tuple[dict[str, Any], Path | None, dict[str, Any]]:
    records = []
    for path in _existing_metrics_paths(run_dir):
        try:
            metrics = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        records.append(
            {
                "path": path,
                "metrics": metrics,
                "map50": _metric_value(metrics, "map50"),
                "map50_95": _metric_value(metrics, "map50_95"),
            }
        )

    if not records:
        return {}, None, {}

    best_map50 = max((record for record in records if record["map50"] is not None), key=lambda r: r["map50"], default=None)
    best_map50_95 = max(
        (record for record in records if record["map50_95"] is not None),
        key=lambda r: r["map50_95"],
        default=None,
    )

    if selection == "best-map50":
        selected = best_map50 or best_map50_95 or records[-1]
    elif selection == "latest":
        selected = records[-1]
    else:
        selected = best_map50_95 or best_map50 or records[-1]

    summary = {
        "num_metrics_json": len(records),
        "best_map50": None if best_map50 is None else best_map50["map50"],
        "best_map50_metrics_json": "" if best_map50 is None else str(best_map50["path"]),
        "best_map50_95": None if best_map50_95 is None else best_map50_95["map50_95"],
        "best_map50_95_metrics_json": "" if best_map50_95 is None else str(best_map50_95["path"]),
        "selected_metric": selection,
    }
    return selected["metrics"], selected["path"], summary


def _metric_model_info(metrics: dict[str, Any]) -> dict[str, int | None]:
    info = metrics.get("model_info") or {}
    params = info.get("params")
    macs = info.get("macs")
    flops = info.get("flops")
    if macs is not None:
        flops = 2 * int(macs)
    elif flops is not None:
        # Historical training metrics store thop.profile's first return value
        # under "flops", but thop returns MACs. Export true FLOPs.
        flops = 2 * int(flops)
    return {
        "params": None if params is None else int(params),
        "flops": None if flops is None else int(flops),
    }


def _recall_snr_columns(metrics: dict[str, Any]) -> dict[str, float]:
    recall_snr = metrics.get("recall_snr", {}).get("global", {})
    bins = recall_snr.get("snr_bins") or recall_snr.get("bins") or []
    recall = recall_snr.get("recall") or []
    out: dict[str, float] = {}
    for idx, value in enumerate(recall):
        if idx + 1 < len(bins):
            name = f"recall_snr_{float(bins[idx]):g}_to_{float(bins[idx + 1]):g}"
        else:
            name = f"recall_snr_bin_{idx}"
        out[name] = float(value)
    return out


def _build_loader(job: Any, data_dir: Path, split: str, batch_size: int, preprocessing: str, num_workers: int) -> DataLoader:
    labels_dir = data_dir / split / "labels_detect"
    data_split_dir = data_dir / split / "data"
    if job.dataset == "specificres":
        dataset = YOLODatasetSpecificRes(
            data_dir=str(data_split_dir),
            labels_dir=str(labels_dir),
            res_hw=job.select_res["res_hw"],
            res_key=job.select_res["res_key"],
            preprocessing=preprocessing,
        )
    else:
        dataset = job.dataset(
            data_dir=str(data_split_dir),
            labels_dir=str(labels_dir),
            preprocessing=preprocessing,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=dataset.collate_fn,
    )


def _infer_img_size(job: Any, input_resolutions: list[tuple[int, int]]) -> tuple[int, int]:
    if job.dataset == "specificres":
        return tuple(job.select_res["res_hw"])
    return (max(h for h, _ in input_resolutions), max(w for _, w in input_resolutions))


def _resolution_label(job: Any) -> str:
    if job.dataset == "specificres":
        return str(job.select_res["res_key"])
    if "_fused_" in job.output_dir_name:
        return job.output_dir_name.split("_fused_", 1)[1].replace("_", ",")
    return ""


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    headers = [
        "model",
        "resolutions",
        "params_m",
        "flops_g",
        "map50",
        "map50_95",
        "latency_cpu_ms_per_sample",
        "latency_gpu_ms_per_sample",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export metrics, cost, and latency for train_benchmark_suite runs.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--weights-root", default=DEFAULT_OUTPUT_DIR_PARENT)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--split", default="val")
    parser.add_argument("--preprocessing", default=DEFAULT_PREPROCESSING)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--reg-max", type=int, default=DEFAULT_REG_MAX)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--latency-iters", type=int, default=50)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--fa-target", type=float, default=0.01)
    parser.add_argument(
        "--recompute-missing-metrics",
        action="store_true",
        help="Run full evaluation only when no saved metrics JSON exists in the training folder.",
    )
    parser.add_argument("--skip-metrics", action="store_true", help="Only compute params/FLOPs/latency.")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument(
        "--metrics-selection",
        choices=("best-map50-95", "best-map50", "latest"),
        default="best-map50-95",
        help="Metrics JSON used for curves such as recall vs SNR. Scalar mAP columns always use their best value.",
    )
    parser.add_argument(
        "--profile-missing-cost",
        action="store_true",
        help="Compute params/FLOPs with thop only when they are absent from saved metrics.",
    )
    parser.add_argument("--checkpoint-name", default="best.pt")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    weights_root = Path(args.weights_root)
    output_dir = Path(args.output_dir)

    input_resolutions = find_input_resolutions(str(data_dir), split="train")
    res_keys = list(DEFAULT_RES_KEYS)
    central_res_key = "cfg512"
    central_res_hw = input_resolutions[res_keys.index(central_res_key)]
    input_channels = preprocessing_num_channels(args.preprocessing)
    jobs = build_jobs(
        input_resolutions=input_resolutions,
        res_keys=res_keys,
        input_channels=input_channels,
        device=args.device,
        num_classes=args.num_classes,
        reg_max=args.reg_max,
        central_res_key=central_res_key,
        central_res_hw=central_res_hw,
    )

    class_index_to_name = load_class_index_to_name(data_dir)
    rows: list[dict[str, Any]] = []

    print(f"[info] data_dir={data_dir}", flush=True)
    print(f"[info] weights_root={weights_root}", flush=True)
    print(f"[info] output_dir={output_dir}", flush=True)
    print(f"[info] jobs={len(jobs)} | batch_size={args.batch_size}", flush=True)

    for job in tqdm(jobs, desc="Export benchmark", unit="model"):
        checkpoint = weights_root / job.output_dir_name / args.checkpoint_name
        if not checkpoint.is_file():
            tqdm.write(f"[skip] {job.label}: missing checkpoint {checkpoint}")
            continue

        tqdm.write(f"[model] {job.label}")
        run_dir = checkpoint.parent
        metrics: dict[str, Any] = {}
        metrics_path: Path | None = None
        if not args.skip_metrics:
            metrics, metrics_path, metrics_summary = _load_best_existing_metrics(run_dir, args.metrics_selection)
            if metrics_path is None:
                tqdm.write("  [metrics] no saved metrics found")
            else:
                tqdm.write(
                    "  [metrics] selected "
                    f"{metrics_path} ({args.metrics_selection}, scanned={metrics_summary.get('num_metrics_json')})"
                )
        else:
            metrics_summary = {}

        device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
        tqdm.write(f"  [load] checkpoint on {device}")
        model = job.model_builder(str(weights_root / job.output_dir_name))
        model.load_weights(str(checkpoint), device=device, eval_mode=True)
        model.to(device).eval()

        tqdm.write("  [data] building dataloader and reading one batch")
        loader = _build_loader(job, data_dir, args.split, args.batch_size, args.preprocessing, args.num_workers)
        sample_imgs = next(iter(loader))[0]
        metric_cost = _metric_model_info(metrics)
        profile_info: dict[str, int | None] = {
            "params": metric_cost["params"],
            "flops": metric_cost["flops"],
        }
        if args.profile_missing_cost and any(profile_info.get(key) is None for key in ("params", "flops")):
            tqdm.write("  [profile] computing missing params/FLOPs")
            profile_info.update(_profile_model(model, sample_imgs, device))

        if (not args.skip_metrics) and (not metrics) and args.recompute_missing_metrics:
            metrics_path = output_dir / "metrics" / f"{job.output_dir_name}.json"
            tqdm.write("  [metrics] recomputing missing full metrics")
            metrics = _load_or_compute_metrics(
                model=model,
                loader=loader,
                metrics_path=metrics_path,
                reuse=True,
                iou_thresh=args.iou_thresh,
                fa_target=args.fa_target,
                img_size=_infer_img_size(job, input_resolutions),
                class_index_to_name=class_index_to_name,
            )
            metric_cost = _metric_model_info(metrics)
            for key in ("params", "flops"):
                if profile_info.get(key) is None:
                    profile_info[key] = metric_cost[key]
            metrics_summary = {
                "num_metrics_json": 1,
                "best_map50": _metric_value(metrics, "map50"),
                "best_map50_metrics_json": str(metrics_path),
                "best_map50_95": _metric_value(metrics, "map50_95"),
                "best_map50_95_metrics_json": str(metrics_path),
                "selected_metric": "recomputed",
            }

        lat_cpu = lat_gpu = None
        if not args.skip_latency:
            tqdm.write("  [latency] CPU")
            cpu_model = job.model_builder(str(weights_root / job.output_dir_name))
            cpu_model.load_weights(str(checkpoint), device="cpu", eval_mode=True)
            cpu_model.to("cpu").eval()
            lat_cpu = _mean_latency_seconds(
                cpu_model, sample_imgs, torch.device("cpu"), args.warmup, args.latency_iters
            )
            if torch.cuda.is_available():
                tqdm.write("  [latency] GPU")
                lat_gpu = _mean_latency_seconds(model, sample_imgs, device, args.warmup, args.latency_iters)

        map_stats = metrics.get("map_stats", {})
        map50 = metrics_summary.get("best_map50")
        map50_95 = metrics_summary.get("best_map50_95")
        if map50 is None:
            map50 = map_stats.get("mAP50")
        if map50_95 is None:
            map50_95 = map_stats.get("mAP50:95")
        row = {
            "model": job.label,
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint),
            "output_dir_name": job.output_dir_name,
            "is_multires": int(job.dataset != "specificres"),
            "resolutions": _resolution_label(job),
            "params": profile_info["params"],
            "flops": profile_info["flops"],
            "params_m": None if profile_info["params"] is None else round(profile_info["params"] / 1e6, 4),
            "flops_g": None if profile_info["flops"] is None else round(profile_info["flops"] / 1e9, 4),
            "map50": map50,
            "map50_95": map50_95,
            "latency_cpu_s_per_batch": lat_cpu,
            "latency_gpu_s_per_batch": lat_gpu,
            "latency_cpu_ms_per_sample": None if lat_cpu is None else 1000.0 * lat_cpu / args.batch_size,
            "latency_gpu_ms_per_sample": None if lat_gpu is None else 1000.0 * lat_gpu / args.batch_size,
            "batch_size": args.batch_size,
            "metrics_json": "" if metrics_path is None else str(metrics_path),
            "num_metrics_json": metrics_summary.get("num_metrics_json"),
            "selected_metric": metrics_summary.get("selected_metric"),
            "best_map50_metrics_json": metrics_summary.get("best_map50_metrics_json"),
            "best_map50_95_metrics_json": metrics_summary.get("best_map50_95_metrics_json"),
        }
        row.update(_recall_snr_columns(metrics))
        rows.append(row)
        tqdm.write(f"  [done] {job.label}")

        del model
        if "cpu_model" in locals():
            del cpu_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_csv = output_dir / "benchmark_all_models.csv"
    mr_csv = output_dir / "benchmark_multires_table.csv"
    mr_md = output_dir / "benchmark_multires_table.md"
    _write_csv(all_csv, rows)
    mr_rows = [row for row in rows if row.get("is_multires") == 1]
    _write_csv(mr_csv, mr_rows)
    _write_markdown(mr_md, mr_rows)
    print(f"[saved] {all_csv}")
    print(f"[saved] {mr_csv}")
    print(f"[saved] {mr_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
