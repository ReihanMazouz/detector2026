from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from thop import profile
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from detector2026.core.nn.blocks import DFL, SCSA
from detector2026.core.scripts.train_benchmark_suite import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR_PARENT,
    DEFAULT_REG_MAX,
    DEFAULT_RES_KEYS,
    build_jobs,
    find_input_resolutions,
)
from detector2026.core.utils.analysing_results import dataset_analysis_with_metrics
from detector2026.core.utils.dataset import YOLODatasetSpecificRes, load_class_index_to_name
from detector2026.core.utils.preprocess import preprocessing_num_channels


DEFAULT_DEVICE = "cuda:1"
DEFAULT_OUTPUT_DIR = Path(DEFAULT_OUTPUT_DIR_PARENT) / "benchmark_best_recomputed_metrics"


def _zero_ops(module, inputs, outputs):
    module.total_ops += torch.DoubleTensor([0.0])


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_loader(job, data_dir: Path, preprocessing: str, batch_size: int, num_workers: int):
    if job.dataset == "specificres":
        if job.select_res is None:
            raise ValueError(f"{job.output_dir_name}: missing select_res for specificres dataset.")
        dataset = YOLODatasetSpecificRes(
            data_dir=str(data_dir / "val" / "data"),
            labels_dir=str(data_dir / "val" / "labels_detect"),
            res_hw=tuple(job.select_res["res_hw"]),
            res_key=str(job.select_res["res_key"]),
            preprocessing=preprocessing,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=dataset.collate_fn,
        ), tuple(job.select_res["res_hw"])

    dataset = job.dataset(
        data_dir=str(data_dir / "val" / "data"),
        labels_dir=str(data_dir / "val" / "labels_detect"),
        preprocessing=preprocessing,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=dataset.collate_fn,
    ), tuple(job.select_res["res_hw"]) if job.select_res else None


def load_best_weights(model: torch.nn.Module, checkpoint: Path, device: str) -> None:
    if hasattr(model, "load_weights"):
        model.load_weights(str(checkpoint), device=device, eval_mode=True)
    else:
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()


def profile_model(model: torch.nn.Module, loader: DataLoader) -> dict[str, Any]:
    sample = next(iter(loader))
    imgs = sample[0]
    if isinstance(imgs, list):
        dummy_input = [img[:1].to(model.device, dtype=torch.float32) for img in imgs]
    else:
        dummy_input = imgs[:1].to(model.device, dtype=torch.float32)

    params = sum(param.numel() for param in model.parameters())
    try:
        macs, _ = profile(
            model,
            inputs=(dummy_input,),
            custom_ops={DFL: _zero_ops, SCSA: _zero_ops},
            verbose=False,
        )
        macs = int(macs)
        return {"params": int(params), "macs": macs, "flops": int(2 * macs), "profile_error": ""}
    except Exception as exc:
        return {"params": int(params), "macs": "", "flops": "", "profile_error": str(exc)}


def _metric_value(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get("map_stats", {}).get(key)
    return float(value) if value is not None else float("nan")


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


def plot_metric(rows: list[dict[str, Any]], metric: str, x_key: str, output_path: Path) -> None:
    valid = [
        row
        for row in rows
        if row.get(metric) not in ("", None)
        and row.get(x_key) not in ("", None)
        and np.isfinite(float(row[metric]))
        and np.isfinite(float(row[x_key]))
    ]
    if not valid:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for row in valid:
        x = float(row[x_key])
        y = float(row[metric])
        ax.scatter(x, y, s=34)
        ax.annotate(row["model"], (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel(x_key)
    ax.set_ylabel(metric)
    ax.grid(True, linestyle="--", alpha=0.35)
    if x_key in {"params", "flops"}:
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute benchmark best.pt mAP on the validation dataset.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir-parent", default=DEFAULT_OUTPUT_DIR_PARENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--reg-max", type=int, default=DEFAULT_REG_MAX)
    parser.add_argument("--preprocessing", default="none")
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--fa-target", type=float, default=0.01)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir_parent = Path(args.output_dir_parent)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_resolutions = find_input_resolutions(str(data_dir), split="val")
    res_keys = list(DEFAULT_RES_KEYS)
    input_channels = preprocessing_num_channels(args.preprocessing)
    central_res_key = res_keys[0]
    central_res_hw = input_resolutions[0]

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

    print(f"Dataset validation : {data_dir}")
    print(f"Device             : {args.device}")
    print(f"Output             : {args.output_dir}")
    print(f"Jobs               : {len(jobs)}")

    rows: list[dict[str, Any]] = []
    class_names = load_class_index_to_name(data_dir)

    for index, job in enumerate(jobs, start=1):
        run_dir = output_dir_parent / job.output_dir_name
        checkpoint = run_dir / "best.pt"
        print(f"\n[{index:02d}/{len(jobs):02d}] {job.label}")
        print(f"  checkpoint = {checkpoint}")

        row: dict[str, Any] = {
            "model": job.output_dir_name,
            "label": job.label,
            "checkpoint": str(checkpoint),
            "status": "missing" if not checkpoint.is_file() else "ok",
            "map50": "",
            "map50_95": "",
            "params": "",
            "macs": "",
            "flops": "",
            "metrics_json": "",
            "profile_error": "",
        }
        if not checkpoint.is_file():
            rows.append(row)
            print("  -> missing best.pt, skipped")
            continue
        if args.dry_run:
            rows.append(row)
            continue

        model = job.model_builder(str(run_dir))
        try:
            load_best_weights(model, checkpoint, args.device)
            loader, img_size = make_loader(job, data_dir, args.preprocessing, args.batch_size, args.num_workers)
            if img_size is None:
                if hasattr(model, "input_resolutions"):
                    img_size = tuple(max(values) for values in zip(*model.input_resolutions))
                else:
                    img_size = input_resolutions[0]

            metrics = dataset_analysis_with_metrics(
                model=model,
                val_loader=loader,
                iou_thresh=args.iou_thresh,
                fa=args.fa_target,
                img_size=img_size,
                to_save=False,
                to_plot=False,
                class_index_to_name=class_names,
            )
            profile_stats = profile_model(model, loader)
            metrics_json = args.output_dir / f"{job.output_dir_name}_metrics.json"
            with metrics_json.open("w") as handle:
                import json

                json.dump(metrics, handle, indent=2, default=_json_default)

            row.update(
                {
                    "map50": _metric_value(metrics, "mAP50"),
                    "map50_95": _metric_value(metrics, "mAP50:95"),
                    "metrics_json": str(metrics_json),
                    **profile_stats,
                }
            )
            print(f"  -> mAP50={row['map50']:.6f} | mAP50:95={row['map50_95']:.6f}")
        except Exception as exc:
            row["status"] = "error"
            row["profile_error"] = str(exc)
            print(f"  -> ERROR: {exc}")
        finally:
            del model
            cleanup()

        rows.append(row)

    csv_path = args.output_dir / "benchmark_best_recomputed_map.csv"
    fieldnames = [
        "model",
        "label",
        "status",
        "map50",
        "map50_95",
        "params",
        "macs",
        "flops",
        "checkpoint",
        "metrics_json",
        "profile_error",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    plot_dir = args.output_dir / "plots"
    for metric in ("map50", "map50_95"):
        for x_key in ("params", "flops"):
            plot_metric(rows, metric, x_key, plot_dir / f"{metric}_vs_{x_key}.png")

    print(f"\nCSV   : {csv_path}")
    print(f"Plots : {plot_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
