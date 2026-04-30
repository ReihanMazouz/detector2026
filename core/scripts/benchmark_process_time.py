from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from detector2026.core.scripts.train_benchmark_suite import (
    DEFAULT_NUM_CLASSES,
    DEFAULT_PREPROCESSING,
    DEFAULT_REG_MAX,
    DEFAULT_RES_KEYS,
    TrainingJob,
    build_jobs,
)
from detector2026.core.utils.preprocess import preprocessing_num_channels


DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_BATCHES = 100
DEFAULT_WARMUP_BATCHES = 5
DEFAULT_OUTPUT_CSV = "benchmark_process_time.csv"
DEFAULT_INPUT_RESOLUTIONS = [
    (256, 256),  # cfg512
    (128, 512),  # cfg256
    (64, 1024),  # cfg128
    (512, 128),  # cfg1024
    (1024, 64),  # cfg2048
]


def available_devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.extend(f"cuda:{index}" for index in range(torch.cuda.device_count()))
    return devices


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def make_inputs(
    model: torch.nn.Module,
    job: TrainingJob,
    batch_size: int,
    input_channels: int,
    device: str,
) -> torch.Tensor | list[torch.Tensor]:
    if hasattr(model, "input_resolutions"):
        return [
            torch.randn(batch_size, input_channels, height, width, device=device)
            for height, width in model.input_resolutions
        ]

    if not job.select_res or "res_hw" not in job.select_res:
        raise ValueError(f"Cannot infer input resolution for job {job.label!r}.")

    height, width = job.select_res["res_hw"]
    return torch.randn(batch_size, input_channels, height, width, device=device)


def detach_output(output: Any) -> None:
    if torch.is_tensor(output):
        return
    if isinstance(output, (list, tuple)):
        for item in output:
            detach_output(item)
        return
    if isinstance(output, dict):
        for item in output.values():
            detach_output(item)


def benchmark_job(
    job: TrainingJob,
    device: str,
    batch_size: int,
    num_batches: int,
    warmup_batches: int,
    input_channels: int,
    output_dir_root: Path,
) -> dict[str, Any]:
    output_dir = output_dir_root / device.replace(":", "_") / job.output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    model = job.model_builder(str(output_dir))
    model.eval()
    inputs = make_inputs(model, job, batch_size, input_channels, device)

    timings_per_data: list[float] = []
    try:
        with torch.no_grad():
            for _ in range(warmup_batches):
                detach_output(model(inputs))
            synchronize(device)

            for _ in range(num_batches):
                synchronize(device)
                start = time.perf_counter()
                detach_output(model(inputs))
                synchronize(device)
                elapsed = time.perf_counter() - start
                timings_per_data.append(elapsed / batch_size)
    finally:
        del inputs
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    mean_s = statistics.fmean(timings_per_data)
    variance_s2 = statistics.pvariance(timings_per_data) if len(timings_per_data) > 1 else 0.0
    return {
        "device": device,
        "label": job.label,
        "output_dir_name": job.output_dir_name,
        "batch_size": batch_size,
        "num_batches": num_batches,
        "warmup_batches": warmup_batches,
        "mean_s_per_data": mean_s,
        "variance_s2_per_data": variance_s2,
        "mean_ms_per_data": mean_s * 1_000.0,
        "variance_ms2_per_data": variance_s2 * 1_000_000.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark average forward processing time for every model in train_benchmark_suite.py."
    )
    parser.add_argument("--num-classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--reg-max", type=int, default=DEFAULT_REG_MAX)
    parser.add_argument("--preprocessing", default=DEFAULT_PREPROCESSING)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-batches", type=int, default=DEFAULT_NUM_BATCHES)
    parser.add_argument("--warmup-batches", type=int, default=DEFAULT_WARMUP_BATCHES)
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help="Devices to benchmark. Default: cpu and every available cuda device.",
    )
    parser.add_argument(
        "--output-csv",
        default=DEFAULT_OUTPUT_CSV,
        help="CSV path where benchmark results are written.",
    )
    parser.add_argument(
        "--tmp-output-dir",
        default="/private/tmp/detector2026_process_time_models",
        help="Temporary output_dir root passed to model constructors.",
    )
    return parser.parse_args()


def build_benchmark_jobs(
    input_resolutions: Sequence[tuple[int, int]],
    input_channels: int,
    device: str,
    num_classes: int,
    reg_max: int,
) -> list[TrainingJob]:
    res_keys = list(DEFAULT_RES_KEYS)
    central_index = 0
    return build_jobs(
        input_resolutions=input_resolutions,
        res_keys=res_keys,
        input_channels=input_channels,
        device=device,
        num_classes=num_classes,
        reg_max=reg_max,
        central_res_key=res_keys[central_index],
        central_res_hw=input_resolutions[central_index],
    )


def main() -> None:
    args = parse_args()
    devices = args.devices or available_devices()
    input_resolutions = list(DEFAULT_INPUT_RESOLUTIONS)
    input_channels = preprocessing_num_channels(args.preprocessing)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_dir = Path(args.tmp_output_dir)

    rows: list[dict[str, Any]] = []
    for device in devices:
        if device.startswith("cuda") and not torch.cuda.is_available():
            print(f"[SKIP] {device}: CUDA indisponible.")
            continue
        if device.startswith("cuda"):
            index = int(device.split(":", 1)[1])
            if index >= torch.cuda.device_count():
                print(f"[SKIP] {device}: GPU inexistant.")
                continue

        jobs = build_benchmark_jobs(
            input_resolutions=input_resolutions,
            input_channels=input_channels,
            device=device,
            num_classes=args.num_classes,
            reg_max=args.reg_max,
        )

        for index, job in enumerate(jobs, start=1):
            print(f"[RUN ] {device} {index:02d}/{len(jobs)} {job.output_dir_name}")
            row = benchmark_job(
                job=job,
                device=device,
                batch_size=args.batch_size,
                num_batches=args.num_batches,
                warmup_batches=args.warmup_batches,
                input_channels=input_channels,
                output_dir_root=tmp_output_dir,
            )
            rows.append(row)
            print(
                f"       mean={row['mean_ms_per_data']:.6f} ms/data "
                f"var={row['variance_ms2_per_data']:.6f} ms^2/data"
            )

    fieldnames = [
        "device",
        "label",
        "output_dir_name",
        "batch_size",
        "num_batches",
        "warmup_batches",
        "mean_s_per_data",
        "variance_s2_per_data",
        "mean_ms_per_data",
        "variance_ms2_per_data",
    ]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResultats ecrits dans {output_csv}")


if __name__ == "__main__":
    main()
