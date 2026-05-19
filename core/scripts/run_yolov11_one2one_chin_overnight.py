from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_DATA_DIR = "/data/RAWSIM/RMA/rf_dataset_for_real_validation"
DEFAULT_SOURCE_RUN_DIR = "/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation/yolov11n_specificres_cfg512"
DEFAULT_WEIGHTS = f"{DEFAULT_SOURCE_RUN_DIR}/best.pt"
DEFAULT_OUTPUT_DIR = "/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation/one2one_head_chin_overnight"
DEFAULT_GPUS = ("cuda:0",)
DEFAULT_LOSSES = ("tal", "hungarian")


@dataclass(frozen=True)
class ChinConfig:
    label: str
    use_chin: bool
    levels: tuple[str, ...] = ("p4", "p5")
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 1
    ffn_ratio: float = 2.0
    dropout: float = 0.0
    residual_scale: float = 0.0


@dataclass(frozen=True)
class Job:
    name: str
    loss: str
    chin: ChinConfig
    output_dir: Path
    command: list[str]
    log_path: Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Launch an overnight YOLOv11 one2one benchmark on the H100 by default, using a P3/P4/P5 transformer chin, "
            "then generate a CSV and plots comparing the runs."
        )
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--source-run-dir", default=DEFAULT_SOURCE_RUN_DIR)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gpus", nargs="+", default=list(DEFAULT_GPUS), help="Devices used as workers. Default is cuda:0 only.")
    parser.add_argument("--losses", nargs="+", choices=("tal", "hungarian"), default=list(DEFAULT_LOSSES))
    parser.add_argument("--scale", default="n")
    parser.add_argument("--res-key", default="cfg512")
    parser.add_argument("--preprocessing", default="none")
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--reg-max", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--full-eval-every", type=int, default=1)
    parser.add_argument("--save-last-every", type=int, default=5)
    parser.add_argument("--monitor", default="map50_95")
    parser.add_argument("--minimum-possible-candidates", type=int, default=7)
    parser.add_argument("--negative-to-positive-ratio", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--chin-d-model", type=int, default=128)
    parser.add_argument("--chin-num-heads", type=int, default=4)
    parser.add_argument("--chin-num-layers", type=int, default=1)
    parser.add_argument("--chin-ffn-ratio", type=float, default=2.0)
    parser.add_argument("--chin-dropout", type=float, default=0.0)
    parser.add_argument("--chin-residual-scale", type=float, default=0.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def source_name_from_args(args) -> str:
    if args.source_run_dir:
        return Path(args.source_run_dir).name
    return Path(args.weights).stem


def chin_suffix(chin: ChinConfig) -> str:
    if not chin.use_chin:
        return ""
    levels = "".join(chin.levels)
    return f"_chin_{levels}_d{chin.d_model}_l{chin.num_layers}"


def output_name(source_name: str, loss: str, chin: ChinConfig) -> str:
    return f"{source_name}_one2one_{loss}{chin_suffix(chin)}"


def build_chin_configs(args) -> list[ChinConfig]:
    return [
        ChinConfig(
            label=f"chin_p3p4p5_d{args.chin_d_model}_l{args.chin_num_layers}",
            use_chin=True,
            levels=("p3", "p4", "p5"),
            d_model=args.chin_d_model,
            num_heads=args.chin_num_heads,
            num_layers=args.chin_num_layers,
            ffn_ratio=args.chin_ffn_ratio,
            dropout=args.chin_dropout,
            residual_scale=args.chin_residual_scale,
        )
    ]


def add_common_train_args(command: list[str], args, device: str, loss: str):
    command.extend(
        [
            "--data-dir", args.data_dir,
            "--source-run-dir", args.source_run_dir,
            "--weights", args.weights,
            "--output-dir", args.output_dir,
            "--device", device,
            "--losses", loss,
            "--scale", args.scale,
            "--res-key", args.res_key,
            "--preprocessing", args.preprocessing,
            "--num-classes", str(args.num_classes),
            "--reg-max", str(args.reg_max),
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--num-workers", str(args.num_workers),
            "--full-eval-every", str(args.full_eval_every),
            "--save-last-every", str(args.save_last_every),
            "--monitor", args.monitor,
            "--minimum-possible-candidates", str(args.minimum_possible_candidates),
            "--negative-to-positive-ratio", str(args.negative_to_positive_ratio),
        ]
    )
    if args.overwrite:
        command.append("--overwrite")


def add_chin_args(command: list[str], chin: ChinConfig):
    if not chin.use_chin:
        return
    command.extend(
        [
            "--use-transformer-chin",
            "--chin-levels", ",".join(chin.levels),
            "--chin-d-model", str(chin.d_model),
            "--chin-num-heads", str(chin.num_heads),
            "--chin-num-layers", str(chin.num_layers),
            "--chin-ffn-ratio", str(chin.ffn_ratio),
            "--chin-dropout", str(chin.dropout),
            "--chin-residual-scale", str(chin.residual_scale),
        ]
    )


def build_jobs(args) -> list[Job]:
    root = Path(args.output_dir)
    log_dir = root / "launcher_logs"
    train_script = Path(__file__).with_name("train_yolov11_one2one_head.py")
    source_name = source_name_from_args(args)
    jobs = []
    for chin in build_chin_configs(args):
        for loss in tuple(dict.fromkeys(args.losses)):
            run_dir = root / output_name(source_name, loss, chin)
            name = f"{loss}_{chin.label}"
            command = [sys.executable, str(train_script)]
            add_common_train_args(command, args, device="{device}", loss=loss)
            add_chin_args(command, chin)
            jobs.append(Job(name=name, loss=loss, chin=chin, output_dir=run_dir, command=command, log_path=log_dir / f"{name}.log"))
    return jobs


def command_for_device(job: Job, device: str) -> list[str]:
    return [device if part == "{device}" else part for part in job.command]


def should_skip(job: Job, overwrite: bool) -> bool:
    return job.output_dir.exists() and not overwrite


def run_jobs(jobs: list[Job], args) -> dict[str, str]:
    devices = list(args.gpus)

    if args.dry_run:
        print("Planned jobs:")
        for index, job in enumerate(jobs):
            device = devices[index % len(devices)]
            status = "SKIP existing" if should_skip(job, args.overwrite) else "RUN"
            print(f"[{status}] {job.name} on {device}")
            print("  " + " ".join(command_for_device(job, device)))
            print(f"  output_dir = {job.output_dir}")
            print(f"  log = {job.log_path}")
        return {job.name: "dry_run" for job in jobs}

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for job in jobs:
        job.log_path.parent.mkdir(parents=True, exist_ok=True)

    pending = [job for job in jobs if not should_skip(job, args.overwrite)]
    statuses = {job.name: "skipped_exists" for job in jobs if should_skip(job, args.overwrite)}
    running: dict[str, tuple[subprocess.Popen, Job, object]] = {}

    print(f"Launching {len(pending)} jobs on devices: {devices}")
    while pending or running:
        for device in devices:
            if device in running or not pending:
                continue
            job = pending.pop(0)
            log_handle = open(job.log_path, "w")
            command = command_for_device(job, device)
            print(f"[START] {job.name} on {device}")
            print(f"        log = {job.log_path}")
            process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT)
            running[device] = (process, job, log_handle)
            statuses[job.name] = "running"

        time.sleep(max(float(args.poll_seconds), 1.0))

        for device, (process, job, log_handle) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            log_handle.close()
            if return_code == 0:
                statuses[job.name] = "completed"
                print(f"[DONE ] {job.name} on {device}")
            else:
                statuses[job.name] = f"failed_{return_code}"
                print(f"[FAIL ] {job.name} on {device} return_code={return_code} log={job.log_path}")
            del running[device]

    return statuses


def read_log_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def best_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    valid = [row for row in rows if not math.isnan(as_float(row.get("map50_95")))]
    if valid:
        return max(valid, key=lambda row: as_float(row.get("map50_95")))
    return rows[-1] if rows else None


def summarize_jobs(jobs: Iterable[Job], statuses: dict[str, str], output_root: Path):
    rows = []
    for job in jobs:
        train_log = job.output_dir / "train_log.csv"
        log_rows = read_log_rows(train_log)
        chosen = best_row(log_rows)
        last = log_rows[-1] if log_rows else {}
        rows.append(
            {
                "job": job.name,
                "status": statuses.get(job.name, "unknown"),
                "loss": job.loss,
                "chin": job.chin.label,
                "use_chin": str(job.chin.use_chin),
                "chin_levels": ",".join(job.chin.levels) if job.chin.use_chin else "",
                "chin_d_model": job.chin.d_model if job.chin.use_chin else "",
                "chin_num_layers": job.chin.num_layers if job.chin.use_chin else "",
                "output_dir": str(job.output_dir),
                "train_log": str(train_log),
                "epochs_logged": len(log_rows),
                "best_epoch": chosen.get("epoch", "") if chosen else "",
                "best_map50": chosen.get("map50", "") if chosen else "",
                "best_map50_95": chosen.get("map50_95", "") if chosen else "",
                "best_avg_recall_low_snr": chosen.get("avg_recall_low_snr", "") if chosen else "",
                "best_avg_recall_medium_snr": chosen.get("avg_recall_medium_snr", "") if chosen else "",
                "best_avg_recall_high_snr": chosen.get("avg_recall_high_snr", "") if chosen else "",
                "last_epoch": last.get("epoch", ""),
                "last_train_loss": last.get("train_loss", ""),
                "last_val_loss": last.get("val_loss", ""),
                "last_map50": last.get("map50", ""),
                "last_map50_95": last.get("map50_95", ""),
            }
        )

    summary_path = output_root / "overnight_comparison_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SUMMARY] saved {summary_path}")
    return rows, summary_path


def plot_summary(rows: list[dict[str, str]], output_root: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable; comparison plots skipped: {exc}")
        return

    plot_dir = output_root / "overnight_comparison_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    labels = [row["job"] for row in rows]

    for metric in ("best_map50", "best_map50_95", "best_avg_recall_low_snr", "best_avg_recall_medium_snr", "best_avg_recall_high_snr"):
        values = [as_float(row.get(metric)) for row in rows]
        if all(math.isnan(value) for value in values):
            continue
        plt.figure(figsize=(max(12, len(labels) * 1.4), 6))
        plt.bar(labels, values)
        plt.ylabel(metric)
        plt.xticks(rotation=35, ha="right")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        path = plot_dir / f"{metric}.png"
        plt.savefig(path)
        plt.close()
        print(f"[PLOT] saved {path}")

    for metric in ("train_loss", "val_loss", "map50", "map50_95"):
        plt.figure(figsize=(10, 6))
        has_data = False
        for row in rows:
            log_rows = read_log_rows(Path(row["train_log"]))
            epochs = [as_float(log_row.get("epoch")) for log_row in log_rows]
            values = [as_float(log_row.get(metric)) for log_row in log_rows]
            if not epochs or all(math.isnan(value) for value in values):
                continue
            plt.plot(epochs, values, label=row["job"])
            has_data = True
        if not has_data:
            plt.close()
            continue
        plt.xlabel("epoch")
        plt.ylabel(metric)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        path = plot_dir / f"{metric}_curves.png"
        plt.savefig(path)
        plt.close()
        print(f"[PLOT] saved {path}")


def main():
    args = parse_args()
    jobs = build_jobs(args)
    print("YOLOv11 one2one overnight benchmark")
    print(f"  output_dir = {args.output_dir}")
    print(f"  weights = {args.weights}")
    print(f"  gpus = {args.gpus}")
    print(f"  losses = {args.losses}")
    print(f"  negative_to_positive_ratio = {args.negative_to_positive_ratio}")
    print(f"  jobs = {len(jobs)}")

    statuses = run_jobs(jobs, args)
    if args.dry_run:
        return

    rows, _ = summarize_jobs(jobs, statuses, Path(args.output_dir))
    plot_summary(rows, Path(args.output_dir))

    failures = {name: status for name, status in statuses.items() if status.startswith("failed")}
    if failures:
        print("[WARN] Some jobs failed:")
        for name, status in failures.items():
            print(f"  {name}: {status}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
