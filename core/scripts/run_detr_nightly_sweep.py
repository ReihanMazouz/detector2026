from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = PROJECT_ROOT / "core" / "scripts" / "train_detr_benchmark.py"


@dataclass(frozen=True)
class SweepJob:
    name: str
    eos_coef: float
    hidden_dim: int
    num_queries: int
    encoder_layers: int
    decoder_layers: int
    dim_feedforward: int
    aux_loss_weight: float = 1.0
    width_mult: float = 0.50


DEFAULT_JOBS = (
    SweepJob("baseline", eos_coef=0.10, hidden_dim=256, num_queries=100, encoder_layers=2, decoder_layers=3, dim_feedforward=1024),
    SweepJob("eos_005", eos_coef=0.05, hidden_dim=256, num_queries=100, encoder_layers=2, decoder_layers=3, dim_feedforward=1024),
    SweepJob("eos_020", eos_coef=0.20, hidden_dim=256, num_queries=100, encoder_layers=2, decoder_layers=3, dim_feedforward=1024),
    SweepJob("queries_150", eos_coef=0.10, hidden_dim=256, num_queries=150, encoder_layers=2, decoder_layers=3, dim_feedforward=1024),
    SweepJob("deeper_4x6", eos_coef=0.10, hidden_dim=256, num_queries=100, encoder_layers=4, decoder_layers=6, dim_feedforward=1024),
    SweepJob("deeper_queries150", eos_coef=0.10, hidden_dim=256, num_queries=150, encoder_layers=4, decoder_layers=6, dim_feedforward=1024),
)

SMOKE_JOBS = (
    SweepJob("smoke_baseline", eos_coef=0.10, hidden_dim=32, num_queries=4, encoder_layers=1, decoder_layers=1, dim_feedforward=64, width_mult=0.25),
    SweepJob("smoke_eos_020", eos_coef=0.20, hidden_dim=32, num_queries=4, encoder_layers=1, decoder_layers=1, dim_feedforward=64, width_mult=0.25),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run a sequential overnight DETR hyperparameter sweep.")
    parser.add_argument("--data-dir", default="/data/RAWSIM/RMA/rf_dataset_for_real_validation")
    parser.add_argument("--output-dir-parent", default="/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation/detr_nightly_sweep")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--res-key", default="cfg512")
    parser.add_argument("--preprocessing", default="none")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--full-eval-every", type=int, default=5)
    parser.add_argument("--save-last-every", type=int, default=5)
    parser.add_argument("--monitor", default="map50_95", choices=["val_loss", "map50", "map50_95"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--smoke-profile", action="store_true", help="Use tiny jobs for a quick end-to-end validation.")
    return parser.parse_args()


def _safe_float(value: Any) -> float | None:
    try:
        return None if value in ("", None) else float(value)
    except (TypeError, ValueError):
        return None


def _best_row(train_log_path: Path) -> dict[str, str] | None:
    if not train_log_path.is_file():
        return None
    with train_log_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if _safe_float(row.get("map50_95")) is not None]
    return max(rows, key=lambda row: float(row["map50_95"])) if rows else None


def _metrics_summary(run_dir: Path) -> dict[str, Any]:
    row = _best_row(run_dir / "train_log.csv")
    if row is None:
        return {}
    epoch = int(float(row["epoch"]))
    metrics_path = run_dir / "metrics" / f"metrics_epoch_{epoch:03d}.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    operating = payload.get("operating_point", {})
    return {
        "best_epoch": epoch,
        "map50": _safe_float(row.get("map50")),
        "map50_95": _safe_float(row.get("map50_95")),
        "val_loss": _safe_float(row.get("val_loss")),
        "conf_thresh": payload.get("conf_thresh"),
        "tp_raw": operating.get("tp_raw"),
        "fp_raw": operating.get("fp_raw"),
        "tp_at_conf_thresh": operating.get("tp_at_conf_thresh"),
        "fp_at_conf_thresh": operating.get("fp_at_conf_thresh"),
        "fn_total": operating.get("fn_total"),
    }


def _write_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "sweep_summary.json"
    csv_path = output_dir / "sweep_summary.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _command(args: argparse.Namespace, job: SweepJob, run_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--data-dir",
        args.data_dir,
        "--output-dir-parent",
        str(run_dir.parent),
        "--output-dir-name",
        run_dir.name,
        "--device",
        args.device,
        "--num-classes",
        str(args.num_classes),
        "--res-key",
        args.res_key,
        "--preprocessing",
        args.preprocessing,
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--full-eval-every",
        str(args.full_eval_every),
        "--save-last-every",
        str(args.save_last_every),
        "--monitor",
        args.monitor,
        "--width-mult",
        str(job.width_mult),
        "--hidden-dim",
        str(job.hidden_dim),
        "--num-queries",
        str(job.num_queries),
        "--encoder-layers",
        str(job.encoder_layers),
        "--decoder-layers",
        str(job.decoder_layers),
        "--dim-feedforward",
        str(job.dim_feedforward),
        "--aux-loss-weight",
        str(job.aux_loss_weight),
        "--eos-coef",
        str(job.eos_coef),
    ]
    if args.num_workers is not None:
        cmd.extend(["--num-workers", str(args.num_workers)])
    return cmd


def main():
    args = parse_args()
    output_root = Path(args.output_dir_parent)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []

    jobs = SMOKE_JOBS if args.smoke_profile else DEFAULT_JOBS
    for job in jobs:
        run_dir = output_root / job.name
        command = _command(args, job, run_dir)
        row = {
            "name": job.name,
            "run_dir": str(run_dir),
            "eos_coef": job.eos_coef,
            "hidden_dim": job.hidden_dim,
            "num_queries": job.num_queries,
            "encoder_layers": job.encoder_layers,
            "decoder_layers": job.decoder_layers,
            "dim_feedforward": job.dim_feedforward,
            "aux_loss_weight": job.aux_loss_weight,
        }
        if run_dir.exists() and not args.rerun_existing:
            print(f"[SKIP] {job.name}: {run_dir} already exists")
            row["status"] = "skipped_existing"
        else:
            print(f"[RUN ] {job.name}")
            print("       " + " ".join(command))
            if args.dry_run:
                row["status"] = "dry_run"
            else:
                completed = subprocess.run(command, check=False)
                row["status"] = "ok" if completed.returncode == 0 else f"failed_{completed.returncode}"
        row.update(_metrics_summary(run_dir))
        rows.append(row)
        _write_summary(rows, output_root)

    print(f"Summary written to {output_root / 'sweep_summary.csv'}")


if __name__ == "__main__":
    main()
