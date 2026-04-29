from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.classical_detectors.evaluation.benchmark import BenchmarkConfig, run_benchmark


SIMULATOR_ROOT = PROJECT_ROOT.parent / "ICML2026DataSimulator"
DEFAULT_DATASET_ROOT = SIMULATOR_ROOT / "tmp" / "output" / "rf_single_emitter_validation_smoketest_f32"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "classical_detectors" / "single_emitter_benchmark.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark classical detectors on the single-emitter RF dataset.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--pfa", type=float, default=1e-3)
    parser.add_argument("--noise-trials", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=444)
    parser.add_argument("--qmf-levels", type=int, default=6)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_benchmark(
        BenchmarkConfig(
            dataset_root=args.dataset_root.resolve(),
            split=args.split,
            pfa=args.pfa,
            noise_trials=args.noise_trials,
            seed=args.seed,
            qmf_levels=args.qmf_levels,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
