from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.classical_detectors.detectors import FFTDetector, QMFDetector, QuadraticDetector
from baselines.classical_detectors.evaluation.metrics import calibrate_qmf_threshold, empirical_pfa
from baselines.classical_detectors.evaluation.noise_models import ThermalNoiseModel


SIMULATOR_ROOT = PROJECT_ROOT.parent / "ICML2026DataSimulator"
DEFAULT_DATASET_ROOT = SIMULATOR_ROOT / "tmp" / "output" / "rf_single_emitter_validation_smoketest_f32"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "classical_detectors" / "noise_validation.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate empirical Pfa of classical RF detectors on simulated thermal noise.")
    parser.add_argument("--fe", type=float, default=4.0e9)
    parser.add_argument("--signal-length", type=int, default=131072)
    parser.add_argument("--pfa", type=float, default=1e-3)
    parser.add_argument("--n-trials", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=444)
    parser.add_argument("--qmf-levels", type=int, default=6)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    noise_model = ThermalNoiseModel(fe=args.fe)
    detectors = {
        "quadratic": QuadraticDetector(),
        "fft_1d": FFTDetector(),
        "qmf": QMFDetector(levels=args.qmf_levels),
    }

    payload = {
        "fe_hz": float(args.fe),
        "signal_length": int(args.signal_length),
        "pfa_target": float(args.pfa),
        "n_trials": int(args.n_trials),
    }

    for name, detector in detectors.items():
        forced_threshold = None
        if name == "qmf":
            forced_threshold = calibrate_qmf_threshold(
                detector,
                pfa=args.pfa,
                noise_model=noise_model,
                n_trials=args.n_trials,
                signal_length=args.signal_length,
                seed=args.seed,
            )

        result = empirical_pfa(
            detector,
            pfa=args.pfa,
            noise_model=noise_model,
            n_trials=args.n_trials,
            signal_length=args.signal_length,
            seed=args.seed,
            forced_threshold=forced_threshold,
        )
        if name == "qmf":
            result["calibrated_threshold"] = forced_threshold
        payload[name] = result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
