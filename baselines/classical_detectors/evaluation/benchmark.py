from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import json

from ..detectors import FFTDetector, QMFDetector, QuadraticDetector
from ..io.single_emitter_dataset import SingleEmitterDataset
from .metrics import apply_dataset_detector, calibrate_qmf_threshold, empirical_pfa, pd_by_snr
from .noise_models import ThermalNoiseModel


@dataclass(frozen=True)
class BenchmarkConfig:
    dataset_root: Path
    split: str
    pfa: float
    noise_trials: int
    seed: int
    qmf_levels: int


def _save_json(payload: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_benchmark(config: BenchmarkConfig) -> Dict[str, Any]:
    dataset = SingleEmitterDataset(config.dataset_root, split=config.split)
    first_sample = dataset.load(dataset.sample_ids[0])

    representation_cfg = json.loads((config.dataset_root / "representation_config.json").read_text(encoding="utf-8"))
    fe = float(representation_cfg["stft_cfgs"][0]["fs"])
    noise_model = ThermalNoiseModel(fe=fe)

    detectors = {
        "quadratic": QuadraticDetector(),
        "fft_1d": FFTDetector(),
        "qmf": QMFDetector(levels=config.qmf_levels),
    }

    results: Dict[str, Any] = {
        "dataset_root": str(config.dataset_root),
        "split": config.split,
        "pfa_target": config.pfa,
        "fe_hz": fe,
        "signal_length": int(first_sample.signal.size),
    }

    for name, detector in detectors.items():
        forced_threshold = None
        if name == "qmf":
            forced_threshold = calibrate_qmf_threshold(
                detector,
                pfa=config.pfa,
                noise_model=noise_model,
                n_trials=config.noise_trials,
                signal_length=first_sample.signal.size,
                seed=config.seed,
            )

        noise_eval = empirical_pfa(
            detector,
            pfa=config.pfa,
            noise_model=noise_model,
            n_trials=config.noise_trials,
            signal_length=first_sample.signal.size,
            seed=config.seed,
            forced_threshold=forced_threshold,
        )
        if name == "qmf":
            noise_eval["calibrated_threshold"] = forced_threshold

        records = apply_dataset_detector(
            detector,
            dataset,
            pfa=config.pfa,
            noise_variance=noise_model.sample_variance,
            forced_threshold=forced_threshold,
        )
        results[name] = {
            "noise": noise_eval,
            "pd_by_snr": pd_by_snr(records),
        }

    return results
