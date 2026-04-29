from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable

import numpy as np

from ..detectors.base import DetectorResult
from ..io.single_emitter_dataset import SignalSample
from .noise_models import ThermalNoiseModel


@dataclass(frozen=True)
class DetectionRecord:
    sample_id: str
    snr_db: float
    statistic: float
    threshold: float
    decision: bool


def empirical_pfa(
    detector: Any,
    *,
    pfa: float,
    noise_model: ThermalNoiseModel,
    n_trials: int,
    signal_length: int,
    seed: int,
    forced_threshold: float | None = None,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    decisions = []
    statistics = []

    for _ in range(int(n_trials)):
        noise = noise_model.draw(signal_length, rng=rng)
        result = detector.detect(noise, pfa=pfa, noise_variance=noise_model.sample_variance)
        threshold = float(result.threshold if forced_threshold is None else forced_threshold)
        decisions.append(float(result.statistic > threshold))
        statistics.append(result.statistic)

    return {
        "target_pfa": float(pfa),
        "empirical_pfa": float(np.mean(decisions)),
        "mean_statistic_h0": float(np.mean(statistics)),
        "std_statistic_h0": float(np.std(statistics)),
        "n_trials": int(n_trials),
        "signal_length": int(signal_length),
        "threshold": float(threshold),
        "theoretical_binomial_std": float(np.sqrt(pfa * (1.0 - pfa) / float(n_trials))),
        "empirical_ci95_half_width": float(1.96 * np.sqrt(np.mean(decisions) * (1.0 - np.mean(decisions)) / float(n_trials))),
    }


def calibrate_qmf_threshold(
    detector: Any,
    *,
    pfa: float,
    noise_model: ThermalNoiseModel,
    n_trials: int,
    signal_length: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    statistics = []
    for _ in range(int(n_trials)):
        noise = noise_model.draw(signal_length, rng=rng)
        result = detector.detect(noise, pfa=pfa, noise_variance=noise_model.sample_variance)
        statistics.append(result.statistic)
    return float(np.quantile(np.asarray(statistics, dtype=np.float64), 1.0 - pfa))


def apply_dataset_detector(
    detector: Any,
    dataset: Iterable[SignalSample],
    *,
    pfa: float,
    noise_variance: float,
    forced_threshold: float | None = None,
) -> list[DetectionRecord]:
    records: list[DetectionRecord] = []
    for sample in dataset:
        result: DetectorResult = detector.detect(sample.signal, pfa=pfa, noise_variance=noise_variance)
        threshold = float(result.threshold if forced_threshold is None else forced_threshold)
        decision = bool(result.statistic > threshold)
        records.append(
            DetectionRecord(
                sample_id=sample.sample_id,
                snr_db=sample.snr_db,
                statistic=float(result.statistic),
                threshold=threshold,
                decision=decision,
            )
        )
    return records


def pd_by_snr(records: Iterable[DetectionRecord], *, snr_bin_width_db: float = 1.0) -> Dict[str, Any]:
    if snr_bin_width_db <= 0.0:
        raise ValueError("snr_bin_width_db must be strictly positive.")
    grouped: dict[float, list[bool]] = defaultdict(list)
    for record in records:
        snr_bin = float(np.round(record.snr_db / snr_bin_width_db) * snr_bin_width_db)
        grouped[snr_bin].append(bool(record.decision))

    rows = []
    for snr_db in sorted(grouped):
        decisions = np.asarray(grouped[snr_db], dtype=np.float64)
        rows.append(
            {
                "snr_db": float(snr_db),
                "pd": float(np.mean(decisions)),
                "n_samples": int(decisions.size),
            }
        )

    return {"rows": rows}
