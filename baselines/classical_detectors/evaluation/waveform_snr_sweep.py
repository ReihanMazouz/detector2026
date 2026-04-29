from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np

from ..detectors import FFTDetector, QMFDetector, QuadraticDetector, TimeFrequencyGLRTDetector
from ..io import WaveformManifestDataset
from .noise_models import draw_real_awgn


@dataclass(frozen=True)
class WaveformSweepConfig:
    dataset_root: Path
    pfa: float = 1e-3
    noise_trials: int = 5000
    seed: int = 444
    qmf_levels: int = 6
    noise_variance: float = 1.0
    snr_values_db: tuple[float, ...] = tuple(range(-20, 21, 2))
    waveforms: tuple[str, ...] = ()
    progress_log: Callable[[str], None] | None = None


def _detectors(qmf_levels: int) -> dict[str, Any]:
    return {
        "quadratic": QuadraticDetector(),
        "fft_1d": FFTDetector(),
        "tf_glrt": TimeFrequencyGLRTDetector(),
        "qmf": QMFDetector(levels=qmf_levels),
    }


def _noise_seed(seed: int, scenario_index: int) -> int:
    return int(seed + 1_000_003 * scenario_index)


def _calibrate_threshold(detector: Any, *, pfa: float, noise_variance: float, signal_length: int, n_trials: int, seed: int) -> dict:
    if int(n_trials) < int(np.ceil(1.0 / float(pfa))):
        raise ValueError("n_trials must be at least ceil(1 / pfa) to control empirical Pfa.")
    rng = np.random.default_rng(seed)
    statistics = []
    for _ in range(int(n_trials)):
        noise = draw_real_awgn(signal_length, noise_variance=noise_variance, rng=rng)
        statistics.append(float(detector.detect(noise, pfa=pfa, noise_variance=noise_variance).statistic))
    values = np.asarray(statistics, dtype=np.float64)
    allowed_false_alarms = int(np.floor(float(pfa) * float(values.size)))
    sorted_values = np.sort(values)
    threshold_index = max(0, int(values.size) - allowed_false_alarms - 1)
    threshold = float(sorted_values[threshold_index])
    empirical_pfa = float(np.mean(values > threshold))
    return {
        "threshold": threshold,
        "empirical_pfa": empirical_pfa,
        "allowed_false_alarms": allowed_false_alarms,
        "observed_false_alarms": int(np.sum(values > threshold)),
        "mean_statistic_h0": float(np.mean(values)),
        "std_statistic_h0": float(np.std(values)),
        "n_trials": int(n_trials),
    }


def _scale_for_target_snr(signal: np.ndarray, *, noise_variance: float, snr_db: float, duration_samples: int) -> float:
    energy = float(np.sum(np.abs(signal) ** 2))
    if energy <= 0.0 or int(duration_samples) <= 0:
        return 0.0
    target_energy = float(noise_variance) * float(duration_samples) * 10.0 ** (float(snr_db) / 10.0)
    return float(np.sqrt(target_energy / energy))


def _summarize(rows: list[dict], x_key: str) -> list[dict]:
    grouped: dict[tuple[str, float], list[bool]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["waveform_label"]), float(row[x_key]))].append(bool(row["decision"]))
    summary = []
    for (waveform_label, x_value), decisions in sorted(grouped.items()):
        arr = np.asarray(decisions, dtype=np.float64)
        summary.append(
            {
                "waveform_label": waveform_label,
                x_key: x_value,
                "pd": float(np.mean(arr)),
                "n_samples": int(arr.size),
            }
        )
    return summary


def _group_scenarios_by_waveform(scenarios: list[Any]) -> dict[str, list[tuple[int, Any]]]:
    grouped: dict[str, list[tuple[int, Any]]] = defaultdict(list)
    for scenario_index, scenario in enumerate(scenarios):
        grouped[str(scenario.waveform_label)].append((scenario_index, scenario))
    return dict(grouped)


def _log(config: WaveformSweepConfig, message: str) -> None:
    if config.progress_log is not None:
        config.progress_log(message)


def run_waveform_snr_sweep(config: WaveformSweepConfig) -> Dict[str, Any]:
    selected_waveforms = set(config.waveforms) if config.waveforms else None
    dataset = WaveformManifestDataset(config.dataset_root, waveforms=selected_waveforms)
    scenarios = list(dataset)
    signal_length = int(scenarios[0].signal.size)

    results: Dict[str, Any] = {
        "dataset_root": str(config.dataset_root),
        "pfa_target": float(config.pfa),
        "seed": int(config.seed),
        "signal_length": signal_length,
        "n_scenarios": len(scenarios),
        "signal_domain": "real_1d",
        "snr_definition": "(sum(|s|^2) / duration_samples) / noise_variance",
        "noise_seed_policy": "same noise realization for each scenario across all SNR levels",
        "snr_values_db": [float(v) for v in config.snr_values_db],
        "noise_variance": float(config.noise_variance),
        "detectors": {},
    }

    for detector_name, detector in _detectors(config.qmf_levels).items():
        _log(config, f"[classical:{detector_name}] calibrating threshold on H0, trials={config.noise_trials}")
        calibration = _calibrate_threshold(
            detector,
            pfa=config.pfa,
            noise_variance=float(config.noise_variance),
            signal_length=signal_length,
            n_trials=config.noise_trials,
            seed=config.seed,
        )
        threshold = float(calibration["threshold"])
        detector_payload = {"threshold": calibration, "by_snr": []}
        _log(
            config,
            f"[classical:{detector_name}] threshold={threshold:.6g}, empirical_pfa={calibration['empirical_pfa']:.6g}",
        )

        grouped_scenarios = _group_scenarios_by_waveform(scenarios)

        _log(config, f"[classical:{detector_name}] sweep Pd vs SNR")
        for waveform_label, waveform_scenarios in grouped_scenarios.items():
            for snr_db in sorted((float(v) for v in config.snr_values_db), reverse=True):
                decisions = []
                for scenario_index, scenario in waveform_scenarios:
                    rng = np.random.default_rng(_noise_seed(config.seed, scenario_index))
                    noise = draw_real_awgn(signal_length, noise_variance=float(config.noise_variance), rng=rng)
                    scale = _scale_for_target_snr(
                        scenario.signal,
                        noise_variance=float(config.noise_variance),
                        snr_db=float(snr_db),
                        duration_samples=int(scenario.duration_samples),
                    )
                    statistic = float(detector.detect(scale * scenario.signal + noise, pfa=config.pfa, noise_variance=float(config.noise_variance)).statistic)
                    decisions.append(bool(statistic > threshold))
                pd = float(np.mean(np.asarray(decisions, dtype=np.float64)))
                detector_payload["by_snr"].append(
                    {
                        "waveform_label": waveform_label,
                        "snr_db": float(snr_db),
                        "pd": pd,
                        "n_samples": int(len(decisions)),
                        "early_stopped": bool(pd == 0.0),
                    }
                )
                _log(config, f"[classical:{detector_name}] waveform={waveform_label} snr_db={snr_db:.2f} pd={pd:.4f}")
                if pd == 0.0:
                    _log(config, f"[classical:{detector_name}] waveform={waveform_label} stop SNR sweep at {snr_db:.2f} dB")
                    break

        results["detectors"][detector_name] = detector_payload
        _log(config, f"[classical:{detector_name}] done")

    return results
