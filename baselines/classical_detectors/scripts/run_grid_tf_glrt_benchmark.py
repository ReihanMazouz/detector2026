from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.classical_detectors.detectors import GridTimeFrequencyGLRTDetector
from baselines.classical_detectors.evaluation.metrics import DetectionRecord, pd_by_snr
from baselines.classical_detectors.evaluation.noise_models import ThermalNoiseModel
from baselines.classical_detectors.io.single_emitter_dataset import SingleEmitterDataset


SIMULATOR_ROOT = PROJECT_ROOT.parent / "ICML2026DataSimulator"
DEFAULT_DATASET_ROOT = SIMULATOR_ROOT / "tmp" / "output" / "rf_single_emitter_validation_smoketest_f32"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "classical_detectors" / "grid_tf_glrt_single_emitter_benchmark.json"
DEFAULT_PLOT_OUTPUT = PROJECT_ROOT / "runs" / "classical_detectors" / "grid_tf_glrt_single_emitter_benchmark.png"
DEFAULT_GRID_SIZES = (8, 16, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark grid GLRT detectors on the single-emitter RF time-frequency representation."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--pfa", type=float, default=1e-3)
    parser.add_argument("--noise-trials", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=444)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop", type=int, default=None)
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=list(DEFAULT_GRID_SIZES))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot-output", type=Path, default=DEFAULT_PLOT_OUTPUT)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def _max_statistics_by_grid(
    detectors: dict[int, GridTimeFrequencyGLRTDetector],
    *,
    noise_model: ThermalNoiseModel,
    n_trials: int,
    signal_length: int,
    seed: int,
) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    statistics = {grid_size: [] for grid_size in detectors}
    for _ in range(int(n_trials)):
        noise = noise_model.draw(signal_length, rng=rng)
        for grid_size, detector in detectors.items():
            result = detector.detect(noise, pfa=0.5, noise_variance=noise_model.sample_variance)
            statistics[grid_size].append(result.statistic)
    return {grid_size: np.asarray(values, dtype=np.float64) for grid_size, values in statistics.items()}


def _thresholds_for_alpha(max_statistics: dict[int, np.ndarray], alpha: float) -> dict[int, float]:
    return {
        grid_size: float(np.quantile(statistics, 1.0 - alpha))
        for grid_size, statistics in max_statistics.items()
    }


def _joint_empirical_pfa(max_statistics: dict[int, np.ndarray], thresholds: dict[int, float]) -> float:
    decisions = None
    for grid_size, statistics in max_statistics.items():
        grid_decisions = statistics > thresholds[grid_size]
        decisions = grid_decisions if decisions is None else np.logical_or(decisions, grid_decisions)
    if decisions is None:
        raise ValueError("No grid statistics available for joint calibration.")
    return float(np.mean(decisions))


def calibrate_joint_thresholds(
    detectors: dict[int, GridTimeFrequencyGLRTDetector],
    *,
    pfa: float,
    noise_model: ThermalNoiseModel,
    n_trials: int,
    signal_length: int,
    seed: int,
) -> tuple[dict[int, float], dict[str, Any]]:
    max_statistics = _max_statistics_by_grid(
        detectors,
        noise_model=noise_model,
        n_trials=n_trials,
        signal_length=signal_length,
        seed=seed,
    )

    low = 0.0
    high = float(pfa)
    best_alpha = 0.0
    best_thresholds = _thresholds_for_alpha(max_statistics, best_alpha)
    best_pfa = _joint_empirical_pfa(max_statistics, best_thresholds)
    for _ in range(40):
        mid = 0.5 * (low + high)
        thresholds = _thresholds_for_alpha(max_statistics, mid)
        empirical = _joint_empirical_pfa(max_statistics, thresholds)
        if empirical <= pfa:
            best_alpha = mid
            best_thresholds = thresholds
            best_pfa = empirical
            low = mid
        else:
            high = mid

    per_grid_noise = {}
    for grid_size, statistics in max_statistics.items():
        threshold = best_thresholds[grid_size]
        decisions = statistics > threshold
        per_grid_noise[str(grid_size)] = {
            "threshold": float(threshold),
            "empirical_pfa": float(np.mean(decisions)),
            "mean_statistic_h0": float(np.mean(statistics)),
            "std_statistic_h0": float(np.std(statistics)),
        }

    return best_thresholds, {
        "target_pfa": float(pfa),
        "joint_empirical_pfa": float(best_pfa),
        "per_grid_alpha": float(best_alpha),
        "n_trials": int(n_trials),
        "signal_length": int(signal_length),
        "per_grid": per_grid_noise,
        "theoretical_binomial_std": float(np.sqrt(pfa * (1.0 - pfa) / float(n_trials))),
    }


def _block_is_inside(inner: dict[str, Any], outer: dict[str, Any]) -> bool:
    return (
        int(outer["n_cells"]) > int(inner["n_cells"])
        and int(outer["time_start"]) <= int(inner["time_start"])
        and int(inner["time_stop"]) <= int(outer["time_stop"])
        and int(outer["freq_start"]) <= int(inner["freq_start"])
        and int(inner["freq_stop"]) <= int(outer["freq_stop"])
    )


def _prune_included_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for block in sorted(blocks, key=lambda item: int(item["n_cells"]), reverse=True):
        if any(_block_is_inside(block, previous) for previous in kept):
            continue
        kept.append(block)
    return kept


def _apply_grid_detectors(
    detectors: dict[int, GridTimeFrequencyGLRTDetector],
    dataset: SingleEmitterDataset,
    *,
    thresholds: dict[int, float],
    noise_variance: float,
) -> tuple[dict[int, list[DetectionRecord]], list[DetectionRecord], list[dict[str, Any]]]:
    per_grid_records: dict[int, list[DetectionRecord]] = {grid_size: [] for grid_size in detectors}
    final_records: list[DetectionRecord] = []
    characterizations: list[dict[str, Any]] = []

    for sample in dataset:
        detected_blocks = []
        final_statistic = float("-inf")
        final_threshold_margin = float("-inf")

        for grid_size, detector in detectors.items():
            blocks, _ = detector.block_statistics(sample.signal, noise_variance=noise_variance)
            threshold = thresholds[grid_size]
            grid_statistic = float(max(block["statistic"] for block in blocks))
            grid_decision = bool(grid_statistic > threshold)
            per_grid_records[grid_size].append(
                DetectionRecord(
                    sample_id=sample.sample_id,
                    snr_db=sample.snr_db,
                    statistic=grid_statistic,
                    threshold=float(threshold),
                    decision=grid_decision,
                )
            )
            final_statistic = max(final_statistic, grid_statistic / float(threshold))
            final_threshold_margin = max(final_threshold_margin, grid_statistic - float(threshold))

            for block in blocks:
                if float(block["statistic"]) > threshold:
                    detected_block = dict(block)
                    detected_block["threshold"] = float(threshold)
                    detected_block["margin"] = float(block["statistic"] - threshold)
                    detected_blocks.append(detected_block)

        kept_blocks = _prune_included_blocks(detected_blocks)
        final_records.append(
            DetectionRecord(
                sample_id=sample.sample_id,
                snr_db=sample.snr_db,
                statistic=float(final_statistic),
                threshold=1.0,
                decision=bool(kept_blocks),
            )
        )
        characterizations.append(
            {
                "sample_id": sample.sample_id,
                "snr_db": float(sample.snr_db),
                "decision": bool(kept_blocks),
                "max_margin": float(final_threshold_margin),
                "n_detected_blocks_raw": int(len(detected_blocks)),
                "n_detected_blocks_after_inclusion": int(len(kept_blocks)),
                "blocks": kept_blocks,
            }
        )

    return per_grid_records, final_records, characterizations


def run_grid_tf_glrt_benchmark(
    *,
    dataset_root: Path,
    split: str,
    pfa: float,
    noise_trials: int,
    seed: int,
    n_fft: int,
    hop: int | None,
    grid_sizes: list[int],
) -> dict[str, Any]:
    dataset = SingleEmitterDataset(dataset_root, split=split)
    first_sample = dataset.load(dataset.sample_ids[0])

    representation_cfg = json.loads((dataset_root / "representation_config.json").read_text(encoding="utf-8"))
    fe = float(representation_cfg["stft_cfgs"][0]["fs"])
    noise_model = ThermalNoiseModel(fe=fe)

    results: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "split": split,
        "pfa_target": float(pfa),
        "fe_hz": fe,
        "signal_length": int(first_sample.signal.size),
        "n_fft": int(n_fft),
        "hop": None if hop is None else int(hop),
        "threshold_calibration": "joint_empirical_h0_quantile_for_or_rule",
    }

    detectors = {
        int(grid_size): GridTimeFrequencyGLRTDetector(grid_size=int(grid_size), n_fft=n_fft, hop=hop)
        for grid_size in grid_sizes
    }
    thresholds, calibration = calibrate_joint_thresholds(
        detectors,
        pfa=pfa,
        noise_model=noise_model,
        n_trials=noise_trials,
        signal_length=first_sample.signal.size,
        seed=seed,
    )
    results["joint_calibration"] = calibration

    per_grid_records, final_records, characterizations = _apply_grid_detectors(
        detectors,
        dataset,
        thresholds=thresholds,
        noise_variance=noise_model.sample_variance,
    )

    for grid_size, records in per_grid_records.items():
        grid_calibration = calibration["per_grid"][str(grid_size)]
        results[f"grid_tf_glrt_{grid_size}x{grid_size}"] = {
            "noise": {
                "target_joint_pfa": float(pfa),
                "empirical_pfa_under_joint_threshold": float(grid_calibration["empirical_pfa"]),
                "calibrated_threshold": float(thresholds[grid_size]),
                "mean_statistic_h0": float(grid_calibration["mean_statistic_h0"]),
                "std_statistic_h0": float(grid_calibration["std_statistic_h0"]),
                "n_trials": int(noise_trials),
                "signal_length": int(first_sample.signal.size),
            },
            "pd_by_snr": pd_by_snr(records),
        }
    results["grid_tf_glrt_aggregated"] = {
        "rule": "detect if any grid block exceeds its jointly calibrated threshold; remove detected blocks included in a larger detected block",
        "noise": {
            "target_pfa": float(pfa),
            "empirical_pfa": float(calibration["joint_empirical_pfa"]),
            "per_grid_thresholds": {f"{grid_size}x{grid_size}": float(threshold) for grid_size, threshold in thresholds.items()},
            "n_trials": int(noise_trials),
            "signal_length": int(first_sample.signal.size),
        },
        "pd_by_snr": pd_by_snr(final_records),
        "characterizations": characterizations,
    }

    return results


def _plot_pd_curves(payload: dict[str, Any], output_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for key, value in payload.items():
        if not key.startswith("grid_tf_glrt_") or "pd_by_snr" not in value:
            continue
        rows = value["pd_by_snr"]["rows"]
        if not rows:
            continue
        snr = [row["snr_db"] for row in rows]
        pd = [row["pd"] for row in rows]
        if key == "grid_tf_glrt_aggregated":
            label = "aggregated"
            ax.plot(snr, pd, marker="o", linewidth=2.4, label=label)
        else:
            label = key.replace("grid_tf_glrt_", "").replace("x", "x")
            ax.plot(snr, pd, marker="o", linewidth=1.4, label=label)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Probability of detection")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    payload = run_grid_tf_glrt_benchmark(
        dataset_root=args.dataset_root.resolve(),
        split=args.split,
        pfa=args.pfa,
        noise_trials=args.noise_trials,
        seed=args.seed,
        n_fft=args.n_fft,
        hop=args.hop,
        grid_sizes=args.grid_sizes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not args.no_plot:
        _plot_pd_curves(payload, args.plot_output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
