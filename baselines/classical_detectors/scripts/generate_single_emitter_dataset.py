from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SIMULATOR_ROOT = PROJECT_ROOT.parent / "ICML2026DataSimulator"
SIMULATOR_EXAMPLES = SIMULATOR_ROOT / "examples"

for candidate in (SIMULATOR_ROOT, SIMULATOR_EXAMPLES):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from rf_data_generation import ACQUISITION_TIME, F_E, SIGNAL_DEFS, STFT_CFGS, build_class_index_to_name  # type: ignore  # noqa: E402
from simulator.multi_res_generation_with_seg import (  # type: ignore  # noqa: E402
    FIXED_DISTANCE,
    LIGHT_SPEED,
    NOISE_POWER,
    generate_and_store_spectrum_multi,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "classical_detectors" / "datasets" / "single_emitter_validation"
DEFAULT_SEED = 444
DEFAULT_SCENARIOS_PER_WAVEFORM = 5
DEFAULT_SNR_VALUES = list(range(-20, 21, 2))


def _draw_fp_with_bw(rng: np.random.Generator, bw: float, fp_min: float, fp_max: float) -> float:
    low = fp_min + bw / 2.0
    high = fp_max - bw / 2.0
    if low >= high:
        return 0.5 * (fp_min + fp_max)
    return float(rng.uniform(low, high))


def _erp_for_snr_db(snr_db: float, fp: float) -> float:
    wavelength = LIGHT_SPEED / float(fp)
    return float(
        (10 ** (float(snr_db) / 10.0) * NOISE_POWER * (4.0 * np.pi * FIXED_DISTANCE) ** 2)
        / wavelength**2
    )


def _build_single_emitter_signal_defs(selected_waveforms: set[str] | None) -> dict:
    waveforms = {}
    for label, definition in SIGNAL_DEFS["waveforms"].items():
        if label == "none" or float(definition.get("p", 0.0)) <= 0.0:
            continue
        if selected_waveforms is not None and label not in selected_waveforms:
            continue
        copied = dict(definition)
        copied["p"] = 1.0
        waveforms[label] = copied

    if not waveforms:
        raise ValueError("No waveform selected for generation.")

    return {
        "waveforms": waveforms,
        "interferences": {
            "none": {
                "duration_min": 0.0,
                "duration_max": 0.0,
                "bandwidth_min": 0.0,
                "bandwidth_max": 0.0,
                "p": 0.0,
            }
        },
    }


def _sample_base_waveform(
    *,
    rng: np.random.Generator,
    label: str,
    definition: dict,
) -> dict:
    pw = float(rng.uniform(definition["duration_min"], definition["duration_max"]))
    bw = float(rng.uniform(definition["bandwidth_min"], definition["bandwidth_max"]))
    fp = _draw_fp_with_bw(rng, bw, 0.1 * F_E / 2.0, 0.9 * F_E / 2.0)
    pri = float(rng.uniform(min(2.0 * pw, 1e3 / LIGHT_SPEED), 10.0 * pw))
    waveform = {
        "waveform_label": label,
        "waveform_type": str(definition.get("modulation", label)),
        "erp": 0.0,
        "pw": pw,
        "fe": F_E,
        "fp": fp,
        "bandwidth": bw,
        "numberOfCycles": 1,
        "pri": pri,
    }
    if waveform["waveform_type"] == "FMCW_TRI":
        cycle_duration = definition.get("cycle_duration")
        if cycle_duration is None:
            cycle_duration = float(rng.uniform(pw / 50.0, pw / 2.0))
        waveform["cycle_duration"] = float(cycle_duration)
        waveform["start_direction"] = int(rng.choice([-1, 1]))
    return waveform


def build_single_emitter_snr_sweep_scenarios(
    *,
    signal_defs: dict,
    scenarios_per_waveform: int,
    snr_values: Iterable[float],
    seed: int,
) -> list[list[dict]]:
    scenarios: list[list[dict]] = []
    active_waveforms = list(signal_defs["waveforms"].items())
    group_id = 0

    for waveform_label, definition in active_waveforms:
        for scenario_index in range(scenarios_per_waveform):
            scenario_seed = int(seed + group_id * 1009)
            acquisition_seed = int(seed + group_id * 10007)
            rng = np.random.default_rng(scenario_seed)
            base_waveform = _sample_base_waveform(
                rng=rng,
                label=waveform_label,
                definition=definition,
            )
            waveform_type = str(base_waveform["waveform_type"])

            for snr_db in snr_values:
                waveform = copy.deepcopy(base_waveform)
                waveform["waveform_id"] = f"{waveform_label}_g{group_id:05d}_snr{snr_db:+05.1f}"
                waveform["erp"] = _erp_for_snr_db(float(snr_db), waveform["fp"])
                waveform["target_snr_db"] = float(snr_db)
                waveform["scenario_group_id"] = group_id
                waveform["sweep_snr_db"] = float(snr_db)
                waveform["acquisition_seed"] = acquisition_seed

                emitter = {
                    "label": "Emitter_1",
                    "emitter_type": "radar",
                    "x": FIXED_DISTANCE,
                    "y": 0.0,
                    "waveforms": [waveform],
                    "target_snr_db": float(snr_db),
                    "scenario_group_id": group_id,
                    "sweep_snr_db": float(snr_db),
                    "acquisition_seed": acquisition_seed,
                    "scenario_waveform_label": waveform_label,
                    "scenario_waveform_type": waveform_type,
                    "scenario_index_per_waveform": scenario_index,
                }
                scenarios.append([emitter])
            group_id += 1

    return scenarios


def _parse_snr_values(raw: str) -> list[float]:
    values = []
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError("At least one SNR value must be provided.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a single-emitter raw 1D RF dataset compatible with "
            "baselines/classical_detectors."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--scenarios-per-waveform", type=int, default=DEFAULT_SCENARIOS_PER_WAVEFORM)
    parser.add_argument(
        "--waveforms",
        type=str,
        default="all",
        help="Comma-separated waveform labels from SIGNAL_DEFS, or 'all'.",
    )
    parser.add_argument(
        "--snr-values",
        type=str,
        default=",".join(str(v) for v in DEFAULT_SNR_VALUES),
        help="Comma-separated SNR values in dB, e.g. '-20,-10,0,10,20'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()

    selected_waveforms = None
    if str(args.waveforms).strip().lower() != "all":
        selected_waveforms = {item.strip() for item in str(args.waveforms).split(",") if item.strip()}

    signal_defs = _build_single_emitter_signal_defs(selected_waveforms)
    class_index_to_name = build_class_index_to_name(signal_defs)
    snr_values = _parse_snr_values(args.snr_values)

    scenarios = build_single_emitter_snr_sweep_scenarios(
        signal_defs=signal_defs,
        scenarios_per_waveform=args.scenarios_per_waveform,
        snr_values=snr_values,
        seed=args.seed,
    )

    generate_and_store_spectrum_multi(
        scenarios=scenarios,
        base_path=str(output_dir),
        split_train_test=True,
        train_ratio=0.0,
        acquisition_time=ACQUISITION_TIME,
        stft_cfgs=STFT_CFGS,
        seed=args.seed,
        pow2_resize=False,
        preprocessing="log_snr_estimated",
        store_complex_spectrum=False,
        save_raw_data=True,
        representation="stft",
        class_index_to_name=class_index_to_name,
    )

    print(f"dataset written to {output_dir}")
    print(f"waveforms: {sorted(signal_defs['waveforms'].keys())}")
    print(f"snr values (dB): {snr_values}")
    print(f"validation samples: {len(scenarios)}")
    print("benchmark example:")
    print(
        "python baselines/classical_detectors/scripts/run_single_emitter_benchmark.py "
        f"--dataset-root {output_dir}"
    )


if __name__ == "__main__":
    main()
