from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.classical_detectors.evaluation.deep_waveform_sweep import DEFAULT_RES_HW  # noqa: E402
from baselines.classical_detectors.scripts.train_resnet_snr_curriculum import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TEST_ROOT,
    _calibrate_threshold,
    _diagonal_rows,
    _evaluate_pd_by_snr,
    _load_model_state_dict,
    _load_scenarios,
    _parse_snr_values,
    _resolve_device,
    _snr_sequence,
    _write_csv,
)
from core.models.resnet import resnet50d_classifier  # noqa: E402


_SNR_DIR_RE = re.compile(r"^snr_([+-]\d+)$")


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def _train_snr_from_dir(path: Path) -> float | None:
    match = _SNR_DIR_RE.match(path.name)
    if match is None:
        return None
    return float(int(match.group(1)))


def _load_checkpoint_state(checkpoint_path: Path) -> dict[str, Any]:
    state_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_state.json")
    if not state_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint metadata: {state_path}")
    return json.loads(state_path.read_text(encoding="utf-8"))


def _discover_checkpoints(root: Path) -> list[dict[str, Any]]:
    candidates = []
    for checkpoint_path in root.glob("snr_*/best.pt"):
        state = _load_checkpoint_state(checkpoint_path)
        train_snr = state.get("train_snr", _train_snr_from_dir(checkpoint_path.parent))
        if train_snr is None:
            raise ValueError(f"Cannot infer train_snr for checkpoint: {checkpoint_path}")
        candidates.append(
            {
                "train_snr": float(train_snr),
                "checkpoint_path": checkpoint_path,
                "state": state,
            }
        )
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under {root}/snr_*/best.pt")
    return sorted(candidates, key=lambda item: float(item["train_snr"]), reverse=True)


def _filter_checkpoints(
    checkpoints: Sequence[dict[str, Any]],
    *,
    train_snr_values: Sequence[float] | None,
) -> list[dict[str, Any]]:
    if train_snr_values is None:
        return list(checkpoints)
    requested = {float(value) for value in train_snr_values}
    selected = [item for item in checkpoints if float(item["train_snr"]) in requested]
    missing = sorted(requested - {float(item["train_snr"]) for item in selected}, reverse=True)
    if missing:
        raise FileNotFoundError(f"Missing best.pt checkpoints for train SNR values: {missing}")
    return selected


def _class_index_to_name_from_state(state: dict[str, Any]) -> dict[int, str]:
    raw = state.get("class_index_to_name")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Checkpoint state does not contain class_index_to_name.")
    return {int(key): str(value) for key, value in raw.items()}


def _write_by_waveform_diagonal_plot(rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in _diagonal_rows(rows):
        grouped.setdefault(str(row["waveform_label"]), []).append(row)

    fig, axis = plt.subplots(figsize=(10.0, 6.0))
    for waveform_label, items in sorted(grouped.items()):
        items = sorted(items, key=lambda item: float(item["eval_snr"]))
        axis.plot(
            [float(item["eval_snr"]) for item in items],
            [float(item["pd"]) for item in items],
            marker="o",
            linewidth=1.5,
            markersize=3.0,
            label=waveform_label,
        )
    axis.set_xlabel("Training/evaluation SNR (dB)")
    axis.set_ylabel("Pd")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", fontsize="x-small", ncol=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_threshold_csv(rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "train_snr",
        "checkpoint_path",
        "threshold",
        "empirical_pfa",
        "allowed_false_alarms",
        "observed_false_alarms",
        "n_trials",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate existing ResNet SNR-curriculum best.pt checkpoints on the clean test set."
    )
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT)
    parser.add_argument("--res-key", default="cfg512")
    parser.add_argument("--preprocessing", default="log_snr_estimated")
    parser.add_argument("--pfa", type=float, default=1e-2)
    parser.add_argument("--noise-trials", type=int, default=1000)
    parser.add_argument("--noise-variance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=400)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--train-snr-start", type=int, default=None)
    parser.add_argument("--train-snr-end", type=int, default=None)
    parser.add_argument("--train-snr-step", type=int, default=1)
    parser.add_argument(
        "--train-snr-values",
        default=None,
        help="Comma-separated checkpoint SNR filter, e.g. '0,-5,-10'. Defaults to all discovered checkpoints.",
    )
    parser.add_argument(
        "--eval-snr-values",
        default=None,
        help="Comma-separated eval SNR values. By default each checkpoint is evaluated only at its own train_snr.",
    )
    parser.add_argument(
        "--full-sweep",
        action="store_true",
        help="Evaluate every selected checkpoint on every --eval-snr-values value instead of the diagonal only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_root = args.checkpoint_root.resolve()
    output_dir = (args.output_dir or checkpoint_root).resolve()
    test_root = args.test_root.resolve()

    if args.res_key not in DEFAULT_RES_HW:
        raise ValueError(f"Unknown res_key '{args.res_key}'. Expected one of {sorted(DEFAULT_RES_HW)}.")
    if not (test_root / "manifest.json").is_file():
        raise FileNotFoundError(f"Missing test manifest: {test_root / 'manifest.json'}")

    train_snr_values = None
    if args.train_snr_values is not None or args.train_snr_start is not None or args.train_snr_end is not None:
        if args.train_snr_values is not None:
            train_snr_values = _parse_snr_values(args.train_snr_values, start=0, end=0, step=1)
        else:
            if args.train_snr_start is None or args.train_snr_end is None:
                raise ValueError("--train-snr-start and --train-snr-end must be provided together.")
            train_snr_values = tuple(
                float(value)
                for value in _snr_sequence(
                    int(args.train_snr_start),
                    int(args.train_snr_end),
                    int(args.train_snr_step),
                )
            )

    checkpoints = _filter_checkpoints(_discover_checkpoints(checkpoint_root), train_snr_values=train_snr_values)
    device = _resolve_device(args.device)

    _log(f"loading clean test scenarios from {test_root}")
    test_scenarios = _load_scenarios(test_root)
    if not test_scenarios:
        raise RuntimeError(f"No test scenarios found in {test_root}")

    all_rows: list[dict[str, Any]] = []
    all_waveform_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []

    for item in checkpoints:
        train_snr = float(item["train_snr"])
        checkpoint_path = Path(item["checkpoint_path"])
        state = dict(item["state"])
        class_index_to_name = _class_index_to_name_from_state(state)
        class_name_to_index = {name: index for index, name in class_index_to_name.items()}

        unknown_test_classes = sorted(
            {str(scenario.class_name) for scenario in test_scenarios}
            - set(class_name_to_index)
        )
        if unknown_test_classes:
            raise ValueError(
                f"Test dataset contains classes absent from {checkpoint_path}: {unknown_test_classes}"
            )

        _log(f"loading checkpoint train_snr={train_snr:g}: {checkpoint_path}")
        model = resnet50d_classifier(
            num_classes=len(class_index_to_name),
            input_canals=1,
            device=device,
        )
        state_dict = torch.load(checkpoint_path, map_location=device)
        _load_model_state_dict(model, state_dict)

        signal_length = int(len(test_scenarios[0].signal))
        calibration = _calibrate_threshold(
            model=model,
            signal_length=signal_length,
            noise_variance=float(args.noise_variance),
            pfa=float(args.pfa),
            n_trials=int(args.noise_trials),
            seed=int(args.seed + 1009 + round((train_snr + 1000.0) * 1000.0)),
            device=device,
            res_key=args.res_key,
            preprocessing=args.preprocessing,
            batch_size=int(args.eval_batch_size),
        )
        threshold_rows.append(
            {
                "train_snr": train_snr,
                "checkpoint_path": str(checkpoint_path),
                "threshold": float(calibration.threshold),
                "empirical_pfa": float(calibration.empirical_pfa),
                "allowed_false_alarms": int(calibration.allowed_false_alarms),
                "observed_false_alarms": int(calibration.observed_false_alarms),
                "n_trials": int(calibration.n_trials),
            }
        )

        if args.full_sweep:
            eval_snr_values = _parse_snr_values(
                args.eval_snr_values,
                start=int(min(float(row["train_snr"]) for row in checkpoints)),
                end=int(max(float(row["train_snr"]) for row in checkpoints)),
                step=1,
            )
        else:
            eval_snr_values = (train_snr,)

        eval_rows, eval_waveform_rows = _evaluate_pd_by_snr(
            model=model,
            scenarios=test_scenarios,
            class_name_to_index=class_name_to_index,
            eval_snr_values=eval_snr_values,
            threshold=calibration.threshold,
            noise_variance=float(args.noise_variance),
            seed=int(args.seed),
            device=device,
            res_key=args.res_key,
            preprocessing=args.preprocessing,
            batch_size=int(args.eval_batch_size),
        )
        for row in eval_rows:
            all_rows.append(
                {
                    "train_snr": train_snr,
                    "eval_snr": float(row["eval_snr"]),
                    "pd": float(row["pd"]),
                    "classification_accuracy": float(row["classification_accuracy"]),
                    "classification_accuracy_detected": float(row["classification_accuracy_detected"]),
                    "mean_score": float(row["mean_score"]),
                    "n_samples": int(row["n_samples"]),
                    "n_detected": int(row["n_detected"]),
                    "threshold": float(calibration.threshold),
                    "empirical_pfa": float(calibration.empirical_pfa),
                }
            )
        for row in eval_waveform_rows:
            all_waveform_rows.append(
                {
                    "waveform_label": str(row["waveform_label"]),
                    "train_snr": train_snr,
                    "eval_snr": float(row["eval_snr"]),
                    "pd": float(row["pd"]),
                    "classification_accuracy": float(row["classification_accuracy"]),
                    "classification_accuracy_detected": float(row["classification_accuracy_detected"]),
                    "mean_score": float(row["mean_score"]),
                    "n_samples": int(row["n_samples"]),
                    "n_detected": int(row["n_detected"]),
                    "threshold": float(calibration.threshold),
                    "empirical_pfa": float(calibration.empirical_pfa),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(all_rows, output_dir / "resnet_snr_curriculum_pd_from_checkpoints.csv")
    _write_csv(all_waveform_rows, output_dir / "resnet_snr_curriculum_pd_by_waveform_from_checkpoints.csv")
    _write_csv(
        _diagonal_rows(all_waveform_rows),
        output_dir / "resnet_snr_curriculum_pd_diagonal_by_waveform.csv",
    )
    _write_threshold_csv(threshold_rows, output_dir / "resnet_snr_curriculum_thresholds_from_checkpoints.csv")
    _write_by_waveform_diagonal_plot(
        all_waveform_rows,
        output_dir / "resnet_snr_curriculum_pd_diagonal_by_waveform.png",
    )

    summary = {
        "checkpoint_root": str(checkpoint_root),
        "test_root": str(test_root),
        "res_key": str(args.res_key),
        "preprocessing": str(args.preprocessing),
        "pfa": float(args.pfa),
        "noise_trials": int(args.noise_trials),
        "noise_variance": float(args.noise_variance),
        "seed": int(args.seed),
        "full_sweep": bool(args.full_sweep),
        "threshold_rows": threshold_rows,
        "pd_rows": all_rows,
        "pd_by_waveform_rows": all_waveform_rows,
        "pd_definition": "Pd = P(max sigmoid(logits) > threshold), threshold calibrated per checkpoint on noise-only H0 at target Pfa. The diagonal by-waveform plot uses eval_snr == train_snr.",
    }
    summary_path = output_dir / "resnet_snr_curriculum_eval_from_checkpoints_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log(f"wrote {summary_path}")
    _log(f"wrote {output_dir / 'resnet_snr_curriculum_pd_diagonal_by_waveform.png'}")


if __name__ == "__main__":
    main()
