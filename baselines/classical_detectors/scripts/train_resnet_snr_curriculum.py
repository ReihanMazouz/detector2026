from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.classical_detectors.evaluation.deep_waveform_sweep import (  # noqa: E402
    DEFAULT_RES_HW,
    _noise_seed,
    _preprocess_tensor,
    _scale_for_target_snr,
    _stft_spectrum,
)
from baselines.classical_detectors.evaluation.noise_models import draw_real_awgn  # noqa: E402
from baselines.classical_detectors.io import WaveformManifestDataset, WaveformScenario  # noqa: E402
from core.models.resnet import resnet50d_classifier  # noqa: E402


DEFAULT_GENERATOR_SCRIPT = (
    PROJECT_ROOT.parent / "ICML2026DataSimulator" / "examples" / "rf_single_emitter_raw_generation.py"
)
DEFAULT_VAL_ROOT = Path("/data/RAWSIM/RMA/rf_single_emitter_real_validation")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "classical_detectors" / "resnet_snr_curriculum"
DEFAULT_WORK_DIR = PROJECT_ROOT / "runs" / "classical_detectors" / "tmp" / "resnet_snr_curriculum_dataset"


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def _snr_sequence(start: int, end: int, step: int) -> list[int]:
    if step <= 0:
        raise ValueError("SNR step must be strictly positive.")
    if start >= end:
        return list(range(start, end - 1, -step))
    return list(range(start, end + 1, step))


def _parse_snr_values(raw: str | None, *, start: int, end: int, step: int) -> tuple[float, ...]:
    if raw:
        values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
        if not values:
            raise argparse.ArgumentTypeError("Expected at least one SNR value.")
        return values
    return tuple(float(v) for v in _snr_sequence(start, end, step))


def _resolve_device(raw: str) -> torch.device:
    requested = torch.device(raw)
    if requested.type == "cuda" and not torch.cuda.is_available():
        _log(f"cuda requested but unavailable; using cpu instead of {raw}")
        return torch.device("cpu")
    return requested


def _build_class_table(scenarios: Sequence[WaveformScenario]) -> tuple[dict[int, str], dict[str, int]]:
    class_names = sorted({str(scenario.class_name) for scenario in scenarios if str(scenario.class_name)})
    if not class_names:
        raise ValueError("No class_name found in generated scenarios.")
    class_index_to_name = {index: name for index, name in enumerate(class_names)}
    class_name_to_index = {name: index for index, name in class_index_to_name.items()}
    return class_index_to_name, class_name_to_index


def _load_scenarios(dataset_root: Path) -> list[WaveformScenario]:
    return list(WaveformManifestDataset(dataset_root))


def _signal_to_image(signal: np.ndarray, *, res_key: str, preprocessing: str) -> torch.Tensor:
    spectrum = _stft_spectrum(signal, res_key=res_key)
    return _preprocess_tensor(spectrum, name=preprocessing, res_key=res_key)


class NoisyWaveformClassificationDataset(Dataset):
    def __init__(
        self,
        scenarios: Sequence[WaveformScenario],
        *,
        class_name_to_index: dict[str, int],
        num_classes: int,
        snr_db: float,
        noise_variance: float,
        noise_samples_per_signal: float,
        seed: int,
        res_key: str,
        preprocessing: str,
    ) -> None:
        self.scenarios = list(scenarios)
        self.class_name_to_index = dict(class_name_to_index)
        self.num_classes = int(num_classes)
        self.snr_db = float(snr_db)
        self.noise_variance = float(noise_variance)
        self.noise_sample_count = int(round(len(self.scenarios) * float(noise_samples_per_signal)))
        self.seed = int(seed)
        self.res_key = str(res_key)
        self.preprocessing = str(preprocessing)
        if not self.scenarios:
            raise ValueError("At least one signal scenario is required.")
        self.signal_length = int(len(self.scenarios[0].signal))

    def __len__(self) -> int:
        return len(self.scenarios) + self.noise_sample_count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index >= len(self.scenarios):
            noise_index = index - len(self.scenarios)
            rng = np.random.default_rng(_noise_seed(self.seed, len(self.scenarios) + noise_index))
            noise = draw_real_awgn(self.signal_length, noise_variance=self.noise_variance, rng=rng)
            image = _signal_to_image(noise, res_key=self.res_key, preprocessing=self.preprocessing)
            return image, torch.zeros(self.num_classes, dtype=torch.float32)

        scenario = self.scenarios[index]
        try:
            label = self.class_name_to_index[str(scenario.class_name)]
        except KeyError as exc:
            raise KeyError(f"Unknown class_name='{scenario.class_name}' for scenario={scenario.scenario_id}") from exc

        rng = np.random.default_rng(_noise_seed(self.seed, index))
        noise = draw_real_awgn(len(scenario.signal), noise_variance=self.noise_variance, rng=rng)
        scale = _scale_for_target_snr(
            scenario.signal,
            noise_variance=self.noise_variance,
            snr_db=self.snr_db,
            duration_samples=scenario.duration_samples,
        )
        image = _signal_to_image(
            scale * scenario.signal + noise,
            res_key=self.res_key,
            preprocessing=self.preprocessing,
        )
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        target[label] = 1.0
        return image, target


@dataclass(frozen=True)
class CalibrationResult:
    threshold: float
    empirical_pfa: float
    allowed_false_alarms: int
    observed_false_alarms: int
    n_trials: int


def _generate_dataset(
    *,
    generator_script: Path,
    output_dir: Path,
    seed: int,
    scenarios_per_waveform: int,
    python_executable: str,
) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_executable,
        str(generator_script),
        "--seed",
        str(seed),
        "--scenarios-per-waveform",
        str(scenarios_per_waveform),
        "--output-dir",
        str(output_dir),
        "--overwrite",
    ]
    _log("generating clean dataset: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def _train_one_snr(
    *,
    model: torch.nn.Module,
    train_dataset: Dataset,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    num_workers: int,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
) -> list[dict[str, float]]:
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    criterion = torch.nn.BCEWithLogitsLoss()
    history = []
    best_loss = float("inf")
    epochs_without_improvement = 0

    model.train()
    for epoch in range(1, int(epochs) + 1):
        total_loss = 0.0
        total_signal_correct = 0
        total_signal_samples = 0
        total_noise_rejected = 0
        total_noise_samples = 0
        total_samples = 0
        for images, targets in loader:
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            targets = targets.to(device, dtype=torch.float32, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            batch_size_actual = int(targets.shape[0])
            total_loss += float(loss.item()) * batch_size_actual
            probs = torch.sigmoid(logits)
            target_is_signal = targets.sum(dim=1) > 0.0
            if bool(target_is_signal.any()):
                total_signal_correct += int(
                    (probs[target_is_signal].argmax(dim=1) == targets[target_is_signal].argmax(dim=1)).sum().item()
                )
                total_signal_samples += int(target_is_signal.sum().item())
            target_is_noise = ~target_is_signal
            if bool(target_is_noise.any()):
                total_noise_rejected += int((probs[target_is_noise].max(dim=1).values <= 0.5).sum().item())
                total_noise_samples += int(target_is_noise.sum().item())
            total_samples += batch_size_actual

        row = {
            "epoch": float(epoch),
            "loss": float(total_loss / max(total_samples, 1)),
            "signal_accuracy": float(total_signal_correct / max(total_signal_samples, 1)),
            "noise_rejection_at_0_5": float(total_noise_rejected / max(total_noise_samples, 1)),
        }
        history.append(row)
        _log(
            f"epoch={epoch} loss={row['loss']:.6f} "
            f"signal_accuracy={row['signal_accuracy']:.4f} "
            f"noise_rejection_at_0_5={row['noise_rejection_at_0_5']:.4f}"
        )
        if int(early_stopping_patience) > 0:
            improvement = best_loss - float(row["loss"])
            if improvement > float(early_stopping_min_delta):
                best_loss = float(row["loss"])
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= int(early_stopping_patience):
                    _log(
                        "early stopping: "
                        f"loss did not improve by {float(early_stopping_min_delta):.6g} "
                        f"for {int(early_stopping_patience)} epochs"
                    )
                    break
    return history


def _confidence_scores(
    *,
    model: torch.nn.Module,
    signals: Sequence[np.ndarray],
    device: torch.device,
    res_key: str,
    preprocessing: str,
    batch_size: int,
) -> list[float]:
    scores: list[float] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(signals), max(1, int(batch_size))):
            batch_signals = signals[start : start + max(1, int(batch_size))]
            images = [
                _signal_to_image(signal, res_key=res_key, preprocessing=preprocessing)
                for signal in batch_signals
            ]
            batch = torch.stack(images, dim=0).to(device, dtype=torch.float32)
            probs = torch.sigmoid(model(batch))
            scores.extend(probs.max(dim=1).values.detach().cpu().tolist())
    return [float(score) for score in scores]


def _calibrate_threshold(
    *,
    model: torch.nn.Module,
    signal_length: int,
    noise_variance: float,
    pfa: float,
    n_trials: int,
    seed: int,
    device: torch.device,
    res_key: str,
    preprocessing: str,
    batch_size: int,
) -> CalibrationResult:
    if int(n_trials) < int(np.ceil(1.0 / float(pfa))):
        raise ValueError("noise-trials must be at least ceil(1 / pfa) to control empirical Pfa.")
    rng = np.random.default_rng(seed)
    noises = [
        draw_real_awgn(signal_length, noise_variance=float(noise_variance), rng=rng)
        for _ in range(int(n_trials))
    ]
    values = np.asarray(
        _confidence_scores(
            model=model,
            signals=noises,
            device=device,
            res_key=res_key,
            preprocessing=preprocessing,
            batch_size=batch_size,
        ),
        dtype=np.float64,
    )
    allowed_false_alarms = int(np.floor(float(pfa) * float(values.size)))
    sorted_values = np.sort(values)
    threshold_index = max(0, int(values.size) - allowed_false_alarms - 1)
    threshold = float(sorted_values[threshold_index])
    return CalibrationResult(
        threshold=threshold,
        empirical_pfa=float(np.mean(values > threshold)),
        allowed_false_alarms=allowed_false_alarms,
        observed_false_alarms=int(np.sum(values > threshold)),
        n_trials=int(n_trials),
    )


def _evaluate_pd_by_snr(
    *,
    model: torch.nn.Module,
    scenarios: Sequence[WaveformScenario],
    eval_snr_values: Sequence[float],
    threshold: float,
    noise_variance: float,
    seed: int,
    device: torch.device,
    res_key: str,
    preprocessing: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    rows = []
    signal_length = int(len(scenarios[0].signal))
    for snr_db in eval_snr_values:
        signals = []
        for scenario_index, scenario in enumerate(scenarios):
            rng = np.random.default_rng(_noise_seed(seed, scenario_index))
            noise = draw_real_awgn(signal_length, noise_variance=float(noise_variance), rng=rng)
            scale = _scale_for_target_snr(
                scenario.signal,
                noise_variance=float(noise_variance),
                snr_db=float(snr_db),
                duration_samples=int(scenario.duration_samples),
            )
            signals.append(scale * scenario.signal + noise)
        scores = _confidence_scores(
            model=model,
            signals=signals,
            device=device,
            res_key=res_key,
            preprocessing=preprocessing,
            batch_size=batch_size,
        )
        decisions = np.asarray(scores, dtype=np.float64) > float(threshold)
        rows.append(
            {
                "eval_snr": float(snr_db),
                "pd": float(np.mean(decisions)),
                "n_samples": int(decisions.size),
            }
        )
        _log(f"eval_snr={float(snr_db):.2f} pd={rows[-1]['pd']:.4f}")
    return rows


def _write_csv(rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["train_snr", "eval_snr", "pd", "n_samples", "threshold", "empirical_pfa"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _plot_pd_curves(rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    grouped: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(float(row["train_snr"]), []).append(row)

    fig, axis = plt.subplots(figsize=(10.0, 6.0))
    for train_snr, train_rows in sorted(grouped.items(), reverse=True):
        train_rows = sorted(train_rows, key=lambda item: float(item["eval_snr"]))
        axis.plot(
            [float(row["eval_snr"]) for row in train_rows],
            [float(row["pd"]) for row in train_rows],
            linewidth=1.2,
            alpha=0.75,
            label=f"train {train_snr:g} dB",
        )
    axis.set_xlabel("Evaluation SNR (dB)")
    axis.set_ylabel("Pd")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", fontsize="x-small", ncol=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ResNet classifier with a descending SNR curriculum.")
    parser.add_argument("--train-snr-start", type=int, default=30)
    parser.add_argument("--train-snr-end", type=int, default=-30)
    parser.add_argument("--train-snr-step", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=400)
    parser.add_argument("--scenarios-per-waveform", type=int, default=500)
    parser.add_argument("--generator-script", type=Path, default=DEFAULT_GENERATOR_SCRIPT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val-root", type=Path, default=DEFAULT_VAL_ROOT)
    parser.add_argument("--res-key", default="cfg512")
    parser.add_argument("--preprocessing", default="log_snr_estimated")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--pfa", type=float, default=1e-2)
    parser.add_argument("--noise-trials", type=int, default=1000)
    parser.add_argument("--noise-variance", type=float, default=1.0)
    parser.add_argument(
        "--noise-samples-per-signal",
        type=float,
        default=1.0,
        help="Number of noise-only training samples per signal sample. Noise targets are all-zero BCE vectors.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Stop training a SNR stage after this many epochs without training-loss improvement. Use 0 to disable.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-4,
        help="Minimum training-loss decrease required to reset early-stopping patience.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--eval-snr-start", type=int, default=-30)
    parser.add_argument("--eval-snr-end", type=int, default=30)
    parser.add_argument("--eval-snr-step", type=int, default=2)
    parser.add_argument("--eval-snr-values", default=None, help="Comma-separated override, e.g. '-30,-20,-10,0,10,20,30'.")
    parser.add_argument("--keep-last-dataset", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.res_key not in DEFAULT_RES_HW:
        raise ValueError(f"Unknown res_key '{args.res_key}'. Expected one of {sorted(DEFAULT_RES_HW)}.")
    if float(args.noise_samples_per_signal) < 0.0:
        raise ValueError("--noise-samples-per-signal must be non-negative.")

    generator_script = args.generator_script.resolve()
    val_root = args.val_root.resolve()
    if not generator_script.is_file():
        raise FileNotFoundError(f"Missing generator script: {generator_script}")
    if not (val_root / "manifest.json").is_file():
        raise FileNotFoundError(f"Missing validation manifest: {val_root / 'manifest.json'}")

    train_snr_values = _snr_sequence(args.train_snr_start, args.train_snr_end, args.train_snr_step)
    if len(train_snr_values) > 61:
        raise ValueError("Training SNR sequence is longer than the locked seed range 400..460.")
    eval_snr_values = _parse_snr_values(
        args.eval_snr_values,
        start=args.eval_snr_start,
        end=args.eval_snr_end,
        step=args.eval_snr_step,
    )

    device = _resolve_device(args.device)
    eval_batch_size = int(args.eval_batch_size or args.batch_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _log(f"loading validation scenarios from {val_root}")
    val_scenarios = _load_scenarios(val_root)
    if not val_scenarios:
        raise RuntimeError(f"No validation scenarios found in {val_root}")

    model = None
    class_index_to_name: dict[int, str] = {}
    class_name_to_index: dict[str, int] = {}
    all_rows: list[dict[str, Any]] = []
    training_history: list[dict[str, Any]] = []

    for step_index, train_snr in enumerate(train_snr_values):
        seed = int(args.seed_start + step_index)
        if seed > 460:
            raise ValueError(f"Seed {seed} exceeds locked maximum seed 460.")
        _log(f"=== train_snr={train_snr} dB seed={seed} ===")
        _generate_dataset(
            generator_script=generator_script,
            output_dir=args.work_dir.resolve(),
            seed=seed,
            scenarios_per_waveform=args.scenarios_per_waveform,
            python_executable=args.python_executable,
        )

        train_scenarios = _load_scenarios(args.work_dir.resolve())
        train_classes = sorted({str(scenario.class_name) for scenario in train_scenarios})
        if model is None:
            class_index_to_name, class_name_to_index = _build_class_table(train_scenarios)
            if args.num_classes is not None and int(args.num_classes) != len(class_index_to_name):
                raise ValueError(
                    f"--num-classes={args.num_classes} does not match classes produced by "
                    f"{generator_script}: {len(class_index_to_name)} classes={list(class_name_to_index)}"
                )
            model = resnet50d_classifier(num_classes=len(class_index_to_name), input_canals=1, device=device)
            _log(f"using {len(class_index_to_name)} classes: {class_index_to_name}")
        unknown_classes = [name for name in train_classes if name not in class_name_to_index]
        if unknown_classes:
            raise ValueError(f"Generated dataset contains unknown classes: {unknown_classes}")
        val_classes = sorted({str(scenario.class_name) for scenario in val_scenarios})
        unknown_val_classes = [name for name in val_classes if name not in class_name_to_index]
        if unknown_val_classes:
            raise ValueError(
                "Validation dataset contains classes absent from the generated training class table: "
                f"{unknown_val_classes}"
            )
        _log(f"training samples={len(train_scenarios)} classes={train_classes}")

        train_dataset = NoisyWaveformClassificationDataset(
            train_scenarios,
            class_name_to_index=class_name_to_index,
            num_classes=len(class_index_to_name),
            snr_db=float(train_snr),
            noise_variance=float(args.noise_variance),
            noise_samples_per_signal=float(args.noise_samples_per_signal),
            seed=seed,
            res_key=args.res_key,
            preprocessing=args.preprocessing,
        )
        epoch_history = _train_one_snr(
            model=model,
            train_dataset=train_dataset,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
        )
        training_history.append(
            {
                "train_snr": float(train_snr),
                "seed": seed,
                "n_train_samples": len(train_dataset),
                "n_signal_samples": len(train_scenarios),
                "n_noise_samples": train_dataset.noise_sample_count,
                "epochs": epoch_history,
            }
        )

        signal_length = int(len(val_scenarios[0].signal))
        calibration = _calibrate_threshold(
            model=model,
            signal_length=signal_length,
            noise_variance=float(args.noise_variance),
            pfa=float(args.pfa),
            n_trials=int(args.noise_trials),
            seed=int(seed + 1009),
            device=device,
            res_key=args.res_key,
            preprocessing=args.preprocessing,
            batch_size=eval_batch_size,
        )
        _log(
            f"threshold={calibration.threshold:.6g} empirical_pfa={calibration.empirical_pfa:.6g}"
        )
        eval_rows = _evaluate_pd_by_snr(
            model=model,
            scenarios=val_scenarios,
            eval_snr_values=eval_snr_values,
            threshold=calibration.threshold,
            noise_variance=float(args.noise_variance),
            seed=seed,
            device=device,
            res_key=args.res_key,
            preprocessing=args.preprocessing,
            batch_size=eval_batch_size,
        )
        for row in eval_rows:
            all_rows.append(
                {
                    "train_snr": float(train_snr),
                    "eval_snr": float(row["eval_snr"]),
                    "pd": float(row["pd"]),
                    "n_samples": int(row["n_samples"]),
                    "threshold": float(calibration.threshold),
                    "empirical_pfa": float(calibration.empirical_pfa),
                }
            )

        csv_path = args.output_dir / "resnet_snr_curriculum_pd.csv"
        _write_csv(all_rows, csv_path)
        _log(f"updated {csv_path}")

    payload = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "class_index_to_name": {str(key): value for key, value in class_index_to_name.items()},
        "train_snr_values": [float(v) for v in train_snr_values],
        "eval_snr_values": [float(v) for v in eval_snr_values],
        "training_history": training_history,
        "pd_rows": all_rows,
        "pd_definition": "Pd = P(max sigmoid(logits) > threshold), threshold calibrated on noise-only H0 at target Pfa. Noise-only training targets are all-zero BCE vectors.",
    }
    json_path = args.output_dir / "resnet_snr_curriculum_summary.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _log(f"wrote {json_path}")

    if not args.no_plot:
        plot_path = args.output_dir / "resnet_snr_curriculum_pd_vs_snr.png"
        _plot_pd_curves(all_rows, plot_path)
        _log(f"wrote {plot_path}")

    if args.work_dir.exists() and not args.keep_last_dataset:
        shutil.rmtree(args.work_dir)
        _log(f"removed temporary dataset {args.work_dir}")


if __name__ == "__main__":
    main()
