from __future__ import annotations

import argparse
import copy
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
DEFAULT_TEST_ROOT = Path("/data/RAWSIM/RMA/rf_single_emitter_real_validation")
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


class CachedClassificationDataset(Dataset):
    def __init__(
        self,
        source: Dataset,
        *,
        dtype: torch.dtype,
        label: str,
    ) -> None:
        self.label = str(label)
        images = []
        targets = []
        _log(f"precomputing {self.label} tensor cache with {len(source)} samples")
        for index in range(len(source)):
            image, target = source[index]
            images.append(image.to(dtype=dtype).contiguous())
            targets.append(target.to(dtype=torch.float32).contiguous())
        self.images = torch.stack(images, dim=0)
        self.targets = torch.stack(targets, dim=0)
        nbytes = self.images.numel() * self.images.element_size() + self.targets.numel() * self.targets.element_size()
        _log(f"{self.label} tensor cache ready: {nbytes / (1024.0 ** 3):.2f} GiB")

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.targets[index]


def _parse_cache_dtype(raw: str) -> torch.dtype:
    normalized = str(raw).strip().lower()
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise argparse.ArgumentTypeError("--cache-dtype must be one of: float16, float32.")


def _maybe_cache_dataset(
    dataset: Dataset,
    *,
    enabled: bool,
    dtype: torch.dtype,
    label: str,
) -> Dataset:
    if not enabled:
        return dataset
    return CachedClassificationDataset(dataset, dtype=dtype, label=label)


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


def _model_state_dict(model: torch.nn.Module) -> dict:
    return model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()


def _load_model_state_dict(model: torch.nn.Module, state_dict: dict) -> None:
    target = model.module if isinstance(model, torch.nn.DataParallel) else model
    target.load_state_dict(state_dict)


def _save_resnet_checkpoint(
    *,
    model: torch.nn.Module,
    output_dir: Path,
    filename: str,
    train_snr: float,
    seed: int,
    epoch: int,
    class_index_to_name: dict[int, str],
    completed: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / filename
    state_path = output_dir / f"{Path(filename).stem}_state.json"
    torch.save(_model_state_dict(model), checkpoint_path)
    state = {
        "last_checkpoint": str(checkpoint_path),
        "train_snr": float(train_snr),
        "seed": int(seed),
        "epoch": int(epoch),
        "completed": bool(completed),
        "class_index_to_name": {str(key): value for key, value in class_index_to_name.items()},
    }
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _evaluate_classification_dataset(
    *,
    model: torch.nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    criterion: torch.nn.Module,
) -> dict[str, float]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    total_loss = 0.0
    total_signal_correct = 0
    total_signal_samples = 0
    total_noise_rejected = 0
    total_noise_samples = 0
    total_samples = 0

    model.eval()
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            targets = targets.to(device, dtype=torch.float32, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)

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

    return {
        "loss": float(total_loss / max(total_samples, 1)),
        "signal_accuracy": float(total_signal_correct / max(total_signal_samples, 1)),
        "noise_rejection_at_0_5": float(total_noise_rejected / max(total_noise_samples, 1)),
    }


def _train_one_snr(
    *,
    model: torch.nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset | None,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    num_workers: int,
    val_every: int,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    checkpoint_every: int,
    checkpoint_dir: Path,
    train_snr: float,
    seed: int,
    class_index_to_name: dict[int, str],
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
    monitor_name = "val_loss" if val_dataset is not None else "train_loss"
    best_monitor = float("inf")
    best_state_dict: dict | None = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, int(epochs) + 1):
        model.train()
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

        train_metrics = {
            "loss": float(total_loss / max(total_samples, 1)),
            "signal_accuracy": float(total_signal_correct / max(total_signal_samples, 1)),
            "noise_rejection_at_0_5": float(total_noise_rejected / max(total_noise_samples, 1)),
        }
        should_validate = val_dataset is not None and (epoch == 1 or epoch % max(1, int(val_every)) == 0)
        val_metrics = (
            _evaluate_classification_dataset(
                model=model,
                dataset=val_dataset,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
                criterion=criterion,
            )
            if should_validate
            else None
        )
        row = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "train_signal_accuracy": train_metrics["signal_accuracy"],
            "train_noise_rejection_at_0_5": train_metrics["noise_rejection_at_0_5"],
        }
        if val_metrics is not None:
            row.update(
                {
                    "val_loss": val_metrics["loss"],
                    "val_signal_accuracy": val_metrics["signal_accuracy"],
                    "val_noise_rejection_at_0_5": val_metrics["noise_rejection_at_0_5"],
                }
            )
        else:
            row["loss"] = train_metrics["loss"]
            row["signal_accuracy"] = train_metrics["signal_accuracy"]
            row["noise_rejection_at_0_5"] = train_metrics["noise_rejection_at_0_5"]
        history.append(row)
        log_message = (
            f"epoch={epoch} train_loss={row['train_loss']:.6f} "
            f"train_signal_accuracy={row['train_signal_accuracy']:.4f} "
            f"train_noise_rejection_at_0_5={row['train_noise_rejection_at_0_5']:.4f}"
        )
        if val_metrics is not None:
            log_message += (
                f" val_loss={row['val_loss']:.6f} "
                f"val_signal_accuracy={row['val_signal_accuracy']:.4f} "
                f"val_noise_rejection_at_0_5={row['val_noise_rejection_at_0_5']:.4f}"
            )
        elif val_dataset is not None:
            log_message += f" val_loss=skipped val_every={max(1, int(val_every))}"
        _log(log_message)
        if int(checkpoint_every) > 0 and epoch % int(checkpoint_every) == 0:
            _save_resnet_checkpoint(
                model=model,
                output_dir=checkpoint_dir,
                filename="last.pt",
                train_snr=float(train_snr),
                seed=int(seed),
                epoch=epoch,
                class_index_to_name=class_index_to_name,
                completed=False,
            )

        if monitor_name in row:
            monitor_value = float(row[monitor_name])
            improvement = best_monitor - monitor_value
            if improvement > float(early_stopping_min_delta) or best_state_dict is None:
                best_monitor = monitor_value
                best_epoch = epoch
                best_state_dict = copy.deepcopy(_model_state_dict(model))
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        if int(early_stopping_patience) > 0:
            if epochs_without_improvement >= int(early_stopping_patience):
                _log(
                    "early stopping: "
                    f"{monitor_name} did not improve by {float(early_stopping_min_delta):.6g} "
                    f"for {int(early_stopping_patience)} epochs"
                )
                break

    last_epoch = int(history[-1]["epoch"]) if history else 0
    _save_resnet_checkpoint(
        model=model,
        output_dir=checkpoint_dir,
        filename="last.pt",
        train_snr=float(train_snr),
        seed=int(seed),
        epoch=last_epoch,
        class_index_to_name=class_index_to_name,
        completed=True,
    )
    if best_state_dict is not None:
        _load_model_state_dict(model, best_state_dict)
        if history:
            history[-1]["restored_best_epoch"] = float(best_epoch)
            history[-1]["restored_best_monitor"] = float(best_monitor)
            history[-1]["restored_best_monitor_name"] = monitor_name
        _save_resnet_checkpoint(
            model=model,
            output_dir=checkpoint_dir,
            filename="best.pt",
            train_snr=float(train_snr),
            seed=int(seed),
            epoch=int(best_epoch),
            class_index_to_name=class_index_to_name,
            completed=True,
        )
        _log(f"restored best stage weights from epoch={best_epoch} {monitor_name}={best_monitor:.6f}")
    return history


def _prediction_scores(
    *,
    model: torch.nn.Module,
    signals: Sequence[np.ndarray],
    device: torch.device,
    res_key: str,
    preprocessing: str,
    batch_size: int,
) -> tuple[list[float], list[int]]:
    scores: list[float] = []
    predictions: list[int] = []
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
            max_scores, pred_indices = probs.max(dim=1)
            scores.extend(max_scores.detach().cpu().tolist())
            predictions.extend(pred_indices.detach().cpu().tolist())
    return [float(score) for score in scores], [int(prediction) for prediction in predictions]


def _confidence_scores(
    *,
    model: torch.nn.Module,
    signals: Sequence[np.ndarray],
    device: torch.device,
    res_key: str,
    preprocessing: str,
    batch_size: int,
) -> list[float]:
    scores, _ = _prediction_scores(
        model=model,
        signals=signals,
        device=device,
        res_key=res_key,
        preprocessing=preprocessing,
        batch_size=batch_size,
    )
    return scores


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
    class_name_to_index: dict[str, int],
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
        targets = []
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
            targets.append(int(class_name_to_index[str(scenario.class_name)]))
        scores, predictions = _prediction_scores(
            model=model,
            signals=signals,
            device=device,
            res_key=res_key,
            preprocessing=preprocessing,
            batch_size=batch_size,
        )
        decisions = np.asarray(scores, dtype=np.float64) > float(threshold)
        pred_array = np.asarray(predictions, dtype=np.int64)
        target_array = np.asarray(targets, dtype=np.int64)
        correct = pred_array == target_array
        detected_correct = correct[decisions]
        rows.append(
            {
                "eval_snr": float(snr_db),
                "pd": float(np.mean(decisions)),
                "classification_accuracy": float(np.mean(correct)) if correct.size else float("nan"),
                "classification_accuracy_detected": (
                    float(np.mean(detected_correct)) if detected_correct.size else float("nan")
                ),
                "mean_score": float(np.mean(np.asarray(scores, dtype=np.float64))) if scores else float("nan"),
                "n_samples": int(decisions.size),
                "n_detected": int(np.sum(decisions)),
            }
        )
        _log(
            f"eval_snr={float(snr_db):.2f} pd={rows[-1]['pd']:.4f} "
            f"class_acc_detected={rows[-1]['classification_accuracy_detected']:.4f}"
        )
    return rows


def _write_csv(rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "train_snr",
        "eval_snr",
        "pd",
        "classification_accuracy",
        "classification_accuracy_detected",
        "mean_score",
        "n_samples",
        "n_detected",
        "threshold",
        "empirical_pfa",
    ]
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


def _diagonal_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if abs(float(row["train_snr"]) - float(row["eval_snr"])) < 1e-9
    ]


def _plot_diagonal_pd(rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    diagonal = sorted(_diagonal_rows(rows), key=lambda item: float(item["train_snr"]))
    fig, axis = plt.subplots(figsize=(8.0, 5.0))
    axis.plot(
        [float(row["train_snr"]) for row in diagonal],
        [float(row["pd"]) for row in diagonal],
        marker="o",
        linewidth=1.5,
    )
    axis.set_xlabel("Training/evaluation SNR (dB)")
    axis.set_ylabel("Pd")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ResNet classifier with a descending SNR curriculum.")
    parser.add_argument("--train-snr-start", type=int, default=0)
    parser.add_argument("--train-snr-end", type=int, default=-30)
    parser.add_argument("--train-snr-step", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=400)
    parser.add_argument("--scenarios-per-waveform", type=int, default=500)
    parser.add_argument(
        "--first-snr-scenarios-per-waveform",
        type=int,
        default=1000,
        help="Number of scenarios per waveform for the first SNR stage. Later stages use --scenarios-per-waveform.",
    )
    parser.add_argument(
        "--val-scenarios-per-waveform",
        type=int,
        default=10,
        help="Number of clean validation scenarios generated per waveform at every SNR stage.",
    )
    parser.add_argument(
        "--stage-val-ratio",
        type=float,
        default=0.0,
        help="Deprecated; stage validation now uses --val-scenarios-per-waveform with a separate generated dataset.",
    )
    parser.add_argument("--generator-script", type=Path, default=DEFAULT_GENERATOR_SCRIPT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--test-root",
        type=Path,
        default=DEFAULT_TEST_ROOT,
        help="Clean/noiseless test dataset root. It is noised on the fly for every evaluation SNR.",
    )
    parser.add_argument(
        "--val-root",
        type=Path,
        default=None,
        help="Deprecated alias for --test-root.",
    )
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
    parser.add_argument(
        "--no-cache-stage-tensors",
        action="store_true",
        help="Disable in-memory tensor caching for train/stage validation datasets.",
    )
    parser.add_argument(
        "--cache-dtype",
        default="float16",
        choices=("float16", "float32"),
        help="Tensor dtype used by the in-memory stage cache.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--val-every",
        type=int,
        default=1,
        help="Run stage validation every N epochs. Epoch 1 is always validated when a stage holdout exists.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Stop a SNR stage after this many validation checks without validation-loss improvement. Use 0 to disable.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-4,
        help="Minimum validation-loss decrease required to reset early-stopping patience.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save an incomplete last.pt every N epochs. Use 0 to save only the restored best checkpoint at stage end.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--eval-snr-start", type=int, default=-30)
    parser.add_argument("--eval-snr-end", type=int, default=0)
    parser.add_argument("--eval-snr-step", type=int, default=1)
    parser.add_argument("--eval-snr-values", default=None, help="Comma-separated override, e.g. '-30,-20,-10,0'.")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.res_key not in DEFAULT_RES_HW:
        raise ValueError(f"Unknown res_key '{args.res_key}'. Expected one of {sorted(DEFAULT_RES_HW)}.")
    if float(args.noise_samples_per_signal) < 0.0:
        raise ValueError("--noise-samples-per-signal must be non-negative.")
    if int(args.val_scenarios_per_waveform) <= 0:
        raise ValueError("--val-scenarios-per-waveform must be strictly positive.")
    if int(args.val_every) <= 0:
        raise ValueError("--val-every must be strictly positive.")
    if int(args.checkpoint_every) < 0:
        raise ValueError("--checkpoint-every must be non-negative.")

    generator_script = args.generator_script.resolve()
    test_root = Path(args.val_root if args.val_root is not None else args.test_root).resolve()
    if not generator_script.is_file():
        raise FileNotFoundError(f"Missing generator script: {generator_script}")
    if not (test_root / "manifest.json").is_file():
        raise FileNotFoundError(f"Missing test manifest: {test_root / 'manifest.json'}")

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
    cache_stage_tensors = not bool(args.no_cache_stage_tensors)
    cache_dtype = _parse_cache_dtype(args.cache_dtype)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _log(f"loading clean test scenarios from {test_root}")
    test_scenarios = _load_scenarios(test_root)
    if not test_scenarios:
        raise RuntimeError(f"No test scenarios found in {test_root}")

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
        scenarios_per_waveform = (
            int(args.first_snr_scenarios_per_waveform)
            if step_index == 0
            else int(args.scenarios_per_waveform)
        )
        stage_work_dir = args.work_dir.resolve() / f"snr_{train_snr:+04d}"
        stage_train_dir = stage_work_dir / "train"
        stage_val_dir = stage_work_dir / "val"
        if stage_work_dir.exists():
            shutil.rmtree(stage_work_dir)
        _generate_dataset(
            generator_script=generator_script,
            output_dir=stage_train_dir,
            seed=seed,
            scenarios_per_waveform=scenarios_per_waveform,
            python_executable=args.python_executable,
        )
        _generate_dataset(
            generator_script=generator_script,
            output_dir=stage_val_dir,
            seed=int(seed + 100_003),
            scenarios_per_waveform=int(args.val_scenarios_per_waveform),
            python_executable=args.python_executable,
        )

        train_scenarios = _load_scenarios(stage_train_dir)
        stage_val_scenarios = _load_scenarios(stage_val_dir)
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
        test_classes = sorted({str(scenario.class_name) for scenario in test_scenarios})
        unknown_test_classes = [name for name in test_classes if name not in class_name_to_index]
        if unknown_test_classes:
            raise ValueError(
                "Test dataset contains classes absent from the generated training class table: "
                f"{unknown_test_classes}"
            )
        stage_unknown_classes = [
            name
            for name in sorted({str(scenario.class_name) for scenario in stage_val_scenarios})
            if name not in class_name_to_index
        ]
        if stage_unknown_classes:
            raise ValueError(f"Stage validation dataset contains unknown classes: {stage_unknown_classes}")
        _log(
            f"training samples={len(train_scenarios)} "
            f"stage_val_samples={len(stage_val_scenarios)} classes={train_classes}"
        )

        train_dataset_raw = NoisyWaveformClassificationDataset(
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
        stage_val_dataset_raw = NoisyWaveformClassificationDataset(
            stage_val_scenarios,
            class_name_to_index=class_name_to_index,
            num_classes=len(class_index_to_name),
            snr_db=float(train_snr),
            noise_variance=float(args.noise_variance),
            noise_samples_per_signal=float(args.noise_samples_per_signal),
            seed=int(seed + 200_003),
            res_key=args.res_key,
            preprocessing=args.preprocessing,
        )
        train_dataset = _maybe_cache_dataset(
            train_dataset_raw,
            enabled=cache_stage_tensors,
            dtype=cache_dtype,
            label=f"train snr={train_snr:g}",
        )
        stage_val_dataset = _maybe_cache_dataset(
            stage_val_dataset_raw,
            enabled=cache_stage_tensors,
            dtype=cache_dtype,
            label=f"stage_val snr={train_snr:g}",
        )
        stage_checkpoint_dir = args.output_dir / f"snr_{train_snr:+04d}"
        epoch_history = _train_one_snr(
            model=model,
            train_dataset=train_dataset,
            val_dataset=stage_val_dataset,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            val_every=args.val_every,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            checkpoint_every=args.checkpoint_every,
            checkpoint_dir=stage_checkpoint_dir,
            train_snr=float(train_snr),
            seed=seed,
            class_index_to_name=class_index_to_name,
        )
        training_history.append(
            {
                "train_snr": float(train_snr),
                "seed": seed,
                "scenarios_per_waveform": scenarios_per_waveform,
                "val_scenarios_per_waveform": int(args.val_scenarios_per_waveform),
                "checkpoint_dir": str(stage_checkpoint_dir),
                "best_checkpoint": str(stage_checkpoint_dir / "best.pt"),
                "last_checkpoint": str(stage_checkpoint_dir / "last.pt"),
                "n_train_samples": len(train_dataset),
                "n_stage_val_samples": len(stage_val_dataset),
                "n_fit_signal_samples": len(train_scenarios),
                "n_stage_val_signal_samples": len(stage_val_scenarios),
                "n_signal_samples": len(train_scenarios),
                "n_noise_samples": train_dataset_raw.noise_sample_count,
                "n_stage_val_noise_samples": stage_val_dataset_raw.noise_sample_count,
                "cache_stage_tensors": cache_stage_tensors,
                "cache_dtype": str(args.cache_dtype),
                "val_every": int(args.val_every),
                "checkpoint_every": int(args.checkpoint_every),
                "epochs": epoch_history,
            }
        )

        signal_length = int(len(test_scenarios[0].signal))
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
            scenarios=test_scenarios,
            class_name_to_index=class_name_to_index,
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
                    "classification_accuracy": float(row["classification_accuracy"]),
                    "classification_accuracy_detected": float(row["classification_accuracy_detected"]),
                    "mean_score": float(row["mean_score"]),
                    "n_samples": int(row["n_samples"]),
                    "n_detected": int(row["n_detected"]),
                    "threshold": float(calibration.threshold),
                    "empirical_pfa": float(calibration.empirical_pfa),
                }
            )

        csv_path = args.output_dir / "resnet_snr_curriculum_pd.csv"
        _write_csv(all_rows, csv_path)
        _log(f"updated {csv_path}")
        diagonal_csv_path = args.output_dir / "resnet_snr_curriculum_pd_diagonal.csv"
        _write_csv(_diagonal_rows(all_rows), diagonal_csv_path)
        _log(f"updated {diagonal_csv_path}")

        if stage_work_dir.exists():
            shutil.rmtree(stage_work_dir)
            _log(f"removed stage dataset {stage_work_dir}")

    payload = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "test_root": str(test_root),
        "test_sweep_definition": "The clean test dataset is reused for every eval_snr; each test signal is scaled and noised on the fly at that SNR.",
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
        diagonal_plot_path = args.output_dir / "resnet_snr_curriculum_pd_diagonal.png"
        _plot_diagonal_pd(all_rows, diagonal_plot_path)
        _log(f"wrote {diagonal_plot_path}")

    if args.work_dir.exists():
        shutil.rmtree(args.work_dir)
        _log(f"removed temporary dataset {args.work_dir}")


if __name__ == "__main__":
    main()
