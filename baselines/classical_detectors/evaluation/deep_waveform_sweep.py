from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

import numpy as np
import torch

from baselines.classical_detectors.io import WaveformManifestDataset
from baselines.classical_detectors.evaluation.noise_models import draw_real_awgn


DEFAULT_RES_HW = {
    "cfg128": (64, 1024),
    "cfg256": (128, 512),
    "cfg512": (256, 256),
    "cfg1024": (512, 128),
    "cfg2048": (1024, 64),
}


YOLO11_WIDTH_MULT = {"n": 0.25, "s": 0.50, "m": 1.00}
MR_WIDTH_MULT = {"n": 0.25, "s": 0.50, "m": 1.00}

F_E = 4.0e9
K_BOLTZMANN = 1.38e-23
STANDARD_TEMP = 290.0
NOISE_POWER = K_BOLTZMANN * STANDARD_TEMP * (F_E / 2.0)
PSNR_MIN = -3.0
_INTERIOR_LOG_NOISE_MEAN_DB = -2.5068150173550796
_EDGE_LOG_NOISE_MEAN_DB = -5.516462134585023
_EDGE_LOG_NOISE_STD_RATIO = math.sqrt(3.0)
_EDGE_LOG_NOISE_SCALE = 1.0 / _EDGE_LOG_NOISE_STD_RATIO
_EDGE_LOG_NOISE_SHIFT_DB = _INTERIOR_LOG_NOISE_MEAN_DB - _EDGE_LOG_NOISE_SCALE * _EDGE_LOG_NOISE_MEAN_DB
_DEFAULT_NOISE_EST_QUANTILE = 0.10


@dataclass(frozen=True)
class DeepModelSpec:
    name: str
    family: str
    weights_path: Path
    scale: str = "n"
    res_key: str = "cfg512"
    res_keys: tuple[str, ...] = ("cfg128", "cfg256", "cfg512", "cfg1024", "cfg2048")
    preprocessing: str = "log_snr_estimated"
    num_classes: int = 20
    reg_max: int = 16
    conf_floor: float = 1e-6
    iou_thres: float = 0.1
    iou_same_box: float = 0.9
    backbone_mode: str = "F"
    outfusion_channels_mult: int = 1


@dataclass(frozen=True)
class DeepWaveformSweepConfig:
    dataset_root: Path
    models: tuple[DeepModelSpec, ...]
    pfa: float = 1e-3
    noise_trials: int = 1000
    seed: int = 444
    noise_variance: float = 1.0
    snr_values_db: tuple[float, ...] = tuple(range(-20, 21, 2))
    waveforms: tuple[str, ...] = ()
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    progress_log: Callable[[str], None] | None = None


def _noise_seed(seed: int, scenario_index: int) -> int:
    return int(seed + 1_000_003 * scenario_index)


def _scale_for_target_snr(signal: np.ndarray, *, noise_variance: float, snr_db: float, duration_samples: int) -> float:
    energy = float(np.sum(np.abs(signal) ** 2))
    if energy <= 0.0 or int(duration_samples) <= 0:
        return 0.0
    target_energy = float(noise_variance) * float(duration_samples) * 10.0 ** (float(snr_db) / 10.0)
    return float(np.sqrt(target_energy / energy))


def _stft_spectrum(signal: np.ndarray, *, res_key: str) -> torch.Tensor:
    if res_key not in DEFAULT_RES_HW:
        raise ValueError(f"Unknown res_key '{res_key}'. Expected one of {sorted(DEFAULT_RES_HW)}.")
    target_h, target_w = DEFAULT_RES_HW[res_key]
    n_fft = int(res_key.replace("cfg", ""))
    raw = np.asarray(signal).reshape(-1)
    is_complex = np.iscomplexobj(raw)
    x = torch.as_tensor(raw, dtype=torch.complex64 if is_complex else torch.float32).reshape(-1)
    hop = n_fft
    if x.numel() < n_fft:
        frames = torch.zeros((1, n_fft), dtype=x.dtype)
        frames[0, : x.numel()] = x
    else:
        frames = x.unfold(0, n_fft, hop)
    if is_complex:
        spec = torch.fft.fft(frames, n=n_fft, dim=-1).transpose(0, 1)
    else:
        spec = torch.fft.rfft(frames, n=n_fft, dim=-1).transpose(0, 1)
    spec = spec[:target_h, :target_w]
    if spec.shape[0] < target_h or spec.shape[1] < target_w:
        padded = torch.zeros((target_h, target_w), dtype=torch.complex64)
        padded[: spec.shape[0], : spec.shape[1]] = spec
        spec = padded
    return spec


def _spectrogram_interior_power_values(power: torch.Tensor, *, exclude_dc: bool = True) -> torch.Tensor:
    values = power
    if exclude_dc and values.shape[0] > 1:
        values = values[1:]
    flattened = values.reshape(-1)
    return flattened if flattened.numel() > 0 else power.reshape(-1)


def _estimate_noise_power_per_cell_from_spectrum(
    spectrum: torch.Tensor,
    *,
    noise_est_quantile: float = _DEFAULT_NOISE_EST_QUANTILE,
    exclude_dc: bool = True,
    min_value: float = float(torch.finfo(torch.float32).tiny),
) -> float:
    if not 0.0 < float(noise_est_quantile) < 1.0:
        raise ValueError("noise_est_quantile must be in the open interval (0, 1).")
    power = spectrum.abs().pow(2).to(torch.float32)
    values = _spectrogram_interior_power_values(power, exclude_dc=exclude_dc)
    num_values = int(values.numel())
    if num_values == 0:
        return float(min_value)
    quantile_rank = max(1, min(num_values, int(math.ceil(float(noise_est_quantile) * num_values))))
    quantile_value = values.kthvalue(quantile_rank).values
    sigma2_hat = float(quantile_value.item()) / (-math.log1p(-float(noise_est_quantile)))
    return max(sigma2_hat, float(min_value))


def _preprocess_spectrogram_log_snr_estimated(
    tensor: torch.Tensor,
    *,
    noise_est_quantile: float = _DEFAULT_NOISE_EST_QUANTILE,
    exclude_dc_from_noise_est: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    if tensor.ndim == 3:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D spectrum, got shape {tuple(tensor.shape)}.")
    power = tensor.abs().pow(2).to(torch.float32)
    noise_power_per_cell = _estimate_noise_power_per_cell_from_spectrum(
        tensor,
        noise_est_quantile=noise_est_quantile,
        exclude_dc=exclude_dc_from_noise_est,
    )
    log_snr = 10.0 * torch.log10(power / noise_power_per_cell + eps)
    if log_snr.shape[0] >= 1:
        log_snr[0] = _EDGE_LOG_NOISE_SCALE * log_snr[0] + _EDGE_LOG_NOISE_SHIFT_DB
    return log_snr.unsqueeze(0).to(torch.float32)


def _preprocess_spectrogram_psnr(
    tensor: torch.Tensor,
    *,
    res_key: str,
    psnr_min: float = PSNR_MIN,
    snr_max_base: float = 20.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    if tensor.ndim == 3:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D spectrum, got shape {tuple(tensor.shape)}.")
    power = tensor.abs().pow(2).to(torch.float32)
    nperseg = int(res_key.replace("cfg", "")) if res_key.startswith("cfg") else int(tensor.shape[-2])
    nfft = max(nperseg, 1)
    psnr_max = float(snr_max_base) + 10.0 * np.log10(max(nfft / 2.0, 1.0))
    noise_power_per_cell = float(NOISE_POWER) / float(F_E) * (float(nperseg) / float(nfft**2))
    noise_power_per_cell = max(noise_power_per_cell, eps)
    psnr = 10.0 * torch.log10(power / noise_power_per_cell + eps)
    psnr = psnr.clamp(float(psnr_min), float(psnr_max))
    psnr = (psnr - float(psnr_min)) / (float(psnr_max) - float(psnr_min))
    return psnr.unsqueeze(0).to(torch.float32)


def _preprocess_minmax(tensor: torch.Tensor, *, use_log: bool = False, eps: float = 1e-12) -> torch.Tensor:
    if tensor.ndim == 3:
        tensor = tensor.squeeze(0)
    values = tensor.abs().pow(2).to(torch.float32)
    if use_log:
        values = 10.0 * torch.log10(values + eps)
    min_value = values.amin()
    max_value = values.amax()
    return ((values - min_value) / (max_value - min_value + eps)).unsqueeze(0).to(torch.float32)


def _preprocess_tensor(tensor: torch.Tensor, *, name: str, res_key: str) -> torch.Tensor:
    if name in {"log_snr_estimated", "spectrogram_log_snr_estimated"}:
        return _preprocess_spectrogram_log_snr_estimated(tensor)
    if name in {"spectrogram_psnr", "psnr"}:
        return _preprocess_spectrogram_psnr(tensor, res_key=res_key)
    if name in {"spectrogram_minmax", "minmax"}:
        return _preprocess_minmax(tensor, use_log=False)
    if name in {"spectrogram_minmax_log", "minmax_log"}:
        return _preprocess_minmax(tensor, use_log=True)
    if name in {"none", "identity"}:
        return tensor.unsqueeze(0).to(torch.float32) if tensor.ndim == 2 else tensor.to(torch.float32)
    raise ValueError("Deep waveform evaluation currently supports log_snr_estimated, psnr, minmax, minmax_log, and none.")


def _preprocessing_num_channels(name: str) -> int:
    if name in {"log_snr_estimated", "spectrogram_log_snr_estimated", "spectrogram_psnr", "psnr", "spectrogram_minmax", "minmax", "spectrogram_minmax_log", "minmax_log", "none", "identity"}:
        return 1
    raise ValueError("Deep waveform evaluation currently supports one-channel spectrogram preprocessing only.")


class _DeepDetector:
    def __init__(self, spec: DeepModelSpec, *, device: str) -> None:
        self.spec = spec
        self.device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
        input_channels = _preprocessing_num_channels(spec.preprocessing)

        if spec.family == "yolov11":
            from core.models.yolov11 import YOLOv11

            res_hw = DEFAULT_RES_HW[spec.res_key]
            self.model = YOLOv11(
                output_dir=str(spec.weights_path.parent),
                num_classes=spec.num_classes,
                reg_max=spec.reg_max,
                device=str(self.device),
                input_canals=input_channels,
                width_mult=YOLO11_WIDTH_MULT[spec.scale],
                input_hw=res_hw,
            )
            self.input_res_keys = (spec.res_key,)
        elif spec.family == "mr_yolo":
            from core.models.mr_yolo import MR_YOLO

            input_resolutions = [DEFAULT_RES_HW[key] for key in spec.res_keys]
            self.model = MR_YOLO(
                input_resolutions=input_resolutions,
                output_dir=str(spec.weights_path.parent),
                num_classes=spec.num_classes,
                reg_max=spec.reg_max,
                device=str(self.device),
                in_ch=input_channels,
                width_mult=MR_WIDTH_MULT[spec.scale],
                backbone_mode=spec.backbone_mode,
                outfusion_channels_mult=spec.outfusion_channels_mult,
            )
            self.input_res_keys = tuple(spec.res_keys)
        else:
            raise ValueError("Deep model family must be 'yolov11' or 'mr_yolo'.")

        if not spec.weights_path.is_file():
            raise FileNotFoundError(f"Missing weights for {spec.name}: {spec.weights_path}")
        self.model.load_weights(str(spec.weights_path), device=str(self.device), eval_mode=True)
        self.model.eval()

    def _make_inputs(self, signal: np.ndarray) -> torch.Tensor | list[torch.Tensor]:
        images = []
        for res_key in self.input_res_keys:
            spec = _stft_spectrum(signal, res_key=res_key)
            image = _preprocess_tensor(spec, name=self.spec.preprocessing, res_key=res_key).unsqueeze(0).to(self.device, dtype=torch.float32)
            images.append(image)
        return images[0] if len(images) == 1 else images

    def statistic(self, signal: np.ndarray) -> float:
        inputs = self._make_inputs(signal)
        with torch.no_grad():
            dist_out, cls_out = self.model(inputs)
            processed = self.model.postprocess(
                dist_out,
                cls_out,
                dist_out,
                conf_thres=self.spec.conf_floor,
                iou_thres=self.spec.iou_thres,
                iou_same_box=self.spec.iou_same_box,
            )
        detections = processed[0] if processed else None
        if detections is None or len(detections) == 0:
            return 0.0
        return float(detections[:, 4].detach().max().cpu())


def _calibrate(detector: _DeepDetector, *, signal_length: int, noise_variance: float, pfa: float, n_trials: int, seed: int) -> dict:
    if int(n_trials) < int(np.ceil(1.0 / float(pfa))):
        raise ValueError("n_trials must be at least ceil(1 / pfa) to control empirical Pfa.")
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(int(n_trials)):
        noise = draw_real_awgn(signal_length, noise_variance=noise_variance, rng=rng)
        stats.append(detector.statistic(noise))
    values = np.asarray(stats, dtype=np.float64)
    allowed_false_alarms = int(np.floor(float(pfa) * float(values.size)))
    sorted_values = np.sort(values)
    threshold_index = max(0, int(values.size) - allowed_false_alarms - 1)
    threshold = float(sorted_values[threshold_index])
    return {
        "threshold": threshold,
        "empirical_pfa": float(np.mean(values > threshold)),
        "allowed_false_alarms": allowed_false_alarms,
        "observed_false_alarms": int(np.sum(values > threshold)),
        "mean_statistic_h0": float(np.mean(values)),
        "std_statistic_h0": float(np.std(values)),
        "n_trials": int(n_trials),
    }


def _summarize(rows: list[dict], x_key: str) -> list[dict]:
    grouped: dict[tuple[str, float], list[bool]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["waveform_label"]), float(row[x_key]))].append(bool(row["decision"]))
    return [
        {"waveform_label": waveform, x_key: x_value, "pd": float(np.mean(decisions)), "n_samples": len(decisions)}
        for (waveform, x_value), decisions in sorted(grouped.items())
    ]


def _group_scenarios_by_waveform(scenarios: list[Any]) -> dict[str, list[tuple[int, Any]]]:
    grouped: dict[str, list[tuple[int, Any]]] = defaultdict(list)
    for scenario_index, scenario in enumerate(scenarios):
        grouped[str(scenario.waveform_label)].append((scenario_index, scenario))
    return dict(grouped)


def _log(config: DeepWaveformSweepConfig, message: str) -> None:
    if config.progress_log is not None:
        config.progress_log(message)


def run_deep_waveform_snr_sweep(config: DeepWaveformSweepConfig) -> Dict[str, Any]:
    selected_waveforms = set(config.waveforms) if config.waveforms else None
    dataset = WaveformManifestDataset(config.dataset_root, waveforms=selected_waveforms)
    scenarios = list(dataset)
    signal_length = int(scenarios[0].signal.size)

    payload: Dict[str, Any] = {
        "dataset_root": str(config.dataset_root),
        "pfa_target": float(config.pfa),
        "seed": int(config.seed),
        "signal_length": signal_length,
        "n_scenarios": len(scenarios),
        "signal_domain": "real_1d",
        "snr_definition": "(sum(|s|^2) / duration_samples) / noise_variance",
        "noise_seed_policy": "same noise realization for each scenario across all SNR levels",
        "noise_variance": float(config.noise_variance),
        "snr_values_db": [float(v) for v in config.snr_values_db],
        "detectors": {},
    }

    for model_index, spec in enumerate(config.models):
        _log(config, f"[deep:{spec.name}] loading model family={spec.family} weights={spec.weights_path}")
        detector = _DeepDetector(spec, device=config.device)
        _log(config, f"[deep:{spec.name}] calibrating confidence threshold on H0, trials={config.noise_trials}")
        calibration = _calibrate(
            detector,
            signal_length=signal_length,
            noise_variance=float(config.noise_variance),
            pfa=config.pfa,
            n_trials=config.noise_trials,
            seed=int(config.seed + 1009 * model_index),
        )
        threshold = float(calibration["threshold"])
        _log(
            config,
            f"[deep:{spec.name}] threshold={threshold:.6g}, empirical_pfa={calibration['empirical_pfa']:.6g}",
        )
        by_snr = []
        grouped_scenarios = _group_scenarios_by_waveform(scenarios)

        _log(config, f"[deep:{spec.name}] sweep Pd vs SNR")
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
                    statistic = detector.statistic(scale * scenario.signal + noise)
                    decisions.append(bool(statistic > threshold))
                pd = float(np.mean(np.asarray(decisions, dtype=np.float64)))
                by_snr.append(
                    {
                        "waveform_label": waveform_label,
                        "snr_db": float(snr_db),
                        "pd": pd,
                        "n_samples": int(len(decisions)),
                        "early_stopped": bool(pd == 0.0),
                    }
                )
                _log(config, f"[deep:{spec.name}] waveform={waveform_label} snr_db={snr_db:.2f} pd={pd:.4f}")
                if pd == 0.0:
                    _log(config, f"[deep:{spec.name}] waveform={waveform_label} stop SNR sweep at {snr_db:.2f} dB")
                    break

        payload["detectors"][spec.name] = {
            "family": spec.family,
            "weights_path": str(spec.weights_path),
            "threshold": calibration,
            "by_snr": by_snr,
        }
        _log(config, f"[deep:{spec.name}] done")

    return payload
