from __future__ import annotations

import math

import torch

from ._common import ensure_single_spectrum_2d, minmax_scale

F_E = 4.0e9
K_BOLTZMANN = 1.38e-23
STANDARD_TEMP = 290.0
RX_BW = F_E / 2.0
NOISE_POWER = K_BOLTZMANN * STANDARD_TEMP * RX_BW
PSNR_MIN = -3.0


def _infer_nperseg(tensor: torch.Tensor, cfg_key: str | None = None) -> int:
    if cfg_key and cfg_key.startswith("cfg"):
        try:
            parsed = int(cfg_key[3:])
            if parsed > 0:
                return parsed
        except ValueError:
            pass
    return max(int(tensor.shape[-2]), 1)


def preprocess_spectrogram_psnr(
    tensor: torch.Tensor,
    *,
    cfg_key: str | None = None,
    psnr_min: float = PSNR_MIN,
    psnr_max: float | None = None,
    snr_max_base: float = 20.0,
    fe: float = F_E,
    noise_power: float = NOISE_POWER,
    nfft: int | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    spectrum = ensure_single_spectrum_2d(tensor)
    power = spectrum.abs().pow(2).to(torch.float32)

    nperseg = _infer_nperseg(spectrum, cfg_key=cfg_key)
    nfft_value = nfft if nfft is not None else nperseg
    nfft_value = max(int(nfft_value), 1)

    if psnr_max is None:
        psnr_max = float(snr_max_base) + 10.0 * math.log10(max(nfft_value / 2.0, 1.0))

    noise_power_per_cell = float(noise_power) / float(fe) * (float(nperseg) / float(nfft_value**2))
    noise_power_per_cell = max(noise_power_per_cell, eps)

    psnr = 10.0 * torch.log10(power / noise_power_per_cell + eps)
    psnr = psnr.clamp(float(psnr_min), float(psnr_max))
    psnr = (psnr - float(psnr_min)) / (float(psnr_max) - float(psnr_min))
    return psnr.unsqueeze(0).to(torch.float32)


def preprocess_spectrogram_minmax(
    tensor: torch.Tensor,
    *,
    use_log: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    spectrum = ensure_single_spectrum_2d(tensor)
    values = spectrum.abs().pow(2).to(torch.float32)
    if use_log:
        values = 10.0 * torch.log10(values + eps)
    return minmax_scale(values).unsqueeze(0).to(torch.float32)
