from __future__ import annotations

import torch

from ._common import ensure_single_spectrum_2d


def preprocess_complex_real_imag(tensor: torch.Tensor) -> torch.Tensor:
    spectrum = ensure_single_spectrum_2d(tensor)

    if torch.is_complex(spectrum):
        real = spectrum.real
        imag = spectrum.imag
    else:
        real = spectrum
        imag = torch.zeros_like(spectrum)

    return torch.stack((real, imag), dim=0).to(torch.float32)


def preprocess_complex_amplitude_phase(tensor: torch.Tensor) -> torch.Tensor:
    spectrum = ensure_single_spectrum_2d(tensor)

    amplitude = spectrum.abs().to(torch.float32)
    if torch.is_complex(spectrum):
        phase = torch.angle(spectrum).to(torch.float32)
    else:
        phase = torch.zeros_like(amplitude)

    return torch.stack((amplitude, phase), dim=0)
