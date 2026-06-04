from __future__ import annotations

from typing import Any, Callable

import torch

from ._common import ensure_chw, ensure_chw_float
from .complex_spectrum import preprocess_complex_amplitude_phase, preprocess_complex_real_imag
from .spectrogram import preprocess_spectrogram_minmax, preprocess_spectrogram_psnr

Preprocessor = Callable[[torch.Tensor], torch.Tensor]

_ALIASES = {
    None: "spectrogram_psnr",
    "none": "none",
    "identity": "none",
    "psnr": "spectrogram_psnr",
    "spectrogram_psnr": "spectrogram_psnr",
    "minmax": "spectrogram_minmax",
    "spectrogram_minmax": "spectrogram_minmax",
    "minmax_log": "spectrogram_minmax_log",
    "spectrogram_minmax_log": "spectrogram_minmax_log",
    "real_imag": "complex_real_imag",
    "complex_real_imag": "complex_real_imag",
    "amplitude_phase": "complex_amplitude_phase",
    "complex_amplitude_phase": "complex_amplitude_phase",
}

_CHANNELS = {
    "none": 1,
    "spectrogram_psnr": 1,
    "spectrogram_minmax": 1,
    "spectrogram_minmax_log": 1,
    "complex_real_imag": 2,
    "complex_amplitude_phase": 2,
}


def _resolve_name(name: str | None) -> str:
    key = name.lower() if isinstance(name, str) else name
    resolved = _ALIASES.get(key)
    if resolved is None:
        raise ValueError(f"Unknown preprocessing '{name}'. Expected one of {sorted(v for v in _CHANNELS)}.")
    return resolved


def preprocessing_num_channels(name: str | None) -> int:
    return _CHANNELS[_resolve_name(name)]


def build_preprocessor(
    name: str | None = "spectrogram_psnr",
    preprocessing_kwargs: dict[str, Any] | None = None,
) -> Callable[..., torch.Tensor]:
    resolved = _resolve_name(name)
    kwargs = dict(preprocessing_kwargs or {})

    if resolved == "none":
        return lambda tensor, **_: ensure_chw(tensor)
    if resolved == "spectrogram_psnr":
        return lambda tensor, **extra: preprocess_spectrogram_psnr(tensor, **kwargs, **extra)
    if resolved == "spectrogram_minmax":
        minmax_kwargs = dict(kwargs)
        minmax_kwargs.pop("use_log", None)
        return lambda tensor, **_: preprocess_spectrogram_minmax(tensor, use_log=False, **minmax_kwargs)
    if resolved == "spectrogram_minmax_log":
        minmax_kwargs = dict(kwargs)
        minmax_kwargs.pop("use_log", None)
        return lambda tensor, **_: preprocess_spectrogram_minmax(tensor, use_log=True, **minmax_kwargs)
    if resolved == "complex_real_imag":
        return lambda tensor, **_: preprocess_complex_real_imag(tensor)
    if resolved == "complex_amplitude_phase":
        return lambda tensor, **_: preprocess_complex_amplitude_phase(tensor)
    raise ValueError(f"Unhandled preprocessing '{resolved}'.")


def preprocess_tensor(
    tensor: torch.Tensor,
    name: str | None = "spectrogram_psnr",
    preprocessing_kwargs: dict[str, Any] | None = None,
    **extra: Any,
) -> torch.Tensor:
    return build_preprocessor(name, preprocessing_kwargs)(tensor, **extra)
