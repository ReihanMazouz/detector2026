from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import DetectorResult


@dataclass
class TimeFrequencyGLRTDetector:
    name: str = "tf_glrt"
    n_fft: int = 512
    hop: int | None = None
    positive_frequencies_only: bool = True
    exclude_dc: bool = True

    def detect(self, signal: np.ndarray, *, pfa: float, noise_variance: float) -> DetectorResult:
        x = np.asarray(signal).reshape(-1)
        if x.size == 0:
            raise ValueError("Signal must be non-empty.")
        if self.n_fft <= 0:
            raise ValueError("n_fft must be strictly positive.")
        if noise_variance <= 0.0:
            raise ValueError("noise_variance must be strictly positive.")
        if not 0.0 < pfa < 1.0:
            raise ValueError("pfa must be in (0, 1).")

        hop = int(self.hop if self.hop is not None else self.n_fft)
        if hop <= 0:
            raise ValueError("hop must be strictly positive.")

        is_complex = np.iscomplexobj(x)
        frame_dtype = np.result_type(x.dtype, np.complex64 if is_complex else np.float32)
        if x.size < self.n_fft:
            frames = np.zeros((1, self.n_fft), dtype=frame_dtype)
            frames[0, : x.size] = x
        else:
            n_frames = 1 + (x.size - self.n_fft) // hop
            starts = hop * np.arange(n_frames)
            frames = np.stack([x[start : start + self.n_fft] for start in starts], axis=0).astype(frame_dtype, copy=False)

        if is_complex:
            spectrum = np.fft.fft(frames, n=self.n_fft, axis=-1)
        else:
            spectrum = np.fft.rfft(frames, n=self.n_fft, axis=-1)
        if self.positive_frequencies_only and is_complex:
            spectrum = spectrum[:, : max(1, self.n_fft // 2)]
        elif self.positive_frequencies_only:
            spectrum = spectrum[:, : max(1, self.n_fft // 2)]
        if self.exclude_dc and spectrum.shape[1] > 1:
            spectrum = spectrum[:, 1:]
        if spectrum.size == 0:
            raise ValueError("No time-frequency cells available for GLRT.")

        normalized_cell_energies = (np.abs(spectrum) ** 2) / (float(self.n_fft) * float(noise_variance))
        statistic = float(np.max(normalized_cell_energies))
        n_cells = int(normalized_cell_energies.size)
        threshold = float(-np.log(1.0 - (1.0 - pfa) ** (1.0 / n_cells)))

        return DetectorResult(
            statistic=statistic,
            threshold=threshold,
            decision=statistic > threshold,
            metadata={
                "n_fft": float(self.n_fft),
                "hop": float(hop),
                "n_cells": float(n_cells),
            },
        )
