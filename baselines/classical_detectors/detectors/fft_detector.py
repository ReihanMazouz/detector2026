from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import DetectorResult


@dataclass
class FFTDetector:
    name: str = "fft_1d"
    exclude_edges: bool = True

    def detect(self, signal: np.ndarray, *, pfa: float, noise_variance: float) -> DetectorResult:
        x = np.asarray(signal).reshape(-1)
        if x.size < 4:
            raise ValueError("Signal must contain at least 4 samples for FFT detection.")
        if noise_variance <= 0.0:
            raise ValueError("noise_variance must be strictly positive.")
        if not 0.0 < pfa < 1.0:
            raise ValueError("pfa must be in (0, 1).")

        spectrum = np.fft.fft(x) if np.iscomplexobj(x) else np.fft.rfft(x)
        if self.exclude_edges and spectrum.size > 2:
            active_bins = spectrum[1:-1]
        else:
            active_bins = spectrum
        if active_bins.size == 0:
            raise ValueError("No FFT bins available after edge exclusion.")

        normalized_bin_energies = (np.abs(active_bins) ** 2) / (x.size * noise_variance)
        statistic = float(np.max(normalized_bin_energies))
        threshold = float(-np.log(1.0 - (1.0 - pfa) ** (1.0 / active_bins.size)))

        return DetectorResult(
            statistic=statistic,
            threshold=threshold,
            decision=statistic > threshold,
            metadata={"n_bins": float(active_bins.size)},
        )
