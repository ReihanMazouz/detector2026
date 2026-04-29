from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .base import DetectorResult


def _haar_qmf_decompose(signal: np.ndarray, levels: int) -> Iterable[np.ndarray]:
    current = np.asarray(signal).reshape(-1)
    if current.size < 2:
        raise ValueError("Signal must contain at least 2 samples for QMF decomposition.")

    sqrt2 = np.sqrt(2.0)
    for _ in range(levels):
        n = current.size - (current.size % 2)
        if n < 2:
            break
        trimmed = current[:n]
        even = trimmed[0::2]
        odd = trimmed[1::2]
        approx = (even + odd) / sqrt2
        detail = (even - odd) / sqrt2
        yield detail
        current = approx


@dataclass
class QMFDetector:
    name: str = "qmf"
    levels: int = 6

    def detect(self, signal: np.ndarray, *, pfa: float, noise_variance: float) -> DetectorResult:
        x = np.asarray(signal).reshape(-1)
        if x.size < 2:
            raise ValueError("Signal must be non-empty.")
        if noise_variance <= 0.0:
            raise ValueError("noise_variance must be strictly positive.")

        band_statistics = []
        for detail in _haar_qmf_decompose(x, self.levels):
            if detail.size == 0:
                continue
            band_statistics.append(float(np.mean(np.abs(detail) ** 2) / noise_variance))

        if not band_statistics:
            raise ValueError("QMF decomposition produced no detail band.")

        statistic = max(band_statistics)
        threshold = float("nan")
        return DetectorResult(
            statistic=statistic,
            threshold=threshold,
            decision=False,
            metadata={"n_bands": float(len(band_statistics))},
        )
