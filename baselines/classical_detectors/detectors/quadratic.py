from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from .base import DetectorResult


@dataclass
class QuadraticDetector:
    name: str = "quadratic"

    def detect(self, signal: np.ndarray, *, pfa: float, noise_variance: float) -> DetectorResult:
        x = np.asarray(signal).reshape(-1)
        if x.size == 0:
            raise ValueError("Signal must be non-empty.")
        if noise_variance <= 0.0:
            raise ValueError("noise_variance must be strictly positive.")
        if not 0.0 < pfa < 1.0:
            raise ValueError("pfa must be in (0, 1).")

        statistic = float(np.sum(np.abs(x) ** 2) / noise_variance)
        threshold = float(0.5 * chi2.isf(pfa, df=2 * x.size))
        return DetectorResult(
            statistic=statistic,
            threshold=threshold,
            decision=statistic > threshold,
            metadata={"n_samples": float(x.size)},
        )
