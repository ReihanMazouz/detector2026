from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol

import numpy as np


@dataclass(frozen=True)
class DetectorResult:
    statistic: float
    threshold: float
    decision: bool
    metadata: Dict[str, float]


class SignalDetector(Protocol):
    name: str

    def detect(self, signal: np.ndarray, *, pfa: float, noise_variance: float) -> DetectorResult:
        ...
