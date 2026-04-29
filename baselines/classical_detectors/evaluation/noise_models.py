from __future__ import annotations

from dataclasses import dataclass

import numpy as np


K_BOLTZMANN = 1.38e-23
STANDARD_TEMP = 290.0


@dataclass(frozen=True)
class ThermalNoiseModel:
    fe: float
    temperature_k: float = STANDARD_TEMP
    boltzmann_constant: float = K_BOLTZMANN

    @property
    def bandwidth_hz(self) -> float:
        return self.fe / 2.0

    @property
    def noise_power(self) -> float:
        return self.boltzmann_constant * self.temperature_k * self.bandwidth_hz

    @property
    def sample_variance(self) -> float:
        return self.noise_power / self.fe

    @property
    def sample_std(self) -> float:
        return float(np.sqrt(self.sample_variance))

    def draw(self, n_samples: int, *, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(loc=0.0, scale=self.sample_std, size=int(n_samples)).astype(np.float64, copy=False)


def draw_complex_awgn(n_samples: int, *, noise_variance: float, rng: np.random.Generator) -> np.ndarray:
    if noise_variance <= 0.0:
        raise ValueError("noise_variance must be strictly positive.")
    component_std = float(np.sqrt(noise_variance / 2.0))
    real = rng.normal(loc=0.0, scale=component_std, size=int(n_samples))
    imag = rng.normal(loc=0.0, scale=component_std, size=int(n_samples))
    return (real + 1j * imag).astype(np.complex64, copy=False)


def draw_real_awgn(n_samples: int, *, noise_variance: float, rng: np.random.Generator) -> np.ndarray:
    if noise_variance <= 0.0:
        raise ValueError("noise_variance must be strictly positive.")
    sample_std = float(np.sqrt(noise_variance))
    return rng.normal(loc=0.0, scale=sample_std, size=int(n_samples)).astype(np.float32, copy=False)
