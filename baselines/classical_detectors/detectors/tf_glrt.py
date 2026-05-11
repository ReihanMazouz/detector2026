from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import DetectorResult


def _time_frequency_normalized_energies(
    signal: np.ndarray,
    *,
    n_fft: int,
    hop: int | None,
    positive_frequencies_only: bool,
    exclude_dc: bool,
    noise_variance: float,
) -> tuple[np.ndarray, int]:
    x = np.asarray(signal).reshape(-1)
    if x.size == 0:
        raise ValueError("Signal must be non-empty.")
    if n_fft <= 0:
        raise ValueError("n_fft must be strictly positive.")
    if noise_variance <= 0.0:
        raise ValueError("noise_variance must be strictly positive.")

    resolved_hop = int(hop if hop is not None else n_fft)
    if resolved_hop <= 0:
        raise ValueError("hop must be strictly positive.")

    is_complex = np.iscomplexobj(x)
    frame_dtype = np.result_type(x.dtype, np.complex64 if is_complex else np.float32)
    if x.size < n_fft:
        frames = np.zeros((1, n_fft), dtype=frame_dtype)
        frames[0, : x.size] = x
    else:
        n_frames = 1 + (x.size - n_fft) // resolved_hop
        starts = resolved_hop * np.arange(n_frames)
        frames = np.stack([x[start : start + n_fft] for start in starts], axis=0).astype(frame_dtype, copy=False)

    if is_complex:
        spectrum = np.fft.fft(frames, n=n_fft, axis=-1)
    else:
        spectrum = np.fft.rfft(frames, n=n_fft, axis=-1)
    if positive_frequencies_only and is_complex:
        spectrum = spectrum[:, : max(1, n_fft // 2)]
    elif positive_frequencies_only:
        spectrum = spectrum[:, : max(1, n_fft // 2)]
    if exclude_dc and spectrum.shape[1] > 1:
        spectrum = spectrum[:, 1:]
    if spectrum.size == 0:
        raise ValueError("No time-frequency cells available for GLRT.")

    normalized_cell_energies = (np.abs(spectrum) ** 2) / (float(n_fft) * float(noise_variance))
    return normalized_cell_energies, resolved_hop


@dataclass
class TimeFrequencyGLRTDetector:
    name: str = "tf_glrt"
    n_fft: int = 512
    hop: int | None = None
    positive_frequencies_only: bool = True
    exclude_dc: bool = True

    def detect(self, signal: np.ndarray, *, pfa: float, noise_variance: float) -> DetectorResult:
        if not 0.0 < pfa < 1.0:
            raise ValueError("pfa must be in (0, 1).")

        normalized_cell_energies, hop = _time_frequency_normalized_energies(
            signal,
            n_fft=self.n_fft,
            hop=self.hop,
            positive_frequencies_only=self.positive_frequencies_only,
            exclude_dc=self.exclude_dc,
            noise_variance=noise_variance,
        )
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


@dataclass
class GridTimeFrequencyGLRTDetector:
    name: str = "grid_tf_glrt"
    grid_size: int = 4
    n_fft: int = 512
    hop: int | None = None
    positive_frequencies_only: bool = True
    exclude_dc: bool = True

    def block_statistics(self, signal: np.ndarray, *, noise_variance: float) -> tuple[list[dict[str, Any]], int]:
        if self.grid_size <= 0:
            raise ValueError("grid_size must be strictly positive.")

        normalized_cell_energies, hop = _time_frequency_normalized_energies(
            signal,
            n_fft=self.n_fft,
            hop=self.hop,
            positive_frequencies_only=self.positive_frequencies_only,
            exclude_dc=self.exclude_dc,
            noise_variance=noise_variance,
        )

        time_blocks = np.array_split(np.arange(normalized_cell_energies.shape[0]), self.grid_size)
        freq_blocks = np.array_split(np.arange(normalized_cell_energies.shape[1]), self.grid_size)
        block_statistics = []
        for time_bin, time_idx in enumerate(time_blocks):
            if time_idx.size == 0:
                continue
            for freq_bin, freq_idx in enumerate(freq_blocks):
                if freq_idx.size == 0:
                    continue
                block = normalized_cell_energies[np.ix_(time_idx, freq_idx)]
                block_statistics.append(
                    {
                        "grid_size": int(self.grid_size),
                        "time_bin": int(time_bin),
                        "freq_bin": int(freq_bin),
                        "time_start": int(time_idx[0]),
                        "time_stop": int(time_idx[-1] + 1),
                        "freq_start": int(freq_idx[0]),
                        "freq_stop": int(freq_idx[-1] + 1),
                        "n_cells": int(block.size),
                        "statistic": float(np.sum(block)),
                    }
                )

        if not block_statistics:
            raise ValueError("No non-empty grid block available for GLRT.")

        return block_statistics, hop

    def detect(self, signal: np.ndarray, *, pfa: float, noise_variance: float) -> DetectorResult:
        if not 0.0 < pfa < 1.0:
            raise ValueError("pfa must be in (0, 1).")

        block_statistics, hop = self.block_statistics(signal, noise_variance=noise_variance)
        statistic = float(max(block["statistic"] for block in block_statistics))
        # The script calibrates this detector empirically because grid blocks
        # may have unequal sizes. This analytic value is only a usable fallback.
        block_sizes = [int(block["n_cells"]) for block in block_statistics]
        max_block_size = int(max(block_sizes))
        threshold = float(max_block_size * -np.log(1.0 - (1.0 - pfa) ** (1.0 / len(block_statistics))))

        return DetectorResult(
            statistic=statistic,
            threshold=threshold,
            decision=statistic > threshold,
            metadata={
                "grid_size": float(self.grid_size),
                "n_fft": float(self.n_fft),
                "hop": float(hop),
                "n_blocks": float(len(block_statistics)),
                "min_block_cells": float(min(block_sizes)),
                "max_block_cells": float(max_block_size),
            },
        )
