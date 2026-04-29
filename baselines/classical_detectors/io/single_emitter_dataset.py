from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch


@dataclass(frozen=True)
class SignalSample:
    sample_id: str
    signal: np.ndarray
    snr_db: float
    n_labels: int
    split: str


class SingleEmitterDataset:
    def __init__(self, dataset_root: str | Path, *, split: str = "val") -> None:
        self.dataset_root = Path(dataset_root)
        self.split = str(split)
        self.split_root = self.dataset_root / self.split
        self.raw_dir = self.split_root / "raw_data"
        self.labels_dir = self.split_root / "labels_detect"

        if not self.raw_dir.is_dir():
            raise FileNotFoundError(f"Missing raw_data directory: {self.raw_dir}")
        if not self.labels_dir.is_dir():
            raise FileNotFoundError(f"Missing labels_detect directory: {self.labels_dir}")

        self.sample_ids = sorted(path.stem for path in self.raw_dir.glob("*.pt"))
        if not self.sample_ids:
            raise FileNotFoundError(f"No raw signal found in {self.raw_dir}")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __iter__(self) -> Iterator[SignalSample]:
        for sample_id in self.sample_ids:
            yield self.load(sample_id)

    def load(self, sample_id: str) -> SignalSample:
        signal_path = self.raw_dir / f"{sample_id}.pt"
        labels_path = self.labels_dir / f"{sample_id}.json"

        signal_tensor = torch.load(signal_path, map_location="cpu")
        signal = np.asarray(signal_tensor, dtype=np.float64).reshape(-1)
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        snr_db = float(labels[0]["snr"]) if labels else float("nan")

        return SignalSample(
            sample_id=sample_id,
            signal=signal,
            snr_db=snr_db,
            n_labels=len(labels),
            split=self.split,
        )
