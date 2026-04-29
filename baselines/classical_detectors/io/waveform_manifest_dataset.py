from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch


@dataclass(frozen=True)
class WaveformScenario:
    scenario_id: str
    waveform_label: str
    class_name: str
    signal: np.ndarray
    pulse: dict

    @property
    def signal_energy(self) -> float:
        return float(np.sum(np.abs(self.signal) ** 2))

    @property
    def duration_samples(self) -> int:
        value = self.pulse.get("duration_samples", self.pulse.get("pulse_duration_samples"))
        if value is not None:
            duration = int(value)
            if duration > 0:
                return duration
        nonzero = np.flatnonzero(np.abs(self.signal) > 0)
        return int(nonzero.size)


class WaveformManifestDataset:
    """Dataset reader for <root>/manifest.json waveform/scenario layouts."""

    def __init__(self, dataset_root: str | Path, *, waveforms: set[str] | None = None) -> None:
        self.dataset_root = Path(dataset_root)
        manifest_path = self.dataset_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing manifest file: {manifest_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest if isinstance(manifest, list) else manifest.get("scenarios", manifest)
        if not isinstance(entries, list):
            raise ValueError("manifest.json must be a list or contain a 'scenarios' list.")

        self.entries = []
        for entry in entries:
            waveform_label = str(entry.get("waveform_label", entry.get("waveform", "")))
            if waveforms is not None and waveform_label not in waveforms:
                continue
            self.entries.append(entry)
        if not self.entries:
            raise FileNotFoundError("No scenario found in manifest for the requested waveforms.")

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[WaveformScenario]:
        for index in range(len(self.entries)):
            yield self.load(index)

    def _resolve(self, entry: dict, default_name: str, *keys: str) -> Path:
        for key in keys:
            value = entry.get(key)
            if value:
                path = Path(value)
                return path if path.is_absolute() else self.dataset_root / path
        scenario_dir = entry.get("scenario_dir", entry.get("path", entry.get("relative_dir")))
        if scenario_dir:
            return self.dataset_root / scenario_dir / default_name
        raise KeyError(f"Missing path key among {keys}.")

    def load(self, index: int) -> WaveformScenario:
        entry = self.entries[index]
        signal_path = self._resolve(entry, "signal.pt", "signal_path", "signal", "signal_pt")
        pulse_path = self._resolve(entry, "pulse.json", "pulse_path", "pulse", "pulse_json")

        signal_tensor = torch.load(signal_path, map_location="cpu")
        signal = np.asarray(signal_tensor.cpu().numpy()).reshape(-1)
        pulse = json.loads(pulse_path.read_text(encoding="utf-8"))

        return WaveformScenario(
            scenario_id=str(entry.get("scenario_id", pulse.get("scenario_id", signal_path.parent.name))),
            waveform_label=str(entry.get("waveform_label", pulse.get("waveform_label", signal_path.parent.parent.name))),
            class_name=str(entry.get("class_name", pulse.get("class_name", ""))),
            signal=signal,
            pulse=pulse,
        )
