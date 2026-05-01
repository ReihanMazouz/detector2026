from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.classical_detectors.evaluation import WaveformSweepConfig, run_waveform_snr_sweep
from baselines.classical_detectors.evaluation.deep_waveform_sweep import (
    DeepModelSpec,
    DeepWaveformSweepConfig,
    run_deep_waveform_snr_sweep,
)


# =============================================================================
# Configuration a modifier ici
# =============================================================================

DATASET_ROOT = Path("/Users/tailleesarah/Documents/thèse/icml/ICML2026DataSimulator/tmp/output/rf_single_emitter_real_smoketest_5x5")
OUTPUT_JSON = PROJECT_ROOT / "runs" / "baselines" / "dataset_evaluation.json"
PROGRESS_LOG = PROJECT_ROOT / "runs" / "baselines" / "dataset_evaluation_progress.log"
PLOT_CURVES = True

RUN_CLASSICAL_BASELINES = True
RUN_YOLOV11 = True
RUN_MR_YOLO = False

YOLOV11_WEIGHTS = Path("/Users/tailleesarah/Documents/thèse/icml/real_data_validation_minimal/weights/yolov11n_cfg512_best.pt")
MR_YOLO_WEIGHTS = Path("/Users/tailleesarah/Documents/thèse/icml/real_data_validation_minimal/weights/mr_yolovn_best.pt")

PFA = 1e-2
SEED = 444
NOISE_VARIANCE = 1.0
NOISE_TRIALS_CLASSICAL = 500
NOISE_TRIALS_DEEP = 500
DEEP_BATCH_SIZE = 64
SNR_VALUES_DB = tuple(range(-20, 61, 2))
WAVEFORMS: tuple[str, ...] = ()

DEVICE = "cuda:0"
NUM_CLASSES = 20
PREPROCESSING = "log_snr_estimated"

YOLOV11_SCALE = "n"
YOLOV11_RES_KEY = "cfg512"

MR_YOLO_SCALE = "n"
MR_YOLO_RES_KEYS = ("cfg128", "cfg256", "cfg512", "cfg1024", "cfg2048")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _build_deep_specs() -> list[DeepModelSpec]:
    specs = []
    if RUN_YOLOV11:
        specs.append(
            DeepModelSpec(
                name="yolov11",
                family="yolov11",
                weights_path=YOLOV11_WEIGHTS.resolve(),
                scale=YOLOV11_SCALE,
                res_key=YOLOV11_RES_KEY,
                preprocessing=PREPROCESSING,
                num_classes=NUM_CLASSES,
            )
        )
    if RUN_MR_YOLO:
        specs.append(
            DeepModelSpec(
                name="mr_yolo",
                family="mr_yolo",
                weights_path=MR_YOLO_WEIGHTS.resolve(),
                scale=MR_YOLO_SCALE,
                res_keys=MR_YOLO_RES_KEYS,
                preprocessing=PREPROCESSING,
                num_classes=NUM_CLASSES,
            )
        )
    return specs


def _iter_detector_results(payload: dict[str, Any]):
    classical = payload.get("results", {}).get("classical", {}).get("detectors", {})
    for detector_name, detector_payload in classical.items():
        yield f"classical/{detector_name}", detector_payload

    deep = payload.get("results", {}).get("deep", {}).get("detectors", {})
    for detector_name, detector_payload in deep.items():
        yield f"deep/{detector_name}", detector_payload


def _plot_curve(payload: dict[str, Any], *, curve_key: str, x_key: str, x_label: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    detector_items = list(_iter_detector_results(payload))
    if not detector_items:
        return

    waveforms = sorted(
        {
            str(row["waveform_label"])
            for _, detector_payload in detector_items
            for row in detector_payload.get(curve_key, [])
        }
    )
    if not waveforms:
        return

    n_cols = min(4, len(waveforms))
    n_rows = (len(waveforms) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 3.0 * n_rows),
        sharex=True,
        sharey=True,
    )
    if hasattr(axes, "ravel"):
        axes = list(axes.ravel())
    else:
        axes = [axes]

    rows_by_detector_and_waveform: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for detector_name, detector_payload in detector_items:
        rows_by_waveform: dict[str, list[dict[str, Any]]] = {}
        for row in detector_payload.get(curve_key, []):
            rows_by_waveform.setdefault(str(row["waveform_label"]), []).append(row)
        rows_by_detector_and_waveform[detector_name] = rows_by_waveform

    legend_handles = []
    legend_labels = []
    for axis, waveform_label in zip(axes, waveforms):
        for detector_name, rows_by_waveform in rows_by_detector_and_waveform.items():
            rows = rows_by_waveform.get(waveform_label, [])
            if not rows:
                continue
            rows = sorted(rows, key=lambda item: float(item[x_key]))
            (line,) = axis.plot(
                [float(row[x_key]) for row in rows],
                [float(row["pd"]) for row in rows],
                marker="o",
                linewidth=1.5,
                markersize=3.0,
                label=detector_name,
            )
            if detector_name not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(detector_name)
        axis.set_title(waveform_label)
        axis.set_ylabel("Pd")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(True, alpha=0.3)

    for axis in axes[len(waveforms):]:
        axis.set_visible(False)
    for axis in axes[-n_cols:]:
        axis.set_xlabel(x_label)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            fontsize="small",
            ncol=min(4, len(legend_labels)),
        )
    fig.suptitle(f"Detection probability at Pfa={payload['pfa_target']}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_results(payload: dict[str, Any]) -> None:
    base = OUTPUT_JSON.with_suffix("")
    _plot_curve(
        payload,
        curve_key="by_snr",
        x_key="snr_db",
        x_label="SNR (dB)",
        output_path=base.with_name(f"{base.name}_pd_vs_snr.png"),
    )


def main() -> None:
    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_LOG.write_text("", encoding="utf-8")
    _log("start dataset evaluation")
    dataset_root = DATASET_ROOT.resolve()
    _log(f"dataset_root={dataset_root}")
    if not (dataset_root / "manifest.json").is_file():
        raise FileNotFoundError(f"Missing manifest.json in dataset root: {dataset_root}")

    payload: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "pfa_target": float(PFA),
        "seed": int(SEED),
        "noise_variance": float(NOISE_VARIANCE),
        "signal_domain": "real_1d",
        "snr_definition": "(sum(|s|^2) / duration_samples) / noise_variance",
        "noise_seed_policy": "same noise realization for each scenario across all SNR levels",
        "snr_values_db": [float(v) for v in SNR_VALUES_DB],
        "preprocessing": PREPROCESSING,
        "results": {},
    }

    if RUN_CLASSICAL_BASELINES:
        _log("running classical baselines")
        payload["results"]["classical"] = run_waveform_snr_sweep(
            WaveformSweepConfig(
                dataset_root=dataset_root,
                pfa=PFA,
                noise_trials=NOISE_TRIALS_CLASSICAL,
                seed=SEED,
                noise_variance=NOISE_VARIANCE,
                snr_values_db=SNR_VALUES_DB,
                waveforms=WAVEFORMS,
                progress_log=_log,
            )
        )
        _log("classical baselines done")

    deep_specs = _build_deep_specs()
    if deep_specs:
        _log(f"running deep models: {[spec.name for spec in deep_specs]}")
        payload["results"]["deep"] = run_deep_waveform_snr_sweep(
            DeepWaveformSweepConfig(
                dataset_root=dataset_root,
                models=tuple(deep_specs),
                pfa=PFA,
                noise_trials=NOISE_TRIALS_DEEP,
                seed=SEED,
                noise_variance=NOISE_VARIANCE,
                snr_values_db=SNR_VALUES_DB,
                waveforms=WAVEFORMS,
                device=DEVICE,
                batch_size=DEEP_BATCH_SIZE,
                progress_log=_log,
            )
        )
        _log("deep models done")

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    _log(f"wrote json: {OUTPUT_JSON}")
    if PLOT_CURVES:
        _log("plotting curves")
        _plot_results(payload)
        _log(f"wrote plots next to: {OUTPUT_JSON}")
    _log("evaluation done")
    print(json.dumps(payload, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
