from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.classical_detectors.evaluation import WaveformSweepConfig, run_waveform_snr_sweep


DEFAULT_DATASET_ROOT = PROJECT_ROOT / "runs" / "classical_detectors" / "datasets" / "single_emitter_noiseless"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "classical_detectors" / "waveform_snr_sweep.json"


def _parse_float_list(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one numeric value.")
    return values


def _parse_waveforms(raw: str) -> tuple[str, ...]:
    if raw.strip().lower() in {"", "all"}:
        return ()
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate classical detectors on a noiseless waveform/scenario dataset. "
            "Outputs Pd curves per waveform versus target SNR."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pfa", type=float, default=1e-3)
    parser.add_argument("--noise-trials", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=444)
    parser.add_argument("--qmf-levels", type=int, default=6)
    parser.add_argument(
        "--noise-variance",
        type=float,
        default=1.0,
        help="Fixed real noise variance E[n^2] used for the Pd-vs-SNR curve.",
    )
    parser.add_argument(
        "--snr-values-db",
        type=_parse_float_list,
        default=tuple(range(-20, 21, 2)),
        help="Comma-separated target SNR values in dB.",
    )
    parser.add_argument("--waveforms", type=_parse_waveforms, default=(), help="Comma-separated waveform labels, or all.")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Only write JSON results. By default, PNG curves are written next to --output.",
    )
    return parser.parse_args()


def _plot_curves(payload: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(payload["detectors"]),
        1,
        figsize=(9.0, max(3.0, 2.7 * len(payload["detectors"]))),
        sharex=True,
        sharey=True,
    )
    if hasattr(axes, "ravel"):
        axes = list(axes.ravel())
    else:
        axes = [axes]
    for axis, (detector_name, detector_payload) in zip(axes, payload["detectors"].items()):
        grouped = {}
        for row in detector_payload["by_snr"]:
            grouped.setdefault(row["waveform_label"], []).append(row)
        for waveform_label, rows in sorted(grouped.items()):
            rows = sorted(rows, key=lambda item: item["snr_db"])
            axis.plot([row["snr_db"] for row in rows], [row["pd"] for row in rows], marker="o", label=waveform_label)
        axis.set_title(detector_name)
        axis.set_ylabel("Pd")
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Target SNR (dB)")
    axes[0].legend(loc="best", fontsize="small")
    fig.suptitle(f"Detection probability at Pfa={payload['pfa_target']}")
    fig.tight_layout()
    fig.savefig(output.with_name(f"{output.stem}_by_snr.png"), dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    payload = run_waveform_snr_sweep(
        WaveformSweepConfig(
            dataset_root=args.dataset_root.resolve(),
            pfa=args.pfa,
            noise_trials=args.noise_trials,
            seed=args.seed,
            qmf_levels=args.qmf_levels,
            noise_variance=args.noise_variance,
            snr_values_db=args.snr_values_db,
            waveforms=args.waveforms,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not args.no_plots:
        try:
            _plot_curves(payload, args.output)
        except ImportError as exc:
            print(f"plots skipped: {exc}", file=sys.stderr)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
