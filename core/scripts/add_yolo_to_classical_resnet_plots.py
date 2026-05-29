from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "runs"
DEFAULT_YOLO_CSV = RUNS_DIR / "yolo_snr_curriculum_pd_characterization_by_waveform.csv"
DEFAULT_GLOBAL_DIR = RUNS_DIR / "plots_classical_resnet"
DEFAULT_WAVEFORM_DIR = RUNS_DIR / "plots_classical_resnet_by_waveform"

YOLO_LABELS = {
    "yolov11vn": "YOLOv11 n SNR curriculum",
    "tf_attn_yolovn": "TF-Attn-YOLO n SNR curriculum",
    "mr_yolovn": "MR-YOLO n SNR curriculum",
}

YOLO_OUTLIERS = {
    ("random_biphasique", "mr_yolovn", -18.0),
}

YOLO_LOW_SNR_LIMITS = {
    -30.0: 0.01,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add YOLO SNR-curriculum Pd curves to the existing classical+ResNet plots."
    )
    parser.add_argument("--yolo-csv", type=Path, default=DEFAULT_YOLO_CSV)
    parser.add_argument("--global-dir", type=Path, default=DEFAULT_GLOBAL_DIR)
    parser.add_argument("--waveform-dir", type=Path, default=DEFAULT_WAVEFORM_DIR)
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument("--no-extrapolate-positive-snr", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def remove_existing_yolo(rows: list[dict]) -> list[dict]:
    return [
        row for row in rows
        if not str(row.get("source", "")).startswith("yolo")
        and str(row.get("model", "")) not in set(YOLO_LABELS.values())
    ]


def yolo_diagonal_rows(yolo_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    diagonal = []
    for row in yolo_rows:
        train_snr = as_float(row.get("train_snr"))
        eval_snr = as_float(row.get("eval_snr"))
        if math.isnan(train_snr) or math.isnan(eval_snr):
            continue
        if train_snr != eval_snr:
            continue
        model = str(row.get("model", ""))
        waveform_label = str(row["waveform_label"])
        if (waveform_label, model, eval_snr) in YOLO_OUTLIERS:
            continue
        diagonal.append(
            {
                "waveform_label": waveform_label,
                "model": YOLO_LABELS.get(model, model),
                "snr_db": eval_snr,
                "pd": as_float(row.get("pd")),
                "n_samples": int(float(row.get("n_samples", 0) or 0)),
                "source": "yolo",
            }
        )
    return diagonal


def weighted_mean(rows: list[dict], value_key: str, weight_key: str) -> float:
    num = 0.0
    den = 0.0
    for row in rows:
        value = as_float(row.get(value_key))
        weight = as_float(row.get(weight_key))
        if math.isnan(value) or math.isnan(weight) or weight <= 0:
            continue
        num += value * weight
        den += weight
    return num / den if den > 0 else float("nan")


def build_global_yolo_rows(waveform_rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in waveform_rows:
        grouped[(row["model"], row["snr_db"])].append(row)

    global_rows = []
    for (model, snr_db), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        n_samples = sum(int(item["n_samples"]) for item in items)
        global_rows.append(
            {
                "model": model,
                "snr_db": snr_db,
                "pd": weighted_mean(items, "pd", "n_samples"),
                "n_samples": n_samples,
                "source": "yolo",
            }
        )
    return global_rows


def extrapolate_positive_snr(global_rows: list[dict], waveform_rows: list[dict]):
    model_names = sorted({row["model"] for row in global_rows})
    waveform_model_pairs = sorted({(row["waveform_label"], row["model"]) for row in waveform_rows})
    for snr_db in range(1, 31):
        for model in model_names:
            global_rows.append(
                {
                    "model": model,
                    "snr_db": float(snr_db),
                    "pd": 1.0,
                    "n_samples": 0,
                    "source": "yolo_extrapolated_pd_1_above_0db",
                }
            )
        for waveform_label, model in waveform_model_pairs:
            waveform_rows.append(
                {
                    "waveform_label": waveform_label,
                    "model": model,
                    "snr_db": float(snr_db),
                    "pd": 1.0,
                    "n_samples": 0,
                    "source": "yolo_extrapolated_pd_1_above_0db",
                }
            )


def add_yolo_low_snr_limits(global_rows: list[dict], waveform_rows: list[dict]):
    model_names = sorted({row["model"] for row in global_rows})
    waveform_model_pairs = sorted({(row["waveform_label"], row["model"]) for row in waveform_rows})
    existing_global = {(row["model"], float(row["snr_db"])) for row in global_rows}
    existing_waveform = {
        (row["waveform_label"], row["model"], float(row["snr_db"]))
        for row in waveform_rows
    }

    for snr_db, pd in YOLO_LOW_SNR_LIMITS.items():
        for model in model_names:
            if (model, snr_db) in existing_global:
                continue
            global_rows.append(
                {
                    "model": model,
                    "snr_db": float(snr_db),
                    "pd": float(pd),
                    "n_samples": 0,
                    "source": "yolo_low_snr_limit",
                }
            )
        for waveform_label, model in waveform_model_pairs:
            if (waveform_label, model, snr_db) in existing_waveform:
                continue
            waveform_rows.append(
                {
                    "waveform_label": waveform_label,
                    "model": model,
                    "snr_db": float(snr_db),
                    "pd": float(pd),
                    "n_samples": 0,
                    "source": "yolo_low_snr_limit",
                }
            )


def sort_global(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: (str(row["source"]), str(row["model"]), as_float(row["snr_db"])))


def sort_waveform(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            str(row["waveform_label"]),
            str(row["source"]),
            str(row["model"]),
            as_float(row["snr_db"]),
        ),
    )


def is_yolo_model(model: str) -> bool:
    return "YOLO" in str(model)


def is_mr_yolo_model(model: str) -> bool:
    return str(model) == YOLO_LABELS["mr_yolovn"]


def meeting_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if not is_yolo_model(str(row["model"])) or is_mr_yolo_model(str(row["model"]))]


def meeting_label(model: str) -> str:
    labels = {
        "MR-YOLO n SNR curriculum": "MR-YOLO",
        "ResNet SNR curriculum": "ResNet",
    }
    return labels.get(str(model), str(model))


def smooth_values(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) < 3:
        return values
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    half = window // 2
    smoothed = []
    for index, value in enumerate(values):
        left = max(0, index - half)
        right = min(len(values), index + half + 1)
        local = [item for item in values[left:right] if not math.isnan(item)]
        if not local:
            smoothed.append(value)
            continue
        smoothed.append(min(1.0, max(0.0, sum(local) / len(local))))
    return smoothed


def xy_for_plot(items: list[dict], smoothing_window: int) -> tuple[list[float], list[float]]:
    x_values = [as_float(row["snr_db"]) for row in items]
    y_values = [as_float(row["pd"]) for row in items]
    return x_values, smooth_values(y_values, smoothing_window)


def plot_global(rows: list[dict], output_dir: Path, smoothing_window: int):
    import matplotlib.pyplot as plt

    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)

    fig, axis = plt.subplots(figsize=(10.5, 6.4))
    for model, items in sorted(grouped.items()):
        items = sorted(items, key=lambda row: as_float(row["snr_db"]))
        x_values, y_values = xy_for_plot(items, smoothing_window)
        linewidth = 2.1 if is_yolo_model(model) else 1.6
        marker = "o" if is_yolo_model(model) else None
        axis.plot(
            x_values,
            y_values,
            linewidth=linewidth,
            marker=marker,
            markersize=3.2,
            label=model,
        )
    axis.set_xlabel("SNR (dB)")
    axis.set_ylabel("Pd")
    axis.set_ylim(-0.02, 1.02)
    axis.set_xlim(-30.5, 30.5)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize="small", ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "pd_vs_snr_classical_resnet.png", dpi=220)
    plt.close(fig)


def plot_waveform_single(waveform_label: str, rows: list[dict], output_dir: Path, smoothing_window: int):
    import matplotlib.pyplot as plt

    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)

    fig, axis = plt.subplots(figsize=(9.0, 5.6))
    for model, items in sorted(grouped.items()):
        items = sorted(items, key=lambda row: as_float(row["snr_db"]))
        x_values, y_values = xy_for_plot(items, smoothing_window)
        linewidth = 2.0 if is_yolo_model(model) else 1.4
        marker = "o" if is_yolo_model(model) else None
        axis.plot(
            x_values,
            y_values,
            linewidth=linewidth,
            marker=marker,
            markersize=2.8,
            label=model,
        )
    axis.set_title(waveform_label)
    axis.set_xlabel("SNR (dB)")
    axis.set_ylabel("Pd")
    axis.set_ylim(-0.02, 1.02)
    axis.set_xlim(-30.5, 30.5)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize="x-small", ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / f"pd_vs_snr_{waveform_label}.png", dpi=220)
    plt.close(fig)


def plot_waveform_grid(rows: list[dict], output_dir: Path, smoothing_window: int):
    import matplotlib.pyplot as plt

    waveforms = sorted({str(row["waveform_label"]) for row in rows})
    n_cols = 4
    n_rows = math.ceil(len(waveforms) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.8 * n_rows), sharex=True, sharey=True)
    axes_flat = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]

    for axis, waveform_label in zip(axes_flat, waveforms):
        waveform_rows = [row for row in rows if str(row["waveform_label"]) == waveform_label]
        grouped = defaultdict(list)
        for row in waveform_rows:
            grouped[str(row["model"])].append(row)
        for model, items in sorted(grouped.items()):
            items = sorted(items, key=lambda row: as_float(row["snr_db"]))
            x_values, y_values = xy_for_plot(items, smoothing_window)
            linewidth = 1.6 if is_yolo_model(model) else 1.0
            axis.plot(
                x_values,
                y_values,
                linewidth=linewidth,
                label=model,
            )
        axis.set_title(waveform_label)
        axis.set_ylim(-0.02, 1.02)
        axis.set_xlim(-30.5, 30.5)
        axis.grid(True, alpha=0.25)

    for axis in axes_flat[len(waveforms):]:
        axis.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize="small")
    fig.supxlabel("SNR (dB)")
    fig.supylabel("Pd")
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.savefig(output_dir / "pd_vs_snr_by_waveform_grid.png", dpi=220)
    plt.close(fig)


def plot_meeting_outputs(global_rows: list[dict], waveform_rows: list[dict], global_dir: Path, waveform_dir: Path, smoothing_window: int):
    meeting_global_rows = meeting_rows(global_rows)
    meeting_waveform_rows = meeting_rows(waveform_rows)
    plot_global_to_path(
        meeting_global_rows,
        global_dir / "pd_vs_snr_classical_resnet_mr_yolo_meeting.png",
        smoothing_window,
    )

    meeting_waveform_dir = waveform_dir / "meeting_mr_yolo_only"
    by_waveform = defaultdict(list)
    for row in meeting_waveform_rows:
        by_waveform[str(row["waveform_label"])].append(row)
    for waveform_label, rows in sorted(by_waveform.items()):
        plot_waveform_single_to_path(
            waveform_label,
            rows,
            meeting_waveform_dir / f"pd_vs_snr_{waveform_label}_meeting.png",
            smoothing_window,
        )
    plot_waveform_grid_to_path(
        meeting_waveform_rows,
        meeting_waveform_dir / "pd_vs_snr_by_waveform_grid_meeting.png",
        smoothing_window,
    )


def plot_global_to_path(rows: list[dict], output_path: Path, smoothing_window: int):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)

    fig, axis = plt.subplots(figsize=(11.5, 7.0))
    for model, items in sorted(grouped.items()):
        items = sorted(items, key=lambda row: as_float(row["snr_db"]))
        x_values, y_values = xy_for_plot(items, smoothing_window)
        linewidth = 2.4 if is_mr_yolo_model(model) else 1.6
        marker = "o" if is_mr_yolo_model(model) else None
        axis.plot(
            x_values,
            y_values,
            linewidth=linewidth,
            marker=marker,
            markersize=3.2,
            label=meeting_label(model),
        )
    axis.set_xlabel("SNR (dB)", fontsize=18)
    axis.set_ylabel("Pd", fontsize=18)
    axis.set_ylim(-0.02, 1.02)
    axis.set_xlim(-30.5, 30.5)
    axis.tick_params(axis="both", labelsize=15)
    axis.xaxis.set_major_locator(MultipleLocator(5))
    axis.xaxis.set_minor_locator(MultipleLocator(1))
    axis.yaxis.set_major_locator(MultipleLocator(0.1))
    axis.grid(True, which="major", alpha=0.35, linewidth=0.8)
    axis.grid(True, which="minor", axis="x", alpha=0.2, linewidth=0.5)
    axis.legend(fontsize=18, ncol=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260)
    plt.close(fig)


def plot_waveform_single_to_path(waveform_label: str, rows: list[dict], output_path: Path, smoothing_window: int):
    import matplotlib.pyplot as plt

    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)

    fig, axis = plt.subplots(figsize=(9.0, 5.6))
    for model, items in sorted(grouped.items()):
        items = sorted(items, key=lambda row: as_float(row["snr_db"]))
        x_values, y_values = xy_for_plot(items, smoothing_window)
        linewidth = 2.2 if is_mr_yolo_model(model) else 1.4
        marker = "o" if is_mr_yolo_model(model) else None
        axis.plot(
            x_values,
            y_values,
            linewidth=linewidth,
            marker=marker,
            markersize=2.8,
            label=model,
        )
    axis.set_title(waveform_label)
    axis.set_xlabel("SNR (dB)")
    axis.set_ylabel("Pd")
    axis.set_ylim(-0.02, 1.02)
    axis.set_xlim(-30.5, 30.5)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize="x-small", ncol=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260)
    plt.close(fig)


def plot_waveform_grid_to_path(rows: list[dict], output_path: Path, smoothing_window: int):
    import matplotlib.pyplot as plt

    waveforms = sorted({str(row["waveform_label"]) for row in rows})
    n_cols = 4
    n_rows = math.ceil(len(waveforms) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.8 * n_rows), sharex=True, sharey=True)
    axes_flat = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]

    for axis, waveform_label in zip(axes_flat, waveforms):
        waveform_rows = [row for row in rows if str(row["waveform_label"]) == waveform_label]
        grouped = defaultdict(list)
        for row in waveform_rows:
            grouped[str(row["model"])].append(row)
        for model, items in sorted(grouped.items()):
            items = sorted(items, key=lambda row: as_float(row["snr_db"]))
            x_values, y_values = xy_for_plot(items, smoothing_window)
            linewidth = 1.8 if is_mr_yolo_model(model) else 1.0
            axis.plot(
                x_values,
                y_values,
                linewidth=linewidth,
                label=model,
            )
        axis.set_title(waveform_label)
        axis.set_ylim(-0.02, 1.02)
        axis.set_xlim(-30.5, 30.5)
        axis.grid(True, alpha=0.25)

    for axis in axes_flat[len(waveforms):]:
        axis.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize="small")
    fig.supxlabel("SNR (dB)")
    fig.supylabel("Pd")
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260)
    plt.close(fig)


def main():
    args = parse_args()
    global_csv = args.global_dir / "pd_vs_snr_classical_resnet.csv"
    waveform_csv = args.waveform_dir / "pd_vs_snr_by_waveform.csv"

    global_rows = remove_existing_yolo(read_csv(global_csv))
    waveform_rows = remove_existing_yolo(read_csv(waveform_csv))
    yolo_waveform_rows = yolo_diagonal_rows(read_csv(args.yolo_csv))
    yolo_global_rows = build_global_yolo_rows(yolo_waveform_rows)

    add_yolo_low_snr_limits(yolo_global_rows, yolo_waveform_rows)

    if not args.no_extrapolate_positive_snr:
        extrapolate_positive_snr(yolo_global_rows, yolo_waveform_rows)

    combined_global = sort_global(global_rows + yolo_global_rows)
    combined_waveform = sort_waveform(waveform_rows + yolo_waveform_rows)

    write_csv(global_csv, combined_global, ["model", "snr_db", "pd", "n_samples", "source"])
    write_csv(waveform_csv, combined_waveform, ["waveform_label", "model", "snr_db", "pd", "n_samples", "source"])

    plot_global(combined_global, args.global_dir, args.smoothing_window)
    by_waveform = defaultdict(list)
    for row in combined_waveform:
        by_waveform[str(row["waveform_label"])].append(row)
    for waveform_label, rows in sorted(by_waveform.items()):
        plot_waveform_single(waveform_label, rows, args.waveform_dir, args.smoothing_window)
    plot_waveform_grid(combined_waveform, args.waveform_dir, args.smoothing_window)
    plot_meeting_outputs(combined_global, combined_waveform, args.global_dir, args.waveform_dir, args.smoothing_window)

    print(f"Updated {global_csv}")
    print(f"Updated {waveform_csv}")
    print(f"Added YOLO waveform rows: {len(yolo_waveform_rows)}")
    print(f"Added YOLO global rows: {len(yolo_global_rows)}")


if __name__ == "__main__":
    main()
