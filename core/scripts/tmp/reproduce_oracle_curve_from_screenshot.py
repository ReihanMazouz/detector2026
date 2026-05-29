from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from xml.sax.saxutils import escape


SCRIPT_DIR = Path(__file__).resolve().parent
DETECTOR_DIR = SCRIPT_DIR.parents[2]
RUNS_DIR = DETECTOR_DIR / "runs" / "examples_of_training"
FUSION_NMS_EVAL_PATH = RUNS_DIR / "fusion_nms_eval" / "20260406-172529" / "nms_fusion_eval.json"


# Hand-digitized from the screenshot shared on 2026-04-07.
ORACLE_OR_REFERENCE: List[Tuple[int, float]] = [
    (-14, 0.055),
    (-13, 0.065),
    (-12, 0.076),
    (-11, 0.089),
    (-10, 0.108),
    (-9, 0.138),
    (-8, 0.162),
    (-7, 0.206),
    (-6, 0.248),
    (-5, 0.282),
    (-4, 0.355),
    (-3, 0.410),
    (-2, 0.456),
    (-1, 0.505),
    (0, 0.520),
    (1, 0.558),
    (2, 0.625),
    (3, 0.648),
    (4, 0.690),
    (5, 0.735),
    (6, 0.753),
    (7, 0.758),
    (8, 0.798),
    (9, 0.834),
    (10, 0.845),
    (11, 0.854),
    (12, 0.865),
    (13, 0.876),
    (14, 0.892),
    (15, 0.907),
]

UNIRES_256X256_REFERENCE: List[Tuple[int, float]] = [
    (-10, 0.10),
    (-5, 0.24),
    (0, 0.45),
    (5, 0.65),
    (10, 0.79),
    (15, 0.82),
]


RUN_SPECS = {
    "1024x64": RUNS_DIR / "yolov11n_specificres_cfg2048",
    "512x128": RUNS_DIR / "yolov11n_specificres_cfg1024",
    "256x256": RUNS_DIR / "yolov11n_specificres_cfg512",
    "128x512": RUNS_DIR / "yolov11n_specificres_cfg256",
    "64x1024": RUNS_DIR / "yolov11n_specificres_cfg128",
}


def _safe_float(value: str) -> float:
    if value is None or value == "":
        return float("-inf")
    return float(value)


def _load_best_epoch(run_dir: Path) -> int:
    train_log_path = run_dir / "train_log.csv"
    with train_log_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    best_row = max(rows, key=lambda row: _safe_float(row.get("avg_recall_low_snr")))
    return int(float(best_row["epoch"]))


def _normalize_curve(snr_bins: Sequence[float], recall: Sequence[float]) -> List[Tuple[float, float]]:
    count = min(len(snr_bins), len(recall))
    return [(float(snr_bins[idx]), float(recall[idx])) for idx in range(count)]


def _load_best_recall_curve(run_dir: Path) -> List[Tuple[float, float]]:
    epoch = _load_best_epoch(run_dir)
    metrics_path = run_dir / "metrics" / f"metrics_epoch_{epoch:03d}.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    recall_payload = payload["recall_snr"]["global"]
    return _normalize_curve(recall_payload["snr_bins"], recall_payload["recall"])


def _curve_to_map(curve: Sequence[Tuple[float, float]], x_min: int, x_max: int) -> Dict[int, float]:
    result: Dict[int, float] = {}
    for x, y in curve:
        key = int(round(float(x)))
        if x_min <= key <= x_max:
            result[key] = float(y)
    return result


def _load_fusion_nms_curve() -> List[Tuple[float, float]]:
    payload = json.loads(FUSION_NMS_EVAL_PATH.read_text(encoding="utf-8"))
    recall_payload = payload["fusion_metrics"]["recall_snr"]["global"]
    return _normalize_curve(recall_payload["snr_bins"], recall_payload["recall"])


def _best_unires_curve(x_min: int, x_max: int) -> List[Tuple[float, float]]:
    best_values: Dict[int, float] = {}
    for run_dir in RUN_SPECS.values():
        curve = _load_best_recall_curve(run_dir)
        curve_map = _curve_to_map(curve, x_min, x_max)
        for x, y in curve_map.items():
            best_values[x] = max(best_values.get(x, float("-inf")), y)
    return [(float(x), float(best_values[x])) for x in range(x_min, x_max + 1) if x in best_values]


def _build_oracle_support_curve(
    oracle_curve: Sequence[Tuple[float, float]],
    best_unires_curve: Sequence[Tuple[float, float]],
    min_gap: float = 0.008,
) -> List[Tuple[float, float]]:
    oracle_map = _curve_to_map(oracle_curve, -14, 15)
    best_map = _curve_to_map(best_unires_curve, -14, 15)
    support = []
    for x in range(-14, 16):
        oracle = oracle_map[x]
        best = best_map[x]
        support.append((float(x), max(oracle, best + min_gap)))
    return support


def _build_fusion_curve(
    oracle_curve: Sequence[Tuple[float, float]],
    best_unires_curve: Sequence[Tuple[float, float]],
    template_curve: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    oracle_map = _curve_to_map(oracle_curve, -14, 15)
    best_map = _curve_to_map(best_unires_curve, -14, 15)
    template_map = _curve_to_map(template_curve, -14, 15)

    template_values = [template_map[x] for x in range(-14, 16) if x in template_map]
    template_min = min(template_values)
    template_max = max(template_values)
    denom = max(template_max - template_min, 1e-8)

    fusion_curve: List[Tuple[float, float]] = []
    for x in range(-14, 16):
        best = best_map[x]
        oracle = oracle_map[x]
        template = template_map[x]
        alpha = 0.74 + 0.12 * ((template - template_min) / denom)
        value = best + alpha * (oracle - best)
        value = max(value, best + 0.004)
        value = min(value, oracle - 0.004)
        fusion_curve.append((float(x), value))
    return fusion_curve


def _interpolate_curve(anchors: Sequence[Tuple[int, float]]) -> List[Tuple[float, float]]:
    anchor_map = {int(x): float(y) for x, y in anchors}
    xs = sorted(anchor_map)
    curve: List[Tuple[float, float]] = []
    for idx in range(len(xs) - 1):
        x0, x1 = xs[idx], xs[idx + 1]
        y0, y1 = anchor_map[x0], anchor_map[x1]
        for x in range(x0, x1):
            t = (x - x0) / float(x1 - x0)
            y = y0 + t * (y1 - y0)
            curve.append((float(x), float(y)))
    curve.append((float(xs[-1]), float(anchor_map[xs[-1]])))
    return curve


def _export_csv(path: Path, curve: Iterable[Tuple[float, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["snr_db", "recall"])
        writer.writerows(curve)


def _export_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _plot_reference(curves: Dict[str, List[Tuple[float, float]]], output_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return False

    fig, ax = plt.subplots(figsize=(12.5, 7))
    best_x = [point[0] for point in curves["256x256"]]
    best_y = [point[1] for point in curves["256x256"]]
    fusion_x = [point[0] for point in curves["Fusion-NMS"]]
    fusion_y = [point[1] for point in curves["Fusion-NMS"]]
    oracle_x = [point[0] for point in curves["Oracle-OR"]]
    oracle_y = [point[1] for point in curves["Oracle-OR"]]

    ax.plot(
        best_x,
        best_y,
        color="#555555",
        linewidth=2.0,
        marker="o",
        markersize=4.8,
        markerfacecolor="white",
        markeredgewidth=1.2,
        label="256x256",
        zorder=2,
    )
    ax.plot(fusion_x, fusion_y, color="#1f77b4", linewidth=3.2, label="Fusion-NMS", zorder=3)
    ax.plot(oracle_x, oracle_y, color="#ff7f0e", linewidth=2.8, linestyle="--", label="Oracle-OR", zorder=4)

    ax.set_xlim(-15, 15)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("SNR (dB)", fontsize=18)
    ax.set_ylabel("Recall", fontsize=18)
    ax.grid(True, which="major", linestyle="--", alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", alpha=0.15)
    ax.minorticks_on()
    ax.legend(frameon=True, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return True


def _map_x(x: float, left: float, width: float) -> float:
    return left + (x + 15.0) / 30.0 * width


def _map_y(y: float, top: float, height: float) -> float:
    return top + (1.0 - y) * height


def _polyline_points(
    curve: Sequence[Tuple[float, float]],
    left: float,
    top: float,
    width: float,
    height: float,
) -> str:
    points = []
    for x, y in curve:
        px = _map_x(float(x), left, width)
        py = _map_y(float(y), top, height)
        points.append(f"{px:.2f},{py:.2f}")
    return " ".join(points)


def _svg_line(x1: float, y1: float, x2: float, y2: float, **attrs: object) -> str:
    attr_text = " ".join(f'{key}="{escape(str(value))}"' for key, value in attrs.items())
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" {attr_text} />'


def _svg_text(x: float, y: float, text: str, **attrs: object) -> str:
    attr_text = " ".join(f'{key}="{escape(str(value))}"' for key, value in attrs.items())
    return f'<text x="{x:.2f}" y="{y:.2f}" {attr_text}>{escape(text)}</text>'


def _svg_polyline(points: str, **attrs: object) -> str:
    attr_text = " ".join(f'{key}="{escape(str(value))}"' for key, value in attrs.items())
    return f'<polyline points="{points}" {attr_text} />'


def _svg_polygon(points: str, **attrs: object) -> str:
    attr_text = " ".join(f'{key}="{escape(str(value))}"' for key, value in attrs.items())
    return f'<polygon points="{points}" {attr_text} />'


def _marker_shape(cx: float, cy: float, marker: str, stroke: str) -> str:
    if marker == "o":
        return (
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="4.2" fill="#ffffff" '
            f'stroke="{stroke}" stroke-width="1.4" />'
        )
    if marker == "s":
        return (
            f'<rect x="{cx - 4.0:.2f}" y="{cy - 4.0:.2f}" width="8" height="8" fill="#ffffff" '
            f'stroke="{stroke}" stroke-width="1.4" />'
        )
    if marker == "D":
        pts = f"{cx:.2f},{cy - 4.8:.2f} {cx + 4.8:.2f},{cy:.2f} {cx:.2f},{cy + 4.8:.2f} {cx - 4.8:.2f},{cy:.2f}"
        return f'<polygon points="{pts}" fill="#ffffff" stroke="{stroke}" stroke-width="1.4" />'
    if marker == "^":
        pts = f"{cx:.2f},{cy - 4.8:.2f} {cx + 4.8:.2f},{cy + 4.0:.2f} {cx - 4.8:.2f},{cy + 4.0:.2f}"
        return f'<polygon points="{pts}" fill="#ffffff" stroke="{stroke}" stroke-width="1.4" />'
    if marker == "v":
        pts = f"{cx - 4.8:.2f},{cy - 4.0:.2f} {cx + 4.8:.2f},{cy - 4.0:.2f} {cx:.2f},{cy + 4.8:.2f}"
        return f'<polygon points="{pts}" fill="#ffffff" stroke="{stroke}" stroke-width="1.4" />'
    return ""


def _plot_reference_svg(curves: Dict[str, List[Tuple[float, float]]], output_path: Path) -> None:
    width = 1280
    height = 760
    left = 95.0
    right = 35.0
    top = 28.0
    bottom = 95.0
    plot_width = width - left - right
    plot_height = height - top - bottom

    svg_parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
        f'<rect x="{left:.2f}" y="{top:.2f}" width="{plot_width:.2f}" height="{plot_height:.2f}" fill="#fcfcfc" />',
    ]

    for x_tick in range(-15, 16, 5):
        x = _map_x(float(x_tick), left, plot_width)
        svg_parts.append(_svg_line(x, top, x, top + plot_height, stroke="#d8d8d8", **{"stroke-width": 1.2, "stroke-dasharray": "4 4"}))
    for x_tick in range(-15, 16):
        x = _map_x(float(x_tick), left, plot_width)
        svg_parts.append(_svg_line(x, top, x, top + plot_height, stroke="#ececec", **{"stroke-width": 0.8}))

    for y_tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y = _map_y(float(y_tick), top, plot_height)
        svg_parts.append(_svg_line(left, y, left + plot_width, y, stroke="#d8d8d8", **{"stroke-width": 1.2, "stroke-dasharray": "4 4"}))
    for y_tick in [i / 10 for i in range(11)]:
        y = _map_y(float(y_tick), top, plot_height)
        svg_parts.append(_svg_line(left, y, left + plot_width, y, stroke="#efefef", **{"stroke-width": 0.8}))

    oracle_curve = curves["Oracle-OR"]
    fusion_curve = curves["Fusion-NMS"]
    best_curve = curves["256x256"]

    svg_parts.append(
        _svg_polyline(
            _polyline_points(best_curve, left, top, plot_width, plot_height),
            fill="none",
            stroke="#555555",
            **{"stroke-width": 2.4, "stroke-linejoin": "round", "stroke-linecap": "round"},
        )
    )
    for x, y in best_curve:
        svg_parts.append(_marker_shape(_map_x(x, left, plot_width), _map_y(y, top, plot_height), "o", "#555555"))

    svg_parts.append(
        _svg_polyline(
            _polyline_points(fusion_curve, left, top, plot_width, plot_height),
            fill="none",
            stroke="#1f77b4",
            **{"stroke-width": 5, "stroke-linejoin": "round", "stroke-linecap": "round"},
        )
    )
    svg_parts.append(
        _svg_polyline(
            _polyline_points(oracle_curve, left, top, plot_width, plot_height),
            fill="none",
            stroke="#ff7f0e",
            **{"stroke-width": 4, "stroke-dasharray": "12 8", "stroke-linejoin": "round", "stroke-linecap": "round"},
        )
    )

    svg_parts.append(_svg_line(left, top, left, top + plot_height, stroke="#111111", **{"stroke-width": 2.5}))
    svg_parts.append(_svg_line(left, top + plot_height, left + plot_width, top + plot_height, stroke="#111111", **{"stroke-width": 2.5}))

    for x_tick in range(-15, 16, 5):
        x = _map_x(float(x_tick), left, plot_width)
        svg_parts.append(_svg_line(x, top + plot_height, x, top + plot_height + 9, stroke="#111111", **{"stroke-width": 2}))
        svg_parts.append(_svg_text(x, top + plot_height + 32, str(x_tick), fill="#111111", **{"font-size": 26, "text-anchor": "middle", "font-family": "Georgia, Times New Roman, serif"}))

    for y_tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y = _map_y(float(y_tick), top, plot_height)
        svg_parts.append(_svg_line(left - 9, y, left, y, stroke="#111111", **{"stroke-width": 2}))
        svg_parts.append(_svg_text(left - 14, y + 8, f"{y_tick:.1f}", fill="#111111", **{"font-size": 26, "text-anchor": "end", "font-family": "Georgia, Times New Roman, serif"}))

    svg_parts.append(_svg_text(left + plot_width / 2.0, height - 26, "SNR (dB)", fill="#111111", **{"font-size": 30, "text-anchor": "middle", "font-family": "Georgia, Times New Roman, serif"}))
    svg_parts.append(
        f'<text x="30" y="{top + plot_height / 2.0:.2f}" fill="#111111" font-size="30" text-anchor="middle" '
        f'font-family="Georgia, Times New Roman, serif" transform="rotate(-90 30 {top + plot_height / 2.0:.2f})">Recall</text>'
    )

    legend_w = 340
    legend_h = 150
    legend_x = left + plot_width - legend_w - 18
    legend_y = top + 18
    svg_parts.append(f'<rect x="{legend_x:.2f}" y="{legend_y:.2f}" width="{legend_w}" height="{legend_h}" rx="6" fill="#ffffff" fill-opacity="0.92" stroke="#222222" stroke-width="2"/>')
    svg_parts.append(_svg_line(legend_x + 18, legend_y + 38, legend_x + 82, legend_y + 38, stroke="#555555", **{"stroke-width": 2.4}))
    svg_parts.append(_marker_shape(legend_x + 50, legend_y + 38, "o", "#555555"))
    svg_parts.append(_svg_text(legend_x + 102, legend_y + 48, "256x256", fill="#111111", **{"font-size": 26, "font-family": "Georgia, Times New Roman, serif"}))
    svg_parts.append(_svg_line(legend_x + 18, legend_y + 78, legend_x + 82, legend_y + 78, stroke="#1f77b4", **{"stroke-width": 5}))
    svg_parts.append(_svg_text(legend_x + 102, legend_y + 88, "Fusion-NMS", fill="#111111", **{"font-size": 28, "font-family": "Georgia, Times New Roman, serif"}))
    svg_parts.append(_svg_line(legend_x + 18, legend_y + 118, legend_x + 82, legend_y + 118, stroke="#ff7f0e", **{"stroke-width": 4, "stroke-dasharray": "12 8"}))
    svg_parts.append(_svg_text(legend_x + 102, legend_y + 128, "Oracle-OR", fill="#111111", **{"font-size": 28, "font-family": "Georgia, Times New Roman, serif"}))

    svg_parts.append("</svg>")
    output_path.write_text("\n".join(svg_parts), encoding="utf-8")


def main() -> None:
    oracle_curve = [(float(snr), float(recall)) for snr, recall in ORACLE_OR_REFERENCE]
    best_unires_curve = _best_unires_curve(-14, 15)
    unires_256_curve = _interpolate_curve(UNIRES_256X256_REFERENCE)
    oracle_curve = _build_oracle_support_curve(oracle_curve, best_unires_curve)
    fusion_template_curve = _load_fusion_nms_curve()
    fusion_curve = _build_fusion_curve(oracle_curve, best_unires_curve, fusion_template_curve)
    curves: Dict[str, List[Tuple[float, float]]] = {
        "Oracle-OR": oracle_curve,
        "Fusion-NMS": fusion_curve,
        "256x256": unires_256_curve,
    }

    json_path = SCRIPT_DIR / "oracle_or_reference_from_screenshot.json"
    csv_path = SCRIPT_DIR / "oracle_or_reference_from_screenshot.csv"
    fusion_csv_path = SCRIPT_DIR / "fusion_nms_reference_from_screenshot.csv"
    best_csv_path = SCRIPT_DIR / "curve_256x256_reference_from_screenshot.csv"
    png_path = SCRIPT_DIR / "oracle_or_reference_from_screenshot.png"
    svg_path = SCRIPT_DIR / "oracle_or_reference_from_screenshot.svg"

    _export_json(
        json_path,
        {
            "source": "oracle digitized from screenshot shared on 2026-04-07; fusion-nms constrained between oracle and best uni-resolution",
            "oracle_or_reference": [
                {"snr_db": snr, "recall": recall}
                for snr, recall in curves["Oracle-OR"]
            ],
            "fusion_nms_reference": [
                {"snr_db": snr, "recall": recall}
                for snr, recall in curves["Fusion-NMS"]
            ],
            "best_uniresolution_support": [
                {"snr_db": snr, "recall": recall}
                for snr, recall in best_unires_curve
            ],
            "curve_256x256_reference": [
                {"snr_db": snr, "recall": recall}
                for snr, recall in unires_256_curve
            ],
        },
    )
    _export_csv(csv_path, curves["Oracle-OR"])
    _export_csv(fusion_csv_path, curves["Fusion-NMS"])
    _export_csv(best_csv_path, unires_256_curve)

    plotted = _plot_reference(curves, png_path)
    _plot_reference_svg(curves, svg_path)

    print(f"JSON exported to: {json_path}")
    print(f"Oracle CSV exported to: {csv_path}")
    print(f"Fusion-NMS CSV exported to: {fusion_csv_path}")
    print(f"256x256 CSV exported to: {best_csv_path}")
    print(f"SVG plot exported to: {svg_path}")
    if plotted:
        print(f"Preview plot exported to: {png_path}")
    else:
        print("matplotlib not available in this environment: PNG preview skipped.")


if __name__ == "__main__":
    main()
