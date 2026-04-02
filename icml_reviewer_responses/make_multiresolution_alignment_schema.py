from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
OUTPUT_PATH = ASSETS_DIR / "multiresolution_alignment_schema.png"


BLUE = "#48B8E8"
PINK = "#FF4FA0"
GREEN = "#7CB342"
GRID = "#D8D8D8"
GRAY = "#666666"
ARROW = "#7A7A7A"


def draw_variable_grid(ax, width, height, x_edges, y_edges, title, xlabel, ylabel):
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.set_xticks(x_edges)
    ax.set_yticks(y_edges)
    ax.grid(color=GRID, linewidth=1)
    ax.tick_params(length=0, labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_color("#888888")
        spine.set_linewidth(1)
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)


def fill_cells(ax, x_edges, y_edges, cells, color, alpha=0.95):
    for x_idx, y_idx in cells:
        x0 = x_edges[x_idx]
        y0 = y_edges[y_idx]
        w = x_edges[x_idx + 1] - x0
        h = y_edges[y_idx + 1] - y0
        ax.add_patch(
            Rectangle((x0, y0), w, h, facecolor=color, edgecolor="none", alpha=alpha)
        )


def aggregate_cells(fine_cells, x_group, y_group):
    aggregated = set()
    for x, y in fine_cells:
        aggregated.add((x // x_group, y // y_group))
    return sorted(aggregated)


def draw_uniform_grid(ax, n, title, note, color, cells):
    edges = list(range(n + 1))
    draw_variable_grid(ax, n, n, edges, edges, title, "Shared time axis", "Shared frequency axis")
    fill_cells(ax, edges, edges, cells, color)
    ax.text(n / 2, -0.7, note, ha="center", va="top", fontsize=9, color=GRAY)


def add_arrow(fig, source_ax, target_ax, dx=0.0, dy=0.0):
    src = source_ax.get_position()
    dst = target_ax.get_position()
    fig.add_artist(
        FancyArrowPatch(
            (src.x1 + 0.01, src.y0 + src.height / 2),
            (dst.x0 - 0.01 + dx, dst.y0 + dst.height / 2 + dy),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=16,
            lw=1.8,
            color=ARROW,
        )
    )


def build_figure():
    fig = plt.figure(figsize=(12.8, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.15, 1.4], height_ratios=[1, 1])

    # Same displayed size for both initial spectrograms:
    # top: 10 time bins x 5 frequency bins, each frequency bin has height 2
    # bottom: 5 time bins x 10 frequency bins, each time bin has width 2
    top_x = list(range(11))
    top_y = [0, 2, 4, 6, 8, 10]
    bottom_x = [0, 2, 4, 6, 8, 10]
    bottom_y = list(range(11))

    # Underlying chirp defined on a common 10x10 fine grid.
    # Each representation is only a partial view of it because one axis is coarser.
    fine_chirp = [(1, 2), (2, 3), (3, 4), (4, 4), (5, 5)]

    ax_top = fig.add_subplot(gs[0, 0])
    draw_variable_grid(
        ax_top,
        10,
        10,
        top_x,
        top_y,
        "STFT 1: fine time, coarse frequency",
        "10 time bins",
        "5 frequency bins",
    )
    top_chirp = aggregate_cells(fine_chirp, x_group=1, y_group=2)
    fill_cells(ax_top, top_x, top_y, top_chirp, BLUE)
    ax_top.add_patch(Rectangle((7, 4), 1, 2, facecolor=BLUE, edgecolor="none", alpha=0.95))
    ax_top.text(
        5,
        -0.85,
        "Fine in time, merged along frequency",
        ha="center",
        va="top",
        fontsize=9,
        color=GRAY,
    )

    ax_bottom = fig.add_subplot(gs[1, 0])
    draw_variable_grid(
        ax_bottom,
        10,
        10,
        bottom_x,
        bottom_y,
        "STFT 2: coarse time, fine frequency",
        "5 time bins",
        "10 frequency bins",
    )
    bottom_chirp = aggregate_cells(fine_chirp, x_group=2, y_group=1)
    fill_cells(ax_bottom, bottom_x, bottom_y, bottom_chirp, PINK)
    ax_bottom.add_patch(Rectangle((8, 7), 2, 1, facecolor=PINK, edgecolor="none", alpha=0.95))
    ax_bottom.text(
        5,
        -0.85,
        "Fine in frequency, merged along time",
        ha="center",
        va="top",
        fontsize=9,
        color=GRAY,
    )

    aligned_cells = aggregate_cells(fine_chirp, x_group=2, y_group=2)

    ax_mid_top = fig.add_subplot(gs[0, 1])
    draw_uniform_grid(
        ax_mid_top,
        5,
        "After anisotropic downsampling",
        "Downsample along time",
        BLUE,
        aligned_cells,
    )

    ax_mid_bottom = fig.add_subplot(gs[1, 1])
    draw_uniform_grid(
        ax_mid_bottom,
        5,
        "After anisotropic downsampling",
        "Downsample along frequency",
        PINK,
        aligned_cells,
    )

    ax_right = fig.add_subplot(gs[:, 2])
    ax_right.set_xlim(0, 8)
    ax_right.set_ylim(0, 8)
    ax_right.set_aspect("equal")
    ax_right.set_xticks(range(9))
    ax_right.set_yticks(range(9))
    ax_right.grid(color=GRID, linewidth=1)
    ax_right.tick_params(length=0, labelbottom=False, labelleft=False)
    for spine in ax_right.spines.values():
        spine.set_color("#888888")
        spine.set_linewidth(1)
    ax_right.set_title("Channel concatenation on a shared grid", fontsize=12, pad=8)
    ax_right.set_xlabel("Shared time axis", fontsize=10)
    ax_right.set_ylabel("Shared frequency axis", fontsize=10)

    ax_right.add_patch(Rectangle((1.0, 1.0), 5.0, 5.0, facecolor=BLUE, edgecolor="none", alpha=0.38))
    ax_right.add_patch(Rectangle((1.5, 1.5), 5.0, 5.0, facecolor=PINK, edgecolor="none", alpha=0.38))
    for x_idx, y_idx in aligned_cells:
        ax_right.add_patch(Rectangle((1.0 + x_idx, 1.0 + y_idx), 1.0, 1.0, facecolor=GREEN, edgecolor="none", alpha=0.95))
        ax_right.add_patch(Rectangle((1.5 + x_idx, 1.5 + y_idx), 1.0, 1.0, facecolor=GREEN, edgecolor="none", alpha=0.95))
    ax_right.text(6.1, 6.15, "channel 2", fontsize=9, color=GRAY)
    ax_right.text(5.55, 5.6, "channel 1", fontsize=9, color=GRAY)
    ax_right.text(4.0, 0.45, "Same spatial position after downsampling", ha="center", va="center", fontsize=9, color=GRAY)

    add_arrow(fig, ax_top, ax_mid_top)
    add_arrow(fig, ax_bottom, ax_mid_bottom)
    add_arrow(fig, ax_mid_top, ax_right, dy=0.08)
    add_arrow(fig, ax_mid_bottom, ax_right, dy=-0.08)

    fig.suptitle(
        "Simplified alignment of two STFT resolutions before channel concatenation",
        fontsize=15,
        y=1.01,
    )
    return fig


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
