"""Assemble the selected Figure 4 candidates into an A4 JoVE-style draft.

This is an assembly/export script only. It reads existing candidate source data
and contour files, and writes new Figure4_A4_autolayout_draft_v2 outputs.
"""

# R1 revision (2026-10-11), copied from assemble_figure4_a4_autolayout.py (original kept unchanged):
#   - Panel B: individual-animal ECDFs coloured by group (ANIMAL_LW/ANIMAL_ALPHA); legend updated (R2-13)
#   - Labels: um -> \u00b5m, um2 -> \u00b5m\u00b2, "fibers" -> "myofibers" (E1)
#   - Panel B: x labels on both rows, wider row gap; separate export puts the legend outside (submitted layout)
#   - New outputs: OUT_STEM *_v3_R1 (full page); --panels ABCDEF -> Figure4/R1_panels/Figure4_R1_Panel{X}.*
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
FIG_DIR = HERE.parent / "generated"
SOURCE_DIR = HERE.parent / "expected"
GRAPH_DIR = HERE.parent / "graphs"

OUT_STEM = "Figure4_A4_autolayout_draft_v3_R1"
PANEL_DIR = FIG_DIR / "R1_panels"
OUT_PDF = FIG_DIR / f"{OUT_STEM}.pdf"
OUT_SVG = FIG_DIR / f"{OUT_STEM}.svg"
OUT_PNG = FIG_DIR / f"{OUT_STEM}.png"

PAGE_W, PAGE_H = 8.27, 11.69
DPI = 300
FONT = "Arial"
# R1: if Arial is not installed (e.g., Linux), use the metric-compatible Liberation Sans so the layout is unchanged
try:
    from matplotlib import font_manager as _fm
    _fm.findfont(FONT, fallback_to_default=False)
except Exception:
    FONT = "Liberation Sans"
PANEL_LABEL_SIZE = 12
MAIN_LABEL_SIZE = 8
AXIS_TITLE_SIZE = 7.5
TICK_SIZE = 6.5
LEGEND_SIZE = 7

TYPE_ORDER = ["Type I", "Type IIA", "Type IIX", "Type IIB"]
TYPE_II_ORDER = ["Type IIA", "Type IIX", "Type IIB"]
SAMPLE_ORDER = ["Y1", "Y2", "Y3", "O1", "O2", "O3"]
GROUP_ORDER = ["Young", "Aged"]
REPRESENTATIVE_SAMPLES = ["Y3", "O2"]
GROUP_FOR_SAMPLE = {"Y1": "Young", "Y2": "Young", "Y3": "Young", "O1": "Aged", "O2": "Aged", "O3": "Aged"}
GROUP_LABEL = {"Young": "Young", "Aged": "Old"}

TYPE_COLORS = {
    "Type I": "#E8A32C",
    "Type IIA": "#3AA655",
    "Type IIX": "#9AA0A6",
    "Type IIB": "#C0392B",
}
GROUP_COLORS = {"Young": "#3867A6", "Aged": "#B45A4A"}
GROUP_MARKERS = {"Young": "o", "Aged": "^"}
# R1 (R2-13): thin individual-animal ECDF lines in panel B
ANIMAL_LW = 0.6
ANIMAL_ALPHA = 0.45
PANEL_B_GAP_Y = 0.50  # R1: room for x labels on the upper row

plt.rcParams.update(
    {
        "font.family": [FONT, "Liberation Sans", "DejaVu Sans"],
        "font.sans-serif": [FONT, "Liberation Sans", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": TICK_SIZE,
    }
)


def sample_label(sample: str) -> str:
    prefix = "Young" if str(sample).startswith("Y") else "Old"
    return f"{prefix} {str(sample)[1:]}"


def add_axes(fig: plt.Figure, x: float, y: float, w: float, h: float) -> plt.Axes:
    return fig.add_axes([x / PAGE_W, y / PAGE_H, w / PAGE_W, h / PAGE_H])


SHOW_PANEL_LABELS = True


def panel_label(fig: plt.Figure, label: str, x: float, y_top: float) -> None:
    if not SHOW_PANEL_LABELS:
        return
    fig.text(
        x / PAGE_W,
        y_top / PAGE_H,
        label,
        ha="left",
        va="top",
        fontsize=PANEL_LABEL_SIZE,
        fontfamily=FONT,
        fontweight="bold",
        color="#111827",
    )


def style_axis(ax: plt.Axes, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D1D5DB")
    ax.spines["bottom"].set_color("#D1D5DB")
    ax.tick_params(length=0, colors="#374151", labelsize=TICK_SIZE)
    ax.xaxis.label.set_size(AXIS_TITLE_SIZE)
    ax.yaxis.label.set_size(AXIS_TITLE_SIZE)
    ax.xaxis.label.set_fontfamily(FONT)
    ax.yaxis.label.set_fontfamily(FONT)
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_fontfamily(FONT)
    if grid:
        ax.grid(axis="y", color="#E8EAED", linewidth=0.6)
        ax.set_axisbelow(True)


def style_legend(legend) -> None:
    if legend is None:
        return
    for text in legend.get_texts():
        text.set_fontsize(LEGEND_SIZE)
        text.set_fontfamily(FONT)
    title = legend.get_title()
    if title:
        title.set_fontsize(LEGEND_SIZE)
        title.set_fontfamily(FONT)


def add_panel_a(fig: plt.Figure, box: tuple[float, float, float, float]) -> None:
    x, y, w, h = box
    panel_label(fig, "A", x, y + h)
    ax = add_axes(fig, x + 0.42, y + 0.38, 2.36, 1.08)
    comp = pd.read_csv(SOURCE_DIR / "Figure4A_composition_source_data.csv")
    positions = np.arange(len(SAMPLE_ORDER), dtype=float)
    bottom = np.zeros(len(SAMPLE_ORDER))
    for fiber_type in TYPE_ORDER:
        vals = []
        for sample in SAMPLE_ORDER:
            rows = comp[(comp["Image"].astype(str) == sample) & (comp["Fiber_Type"].astype(str) == fiber_type)]
            vals.append(float(rows["Proportion_percent"].iloc[0]) if not rows.empty else 0.0)
        ax.bar(
            positions,
            vals,
            bottom=bottom,
            width=0.72,
            color=TYPE_COLORS[fiber_type],
            edgecolor="white",
            linewidth=0.45,
            label=fiber_type,
        )
        bottom += np.asarray(vals)
    ax.axvline(2.5, color="#D1D5DB", lw=0.8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Fiber-type composition (%)")
    ax.set_xticks(positions)
    ax.set_xticklabels([sample_label(s) for s in SAMPLE_ORDER], rotation=35, ha="right")
    ax.text(1.0, -0.40, "Young", ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=MAIN_LABEL_SIZE)
    ax.text(4.0, -0.40, "Old", ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=MAIN_LABEL_SIZE)
    style_axis(ax)
    handles = [mpatches.Patch(color=TYPE_COLORS[t], label=t) for t in TYPE_ORDER[::-1]]
    leg = ax.legend(handles=handles, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.50), handlelength=1.0)
    style_legend(leg)


def add_panel_b(fig: plt.Figure, box: tuple[float, float, float, float], show_label: bool = True, legend_outside: bool = False) -> None:
    x, y, w, h = box
    if show_label:
        panel_label(fig, "B", x, y + h)
    src = pd.read_csv(SOURCE_DIR / "Figure4B_ECDF_source_data.csv")
    facets = ["All", "Type IIA", "Type IIX", "Type IIB"]
    titles = ["All myofibers", "Type IIA", "Type IIX", "Type IIB"]
    left = x + 0.48
    bottom = y + 0.34
    ax_w = (w - 0.70) / 2
    ax_h = (h - 0.64) / 2
    gap_x, gap_y = 0.26, PANEL_B_GAP_Y
    ax_h = (h - 0.36 - gap_y) / 2
    axes = []
    for i, (facet, title) in enumerate(zip(facets, titles)):
        col = i % 2
        row = 1 - (i // 2)
        ax = add_axes(fig, left + col * (ax_w + gap_x), bottom + row * (ax_h + gap_y), ax_w, ax_h)
        axes.append(ax)
        facet_rows = src[src["Facet"].astype(str) == facet]
        for group in GROUP_ORDER:
            for sample in [s for s in SAMPLE_ORDER if GROUP_FOR_SAMPLE[s] == group]:
                sub = facet_rows[
                    (facet_rows["Curve_Type"].astype(str) == "individual_sample")
                    & (facet_rows["Group"].astype(str) == group)
                    & (facet_rows["Image"].astype(str) == sample)
                ]
                if not sub.empty:
                    ax.plot(sub["CSA_um2"], sub["ECDF"], color=GROUP_COLORS[group], lw=ANIMAL_LW, alpha=ANIMAL_ALPHA, zorder=1)
            med = facet_rows[
                (facet_rows["Curve_Type"].astype(str) == "group_median_of_samples")
                & (facet_rows["Group"].astype(str) == group)
            ]
            if not med.empty:
                ax.plot(med["CSA_um2"], med["ECDF"], color=GROUP_COLORS[group], lw=1.35, zorder=3)
        ax.set_title(title, fontsize=MAIN_LABEL_SIZE, fontfamily=FONT, pad=2)
        ax.set_xlim(0, float(facet_rows["CSA_um2"].max()))
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("CSA (\u00b5m\u00b2)")
        if col == 0:
            ax.set_ylabel("Cumulative fraction")
        else:
            ax.set_ylabel("")
        style_axis(ax)
    # Bold = pointwise median of the 3 animal ECDFs; thin = individual animals (same group colour, translucent)
    handles = []
    for g in GROUP_ORDER:
        handles.append(mlines.Line2D([], [], color=GROUP_COLORS[g], lw=1.35, label=f"{GROUP_LABEL[g]} median"))
        handles.append(mlines.Line2D([], [], color=GROUP_COLORS[g], lw=ANIMAL_LW + 0.1, alpha=ANIMAL_ALPHA + 0.15,
                                     label=f"{GROUP_LABEL[g]} mice (n = 3)"))
    if legend_outside:  # beside the right column, between the two rows (as in the submitted Affinity layout)
        leg = axes[1].legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(1.03, 0.02),
                             handlelength=1.4, labelspacing=0.3)
    else:
        leg = axes[1].legend(handles=handles, frameon=False, loc="lower right", handlelength=1.4, labelspacing=0.3)
    style_legend(leg)


def add_panel_c(fig: plt.Figure, box: tuple[float, float, float, float]) -> None:
    x, y, w, h = box
    panel_label(fig, "C", x, y + h)
    ax = add_axes(fig, x + 0.42, y + 0.34, 2.68, 1.20)
    src = pd.read_csv(SOURCE_DIR / "Figure4F_PCA_and_fiber_source_data.csv")
    facets = [("Total", TYPE_ORDER), ("IIA", ["Type IIA"]), ("IIX", ["Type IIX"]), ("IIB", ["Type IIB"])]
    base = np.arange(len(facets), dtype=float)
    offset = 0.17
    width = 0.26
    for group in GROUP_ORDER:
        positions, data = [], []
        for i, (_, fiber_types) in enumerate(facets):
            sub = src[(src["Group"].astype(str) == group) & (src["Fiber_Type"].astype(str).isin(fiber_types))]
            values = sub["MinFeret_um"].dropna().to_numpy(dtype=float)
            if len(values):
                data.append(values)
                positions.append(base[i] + (-offset if group == "Young" else offset))
        parts = ax.violinplot(data, positions=positions, widths=width, showextrema=False, showmedians=False)
        for body in parts["bodies"]:
            body.set_facecolor(GROUP_COLORS[group])
            body.set_edgecolor(GROUP_COLORS[group])
            body.set_alpha(0.28)
            body.set_linewidth(0.7)
        for pos, values in zip(positions, data):
            med = float(np.nanmedian(values))
            ax.plot([pos - width * 0.34, pos + width * 0.34], [med, med], color=GROUP_COLORS[group], lw=1.1, zorder=4)
        for i, (_, fiber_types) in enumerate(facets):
            for k, sample in enumerate([s for s in SAMPLE_ORDER if GROUP_FOR_SAMPLE[s] == group]):
                sub = src[(src["Image"].astype(str) == sample) & (src["Fiber_Type"].astype(str).isin(fiber_types))]
                if not sub.empty:
                    xpos = base[i] + (-offset if group == "Young" else offset) + (k - 1) * 0.025
                    ax.scatter(xpos, float(sub["MinFeret_um"].median()), s=10, facecolor="white", edgecolor=GROUP_COLORS[group], linewidth=0.6, zorder=5)
    ax.set_xticks(base)
    ax.set_xticklabels([name for name, _ in facets])
    ax.set_ylabel("MinFeret diameter (\u00b5m)")
    style_axis(ax)
    handles = [mpatches.Patch(facecolor=GROUP_COLORS[g], edgecolor=GROUP_COLORS[g], alpha=0.38, label=GROUP_LABEL[g]) for g in GROUP_ORDER]
    leg = ax.legend(handles=handles, frameon=False, loc="upper left")
    style_legend(leg)


def sample_pixel_size(sample_df: pd.DataFrame) -> float:
    good = sample_df[(sample_df["X_Centroid_px"] > 0) & (sample_df["X_Centroid_um"] > 0)]
    ratios = good["X_Centroid_um"] / good["X_Centroid_px"]
    return float(ratios.median()) if ratios.notna().any() else 0.36


def load_contours(sample: str, pixel_size_um: float) -> tuple[list[np.ndarray], np.ndarray]:
    path = GRAPH_DIR / f"{sample}_graph_contours.npz"
    z = np.load(path, allow_pickle=True)
    fiber_ids = z["fiber_ids"].astype(int)
    contours = []
    for fid in fiber_ids:
        arr = np.asarray(z[f"contour_{fid}"], dtype=float).reshape(-1, 2)
        contours.append(arr * pixel_size_um)
    return contours, fiber_ids


def draw_heatmap_axis(ax: plt.Axes, src: pd.DataFrame, sample: str, metric: str, norm: Normalize) -> None:
    sub = src[src["Image"].astype(str) == sample].copy().sort_values("Fiber_ID")
    px_um = sample_pixel_size(sub)
    contours, contour_fids = load_contours(sample, px_um)
    contour_by_id = dict(zip(contour_fids, contours))
    valid = sub[sub["Fiber_ID"].astype(int).isin(contour_by_id)].copy()
    polys = [contour_by_id[int(fid)] for fid in valid["Fiber_ID"]]
    values = valid[metric].to_numpy(dtype=float)
    coll = PolyCollection(
        polys,
        facecolors=plt.get_cmap("turbo")(norm(values)),
        edgecolors="#1F2937",
        linewidths=0.035,
        rasterized=False,
    )
    ax.add_collection(coll)
    all_pts = np.vstack(polys)
    xmin, ymin = all_pts.min(axis=0)
    xmax, ymax = all_pts.max(axis=0)
    padx = (xmax - xmin) * 0.02
    pady = (ymax - ymin) * 0.02
    ax.set_xlim(xmin - padx, xmax + padx)
    ax.set_ylim(ymin - pady, ymax + pady + 95.0)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_facecolor("#080808")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(sample_label(sample), loc="left", fontsize=MAIN_LABEL_SIZE, fontfamily=FONT, pad=2)
    bar_um = 250.0
    x0 = xmax - padx - bar_um
    y0 = ymax + 52.0
    ax.plot([x0, x0 + bar_um], [y0, y0], color="white", lw=1.7, solid_capstyle="butt", zorder=10)
    ax.text(x0 + bar_um / 2, y0 - 14.0, "250 \u00b5m", color="white", ha="center", va="bottom", fontsize=AXIS_TITLE_SIZE, zorder=10)


def add_heatmap_panel(
    fig: plt.Figure,
    box: tuple[float, float, float, float],
    label: str,
    source_file: str,
    metric: str,
    colorbar_label: str,
) -> None:
    x, y, w, h = box
    panel_label(fig, label, x, y + h)
    src = pd.read_csv(SOURCE_DIR / source_file)
    vmin = float(src["Scale_Vmin"].dropna().iloc[0])
    vmax = float(src["Scale_Vmax"].dropna().iloc[0])
    norm = Normalize(vmin=vmin, vmax=vmax)
    ax_y = y + 0.16
    ax_h = h - 0.38
    ax_w = 3.18
    ax1 = add_axes(fig, x + 0.34, ax_y, ax_w, ax_h)
    ax2 = add_axes(fig, x + 3.76, ax_y, ax_w, ax_h)
    draw_heatmap_axis(ax1, src, "Y3", metric, norm)
    draw_heatmap_axis(ax2, src, "O2", metric, norm)
    cax = add_axes(fig, x + 7.12, ax_y + 0.10, 0.10, ax_h - 0.20)
    sm = ScalarMappable(norm=norm, cmap=plt.get_cmap("turbo"))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(colorbar_label, fontsize=AXIS_TITLE_SIZE, fontfamily=FONT)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=TICK_SIZE, length=0)
    for text in cb.ax.get_yticklabels():
        text.set_fontfamily(FONT)


def add_panel_f(fig: plt.Figure, box: tuple[float, float, float, float]) -> None:
    x, y, w, h = box
    panel_label(fig, "F", x, y + h)
    ax = add_axes(fig, x + 0.86, y + 0.28, 1.52, 1.52)
    src = pd.read_csv(SOURCE_DIR / "Figure4F_PCA_and_fiber_source_data.csv")
    for group in GROUP_ORDER:
        for fiber_type in TYPE_ORDER:
            sub = src[(src["Group"].astype(str) == group) & (src["Fiber_Type"].astype(str) == fiber_type)]
            ax.scatter(
                sub["PC1"],
                sub["PC2"],
                s=3.6,
                c=TYPE_COLORS[fiber_type],
                alpha=0.32,
                linewidths=0,
                marker=GROUP_MARKERS[group],
            )
    pc1 = float(src["PC1_Explained_Variance_percent"].iloc[0])
    pc2 = float(src["PC2_Explained_Variance_percent"].iloc[0])
    ax.set_xlabel(f"PC1 ({pc1:.1f}%)")
    ax.set_ylabel(f"PC2 ({pc2:.1f}%)")
    style_axis(ax)
    ax.set_aspect("equal", adjustable="box")
    type_handles = [
        mlines.Line2D([], [], marker="o", color="none", markerfacecolor=TYPE_COLORS[t], markeredgecolor="none", markersize=4.8, label=t)
        for t in TYPE_ORDER
    ]
    group_handles = [
        mlines.Line2D([], [], marker=GROUP_MARKERS[g], color="#374151", markerfacecolor="none", linestyle="None", markersize=4.8, label=GROUP_LABEL[g])
        for g in GROUP_ORDER
    ]
    leg_ax = add_axes(fig, x + 2.72, y + 0.58, 3.9, 0.78)
    leg_ax.axis("off")
    leg1 = leg_ax.legend(handles=type_handles, frameon=False, loc="upper left", ncol=4, title="Fiber type", columnspacing=0.8, handletextpad=0.3)
    style_legend(leg1)
    leg_ax.add_artist(leg1)
    leg2 = leg_ax.legend(handles=group_handles, frameon=False, loc="lower left", ncol=2, title="Group", columnspacing=0.8, handletextpad=0.3)
    style_legend(leg2)


def main() -> None:
    fig = plt.figure(figsize=(PAGE_W, PAGE_H), dpi=DPI, facecolor="white")
    left = 0.48
    right = 0.35
    gap_x = 0.36
    gap_y = 0.18
    usable_w = PAGE_W - left - right
    col_w = (usable_w - gap_x) / 2
    full_w = usable_w

    row0_h, row_b_h, row_d_h, row_e_h, row_f_h = 1.70, 2.80, 2.05, 2.05, 1.85
    top = PAGE_H - 0.35
    row0_y = top - row0_h
    row_b_y = row0_y - gap_y - row_b_h
    row_d_y = row_b_y - gap_y - row_d_h
    row_e_y = row_d_y - gap_y - row_e_h
    row_f_y = row_e_y - gap_y - row_f_h

    add_panel_a(fig, (left, row0_y, col_w, row0_h))
    add_panel_c(fig, (left + col_w + gap_x, row0_y, col_w, row0_h))
    add_panel_b(fig, (left, row_b_y, full_w, row_b_h))
    add_heatmap_panel(
        fig,
        (left, row_d_y, full_w, row_d_h),
        "D",
        "Figure4D_CSA_map_source_data.csv",
        "CSA_um2",
        "CSA (\u00b5m\u00b2)",
    )
    add_heatmap_panel(
        fig,
        (left, row_e_y, full_w, row_e_h),
        "E",
        "Figure4E_density_map_source_data.csv",
        "Local_Density_per_100um2",
        "Local density (myofibers/100 \u00b5m\u00b2)",
    )
    add_panel_f(fig, (left, row_f_y, full_w, row_f_h))

    fig.savefig(OUT_PDF, facecolor="white")
    fig.savefig(OUT_SVG, facecolor="white")
    fig.savefig(OUT_PNG, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(OUT_PDF.resolve())
    print(OUT_SVG.resolve())
    print(OUT_PNG.resolve())


def _export_single(name: str, draw, w: float, h: float, extra_right: float = 0.0) -> None:
    """Render one panel alone at its A4-layout size (no panel letter; letters stay in the Affinity layout)."""
    global PAGE_W, PAGE_H, SHOW_PANEL_LABELS
    pad = 0.10
    fig_w, fig_h = w + 2 * pad + extra_right, h + 2 * pad
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI, facecolor="white")
    saved = (PAGE_W, PAGE_H, SHOW_PANEL_LABELS)
    PAGE_W, PAGE_H, SHOW_PANEL_LABELS = fig_w, fig_h, False
    try:
        draw(fig, (pad, pad, w, h))
    finally:
        PAGE_W, PAGE_H, SHOW_PANEL_LABELS = saved
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        out = PANEL_DIR / f"Figure4_R1_Panel{name}.{ext}"
        fig.savefig(out, dpi=DPI, facecolor="white")
        print(out.resolve())
    plt.close(fig)


def export_panels(which: str = "ABCDEF") -> None:
    left, right, gap_x = 0.48, 0.35, 0.36
    usable_w = PAGE_W - left - right
    col_w = (usable_w - gap_x) / 2
    jobs = {
        "A": (lambda f, b: add_panel_a(f, b), col_w, 1.70, 0.0),
        "B": (lambda f, b: add_panel_b(f, b, show_label=False, legend_outside=True), usable_w - 1.45, 2.80, 1.45),
        "C": (lambda f, b: add_panel_c(f, b), col_w, 1.70, 0.0),
        "D": (lambda f, b: add_heatmap_panel(f, b, "D", "Figure4D_CSA_map_source_data.csv",
                                             "CSA_um2", "CSA (\u00b5m\u00b2)"), usable_w, 2.05, 0.0),
        "E": (lambda f, b: add_heatmap_panel(f, b, "E", "Figure4E_density_map_source_data.csv",
                                             "Local_Density_per_100um2", "Local density (myofibers/100 \u00b5m\u00b2)"), usable_w, 2.05, 0.0),
        "F": (lambda f, b: add_panel_f(f, b), usable_w, 1.85, 0.0),
    }
    for name in which:
        draw, w, h, extra = jobs[name]
        _export_single(name, draw, w, h, extra)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Assemble Figure 4 (A4 draft page) and/or export separate R1 panels for Affinity.")
    parser.add_argument("--panels", default="", help='Export separate panels, e.g. "BCDE" or "ABCDEF" (written to Figure4/R1_panels).')
    parser.add_argument("--no-page", action="store_true", help="Skip the full A4 draft page.")
    args = parser.parse_args()
    if args.panels:
        export_panels(args.panels.upper())
    if not args.no_page:
        main()
