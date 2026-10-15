"""Generate Option 1 Figure 4 candidate panels from current AI-Myotally output.

Existing final Figure 4 panel files are not overwritten. Candidate files are
written with "candidate" in the filename and paired with panel-specific source
CSV files under Figure4/SourceData.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent.parent
FIG_DIR = HERE / "Figure4"
SOURCE_DIR = FIG_DIR / "SourceData"
RESULTS_DIR = WORKSPACE / "Muscle_Fiber" / "OY_myotally_final" / "Results"
DATA_DIR = RESULTS_DIR / "AI_Myotally_Results" / "Data"
GRAPH_DIR = RESULTS_DIR / "_internal" / "Graphs"

TYPE_ORDER = ["Type I", "Type IIA", "Type IIX", "Type IIB"]
TYPE_II_ORDER = ["Type IIA", "Type IIX", "Type IIB"]
GROUP_ORDER = ["Young", "Aged"]
SAMPLE_ORDER = ["Y1", "Y2", "Y3", "O1", "O2", "O3"]
GROUP_MAP = {"Y": "Young", "O": "Aged"}
REPRESENTATIVE_SAMPLES = ["Y3", "O2"]

TYPE_COLORS = {
    "Type I": "#E8A32C",
    "Type IIA": "#3AA655",
    "Type IIX": "#9AA0A6",
    "Type IIB": "#C0392B",
}
GROUP_COLORS = {"Young": "#3867A6", "Aged": "#B45A4A"}
SAMPLE_COLORS = {
    "Y1": "#1F4E79",
    "Y2": "#5B8CC0",
    "Y3": "#9AB7D7",
    "O1": "#8B3E34",
    "O2": "#C46B5A",
    "O3": "#E0A093",
}
GROUP_MARKERS = {"Young": "o", "Aged": "^"}
PCA_FEATURES = [
    "CSA_um2",
    "MinFeret_um",
    "Circularity",
    "Eccentricity",
    "Aspect_Ratio",
    "Neighbor_Count",
    "Local_Density_per_100um2",
    "Local_CSA_CV_percent",
    "Local_Type_Entropy",
    "IIB_Norm_Score",
    "IIA_Norm_Score",
    "I_Norm_Score",
]
DPI = 300
PIXEL_SIZE_UM = 0.36

plt.rcParams.update(
    {
        "font.family": ["Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 8.5,
    }
)


def save_panel(fig: plt.Figure, stem: str) -> dict[str, str]:
    paths = {
        "pdf": FIG_DIR / f"{stem}.pdf",
        "svg": FIG_DIR / f"{stem}.svg",
        "png": FIG_DIR / f"{stem}.png",
    }
    for suffix, path in paths.items():
        if suffix == "png":
            fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
        else:
            fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {key: str(path) for key, path in paths.items()}


def clean_axis(ax: plt.Axes, y_grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D1D5DB")
    ax.spines["bottom"].set_color("#D1D5DB")
    ax.tick_params(length=0, colors="#374151")
    if y_grid:
        ax.grid(axis="y", color="#E8EAED", linewidth=0.8)
        ax.set_axisbelow(True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fiber = pd.read_csv(DATA_DIR / "Fiber_Measurements.csv")
    fiber = fiber[
        (fiber["Include_In_Analysis"] == 1)
        & (fiber["Fiber_Type"].isin(TYPE_ORDER))
    ].copy()
    fiber["Group"] = fiber["Image"].str[0].map(GROUP_MAP)
    if "X_Centroid_um" not in fiber.columns:
        fiber["X_Centroid_um"] = fiber["X_Centroid_px"] * PIXEL_SIZE_UM
    if "Y_Centroid_um" not in fiber.columns:
        fiber["Y_Centroid_um"] = fiber["Y_Centroid_px"] * PIXEL_SIZE_UM
    fiber["Image"] = pd.Categorical(fiber["Image"], SAMPLE_ORDER, ordered=True)
    fiber["Group"] = pd.Categorical(fiber["Group"], GROUP_ORDER, ordered=True)
    fiber["Fiber_Type"] = pd.Categorical(fiber["Fiber_Type"], TYPE_ORDER, ordered=True)

    image_summary = pd.read_csv(DATA_DIR / "Image_Summary.csv")
    image_summary["Group"] = image_summary["Image"].str[0].map(GROUP_MAP)
    image_summary["Image"] = pd.Categorical(image_summary["Image"], SAMPLE_ORDER, ordered=True)
    image_summary["Group"] = pd.Categorical(image_summary["Group"], GROUP_ORDER, ordered=True)

    type_summary = pd.read_csv(DATA_DIR / "Fiber_Type_Summary.csv")
    type_summary["Group"] = type_summary["Image"].str[0].map(GROUP_MAP)
    type_summary["Image"] = pd.Categorical(type_summary["Image"], SAMPLE_ORDER, ordered=True)
    type_summary["Group"] = pd.Categorical(type_summary["Group"], GROUP_ORDER, ordered=True)
    type_summary["Fiber_Type"] = pd.Categorical(type_summary["Fiber_Type"], TYPE_ORDER, ordered=True)
    return fiber, image_summary, type_summary


def panel_a_sample_stacked(fiber: pd.DataFrame) -> dict[str, str]:
    counts = (
        fiber.groupby(["Group", "Image", "Fiber_Type"], observed=True)
        .size()
        .rename("Fiber_Count")
        .reset_index()
    )
    totals = counts.groupby(["Group", "Image"], observed=True)["Fiber_Count"].sum().rename("Total_Fibers")
    comp = counts.merge(totals.reset_index(), on=["Group", "Image"], how="left")
    comp["Proportion_percent"] = 100 * comp["Fiber_Count"] / comp["Total_Fibers"]
    group_stats = (
        comp.groupby(["Group", "Fiber_Type"], observed=True)["Proportion_percent"]
        .agg(Group_Mean_percent="mean", Group_SD_percent="std")
        .reset_index()
    )
    comp = comp.merge(group_stats, on=["Group", "Fiber_Type"], how="left")
    comp["Panel"] = "A_candidate_sampleStacked"
    out_csv = SOURCE_DIR / "Figure4_PanelA_candidate_sampleStacked_source_data.csv"
    comp.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(5.2, 3.35), dpi=DPI)
    x = np.arange(len(SAMPLE_ORDER), dtype=float)
    bottom = np.zeros(len(SAMPLE_ORDER))
    for fiber_type in TYPE_ORDER:
        vals = []
        for sample in SAMPLE_ORDER:
            row = comp[(comp["Image"].astype(str) == sample) & (comp["Fiber_Type"] == fiber_type)]
            vals.append(float(row["Proportion_percent"].iloc[0]) if not row.empty else 0.0)
        ax.bar(
            x,
            vals,
            bottom=bottom,
            width=0.72,
            color=TYPE_COLORS[fiber_type],
            edgecolor="white",
            linewidth=0.75,
            label=fiber_type,
        )
        bottom += np.asarray(vals)

    ax.axvline(2.5, color="#D1D5DB", lw=1.0)
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(SAMPLE_ORDER)
    ax.set_ylabel("Fiber-type composition (%)")
    ax.set_title("A  Fiber-type composition", loc="left", fontweight="bold", pad=8)
    ax.text(1.0, -0.19, "Young", ha="center", va="top", transform=ax.get_xaxis_transform())
    ax.text(4.0, -0.19, "Aged", ha="center", va="top", transform=ax.get_xaxis_transform())
    handles = [mpatches.Patch(color=TYPE_COLORS[t], label=t) for t in TYPE_ORDER[::-1]]
    ax.legend(handles=handles, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    clean_axis(ax)
    fig.subplots_adjust(bottom=0.24, right=0.80)
    paths = save_panel(fig, "Figure4_PanelA_candidate_sampleStacked")
    paths["source_csv"] = str(out_csv)
    return paths


def ecdf_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values[~np.isnan(values)])
    y = np.arange(1, len(x) + 1, dtype=float) / len(x)
    return x, y


def panel_b_ecdf(fiber: pd.DataFrame) -> dict[str, str]:
    facets = [("All", TYPE_ORDER), ("Type IIA", ["Type IIA"]), ("Type IIX", ["Type IIX"]), ("Type IIB", ["Type IIB"])]
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.0), dpi=DPI, sharey=True)
    axes = axes.ravel()
    for ax, (facet, types) in zip(axes, facets):
        sub_facet = fiber[fiber["Fiber_Type"].isin(types)]
        x_max = float(np.nanpercentile(sub_facet["CSA_um2"], 99.3))
        grid = np.linspace(0, x_max, 260)
        for group in GROUP_ORDER:
            sample_curves = []
            for sample in [s for s in SAMPLE_ORDER if GROUP_MAP[s[0]] == group]:
                sub = sub_facet[sub_facet["Image"].astype(str) == sample]
                if sub.empty:
                    continue
                xs, ys = ecdf_values(sub["CSA_um2"].to_numpy(dtype=float))
                for rank, (value, frac) in enumerate(zip(xs, ys), start=1):
                    rows.append(
                        {
                            "Panel": "B_candidate_ECDF_Total_TypeSpecific",
                            "Curve_Type": "individual_sample",
                            "Facet": facet,
                            "Group": group,
                            "Image": sample,
                            "Fiber_Type_Set": ";".join(types),
                            "CSA_um2": value,
                            "ECDF": frac,
                            "Rank_Within_Curve": rank,
                            "Fiber_Count_Curve": len(xs),
                        }
                    )
                idx = np.unique(np.linspace(0, len(xs) - 1, min(180, len(xs))).astype(int))
                ax.plot(xs[idx], ys[idx], color="#9CA3AF", lw=0.8, alpha=0.55, zorder=1)
                sample_curves.append(np.interp(grid, xs, ys, left=0.0, right=1.0))
            if sample_curves:
                median_curve = np.median(np.vstack(sample_curves), axis=0)
                for rank, (value, frac) in enumerate(zip(grid, median_curve), start=1):
                    rows.append(
                        {
                            "Panel": "B_candidate_ECDF_Total_TypeSpecific",
                            "Curve_Type": "group_median_of_samples",
                            "Facet": facet,
                            "Group": group,
                            "Image": "group_median",
                            "Fiber_Type_Set": ";".join(types),
                            "CSA_um2": value,
                            "ECDF": frac,
                            "Rank_Within_Curve": rank,
                            "Fiber_Count_Curve": len(sample_curves),
                        }
                    )
                ax.plot(grid, median_curve, color=GROUP_COLORS[group], lw=2.2, zorder=3)
        title = "All fibers" if facet == "All" else facet
        ax.set_title(title, fontweight="bold", fontsize=9.5)
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("CSA (um2)")
        clean_axis(ax)
    axes[0].set_ylabel("Cumulative fraction")
    axes[2].set_ylabel("Cumulative fraction")
    fig.suptitle("B  CSA distributions by fiber type", x=0.02, y=0.99, ha="left", fontweight="bold")
    handles = [
        mlines.Line2D([], [], color=GROUP_COLORS[g], lw=2.2, label=f"{g} median") for g in GROUP_ORDER
    ]
    handles.append(mlines.Line2D([], [], color="#9CA3AF", lw=0.9, alpha=0.65, label="Individual sample"))
    axes[1].legend(handles=handles, frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = pd.DataFrame(rows)
    out_csv = SOURCE_DIR / "Figure4_PanelB_candidate_ECDF_Total_TypeSpecific_source_data.csv"
    out.to_csv(out_csv, index=False)
    paths = save_panel(fig, "Figure4_PanelB_candidate_ECDF_Total_TypeSpecific")
    paths["source_csv"] = str(out_csv)
    return paths


def panel_b_violin(fiber: pd.DataFrame) -> dict[str, str]:
    facets = [("All", TYPE_ORDER), ("Type IIA", ["Type IIA"]), ("Type IIX", ["Type IIX"]), ("Type IIB", ["Type IIB"])]
    rows = []
    data = []
    positions = []
    colors = []
    for i, (facet, types) in enumerate(facets):
        for j, group in enumerate(GROUP_ORDER):
            vals = fiber[(fiber["Fiber_Type"].isin(types)) & (fiber["Group"].astype(str) == group)]["CSA_um2"].dropna().to_numpy()
            data.append(vals)
            positions.append(i * 2.2 + j * 0.70)
            colors.append(GROUP_COLORS[group])
            for value in vals:
                rows.append(
                    {
                        "Panel": "B_candidate_violin",
                        "Facet": facet,
                        "Group": group,
                        "CSA_um2": value,
                        "Fiber_Type_Set": ";".join(types),
                    }
                )
    fig, ax = plt.subplots(figsize=(6.5, 3.4), dpi=DPI)
    parts = ax.violinplot(data, positions=positions, widths=0.56, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.28)
        body.set_linewidth(0.8)
    for pos, vals, color in zip(positions, data, colors):
        q1, med, q3 = np.nanpercentile(vals, [25, 50, 75])
        ax.vlines(pos, q1, q3, color=color, lw=2.0)
        ax.scatter([pos], [med], s=18, color=color, edgecolor="white", linewidth=0.6, zorder=3)
    for i, (facet, types) in enumerate(facets):
        for j, group in enumerate(GROUP_ORDER):
            sample_medians = []
            for sample in [s for s in SAMPLE_ORDER if GROUP_MAP[s[0]] == group]:
                sub = fiber[(fiber["Image"].astype(str) == sample) & (fiber["Fiber_Type"].isin(types))]
                if not sub.empty:
                    sample_medians.append(float(sub["CSA_um2"].median()))
            offsets = np.linspace(-0.13, 0.13, len(sample_medians))
            ax.scatter(
                i * 2.2 + j * 0.70 + offsets,
                sample_medians,
                s=24,
                facecolor="white",
                edgecolor=GROUP_COLORS[group],
                linewidth=1.0,
                zorder=4,
            )
    ax.set_xticks([i * 2.2 + 0.35 for i in range(len(facets))])
    ax.set_xticklabels(["All", "IIA", "IIX", "IIB"])
    ax.set_ylabel("CSA (um2)")
    ax.set_title("B  CSA distributions", loc="left", fontweight="bold", pad=8)
    clean_axis(ax)
    handles = [mpatches.Patch(color=GROUP_COLORS[g], alpha=0.35, label=g) for g in GROUP_ORDER]
    handles.append(mlines.Line2D([], [], marker="o", linestyle="None", markerfacecolor="white", markeredgecolor="#374151", label="Sample median"))
    ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=8)
    out_csv = SOURCE_DIR / "Figure4_PanelB_candidate_violin_source_data.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    paths = save_panel(fig, "Figure4_PanelB_candidate_violin")
    paths["source_csv"] = str(out_csv)
    return paths


def panel_b_histogram(fiber: pd.DataFrame) -> dict[str, str]:
    facets = [("All", TYPE_ORDER), ("Type IIA", ["Type IIA"]), ("Type IIX", ["Type IIX"]), ("Type IIB", ["Type IIB"])]
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.0), dpi=DPI, sharey=False)
    axes = axes.ravel()
    for ax, (facet, types) in zip(axes, facets):
        sub_facet = fiber[fiber["Fiber_Type"].isin(types)]
        max_x = float(np.nanpercentile(sub_facet["CSA_um2"], 99.3))
        bins = np.linspace(0, max_x, 36)
        for group in GROUP_ORDER:
            vals = sub_facet[sub_facet["Group"].astype(str) == group]["CSA_um2"].dropna().to_numpy()
            hist, edges = np.histogram(vals, bins=bins, density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            ax.step(centers, hist, where="mid", color=GROUP_COLORS[group], lw=2.0, label=group)
            for left, right, center, density in zip(edges[:-1], edges[1:], centers, hist):
                rows.append(
                    {
                        "Panel": "B_candidate_histogram_or_ridgeline",
                        "Facet": facet,
                        "Group": group,
                        "Fiber_Type_Set": ";".join(types),
                        "Bin_Left_CSA_um2": left,
                        "Bin_Right_CSA_um2": right,
                        "Bin_Center_CSA_um2": center,
                        "Density": density,
                        "Fiber_Count_Group_Facet": len(vals),
                    }
                )
        ax.set_title("All fibers" if facet == "All" else facet, fontweight="bold", fontsize=9.5)
        ax.set_xlim(0, max_x)
        ax.set_xlabel("CSA (um2)")
        clean_axis(ax)
    axes[0].set_ylabel("Density")
    axes[2].set_ylabel("Density")
    fig.suptitle("B  CSA histogram overlay", x=0.02, y=0.99, ha="left", fontweight="bold")
    axes[1].legend(frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_csv = SOURCE_DIR / "Figure4_PanelB_candidate_histogram_or_ridgeline_source_data.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    paths = save_panel(fig, "Figure4_PanelB_candidate_histogram_or_ridgeline")
    paths["source_csv"] = str(out_csv)
    return paths


def sample_cv(x: pd.Series) -> float:
    return float(x.std(ddof=1) / x.mean() * 100.0)


def morphometry_tables(fiber: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = (
        fiber.groupby(["Group", "Image"], observed=True)
        .agg(
            Fiber_Count=("Fiber_ID", "count"),
            Mean_CSA_um2=("CSA_um2", "mean"),
            Median_CSA_um2=("CSA_um2", "median"),
            SD_CSA_um2=("CSA_um2", "std"),
            CSA_Variance_um4=("CSA_um2", "var"),
            CSA_CV_percent=("CSA_um2", sample_cv),
            Mean_MinFeret_um=("MinFeret_um", "mean"),
            Median_MinFeret_um=("MinFeret_um", "median"),
            Mean_Circularity=("Circularity", "mean"),
            Median_Circularity=("Circularity", "median"),
        )
        .reset_index()
        .sort_values(["Group", "Image"])
    )
    overall.to_csv(SOURCE_DIR / "Figure4_PanelC_candidate_Morphometry_Heterogeneity_summary.csv", index=False)
    by_type = (
        fiber.groupby(["Group", "Image", "Fiber_Type"], observed=True)
        .agg(
            Fiber_Count=("Fiber_ID", "count"),
            Mean_MinFeret_um=("MinFeret_um", "mean"),
            Median_MinFeret_um=("MinFeret_um", "median"),
            Mean_Circularity=("Circularity", "mean"),
            CSA_CV_percent=("CSA_um2", sample_cv),
        )
        .reset_index()
        .sort_values(["Group", "Image", "Fiber_Type"])
    )
    by_type.to_csv(SOURCE_DIR / "Figure4_PanelC_candidate_TypeSpecific_summary.csv", index=False)
    return overall, by_type


def panel_c_minferet_circularity(fiber: pd.DataFrame, overall: pd.DataFrame) -> dict[str, str]:
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(5.9, 3.25), dpi=DPI)
    metric_specs = [
        ("MinFeret_um", "Mean_MinFeret_um", "MinFeret (um)", "C  MinFeret diameter"),
        ("Circularity", "Mean_Circularity", "Circularity", "   Circularity"),
    ]
    for ax, (fiber_metric, sample_metric, ylabel, title) in zip(axes, metric_specs):
        data = []
        positions = []
        colors = []
        for i, group in enumerate(GROUP_ORDER):
            vals = fiber[fiber["Group"].astype(str) == group][fiber_metric].dropna().to_numpy()
            data.append(vals)
            positions.append(i)
            colors.append(GROUP_COLORS[group])
            for value in vals:
                rows.append(
                    {
                        "Panel": "C_candidate_MinFeret_Circularity",
                        "Metric": fiber_metric,
                        "Curve_Type": "individual_fiber",
                        "Group": group,
                        "Value": value,
                    }
                )
        parts = ax.violinplot(data, positions=positions, widths=0.55, showextrema=False)
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.25)
        for i, group in enumerate(GROUP_ORDER):
            sub = overall[overall["Group"].astype(str) == group]
            offsets = np.linspace(-0.09, 0.09, len(sub))
            ax.scatter(i + offsets, sub[sample_metric], s=32, facecolor="white", edgecolor=GROUP_COLORS[group], linewidth=1.1, zorder=4)
            mean = sub[sample_metric].mean()
            sd = sub[sample_metric].std(ddof=1)
            ax.errorbar([i], [mean], yerr=[sd], fmt="_", color="#111827", ms=16, lw=1.2, capsize=3, zorder=5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(GROUP_ORDER)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold", pad=8)
        if fiber_metric == "Circularity":
            ax.set_ylim(0.45, 1.02)
        clean_axis(ax)
    fig.tight_layout()
    out_csv = SOURCE_DIR / "Figure4_PanelC_candidate_MinFeret_Circularity_source_data.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    paths = save_panel(fig, "Figure4_PanelC_candidate_MinFeret_Circularity")
    paths["source_csv"] = str(out_csv)
    return paths


def panel_c_minferet_cvcsa(overall: pd.DataFrame) -> dict[str, str]:
    plot_df = overall.copy()
    plot_df["Panel"] = "C_candidate_MinFeret_CVCSA"
    out_csv = SOURCE_DIR / "Figure4_PanelC_candidate_MinFeret_CVCSA_source_data.csv"
    plot_df.to_csv(out_csv, index=False)
    fig, axes = plt.subplots(1, 2, figsize=(5.7, 3.15), dpi=DPI)
    specs = [
        ("Mean_MinFeret_um", "Mean MinFeret (um)", "C  MinFeret diameter"),
        ("CSA_CV_percent", "CSA CV (%)", "   CSA heterogeneity"),
    ]
    for ax, (metric, ylabel, title) in zip(axes, specs):
        for i, group in enumerate(GROUP_ORDER):
            sub = plot_df[plot_df["Group"].astype(str) == group]
            x = np.full(len(sub), i) + np.linspace(-0.08, 0.08, len(sub))
            ax.scatter(x, sub[metric], s=34, color=GROUP_COLORS[group], edgecolor="white", linewidth=0.7, zorder=3)
            ax.errorbar([i], [sub[metric].mean()], yerr=[sub[metric].std(ddof=1)], fmt="_", color="#111827", ms=16, lw=1.2, capsize=3, zorder=4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(GROUP_ORDER)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold", pad=8)
        clean_axis(ax)
    fig.tight_layout()
    paths = save_panel(fig, "Figure4_PanelC_candidate_MinFeret_CVCSA")
    paths["source_csv"] = str(out_csv)
    return paths


def panel_c_type_specific_minferet(by_type: pd.DataFrame) -> dict[str, str]:
    plot_df = by_type[by_type["Fiber_Type"].isin(TYPE_II_ORDER)].copy()
    plot_df["Panel"] = "C_candidate_TypeSpecific_MinFeret"
    out_csv = SOURCE_DIR / "Figure4_PanelC_candidate_TypeSpecific_MinFeret_source_data.csv"
    plot_df.to_csv(out_csv, index=False)
    fig, ax = plt.subplots(figsize=(5.8, 3.25), dpi=DPI)
    base = np.arange(len(TYPE_II_ORDER))
    width = 0.30
    for j, group in enumerate(GROUP_ORDER):
        pos = base + (j - 0.5) * width
        means = []
        sds = []
        for fiber_type in TYPE_II_ORDER:
            sub = plot_df[(plot_df["Group"].astype(str) == group) & (plot_df["Fiber_Type"] == fiber_type)]
            means.append(sub["Mean_MinFeret_um"].mean())
            sds.append(sub["Mean_MinFeret_um"].std(ddof=1))
            offsets = np.linspace(-0.06, 0.06, len(sub))
            ax.scatter(
                np.full(len(sub), pos[len(means) - 1]) + offsets,
                sub["Mean_MinFeret_um"],
                s=30,
                facecolor="white",
                edgecolor=GROUP_COLORS[group],
                linewidth=1.0,
                zorder=4,
            )
        ax.errorbar(pos, means, yerr=sds, fmt="o", color=GROUP_COLORS[group], markerfacecolor=GROUP_COLORS[group], markeredgecolor="white", markersize=5.5, lw=1.2, capsize=3, label=group, zorder=3)
    ax.set_xticks(base)
    ax.set_xticklabels(["IIA", "IIX", "IIB"])
    ax.set_ylabel("Mean MinFeret (um)")
    ax.set_title("C  Type-specific MinFeret diameter", loc="left", fontweight="bold", pad=8)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    clean_axis(ax)
    fig.tight_layout()
    paths = save_panel(fig, "Figure4_PanelC_candidate_TypeSpecific_MinFeret")
    paths["source_csv"] = str(out_csv)
    return paths


def representative_ranking(fiber: pd.DataFrame, type_summary: pd.DataFrame) -> pd.DataFrame:
    sample = (
        fiber.groupby(["Group", "Image"], observed=True)
        .agg(
            Fiber_Count=("Fiber_ID", "count"),
            Median_CSA_um2=("CSA_um2", "median"),
            Mean_CSA_um2=("CSA_um2", "mean"),
            Avg_Local_Density_per_100um2=("Local_Density_per_100um2", "mean"),
            CSA_CV_percent=("CSA_um2", sample_cv),
        )
        .reset_index()
    )
    iib = type_summary[type_summary["Fiber_Type"] == "Type IIB"][
        ["Image", "Median_CSA_um2"]
    ].rename(columns={"Median_CSA_um2": "TypeIIB_Median_CSA_um2"})
    sample = sample.merge(iib, on="Image", how="left")
    prop = (
        type_summary.pivot_table(index="Image", columns="Fiber_Type", values="Proportion_percent", observed=True)
        .fillna(0)
        .reset_index()
    )
    for fiber_type in TYPE_ORDER:
        if fiber_type not in prop.columns:
            prop[fiber_type] = 0.0
    prop = prop.rename(columns={t: t.replace(" ", "_") + "_percent" for t in TYPE_ORDER})
    sample = sample.merge(prop, on="Image", how="left")
    metrics = ["Median_CSA_um2", "TypeIIB_Median_CSA_um2", "Avg_Local_Density_per_100um2", "Fiber_Count"]
    comp_cols = [t.replace(" ", "_") + "_percent" for t in TYPE_ORDER]
    ranked = []
    for group, sub in sample.groupby("Group", observed=True):
        tmp = sub.copy()
        for metric in metrics:
            center = tmp[metric].median()
            tmp[f"{metric}_AbsDiffToGroupMedian"] = (tmp[metric] - center).abs()
            tmp[f"{metric}_Rank"] = tmp[f"{metric}_AbsDiffToGroupMedian"].rank(method="min").astype(int)
        comp_center = tmp[comp_cols].median()
        tmp["Composition_Distance_to_GroupMedian"] = np.sqrt(((tmp[comp_cols] - comp_center) ** 2).sum(axis=1))
        tmp["Composition_Rank"] = tmp["Composition_Distance_to_GroupMedian"].rank(method="min").astype(int)
        rank_cols = [f"{m}_Rank" for m in metrics] + ["Composition_Rank"]
        tmp["Rank_Sum"] = tmp[rank_cols].sum(axis=1)
        tmp["Overall_Representative_Rank"] = tmp["Rank_Sum"].rank(method="min").astype(int)
        tmp["Current_Figure4_Heatmap_Sample"] = tmp["Image"].astype(str).isin(REPRESENTATIVE_SAMPLES)
        tmp["Figure3_Sample"] = tmp["Image"].astype(str).isin(["Y1", "O3"])
        ranked.append(tmp)
    out = pd.concat(ranked, ignore_index=True).sort_values(["Group", "Overall_Representative_Rank", "Image"])
    out.to_csv(SOURCE_DIR / "Figure4_Option1_RepresentativeSampleRanking.csv", index=False)
    return out


def sample_pixel_size(sample_df: pd.DataFrame) -> float:
    good = sample_df[(sample_df["X_Centroid_px"] > 0) & (sample_df["X_Centroid_um"] > 0)]
    ratios = good["X_Centroid_um"] / good["X_Centroid_px"]
    if ratios.notna().any():
        return float(ratios.median())
    return PIXEL_SIZE_UM


def load_contours(sample: str, pixel_size_um: float) -> tuple[list[np.ndarray], np.ndarray]:
    z = np.load(GRAPH_DIR / f"{sample}_graph_contours.npz", allow_pickle=True)
    fiber_ids = z["fiber_ids"].astype(int)
    contours = []
    for fid in fiber_ids:
        arr = np.asarray(z[f"contour_{fid}"], dtype=float).reshape(-1, 2)
        contours.append(arr * pixel_size_um)
    return contours, fiber_ids


def choose_scale_bar(sample_df: pd.DataFrame, xlim: tuple[float, float], ylim: tuple[float, float], bar_um: float) -> dict:
    xmin, xmax = xlim
    ymin, ymax = ylim
    pad_x = (xmax - xmin) * 0.035
    pad_y = (ymax - ymin) * 0.035
    candidates = {
        "lower-right": (xmax - pad_x - bar_um, ymax - pad_y),
        "lower-left": (xmin + pad_x, ymax - pad_y),
        "upper-right": (xmax - pad_x - bar_um, ymin + pad_y + 35),
        "upper-left": (xmin + pad_x, ymin + pad_y + 35),
    }
    best_name = None
    best_count = None
    for name, (x0, y0) in candidates.items():
        box_x0, box_x1 = x0 - 30, x0 + bar_um + 30
        box_y0, box_y1 = y0 - 45, y0 + 20
        count = int(
            (
                (sample_df["X_Centroid_um"] >= box_x0)
                & (sample_df["X_Centroid_um"] <= box_x1)
                & (sample_df["Y_Centroid_um"] >= box_y0)
                & (sample_df["Y_Centroid_um"] <= box_y1)
            ).sum()
        )
        if best_count is None or count < best_count:
            best_name = name
            best_count = count
    x0, y0 = candidates[best_name]
    return {"corner": best_name, "x0_um": x0, "y0_um": y0, "overlapping_centroid_count": best_count}


def heatmap_panel(fiber: pd.DataFrame, metric: str, stem: str, title: str, colorbar_label: str) -> dict[str, str]:
    vmin, vmax = np.nanpercentile(fiber[metric], [5, 95])
    cmap = "turbo"
    sample_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.75), dpi=DPI)
    all_collection = None
    for ax, sample in zip(axes, REPRESENTATIVE_SAMPLES):
        sub = fiber[fiber["Image"].astype(str) == sample].copy().sort_values("Fiber_ID")
        px_um = sample_pixel_size(sub)
        contours, contour_fids = load_contours(sample, px_um)
        value_by_id = sub.set_index("Fiber_ID")[metric].to_dict()
        contour_by_id = dict(zip(contour_fids, contours))
        valid = sub[sub["Fiber_ID"].isin(contour_by_id)].copy()
        polys = [contour_by_id[int(fid)] for fid in valid["Fiber_ID"]]
        values = valid[metric].to_numpy(dtype=float)
        norm = Normalize(vmin=vmin, vmax=vmax)
        colors = plt.get_cmap(cmap)(norm(values))
        coll = PolyCollection(
            polys,
            facecolors=colors,
            edgecolors="#1F2937",
            linewidths=0.07,
            rasterized=True,
        )
        ax.add_collection(coll)
        all_collection = coll
        all_pts = np.vstack(polys)
        xmin, ymin = all_pts.min(axis=0)
        xmax, ymax = all_pts.max(axis=0)
        padx = (xmax - xmin) * 0.02
        pady = (ymax - ymin) * 0.02
        extra_bottom = 95.0
        ax.set_xlim(xmin - padx, xmax + padx)
        ax.set_ylim(ymin - pady, ymax + pady + extra_bottom)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_facecolor("#080808")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        group = str(valid["Group"].iloc[0])
        ax.set_title(f"{group} {sample}", loc="left", fontweight="bold", fontsize=10, color="#111827", pad=6)
        bar_um = 250.0
        placement = choose_scale_bar(valid, ax.get_xlim(), tuple(sorted(ax.get_ylim())), bar_um)
        x0 = placement["x0_um"]
        y0 = placement["y0_um"] + 55.0
        # Put the final bar in the reserved lower margin so it avoids tissue.
        x0 = xmax - padx - bar_um
        y0 = ymax + 52.0
        ax.plot([x0, x0 + bar_um], [y0, y0], color="white", lw=2.2, solid_capstyle="butt", zorder=10)
        ax.text(x0 + bar_um / 2, y0 - 14.0, "250 um", color="white", ha="center", va="bottom", fontsize=7.5, zorder=10)
        valid["Panel"] = stem
        valid["Heatmap_Value_Column"] = metric
        valid["Scale_Vmin"] = float(vmin)
        valid["Scale_Vmax"] = float(vmax)
        valid["Scale_Mode"] = "dataset-wide 5th-95th percentile across six samples"
        valid["Colormap"] = cmap
        valid["Scale_Bar_um"] = bar_um
        valid["Scale_Bar_Placement"] = "lower-right reserved margin"
        valid["Pixel_Size_um"] = px_um
        sample_rows.append(valid)
    sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=plt.get_cmap(cmap))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, fraction=0.030, pad=0.018)
    cb.set_label(colorbar_label, fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=0)
    fig.suptitle(title, x=0.02, y=0.98, ha="left", fontweight="bold", fontsize=10)
    fig.subplots_adjust(left=0.02, right=0.90, bottom=0.03, top=0.90, wspace=0.05)
    out_cols = [
        "Panel",
        "Image",
        "Group",
        "Fiber_ID",
        "X_Centroid_px",
        "Y_Centroid_px",
        "X_Centroid_um",
        "Y_Centroid_um",
        "Fiber_Type",
        "CSA_um2",
        "Local_Density_per_100um2",
        "Heatmap_Value_Column",
        "Scale_Vmin",
        "Scale_Vmax",
        "Scale_Mode",
        "Colormap",
        "Scale_Bar_um",
        "Scale_Bar_Placement",
        "Pixel_Size_um",
    ]
    out = pd.concat(sample_rows, ignore_index=True)
    out_csv = SOURCE_DIR / f"{stem}_source_data.csv"
    out[out_cols].to_csv(out_csv, index=False)
    paths = save_panel(fig, stem)
    paths["source_csv"] = str(out_csv)
    return paths


def run_pca(fiber: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pca_df = fiber.dropna(subset=PCA_FEATURES).copy()
    features = pca_df[PCA_FEATURES].astype(float)
    scaled = StandardScaler().fit_transform(features)
    pca = PCA(n_components=2, random_state=0)
    pcs = pca.fit_transform(scaled)
    pca_df["PC1"] = pcs[:, 0]
    pca_df["PC2"] = pcs[:, 1]
    pca_df["PC1_Explained_Variance_percent"] = pca.explained_variance_ratio_[0] * 100.0
    pca_df["PC2_Explained_Variance_percent"] = pca.explained_variance_ratio_[1] * 100.0
    pca_df["Embedding_Method"] = "PCA on standardized AI-Myotally single-fiber features"
    pca_df["PCA_Features"] = ";".join(PCA_FEATURES)
    loadings = pd.DataFrame(
        {
            "Feature": PCA_FEATURES,
            "PC1_Loading": pca.components_[0],
            "PC2_Loading": pca.components_[1],
            "PC1_Explained_Variance_percent": pca.explained_variance_ratio_[0] * 100.0,
            "PC2_Explained_Variance_percent": pca.explained_variance_ratio_[1] * 100.0,
        }
    )
    loadings.to_csv(SOURCE_DIR / "Figure4_PanelF_candidate_PCA_Loadings.csv", index=False)
    cols = [
        "Image",
        "Group",
        "Fiber_ID",
        "Fiber_Type",
        "CSA_um2",
        "MinFeret_um",
        "Circularity",
        "Eccentricity",
        "Aspect_Ratio",
        "Neighbor_Count",
        "Local_Density_per_100um2",
        "Local_CSA_CV_percent",
        "Local_Type_Entropy",
        "IIB_Norm_Score",
        "IIA_Norm_Score",
        "I_Norm_Score",
        "PC1",
        "PC2",
        "PC1_Explained_Variance_percent",
        "PC2_Explained_Variance_percent",
        "Embedding_Method",
        "PCA_Features",
        "X_Centroid_um",
        "Y_Centroid_um",
    ]
    return pca_df[cols].copy(), loadings


def pca_panel_base(ax: plt.Axes, pca_df: pd.DataFrame, title: str) -> None:
    pc1 = pca_df["PC1_Explained_Variance_percent"].iloc[0]
    pc2 = pca_df["PC2_Explained_Variance_percent"].iloc[0]
    ax.set_xlabel(f"PC1 ({pc1:.1f}%)")
    ax.set_ylabel(f"PC2 ({pc2:.1f}%)")
    ax.set_title(title, loc="left", fontweight="bold", pad=8)
    clean_axis(ax)


def panel_f_by_fiber_type(pca_df: pd.DataFrame) -> dict[str, str]:
    out = pca_df.copy()
    out["Panel"] = "F_candidate_byFiberType"
    out_csv = SOURCE_DIR / "Figure4_PanelF_candidate_byFiberType_source_data.csv"
    out.to_csv(out_csv, index=False)
    fig, ax = plt.subplots(figsize=(4.7, 3.65), dpi=DPI)
    for group in GROUP_ORDER:
        for fiber_type in TYPE_ORDER:
            sub = out[(out["Group"].astype(str) == group) & (out["Fiber_Type"].astype(str) == fiber_type)]
            ax.scatter(
                sub["PC1"],
                sub["PC2"],
                s=5.0,
                c=TYPE_COLORS[fiber_type],
                alpha=0.35,
                linewidths=0,
                marker=GROUP_MARKERS[group],
            )
    pca_panel_base(ax, out, "F  Single-fiber feature embedding")
    type_handles = [mlines.Line2D([], [], marker="o", color="none", markerfacecolor=TYPE_COLORS[t], markeredgecolor="none", markersize=5.5, label=t) for t in TYPE_ORDER]
    group_handles = [mlines.Line2D([], [], marker=GROUP_MARKERS[g], color="#374151", markerfacecolor="none", linestyle="None", markersize=5.5, label=g) for g in GROUP_ORDER]
    leg1 = ax.legend(handles=type_handles, frameon=False, loc="upper right", title="Fiber type", fontsize=7.5, title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=group_handles, frameon=False, loc="lower right", title="Group", fontsize=7.5, title_fontsize=8)
    paths = save_panel(fig, "Figure4_PanelF_candidate_byFiberType")
    paths["source_csv"] = str(out_csv)
    return paths


def panel_f_by_group(pca_df: pd.DataFrame) -> dict[str, str]:
    out = pca_df.copy()
    out["Panel"] = "F_candidate_byGroup"
    out_csv = SOURCE_DIR / "Figure4_PanelF_candidate_byGroup_source_data.csv"
    out.to_csv(out_csv, index=False)
    fig, ax = plt.subplots(figsize=(4.5, 3.65), dpi=DPI)
    for group in GROUP_ORDER:
        sub = out[out["Group"].astype(str) == group]
        ax.scatter(sub["PC1"], sub["PC2"], s=5.0, c=GROUP_COLORS[group], alpha=0.30, linewidths=0, marker=GROUP_MARKERS[group], label=group)
    pca_panel_base(ax, out, "F  Embedding colored by group")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    paths = save_panel(fig, "Figure4_PanelF_candidate_byGroup")
    paths["source_csv"] = str(out_csv)
    return paths


def panel_f_by_sample(pca_df: pd.DataFrame) -> dict[str, str]:
    out = pca_df.copy()
    out["Panel"] = "F_candidate_bySample"
    out_csv = SOURCE_DIR / "Figure4_PanelF_candidate_bySample_source_data.csv"
    out.to_csv(out_csv, index=False)
    fig, ax = plt.subplots(figsize=(4.65, 3.65), dpi=DPI)
    for sample in SAMPLE_ORDER:
        group = GROUP_MAP[sample[0]]
        sub = out[out["Image"].astype(str) == sample]
        ax.scatter(sub["PC1"], sub["PC2"], s=4.8, c=SAMPLE_COLORS[sample], alpha=0.32, linewidths=0, marker=GROUP_MARKERS[group], label=sample)
    pca_panel_base(ax, out, "F  Embedding colored by sample")
    ax.legend(frameon=False, loc="upper right", fontsize=7.5, ncol=2, title="Sample", title_fontsize=8)
    paths = save_panel(fig, "Figure4_PanelF_candidate_bySample")
    paths["source_csv"] = str(out_csv)
    return paths


def panel_f_continuous(pca_df: pd.DataFrame, metric: str, stem: str, title: str, label: str, cmap: str) -> dict[str, str]:
    out = pca_df.copy()
    out["Panel"] = stem
    out["Color_Value"] = out[metric]
    out_csv = SOURCE_DIR / f"{stem}_source_data.csv"
    out.to_csv(out_csv, index=False)
    fig, ax = plt.subplots(figsize=(4.85, 3.65), dpi=DPI)
    vmin, vmax = np.nanpercentile(out[metric], [2, 98])
    sc = ax.scatter(out["PC1"], out["PC2"], s=4.8, c=out[metric], cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.35, linewidths=0)
    pca_panel_base(ax, out, title)
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.018)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7, length=0)
    cb.outline.set_visible(False)
    paths = save_panel(fig, stem)
    paths["source_csv"] = str(out_csv)
    return paths


def panel_f_xy_centroid_map(fiber: pd.DataFrame) -> dict[str, str]:
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.45), dpi=DPI)
    for ax, sample in zip(axes, REPRESENTATIVE_SAMPLES):
        sub = fiber[fiber["Image"].astype(str) == sample].copy()
        for fiber_type in TYPE_ORDER:
            s = sub[sub["Fiber_Type"].astype(str) == fiber_type]
            ax.scatter(s["X_Centroid_um"], s["Y_Centroid_um"], s=2.8, c=TYPE_COLORS[fiber_type], alpha=0.75, linewidths=0, label=fiber_type)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor("#080808")
        ax.set_title(f"{str(sub['Group'].iloc[0])} {sample}", loc="left", fontweight="bold", fontsize=10)
        sub["Panel"] = "F_candidate_xyCentroidMap"
        rows.append(sub)
    fig.suptitle("F option  Anatomical centroid map", x=0.02, y=0.98, ha="left", fontweight="bold", fontsize=10)
    handles = [mlines.Line2D([], [], marker="o", linestyle="None", markerfacecolor=TYPE_COLORS[t], markeredgecolor="none", markersize=5, label=t) for t in TYPE_ORDER]
    axes[1].legend(handles=handles, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.subplots_adjust(left=0.02, right=0.84, bottom=0.02, top=0.88, wspace=0.05)
    out = pd.concat(rows, ignore_index=True)
    out_cols = ["Panel", "Image", "Group", "Fiber_ID", "Fiber_Type", "X_Centroid_um", "Y_Centroid_um", "CSA_um2", "MinFeret_um", "Local_Density_per_100um2"]
    out_csv = SOURCE_DIR / "Figure4_PanelF_candidate_xyCentroidMap_source_data.csv"
    out[out_cols].to_csv(out_csv, index=False)
    paths = save_panel(fig, "Figure4_PanelF_candidate_xyCentroidMap")
    paths["source_csv"] = str(out_csv)
    return paths


def audit_tables(fiber: pd.DataFrame, image_summary: pd.DataFrame, type_summary: pd.DataFrame, overall: pd.DataFrame, by_type: pd.DataFrame, loadings: pd.DataFrame) -> dict:
    panel_a = []
    for fiber_type in TYPE_ORDER:
        sub = type_summary[type_summary["Fiber_Type"] == fiber_type]
        y = sub[sub["Group"].astype(str) == "Young"]["Proportion_percent"].to_numpy(dtype=float)
        o = sub[sub["Group"].astype(str) == "Aged"]["Proportion_percent"].to_numpy(dtype=float)
        panel_a.append(
            {
                "Fiber_Type": fiber_type,
                "Young_mean_percent": float(np.mean(y)),
                "Young_sd_percent": float(np.std(y, ddof=1)),
                "Aged_mean_percent": float(np.mean(o)),
                "Aged_sd_percent": float(np.std(o, ddof=1)),
                "Aged_minus_Young_percent": float(np.mean(o) - np.mean(y)),
                "Pairwise_Aged_Greater_Count_of_9": int(sum(oo > yy for oo in o for yy in y)),
            }
        )
    pd.DataFrame(panel_a).to_csv(SOURCE_DIR / "Figure4_Option1_PanelA_Audit.csv", index=False)

    facets = [("All", TYPE_ORDER), ("Type IIA", ["Type IIA"]), ("Type IIX", ["Type IIX"]), ("Type IIB", ["Type IIB"])]
    panel_b = []
    for facet, types in facets:
        sample_medians = (
            fiber[fiber["Fiber_Type"].isin(types)]
            .groupby(["Group", "Image"], observed=True)["CSA_um2"]
            .median()
            .reset_index(name="Sample_Median_CSA_um2")
        )
        y = sample_medians[sample_medians["Group"].astype(str) == "Young"]["Sample_Median_CSA_um2"].to_numpy(dtype=float)
        o = sample_medians[sample_medians["Group"].astype(str) == "Aged"]["Sample_Median_CSA_um2"].to_numpy(dtype=float)
        panel_b.append(
            {
                "Facet": facet,
                "Young_sample_median_mean_um2": float(np.mean(y)),
                "Young_sample_median_sd_um2": float(np.std(y, ddof=1)),
                "Aged_sample_median_mean_um2": float(np.mean(o)),
                "Aged_sample_median_sd_um2": float(np.std(o, ddof=1)),
                "Aged_minus_Young_um2": float(np.mean(o) - np.mean(y)),
                "Aged_All_Below_Young_Min": bool(np.max(o) < np.min(y)),
            }
        )
    pd.DataFrame(panel_b).to_csv(SOURCE_DIR / "Figure4_Option1_PanelB_Audit.csv", index=False)

    c_rows = []
    for metric in ["Mean_MinFeret_um", "Mean_Circularity", "CSA_CV_percent"]:
        row = {"Metric": metric}
        for group in GROUP_ORDER:
            vals = overall[overall["Group"].astype(str) == group][metric].to_numpy(dtype=float)
            row[f"{group}_mean"] = float(np.mean(vals))
            row[f"{group}_sd"] = float(np.std(vals, ddof=1))
            row[f"{group}_min"] = float(np.min(vals))
            row[f"{group}_max"] = float(np.max(vals))
        row["Aged_minus_Young"] = row["Aged_mean"] - row["Young_mean"]
        c_rows.append(row)
    pd.DataFrame(c_rows).to_csv(SOURCE_DIR / "Figure4_Option1_PanelC_Audit.csv", index=False)

    # Categorical eta-squared for PCA axes.
    pca_source = pd.read_csv(SOURCE_DIR / "Figure4_PanelF_candidate_byFiberType_source_data.csv")
    def eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        grand = np.nanmean(values)
        ss_total = np.nansum((values - grand) ** 2)
        ss_between = 0.0
        for label in pd.Series(labels).dropna().unique():
            sub = values[labels == label]
            ss_between += len(sub) * (np.nanmean(sub) - grand) ** 2
        return float(ss_between / ss_total)
    f_rows = []
    for pc in ["PC1", "PC2"]:
        for factor in ["Group", "Fiber_Type", "Image"]:
            f_rows.append({"PC": pc, "Factor": factor, "EtaSquared": eta_squared(pca_source[pc].to_numpy(), pca_source[factor].to_numpy())})
    pd.DataFrame(f_rows).to_csv(SOURCE_DIR / "Figure4_Option1_PanelF_EtaSquared_Audit.csv", index=False)

    summary = {
        "panel_A": panel_a,
        "panel_B": panel_b,
        "panel_C": c_rows,
        "pca_top_loadings_PC1": loadings.assign(abs_loading=loadings["PC1_Loading"].abs()).sort_values("abs_loading", ascending=False).head(6).drop(columns=["abs_loading"]).to_dict(orient="records"),
        "pca_top_loadings_PC2": loadings.assign(abs_loading=loadings["PC2_Loading"].abs()).sort_values("abs_loading", ascending=False).head(6).drop(columns=["abs_loading"]).to_dict(orient="records"),
    }
    with open(SOURCE_DIR / "Figure4_Option1_FinalCandidate_AuditSummary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    fiber, image_summary, type_summary = load_data()
    generated: dict[str, dict[str, str]] = {}

    generated["PanelA_candidate_sampleStacked"] = panel_a_sample_stacked(fiber)
    generated["PanelB_candidate_ECDF_Total_TypeSpecific"] = panel_b_ecdf(fiber)
    generated["PanelB_candidate_violin"] = panel_b_violin(fiber)
    generated["PanelB_candidate_histogram_or_ridgeline"] = panel_b_histogram(fiber)

    overall, by_type = morphometry_tables(fiber)
    generated["PanelC_candidate_MinFeret_Circularity"] = panel_c_minferet_circularity(fiber, overall)
    generated["PanelC_candidate_MinFeret_CVCSA"] = panel_c_minferet_cvcsa(overall)
    generated["PanelC_candidate_TypeSpecific_MinFeret"] = panel_c_type_specific_minferet(by_type)

    ranking = representative_ranking(fiber, type_summary)
    generated["PanelD_candidate_CSA_Heatmap"] = heatmap_panel(
        fiber,
        "CSA_um2",
        "Figure4_PanelD_candidate_CSA_Heatmap",
        "D  Representative CSA heatmaps",
        "CSA (um2)",
    )
    generated["PanelE_candidate_LocalDensity_Heatmap"] = heatmap_panel(
        fiber,
        "Local_Density_per_100um2",
        "Figure4_PanelE_candidate_LocalDensity_Heatmap",
        "E  Representative local-density heatmaps",
        "Local density (fibers/100 um2)",
    )

    pca_df, loadings = run_pca(fiber)
    generated["PanelF_candidate_byFiberType"] = panel_f_by_fiber_type(pca_df)
    generated["PanelF_candidate_byGroup"] = panel_f_by_group(pca_df)
    generated["PanelF_candidate_bySample"] = panel_f_by_sample(pca_df)
    generated["PanelF_candidate_byCSA"] = panel_f_continuous(
        pca_df,
        "CSA_um2",
        "Figure4_PanelF_candidate_byCSA",
        "F option  Embedding colored by CSA",
        "CSA (um2)",
        "plasma",
    )
    generated["PanelF_candidate_byLocalDensity"] = panel_f_continuous(
        pca_df,
        "Local_Density_per_100um2",
        "Figure4_PanelF_candidate_byLocalDensity",
        "F option  Embedding colored by local density",
        "Local density (fibers/100 um2)",
        "viridis",
    )
    generated["PanelF_candidate_xyCentroidMap"] = panel_f_xy_centroid_map(fiber)

    audit = audit_tables(fiber, image_summary, type_summary, overall, by_type, loadings)
    panel_decisions = pd.DataFrame(
        [
            {
                "Panel": "A",
                "Recommended_File_Stem": "Figure4_PanelA_candidate_sampleStacked",
                "Decision": "Use candidate",
                "Reason": "One bar per biological replicate shows composition variability and avoids treating fibers as independent samples.",
            },
            {
                "Panel": "B",
                "Recommended_File_Stem": "Figure4_PanelB_candidate_ECDF_Total_TypeSpecific",
                "Decision": "Use candidate",
                "Reason": "ECDF shows the full leftward CSA shift, adds an all-fiber overview, and uses gray sample lines plus group median curves.",
            },
            {
                "Panel": "C",
                "Recommended_File_Stem": "Figure4_PanelC_candidate_TypeSpecific_MinFeret",
                "Decision": "Use candidate",
                "Reason": "Type-specific MinFeret is the cleanest morphometry summary; circularity is inconsistent and CSA CV is not increased in aged samples.",
            },
            {
                "Panel": "D",
                "Recommended_File_Stem": "Figure4_PanelD_candidate_CSA_Heatmap",
                "Decision": "Use candidate",
                "Reason": "Contour-based heatmap with dataset-wide matched percentile scale and smaller nonoverlapping scale bar.",
            },
            {
                "Panel": "E",
                "Recommended_File_Stem": "Figure4_PanelE_candidate_LocalDensity_Heatmap",
                "Decision": "Use candidate",
                "Reason": "Matched representative samples and dataset-wide scale show spatial packing differences descriptively.",
            },
            {
                "Panel": "F",
                "Recommended_File_Stem": "Figure4_PanelF_candidate_byFiberType",
                "Decision": "Use candidate",
                "Reason": "Fiber-type coloring is biologically interpretable and avoids overstating age separation; group is encoded by marker shape.",
            },
        ]
    )
    panel_decisions.to_csv(SOURCE_DIR / "Figure4_Option1_FinalPanelDecisionTable.csv", index=False)
    manifest = {
        "generated": generated,
        "representative_ranking_csv": str(SOURCE_DIR / "Figure4_Option1_RepresentativeSampleRanking.csv"),
        "decision_table_csv": str(SOURCE_DIR / "Figure4_Option1_FinalPanelDecisionTable.csv"),
        "audit_summary_json": str(SOURCE_DIR / "Figure4_Option1_FinalCandidate_AuditSummary.json"),
        "note": "Existing final Figure 4 panel files were not overwritten.",
    }
    with open(SOURCE_DIR / "Figure4_Option1_Candidate_Manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
