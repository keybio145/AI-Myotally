"""Frozen Figure 4 ECDF and morphometry calculations; portable manuscript adaptation."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
TYPE_ORDER = ["Type I", "Type IIA", "Type IIX", "Type IIB"]
GROUP_ORDER = ["Young", "Aged"]
SAMPLE_ORDER = ["Y1", "Y2", "Y3", "O1", "O2", "O3"]
GROUP_MAP = {"Y": "Young", "O": "Aged"}
GROUP_COLORS = {"Young": "#3867A6", "Aged": "#B45A4A"}
DPI = 300
SOURCE_DIR = None
def save_panel(fig, stem):
    plt.close(fig)
    return {}

def clean_axis(ax: plt.Axes, y_grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D1D5DB")
    ax.spines["bottom"].set_color("#D1D5DB")
    ax.tick_params(length=0, colors="#374151")
    if y_grid:
        ax.grid(axis="y", color="#E8EAED", linewidth=0.8)
        ax.set_axisbelow(True)

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
    out_csv = SOURCE_DIR / "Figure4B_ECDF_source_data.csv"
    out.to_csv(out_csv, index=False)
    paths = save_panel(fig, "Figure4_PanelB_candidate_ECDF_Total_TypeSpecific")
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
    overall.to_csv(SOURCE_DIR / "Figure4C_morphometry_by_animal.csv", index=False)
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
    by_type.to_csv(SOURCE_DIR / "Figure4C_morphometry_by_animal_and_type.csv", index=False)
    return overall, by_type
