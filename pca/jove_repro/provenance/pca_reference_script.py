"""R2-14 Supplementary Figure 2: PCA with all 12 features (as in Figure 4F) and without
marker-derived features (marker scores and local fiber-type entropy), plus PCA coefficients.

Usage: python r2_14_supp_figure2.py <Fiber_Measurements.csv> <output_dir>
Method (identical to Figure 4F, make_figure4_option1_candidates.py run_pca):
  myofibers with Include_In_Analysis == 1 and an assigned fiber type; rows with missing
  values dropped (none in the demonstration dataset); no transformation; StandardScaler
  fitted on all myofibers pooled across animals; PCA(n_components=2, random_state=0).
Loadings are PCA coefficients (sklearn components_; unit-norm eigenvector weights on
standardized features), not feature-PC correlations. PC signs are arbitrary; each PC is
oriented so that the CSA coefficient is positive (no flip was needed for the 12-feature PCA,
which therefore matches Figure 4F).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from matplotlib import font_manager
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

FM, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

FONT = "Arial"
try:
    font_manager.findfont(FONT, fallback_to_default=False)
except Exception:
    FONT = "Liberation Sans"  # Arial-metric substitute when Arial is unavailable
plt.rcParams.update({"font.family": [FONT, "Liberation Sans", "DejaVu Sans"], "font.size": 7,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})

# Same colors, markers, and draw order as Figure 4F (assemble_figure4_a4_autolayout_v3_R1.py)
TYPE_ORDER = ["Type I", "Type IIA", "Type IIX", "Type IIB"]
TYPE_COLORS = {"Type I": "#E8A32C", "Type IIA": "#3AA655", "Type IIX": "#9AA0A6", "Type IIB": "#C0392B"}
GROUP_ORDER = ["Young", "Aged"]
GROUP_MARKERS = {"Young": "o", "Aged": "^"}
GROUP_LABEL = {"Young": "Young", "Aged": "Old"}

MARKER_DERIVED = ["Local_Type_Entropy", "IIB_Norm_Score", "IIA_Norm_Score", "I_Norm_Score"]
FEAT12 = ["CSA_um2", "MinFeret_um", "Circularity", "Eccentricity", "Aspect_Ratio", "Neighbor_Count",
          "Local_Density_per_100um2", "Local_CSA_CV_percent", "Local_Type_Entropy",
          "IIB_Norm_Score", "IIA_Norm_Score", "I_Norm_Score"]
FEAT8 = [c for c in FEAT12 if c not in MARKER_DERIVED]
LABELS = {"CSA_um2": "CSA", "MinFeret_um": "MinFeret", "Circularity": "Circularity",
          "Eccentricity": "Eccentricity", "Aspect_Ratio": "Aspect ratio", "Neighbor_Count": "Neighbor count",
          "Local_Density_per_100um2": "Local density", "Local_CSA_CV_percent": "Local CSA CV",
          "Local_Type_Entropy": "Local type entropy", "IIB_Norm_Score": "Type IIB score",
          "IIA_Norm_Score": "Type IIA score", "I_Norm_Score": "Type I score"}

f = pd.read_csv(FM)
f = f[(f["Include_In_Analysis"] == 1) & (f["Fiber_Type"].isin(TYPE_ORDER))].copy()
f["Group"] = f["Image"].str[0].map({"Y": "Young", "O": "Aged"})
n_all = len(f)
f = f.dropna(subset=FEAT12)
log = [f"sklearn {sklearn.__version__}", f"myofibers {n_all}; complete cases {len(f)}"]


def run(cols, tag):
    X = StandardScaler().fit_transform(f[cols].astype(float))
    p = PCA(n_components=2, random_state=0).fit(X)
    s = p.transform(X)
    comp = p.components_.copy()
    for i in range(2):
        if comp[i][cols.index("CSA_um2")] < 0:
            comp[i] *= -1
            s[:, i] *= -1
            log.append(f"{tag}: PC{i + 1} sign flipped (CSA coefficient made positive)")
    ev = p.explained_variance_ratio_ * 100
    load = pd.DataFrame({"Feature": cols, "PC1_coefficient": comp[0], "PC2_coefficient": comp[1],
                         "PC1_explained_variance_pct": ev[0], "PC2_explained_variance_pct": ev[1]})
    load.to_csv(OUT / f"SuppFig2_PCA_coefficients_{tag}.csv", index=False)
    sc = f[["Image", "Group", "Fiber_ID", "Fiber_Type"]].copy() if "Fiber_ID" in f else f[["Image", "Group", "Fiber_Type"]].copy()
    sc["PC1"], sc["PC2"] = s[:, 0], s[:, 1]
    sc.to_csv(OUT / f"SuppFig2_PCA_scores_{tag}.csv", index=False)
    log.append(f"{tag}: {len(cols)} features, EV PC1 {ev[0]:.1f}%, PC2 {ev[1]:.1f}%")
    return sc, ev, load.set_index("Feature")


sc12, ev12, ld12 = run(FEAT12, "12features")
sc8, ev8, ld8 = run(FEAT8, "8features")

fig = plt.figure(figsize=(6.8, 6.3), dpi=300, facecolor="white")
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.62], hspace=0.55, wspace=0.32,
                      left=0.17, right=0.95, top=0.88, bottom=0.17)


def scatter(ax, sc, ev, title):
    for g in GROUP_ORDER:
        for t in TYPE_ORDER:
            s = sc[(sc["Group"] == g) & (sc["Fiber_Type"] == t)]
            ax.scatter(s["PC1"], s["PC2"], s=1.6, c=TYPE_COLORS[t], alpha=0.32, linewidths=0,
                       marker=GROUP_MARKERS[g], rasterized=True)
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
    ax.set_title(title, fontsize=7.5)
    ax.set_aspect("equal", adjustable="datalim")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
scatter(axA, sc12, ev12, "12 features (as in Figure 4F)")
scatter(axB, sc8, ev8, "8 features (marker-derived features excluded)")

axC = fig.add_subplot(gs[1, :])
rows = [("12 features, PC1", ld12, "PC1_coefficient"), ("12 features, PC2", ld12, "PC2_coefficient"),
        ("8 features, PC1", ld8, "PC1_coefficient"), ("8 features, PC2", ld8, "PC2_coefficient")]
M = np.full((len(rows), len(FEAT12)), np.nan)
for r, (_, ld, col) in enumerate(rows):
    for c, feat in enumerate(FEAT12):
        if feat in ld.index:
            M[r, c] = ld.loc[feat, col]
cmap = plt.get_cmap("RdBu_r").copy()
cmap.set_bad("#F2F2F2")
im = axC.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=-0.6, vmax=0.6, aspect="auto")
for r in range(M.shape[0]):
    for c in range(M.shape[1]):
        v = M[r, c]
        txt, color = ("–", "#6B7280") if np.isnan(v) else (f"{v:.2f}".replace("-", "−"),
                                                              "white" if abs(v) > 0.4 else "#111827")
        axC.text(c, r, txt, ha="center", va="center", fontsize=5.6, color=color)
axC.set_xticks(range(len(FEAT12)))
axC.set_xticklabels([LABELS[c] for c in FEAT12], rotation=40, ha="right")
axC.set_yticks(range(len(rows)))
axC.set_yticklabels([r[0] for r in rows])
axC.tick_params(length=0)
for sp in axC.spines.values():
    sp.set_visible(False)
axC.set_xticks(np.arange(-0.5, len(FEAT12)), minor=True)
axC.set_yticks(np.arange(-0.5, len(rows)), minor=True)
axC.grid(which="minor", color="white", linewidth=1.0)
axC.tick_params(which="minor", length=0)
cb = fig.colorbar(im, ax=axC, fraction=0.025, pad=0.01)
cb.set_label("PCA coefficient", fontsize=6.5)
cb.ax.tick_params(labelsize=6)
cb.outline.set_visible(False)

fig.canvas.draw()
for lab, ax in zip("ABC", (axA, axB, axC)):
    bb = ax.get_position()
    fig.text(0.02 if lab == "C" else bb.x0 - 0.09, bb.y1 + 0.03, lab, fontsize=12, fontweight="bold", va="bottom")

th = [mlines.Line2D([], [], ls="", marker="o", ms=4, color=TYPE_COLORS[t], label=t) for t in TYPE_ORDER]
gh = [mlines.Line2D([], [], ls="", marker=GROUP_MARKERS[g], ms=4, mfc="none", color="#374151",
                    label=GROUP_LABEL[g]) for g in GROUP_ORDER]
fig.legend(handles=th + gh, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.55, 0.985),
           fontsize=6.5, handletextpad=0.3, columnspacing=1.0)

for ext in ("png", "pdf", "svg"):
    fig.savefig(OUT / f"Supplementary_Figure_2_PCA_R2-14_draft.{ext}", facecolor="white")
(OUT / "SuppFig2_PCA_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
print("\n".join(log))
