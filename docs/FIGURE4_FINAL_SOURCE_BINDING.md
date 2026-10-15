# Figure 4: finalized artwork and numerical source binding

The final reference PDF is `figure4/reference/Figure4_Revision.pdf`, SHA256 `3ec8dea39b4b072b397b7a51a3a38e84318cd7530949331e68560dd0004a0892` (6,715,293 bytes). It was assembled and its labels edited in Affinity Designer 1.10.6 from script-generated panels. The scripts reproduce panels and SourceData, not the final PDF bytes. This revision preserves that PDF without editing it. Final PNG/SVG exports are not included.

The restored 35-column input has 15,373 fibers and SHA256 `8f0bc762e6c5f10db0c5656eb2e3389b5ff0742d535d8247d19b6196e821b8f3`. Its original 27 columns exactly preserve the numerical reference input, SHA256 `7ee8830e2a4139cc0cdfff90534b78b7ff6615b423d661eb3c500c3ad8a5608f`. Source CSV filenames now describe their panels without the old candidate suffix. Their bytes and scientific values are unchanged; legacy Panel/provenance fields are preserved for traceability. FIGURE4_FINAL_BINDING.json maps old and new names.

## SourceData

Panel A: Figure4A_composition_source_data.csv, including the zero Type I count for Old 3. Panel B: Figure4B_ECDF_source_data.csv, individual ECDF points and the pointwise median on a 260-point common grid per facet. The grid is linspace(0, pooled facet CSA 99.3rd percentile, 260); each animal ECDF is interpolated with left=0 and right=1, then the median of the three interpolated curves is taken. The displayed x limit can exceed this grid because it includes the maximum individual CSA.

Panel C: Figure4C_animal_medians_white_circles.csv contains the 24 animal medians. The violin data are pooled fibers from Figure4F_PCA_and_fiber_source_data.csv; the eight horizontal bars are pooled medians. Animal summaries are also supplied. The separate composition_animal_equal.csv and morphometry_animal_equal.csv summarize animal-level values for descriptive Result 4 calculations; they do not imply animal weighting of the pooled violins or PCA.

Panels D/E: fixed measured values and keyed contour files for Young 3 and Old 2. The scripts redraw maps from these values; they do not recalculate spatial measurements from raw images. Units are CSA in square micrometers and density in myofibers per 100 square micrometers. Panel F: separately verified 12-feature PCA coordinates and coefficients; the figure labels round the explained variance to 24.9% and 20.1%. The independent 8-feature fit remains in pca/ for the descriptive sensitivity analysis.

## PDF comparison

The corrected PDF has Title "Figure 4". Its five nonzero Type I rectangles are aligned with the expected 0% stacking baseline (maximum baseline residual 0.000002442 pt). All 24 bar heights match the r4 numerical expected values within 0.000077119 percentage points. Old 3 has zero Type I fibers; its zero-height, unfilled path has no area and was not moved.

All 26,473 drawing paths were compared with the preceding reference PDF: only the geometry of the five corrected rectangles changed; all other vector properties and the page text are identical. The detailed A–F source comparisons are in figure4/verification/. Rendering was also reviewed. PDF export rounding is separate from the numerical CSV tolerance (rtol 1e-7, atol 1e-8).

Figure 5 and its validation data are outside this archive. The reproduction scripts generate panels and SourceData; Affinity Designer assembly is not replayed and byte-identical final PDFs are not claimed.
