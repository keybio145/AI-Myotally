# Spatial feature data dictionary

These definitions describe the AI-Myotally v1.0.0 spatial features. The source-module identity and relevant functions are listed under Source binding.

## Scope and notation

Spatial features are calculated from finalized myofiber objects and the final
instance-label mask at save time. Removed/rejected objects are not supplied as
neighbor nodes. These features describe centroid neighborhoods, not physical
membrane contacts. `Image` identifies a section/export; `Fiber_ID` identifies a
fiber within that section. `Mask_ID` identifies a positive value in the label
mask and is a different identifier. Fiber IDs can be zero.

Let `q` be the linear calibration in micrometers per pixel. In the source,
`q = 1/sqrt(pmr)`, where `pmr` is pixels per square micrometer. Pixel coordinates
are `(x, y)`, with the origin at the upper left; array indexing is `[y, x]`.
Contour-moment centroids are truncated to integer pixels by the inspected
application before spatial calculation. Physical centroids are `q*(x, y)`.
All distances use the Euclidean distance between physical centroids:

`d_ij = sqrt((q*x_i - q*x_j)^2 + (q*y_i - q*y_j)^2)`.

Two different neighborhoods are used:

- `N_i`: endpoints adjacent to fiber `i` in the undirected centroid Delaunay
  graph after removal of edges with `d_ij > 100 um`. Fiber `i` is excluded.
  Edges at exactly 100 um are retained. Neighbor lists are unique and sorted.
- `B_i`: all other finalized centroids satisfying `d_ij <= 100 um`, regardless
  of Delaunay adjacency. Fiber `i` is excluded. Let `n_i = |B_i|`.

For fewer than three centroids, or a triangulation that raises `QhullError`,
the source returns an empty Delaunay graph. This does not prevent circular
neighborhood density from being calculated.

## Per-fiber and edge columns

| Column | Definition / formula | Unit | Neighborhood | Boundary and missing-value handling | Fixed constants / settings |
| --- | --- | --- | --- | --- | --- |
| `Neighbor_IDs` | Sorted unique `Fiber_ID` values of neighbors in `N_i`, serialized with semicolons. | Identifier list | Distance-filtered Delaunay | Empty list for an isolated node or empty/failed triangulation. No image-external neighbors are imputed. Graph edges are not checked for crossing tissue-mask holes or excluded regions. | Maximum centroid distance 100 um; undirected graph; center excluded. |
| `Neighbor_Count` | `|N_i|`; the number of listed Delaunay neighbors. It is not the circular count `n_i` used by density. | Fibers (count) | Distance-filtered Delaunay | Zero for an isolated node or empty/failed triangulation; not missing solely because the count is below two. | Maximum edge length 100 um. |
| `Local_Density_per_100um2` | `rho_i = 100*n_i/A_i`, where `A_i` is the area of the circular window inside both the tissue mask and image. | Fibers per 100 square micrometers | All other centroids in `B_i` | The denominator excludes pixels outside the tissue mask or image. Gaps filled by closing contribute to tissue area. The implementation returns **0 when `A_i = 0`**, rather than a missing value. No minimum of two neighbors applies; `n_i = 0` gives zero density. | Circular radius 100 um; reporting area 100 square micrometers. |
| `Local_CSA_CV_percent` | From finite CSA values of Delaunay neighbors: `mu_i = sum(a_j)/m_i`, `s_i = sqrt(sum((a_j-mu_i)^2)/m_i)`, and `CV_i = 100*s_i/mu_i`. The central fiber is excluded. | Percent | Finite-CSA members of `N_i` | Missing (`NaN` internally) when fewer than two finite CSA values remain. With at least two values, the code returns **0 if `abs(mu_i) <= eps`**, otherwise the formula shown. No neighbor outside the image is added and no boundary extrapolation is applied. | Minimum finite-neighbor count 2; population SD (`ddof=0`); `eps = numpy.finfo(float).eps`. |
| `Local_Type_Entropy` | `H_i = -sum_k(p_ik*ln(p_ik))/ln(K)`, summing only positive proportions. `p_ik` is the proportion of valid typed Delaunay neighbors in category `k`. By default, `K` is the number of distinct nonempty type strings other than literal `null` among finalized fibers in the section. | Dimensionless normalized Shannon entropy | Valid-typed members of `N_i` | Missing when fewer than two valid typed neighbors remain. With at least two, **`K <= 1` gives 0** rather than division by `ln(1)`. Type composition at the image boundary is not imputed or area-corrected. `K` is section-specific, so values using different `K` require care in comparison. | Minimum valid typed-neighbor count 2; natural logarithm; section-derived `K`, not a fixed number of classes. |
| `Source_Fiber`, `Target_Fiber` | Fiber identifiers for the two endpoints of a saved undirected Delaunay edge. Each retained edge is written once. | Identifiers | Distance-filtered Delaunay | Endpoints refer to `Fiber_ID` within `Image`, not `Mask_ID` or a pooled row number. The graph does not test whether the connecting segment stays inside tissue. | Maximum edge length 100 um; no membrane-contact interpretation. |
| `Distance_um` | `d_ij` for the saved edge endpoints. | Micrometers | Distance-filtered Delaunay | The distance cutoff is applied to unrounded floating-point distances before CSV formatting. The official `Spatial_Edges.csv` formats distances to two decimals; internal graph edge exports can use a different precision. | Euclidean centroid distance; retain `d_ij <= 100 um`. |

CSA here is the source object's contour area in square pixels divided by
`pmr`; it is not silently replaced with a mask-pixel count. Only finite CSA
values enter the local CV; this valid-value count may differ from
`Neighbor_Count`.

The default entropy calculation stringifies labels and excludes only empty
strings and the literal `null`. It does not normalize other spellings such as
`nan`, `None`, or `Unassigned` into missing labels. Importers must preserve the
application's missing-label convention. An explicitly supplied `type_order`
can change the valid categories and `K`; the inspected final-save caller does
not supply this override.

## Tissue mask and the density denominator

The tissue mask starts from `(final_masks > 0)`, converted to a binary image,
then closed with `cv2.morphologyEx(..., cv2.MORPH_CLOSE, kernel)` using
`cv2.MORPH_ELLIPSE`. The closing width is selected by the recorded tissue mode:

| Tissue-mode setting | Requested elliptical closing width | Pixel implementation |
| --- | ---: | --- |
| `Intra-fascicular` | 15 um | `k = max(1, int(15/q))`; OpenCV ellipse of shape `(k, k)` pixels. |
| Any other tissue-mode value, including `Whole Muscle` | 100 um | `k = max(1, int(100/q))`; OpenCV ellipse of shape `(k, k)` pixels. |

The requested width is a kernel width, not a radius. Conversion truncates to
an integer and does not force odd kernel dimensions. The realized pixel width
can therefore differ slightly from the requested physical width. The call
does not override OpenCV's default morphology border handling.

The inspected v1.0.0 launcher does not pass `tissue_opt` to
`call_Myotally.main`, whose default is `Whole Muscle`; the preserved R2-12 exact
analysis script also explicitly selects `Whole Muscle`. The 15 um branch is a
module/API option, not a user-selectable launcher setting certified by this
audit. The 100 um description applies to that inspected launcher path.

For fiber `i`, the source considers image pixel positions satisfying
`(x-x_i)^2 + (y-y_i)^2 <= (100/q)^2`. After clipping the search window to image
dimensions, it counts positions whose tissue-mask value is nonzero:

`A_i = q^2 * number_of_pixels_in(circular_window_i AND tissue_mask AND image)`.

Thus `A_i` is a discrete pixel-area estimate, not the continuous area
`pi*r^2`. The numerator uses centroid inclusion, not fiber-area overlap or a
requirement that the centroid-to-centroid segment stays inside tissue.
Closing can fill inter-fiber gaps and some holes; filled pixels are counted as
tissue. The code does not preserve a separate exclusion-region polygon that
is guaranteed to remain excluded after closing.

The final-save caller fixes the Delaunay cutoff, circular radius and reporting
area to 100 um, 100 um and 100 square micrometers respectively. These are
separate parameters with different meanings. A density-radius change is not
equivalent to a graph-cutoff change.

## Image-level spatial summaries

The inspected `core_data.py` uses the following public column names in
`Image_Summary.csv`. They summarize the same spatial context and do not define
new neighbor selection rules. Other package generations should be checked
against their own headers.

| Column | Definition / formula | Unit | Neighborhood | Boundary and missing-value handling | Fixed constants / settings |
| --- | --- | --- | --- | --- | --- |
| `Avg_Local_Density_per_100um2` | Arithmetic mean of per-fiber `Local_Density_per_100um2` over finalized fibers. | Fibers per 100 square micrometers | Circular neighborhoods, then image-level aggregation | Empty fiber collection gives 0. The exporter uses an ordinary mean, rather than a separate finite-value filter. | Same 100 um radius and reporting area as the per-fiber field. |
| `Avg_Local_CSA_CV_percent` | Arithmetic mean of finite per-fiber local CV values. | Percent | Delaunay neighborhoods, then image-level aggregation | Nonfinite per-fiber values are omitted. The inspected `numeric_distribution_summary` returns **0 when no finite values remain**, so a zero image-level mean does not certify that valid local estimates exist. | Per-fiber threshold 2; `ddof=0`. |
| `Avg_Local_Type_Entropy` | Arithmetic mean of finite per-fiber local entropy values. | Dimensionless | Valid-typed Delaunay neighborhoods, then image-level aggregation | Nonfinite per-fiber values are omitted. Returns **0 when no finite values remain** under the inspected core summary helper. | Per-fiber threshold 2; section-specific `K`. |
| `Tissue_Area_um2` | `q^2 * number_of_nonzero_pixels(tissue_mask)`. | Square micrometers | Whole tissue mask | Uses the same closed final-mask union and image boundary as local density. | Mode-dependent 15/100 um closing widths above. |
| `Overall_Fiber_Density_fib_per_mm2` | `N / (tissue_area_um2/1,000,000)`, where `N` is the number of finalized fiber objects. | Fibers per square millimeter | Whole tissue mask | Returns 0 if tissue area is zero. | Conversion factor 1,000,000 square micrometers per square millimeter. |
| `Non_fiber_Tissue_Area_um2`, `Non_fiber_Tissue_Fraction_percent` | `max(0, tissue_area_um2 - sum(CSA_um2))`; divide by tissue area and multiply by 100 for the fraction. | Square micrometers / percent | Whole tissue mask | Negative area differences are clamped to 0. Fraction is 0 when tissue area is zero. This is a geometric residual within the closed tissue mask, not a separately segmented tissue class. | Same tissue mask; percent scale 100. |
| `Same_Type_Edge_Fraction_percent` | `100 * same_type_edge_count / valid_edge_count`. | Percent | Delaunay edges whose endpoints have valid type strings | Excludes endpoints labeled empty string or `null`; missing if no valid edges remain. | Same 100 um graph. |
| `Same_Type_Edge_Enrichment` | `(same_type_edge_count/valid_edge_count) / sum_k[n_k*(n_k-1)/(N_t*(N_t-1))]`, where `N_t` counts all valid-typed section nodes. | Dimensionless ratio | Valid-typed Delaunay edges, with section type composition as the denominator model | Missing if observed fraction is undefined or expected fraction is undefined/nonpositive. The random-mixing expectation does not model spatial packing or image-boundary constraints. | Finite-population random-mixing expectation; descriptive only. |

## Missing values, precision and interpretation

CV and entropy are `NaN` internally when the valid-neighbor threshold is not
met. The inspected core and graph CSV formatters write nonfinite optional
values as `NA`; consumers should retain this missingness and not replace it
with zero. Core measurement CSVs format local density/CV/entropy to 4/3/4
decimal places; graph-node CSVs use 6/4/5 decimal places. Zero-valued CV,
entropy or density can be legitimate outputs under the per-fiber rules above;
image-level summary helpers also have the all-missing fallback described in
the table. Rounded text should not be used to reconstruct exact cutoffs.

Local density and CSA are geometrically related through tissue occupancy;
they should not automatically be treated as independent dimensions. Entropy
depends on the fiber-type labels and their local composition. These are
descriptive outputs and do not by themselves establish typing accuracy,
independent biological replication, or a causal spatial effect.

## Source binding

Inspected spatial source: `Myotally_072426_v1.0.0/spatial_metrics.py`.

SHA256: `82cfccfacfa21f1d8d7f7f216b3e0ad4e9194e0c40ce654dbdc74fa1da9b580a`.

The same SHA256 was observed for
`Myotally_v1.0.0_manual_261006b/spatial_metrics.py` and
`Myotally_092926/ai_myotally/spatial_metrics.py`. Relevant source functions are
`delaunay_edges_within_cutoff`, `neighbor_ids_from_edges`,
`edge_corrected_local_density`, `build_tissue_mask`, `local_feature_cv`,
`local_type_entropy`, `analyze_finalized_fiber_spatial_context`,
`apply_spatial_context_to_fibers`, and `same_type_neighbor_enrichment`.
The final-save parameter binding was read from `call_Myotally.py`, and output
field/formatting bindings from `core_data.py`, `graph_outputs.py` and
`output_layout.py`.

This dictionary records methods only. It contains no dataset-specific edge ratios, section-specific class counts or sensitivity results. The definitions describe calculations from finalized objects; they do not imply regeneration of segmentation or typing from raw images.
