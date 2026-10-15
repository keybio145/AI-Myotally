# AI-Myotally reproducibility scripts

Application version: **v1.0.0**. Scientific archive revision: **2026-10-15**.

This repository provides downstream analysis code, documentation, pinned dependencies and expected tables accompanying the Figure 3/4 reproducibility archive for JoVE manuscript 73893. It includes Figure 3 saved-state display scripts, Figure 4 panel and SourceData scripts, and separately fitted 12-feature and 8-feature descriptive PCA scripts. Application source and trained RF models are excluded.

## Complete data and execution

A repository checkout is a code/documentation mirror. Complete execution requires original IF channels, final label masks, saved decision checkpoints, the measurement table, graph contours and reference figures from the full reproducibility archive. These files are distributed with the Zenodo archive. Expected CSVs are reference outputs; they do not replace input data.

Identify the complete archive by these exact values:

- Filename: AI-Myotally-v1.0.0-figure3-figure4-reproduction-20261015.zip
- SHA256: 3477c0ca5cccf181f1e9d821c2286128790050cab7d8b4a3f8637009270142e3
- Size: 558,649,447 bytes

The original application release is https://doi.org/10.5281/zenodo.21534540. For supplementary data, use the specific Zenodo version that lists the archive above and matches its checksum. The historical application DOI alone does not identify the supplementary archive.

1. Download the complete archive from the corresponding Zenodo record version and verify its SHA256.
2. Extract it to an ASCII-only path. Open a terminal in the extracted archive directory containing run_reproduction.py.
3. Use Python 3.11.9 in an isolated environment.
4. Run these commands **inside the extracted full archive directory**:

       python -m pip install -r requirements-lock.txt
       python -B run_reproduction.py --output NEW_DIRECTORY

The output directory must be new. Add --render-figure4 to generate Figure 4 panel previews. Scripts generate panel content and numerical sources; final Affinity Designer assembly is not replayed. Numerical comparisons use rtol 1e-7 and atol 1e-8. Figure 5 is outside the supplementary archive's scope.

Run tests from that same extracted full archive directory:

    python -B -m unittest discover -s tests -v
    python -B -m unittest discover -s pca/tests -v

## Repository layout

- run_reproduction.py: full-archive verification and reproduction entry point.
- figure3/scripts/: saved-state display code; figure3/ also holds display parameters and expected RGB hashes.
- figure4/scripts/: panel calculations and rendering; figure4/expected/ contains reference tables.
- pca/jove_repro/scripts/: PCA calculation and verification; pca/expected/ holds reference scores and coefficients.
- tests/ and pca/tests/: archived tests, requiring complete-archive inputs where applicable.
- docs/: measurement/state scope, spatial definitions and figure provenance.
- ARCHIVE_SHA256SUMS.txt: original full-archive member manifest, including inputs omitted from this repository.
- CODE_SHA256SUMS.txt: checksums for files in this repository snapshot.

Copied scientific scripts and expected tables retain their archive bytes and folder structure. The original archive README is in docs/FULL_ARCHIVE_README.md. Input/output folders are excluded by .gitignore.

## Interpretation and citation

Fiber types in the supplied measurement table were obtained by image-level calibration with saved operator labels. These values were used for Figure 4 and Representative Results section 4. Reanalysis from raw images can differ when interactive decisions or calibration differ. Saved checkpoints preserve endpoint decisions and selected stages, not a chronological editing history. Comparisons and PCA summaries are descriptive.

Cite the specific Zenodo version containing the identified archive and record the GitHub tag/commit used. State application version v1.0.0 and archive revision 2026-10-15. Manuscript: Kim E., Hwang Y., Baek Y. E., Cho E. J., Lee J. H., and Kim S., "Artificial Intelligence-Assisted Image Analysis of Mouse Skeletal Muscle Histology for Myofiber Morphometry and Aging-Associated Phenotyping" (JoVE manuscript 73893, under revision).

All files in this repository are licensed under CC BY 4.0. See LICENSE.
