# PCA reproduction

Run the root runner or `python -B pca/jove_repro/scripts/verify_pca_reproduction.py --output NEW_DIRECTORY`. The manifest binds the restored 35-column input and unchanged numerical references. The original numerical reference metadata retains its 27-column input hash. The verifier compares cohort/features/settings and all coefficients/scores; a difference in file hash caused by the added eight metadata columns is recorded separately, not treated as a changed PCA feature value. The root runner verifies the original 27 parsed values' digest.

Both analyses use untransformed features, pooled complete-case StandardScaler, two PCA components, random_state=0, fit then transform, and CSA-positive sign orientation. The axes are fitted separately. Descriptive eta-squared and per-type PC medians are output. Type I has 23 observations and is not substantively interpreted. Python/package environment metadata are written with each run.
