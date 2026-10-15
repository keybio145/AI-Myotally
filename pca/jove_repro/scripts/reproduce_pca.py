"""Recompute the two PCA analyses from a final fiber measurement CSV."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


FEATURES_12 = [
    "CSA_um2", "MinFeret_um", "Circularity", "Eccentricity", "Aspect_Ratio",
    "Neighbor_Count", "Local_Density_per_100um2", "Local_CSA_CV_percent",
    "Local_Type_Entropy", "IIB_Norm_Score", "IIA_Norm_Score", "I_Norm_Score",
]
MARKER_DERIVED = [
    "Local_Type_Entropy", "IIB_Norm_Score", "IIA_Norm_Score", "I_Norm_Score",
]
FEATURES_8 = [feature for feature in FEATURES_12 if feature not in MARKER_DERIVED]
TYPE_ORDER = ["Type I", "Type IIA", "Type IIX", "Type IIB"]
ANALYSES = {"pca_12features": FEATURES_12,
            "pca_marker_excluded_8features": FEATURES_8}
SCORE_KEYS = ["Image", "Fiber_ID"]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment_versions():
    names = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib"]
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "machine": platform.machine(),
        "packages": {name: importlib.metadata.version(name) for name in names},
    }


def descriptive_eta2(values, labels):
    """Between-label / total sums of squares; descriptive, without a test."""
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    if values.ndim != 1 or values.size == 0 or values.shape != labels.shape:
        raise ValueError("Descriptive eta squared needs equally sized nonempty vectors")
    if not np.isfinite(values).all() or pd.isna(labels).any():
        raise ValueError("Missing/nonfinite descriptive inputs")
    mean = float(values.mean())
    total = float(np.sum((values - mean) ** 2))
    if total == 0:
        return None
    between = sum(int(np.sum(labels == label)) * (float(values[labels == label].mean()) - mean) ** 2
                  for label in pd.unique(labels))
    return float(between / total)


def descriptive_statistics(results):
    analyses, eta_rows, median_rows = {}, [], []
    for tag, result in results.items():
        scores = result["scores"]
        counts = scores.groupby("Fiber_Type", observed=True).size().to_dict()
        pcs = {}
        for pc in ("PC1", "PC2"):
            medians = scores.groupby("Fiber_Type", observed=True)[pc].median().to_dict()
            eta = {label: descriptive_eta2(scores[pc], scores[column]) for label, column in
                   (("type", "Fiber_Type"), ("group", "Group"), ("section", "Image"))}
            pcs[pc] = {"median_by_type": medians, "eta2": eta}
            eta_rows.append({"analysis": tag, "PC": pc, "eta2_type": eta["type"],
                             "eta2_group": eta["group"], "eta2_section": eta["section"]})
            for typ in TYPE_ORDER:
                if typ in counts:
                    median_rows.append({"analysis": tag, "PC": pc, "Fiber_Type": typ,
                                        "n": int(counts[typ]), "median": float(medians[typ])})
        analyses[tag] = {"type_counts": {str(k): int(v) for k, v in counts.items()}, "PCs": pcs,
                         "explained_variance_pct": result["metadata"]["explained_variance_pct"]}
    metrics = {"schema_version": 1, "analyses": analyses,
               "axis_relationship": "separately fitted axes; no shared coordinate system",
               "eta2_definition": "between-label sum of squares / total sum of squares; undefined for zero total variance",
               "type_I_interpretation": "count only; n=23 in adopted inputs, no substantive interpretation",
               "scope": "Pooled per-fiber descriptive statistics; no inferential test, classification validation or animal-level aging inference"}
    return metrics, pd.DataFrame(eta_rows), pd.DataFrame(median_rows)


def calculate_pcas(frame):
    required = SCORE_KEYS + ["Fiber_Type", "Include_In_Analysis"] + FEATURES_12
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError("Missing input columns: " + ", ".join(missing))
    selected = frame.loc[(frame["Include_In_Analysis"] == 1)
                         & frame["Fiber_Type"].isin(TYPE_ORDER)].copy()
    selected["Group"] = selected["Image"].str[0].map({"Y": "Young", "O": "Aged"})
    if selected["Group"].isna().any():
        raise ValueError("An included image has no declared Young/Aged group")
    # Both analyses use the complete 12-feature cohort, as in the reference script.
    fiber = selected.dropna(subset=FEATURES_12).copy()
    if len(fiber) < 2 or fiber[SCORE_KEYS].isna().any().any():
        raise ValueError("PCA requires at least two rows and nonmissing fiber keys")
    if fiber.duplicated(SCORE_KEYS).any():
        raise ValueError("Duplicate (Image, Fiber_ID) keys in input")
    counts = {"input_rows": len(frame), "eligible_rows": len(selected),
              "complete_case_rows": len(fiber),
              "missing_feature_rows_excluded": len(selected) - len(fiber)}
    results = {}
    for tag, features in ANALYSES.items():
        values = fiber[features].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("PCA input includes infinite feature values")
        standardized = StandardScaler().fit_transform(values)
        pca = PCA(n_components=2, random_state=0).fit(standardized)
        coordinates = pca.transform(standardized)
        coefficients = pca.components_.copy()
        signs = np.where(coefficients[:, features.index("CSA_um2")] < 0, -1, 1)
        coefficients *= signs[:, None]
        coordinates *= signs[None, :]
        scores = fiber[["Image", "Group", "Fiber_ID", "Fiber_Type"]].copy()
        scores["PC1"], scores["PC2"] = coordinates[:, 0], coordinates[:, 1]
        variance = pca.explained_variance_ratio_ * 100
        loadings = pd.DataFrame({
            "Feature": features,
            "PC1_coefficient": coefficients[0], "PC2_coefficient": coefficients[1],
            "PC1_explained_variance_pct": variance[0],
            "PC2_explained_variance_pct": variance[1],
        })
        results[tag] = {
            "scores": scores, "coefficients": loadings,
            "metadata": {"features": features, "feature_count": len(features),
                         "csa_orientation_signs": signs.tolist(),
                         "svd_solver_requested": "auto",
                         "svd_solver_used": pca._fit_svd_solver,
                         "explained_variance_pct": variance.tolist()},
        }
    return results, counts


def reproduce(input_path, output_dir):
    input_hash = sha256_file(input_path)
    results, counts = calculate_pcas(pd.read_csv(input_path))
    if sha256_file(input_path) != input_hash:
        raise ValueError("Input changed during PCA computation")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for tag, result in results.items():
        for kind in ("scores", "coefficients"):
            result[kind].to_csv(output / f"{tag}_{kind}.csv", index=False)
    metrics, eta, medians = descriptive_statistics(results)
    eta.to_csv(output / "pca_eta2_descriptive.csv", index=False)
    medians.to_csv(output / "pca_pc_medians_by_type.csv", index=False)
    (output / "interpretation_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    metadata = {
        "input_sha256": input_hash, "cohort": counts,
        "environment": environment_versions(),
        "preprocessing": "Untransformed features; StandardScaler fit on pooled complete cases",
        "pca": {"n_components": 2, "random_state": 0,
                "coordinate_method": "fit followed by transform",
                "coefficient_definition": "components_ on standardized features",
                "output_orientation": "Positive CSA coefficient for each PC"},
        "analyses": {tag: result["metadata"] for tag, result in results.items()},
    }
    (output / "pca_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return results, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Final Fiber_Measurements.csv")
    parser.add_argument("--output", type=Path, required=True, help="Directory for numerical outputs")
    args = parser.parse_args()
    results, metadata = reproduce(args.input, args.output)
    metrics, _, _ = descriptive_statistics(results)
    print(json.dumps({"complete_case_rows": metadata["cohort"]["complete_case_rows"],
                      "analyses": metrics["analyses"], "scope": metrics["scope"]}, indent=2))


if __name__ == "__main__":
    main()
