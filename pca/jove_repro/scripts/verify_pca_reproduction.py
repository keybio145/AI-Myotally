"""Compare recomputed PCA numbers with manifest-frozen references."""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from reproduce_pca import ANALYSES, SCORE_KEYS, descriptive_statistics, reproduce, sha256_file


RTOL = 1e-7
ATOL = 1e-8
PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def frozen_sources(manifest_path, source_root, path_column="source_path"):
    root = Path(source_root).resolve()
    with Path(manifest_path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Manifest is empty")
    paths = {}
    for row in rows:
        artifact = row["artifact_id"]
        if artifact in paths:
            raise ValueError("Duplicate manifest artifact: " + artifact)
        path = (root / row[path_column]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Manifest source is outside the workspace")
        if path.stat().st_size != int(row["size_bytes"]) or sha256_file(path) != row["sha256"]:
            raise ValueError("Source size/hash mismatch: " + artifact)
        paths[artifact] = path
    return paths


def numeric_check(actual, expected):
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    if actual.shape != expected.shape or actual.size == 0:
        raise ValueError("Numeric comparison has empty or different shapes")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Nonfinite values in comparison")
    difference = np.abs(actual - expected)
    limit = ATOL + RTOL * np.abs(expected)
    return {"passed": bool(np.all(difference <= limit)),
            "max_abs_difference": float(difference.max()),
            "max_tolerance_ratio": float((difference / limit).max()),
            "values_compared": int(actual.size)}


def keyed_reference(actual, expected, keys):
    for label, table in (("actual", actual), ("reference", expected)):
        if table[keys].isna().any().any() or table.duplicated(keys).any():
            raise ValueError("Missing/duplicate keys in " + label)
    actual_index = actual.set_index(keys).index
    reference = expected.set_index(keys)
    if len(actual) != len(expected) or len(actual_index.difference(reference.index)):
        raise ValueError("Actual/reference key sets differ")
    return reference.reindex(actual_index)


def compare_analysis(result, reference_scores, reference_coefficients, style="supplementary"):
    actual_scores = result["scores"]
    actual_coefficients = result["coefficients"]
    scores = keyed_reference(actual_scores, reference_scores, SCORE_KEYS)
    coefficients = keyed_reference(actual_coefficients, reference_coefficients, ["Feature"])
    for label in ("Group", "Fiber_Type"):
        if label in scores and not np.array_equal(
                actual_scores[label].astype(str).to_numpy(), scores[label].astype(str).to_numpy()):
            raise ValueError("Actual/reference labels differ: " + label)
    checks, signs = {}, []
    for pc in ("PC1", "PC2"):
        coefficient_column = pc + ("_Loading" if style == "figure4f" else "_coefficient")
        variance_column = pc + ("_Explained_Variance_percent" if style == "figure4f"
                                else "_explained_variance_pct")
        actual_vector = actual_coefficients[pc + "_coefficient"].to_numpy(float)
        reference_vector = coefficients[coefficient_column].to_numpy(float)
        # A PC's single sign must be applied to both its coefficients and its scores.
        sign = -1 if np.dot(actual_vector, reference_vector) < 0 else 1
        signs.append(sign)
        checks[pc + "_coefficients"] = numeric_check(sign * actual_vector, reference_vector)
        checks[pc + "_scores"] = numeric_check(
            sign * actual_scores[pc].to_numpy(float), scores[pc].to_numpy(float))
        checks[pc + "_explained_variance_pct"] = numeric_check(
            actual_coefficients[pc + "_explained_variance_pct"].to_numpy(float),
            coefficients[variance_column].to_numpy(float))
        if style == "figure4f":
            checks[pc + "_score_table_explained_variance_pct"] = numeric_check(
                np.full(len(scores), actual_coefficients[pc + "_explained_variance_pct"].iloc[0]),
                scores[variance_column].to_numpy(float))
    return {"passed": all(check["passed"] for check in checks.values()),
            "signs_applied_to_actual": signs, "fiber_rows": len(actual_scores),
            "features": len(actual_coefficients), "checks": checks}


def verify(manifest, workspace_root, output):
    source_root = workspace_root if workspace_root is not None else Path(manifest).resolve().parent.parent
    path_column = "source_path" if workspace_root is not None else "proposed_package_path"
    paths = frozen_sources(manifest, source_root, path_column)
    output = Path(output)
    results, run_metadata = reproduce(paths["fiber_measurements"], output / "recomputed")
    comparisons = {
        "pca_12features_vs_adopted_4f": compare_analysis(
            results["pca_12features"], pd.read_csv(paths["reference_4f_scores"]),
            pd.read_csv(paths["reference_4f_loadings"]), style="figure4f"),
    }
    for tag in ANALYSES:
        reference_coefficients = pd.read_csv(paths[tag + "_coefficients"])
        if reference_coefficients["Feature"].tolist() != ANALYSES[tag]:
            raise ValueError("Reference feature order differs: " + tag)
        comparisons[tag + "_vs_adopted_numeric"] = compare_analysis(
            results[tag], pd.read_csv(paths[tag + "_scores"]),
            reference_coefficients)
    metrics, _, _ = descriptive_statistics(results)
    ref_metrics = json.loads(paths["reference_interpretation_metrics"].read_text(encoding="utf-8"))
    descriptive_checks = {}
    mapping = {"pca_12features": "12features", "pca_marker_excluded_8features": "8features_retained"}
    for tag, legacy in mapping.items():
        analysis = metrics["analyses"][tag]
        for typ, value in ref_metrics["PC2_medians_by_type"][legacy].items():
            descriptive_checks[tag + "_PC2_median_" + typ] = numeric_check(
                [analysis["PCs"]["PC2"]["median_by_type"][typ]], [value])
        for row in ref_metrics["eta2_descriptive"]:
            if row["analysis"] == legacy:
                for target, source in (("type", "type"), ("group", "group"), ("section", "animal")):
                    descriptive_checks[tag + "_" + row["PC"] + "_eta2_" + target] = numeric_check(
                        [analysis["PCs"][row["PC"]]["eta2"][target]], [row["eta2_" + source]])
        if analysis["type_counts"].get("Type I") != ref_metrics["Type_I_n"]:
            raise ValueError("Type I count differs")
        medians = analysis["PCs"]["PC2"]["median_by_type"]
        if not medians["Type IIA"] < medians["Type IIX"] < medians["Type IIB"]:
            raise ValueError("Expected PC2 median ordering differs")
    upstream = json.loads(paths["upstream_pca_run_metadata"].read_text(encoding="utf-8"))
    if upstream["cohort"] != run_metadata["cohort"]:
        raise ValueError("Upstream PCA input/cohort differs")
    for tag in ANALYSES:
        for field in ("features", "feature_count", "svd_solver_requested", "svd_solver_used"):
            if upstream["analyses"][tag][field] != run_metadata["analyses"][tag][field]:
                raise ValueError("Upstream PCA metadata differs: " + field)
    comparisons["descriptive_statistics"] = {"passed": all(c["passed"] for c in descriptive_checks.values()),
        "signs_applied_to_actual": [1, 1], "checks": descriptive_checks,
        "scope": "Descriptive only; separately fitted axes; Type I count only"}
    frozen_sources(manifest, source_root, path_column)
    report = {
        "schema_version": 1, "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(item["passed"] for item in comparisons.values()) else "failed",
        "manifest_sha256": sha256_file(manifest), "source_hash_checks": len(paths),
        "input_layout": "workspace" if workspace_root is not None else "package",
        "code_sha256": {"reproduce_pca.py": sha256_file(Path(__file__).with_name("reproduce_pca.py")),
                        "verify_pca_reproduction.py": sha256_file(__file__)},
        "tolerances": {"rtol": RTOL, "atol": ATOL,
                       "rule": "abs(actual-reference) <= atol + rtol * abs(reference)",
                       "pc_order": "preserved", "sign_alignment": "coefficients then same sign on scores"},
        "environment": run_metadata["environment"], "cohort": run_metadata["cohort"],
        "analyses": run_metadata["analyses"], "comparisons": comparisons,
        "reference_numeric_input_sha256": upstream["input_sha256"],
        "restored_input_sha256": run_metadata["input_sha256"],
        "input_binding": "Eight metadata columns added; original 27 parsed fields verified separately by root runner",
        "scope": "Numerical PCA reproduction only; draft figures and spatial recalculation excluded",
        "limitations": ["This report covers numerical comparison only; installation and archive QA are separate"],
    }
    (output / "pca_verification_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=PACKAGE_ROOT / "metadata" / "release_manifest.csv")
    parser.add_argument("--workspace-root", type=Path,
                        help="Original workspace root; omit to use packaged files")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.manifest, args.workspace_root, args.output)
    print(json.dumps({"status": report["status"], "cohort": report["cohort"],
                      "comparisons": {name: {"passed": item["passed"],
                          "max_abs_difference": max(c["max_abs_difference"] for c in item["checks"].values()),
                          "signs": item["signs_applied_to_actual"]}
                                      for name, item in report["comparisons"].items()}}, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
