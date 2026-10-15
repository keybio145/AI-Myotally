"""Tests for sign alignment, fiber identity, and failure detection."""

import sys
import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jove_repro" / "scripts"))
from reproduce_pca import FEATURES_12, calculate_pcas
from verify_pca_reproduction import compare_analysis, frozen_sources


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.scores = pd.DataFrame({"Image": ["Y1", "Y1", "O1"],
            "Fiber_ID": [1, 2, 1], "Group": ["Young", "Young", "Aged"],
            "Fiber_Type": ["Type IIA", "Type IIB", "Type IIX"],
            "PC1": [1.0, 2.0, -3.0], "PC2": [2.0, -2.0, 0.0]})
        self.coefficients = pd.DataFrame({"Feature": ["CSA_um2", "MinFeret_um"],
            "PC1_coefficient": [0.6, 0.8], "PC2_coefficient": [0.8, -0.6],
            "PC1_explained_variance_pct": [60.0, 60.0],
            "PC2_explained_variance_pct": [40.0, 40.0]})
        self.result = {"scores": self.scores.copy(), "coefficients": self.coefficients.copy()}

    def compare(self, scores=None, coefficients=None):
        return compare_analysis(self.result, self.scores if scores is None else scores,
                                self.coefficients if coefficients is None else coefficients)

    def test_independent_signs_and_shuffled_rows_pass(self):
        self.result["scores"]["PC1"] *= -1
        self.result["coefficients"]["PC1_coefficient"] *= -1
        report = self.compare(self.scores.iloc[::-1], self.coefficients.iloc[::-1])
        self.assertTrue(report["passed"])
        self.assertEqual(report["signs_applied_to_actual"], [-1, 1])

    def test_score_sign_without_matching_coefficient_sign_fails(self):
        self.result["scores"]["PC1"] *= -1
        self.assertFalse(self.compare()["passed"])

    def test_score_coefficient_and_variance_errors_fail(self):
        for table, column in (("scores", "PC1"), ("coefficients", "PC1_coefficient"),
                              ("coefficients", "PC1_explained_variance_pct")):
            with self.subTest(column=column):
                changed = self.result[table].copy()
                self.result[table].loc[0, column] += 0.01
                self.assertFalse(self.compare()["passed"])
                self.result[table] = changed

    def test_components_cannot_be_swapped(self):
        self.result["scores"][["PC1", "PC2"]] = self.scores[["PC2", "PC1"]].to_numpy()
        self.result["coefficients"][["PC1_coefficient", "PC2_coefficient"]] = (
            self.coefficients[["PC2_coefficient", "PC1_coefficient"]].to_numpy())
        self.assertFalse(self.compare()["passed"])

    def test_duplicate_or_changed_fiber_keys_rejected(self):
        for keys in ([1, 1, 1], [1, 99, 1]):
            with self.subTest(keys=keys):
                scores = self.scores.copy()
                scores["Fiber_ID"] = keys
                with self.assertRaises(ValueError):
                    self.compare(scores)

    def test_changed_labels_rejected(self):
        scores = self.scores.copy()
        scores.loc[0, "Fiber_Type"] = "Type IIB"
        with self.assertRaises(ValueError):
            self.compare(scores)

    def test_nonfinite_numbers_rejected(self):
        self.result["scores"].loc[0, "PC1"] = np.nan
        with self.assertRaises(ValueError):
            self.compare()

    def test_original_4f_column_names(self):
        coefficients = self.coefficients.rename(columns={
            "PC1_coefficient": "PC1_Loading", "PC2_coefficient": "PC2_Loading",
            "PC1_explained_variance_pct": "PC1_Explained_Variance_percent",
            "PC2_explained_variance_pct": "PC2_Explained_Variance_percent"})
        scores = self.scores.copy()
        scores["PC1_Explained_Variance_percent"] = 60.0
        scores["PC2_Explained_Variance_percent"] = 40.0
        report = compare_analysis(self.result, scores, coefficients, "figure4f")
        self.assertTrue(report["passed"])


class CohortTests(unittest.TestCase):
    def test_eight_features_use_same_complete_case_cohort(self):
        frame = pd.DataFrame(np.random.default_rng(0).normal(size=(8, 12)), columns=FEATURES_12)
        frame["Image"], frame["Fiber_ID"] = "Y1", np.arange(8)
        frame["Fiber_Type"], frame["Include_In_Analysis"] = "Type IIB", 1
        frame.loc[0, "I_Norm_Score"] = np.nan
        results, counts = calculate_pcas(frame)
        self.assertEqual(counts["complete_case_rows"], 7)
        for result in results.values():
            self.assertNotIn(0, result["scores"]["Fiber_ID"].tolist())


class ManifestTests(unittest.TestCase):
    def test_package_layout_uses_packaged_destination(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            root = Path(folder)
            data = root / "input.csv"
            data.write_bytes(b"x\n1\n")
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["artifact_id", "source_path",
                    "proposed_package_path", "size_bytes", "sha256"])
                writer.writeheader()
                writer.writerow({"artifact_id": "sample", "source_path": "absent/source.csv",
                    "proposed_package_path": "input.csv", "size_bytes": 4,
                    "sha256": hashlib.sha256(data.read_bytes()).hexdigest()})
            self.assertEqual(frozen_sources(manifest, root, "proposed_package_path")["sample"], data)
            data.write_bytes(b"x\n2\n")
            with self.assertRaises(ValueError):
                frozen_sources(manifest, root, "proposed_package_path")


if __name__ == "__main__":
    unittest.main()
