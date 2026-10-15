"""Analytic checks for descriptive effect size and separately fitted axes."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jove_repro/scripts"))
from reproduce_pca import descriptive_eta2, descriptive_statistics


class DescriptiveTests(unittest.TestCase):
    def test_known_between_total_ratio(self):
        self.assertAlmostEqual(descriptive_eta2([0, 2, 4, 6], ["a", "a", "b", "b"]), 0.8)

    def test_constant_axis_is_undefined(self):
        self.assertIsNone(descriptive_eta2([1, 1], ["a", "b"]))

    def test_label_and_nonfinite_failures(self):
        for values, labels in (([0, np.inf], ["a", "b"]), ([1], []), ([1, 2], ["a", None])):
            with self.assertRaises(ValueError):
                descriptive_eta2(values, labels)

    def test_sign_invariance_and_single_label(self):
        self.assertEqual(descriptive_eta2([1, 2, 3], ["a"] * 3), 0.0)
        self.assertEqual(descriptive_eta2([0, 2, 4, 6], ["a", "a", "b", "b"]),
                         descriptive_eta2([0, -2, -4, -6], ["a", "a", "b", "b"]))

    def test_statistics_use_each_separate_analysis(self):
        scores = pd.DataFrame({"Image": ["O1", "O1", "Y1", "Y1"], "Group": ["Aged"] * 2 + ["Young"] * 2,
            "Fiber_Type": ["Type IIA"] * 2 + ["Type IIB"] * 2, "PC1": [0, 2, 4, 6], "PC2": [-2, 0, 2, 4]})
        results = {"a": {"scores": scores, "metadata": {"explained_variance_pct": [60, 40]}},
                   "b": {"scores": scores.assign(PC2=[-8, -4, 0, 4]), "metadata": {"explained_variance_pct": [70, 30]}}}
        metrics, eta, medians = descriptive_statistics(results)
        self.assertEqual(metrics["analyses"]["a"]["PCs"]["PC2"]["median_by_type"]["Type IIA"], -1)
        self.assertEqual(metrics["analyses"]["b"]["PCs"]["PC2"]["median_by_type"]["Type IIA"], -6)
        self.assertEqual(len(eta), 4)
        self.assertEqual(len(medians), 8)


if __name__ == "__main__":
    unittest.main()
