"""Behavioral checks: python -m unittest discover -s tests -v."""
import copy
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from carecompass.business import quantify_value
from carecompass.data import NUMERIC, load_data, split_data, validate_inputs
from carecompass.decision import make_decision, encode_from_artifact
from carecompass.model import FeatureEncoder, fit_candidates, build_artifact

ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "config.json").read_text())
        cls.data = load_data(ROOT / cls.config["data_file"])
        cls.fit = fit_candidates(cls.data, None, cls.config)
        cls.artifact, cls.profiles = build_artifact(cls.fit, cls.data, cls.config)

    def test_exact_under_30_boundary(self):
        train, test = split_data(pd.DataFrame({"row": range(29)}))
        self.assertEqual(len(train), 29)
        self.assertIsNone(test)
        train, test = split_data(pd.DataFrame({"row": range(30)}))
        self.assertEqual((len(train), len(test)), (21, 9))
        self.assertTrue(set(train.row).isdisjoint(test.row))

    def test_scaling_never_learns_test_values(self):
        train, _ = split_data(pd.concat([self.data, self.data.iloc[:1]], ignore_index=True))
        encoder = FeatureEncoder.fit(train)
        expected = train[NUMERIC].mean().to_numpy()
        np.testing.assert_allclose(encoder.scaler.mean_, expected)
        extreme = self.data.iloc[:1].copy()
        extreme["hba1c_pct"] = 99
        encoder.transform(extreme)
        np.testing.assert_allclose(encoder.scaler.mean_, expected)

    def test_test_elbow_uses_training_centers(self):
        source = load_data(ROOT / "data/reference/complete_pool512.csv").iloc[:30]
        train, test = split_data(source)
        fit = fit_candidates(train, test, self.config)
        for k, model in fit["models"].items():
            direct = ((fit["x_test"][:, None, :] - model.cluster_centers_[None, :, :]) ** 2).sum(axis=2).min(axis=1).sum()
            reported = fit["curve"].loc[fit["curve"].k.eq(k), "test_sse"].iloc[0]
            self.assertAlmostEqual(float(direct), float(reported), places=8)

    def test_category_distance_is_explicit(self):
        pair = pd.concat([self.data.iloc[:1]] * 2, ignore_index=True)
        pair["sex_recorded"] = ["Female", "Male"]
        encoded = self.fit["encoder"].transform(pair)
        self.assertAlmostEqual(float(np.square(encoded[0] - encoded[1]).sum()), 1.0)

    def test_portable_artifact_assignments_match_sklearn(self):
        artifact = json.loads(json.dumps(self.artifact))
        x = encode_from_artifact(self.data, artifact)
        predictions = [make_decision(row, artifact)["raw_cluster_id"] for row in self.data.to_dict("records")]
        np.testing.assert_array_equal(predictions, self.fit["model"].predict(x))

    def test_missing_and_unsupported_inputs_do_not_get_a_profile(self):
        user = self.data.iloc[0].to_dict()
        user["hba1c_pct"] = None
        with self.assertRaises(ValueError):
            make_decision(user, self.artifact)
        user = self.data.iloc[0].to_dict()
        user["age_years"] = 25
        with self.assertRaises(ValueError):
            make_decision(user, self.artifact)
        user = self.data.iloc[0].to_dict()
        user["sex_recorded"] = "Unknown"
        with self.assertRaises(ValueError):
            make_decision(user, self.artifact)

    def test_retinal_findings_and_optional_fields_cannot_leak_into_fit(self):
        changed = self.data.copy()
        changed["observed_retinopathy"] = np.arange(len(changed))
        changed["bmi_kg_m2"] = 999
        changed["insulin_use"] = "arbitrary excluded text"
        other = fit_candidates(changed, None, self.config)
        np.testing.assert_allclose(other["model"].cluster_centers_, self.fit["model"].cluster_centers_)

    def test_business_value_uses_declared_assumptions(self):
        value = quantify_value(self.artifact, self.config["business_assumptions"])
        expected = value["expected_enhanced_information_users_per_month"] * 2
        self.assertAlmostEqual(value["assumed_monthly_navigation_minutes_saved"], expected)
        identical = copy.deepcopy(self.config["business_assumptions"])
        identical["enhanced_information_navigation_minutes"] = identical["generic_information_navigation_minutes"]
        self.assertAlmostEqual(quantify_value(self.artifact, identical)["assumed_monthly_navigation_minutes_saved"], 0)

    def test_duplicate_people_are_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicated.csv"
            pd.concat([self.data, self.data.iloc[:1]]).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_data(path)


if __name__ == "__main__":
    unittest.main()
