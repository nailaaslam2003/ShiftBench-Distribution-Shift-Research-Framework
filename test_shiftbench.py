import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shiftbench.data import make_stream
from shiftbench.detectors import two_proportion_pvalue, error_rate_alarms, feature_mean_alarms
from shiftbench.metrics import prediction_metrics, detection_metrics


class TestShiftBench(unittest.TestCase):
    def test_stream_is_reproducible(self):
        a = make_stream(600, 300, 11, "concept_drift")
        b = make_stream(600, 300, 11, "concept_drift")
        self.assertTrue(np.allclose(a["x1"], b["x1"]))
        self.assertEqual(int(a["true_post_change"].sum()), 300)

    def test_stationary_and_invalid_regime(self):
        df = make_stream(600, 300, 2, "stationary")
        self.assertEqual(len(df), 600)
        with self.assertRaises(ValueError):
            make_stream(600, 300, 2, "not_a_regime")

    def test_two_proportion_test(self):
        a = np.array([0, 1] * 50)
        b = np.array([0, 1] * 50)
        self.assertGreater(two_proportion_pvalue(a, b), .9)

    def test_feature_detector_flags_mean_shift(self):
        rng = np.random.default_rng(5)
        x = np.r_[rng.normal(0, 1, 100), rng.normal(4, 1, 200)]
        alarms = feature_mean_alarms(x, 100, .01)
        self.assertTrue(alarms.any())

    def test_metric_invariants(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([.1, .3, .7, .9])
        result = prediction_metrics(y, (p >= .5).astype(int), p, .6)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["coverage"], 1.0)

    def test_detection_delay(self):
        result = detection_metrics(np.array([0, 1, 0]), np.array([100, 120, 140]), 110, 50)
        self.assertEqual(result["detection_delay"], 10)
        self.assertTrue(result["detected_within_tolerance"])

if __name__ == "__main__":
    unittest.main()
