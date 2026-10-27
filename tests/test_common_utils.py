import unittest

import numpy as np
import pandas as pd

from analysis.common import utils
from analysis.common.util_dataloader import crop_full_window_metrics


class TestCircularUtilities(unittest.TestCase):
    def test_circular_difference_wraps_across_pi(self):
        actual = utils.circular_difference(-np.pi + 0.1, np.pi - 0.1)

        self.assertTrue(np.isclose(actual, 0.2))

    def test_circular_mean_handles_wraparound(self):
        actual = utils.cmean([np.pi - 0.1, -np.pi + 0.1])

        self.assertTrue(np.isclose(abs(actual), np.pi))

    def test_signed_error_is_positive_toward_prior(self):
        actual = utils.get_signed_error(
            np.array([-0.2, 0.2, 0.2, -0.2]),
            np.array([-1.0, 1.0, -1.0, 1.0]),
        )

        np.testing.assert_allclose(actual, [0.2, 0.2, -0.2, -0.2])


class TestMetricTimeCropping(unittest.TestCase):
    def test_full_window_metrics_are_cropped_and_rebased(self):
        source = pd.DataFrame(
            {
                "time": [0, 178, 179, 180, 754, 755, 756],
                "accuracy": range(7),
            }
        )

        cropped = crop_full_window_metrics(source)

        self.assertEqual(cropped["time"].tolist(), [0, 1, 2, 575, 576])
        self.assertEqual(cropped["accuracy"].tolist(), [2, 3, 4, 5, 6])
        self.assertEqual(source["time"].tolist(), [0, 178, 179, 180, 754, 755, 756])


if __name__ == "__main__":
    unittest.main()
