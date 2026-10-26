import unittest

import numpy as np

from analysis.common import utils


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


if __name__ == "__main__":
    unittest.main()
