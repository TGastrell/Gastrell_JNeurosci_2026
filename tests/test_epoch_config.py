import unittest

from analysis.common.epoch_config import baseline_window, clean_epoch_label, epoch_label


class TestEpochConfig(unittest.TestCase):
    def test_standard_baseline_preserves_existing_cropped_label(self):
        self.assertEqual(
            epoch_label(-0.25, 2, "standard"),
            "t--0.25-2_bl--0.35--0.25-0.2-40bpf",
        )
        self.assertEqual(
            clean_epoch_label(-0.25, 2, "standard"),
            "t--0.25-2_bl--0.35--0.25-0.2-40bpf-clean",
        )

    def test_s5_baseline_full_and_cropped_labels_are_distinct(self):
        self.assertEqual(baseline_window("s5"), (-0.95, -0.85))
        self.assertEqual(
            epoch_label(-0.95, 2, "s5"),
            "t--0.95-2_bl--0.95--0.85-0.2-40bpf",
        )
        self.assertEqual(
            clean_epoch_label(-0.95, 2, "s5"),
            "t--0.95-2_bl--0.95--0.85-0.2-40bpf-clean",
        )
        self.assertEqual(
            clean_epoch_label(-0.25, 2, "s5"),
            "t--0.25-2_bl--0.95--0.85-0.2-40bpf-clean",
        )

    def test_unknown_baseline_has_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "standard, s5"):
            baseline_window("unknown")


if __name__ == "__main__":
    unittest.main()
