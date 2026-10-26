import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from analysis.common.paths import (
    analysis_cache_dir,
    derivatives_root,
    experiment_rawdata_root,
    figure_output_path,
    subject_directories,
    within_prior_cache_dir,
)


class TestConfiguredPaths(unittest.TestCase):
    def test_data_roots_require_explicit_configuration(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "MOTEST_EXP1_RAWDATA"):
                experiment_rawdata_root(1)

            with self.assertRaisesRegex(RuntimeError, "MOTEST_DERIVATIVES"):
                derivatives_root()

    def test_data_roots_validate_directories(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing = Path(temporary_directory) / "missing"
            with patch.dict(os.environ, {"MOTEST_EXP2_RAWDATA": str(missing)}):
                with self.assertRaisesRegex(FileNotFoundError, "MOTEST_EXP2_RAWDATA"):
                    experiment_rawdata_root(2)

    def test_output_paths_create_configured_directories(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache = Path(temporary_directory) / "cache"
            figures = Path(temporary_directory) / "figures"
            with patch.dict(
                os.environ,
                {
                    "MOTEST_CACHE_DIR": str(cache),
                    "MOTEST_FIGURE_DIR": str(figures),
                },
            ):
                self.assertEqual(analysis_cache_dir(), cache.resolve())
                self.assertEqual(
                    figure_output_path("figure.svg"), figures.resolve() / "figure.svg"
                )
                self.assertTrue(cache.is_dir())
                self.assertTrue(figures.is_dir())

    def test_within_prior_cache_has_separate_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache = Path(temporary_directory) / "cache"
            with patch.dict(os.environ, {"MOTEST_CACHE_DIR": str(cache)}):
                within_prior_cache = within_prior_cache_dir()

            self.assertEqual(within_prior_cache, cache / "within_prior")
            self.assertTrue(within_prior_cache.is_dir())

    def test_subject_directories_selects_one_or_all_participants(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            derivatives = Path(temporary_directory)
            (derivatives / "sub-002").mkdir()
            (derivatives / "sub-001").mkdir()
            (derivatives / "not-a-subject").mkdir()
            with patch.dict(os.environ, {"MOTEST_DERIVATIVES": str(derivatives)}):
                self.assertEqual(
                    [path.name for path in subject_directories()],
                    ["sub-001", "sub-002"],
                )
                self.assertEqual(
                    [path.name for path in subject_directories("sub-002")],
                    ["sub-002"],
                )
                with self.assertRaisesRegex(FileNotFoundError, "sub-003"):
                    subject_directories("sub-003")

    def test_subject_directory_rejects_path_traversal_and_non_subject_names(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch.dict(os.environ, {"MOTEST_DERIVATIVES": temporary_directory}):
                with self.assertRaisesRegex(ValueError, "participant folder name"):
                    subject_directories("..")
                with self.assertRaisesRegex(ValueError, "participant folder name"):
                    subject_directories("participant-001")


if __name__ == "__main__":
    unittest.main()
