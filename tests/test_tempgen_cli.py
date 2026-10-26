import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from analysis.decoding.forward_encoding import FE_tempgen_localDev
from analysis.common.tempgen_utils import _aggregate_needs_refresh, _subject_directories


class TestTemporalGeneralisationCLI(unittest.TestCase):
    def test_subject_option_runs_only_selected_participant(self):
        selected_subject = Path("sub-001")
        with (
            patch.object(
                FE_tempgen_localDev,
                "subject_directories",
                return_value=[selected_subject],
            ) as subject_directories,
            patch.object(FE_tempgen_localDev, "run_subject") as run_subject,
        ):
            FE_tempgen_localDev.main(["--subject", "sub-001"])

        subject_directories.assert_called_once_with("sub-001")
        run_subject.assert_called_once_with(selected_subject)

    def test_default_runs_all_selected_participant_directories(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            subjects = [
                Path(temporary_directory) / "sub-001",
                Path(temporary_directory) / "sub-002",
            ]
            with (
                patch.object(
                    FE_tempgen_localDev,
                    "subject_directories",
                    return_value=subjects,
                ) as subject_directories,
                patch.object(FE_tempgen_localDev, "run_subject") as run_subject,
            ):
                FE_tempgen_localDev.main([])

        subject_directories.assert_called_once_with(None)
        run_subject.assert_has_calls([unittest.mock.call(subject) for subject in subjects])
        self.assertEqual(run_subject.call_count, len(subjects))

    def test_subject_directories_use_the_given_derivatives_path(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            derivatives = Path(temporary_directory)
            (derivatives / "sub-002").mkdir()
            (derivatives / "sub-001").mkdir()
            (derivatives / "other").mkdir()

            self.assertEqual(
                [path.name for path in _subject_directories(derivatives)],
                ["sub-001", "sub-002"],
            )

    def test_hdf_aggregate_refreshes_when_subject_matrices_are_newer(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            aggregate = Path(temporary_directory) / "aggregate.h5"
            matrix = Path(temporary_directory) / "subject_matrix.npy"
            aggregate.touch()
            matrix.touch()

            os_times = (1_000_000_000, 1_000_000_000)
            os.utime(aggregate, ns=os_times)
            os.utime(matrix, ns=(2_000_000_000, 2_000_000_000))
            self.assertTrue(_aggregate_needs_refresh(aggregate, [matrix]))

            os.utime(matrix, ns=(500_000_000, 500_000_000))
            self.assertFalse(_aggregate_needs_refresh(aggregate, [matrix]))


if __name__ == "__main__":
    unittest.main()
