import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from analysis.common.tempgen_utils import load_tempgen_matrices_hdf


class TestTempgenCache(unittest.TestCase):
    def test_hdf_aggregate_is_loaded_from_analysis_cache(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            derivatives = root / "derivatives"
            cache = root / "cache"
            old_group_data = derivatives / "group_data_structures"
            old_group_data.mkdir(parents=True)
            cache.mkdir()

            expected = np.array([1.0])
            with h5py.File(cache / "allTempGenMats.h5", "w") as hdf:
                hdf.create_dataset("all_mats", data=expected)
            with h5py.File(old_group_data / "allTempGenMats.h5", "w") as hdf:
                hdf.create_dataset("all_mats", data=np.array([2.0]))

            with patch.dict(
                os.environ,
                {
                    "MOTEST_DERIVATIVES": str(derivatives),
                    "MOTEST_CACHE_DIR": str(cache),
                },
            ):
                actual = load_tempgen_matrices_hdf(derivatives)

            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
