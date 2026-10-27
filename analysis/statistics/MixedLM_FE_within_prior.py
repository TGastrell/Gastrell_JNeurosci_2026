"""Fit within-prior models used by Supplementary Figures 2/3."""
import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.common import mixedLM_2way, utils
from analysis.common.paths import (
    WITHIN_PRIOR_FE_DIRECTORY,
    derivatives_root,
    within_prior_cache_dir,
)
from analysis.common.util_dataloader import (
    crop_full_window_metrics,
    load_decoding_data,
)


FILE_LABEL = "t--0.95-2_bl--0.35--0.25-0.2-40bpf-clean"
CROPPED_CACHE_LABEL = (
    "t--0.25-2_bl--0.35--0.25-0.2-40bpf-clean_cropped_from_full_window"
)
OUTPUT_FILES = {
    "accuracy": "acc_resamp_binned_model_fit_cropped_from_full_window.pkl",
    "bias": "bias_resamp_binned_model_fit_cropped_from_full_window.pkl",
}


def fit_model(dv):
    derivatives = derivatives_root()
    cache = within_prior_cache_dir()
    data_path = (
        cache
        / f"resamp_binned_decoding_metrics_long_{CROPPED_CACHE_LABEL}_for_mixedlm.parquet"
    )

    if data_path.exists():
        print("Loading aggregated decoding metrics...")
        df = pd.read_parquet(data_path)
    else:
        print("Aggregated file not found; loading individual subject data...")
        df = load_decoding_data(
            derivatives,
            FILE_LABEL,
            [
                "subjectNumber", "trialNr", "session", "blockNr", "time",
                "stimKappa", "priorKappa", "binLabel", "priorDir",
                "responseAngle", "vector_length", "accuracy", "bias",
            ],
            subdirectory=WITHIN_PRIOR_FE_DIRECTORY,
        )
        df = crop_full_window_metrics(df)
        df["stimDir"] = df["binLabel"]
        df["uniqueTrialID"] = df.groupby("subjectNumber").cumcount()
        df["trialInBlock"] = df.groupby(
            ["subjectNumber", "session", "blockNr"]
        ).cumcount() + 1
        df["responseError"] = utils.circular_difference(
            df["responseAngle"], df["stimDir"]
        )
        df["abs_responseError"] = np.abs(df["responseError"])
        df["prior_stim_distance"] = utils.circular_difference(
            df["priorDir"], df["stimDir"]
        )
        df["signed_error"] = utils.get_signed_error(
            df["responseError"], df["prior_stim_distance"]
        )
        df.to_parquet(data_path)

    df["time_secs"] = (df["time"] / 256) - 0.25
    df = df.rename(
        columns={"stimKappa": "Coherence", "priorKappa": "Prior_Precision"}
    )
    float64_cols = df.select_dtypes(include="float64").columns
    df[float64_cols] = df[float64_cols].astype("float32")

    variables = [dv, "Coherence", "Prior_Precision", "subjectNumber"]
    params, pvals, resid = mixedLM_2way.run_parallel_mixedlm(
        df, variables, n_jobs=10
    )
    out = {"params": params, "pvals": pvals, "resid": resid}

    output_path = cache / OUTPUT_FILES[dv]
    print(f"Saving model output to {output_path}...")
    with output_path.open("wb") as file:
        pickle.dump(out, file)
    mixedLM_2way.plot_results_heatmap(out["params"], dv=dv, pvals=out["pvals"])
    mixedLM_2way.plot_fdr_results(out["pvals"], dv=dv, legend=True)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dv",
        required=True,
        choices=OUTPUT_FILES,
        help="Dependent variable/model output to prepare for Figure S2 or S3.",
    )
    args = parser.parse_args()
    fit_model(args.dv)


if __name__ == "__main__":
    main()
