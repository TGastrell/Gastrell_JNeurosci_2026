"""Fit the main time-resolved mixed model used by Figures 6 and 7."""
import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.common import mixedLM_2way, utils
from analysis.common.paths import analysis_cache_dir, derivatives_root
from analysis.common.util_dataloader import load_decoding_data


FILE_LABEL = "t--0.25-2_bl--0.35--0.25-0.2-40bpf-clean"
OUTPUT_FILES = {
    "accuracy": "fe_acc_model_fit.pkl",
    "bias": "original_bias_model_fit.pkl",
}


def fit_model(dv):
    derivatives = derivatives_root()
    cache = analysis_cache_dir()
    data_path = cache / f"decoding_metrics_long_{FILE_LABEL}_for_mixedlm.parquet"

    if data_path.exists():
        print("Loading aggregated decoding metrics...")
        df = pd.read_parquet(data_path)
    else:
        df = load_decoding_data(
            derivatives,
            FILE_LABEL,
            [
                "subjectNumber", "trialNr", "session", "blockNr", "time",
                "stimKappa", "priorKappa", "stimDir", "priorDir",
                "responseAngle", "vector_length", "accuracy", "bias",
            ],
        )
        df["uniqueTrialID"] = df.groupby("subjectNumber").cumcount()
        df["trialInBlock"] = df.groupby(
            ["subjectNumber", "session", "blockNr"]
        ).cumcount() + 1
        df["time_secs"] = (df["time"] / 256) - 0.25
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
        df = df.query("time_secs >= -0.25")
        df.to_parquet(data_path)

    df = df.query("time_secs >= -0.25").copy()
    df = df.rename(
        columns={"stimKappa": "Coherence", "priorKappa": "Prior_Precision"}
    )
    # Figure 6/7 plot model effects in seconds, unlike S2/S3 which use samples.
    df["time"] = df["time_secs"]
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
        help="Dependent variable/model output to prepare for Figure 6 or 7.",
    )
    args = parser.parse_args()
    fit_model(args.dv)


if __name__ == "__main__":
    main()
