from analysis.common import utils
from analysis.common.eeg_utils import load_all_epochs_with_metadata
from analysis.common.paths import analysis_cache_dir, derivatives_root
from pathlib import Path
from analysis.common.mixedLM_2way import run_parallel_mixedlm
import pandas as pd
import pickle


def main():
    derivatives_pth = derivatives_root()
    file_label = "t--0.95-2_bl--0.35--0.25-0.2-40bpf-clean"
    filename = analysis_cache_dir() / f"cpp_long_{file_label}_for_mixedlm.parquet"

    if not filename.exists():
        t_min, t_max = -0.95, 2
        bl_min, bl_max = -0.35, -0.25
        bpf_low, bpf_high = 0.2, 40
        sorted_pts = sorted(derivatives_pth.iterdir(), key=lambda path: path.name.lower())
        df, electrodes = load_all_epochs_with_metadata(
            sorted_pts,
            t_min,
            t_max,
            bl_min,
            bl_max,
            bpf_low,
            bpf_high,
            eogs=True,
            electrode_subset=["CP1", "CPz", "CP2", "Cz", "Pz"],
            crop=(0, 0.75),
        )

        float64_cols = df.select_dtypes(include=["float64"]).columns
        df[float64_cols] = df[float64_cols].astype("float32")
        df["responseError"] = utils.circular_difference(df["responseAngle"], df["stimDir"])
        df["prior_stim_distance"] = utils.circular_difference(df["priorDir"], df["stimDir"])
        df["signed_error"] = utils.get_signed_error(df["responseError"], df["prior_stim_distance"])
        df["cpp"] = df[electrodes].mean(axis=1)
        df = df.rename(
            columns={
                "stimKappa": "Coherence",
                "priorKappa": "Prior_Precision",
                "signed_error": "Behavioural_Bias",
            }
        )
        df.to_parquet(filename)
    else:
        df = pd.read_parquet(filename).rename(columns={"time_secs": "time"})

    print(
        "Fitting mixed effects model... This may take a while depending on "
        "your system and the number of parallel jobs."
    )
    params, pvals, resid = run_parallel_mixedlm(
        df, vars=["cpp", "Coherence", "Prior_Precision", "subjectNumber"], n_jobs=10
    )
    out = {"params": params, "pvals": pvals, "resid": resid}
    output_path = analysis_cache_dir() / "cpp_model_fit.pkl"
    with output_path.open("wb") as file:
        pickle.dump(out, file)
    print(f"Saved model output to {output_path}")


if __name__ == "__main__":
    main()