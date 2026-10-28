# Reproducing analyses and figures

This guide takes you from environment setup to the analysis outputs. Run commands from the repository root. Figure 1 is a task-design schematic and has no analysis script.

Only the de-identified raw study data are available for download from the separate data repository; preprocessed EEG derivatives are not provided. Users must run the preprocessing steps in Section 4 to create derivatives before running EEG analyses. Behavioural Figures 2 and 3 use the raw behavioural files and do not require EEG preprocessing. Do not add raw data, derivatives, caches, or generated figures to this code repository.

> **Reproduction status:** The analysis and figure outputs have been verified, including the supplementary figures (S1–S5). Figure 1 is a task-design schematic and has no analysis script.

## 1. Set up the environment

The analyses were run with Python 3.12.7 and MNE 1.8.0. [`environment.yml`](../environment.yml) records the analysis dependencies and tested versions. It is a portable Conda specification, not a platform-specific lock file. `mne-faster` is installed through pip for EEG preprocessing; `pyarrow` provides Parquet support.

From a Bash terminal, create and activate the environment:

```bash
project="$HOME/path/to/TG-MotEst-02-reproducible-pipeline"
cd "$project"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda env create -f environment.yml
conda activate motest-analysis

python --version
python -c 'import mne; print("MNE", mne.__version__)'
```

If the environment already exists, skip `conda env create`. On HPC systems, initialize or load Conda according to the cluster's instructions before activation.

## 2. Configure data and output paths

Set each raw-data variable to the root containing that experiment's participant folders. If both experiments are stored under one root, the variables can point to the same directory.

```bash
export MOTEST_EXP1_RAWDATA="/path/to/experiment-1/rawdata"
export MOTEST_EXP2_RAWDATA="/path/to/experiment-2/rawdata"
export MOTEST_DERIVATIVES="/path/to/derivatives"

export MOTEST_CACHE_DIR="$project/outputs/cache"
export MOTEST_FIGURE_DIR="$project/figures/generated"
```

The behavioural scripts expect participant folders such as `sub-001/beh/` under the corresponding raw-data root, with tab-delimited behavioural files there. EEG preprocessing expects the participant's `eeg/` and `beh/` folders under the input participant folder.

`MOTEST_DERIVATIVES` must contain participant folders such as `sub-001`. EEG analyses read each participant's `eeg/` and `eeg/FE/` derivatives. If `MOTEST_CACHE_DIR` or `MOTEST_FIGURE_DIR` is unset, outputs default to `outputs/cache/` and `figures/generated/` in the project. The `.env.example` file shows the variable names; it is not loaded automatically.

## 3. Understand the required EEG derivatives

EEG analyses require participant-level derivatives created from the downloaded raw data. These derivatives are not available for download, so run the preprocessing workflow in [Section 4](#4-preprocess-the-eeg-data) for every participant before continuing with the CPP, decoding, or supplementary EEG analyses.

Standard clean epochs span `-0.95..2 s` and use the `-0.35..-0.25 s` baseline. Their label is:

```text
t--0.95-2_bl--0.35--0.25-0.2-40bpf-clean
```

Main-decoder predictions and metrics default to the cropped `-0.25..2 s` window. Within-prior products for S2–S4 must instead be generated over the full `-0.95..2 s` window; their group summaries, models, and plots crop to `-0.25..2 s`. S5 uses full-window products from **both** decoder modes with the alternate `-0.95..-0.85 s` baseline. Keep these product sets distinct.

The standard-baseline product labels are:

```text
Main decoder, cropped:       t--0.25-2_bl--0.35--0.25-0.2-40bpf-clean
Within-prior, full-window:   t--0.95-2_bl--0.35--0.25-0.2-40bpf-clean
```

Participant-level decoder metrics are Parquet files in the decoder's `decoding_metrics/` folder. Prediction epochs are separate files. The decoder folders are:

- Main: `eeg/FE/train_sk2_unif/`
- Within-prior: `eeg/FE/train_within_prior_resamp_binned/`

## 4. Preprocess the EEG data

Preprocessing is required for EEG analyses and runs once per participant. The example uses `MOTEST_EXP2_RAWDATA`; pass the participant folder containing `eeg/` and `beh/`, followed by the derivatives root:

```bash
python -m analysis.eeg_processing.processEEG \
  "$MOTEST_EXP2_RAWDATA/sub-001" \
  "$MOTEST_DERIVATIVES"
```

Repeat for each participant. The script reads BDF and behavioural files, filters and epochs the EEG, applies MNE-Faster/ICA cleaning, and writes derivatives under `$MOTEST_DERIVATIVES/sub-001/eeg/`. It uses the standard baseline by default. Confirm that clean epoch files exist for both sessions and every participant before running the CPP analysis or standard-baseline decoding. For example:

```text
sub-001_ses-MAIN-01_task-TG-MotEst-2.0_t--0.95-2_bl--0.35--0.25-0.2-40bpf-clean_epo.fif.gz
sub-001_ses-MAIN-02_task-TG-MotEst-2.0_t--0.95-2_bl--0.35--0.25-0.2-40bpf-clean_epo.fif.gz
```

For S5, create a separate set of epochs with `--baseline s5`; this does not replace the standard-baseline epochs. The S5-specific preprocessing and decoding commands are in [Section 8.3](#83-generate-supplementary-figure-s5).

## 5. Generate Figures 2 and 3: behaviour

These analyses do not depend on EEG preprocessing. With both experiment data roots configured, run:

```bash
python -m analysis.behaviour.behavioural_figs
```

The script filters aborted and very fast trials, fits the behavioural models, and saves `E1_bhv_results.svg` and `E2_bhv_results.svg` under `$MOTEST_FIGURE_DIR`.

## 6. Generate Figure 4: CPP

This analysis requires standard-baseline clean epochs for both sessions and all participants. First fit the time-resolved model:

```bash
python -m analysis.statistics.CPP
```

This saves `cpp_model_fit.pkl` and a model-input Parquet file in `$MOTEST_CACHE_DIR`. Then generate the figure:

```bash
python -m analysis.erp.generate_fig04
```

The figure script prepares or loads its plotting-data cache and saves `fig04.svg` under `$MOTEST_FIGURE_DIR`.

## 7. Generate main EEG decoding figures

### 7.1 Generate main-decoder predictions and metrics

The default `--time-window` is `cropped`: the decoder loads the standard `-0.95..2 s` clean epochs, crops them to `-0.25..2 s` for training and prediction, and saves predictions and metrics for that cropped window.

```bash
python -m analysis.decoding.forward_encoding.FE_full_pipeline --analysis main
python -m analysis.decoding.forward_encoding.FE_chans2metrics --analysis main
```

The outputs are stored under each participant's `eeg/FE/train_sk2_unif/` directory, with metrics under its `decoding_metrics/` subdirectory.

### 7.2 Generate Figure 5

Figure 5 uses the cropped main-decoder metrics, sensor-regression results, and temporal-generalisation matrices. Run sensor regression and generate temporal-generalisation matrices before plotting:

```bash
python -m analysis.decoding.MixedLM_sensors
python -m analysis.decoding.forward_encoding.FE_tempgen_localDev
```

Temporal-generalisation processing can be run for all participants in one invocation or participant-by-participant with `--subject sub-001`. Wait for all participant jobs to finish before generating the figure. Then run:

```bash
python -m analysis.decoding.generate_fig05
```

The sensor-regression cache is `$MOTEST_CACHE_DIR/sensor_reg_stats_all.pkl`. Subject temporal-generalisation matrices are saved under each participant's `eeg/FE/train_sk2_unif/tempGenMats/`; the group aggregate is cached at `$MOTEST_CACHE_DIR/allTempGenMats.h5`. Figure 5 also loads and crops the standard clean epochs for its sensor topomaps. It saves `fig05a.svg` and `fig05bc.svg`.

Sensor regression is especially compute-intensive, though it is possible this code could be optimized substantially. Based on the project run, budget approximately 10 hours distributed across ~16 CPU cores; actual runtime depends on the system and data. Request adequate wall time, CPU, and memory on the HPC scheduler.

### 7.3 Generate Figures 6 and 7

Fit the relevant model before generating each figure:

```bash
python -m analysis.statistics.MixedLM_FE --dv accuracy
python -m analysis.decoding.generate_fig06

python -m analysis.statistics.MixedLM_FE --dv bias
python -m analysis.decoding.generate_fig07
```

The model files are `$MOTEST_CACHE_DIR/fe_acc_model_fit.pkl` and `$MOTEST_CACHE_DIR/original_bias_model_fit.pkl`. The figures are `fig06.svg` and `fig07.svg`.

### 7.4 Generate Figure 8

Figure 8 uses the same subject temporal-generalisation matrices as Figure 5. If they have not already been generated, run `FE_tempgen_localDev` as described in Section 7.2 and wait for all participant jobs to finish. Then run:

```bash
python -m analysis.decoding.generate_fig08
```

The script builds or updates the group aggregate in `$MOTEST_CACHE_DIR/allTempGenMats.h5` as needed, then saves `fig08.svg`.

## 8. Generate supplementary figures

Follow the input requirements carefully and do not substitute the main-decoder products for within-prior products. Raw-data downloads do not include the derivatives or decoder outputs listed below; generate them by running the preceding preprocessing and analysis steps.

### 8.1 Generate within-prior predictions and metrics for S2–S4

For S2, S3, and S4, both commands must use `--time-window full` so the participant products span `-0.95..2 s`:

```bash
python -m analysis.decoding.forward_encoding.FE_full_pipeline \
  --analysis within-prior --time-window full
python -m analysis.decoding.forward_encoding.FE_chans2metrics \
  --analysis within-prior --time-window full
```

These products are stored separately under `eeg/FE/train_within_prior_resamp_binned/`. The S2/S3 models and S4 group summaries crop full-window metrics to `-0.25..2 s`; S4 also crops the prediction epochs before plotting.

### 8.2 Generate S1–S4

S1 uses the cropped **main** predictions and metrics from Section 7.1:

```bash
python -m analysis.supplementary.plot_tuning_functions --analysis main
```

S4 uses the full-window **within-prior** predictions and metrics from Section 8.1:

```bash
python -m analysis.supplementary.plot_tuning_functions --analysis within-prior
```

The script saves `figS1.svg` or `figS4.svg`. Its default mode is `within-prior`; specify the mode to avoid generating the wrong supplementary figure.

S2 and S3 each require a model fit followed by its figure script:

```bash
python -m analysis.statistics.MixedLM_FE_within_prior --dv accuracy
python -m analysis.supplementary.generate_figS2

python -m analysis.statistics.MixedLM_FE_within_prior --dv bias
python -m analysis.supplementary.generate_figS3
```

The model inputs and fitted models are stored under `$MOTEST_CACHE_DIR/within_prior/`. The model files are `acc_resamp_binned_model_fit_cropped_from_full_window.pkl` and `bias_resamp_binned_model_fit_cropped_from_full_window.pkl`. The figures are `figS2.svg` and `figS3.svg`.

### 8.3 Generate Supplementary Figure S5

S5 requires full-window (`-0.95..2 s`) prediction epochs and metrics from **both** decoders, using the alternate `-0.95..-0.85 s` baseline. First generate alternate-baseline clean epochs for every participant:

```bash
python -m analysis.eeg_processing.processEEG \
  "$MOTEST_EXP2_RAWDATA/sub-001" \
  "$MOTEST_DERIVATIVES" --baseline s5
```

Repeat for all participants. Then run all four decoder/metric steps:

```bash
python -m analysis.decoding.forward_encoding.FE_full_pipeline \
  --analysis main --baseline s5 --time-window full
python -m analysis.decoding.forward_encoding.FE_chans2metrics \
  --analysis main --baseline s5 --time-window full

python -m analysis.decoding.forward_encoding.FE_full_pipeline \
  --analysis within-prior --baseline s5 --time-window full
python -m analysis.decoding.forward_encoding.FE_chans2metrics \
  --analysis within-prior --baseline s5 --time-window full
```

Each participant must have both prediction-epoch files and their matching metric Parquet files under the corresponding decoder folders. The expected label is `t--0.95-2_bl--0.95--0.85-0.2-40bpf-clean`. Do not use standard-baseline or cropped-window products for S5.

Generate the figure after all four products exist for every participant:

```bash
python -m analysis.supplementary.generate_figS5
```

It writes metric summaries to the main and within-prior caches, applies the script's time alignment, and saves `figS5.svg`.

## 9. Running participant-level jobs on HPC

We recommend running preprocessing and decoding as two sequential HPC stages. First, submit preprocessing as one job per participant and wait until preprocessing has completed for all participants. Then submit the decoding analyses as one job per participant. This ensures all clean-epoch inputs are ready before decoding begins and avoids tying decoding job timing to individual preprocessing jobs.

```bash
python -m analysis.eeg_processing.processEEG \
  "$MOTEST_EXP2_RAWDATA/sub-001" "$MOTEST_DERIVATIVES"
python -m analysis.decoding.forward_encoding.FE_full_pipeline \
  --analysis main --subject sub-001
python -m analysis.decoding.forward_encoding.FE_chans2metrics \
  --analysis main --subject sub-001
python -m analysis.decoding.forward_encoding.FE_tempgen_localDev \
  --subject sub-001
```

The commands above show the per-participant sequence; run the preprocessing command for each participant and complete that stage for all participants before submitting the forward-encoding and temporal-generalisation jobs. Use the corresponding `--analysis`, `--time-window`, and `--baseline` options for within-prior or S5 jobs. Keep the project code and environment available on compute nodes and provide the same data-path variables to every job. Forward encoding, temporal generalisation, and sensor regression use parallel workers; forward encoding and temporal generalisation use joblib with `n_jobs=-1`, so each process uses the CPUs visible to it. Match CPUs per job and the number of simultaneous jobs to the resources allocated by the scheduler to avoid oversubscription. Wait for all temporal-generalisation jobs to finish before generating Figures 5 or 8.

## 10. Outputs and tests

Generated figures are written under `$MOTEST_FIGURE_DIR`; intermediate aggregates and fitted models are stored under `$MOTEST_CACHE_DIR`. To list them from Bash:

```bash
find "$MOTEST_FIGURE_DIR" -maxdepth 1 -type f -print
find "$MOTEST_CACHE_DIR" -type f -print
```

Run the lightweight project tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

## Figure input reference

Participant files are expected below `$MOTEST_DERIVATIVES/sub-*/`. Decoder metrics are in the corresponding decoder directory's `decoding_metrics/` subdirectory. This table summarizes the inputs and generated intermediates required by each figure.

| Figure | Participant-level inputs | Generated group/model intermediates |
| --- | --- | --- |
| 1 | None; design schematic. | None. |
| 2 | Experiment 1 behavioural TSVs under `$MOTEST_EXP1_RAWDATA/sub-*/beh/`. | The script fits the behavioural models. |
| 3 | Experiment 2 behavioural TSVs under `$MOTEST_EXP2_RAWDATA/sub-*/beh/`. | The script fits the behavioural models. |
| 4 | Standard-baseline clean epochs for both sessions, for all participants. | `$MOTEST_CACHE_DIR/cpp_model_fit.pkl`; CPP model-input and Figure 4 plotting-data Parquet caches. |
| 5 | Cropped main-decoder metrics; standard-baseline clean epochs; subject matrices under `eeg/FE/train_sk2_unif/tempGenMats/`. | `$MOTEST_CACHE_DIR/sensor_reg_stats_all.pkl` and `$MOTEST_CACHE_DIR/allTempGenMats.h5`. |
| 6 | Cropped main-decoder metrics. | `$MOTEST_CACHE_DIR/fe_acc_model_fit.pkl`. |
| 7 | Cropped main-decoder metrics. | `$MOTEST_CACHE_DIR/original_bias_model_fit.pkl`. |
| 8 | Subject temporal-generalisation matrices under `eeg/FE/train_sk2_unif/tempGenMats/`. | `$MOTEST_CACHE_DIR/allTempGenMats.h5`. |
| S1 | Cropped main-decoder prediction epochs and metrics (`-0.25..2 s`). | Main-cache channel-activity and metric summaries. |
| S2 | Full-window within-prior metrics (`-0.95..2 s`), cropped to `-0.25..2 s` by the model workflow. | `$MOTEST_CACHE_DIR/within_prior/acc_resamp_binned_model_fit_cropped_from_full_window.pkl`. |
| S3 | Full-window within-prior metrics (`-0.95..2 s`), cropped to `-0.25..2 s` by the model workflow. | `$MOTEST_CACHE_DIR/within_prior/bias_resamp_binned_model_fit_cropped_from_full_window.pkl`. |
| S4 | Full-window within-prior prediction epochs and metrics (`-0.95..2 s`), cropped for group summaries and plotting. | Cropped channel-activity and metric summaries under `$MOTEST_CACHE_DIR/within_prior/`. |
| S5 | Full-window main and within-prior prediction epochs and metrics using the alternate baseline, for all participants. | Metric summaries under `$MOTEST_CACHE_DIR/` and `$MOTEST_CACHE_DIR/within_prior/`; no separate model-fit file. |

For the exact filename labels, see Sections 3 and 8.3.
