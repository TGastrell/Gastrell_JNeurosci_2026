# Figure provenance

This map follows the submitted manuscript and supplement. Generated intermediates and rendered figure files are not part of the curated analysis source.

| Figure | Analysis | Analysis source |
| --- | --- | --- |
| 1 | Task and experimental design schematic | No analysis script |
| 2 | Experiment 1 behavioural results | `analysis/behaviour/behavioural_figs.py` |
| 3 | Experiment 2 behavioural results | `analysis/behaviour/behavioural_figs.py` |
| 4 | CPP waveforms and time-resolved model statistics | `analysis/erp/generate_fig04.py`; model fitting in `analysis/statistics/CPP.py` |
| 5 | EEG sensor sensitivity, decoding accuracy, and temporal generalisation | `analysis/decoding/generate_fig05.py`; sensor model in `analysis/decoding/MixedLM_sensors.py` |
| 6 | EEG decoding accuracy by stimulus and prior precision | `analysis/decoding/generate_fig06.py`; model fitting in `analysis/statistics/MixedLM_FE.py` |
| 7 | EEG decoding bias | `analysis/decoding/generate_fig07.py` |
| 8 | Cross-generalisation of stimulus-observation models to the response period | `analysis/decoding/generate_fig08.py` |
| S1 | Main-model EEG channel-activity profiles | `analysis/supplementary/plot_tuning_functions.py`, original/unbiased decoder mode |
| S2 | EEG decoding-accuracy analysis | `analysis/supplementary/generate_figS2.py` |
| S3 | EEG decoding-bias analysis | `analysis/supplementary/generate_figS3.py` |
| S4 | Prior-informed-model EEG channel-activity profiles | `analysis/supplementary/plot_tuning_functions.py`, within-prior decoder mode |
| S5 | Re-baselined/pre-stimulus EEG analyses | `analysis/supplementary/generate_figS5.py` |

Figures S1 and S4 share a plotting script and use different intermediate data/decoder modes. Select the mode with `--analysis main` for S1 or `--analysis within-prior` for S4. The former main-workspace `generate_figS1.py` produced a superseded decoding-bias figure and is intentionally excluded.

The forward-encoding source and shared analysis utilities are in `analysis/decoding/forward_encoding/` and `analysis/common/`. Before fitting the decoding statistics models or generating figures that use decoder metrics, convert the corresponding prediction epochs to per-participant decoding-metric Parquet files.

The forward-encoding module can run either the main decoder (`--analysis main`) or within-prior decoder (`--analysis within-prior`). Run `python -m analysis.decoding.forward_encoding.FE_chans2metrics --analysis main` for main predictions in `train_sk2_unif`, or use `--analysis within-prior` for predictions in `train_within_prior_resamp_binned`. Main analyses and S1 use cropped prediction/metric products. S2–S4 use full-window within-prior products; their group summaries, models, and plots crop metrics to `-0.25..2 s` as appropriate. Group caches and fitted models live in `outputs/cache/within_prior` (or `$MOTEST_CACHE_DIR/within_prior`), with model-input cache filenames marked `cropped_from_full_window`. The within-prior figures use the decoder's `binLabel` as their stimulus direction. The abandoned high-coherence test decoder is excluded.

Run these commands from the project root to create model files consumed by the figures:

| Model output | Command | Used by |
| --- | --- | --- |
| `cpp_model_fit.pkl` | `python -m analysis.statistics.CPP` | Figure 4 |
| `fe_acc_model_fit.pkl` | `python -m analysis.statistics.MixedLM_FE --dv accuracy` | Figure 6 |
| `original_bias_model_fit.pkl` | `python -m analysis.statistics.MixedLM_FE --dv bias` | Figure 7 |
| `acc_resamp_binned_model_fit_cropped_from_full_window.pkl` in `within_prior/` | `python -m analysis.statistics.MixedLM_FE_within_prior --dv accuracy` | Figure S2 |
| `bias_resamp_binned_model_fit_cropped_from_full_window.pkl` in `within_prior/` | `python -m analysis.statistics.MixedLM_FE_within_prior --dv bias` | Figure S3 |

The CPP model reads EEG derivatives directly. The decoding model scripts use per-subject decoding metrics under the derivatives tree and aggregate them into the configured cache when needed. Figure 6/7 model times are stored in seconds; S2/S3 model times remain sample indices because their plotting scripts convert them to seconds. The analysis and figure outputs, including supplementary figures S1–S5, have been verified. Only raw study data are available for download; users must generate the EEG derivatives and decoding products by following the [reproduction guide](reproduction-guide.md).
