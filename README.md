# Gastrell et al. (2026) reproducible analysis

Analysis code for the manuscript **Late Integration of Prior Expectations During Precision-Weighted Perceptual Decisions** Gastrell, Rangelov & Mattingley (2026) 10.1523/JNEUROSCI.0612-26.2026.

Only the de-identified raw study data are available for download from the separate data repository (https://doi.org/10.48610/777ae2d). Users must run the preprocessing steps in the reproduction guide to create the EEG derivatives required for EEG analyses; preprocessed derivatives are not provided.

## Getting started

Create the analysis environment from [`environment.yml`](environment.yml), then see the [reproduction guide](docs/reproduction-guide.md) for the data directory layout, preprocessing instructions, order of operations, and commands for each figure. The [figure provenance map](docs/figure-provenance.md) links the figures to their analysis scripts.

Set `MOTEST_EXP1_RAWDATA`, `MOTEST_EXP2_RAWDATA`, and `MOTEST_DERIVATIVES` to the local data directories before running analyses. Optional variables `MOTEST_CACHE_DIR` and `MOTEST_FIGURE_DIR` control generated caches and figures; defaults are `outputs/cache` and `figures/generated`. `.env.example` illustrates the variables but is not loaded automatically.

Run modules from the project root so package imports resolve consistently.

## Development and AI assistance

This reproducible pipeline and guide was adapted from the original hand-coded data-analysis environment with assistance from AI using GitHub Copilot in VS Code, through the Copilot SDK harness. The analysis decisions and validation of this pipeline remain the responsibility of the authors.

## Validation status

The analysis and figure outputs have been verified, including the supplementary figures. Figure 1 is a design schematic and has no analysis script. The reproduction guide documents the execution order and prerequisites.

Run the lightweight project tests from the project root using the Python interpreter in the MNE environment:

```bash
python -m unittest discover -s tests -v
```
