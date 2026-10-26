# Gastrell et al. (2026) reproducible analysis

Analysis code for the manuscript **Late Integration of Prior Expectations During Precision-Weighted Perceptual Decisions** Gastrell, Rangelov & Mattingley (2026) 10.1523/JNEUROSCI.0612-26.2026.

De-identified study data are hosted separately in a University of Queensland repository [LINK].

## Getting started

Create the analysis environment from [`environment.yml`](environment.yml), then see the [reproduction guide](docs/reproduction-guide.md) for the data directory layout, order of operations, commands for each figure, and known prerequisite gaps. The [figure provenance map](docs/figure-provenance.md) links the figures to their analysis scripts.

Set `MOTEST_EXP1_RAWDATA`, `MOTEST_EXP2_RAWDATA`, and `MOTEST_DERIVATIVES` to the local data directories before running analyses. Optional variables `MOTEST_CACHE_DIR` and `MOTEST_FIGURE_DIR` control generated caches and figures; defaults are `outputs/cache` and `figures/generated`. `.env.example` illustrates the variables but is not loaded automatically.

Run modules from the project root so package imports resolve consistently.

## Development and AI assistance

This reproducible pipeline and guide was adapted from the original hand-coded data-analysis environment with assistance from AI using GitHub Copilot in VS Code, through the Copilot SDK harness. The analysis decisions and validation of this pipeline remain the responsibility of the authors.

## Validation status

All main analyses and main-text figures (Figures 2–8; Figure 1 is a design schematic) have been reproduced successfully. Supplementary analyses and figures (S1–S5) have not yet been tested. The reproduction guide documents the execution order and prerequisites.

Run the lightweight project tests from the project root using the Python interpreter in the MNE environment:

```bash
python -m unittest discover -s tests -v
```
