# CODE_PACKAGE_REPORT

## Package Scope

This repository is a GitHub-safe release of the CET-STGL code and paper artifacts. It intentionally excludes the raw private cellular operational dataset.

## Included

- Runnable code for data loading, feature construction, classical baselines, lightweight sklearn-based CET-STGL instantiation, fixed LSTM/TCN sequence baselines, weak degradation labels, and weak alarm-candidate prioritization.
- Synthetic toy data with 187 columns for smoke testing.
- Public schema/statistics metadata for the private `510957.csv` panel.
- Current RESS-style paper LaTeX source, compiled PDF, figures, tables, and bibliography.
- Paper-ready CSV/JSON artifacts for tables and figures.
- Neural sequence-baseline summary metrics and metadata.

## Excluded

- Raw private `510957.csv`.
- Per-row private prediction/ground-truth outputs from private-data experiments.
- Any claim that the synthetic toy data reproduce the paper metrics.

## Reproducibility Boundary

The public package supports code inspection, smoke tests, table/figure artifact inspection, and protocol reproduction. Exact metric reproduction requires authorized access to the private `510957.csv` data.

## Current Boundary

The repository does not claim a public benchmark, telecom foundation model, expert-confirmed fault localization, or causal-effect identification. LSTM and TCN are fixed sequence-model comparators; they are not full neural spatio-temporal graph versions of CET-STGL.

