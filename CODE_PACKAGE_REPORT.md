# CODE_PACKAGE_REPORT

## Package Scope

This repository is a GitHub-safe release of the CET-STGL code and paper artifacts. It intentionally excludes the raw private cellular operational dataset.

## Included

- Runnable sklearn code for data loading, feature construction, baselines, weak degradation labels, and weak root-cause ranking.
- Synthetic toy data with 187 columns for smoke testing.
- Public schema/statistics metadata for the private `510957.csv` panel.
- Paper LaTeX source, compiled PDF, figures, tables, and 24-entry bibliography.
- Paper-ready CSV/JSON artifacts for tables and figures.

## Excluded

- Raw private `510957.csv`.
- Per-row private prediction/ground-truth outputs from private-data experiments.
- Any claim that the synthetic toy data reproduce the paper metrics.

## Reproducibility Boundary

The public package supports code inspection, smoke tests, table/figure artifact inspection, and protocol reproduction. Exact metric reproduction requires authorized access to the private `510957.csv` data.

