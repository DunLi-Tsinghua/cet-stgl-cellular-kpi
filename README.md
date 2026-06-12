# CET-STGL: Causal-Inspired Event-Token Spatio-Temporal Learning for Cellular KPI Degradation

This repository contains the reproducible code package for:

**Causal-Inspired Event-Token Spatio-Temporal Graph Learning for Alarm-Driven KPI Degradation Prediction in Cellular Networks**

The package includes the runnable sklearn implementation, feature construction code, weak-label generation code, paper-ready artifacts, and a synthetic toy dataset for interface testing.

## What Is Included

- `src/`: feature construction, weak-labeling, baselines, and ranking utilities.
- `run_smoke.py`: lightweight smoke test for forecasting and weak degradation classification.
- `run_full_experiment.py`: full sklearn experiment runner used for the reported runnable protocol.
- `scripts/`: paper artifact builders.
- `configs/default.json`: default horizons, lookback, and primary KPI settings.
- `data/synthetic_toy_510957.csv`: synthetic toy panel with the same column scale as the private panel.
- `data/schema_510957_public.json`: public schema/statistics record for the private panel.
- `results/paper_artifacts/`: figure-ready and table-ready CSV/JSON artifacts used by the manuscript.
- `paper/`: LaTeX paper source and compiled PDF.

## Data Availability

The raw `510957.csv` file is **not included** in this repository.

Reason: the original data are private cellular operational records containing business-sensitive KPI/alarm information and possible network-operation confidentiality. The reported paper metrics are bound to that private data slice or weak labels derived from it. Public release would require data-owner approval and anonymization review.

For code testing, use:

```bash
python run_smoke.py --csv data/synthetic_toy_510957.csv --horizons 1,3 --max-train-samples 2000 --max-test-samples 1000
```

To reproduce the private-data experiments after obtaining authorization, place the authorized raw file at a local path such as:

```text
data_private/510957.csv
```

Then run:

```bash
python run_smoke.py --csv data_private/510957.csv
python run_full_experiment.py --csv data_private/510957.csv --out-dir results/private_run
```

`data_private/` and raw `510957.csv` are ignored by git.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The runnable implementation is sklearn-based. Neural baselines such as LSTM, TCN, Informer, Autoformer, STGCN, and GAT are not reported because the original runtime did not provide the required PyTorch backend.

## Claim Boundaries

- This is not a telecom foundation-model repository.
- This is not a public benchmark release.
- The event graph is causal-inspired / causal discovery-based, not confirmed causality.
- Root-cause ranking is weak-label/proxy consistency only, not expert-confirmed root-cause localization.
- Reported results should not be interpreted as state-of-the-art claims.

## Citation

If you use this code package, please cite the paper draft in `paper/main.pdf`. A formal BibTeX entry can be added after publication.

