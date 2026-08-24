# Neural Baseline Run Report

## Scope

This version adds two runnable neural sequence baselines, LSTM and TCN, to the CET-STGL paper package. The neural baselines are used as additional sequence-model comparators only. They are not full neural spatio-temporal graph implementations and are not reported as CET-STGL neural backends.

## Protocol

- Data source: anonymized Huawei MBB operations data used by the existing CET-STGL experiments.
- Split: same chronological 70/15/15 train/validation/test protocol.
- Horizons: 1, 3, 6 and 12 hours.
- Lookback: 24 hours.
- Inputs: all 48 KPI/business counters, 11 active alarm/fault columns, and cell/time context features.
- Outputs: all 48 KPI/business counters for forecasting; weak degradation label for classification.
- Scaling: train-only input and target standardization.
- Hyperparameters: fixed seed 7, hidden dimension 32, batch size 4096, maximum 5 epochs, patience 2, AdamW learning rate 1e-3, weight decay 1e-4.
- Tuning: no hyperparameter search.
- Ranking: not evaluated for LSTM/TCN because the ranking protocol is association-rule based rather than a neural ranking head.

## Results Summary

Primary-KPI forecasting:

| Model | H | MAE | RMSE | MAPE |
| --- | ---: | ---: | ---: | ---: |
| LSTM | 1 | 3.774 | 6.606 | 68.954 |
| TCN | 1 | 13.360 | 51.673 | 131.255 |
| LSTM | 3 | 3.822 | 6.907 | 72.139 |
| TCN | 3 | 6.580 | 16.235 | 116.662 |
| LSTM | 6 | 3.991 | 7.053 | 73.741 |
| TCN | 6 | 7.660 | 19.911 | 121.740 |
| LSTM | 12 | 3.892 | 7.066 | 73.416 |
| TCN | 12 | 11.522 | 41.975 | 124.861 |

Weak degradation classification:

| Model | H | AUPRC | F1 | Precision | Recall | AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LSTM | 1 | 0.448 | 0.404 | 0.277 | 0.748 | 0.755 |
| TCN | 1 | 0.270 | 0.389 | 0.277 | 0.652 | 0.685 |
| LSTM | 3 | 0.370 | 0.390 | 0.268 | 0.717 | 0.721 |
| TCN | 3 | 0.220 | 0.334 | 0.226 | 0.640 | 0.619 |
| LSTM | 6 | 0.350 | 0.368 | 0.246 | 0.727 | 0.702 |
| TCN | 6 | 0.199 | 0.313 | 0.211 | 0.605 | 0.583 |
| LSTM | 12 | 0.380 | 0.376 | 0.262 | 0.666 | 0.710 |
| TCN | 12 | 0.190 | 0.289 | 0.196 | 0.548 | 0.569 |

## Interpretation Boundary

The added neural baselines do not change the main empirical reading. KPI-history and feature-engineered baselines remain strong; CET-STGL-sklearn remains competitive on several primary-KPI forecasting metrics; LSTM provides an additional runnable sequence comparator but does not dominate the classical or CET-STGL variants; fixed TCN is weaker in the current leakage-controlled protocol. These results do not support a claim that a neural CET-STGL backend has been validated.

## Files

- `run_neural_baselines.py`
- `src/neural_sequence.py`
- `tests/test_neural_sequence.py`
- `results/neural_baselines/neural_metrics_all.csv`
- `results/neural_baselines/neural_forecasting_metrics.csv`
- `results/neural_baselines/neural_classification_metrics.csv`
- `results/neural_baselines/neural_run_metadata.json`
