# EXPERIMENT_PLAN

Title: **Causal Event-Token Spatio-Temporal Graph Learning for Alarm-Driven KPI Degradation Prediction in Cellular Networks**

Scope: technical paper on alarm-driven KPI degradation prediction using the current `510957.csv` data. This is not a telecom foundation-model paper, not a benchmark paper, and not a survey extension.

## Research Question

Can sparse network alarm events improve multi-step KPI forecasting, KPI degradation classification, and weak root-cause ranking in a cellular cell-time panel when represented as causal-inspired event tokens over a dynamic event influence graph?

The paper should use the phrase **causal-inspired** or **causal discovery-based event influence graph**. It should not claim strict causal identification because the data are observational and no intervention/randomization is available.

## Data Unit

Basic sample unit:

- Cell `c`
- Timestamp `t`
- Lookback window `L`
- Forecast horizon `h in {1, 3, 6, 12}` hours

Input:

- KPI history: `X_kpi[c, t-L+1:t]`
- Alarm history: `X_alarm[c, t-L+1:t]`
- Optional cross-cell context from inferred neighboring cells
- Time encodings and cell encodings

Output:

- Future KPI vector at `t+h`
- Future degradation label at `t+h`
- Weak root-cause ranking over alarm types observed before `t+h`

Windows crossing the 21-day time gap must be excluded.

## Task 1: Multi-Step KPI Forecasting

Horizons:

- 1 hour
- 3 hours
- 6 hours
- 12 hours

Primary target KPIs:

- `Call Drop Rate(%)`
- `RRC Setup Success Rate(%)`
- `ERAB Setup Success Rate(%)`
- `LTE_User DL Average Throughput(Mbps)`
- `LTE_User UL Average Throughput(Mbps)`
- `LTE_DL PRB Utilizing Rate(%)`
- `LTE_UL PRB Utilizing Rate(%)`
- `CQI AVG`

Secondary target KPIs:

- `LTE_DL Traffic Volume(GB)`
- `LTE_UL Traffic Volume(GB)`
- `Call Drop Times`
- `L.UL.Interference.Avg(dBm)`

Metrics:

- MAE and RMSE per KPI and horizon
- normalized RMSE or MASE for cross-KPI comparability
- event-conditioned MAE/RMSE on windows with recent alarm events
- degradation-conditioned MAE/RMSE on future degraded windows

Recommended main reporting:

- Table: average forecasting error across horizons
- Table: event-conditioned improvement over strongest baseline
- Figure: forecast trajectory examples around alarm bursts

## Task 2: KPI Degradation Classification

Prediction target:

`y_deg[c, t+h] = 1` if future KPI state is degraded.

Primary weak-label rule, fitted with train-only thresholds:

- `Call Drop Rate(%) > max(1.0, train-cell p99 or calibrated high threshold)`
- OR `RRC Setup Success Rate(%) < min(98.0, train-cell p05)`
- OR `ERAB Setup Success Rate(%) < min(98.0, train-cell p05)`
- OR `LTE_User DL Average Throughput(Mbps) < train-cell p10`
- OR `CQI AVG < train-cell p10`

Use train-cell thresholds as primary and global thresholds as sensitivity analysis. If a cell has too few valid values for stable quantiles, fall back to train-global thresholds.

Metrics:

- AUROC
- AUPRC, primary due to class imbalance
- F1 and recall at fixed precision
- event-conditioned AUPRC for windows with recent alarms
- calibration error for risk scores

Class imbalance handling:

- Report positive prevalence per split and horizon
- Use class weights or focal loss for neural models
- Compare threshold-free AUPRC and thresholded F1

## Task 3: Weak Root-Cause Ranking

No true expert root-cause labels are available in the audited CSV. Ranking should be explicitly presented as **weakly supervised root-cause candidate ranking**, not verified causal diagnosis.

Candidate alarm set for a positive degradation sample:

- Same-cell alarms observed in `[t+h-R, t+h]`, with `R = 24` hours by default
- Optional neighbor-cell alarms from inferred cross-cell graph, with lower spatial weight
- Alarm columns with all-zero train values excluded

Weak label scoring:

For degradation sample `(c, t+h)` and alarm type `a`:

`score_weak(a, c, t+h) = recent_intensity(a, c, t+h) * train_lift(a -> degradation_h) * cell_weight`

Where:

- `recent_intensity` is an exponential time-decay count, `sum exp(-(t+h - tau) / eta)`, default `eta = 6` hours
- `train_lift` is estimated on train only as lagged log-lift or log odds ratio between alarm presence in the lookback window and future degradation
- `cell_weight = 1` for same cell and lower for inferred neighbors

Weak positive labels:

- Top-1 or top-k alarm types by `score_weak`
- If no recent alarm exists, assign `NO_OBSERVED_ALARM` and exclude from ranking metrics or report separately

Ranking metrics:

- Hits@1, Hits@3, Hits@5
- MRR
- nDCG@k

Important paper language:

- "The ranking head learns to recover weak root-cause candidates induced by event-to-degradation temporal association."
- Avoid "ground-truth root cause" unless expert labels are later added.

## Splits

Chronological split by unique timestamp:

- Train: first 70%
- Validation: next 15%
- Test: final 15%

All preprocessing parameters must be fitted on train:

- KPI scaling
- degradation thresholds
- inferred cell graph
- event influence graph
- alarm-to-degradation lift

## Lookback and Patch Settings

Default lookback:

- `L = 24` hours for smoke tests and first experiments
- sensitivity: `L in {12, 24, 48, 72}`

KPI patch length:

- `P = 3` or `6` hours

Alarm event lookback:

- `R = 24` hours
- decay temperature `eta in {3, 6, 12}` hours

## Ablation Protocol

Main model compared with:

- No alarms: remove all alarm event tokens and event intensity features
- No dynamic graph: replace event influence graph with static identity or train-global graph
- No Hawkes/event intensity: keep event tokens but remove time-decayed intensity
- No tokenization: feed raw aligned KPI/alarm vectors instead of patch/event tokens
- No cross-cell graph: use per-cell temporal model only
- No ranking loss: train forecasting/classification only

Each ablation should report:

- Forecasting MAE/RMSE
- Degradation AUPRC
- Weak ranking MRR/Hits@k where applicable
- Event-conditioned metrics

## Minimum Viable Experiment

MVP for first technical draft:

1. Build clean cell-time windows from `510957.csv`.
2. Run chronological split.
3. Generate train-only thresholds for degradation labels.
4. Implement lag-feature XGBoost-style baseline using sklearn gradient boosting proxy.
5. Implement event-attention-only baseline using decayed alarm counts and lagged lift.
6. Implement first neural version after PyTorch is available.
7. Report whether alarms improve event-conditioned degradation AUPRC and forecasting errors.

The first code skeleton currently focuses on steps 1-5 with available local packages.
