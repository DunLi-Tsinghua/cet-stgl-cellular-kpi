# CLAIM_CALIBRATION

Paper line: **Causal Event-Token Spatio-Temporal Graph Learning for Alarm-Driven KPI Degradation Prediction in Cellular Networks**

This document calibrates what the current CET-STGL project can and cannot claim based on the generated artifacts and experiments. It should be treated as the guardrail for the manuscript.

## Evidence Base

Primary evidence files:

- `DATA_AUDIT_REPORT.md`
- `EXPERIMENT_PLAN.md`
- `MODEL_DESIGN.md`
- `EXPERIMENT_RESULTS.md`
- `PAPER_TABLES.md`
- `PAPER_ANALYSIS.md`
- `experiments/causal_event_token_stgl/results/paper_artifacts/paper_tables_forecasting.csv`
- `experiments/causal_event_token_stgl/results/paper_artifacts/paper_tables_classification.csv`
- `experiments/causal_event_token_stgl/results/paper_artifacts/paper_tables_ranking.csv`
- `experiments/causal_event_token_stgl/results/paper_artifacts/paper_model_status.csv`

Data basis:

- Real private cellular panel from `510957.csv`
- 33 cells
- 3,151 hourly timestamps
- 48 KPI/business columns
- 137 alarm/fault columns
- 11 active alarm/fault columns
- active-alarm row rate about 1.62%
- forecasting uses real future KPI values
- degradation and root-cause ranking use weak labels derived from the current data

## Claims That Are Supported

### C1. Real cellular structured operations data

Supported wording:

> We study alarm-driven KPI degradation prediction on a real private cellular operations panel containing KPI/business indicators and alarm/fault signals across 33 cells and 3,151 hourly timestamps.

Do not imply public benchmark status.

### C2. Causal-inspired event-token pipeline

Supported wording:

> We propose a causal-inspired event-token representation pipeline that combines KPI patch features, sparse alarm event features, cell/time encodings, train-only alarm-degradation association features, and an inferred cross-cell graph.

Acceptable phrases:

- causal-inspired event influence graph
- causal discovery-based event influence graph
- alarm-to-degradation temporal association
- event influence prior

### C3. Structured representation of KPI, event, cell and time context

Supported wording:

> The framework organizes heterogeneous operations data into KPI patch tokens, alarm event tokens, cell/time positional features, and inferred graph-context features.

This is supported by `MODEL_DESIGN.md`, `EXPERIMENT_PLAN.md`, and the runnable feature-block implementation.

### C4. Three-task evaluation

Supported wording:

> We evaluate the representation pipeline on multi-horizon KPI forecasting, weak KPI degradation classification, and weak root-cause candidate ranking.

Important qualifier:

- forecasting targets are real future KPI values
- classification labels are weak KPI-threshold labels
- ranking labels are weak alarm-lift labels

### C5. CET-STGL-sklearn is competitive in some forecasting settings

Supported wording:

> The runnable CET-STGL-sklearn proxy is competitive with the Ridge/Logistic feature baseline and is slightly better on several primary-KPI forecasting metrics.

Concrete supported numbers:

- 1h primary KPI MAE/RMSE: `CET-STGL-sklearn` 2.183/4.635 vs `ridge_logistic` 2.219/4.698
- 3h primary KPI MAE/RMSE: `CET-STGL-sklearn` 2.537/5.083 vs `ridge_logistic` 3.071/5.684
- 6h primary KPI MAE/RMSE: `CET-STGL-sklearn` 3.197/5.853 vs `ridge_logistic` 3.338/5.940
- 12h primary KPI MAE/RMSE: `CET-STGL-sklearn` 2.604/5.442 vs `ridge_logistic` 2.640/5.391

Careful interpretation:

- At 12h, CET-STGL-sklearn has slightly lower MAE but slightly higher RMSE than ridge.
- The no-alarm CET ablation is the best primary-MAE row at all four horizons, so do not claim alarm tokens drive forecasting gains.

### C6. Event-only and Hawkes-only are weak standalone predictors

Supported wording:

> Event-only and Hawkes-only variants are substantially weaker than KPI-history baselines for forecasting and weak degradation classification, reflecting the sparsity of active alarms and the KPI-threshold nature of the weak labels.

Concrete supported example:

- 1h primary KPI MAE/RMSE: `hawkes_only` 4.578/7.981 vs `ridge_logistic` 2.219/4.698
- 1h AUPRC: `hawkes_only` 0.247, `event_attention_only` 0.257, `ridge_logistic` 0.471

### C7. Alarm signals do not uniformly improve classification

Supported wording:

> Under the current weak degradation labels, adding alarm features does not uniformly improve degradation classification.

Concrete supported example:

- 1h AUPRC: `ridge_logistic_no_alarm` 0.476, `ridge_logistic` 0.471, `CET-STGL-sklearn` 0.466

This is a legitimate negative result and should be preserved.

### C8. Root-cause ranking is weak-label consistency only

Supported wording:

> Weak root-cause ranking evaluates whether a method recovers alarm candidates induced by train-only alarm-to-degradation temporal association.

Supported but bounded:

- `CET-STGL-sklearn` achieves perfect weak-label ranking in the current setup, but this is not expert root-cause validation because the same alarm-lift signal is used in weak-label construction and model scoring.

## Claims That Must Not Be Written

### N1. Do not claim a telecom foundation model

Forbidden:

- "We train a telecom foundation model."
- "CET-STGL is a foundation model for cellular networks."
- "The model learns a general telecom foundation representation."

Allowed:

- "a structured event-token representation pipeline"
- "a runnable proxy for the proposed causal-inspired graph representation"

### N2. Do not claim a benchmark paper

Forbidden:

- "We release a benchmark."
- "This is the first benchmark for telecom KPI degradation."

Allowed:

- "we evaluate on a private cellular operations slice"
- "we report a reproducible experimental protocol within the project workspace"

### N3. Do not claim strict causality

Forbidden:

- "We discover true causal relations."
- "The graph confirms causal propagation."
- "The method identifies causal root causes."

Allowed:

- "causal-inspired"
- "causal discovery-based event influence graph"
- "temporal association and event-influence prior"

### N4. Do not claim true root-cause verification

Forbidden:

- "ground-truth root cause"
- "expert-validated root cause"
- "true fault localization accuracy"

Allowed:

- "weak root-cause candidate ranking"
- "weak-label consistency"
- "proxy ranking under train-only alarm-lift labels"

### N5. Do not claim neural baselines were run

Forbidden:

- "LSTM, TCN, Informer, Autoformer, STGCN and GAT were evaluated."
- "CET-STGL outperforms STGCN/GAT."

Actual status:

- `lstm`, `tcn`, `informer`, `autoformer`, `stgcn`, `gat`: `not_run_backend_missing`
- reason: PyTorch is unavailable in the current runtime

Allowed:

- "Neural baselines are reserved for future implementation."
- "The current experiments use sklearn baselines and a runnable CET-STGL proxy."

### N6. Do not claim alarm tokens significantly improve all tasks

Forbidden:

- "Alarm tokens significantly improve all horizons and tasks."
- "Event information is the main driver of KPI prediction."

Actual result:

- no-alarm variants are often competitive or better
- event-only/Hawkes-only are weak standalone predictors
- alarm sparsity is severe

Allowed:

- "alarm features are useful for constructing weak root-cause candidates and for probing event-conditioned behavior, but gains are not uniform under current weak labels."

## Recommended Main Contribution Statements

Use conservative wording:

1. We formulate alarm-driven KPI degradation prediction as a structured multi-task problem over real cellular KPI, alarm, cell and time signals.
2. We introduce a causal-inspired event-token representation that combines KPI patch features, sparse alarm event features, train-only event influence priors, and inferred cross-cell graph context.
3. We evaluate the runnable proxy on forecasting, weak degradation classification, and weak root-cause candidate ranking over four horizons.
4. We show that KPI-history baselines are strong, event-only models are insufficient, and the CET-STGL proxy is competitive in several forecasting settings while exposing the limitations of alarm sparsity and weak supervision.

## Recommended Abstract Claim Strength

Maximum defensible level:

> Results show that the proposed representation pipeline is feasible on real enterprise cellular operations data and is competitive with strong sklearn feature baselines for several primary-KPI forecasting metrics, while event-only baselines are insufficient under sparse alarm observations.

Do not write:

> Results show state-of-the-art performance across all tasks.

