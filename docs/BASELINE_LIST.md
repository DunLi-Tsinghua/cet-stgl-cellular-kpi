# BASELINE_LIST

This list defines the baselines for the technical paper. The first runnable code skeleton implements lightweight sklearn-compatible baselines; deep baselines should be added under the same data interface.

## Baseline Families

### 1. ARIMA/VAR or XGBoost-Style Tabular Forecasting

Purpose:

- measure how far lagged KPI/alarm feature engineering can go without neural sequence modeling

Variants:

- ARIMA/SARIMA per cell and KPI
- VAR per cell for multivariate KPI forecasting
- XGBoost/LightGBM/CatBoost with lag features, rolling statistics, alarm counts, time features
- sklearn `HistGradientBoosting` or `RandomForest` as local proxy when XGBoost is unavailable

Inputs:

- KPI lags
- rolling mean/std/slope
- recent alarm counts
- time features
- cell one-hot or cell embedding proxy

Tasks:

- forecasting
- degradation classification

Expected role:

- strong non-neural baseline
- useful for feature importance sanity checks

### 2. LSTM/TCN

Purpose:

- standard temporal neural baseline without graph structure

Variants:

- LSTM over per-cell KPI/alarm sequences
- TCN with dilated causal convolutions
- TCN with alarm exogenous channels

Inputs:

- per-cell aligned KPI and alarm matrices
- no cross-cell graph

Tasks:

- forecasting
- degradation classification

Expected role:

- tests whether graph/cross-cell modeling matters beyond temporal memory

### 3. Informer or Autoformer

Purpose:

- long-horizon time-series Transformer baseline

Variants:

- Informer
- Autoformer
- PatchTST if easier to implement

Inputs:

- KPI sequences
- alarm channels as exogenous variables
- time features

Tasks:

- 1/3/6/12-hour KPI forecasting
- optional degradation head

Expected role:

- strong sequence-only Transformer baseline
- distinguishes generic time-series Transformer gains from event graph gains

### 4. STGCN/GAT

Purpose:

- graph temporal baseline using cell graph but without causal-inspired event influence graph

Graph construction:

- primary: train-only KPI correlation graph
- secondary: same-site prefix graph from cell IDs
- sensitivity: fully connected top-k correlation graph

Variants:

- STGCN
- DCRNN
- GAT + temporal convolution
- Graph WaveNet if implementation time allows

Inputs:

- KPI time series per cell
- optional alarm channels as node features

Tasks:

- forecasting
- degradation classification

Expected role:

- tests whether the proposed event influence graph adds value beyond generic cell graph modeling

### 5. Hawkes-Only or Event-Attention-Only

Purpose:

- isolate sparse alarm event contribution without rich KPI temporal modeling

Hawkes-only:

- decayed alarm intensity per alarm type and cell
- optional train-estimated alarm-to-degradation lagged lift
- logistic or gradient boosting classifier for degradation

Event-attention-only:

- event tokens with alarm type, cell/time encoding, and decay
- attention pooling over recent events
- no KPI history or minimal current KPI state

Tasks:

- degradation classification
- weak root-cause ranking

Expected role:

- tests whether alarm events alone carry predictive degradation signal

## Fair Comparison Rules

All baselines must use:

- same chronological splits
- same forecast horizons
- train-only preprocessing
- same weak degradation labels
- same target KPI set

Report both:

- all-window metrics
- event-conditioned metrics on samples with recent alarm events

## Minimum First-Round Baselines

Given the current local environment, run these first:

1. persistence forecast
2. sklearn ridge/gradient boosting with KPI lag and alarm aggregate features
3. event-intensity-only logistic classifier

Then add:

4. XGBoost once dependency is available
5. LSTM/TCN once PyTorch is available
6. STGCN/GAT once PyTorch Geometric or a lightweight graph implementation is available

## Baseline Reporting Table

| Baseline | KPI history | Alarm history | Cross-cell graph | Event influence graph | Forecasting | Classification | Ranking |
|---|---|---|---|---|---|---|---|
| Persistence | yes | no | no | no | yes | no | no |
| ARIMA/VAR | yes | optional | no | no | yes | optional | no |
| XGBoost-style | yes | yes | optional features | no | yes | yes | optional |
| LSTM/TCN | yes | yes | no | no | yes | yes | no |
| Informer/Autoformer | yes | yes | no | no | yes | yes | no |
| STGCN/GAT | yes | optional | yes | no | yes | yes | no |
| Hawkes-only | no/minimal | yes | optional | yes, simplified | no | yes | yes |
| CET-STGL | yes | yes | yes | yes | yes | yes | yes |
