# Reproducibility and Claim Boundary

This repository accompanies the CET-STGL manuscript:

**CET-STGL: A Causal-Inspired Event-Token Spatio-Temporal Framework for Alarm-Aware KPI Degradation Prediction in Cellular Networks**

## Data

The raw Huawei MBB operational records are not released. They contain private cellular KPI, business-counter and alarm information. The repository includes:

- a sanitized public schema/statistics record;
- a synthetic toy panel for interface testing;
- paper table/figure artifacts generated from the authorized private-data experiments.

Exact metric reproduction requires authorized access to the private operational panel and the leakage-controlled protocol described in the paper.

## Code

The public code includes:

- feature construction and weak degradation labeling;
- classical and lightweight sklearn-based runnable baselines;
- weak alarm-candidate prioritization utilities;
- fixed LSTM and TCN sequence-baseline runners;
- paper table/figure artifacts.

The LSTM and TCN baselines are sequence-model comparators. They are not full neural spatio-temporal graph versions of CET-STGL.

## Claims Not Made

The repository and manuscript do not claim:

- a telecom foundation model;
- a public benchmark dataset;
- causal-effect identification;
- expert-confirmed root-cause analysis;
- state-of-the-art performance.

The event influence prior is causal-inspired and association-based. Weak alarm-candidate ranking evaluates proxy consistency only.
