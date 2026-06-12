from __future__ import annotations

import math
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "experiments" / "causal_event_token_stgl" / "results" / "paper_artifacts"
PAPER = ROOT / "paper_cet_stgl"
SECTIONS = PAPER / "sections"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"


DATA_STATS = {
    "Rows": "103,983",
    "Original columns": "187",
    "Cells": "33",
    "Unique hourly timestamps": "3,151",
    "Time range": "2024-01-10 00:00:00 to 2024-06-10 06:00:00",
    "KPI/business columns": "48",
    "Alarm/fault columns": "137",
    "Active alarm/fault columns": "11",
    "Rows with active alarms": "1,681 (1.62%)",
    "Missing values": "None detected",
    "Detected time discontinuity": "One 21 days 01:00:00 gap; crossing windows removed",
}

MODEL_SUBSET = [
    "Ridge/Logistic",
    "Ridge/Logistic w/o alarms",
    "Hawkes-only",
    "Event-attention-only",
    "CET-STGL-sklearn",
    "CET w/o alarm tokens",
]

RANKING_SUBSET = [
    "Ridge/Logistic",
    "Hawkes-only",
    "Event-attention-only",
    "CET-STGL-sklearn",
    "CET w/o dynamic event graph",
]

PALETTE = {
    "Ridge/Logistic": "#4E79A7",
    "Ridge/Logistic w/o alarms": "#86BCB6",
    "Hawkes-only": "#F28E2B",
    "Event-attention-only": "#E15759",
    "CET-STGL-sklearn": "#59A14F",
    "CET w/o alarm tokens": "#B07AA1",
    "CET w/o dynamic event graph": "#9C755F",
    "CET w/o cross-cell graph": "#BAB0AC",
    "CET w/o Hawkes intensity": "#EDC948",
    "CET w/o tokenization": "#AF7AA1",
}


def ensure_dirs() -> None:
    for path in [PAPER, SECTIONS, TABLES, FIGURES]:
        path.mkdir(parents=True, exist_ok=True)


def read_artifacts() -> dict[str, pd.DataFrame]:
    return {
        "forecasting": pd.read_csv(ART / "paper_tables_forecasting.csv"),
        "classification": pd.read_csv(ART / "paper_tables_classification.csv"),
        "ranking": pd.read_csv(ART / "paper_tables_ranking.csv"),
        "status": pd.read_csv(ART / "paper_model_status.csv"),
        "fig_long": pd.read_csv(ART / "paper_figure_long.csv"),
        "ablation": pd.read_csv(ART / "paper_ablation_delta.csv"),
    }


def fmt(x: object, digits: int = 4) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except TypeError:
        pass
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.{digits}f}"
    return str(x)


def latex_escape(text: object) -> str:
    s = "" if text is None or (isinstance(text, float) and math.isnan(text)) else str(text)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def get_row(df: pd.DataFrame, model_label: str, horizon: int) -> pd.Series:
    row = df[(df["model_label"] == model_label) & (df["horizon"] == horizon)]
    if row.empty:
        raise KeyError(f"Missing row: {model_label}, horizon {horizon}")
    return row.iloc[0]


def build_manuscript_v1(d: dict[str, pd.DataFrame]) -> str:
    f = d["forecasting"]
    c = d["classification"]
    r = d["ranking"]

    cet1 = get_row(f, "CET-STGL-sklearn", 1)
    ridge1 = get_row(f, "Ridge/Logistic", 1)
    hawkes1 = get_row(f, "Hawkes-only", 1)
    cet3 = get_row(f, "CET-STGL-sklearn", 3)
    ridge3 = get_row(f, "Ridge/Logistic", 3)
    cet6 = get_row(f, "CET-STGL-sklearn", 6)
    ridge6 = get_row(f, "Ridge/Logistic", 6)
    cet12 = get_row(f, "CET-STGL-sklearn", 12)
    ridge12 = get_row(f, "Ridge/Logistic", 12)

    cls_cet1 = get_row(c, "CET-STGL-sklearn", 1)
    cls_ridge1 = get_row(c, "Ridge/Logistic", 1)
    cls_noalarm1 = get_row(c, "Ridge/Logistic w/o alarms", 1)
    cls_hawkes1 = get_row(c, "Hawkes-only", 1)
    cls_event1 = get_row(c, "Event-attention-only", 1)

    rank_cet1 = get_row(r, "CET-STGL-sklearn", 1)
    rank_hawkes1 = get_row(r, "Hawkes-only", 1)

    return f"""# Causal-Inspired Event-Token Spatio-Temporal Graph Learning for Alarm-Driven KPI Degradation Prediction in Cellular Networks

## Abstract

Cellular network operations are monitored through heterogeneous time-evolving signals, including KPI measurements, business counters, sparse alarm/fault events, cell identifiers and temporal context. Predicting future KPI degradation from these signals is operationally important, but difficult because alarms are sparse, KPI responses may be delayed, physical topology is often unavailable, and expert-confirmed root-cause labels are not always recorded in structured datasets. This paper studies the problem on a real private cellular operations panel extracted from `510957.csv`, containing 103,983 rows, 33 cells, 3,151 hourly timestamps, 48 KPI/business columns and 137 alarm/fault columns, of which 11 alarm/fault columns are active in the observed slice. We propose CET-STGL, a causal-inspired event-token spatio-temporal learning pipeline that organizes KPI histories into patch tokens, sparse alarms into event tokens, cell/time context into positional features, and train-only alarm-degradation associations into an event influence prior with an inferred cross-cell context graph. In the current runnable implementation, we evaluate an sklearn proxy of the representation rather than a neural graph backend, because PyTorch is unavailable in the project runtime. Experiments cover 1/3/6/12-hour KPI forecasting, weak KPI degradation classification and weak root-cause candidate ranking. The results show that KPI-history baselines are strong, event-only and Hawkes-only variants are weak standalone predictors, and the CET-STGL proxy is competitive with Ridge/Logistic baselines on several primary-KPI forecasting metrics. At 1 hour, CET-STGL-sklearn obtains primary-KPI MAE/RMSE of {fmt(cet1['primary_mae'], 3)}/{fmt(cet1['primary_rmse'], 3)}, compared with {fmt(ridge1['primary_mae'], 3)}/{fmt(ridge1['primary_rmse'], 3)} for Ridge/Logistic. The evidence supports a cautious technical claim: causal-inspired event-token structuring is feasible and testable on real enterprise cellular operations data, while alarm-token gains are not uniform, weak ranking is not expert root-cause verification, and neural spatio-temporal baselines remain future work.

**Keywords:** cellular networks; KPI forecasting; alarm correlation; weak root-cause ranking; spatio-temporal graph learning; causal-inspired event influence.

## 1. Introduction

Modern cellular networks generate large volumes of operational telemetry. At the cell level, operators observe KPI and business counters, alarm/fault indicators, timestamps and cell identifiers. These signals are used to detect service degradation, anticipate future performance problems and narrow the set of possible fault candidates. The practical question is not only whether a KPI will change, but whether a future cell state is likely to become degraded and which recent events deserve inspection.

This setting is naturally structured. KPI measurements are continuous and temporally correlated, alarms are sparse and irregular, cells may share operating conditions, and the same alarm type may have different relevance depending on timing and context. A purely sequence-based model can ignore event semantics; a purely event-based model can miss the dominant KPI dynamics. The paper therefore studies a middle path: a structured event-token pipeline that represents KPI, event, cell and time information explicitly, then evaluates the representation on forecasting, weak degradation classification and weak root-cause candidate ranking.

The study is deliberately narrower than telecom foundation-model work. We do not train a telecom foundation model, do not claim a public benchmark, and do not report unrun neural baselines. The current dataset is a private enterprise cellular slice. Forecasting labels are real future KPI values, but degradation labels are weak KPI-threshold labels fitted from the training split, and ranking labels are weak alarm-lift candidates. The event influence graph is causal-inspired or causal discovery-based in the sense that it uses temporal precedence and train-only alarm-degradation association; it is not confirmed causality.

The paper makes four calibrated contributions.

1. It formulates alarm-driven KPI degradation prediction as a multi-task cell-time learning problem over real KPI/business and alarm/fault panels.
2. It proposes a causal-inspired event-token representation pipeline combining KPI patch tokens, alarm event tokens, cell/time positional features, train-only event influence priors and inferred cross-cell graph context.
3. It reports a runnable experimental protocol for 1/3/6/12-hour forecasting, weak degradation classification and weak root-cause candidate ranking.
4. It presents conservative empirical findings: KPI-history baselines remain strong, event-only/Hawkes-only variants are insufficient as standalone predictors, the CET-STGL sklearn proxy is competitive in several forecasting settings, and weak ranking should be interpreted only as proxy consistency.

## 2. Related Work

This work touches five literature areas that should be completed with real citations before submission.

First, KPI forecasting methods for cellular and communication networks model future performance counters using autoregressive models, tree-based regressors, recurrent networks, temporal convolutions or transformer-style sequence models. Second, alarm correlation and root-cause analysis methods seek to group fault events and identify candidate causes, often under sparse and noisy event observations. Third, Hawkes-process and point-process models provide a principled vocabulary for self-exciting or temporally decayed event influence, although the current implementation uses Hawkes-inspired decay features rather than a fitted multi-type point process. Fourth, spatio-temporal graph learning provides tools for dependency modeling across networked entities when topology or inferred relation graphs are available. Fifth, telecom LLM and telecom benchmark work provides useful background contrast, but the present paper is not a foundation-model or benchmark paper.

The final manuscript should add verified references for these areas. This draft intentionally avoids placeholder citations that could be mistaken for checked bibliography.

## 3. Problem Formulation

Let `c` denote a cell and `t` an hourly timestamp. Each cell-time row contains a KPI/business vector `x_{{c,t}}`, an alarm/fault vector `a_{{c,t}}`, a cell identifier and timestamp-derived context. Given a lookback window of length `L`, the model predicts outcomes at horizon `h` in {{1, 3, 6, 12}} hours. Windows that cross the detected 21-day gap are removed, so the learning problem does not assume false temporal continuity.

### 3.1 Multi-step KPI forecasting

The forecasting task predicts the future KPI vector `x_{{c,t+h}}` from historical KPI, alarm, cell and time context. These labels are real observed future values from `510957.csv`. We report aggregate results over all 48 KPI/business columns and primary results over eight core KPIs used for paper-level evaluation.

### 3.2 Weak KPI degradation classification

The degradation task predicts a binary weak label `y_{{c,t+h}}` indicating whether the future cell state satisfies threshold-based degradation criteria. Because no expert incident labels are present in the audited CSV, the thresholds are fitted from the training split and applied chronologically. These labels should be interpreted as weak KPI degradation labels, not confirmed service incidents.

### 3.3 Weak root-cause candidate ranking

For degraded samples with recent alarms, the ranking task orders candidate alarm types. The target is a weak candidate induced by train-only alarm-to-degradation lift and recent event intensity. This evaluates weak-label consistency or proxy ranking only. It is not expert-confirmed root-cause localization.

### 3.4 Causal-inspired graph boundary

The event influence graph uses temporal precedence and train-only associations. Such a graph may be useful for organizing event influence hypotheses, but the observational dataset does not support strict causal identification. Throughout the paper, the graph is described as causal-inspired or causal discovery-based rather than causal proof.

## 4. Methodology

### 4.1 Overview

CET-STGL is designed as a representation and learning pipeline for structured cellular operations data. It has four input blocks: KPI patch tokens, alarm event tokens, cell/time positional encodings and graph-context features. A full neural version would feed these blocks into a spatio-temporal graph encoder with three heads. The current experiments implement a runnable sklearn proxy over the same feature blocks, which is sufficient to test data preparation, weak labeling, ablation structure and the empirical value of event/cell/time features.

### 4.2 KPI patch tokens

KPI histories are represented through short temporal patches. For each KPI in a lookback window, patch-level summaries such as local mean, variation and trend approximate the local temporal shape. This reduces raw sequence length and separates short-term KPI dynamics from cell and time context. In the current proxy, patch summaries are computed for all KPI/business columns, while paper-level forecasting emphasizes the primary KPI subset.

### 4.3 Alarm event tokens

Alarm/fault columns are sparse. The audit found 137 alarm/fault columns but only 11 active columns, and only 1.62% of rows contain at least one active alarm. CET-STGL therefore represents alarms as sparse event tokens with rolling counts, recency and decayed intensity features. The decayed intensity is Hawkes-inspired, but the current code does not fit a full multi-type Hawkes process.

### 4.4 Cell and time positional features

The method includes cell identity and cyclical time features such as hour-of-day and day-of-week. These features encode recurring operational patterns and persistent cell-level offsets. They are fitted within the chronological training protocol to avoid leakage.

### 4.5 Causal-inspired event influence graph

For each alarm type and horizon, train-only lagged lift estimates the association between recent alarm occurrence and future weak degradation. These values define an event influence prior that can weight alarm event tokens. The prior is causal-inspired because it respects temporal ordering and is estimated on the training split, but it is still an observational association. It should be used for event candidate prioritization and hypothesis generation, not as confirmed causality.

### 4.6 Inferred cross-cell graph

The dataset does not contain a verified physical topology. The cross-cell context graph is therefore inferred from training-period KPI correlation or related cell-context heuristics. It is a statistical graph used for context aggregation, not a physical network topology. The ablation without cross-cell graph tests the sensitivity of the proxy to this component.

### 4.7 Prediction heads and runnable proxy

The runnable implementation uses sklearn-compatible heads: Ridge regression for KPI forecasting, logistic-style classification for weak degradation, and event-lift/intensity scoring for weak ranking. LSTM, TCN, Informer, Autoformer, STGCN and GAT are listed in the model-status table as `not_run_backend_missing` because PyTorch is unavailable. The reported CET-STGL result should therefore be read as a proxy over the proposed representation, not as a completed neural graph encoder.

## 5. Experimental Setup

### 5.1 Dataset and preprocessing

The dataset contains 103,983 rows, 187 original columns, 33 cells and 3,151 unique hourly timestamps from 2024-01-10 00:00:00 to 2024-06-10 06:00:00. It is a balanced cell-time panel with 48 KPI/business columns and 137 alarm/fault columns. No missing values were detected. One long gap of 21 days and 1 hour is present; windows crossing this gap are excluded.

### 5.2 Splits and horizons

Experiments use chronological splits by timestamp: 70% train, 15% validation and 15% test. All scalers, weak thresholds, event influence scores and inferred graphs are fitted from the training split only. Prediction horizons are 1, 3, 6 and 12 hours.

### 5.3 Compared models

Runnable models include Ridge/Logistic, Ridge/Logistic without alarms, Hawkes-only, Event-attention-only, CET-STGL-sklearn and CET ablations removing alarm tokens, dynamic event graph, Hawkes/event intensity, tokenization and cross-cell graph. Neural baselines are not reported because the current runtime does not provide PyTorch.

### 5.4 Metrics

Forecasting is evaluated with MAE, RMSE and MAPE with denominator floor `max(abs(y_true), 1.0)`. MAE and RMSE are the primary interpretation metrics. Weak degradation classification is evaluated with AUPRC, AUROC, F1, precision and recall. Weak root-cause candidate ranking is evaluated with Hit@1, Hit@3, Hit@5, MRR and nDCG@5. Ranking metrics are weak-label consistency scores only.

## 6. Results and Analysis

### 6.1 Forecasting

The CET-STGL proxy is competitive with the Ridge/Logistic baseline on several primary-KPI forecasting metrics. At 1 hour, CET-STGL-sklearn obtains MAE/RMSE {fmt(cet1['primary_mae'], 3)}/{fmt(cet1['primary_rmse'], 3)}, compared with {fmt(ridge1['primary_mae'], 3)}/{fmt(ridge1['primary_rmse'], 3)} for Ridge/Logistic. At 3 hours the corresponding values are {fmt(cet3['primary_mae'], 3)}/{fmt(cet3['primary_rmse'], 3)} versus {fmt(ridge3['primary_mae'], 3)}/{fmt(ridge3['primary_rmse'], 3)}. At 6 hours they are {fmt(cet6['primary_mae'], 3)}/{fmt(cet6['primary_rmse'], 3)} versus {fmt(ridge6['primary_mae'], 3)}/{fmt(ridge6['primary_rmse'], 3)}. At 12 hours, CET-STGL-sklearn has slightly lower MAE than Ridge/Logistic ({fmt(cet12['primary_mae'], 3)} versus {fmt(ridge12['primary_mae'], 3)}) but slightly higher RMSE ({fmt(cet12['primary_rmse'], 3)} versus {fmt(ridge12['primary_rmse'], 3)}).

This supports a cautious statement of competitiveness, not a universal superiority claim. The CET ablation without alarm tokens has the lowest primary MAE at all four horizons in the current tables. Therefore, the evidence does not show that alarm tokens consistently improve forecasting. A plausible explanation is that many KPI values are driven by regular temporal and load dynamics, whereas active alarms occur in only 1.62% of rows.

Event-only methods are much weaker for continuous forecasting. At the 1-hour horizon, Hawkes-only obtains primary-KPI MAE/RMSE {fmt(hawkes1['primary_mae'], 3)}/{fmt(hawkes1['primary_rmse'], 3)}, which is far worse than KPI-history baselines. This indicates that sparse alarms alone cannot reconstruct continuous KPI trajectories in this data slice.

### 6.2 Weak degradation classification

At the 1-hour horizon, CET-STGL-sklearn obtains AUPRC/F1 {fmt(cls_cet1['auprc'], 3)}/{fmt(cls_cet1['f1'], 3)}. Ridge/Logistic obtains {fmt(cls_ridge1['auprc'], 3)}/{fmt(cls_ridge1['f1'], 3)}, while Ridge/Logistic without alarms obtains {fmt(cls_noalarm1['auprc'], 3)}/{fmt(cls_noalarm1['f1'], 3)}. Under the current weak KPI-threshold labels, alarm features therefore do not produce a uniform classification gain.

Hawkes-only and event-attention-only are weaker at the same horizon, with AUPRC {fmt(cls_hawkes1['auprc'], 3)} and {fmt(cls_event1['auprc'], 3)}, respectively. This aligns with the data audit: active alarms are sparse, while weak degradation labels are primarily defined from KPI thresholds. Event features are still useful for candidate ranking and event-conditioned analysis, but they are not sufficient standalone predictors in this slice.

### 6.3 Weak root-cause candidate ranking

CET-STGL-sklearn reaches Hit@1/MRR {fmt(rank_cet1['hit_at_1'], 3)}/{fmt(rank_cet1['mrr'], 3)} at the 1-hour horizon under the weak ranking rule. Hawkes-only obtains Hit@1/MRR {fmt(rank_hawkes1['hit_at_1'], 3)}/{fmt(rank_hawkes1['mrr'], 3)}. These values should not be interpreted as true root-cause localization accuracy. The ranking target is constructed from train-only alarm-to-degradation lift and recent event intensity, and some scoring variants use the same signal. The result is a proxy consistency check.

### 6.4 Ablation analysis

The ablation table provides a useful negative result. Removing alarm tokens is competitive or better for several forecasting metrics, which implies that KPI history dominates aggregate prediction in this dataset. Removing Hawkes/event intensity changes several metrics only slightly, suggesting redundancy with rolling alarm counts or insufficient signal density. Removing tokenization and cross-cell context can affect results by horizon and task, but the evidence is not strong enough to claim universal benefit.

Overall, the ablations support the representation framework as a testable engineering pipeline, while also showing that the current dataset does not justify broad claims about alarm-token superiority.

## 7. Discussion and Limitations

The main value of this study is not a claim of state-of-the-art neural graph performance. Its value is a disciplined technical formulation and a runnable, audited pipeline for real cellular operations data. The results show where the signal is strong, where it is weak, and which claims should be avoided.

The limitations are substantial. First, the data are private and come from one project slice. They should not be presented as a public benchmark or evidence of broad cross-operator generalization. Second, degradation and ranking labels are weak labels. Forecasting labels are real future KPI values, but classification and ranking lack expert confirmation. Third, the event influence graph is causal-inspired, not strictly causal. Observational associations cannot establish true causality without interventions, counterfactual assumptions or external validation. Fourth, neural baselines were not run because PyTorch is unavailable; LSTM, TCN, Informer, Autoformer, STGCN and GAT remain future implementation targets. Fifth, alarm sparsity is severe: only 11 of 137 alarm/fault columns are active. Sixth, the cross-cell graph is inferred from available statistics rather than verified physical topology.

These limitations should remain visible in any submission. They make the paper more credible, not weaker, because they prevent the current evidence from being overstated.

## 8. Conclusion

This paper presents a cautious technical study of causal-inspired event-token spatio-temporal learning for alarm-driven KPI degradation prediction in cellular networks. On a real private cellular operations panel, CET-STGL structures KPI histories, sparse alarms, cell/time context and inferred graph information into a multi-task representation for forecasting, weak degradation classification and weak root-cause candidate ranking. The runnable sklearn proxy is competitive with Ridge/Logistic baselines on several primary-KPI forecasting metrics, while event-only and Hawkes-only variants are insufficient as standalone predictors. The same experiments also show important boundaries: alarm-token gains are not uniform, ranking is weak-label consistency rather than expert root-cause validation, and neural graph baselines are not yet executed. Future work should add expert incident/root-cause annotations, evaluate multi-region data, and implement the full neural spatio-temporal graph encoder under the same leakage-controlled protocol.
"""


def make_latex_sections(manuscript: str) -> dict[str, str]:
    # Hand-tuned LaTeX sections derived from MANUSCRIPT_V1.md. Citations are left
    # as TODO comments to avoid fabricated references.
    return {
        "01_introduction.tex": r"""\section{Introduction}

Modern cellular networks generate large volumes of operational telemetry. At the cell level, operators observe KPI and business counters, alarm/fault indicators, timestamps and cell identifiers. These signals are used to detect service degradation, anticipate future performance problems and narrow possible fault candidates.

This paper studies a structured event-token pipeline for alarm-driven KPI degradation prediction. The objective is not to train a telecom foundation model or to release a benchmark. Instead, we focus on a private enterprise cellular operations slice and ask how KPI, event, cell and time signals can be represented for forecasting, weak degradation classification and weak root-cause candidate ranking.

The study uses a real panel extracted from \texttt{510957.csv}. Forecasting labels are real future KPI values. Degradation labels are weak KPI-threshold labels fitted from the training split. Root-cause ranking labels are weak alarm-lift candidates. The event influence graph is causal-inspired or causal discovery-based, not confirmed causality.

The contributions are:
\begin{enumerate}
    \item A multi-task formulation for alarm-driven KPI degradation prediction over real cellular KPI/business and alarm/fault panels.
    \item A causal-inspired event-token representation combining KPI patch tokens, alarm event tokens, cell/time positional features, train-only event influence priors and inferred cross-cell graph context.
    \item A runnable experimental protocol covering 1/3/6/12-hour forecasting, weak degradation classification and weak root-cause candidate ranking.
    \item Calibrated empirical findings showing strong KPI-history baselines, weak standalone event-only predictors and competitive CET-STGL proxy forecasting results under clear limitations.
\end{enumerate}
""",
        "02_related_work.tex": r"""\section{Related Work}

This work connects to five research areas. First, KPI forecasting for communication networks studies temporal prediction of performance counters using statistical, machine-learning and neural sequence models. Second, alarm correlation and root-cause analysis methods seek to group events and prioritize fault candidates under sparse noisy observations. Third, Hawkes-process and point-process models provide a vocabulary for temporally decayed and self-exciting event influence; this paper uses Hawkes-inspired decay features rather than a fully fitted multi-type point process. Fourth, spatio-temporal graph learning models dependencies across networked entities when topology or inferred relations are available. Fifth, telecom LLM and telecom benchmark studies provide background contrast, but the present work is neither a foundation-model paper nor a benchmark paper.

% TODO: Add verified citations for each area before submission. Do not leave fabricated references.
""",
        "03_problem_formulation.tex": r"""\section{Problem Formulation}

Let $c$ denote a cell and $t$ an hourly timestamp. Each cell-time row contains a KPI/business vector $\mathbf{x}_{c,t}$, an alarm/fault vector $\mathbf{a}_{c,t}$, a cell identifier and timestamp-derived context. Given a lookback window of length $L$, the model predicts outcomes at horizon $h \in \{1,3,6,12\}$ hours. Windows crossing the detected long time gap are removed.

\subsection{KPI Forecasting}
The forecasting task predicts future KPI values $\mathbf{x}_{c,t+h}$. These labels are real future observations from the private cellular panel.

\subsection{Weak Degradation Classification}
The degradation task predicts a binary weak label $y_{c,t+h}$ induced by training-split KPI thresholds. These are weak KPI degradation labels, not expert incident labels.

\subsection{Weak Root-Cause Candidate Ranking}
For degraded samples with recent alarms, the ranking task orders alarm candidates. The target candidate is induced by train-only alarm-to-degradation lift and recent event intensity. This task evaluates weak-label consistency only and should not be interpreted as expert-confirmed root-cause localization.

\subsection{Causal-Inspired Graph Boundary}
The event graph uses temporal precedence and train-only association. It is a causal-inspired or causal discovery-based event influence graph, not a confirmed causal graph.
""",
        "04_methodology.tex": r"""\section{Methodology}

\subsection{Overview}
CET-STGL is a representation and learning pipeline for structured cellular operations data. It combines KPI patch tokens, alarm event tokens, cell/time positional encodings and graph-context features. A full neural implementation would use these features in a spatio-temporal graph encoder with forecasting, classification and ranking heads. The current experiments report a runnable sklearn proxy because the PyTorch backend is unavailable.

\subsection{KPI Patch Tokens}
KPI histories are transformed into short temporal patches. Patch summaries capture local level, variation and trend, reducing raw sequence length while preserving short-term KPI dynamics.

\subsection{Alarm Event Tokens}
Alarm/fault signals are sparse: only 11 of 137 alarm/fault columns are active and only 1.62\% of rows contain active alarms. Alarm event tokens therefore use rolling counts, recency and decayed intensity summaries. The decay features are Hawkes-inspired but do not constitute a fitted multi-type Hawkes model.

\subsection{Cell and Time Positional Features}
The model includes cell identity and cyclical time encodings for recurring operational behavior. These features are fitted within the chronological training protocol.

\subsection{Causal-Inspired Event Influence Graph}
For each alarm type and horizon, train-only lagged lift estimates association between recent alarms and future weak degradation. This defines an event influence prior. The prior is useful for organizing event hypotheses but does not prove causality.

\subsection{Inferred Cross-Cell Graph}
Because physical topology is not provided, cross-cell context is inferred from training-period statistics such as KPI correlation. This is a statistical context graph, not verified deployment topology.

\subsection{Prediction Heads}
The runnable proxy uses Ridge regression for forecasting, logistic-style classification for weak degradation and alarm-lift/event-intensity scoring for weak ranking. Neural baselines are listed only as not-run backends.
""",
        "05_experiments.tex": r"""\section{Experimental Setup}

\subsection{Dataset}
The private panel contains 103,983 rows, 187 original columns, 33 cells and 3,151 unique hourly timestamps from 2024-01-10 00:00:00 to 2024-06-10 06:00:00. It includes 48 KPI/business columns and 137 alarm/fault columns. No missing values were detected. One long 21-day and 1-hour gap is present, and crossing windows are removed.

\subsection{Splits and Horizons}
All experiments use chronological splits by timestamp: 70\% train, 15\% validation and 15\% test. All thresholds, scalers, inferred graphs and event influence scores are fitted from training data only. Horizons are 1, 3, 6 and 12 hours.

\subsection{Models}
Runnable models include Ridge/Logistic, Ridge/Logistic without alarms, Hawkes-only, Event-attention-only, CET-STGL-sklearn and CET ablations removing alarm tokens, dynamic event graph, Hawkes/event intensity, tokenization and cross-cell graph. LSTM, TCN, Informer, Autoformer, STGCN and GAT are not run because PyTorch is unavailable.

\subsection{Metrics}
Forecasting uses MAE, RMSE and MAPE with denominator floor $\max(|y|,1.0)$. Weak degradation classification uses AUPRC, AUROC, F1, precision and recall. Weak ranking uses Hit@1, Hit@3, Hit@5, MRR and nDCG@5, interpreted as weak-label consistency only.

\input{tables/table_data_statistics}
\input{tables/table_model_status}
""",
        "06_results.tex": r"""\section{Results and Analysis}

\subsection{Forecasting}
The CET-STGL sklearn proxy is competitive with Ridge/Logistic on several primary-KPI forecasting metrics. At 1 hour, CET-STGL-sklearn obtains MAE/RMSE 2.183/4.635, compared with 2.219/4.698 for Ridge/Logistic. At 3 hours the corresponding values are 2.537/5.083 versus 3.071/5.684. At 6 hours they are 3.197/5.853 versus 3.338/5.940. At 12 hours CET-STGL-sklearn has slightly lower MAE but slightly higher RMSE than Ridge/Logistic.

The no-alarm CET ablation has the lowest primary MAE at all four horizons. Thus the results do not support a claim that alarm tokens universally improve forecasting. Event-only methods are much weaker; Hawkes-only has 1-hour primary MAE/RMSE 4.578/7.981.

\input{tables/table_forecasting_results}

\subsection{Weak Degradation Classification}
At 1 hour, the CET-STGL proxy obtains AUPRC/F1 0.466/0.479. The Ridge/Logistic baseline obtains 0.471/0.477, while its no-alarm variant obtains 0.476/0.480. Under current weak KPI-threshold labels, alarm features do not provide a uniform classification gain.

\input{tables/table_classification_results}

\subsection{Weak Root-Cause Candidate Ranking}
Ranking is evaluated only as weak-label consistency. CET-STGL-sklearn reaches perfect weak-ranking consistency in this setup, but this does not validate true root causes because the labels are derived from alarm-lift/event-intensity proxies.

\input{tables/table_weak_ranking_results}

\subsection{Ablation Analysis}
The ablations show that the representation blocks are testable, but gains depend on task and horizon. Removing alarm tokens is often competitive for forecasting, and removing Hawkes/event intensity has little effect in several rows.

\input{tables/table_ablation_delta}

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/Fig1_forecasting_vs_horizon.pdf}
    \caption{Primary-KPI forecasting error versus horizon. The plot uses real future KPI values from the private panel. Lower values are better.}
    \label{fig:forecasting}
\end{figure}

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/Fig2_classification_comparison.pdf}
    \caption{Weak KPI degradation classification under training-split threshold labels. Scores are not expert incident-detection accuracy.}
    \label{fig:classification}
\end{figure}

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/Fig3_ablation_delta.pdf}
    \caption{Ablation deltas relative to CET-STGL-sklearn. Negative deltas are better for MAE, while positive deltas are better for AUPRC and MRR.}
    \label{fig:ablation}
\end{figure}

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/Fig4_weak_ranking_comparison.pdf}
    \caption{Weak root-cause candidate ranking consistency. These are proxy labels derived from alarm lift and event intensity, not expert-confirmed root causes.}
    \label{fig:ranking}
\end{figure}
""",
        "07_discussion_limitations.tex": r"""\section{Discussion and Limitations}

The current evidence supports a disciplined technical pipeline rather than a broad performance claim. KPI history is strong, event-only signals are weak standalone predictors, and alarm-token gains are not uniform under weak labels.

The limitations are central to interpretation. The data are private and come from a single project slice, so the work is not a public benchmark. Degradation and ranking labels are weak labels; only forecasting uses real future KPI targets. The event influence graph is causal-inspired rather than strictly causal. Neural baselines are not run because PyTorch is unavailable. Alarm sparsity is severe, with only 11 active alarm/fault columns and 1.62\% active-alarm rows. The cross-cell graph is inferred statistically rather than verified from physical topology.
""",
        "08_conclusion.tex": r"""\section{Conclusion}

This paper presents a cautious technical study of causal-inspired event-token spatio-temporal learning for alarm-driven KPI degradation prediction. On a real private cellular operations panel, CET-STGL structures KPI histories, sparse alarms, cell/time context and inferred graph information for forecasting, weak degradation classification and weak root-cause candidate ranking. The sklearn proxy is competitive with Ridge/Logistic baselines on several primary-KPI forecasting metrics, while event-only and Hawkes-only variants are weak standalone predictors. The results also define clear boundaries: no foundation-model claim, no benchmark claim, no strict causality claim, no expert root-cause validation and no fabricated neural baseline performance.
""",
    }


def build_main_tex() -> str:
    return r"""\documentclass[11pt]{article}

% Generic draft template. Replace with a target venue template only after the
% claims, references and table/figure selection are finalized.
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{graphicx}
\usepackage{caption}
\usepackage{hyperref}
\usepackage{placeins}

\title{Causal-Inspired Event-Token Spatio-Temporal Graph Learning for Alarm-Driven KPI Degradation Prediction in Cellular Networks}
\author{Author names withheld for draft}
\date{}

\begin{document}
\maketitle

\begin{abstract}
Cellular network operations are monitored through heterogeneous time-evolving signals, including KPI measurements, business counters, sparse alarm/fault events, cell identifiers and temporal context. Predicting future KPI degradation is operationally important but difficult because alarms are sparse, KPI responses may be delayed, physical topology is often unavailable and expert-confirmed root-cause labels are not always recorded. We study a real private cellular operations panel extracted from \texttt{510957.csv}, containing 103,983 rows, 33 cells, 3,151 hourly timestamps, 48 KPI/business columns and 137 alarm/fault columns, of which 11 are active. We propose CET-STGL, a causal-inspired event-token spatio-temporal learning pipeline that organizes KPI histories into patch tokens, sparse alarms into event tokens, cell/time context into positional features and train-only alarm-degradation associations into an event influence prior with inferred cross-cell context. The current runnable implementation evaluates an sklearn proxy rather than a neural graph backend because PyTorch is unavailable. Experiments cover 1/3/6/12-hour KPI forecasting, weak KPI degradation classification and weak root-cause candidate ranking. Results show strong KPI-history baselines, weak standalone event-only predictors and competitive CET-STGL proxy forecasting on several primary-KPI metrics, while also showing that alarm-token gains are not uniform and weak ranking is not expert root-cause validation.
\end{abstract}

\input{sections/01_introduction}
\input{sections/02_related_work}
\input{sections/03_problem_formulation}
\input{sections/04_methodology}
\input{sections/05_experiments}
\input{sections/06_results}
\input{sections/07_discussion_limitations}
\input{sections/08_conclusion}

% References are intentionally left as TODOs in refs.bib. Add verified BibTeX
% entries before submission; do not use fabricated references.
% \bibliographystyle{IEEEtran}
% \bibliography{refs}

\end{document}
"""


def build_refs_bib() -> str:
    return """% refs.bib
% Verified references must be added manually before submission.
% Do not add placeholder or fabricated bibliography entries.
%
% TODO areas:
% 1. Telecom KPI forecasting.
% 2. Alarm correlation and root-cause analysis in mobile/cellular networks.
% 3. Hawkes processes or point-process event modeling.
% 4. Spatio-temporal graph learning for networked time series.
% 5. Telecom LLM / telecom benchmark papers used only as background contrast.
"""


def table_note() -> str:
    return (
        "Neural baselines are not run because the PyTorch backend is unavailable. "
        "Root-cause ranking is weak-label/proxy consistency, not expert-confirmed root cause. "
        "The event graph is causal-inspired / causal discovery-based, not confirmed causality."
    )


def latex_data_table() -> str:
    rows = "\n".join(
        f"{latex_escape(k)} & {latex_escape(v)} \\\\" for k, v in DATA_STATS.items()
    )
    return rf"""\begin{{table}}[t]
\centering
\caption{{Dataset statistics for the private cellular operations panel.}}
\label{{tab:data_stats}}
\begin{{tabular}}{{ll}}
\toprule
Item & Value \\
\midrule
{rows}
\bottomrule
\end{{tabular}}

\vspace{{0.3em}}
\footnotesize{{Note. The data are a private cellular operations slice from \texttt{{510957.csv}}, not a public benchmark. {table_note()}}}
\end{{table}}
"""


def latex_model_status(df: pd.DataFrame) -> str:
    rows = "\n".join(
        f"{latex_escape(row.model)} & {latex_escape(row.status)} & {latex_escape(row.reason)} \\\\"
        for row in df.itertuples(index=False)
    )
    return rf"""\begin{{table}}[t]
\centering
\caption{{Neural baseline backend status.}}
\label{{tab:model_status}}
\scriptsize
\begin{{tabular}}{{p{{0.15\linewidth}}p{{0.25\linewidth}}p{{0.50\linewidth}}}}
\toprule
Model & Status & Reason \\
\midrule
{rows}
\bottomrule
\end{{tabular}}

\vspace{{0.3em}}
\footnotesize{{Note. No metrics are reported for these neural baselines. {table_note()}}}
\end{{table}}
"""


def latex_long_forecasting(df: pd.DataFrame) -> str:
    cols = [
        "model_label",
        "horizon",
        "primary_mae",
        "primary_rmse",
        "primary_mape_floor1_pct",
        "all_kpi_mae",
        "all_kpi_rmse",
    ]
    rows = []
    for row in df[cols].itertuples(index=False):
        rows.append(
            f"{latex_escape(row.model_label)} & {row.horizon} & {fmt(row.primary_mae)} & {fmt(row.primary_rmse)} & {fmt(row.primary_mape_floor1_pct)} & {fmt(row.all_kpi_mae)} & {fmt(row.all_kpi_rmse)} \\\\"
        )
    return r"""\begingroup
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{longtable}{@{}p{0.26\linewidth}rrrrrr@{}}
\caption{KPI forecasting results over runnable models and horizons.}
\label{tab:forecasting_results}\\
\toprule
Model & H & P-MAE & P-RMSE & P-MAPE & All-MAE & All-RMSE \\
\midrule
\endfirsthead
\toprule
Model & H & P-MAE & P-RMSE & P-MAPE & All-MAE & All-RMSE \\
\midrule
\endhead
""" + "\n".join(rows) + r"""
\bottomrule
\end{longtable}
\endgroup
\noindent\footnotesize{Note. Forecasting labels are real future KPI values. MAPE uses denominator floor $\max(|y|,1.0)$. """ + table_note() + r"""}
"""


def latex_long_classification(df: pd.DataFrame) -> str:
    cols = ["model_label", "horizon", "auprc", "f1", "precision", "recall", "auroc"]
    rows = []
    for row in df[cols].itertuples(index=False):
        rows.append(
            f"{latex_escape(row.model_label)} & {row.horizon} & {fmt(row.auprc)} & {fmt(row.f1)} & {fmt(row.precision)} & {fmt(row.recall)} & {fmt(row.auroc)} \\\\"
        )
    return r"""\begingroup
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{longtable}{@{}p{0.26\linewidth}rrrrrr@{}}
\caption{Weak KPI degradation classification results.}
\label{tab:classification_results}\\
\toprule
Model & H & AUPRC & F1 & Prec. & Recall & AUROC \\
\midrule
\endfirsthead
\toprule
Model & H & AUPRC & F1 & Prec. & Recall & AUROC \\
\midrule
\endhead
""" + "\n".join(rows) + r"""
\bottomrule
\end{longtable}
\endgroup
\noindent\footnotesize{Note. Classification labels are weak KPI-threshold labels fitted from the training split; they are not expert incident labels. """ + table_note() + r"""}
"""


def latex_long_ranking(df: pd.DataFrame) -> str:
    cols = ["model_label", "horizon", "ranking_samples", "hit_at_1", "hit_at_3", "hit_at_5", "mrr", "ndcg_at_5"]
    rows = []
    for row in df[cols].itertuples(index=False):
        samples = "" if pd.isna(row.ranking_samples) else str(int(row.ranking_samples))
        rows.append(
            f"{latex_escape(row.model_label)} & {row.horizon} & {samples} & {fmt(row.hit_at_1)} & {fmt(row.hit_at_3)} & {fmt(row.hit_at_5)} & {fmt(row.mrr)} & {fmt(row.ndcg_at_5)} \\\\"
        )
    return r"""\begingroup
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{longtable}{@{}p{0.24\linewidth}rrrrrrr@{}}
\caption{Weak root-cause candidate ranking consistency.}
\label{tab:weak_ranking_results}\\
\toprule
Model & H & N & Hit@1 & Hit@3 & Hit@5 & MRR & nDCG@5 \\
\midrule
\endfirsthead
\toprule
Model & H & N & Hit@1 & Hit@3 & Hit@5 & MRR & nDCG@5 \\
\midrule
\endhead
""" + "\n".join(rows) + r"""
\bottomrule
\end{longtable}
\endgroup
\noindent\footnotesize{Note. Weak root-cause ranking uses train-only alarm-lift/event-intensity proxy labels; it is not expert-confirmed root-cause localization. Perfect scores indicate weak-label consistency when the scoring rule overlaps with the weak-label construction. """ + table_note() + r"""}
"""


def ablation_summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        ("forecasting", "primary_mae", "delta_primary_mae"),
        ("classification", "auprc", "delta_auprc"),
        ("ranking", "mrr", "delta_mrr"),
    ]
    parts = []
    base_cols = ["model_label", "ablation", "horizon"]
    for task, metric, out in metrics:
        sub = df[(df["task"] == task) & (df["metric"] == metric)][base_cols + ["delta_vs_full"]].copy()
        sub = sub.rename(columns={"delta_vs_full": out})
        parts.append(sub)
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on=base_cols, how="outer")
    return out.sort_values(["ablation", "horizon"])


def latex_ablation_table(df: pd.DataFrame) -> str:
    rows = []
    for row in df.itertuples(index=False):
        rows.append(
            f"{latex_escape(row.model_label)} & {latex_escape(row.ablation)} & {row.horizon} & {fmt(row.delta_primary_mae)} & {fmt(row.delta_auprc)} & {fmt(row.delta_mrr)} \\\\"
        )
    return r"""\begingroup
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{longtable}{@{}p{0.24\linewidth}p{0.19\linewidth}rrrr@{}}
\caption{Ablation deltas relative to CET-STGL-sklearn.}
\label{tab:ablation_delta}\\
\toprule
Ablation row & Removed & H & $\Delta$ P-MAE & $\Delta$ AUPRC & $\Delta$ MRR \\
\midrule
\endfirsthead
\toprule
Ablation row & Removed & H & $\Delta$ P-MAE & $\Delta$ AUPRC & $\Delta$ MRR \\
\midrule
\endhead
""" + "\n".join(rows) + r"""
\bottomrule
\end{longtable}
\endgroup
\noindent\footnotesize{Note. Deltas are computed against CET-STGL-sklearn from existing paper artifacts. Negative MAE deltas are better; positive AUPRC/MRR deltas are better. """ + table_note() + r"""}
"""


def setup_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_fig(fig: plt.Figure, stem: str) -> None:
    for ext in ["png", "pdf", "svg"]:
        path = FIGURES / f"{stem}.{ext}"
        if ext == "png":
            fig.savefig(path, dpi=600, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")


def plot_forecasting(df: pd.DataFrame) -> None:
    setup_matplotlib()
    sub = df[df["model_label"].isin(MODEL_SUBSET)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True)
    for ax, metric, ylabel in [
        (axes[0], "primary_mae", "Primary MAE"),
        (axes[1], "primary_rmse", "Primary RMSE"),
    ]:
        for label in MODEL_SUBSET:
            g = sub[sub["model_label"] == label].sort_values("horizon")
            ax.plot(
                g["horizon"],
                g[metric],
                marker="o",
                linewidth=1.4,
                markersize=3.5,
                color=PALETTE.get(label),
                label=label,
            )
        ax.set_xlabel("Forecast horizon (hours)")
        ax.set_ylabel(ylabel)
        ax.set_xticks([1, 3, 6, 12])
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
    axes[0].set_title("A. Forecasting MAE")
    axes[1].set_title("B. Forecasting RMSE")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.text(0.01, -0.03, "Real future KPI labels. Lower values are better.", fontsize=7)
    save_fig(fig, "Fig1_forecasting_vs_horizon")
    plt.close(fig)


def plot_classification(df: pd.DataFrame) -> None:
    setup_matplotlib()
    sub = df[df["model_label"].isin(MODEL_SUBSET)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True, sharey=False)
    for ax, metric, ylabel in [
        (axes[0], "auprc", "AUPRC"),
        (axes[1], "f1", "F1"),
    ]:
        for label in MODEL_SUBSET:
            g = sub[sub["model_label"] == label].sort_values("horizon")
            ax.plot(
                g["horizon"],
                g[metric],
                marker="o",
                linewidth=1.4,
                markersize=3.5,
                color=PALETTE.get(label),
                label=label,
            )
        ax.set_xlabel("Forecast horizon (hours)")
        ax.set_ylabel(ylabel)
        ax.set_xticks([1, 3, 6, 12])
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
    axes[0].set_title("A. Weak-label AUPRC")
    axes[1].set_title("B. Weak-label F1")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.text(0.01, -0.03, "Weak KPI-threshold labels; not expert incident labels.", fontsize=7)
    save_fig(fig, "Fig2_classification_comparison")
    plt.close(fig)


def plot_ablation(df: pd.DataFrame) -> None:
    setup_matplotlib()
    summary = ablation_summary(df)
    display = (
        summary.groupby(["model_label", "ablation"], as_index=False)[
            ["delta_primary_mae", "delta_auprc", "delta_mrr"]
        ]
        .mean(numeric_only=True)
        .sort_values("ablation")
    )
    labels = [lab.replace("CET w/o ", "w/o ") for lab in display["model_label"].tolist()]
    y = np.arange(len(labels))
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 5.8), sharey=True)
    settings = [
        ("delta_primary_mae", "A. Forecasting delta MAE", "MAE delta"),
        ("delta_auprc", "B. Classification delta AUPRC", "AUPRC delta"),
        ("delta_mrr", "C. Weak ranking delta MRR", "MRR delta"),
    ]
    bar_color = "#6B8EAD"
    for ax, (col, title, ylabel) in zip(axes, settings):
        vals = display[col].astype(float)
        ax.barh(y, vals, color=bar_color, height=0.58)
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel(ylabel)
        ax.grid(axis="x", color="#e5e5e5", linewidth=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        if vals.notna().sum() == 0:
            ax.text(0.02, 0.5, "No ranking samples", transform=ax.transAxes, fontsize=7, va="center")
    fig.subplots_adjust(left=0.34, bottom=0.12, hspace=0.65)
    fig.text(
        0.02,
        0.02,
        "Mean delta across horizons vs CET-STGL-sklearn. Negative is better for MAE; positive is better for AUPRC/MRR. Ranking is weak-label consistency.",
        fontsize=7,
    )
    save_fig(fig, "Fig3_ablation_delta")
    plt.close(fig)


def plot_ranking(df: pd.DataFrame) -> None:
    setup_matplotlib()
    sub = df[df["model_label"].isin(RANKING_SUBSET)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True, sharey=True)
    for ax, metric, ylabel in [
        (axes[0], "hit_at_1", "Hit@1"),
        (axes[1], "mrr", "MRR"),
    ]:
        for label in RANKING_SUBSET:
            g = sub[sub["model_label"] == label].sort_values("horizon")
            if g[metric].notna().sum() == 0:
                continue
            ax.plot(
                g["horizon"],
                g[metric],
                marker="o",
                linewidth=1.4,
                markersize=3.5,
                color=PALETTE.get(label),
                label=label,
            )
        ax.set_xlabel("Forecast horizon (hours)")
        ax.set_ylabel(ylabel)
        ax.set_xticks([1, 3, 6, 12])
        ax.set_ylim(0.75, 1.02)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
    axes[0].set_title("A. Weak ranking Hit@1")
    axes[1].set_title("B. Weak ranking MRR")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.text(0.01, -0.03, "Proxy alarm-lift labels only; not expert-confirmed root cause.", fontsize=7)
    save_fig(fig, "Fig4_weak_ranking_comparison")
    plt.close(fig)


def figure_captions() -> str:
    return """# FIGURE_CAPTIONS

## Fig1. Forecasting vs. Horizon

Primary-KPI forecasting error across 1/3/6/12-hour horizons. MAE and RMSE are computed against real future KPI values from the private `510957.csv` panel. Lower values are better. The figure supports a cautious comparison of runnable sklearn baselines and CET-STGL proxy variants; it does not include unrun neural baselines.

## Fig2. Weak Degradation Classification

Weak KPI degradation classification across horizons. AUPRC and F1 are computed using weak KPI-threshold labels fitted from the training split. These scores should not be interpreted as expert incident-label performance. The comparison shows that alarm/event features do not uniformly improve weak degradation classification in the current sparse-alarm slice.

## Fig3. Ablation Delta

Ablation deltas relative to CET-STGL-sklearn, averaged across horizons for selected metrics. Negative deltas are favorable for forecasting MAE, whereas positive deltas are favorable for AUPRC and MRR. The plot is an ablation diagnostic rather than proof of universal component benefit. Ranking deltas use weak-label/proxy consistency only.

## Fig4. Weak Root-Cause Ranking

Weak root-cause candidate ranking consistency across horizons. Hit@1 and MRR are computed against proxy candidates derived from train-only alarm-to-degradation lift and recent event intensity. These results are not expert-confirmed root-cause localization and should be described as weak-label consistency.
"""


def references_todo() -> str:
    return """# REFERENCES_TODO

This file lists citation needs only. Do not treat any item below as a verified reference. Before submission, manually add real BibTeX entries to `paper_cet_stgl/refs.bib` and cite them in the LaTeX sections.

## Telecom KPI Forecasting

Where needed:

- Introduction: motivate KPI forecasting in cellular operations.
- Related Work: compare statistical, machine-learning and neural sequence methods for KPI/counter prediction.
- Experimental Setup: justify common forecasting metrics and horizons if needed.

Search keywords:

- "cellular network KPI forecasting"
- "mobile network performance prediction"
- "LTE KPI forecasting machine learning"
- "RAN KPI time series forecasting"

## Alarm Correlation / Root Cause Analysis

Where needed:

- Introduction: motivate alarm-driven degradation diagnosis.
- Related Work: position weak root-cause candidate ranking against alarm correlation and RCA.
- Limitations: explain why expert-confirmed root-cause labels are stronger than weak labels.

Search keywords:

- "telecommunication alarm correlation root cause analysis"
- "mobile network alarm correlation"
- "network fault localization alarm data"
- "RAN root cause analysis alarms"

## Hawkes Process for Event Modeling

Where needed:

- Related Work: introduce event intensity and temporal excitation.
- Methodology: justify Hawkes-inspired decay, while stating that the current proxy is not a fitted multi-type Hawkes process.

Search keywords:

- "Hawkes process event sequence modeling"
- "self-exciting point process fault events"
- "temporal point process alarm prediction"

## Spatio-Temporal Graph Learning

Where needed:

- Related Work: compare STGCN/GAT-style graph models for time series.
- Methodology: justify inferred cross-cell graph and spatio-temporal encoder design.
- Limitations: explain that STGCN/GAT were not run because PyTorch is unavailable.

Search keywords:

- "spatio-temporal graph neural network time series forecasting"
- "graph attention network traffic forecasting"
- "STGCN time series forecasting"
- "spatio-temporal graph learning network operations"

## Telecom LLM / Telecom Benchmark as Background Contrast

Where needed:

- Introduction or Related Work only as background contrast.
- Use this literature to explain what this paper is not: it is not a telecom foundation-model paper and not a benchmark paper.

Search keywords:

- "telecom large language model"
- "telecom foundation model"
- "telecom benchmark dataset AI"
- "large language models for network operations"

## Manual BibTeX Checklist

- Add only verified references that have been read or checked.
- Include DOI/arXiv/venue metadata where available.
- Do not add placeholder BibTeX entries.
- Keep claims tied to references: citations should support background and method context, not unsupported superiority claims.
"""


def submission_checklist() -> str:
    return """# SUBMISSION_CHECKLIST

Scope: `MANUSCRIPT_V1.md` and `paper_cet_stgl/` generated from existing CET-STGL artifacts. This checklist is a claim-safety audit, not a new experiment.

| Risk item | Status | Checked locations | Notes / required action |
|---|---|---|---|
| Foundation model misstatement | PASS | `MANUSCRIPT_V1.md`; `paper_cet_stgl/sections/01_introduction.tex`; `08_conclusion.tex` | The manuscript explicitly says this is not a telecom foundation-model paper. Re-check after adding Related Work citations. |
| Benchmark paper misstatement | PASS | `MANUSCRIPT_V1.md`; `paper_cet_stgl/sections/01_introduction.tex`; data table note | The data are described as a private enterprise slice, not a public benchmark. |
| Strict causality misstatement | PASS | Problem Formulation; Methodology; table notes | The graph is described as causal-inspired / causal discovery-based and not confirmed causality. |
| Weak root-cause ranking written as true root cause | PASS | Results; ranking table; Fig4 caption | Ranking is consistently described as weak-label/proxy consistency, not expert-confirmed root-cause localization. |
| Neural baselines implied as run | PASS | Experimental Setup; model status table; Limitations | LSTM, TCN, Informer, Autoformer, STGCN and GAT are marked `not_run_backend_missing` because PyTorch is unavailable. |
| Overclaiming alarm-token improvement | PASS | Abstract; Results; Fig2 caption | The draft explicitly states that alarm-token gains are not uniform and no-alarm variants can be competitive or better. |
| Enterprise/private data limitation missing | PASS | Dataset; Limitations; data table note | The manuscript states that the raw data are private and tied to one project slice. |
| Unverified results added | PASS | All tables/figures | Tables and figures are generated from `paper_artifacts` CSV files only. No new model metrics are introduced. |
| Fabricated references | PASS with TODO | `refs.bib`; `REFERENCES_TODO.md`; Related Work | `refs.bib` contains only comments. Add verified BibTeX before submission. |

## Remaining Before Submission

1. Add verified references and update `refs.bib`.
2. Decide target venue and migrate from generic `article` format only after claims are stable.
3. Add author/affiliation, data availability wording and any enterprise approval language.
4. If neural baselines become available later, run them under the same split and update model-status, tables and claims together.
"""


def write_tables(d: dict[str, pd.DataFrame]) -> None:
    (TABLES / "table_data_statistics.tex").write_text(latex_data_table(), encoding="utf-8")
    (TABLES / "table_model_status.tex").write_text(latex_model_status(d["status"]), encoding="utf-8")
    (TABLES / "table_forecasting_results.tex").write_text(latex_long_forecasting(d["forecasting"]), encoding="utf-8")
    (TABLES / "table_classification_results.tex").write_text(latex_long_classification(d["classification"]), encoding="utf-8")
    (TABLES / "table_weak_ranking_results.tex").write_text(latex_long_ranking(d["ranking"]), encoding="utf-8")
    abl = ablation_summary(d["ablation"])
    (TABLES / "table_ablation_delta.tex").write_text(latex_ablation_table(abl), encoding="utf-8")
    abl.to_csv(TABLES / "table_ablation_delta_source.csv", index=False)


def write_markdown_and_latex(d: dict[str, pd.DataFrame]) -> None:
    manuscript = build_manuscript_v1(d)
    (ROOT / "MANUSCRIPT_V1.md").write_text(manuscript, encoding="utf-8")
    (PAPER / "MANUSCRIPT_V1.md").write_text(manuscript, encoding="utf-8")
    (PAPER / "main.tex").write_text(build_main_tex(), encoding="utf-8")
    for name, text in make_latex_sections(manuscript).items():
        (SECTIONS / name).write_text(text, encoding="utf-8")
    (PAPER / "refs.bib").write_text(build_refs_bib(), encoding="utf-8")

    captions = figure_captions()
    refs = references_todo()
    checklist = submission_checklist()
    for base in [ROOT, PAPER]:
        (base / "FIGURE_CAPTIONS.md").write_text(captions, encoding="utf-8")
        (base / "REFERENCES_TODO.md").write_text(refs, encoding="utf-8")
        (base / "SUBMISSION_CHECKLIST.md").write_text(checklist, encoding="utf-8")


def copy_source_artifacts() -> None:
    source_dir = PAPER / "source_artifacts"
    source_dir.mkdir(exist_ok=True)
    for src in ART.glob("*.csv"):
        shutil.copy2(src, source_dir / src.name)
    for src in ART.glob("*.json"):
        shutil.copy2(src, source_dir / src.name)


def main() -> None:
    ensure_dirs()
    d = read_artifacts()
    write_markdown_and_latex(d)
    write_tables(d)
    plot_forecasting(d["forecasting"])
    plot_classification(d["classification"])
    plot_ablation(d["ablation"])
    plot_ranking(d["ranking"])
    copy_source_artifacts()
    print(f"Wrote paper package to {PAPER}")


if __name__ == "__main__":
    main()
