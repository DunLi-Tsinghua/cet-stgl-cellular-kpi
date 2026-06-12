from __future__ import annotations

import numpy as np
import pandas as pd

from data_utils import CELL_COL, TIMESTAMP_COL, safe_numeric_frame


def estimate_alarm_degradation_lift(
    df: pd.DataFrame,
    alarm_cols: list[str],
    degradation: pd.Series,
    lookback: int = 24,
    eps: float = 1e-6,
) -> pd.Series:
    """Estimate train-only lagged alarm-to-degradation log-lift."""

    base = float(degradation.mean())
    lifts = {}
    active_alarm_cols = [c for c in alarm_cols if df[c].fillna(0).ne(0).any()]
    for col in active_alarm_cols:
        present = (
            safe_numeric_frame(df, [col]).fillna(0)
            .groupby(df[CELL_COL])
            .rolling(lookback, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)[col]
            .gt(0)
        )
        if present.sum() == 0:
            continue
        cond = float(degradation.loc[present.index][present].mean())
        lifts[col] = np.log((cond + eps) / (base + eps))
    return pd.Series(lifts).sort_values(ascending=False)


def recent_alarm_scores(
    row_history: pd.DataFrame,
    alarm_cols: list[str],
    eta_hours: float = 6.0,
) -> pd.Series:
    """Score recent alarms by exponential time decay within one cell history."""

    if row_history.empty:
        return pd.Series(dtype=float)
    end_time = row_history[TIMESTAMP_COL].max()
    ages = (end_time - row_history[TIMESTAMP_COL]).dt.total_seconds() / 3600.0
    decay = np.exp(-ages / eta_hours)
    scores = {}
    alarms = safe_numeric_frame(row_history, alarm_cols).fillna(0)
    for col in alarm_cols:
        value = float((alarms[col] * decay).sum())
        if value > 0:
            scores[col] = value
    return pd.Series(scores).sort_values(ascending=False)
