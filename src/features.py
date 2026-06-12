from __future__ import annotations

import numpy as np
import pandas as pd

from data_utils import CELL_COL, TIMESTAMP_COL, safe_numeric_frame
from labels import DegradationThresholds, apply_degradation_label


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = out[TIMESTAMP_COL]
    hour = ts.dt.hour.astype(float)
    dow = ts.dt.dayofweek.astype(float)
    out["_hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["_hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["_dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["_dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    return out


def _valid_temporal_mask(g: pd.DataFrame, horizon: int, max_lag: int) -> pd.Series:
    current = g[TIMESTAMP_COL]
    future = current.shift(-horizon)
    valid_future = future.sub(current).eq(pd.Timedelta(hours=horizon))
    if max_lag <= 0:
        valid_past = pd.Series(True, index=g.index)
    else:
        past = current.shift(max_lag)
        valid_past = current.sub(past).eq(pd.Timedelta(hours=max_lag))
    return valid_future & valid_past


def build_lagged_dataset(
    df: pd.DataFrame,
    kpi_cols: list[str],
    alarm_cols: list[str],
    target_kpis: list[str],
    thresholds: DegradationThresholds,
    horizon: int,
    lookback: int = 24,
    max_feature_kpis: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build a tabular first-pass dataset.

    This is intentionally simple and leak-safe enough for smoke baselines. It
    is also the compatibility layer for future neural data loaders.
    """

    df = add_time_features(df).sort_values([CELL_COL, TIMESTAMP_COL]).copy()
    kpi_feature_cols = kpi_cols[:max_feature_kpis] if max_feature_kpis else kpi_cols
    target_kpis = [c for c in target_kpis if c in df.columns]
    if not target_kpis:
        raise ValueError("No target KPI columns found")

    deg_now = apply_degradation_label(df, thresholds)
    df["_degradation_now"] = deg_now

    lag_steps = sorted(set([0, 1, 2, 3, 6, 12, lookback - 1]))
    lag_steps = [lag for lag in lag_steps if lag >= 0]
    pieces: list[pd.DataFrame] = []

    numeric_kpis = safe_numeric_frame(df, kpi_feature_cols)
    for lag in lag_steps:
        shifted = numeric_kpis.groupby(df[CELL_COL]).shift(lag)
        shifted.columns = [f"kpi_lag{lag}__{c}" for c in shifted.columns]
        pieces.append(shifted)

    active_alarm_cols = [
        c for c in alarm_cols
        if safe_numeric_frame(df, [c])[c].fillna(0).ne(0).any()
    ]
    if active_alarm_cols:
        alarm_num = safe_numeric_frame(df, active_alarm_cols).fillna(0)
        rolling_sum = (
            alarm_num
            .groupby(df[CELL_COL])
            .rolling(lookback, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        rolling_sum.columns = [f"alarm_sum{lookback}__{c}" for c in rolling_sum.columns]
        pieces.append(rolling_sum)
        any_alarm = rolling_sum.sum(axis=1).gt(0).astype(int).rename(f"any_alarm_sum{lookback}")
        pieces.append(any_alarm.to_frame())

    pieces.append(df[["_hour_sin", "_hour_cos", "_dow_sin", "_dow_cos"]])
    cell_codes = pd.Categorical(df[CELL_COL]).codes.astype(float)
    pieces.append(pd.DataFrame({"cell_code": cell_codes}, index=df.index))

    X = pd.concat(pieces, axis=1)
    y_reg = safe_numeric_frame(df, target_kpis).groupby(df[CELL_COL]).shift(-horizon)
    y_cls = df["_degradation_now"].groupby(df[CELL_COL]).shift(-horizon)

    valid_masks = []
    for _, g in df.groupby(CELL_COL, sort=False):
        valid_masks.append(_valid_temporal_mask(g, horizon=horizon, max_lag=max(lag_steps)))
    valid = pd.concat(valid_masks).sort_index()
    valid &= X.notna().all(axis=1)
    valid &= y_reg.notna().all(axis=1)
    valid &= y_cls.notna()

    meta = df.loc[valid, [CELL_COL, TIMESTAMP_COL]].copy()
    return X.loc[valid].reset_index(drop=True), y_reg.loc[valid].reset_index(drop=True), y_cls.loc[valid].astype(int).reset_index(drop=True), meta.reset_index(drop=True)
