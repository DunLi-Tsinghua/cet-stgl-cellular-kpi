from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_utils import CELL_COL, TIMESTAMP_COL, safe_numeric_frame
from labels import DegradationThresholds, apply_degradation_label


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    use_raw_kpi_lags: bool = True
    use_patch_tokens: bool = False
    use_alarm_raw: bool = True
    use_alarm_decay: bool = False
    use_alarm_lift: bool = False
    use_cross_cell_graph: bool = False
    use_time_cell_positional: bool = True
    patch_len: int = 3
    cross_cell_top_k: int = 5


def add_time_position_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df[TIMESTAMP_COL]
    hour = ts.dt.hour.astype(float)
    dow = ts.dt.dayofweek.astype(float)
    return pd.DataFrame(
        {
            "time_hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "time_hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "time_dow_sin": np.sin(2 * np.pi * dow / 7.0),
            "time_dow_cos": np.cos(2 * np.pi * dow / 7.0),
            "cell_code": pd.Categorical(df[CELL_COL]).codes.astype(float),
        },
        index=df.index,
    )


def decayed_alarm_features(
    df: pd.DataFrame,
    alarm_cols: list[str],
    eta_hours: float,
) -> pd.DataFrame:
    if not alarm_cols:
        return pd.DataFrame(index=df.index)
    decay = float(np.exp(-1.0 / eta_hours))
    out = pd.DataFrame(index=df.index)
    alarm_num = safe_numeric_frame(df, alarm_cols).fillna(0.0)
    for cell, idx in df.groupby(CELL_COL, sort=False).groups.items():
        g = df.loc[idx].sort_values(TIMESTAMP_COL)
        values = alarm_num.loc[g.index, alarm_cols].to_numpy(dtype=float)
        state = np.zeros(values.shape[1], dtype=float)
        prev_ts = None
        cell_result = np.zeros_like(values, dtype=float)
        for i, (_, row) in enumerate(g.iterrows()):
            ts = row[TIMESTAMP_COL]
            if prev_ts is not None and ts - prev_ts != pd.Timedelta(hours=1):
                state = np.zeros(values.shape[1], dtype=float)
            state = values[i] + decay * state
            cell_result[i] = state
            prev_ts = ts
        out.loc[g.index, [f"alarm_decay__{c}" for c in alarm_cols]] = cell_result
    return out.fillna(0.0)


def rolling_alarm_features(
    df: pd.DataFrame,
    alarm_cols: list[str],
    lookback: int,
) -> pd.DataFrame:
    if not alarm_cols:
        return pd.DataFrame(index=df.index)
    alarm_num = safe_numeric_frame(df, alarm_cols).fillna(0.0)
    roll = (
        alarm_num.groupby(df[CELL_COL])
        .rolling(lookback, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    roll.columns = [f"alarm_sum{lookback}__{c}" for c in roll.columns]
    return roll


def kpi_raw_lag_features(df: pd.DataFrame, kpi_cols: list[str], lookback: int) -> pd.DataFrame:
    lag_steps = sorted(set([0, 1, 2, 3, 6, 12, lookback - 1]))
    kpi_num = safe_numeric_frame(df, kpi_cols)
    pieces = []
    for lag in lag_steps:
        shifted = kpi_num.groupby(df[CELL_COL]).shift(lag)
        shifted.columns = [f"kpi_lag{lag}__{c}" for c in shifted.columns]
        pieces.append(shifted)
    return pd.concat(pieces, axis=1)


def kpi_patch_token_features(
    df: pd.DataFrame,
    kpi_cols: list[str],
    patch_len: int,
    lookback: int,
) -> pd.DataFrame:
    kpi_num = safe_numeric_frame(df, kpi_cols)
    starts = [0, patch_len, 2 * patch_len, 4 * patch_len, max(0, lookback - patch_len)]
    starts = sorted(set(s for s in starts if s + patch_len - 1 < lookback))
    pieces = []
    for start in starts:
        shifted_values = [
            kpi_num.groupby(df[CELL_COL]).shift(start + offset)
            for offset in range(patch_len)
        ]
        stack = np.stack([x.to_numpy(dtype=float) for x in shifted_values], axis=2)
        mean = pd.DataFrame(np.nanmean(stack, axis=2), index=df.index, columns=[f"kpi_patch{start}_mean__{c}" for c in kpi_cols])
        last = shifted_values[0].to_numpy(dtype=float)
        first = shifted_values[-1].to_numpy(dtype=float)
        slope = pd.DataFrame((last - first) / max(1, patch_len - 1), index=df.index, columns=[f"kpi_patch{start}_slope__{c}" for c in kpi_cols])
        pieces.extend([mean, slope])
    return pd.concat(pieces, axis=1) if pieces else pd.DataFrame(index=df.index)


def infer_correlation_neighbors(
    train_df: pd.DataFrame,
    target_col: str,
    top_k: int,
) -> dict[object, list[tuple[object, float]]]:
    pivot = train_df.pivot(index=TIMESTAMP_COL, columns=CELL_COL, values=target_col)
    corr = pivot.corr().fillna(0.0)
    neighbors: dict[object, list[tuple[object, float]]] = {}
    for cell in corr.index:
        s = corr.loc[cell].drop(index=cell, errors="ignore").abs().sort_values(ascending=False).head(top_k)
        neighbors[cell] = [(idx, float(val)) for idx, val in s.items() if val > 0]
    return neighbors


def cross_cell_features(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    kpi_cols: list[str],
    top_k: int,
) -> pd.DataFrame:
    if not kpi_cols:
        return pd.DataFrame(index=df.index)
    target_col = "Call Drop Rate(%)" if "Call Drop Rate(%)" in train_df.columns else kpi_cols[0]
    neighbors = infer_correlation_neighbors(train_df, target_col=target_col, top_k=top_k)
    out = pd.DataFrame(index=df.index)
    row_index = pd.MultiIndex.from_frame(df[[TIMESTAMP_COL, CELL_COL]])
    for kpi in kpi_cols:
        pivot = df.pivot(index=TIMESTAMP_COL, columns=CELL_COL, values=kpi)
        result = pd.DataFrame(index=pivot.index, columns=pivot.columns, dtype=float)
        for cell, neigh in neighbors.items():
            if not neigh or cell not in result.columns:
                continue
            nb_cells = [nb for nb, _ in neigh if nb in pivot.columns]
            weights = np.array([w for nb, w in neigh if nb in pivot.columns], dtype=float)
            if not nb_cells or weights.sum() <= 0:
                continue
            result[cell] = pivot[nb_cells].to_numpy(dtype=float).dot(weights) / weights.sum()
        stacked = result.stack()
        out[f"cross_cell_corr_mean__{kpi}"] = row_index.map(stacked)
    return out


def _valid_temporal_mask(g: pd.DataFrame, horizon: int, max_lag: int) -> pd.Series:
    current = g[TIMESTAMP_COL]
    future = current.shift(-horizon)
    past = current.shift(max_lag)
    return future.sub(current).eq(pd.Timedelta(hours=horizon)) & current.sub(past).eq(pd.Timedelta(hours=max_lag))


def build_supervised_features(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    kpi_cols: list[str],
    alarm_cols: list[str],
    target_kpis: list[str],
    thresholds: DegradationThresholds,
    alarm_lift: pd.Series,
    spec: FeatureSpec,
    horizon: int,
    lookback: int,
    eta_hours: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    pieces: list[pd.DataFrame] = []

    if spec.use_raw_kpi_lags:
        pieces.append(kpi_raw_lag_features(df, kpi_cols, lookback))
    if spec.use_patch_tokens:
        pieces.append(kpi_patch_token_features(df, kpi_cols, spec.patch_len, lookback))
    if spec.use_alarm_raw:
        pieces.append(rolling_alarm_features(df, alarm_cols, lookback))
    if spec.use_alarm_decay:
        decay = decayed_alarm_features(df, alarm_cols, eta_hours)
        pieces.append(decay)
    else:
        decay = pd.DataFrame(index=df.index)
    if spec.use_alarm_lift and alarm_cols:
        source = decay if not decay.empty else rolling_alarm_features(df, alarm_cols, lookback)
        lift_features = pd.DataFrame(index=df.index)
        for col in alarm_cols:
            lift = float(alarm_lift.get(col, 0.0))
            source_cols = [c for c in source.columns if c.endswith(f"__{col}")]
            if source_cols:
                lift_features[f"alarm_lift__{col}"] = source[source_cols[0]] * lift
        if not lift_features.empty:
            lift_features["alarm_lift_total_positive"] = lift_features.clip(lower=0).sum(axis=1)
            pieces.append(lift_features)
    if spec.use_cross_cell_graph:
        graph_kpis = target_kpis[: min(8, len(target_kpis))]
        pieces.append(cross_cell_features(df, train_df, graph_kpis, spec.cross_cell_top_k))
    if spec.use_time_cell_positional:
        pieces.append(add_time_position_features(df))

    X = pd.concat(pieces, axis=1) if pieces else pd.DataFrame(index=df.index)
    X = X.replace([np.inf, -np.inf], np.nan)

    y_reg = safe_numeric_frame(df, target_kpis).groupby(df[CELL_COL]).shift(-horizon)
    y_cls_now = apply_degradation_label(df, thresholds)
    y_cls = y_cls_now.groupby(df[CELL_COL]).shift(-horizon)

    max_lag = lookback - 1
    valid = pd.Series(False, index=df.index)
    masks = []
    for _, g in df.groupby(CELL_COL, sort=False):
        masks.append(_valid_temporal_mask(g, horizon=horizon, max_lag=max_lag))
    if masks:
        valid = pd.concat(masks).sort_index()
    valid &= y_reg.notna().all(axis=1) & y_cls.notna()

    # Do not require every feature to be non-null; the model pipeline imputes.
    meta = df.loc[valid, [CELL_COL, TIMESTAMP_COL]].copy()
    return (
        X.loc[valid].reset_index(drop=True),
        y_reg.loc[valid].reset_index(drop=True),
        y_cls.loc[valid].astype(int).reset_index(drop=True),
        meta.reset_index(drop=True),
    )
