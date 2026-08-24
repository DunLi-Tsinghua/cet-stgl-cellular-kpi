from __future__ import annotations

import numpy as np
import pandas as pd

from data_utils import CELL_COL, TIMESTAMP_COL


def make_sequence_mask(df: pd.DataFrame, horizon: int, lookback: int) -> pd.Series:
    """Return rows with contiguous lookback history and exact future horizon."""

    max_lag = lookback - 1
    masks = []
    for _, g in df.groupby(CELL_COL, sort=False):
        current = g[TIMESTAMP_COL]
        future = current.shift(-horizon)
        past = current.shift(max_lag)
        masks.append(
            future.sub(current).eq(pd.Timedelta(hours=horizon))
            & current.sub(past).eq(pd.Timedelta(hours=max_lag))
        )
    if not masks:
        return pd.Series(False, index=df.index)
    return pd.concat(masks).sort_index()


def build_sequence_arrays(
    df: pd.DataFrame,
    input_cols: list[str],
    reg_target_cols: list[str],
    cls_target: pd.Series,
    horizon: int,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.Series]:
    """Build raw sequence tensors without crossing cell boundaries or time gaps."""

    valid = make_sequence_mask(df, horizon=horizon, lookback=lookback)
    numeric_inputs = df[input_cols].apply(pd.to_numeric, errors="coerce")
    numeric_targets = df[reg_target_cols].apply(pd.to_numeric, errors="coerce")
    reg_future = numeric_targets.groupby(df[CELL_COL]).shift(-horizon)
    cls_future = cls_target.groupby(df[CELL_COL]).shift(-horizon)
    valid &= reg_future.notna().all(axis=1) & cls_future.notna()

    sequences: list[np.ndarray] = []
    y_reg: list[np.ndarray] = []
    y_cls: list[int] = []
    rows: list[int] = []
    for _, g in df.groupby(CELL_COL, sort=False):
        positions = list(g.index)
        pos_to_offset = {idx: offset for offset, idx in enumerate(positions)}
        for idx in g.index[valid.loc[g.index]]:
            offset = pos_to_offset[idx]
            window_idx = positions[offset - lookback + 1 : offset + 1]
            sequences.append(numeric_inputs.loc[window_idx].to_numpy(dtype=np.float32))
            y_reg.append(reg_future.loc[idx, reg_target_cols].to_numpy(dtype=np.float32))
            y_cls.append(int(cls_future.loc[idx]))
            rows.append(idx)

    meta = df.loc[rows, [CELL_COL, TIMESTAMP_COL]].reset_index(drop=True)
    return (
        np.stack(sequences).astype(np.float32),
        np.stack(y_reg).astype(np.float32),
        np.asarray(y_cls, dtype=np.float32),
        meta,
        valid,
    )


def fit_sequence_standardizer(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(X_train, axis=(0, 1), keepdims=True).astype(np.float32)
    std = np.nanstd(X_train, axis=(0, 1), keepdims=True).astype(np.float32)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    std = np.where((std > 1e-8) & np.isfinite(std), std, 1.0).astype(np.float32)
    return mean, std


def apply_sequence_standardizer(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    out = (X - mean) / std
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def fit_target_standardizer(y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(y_train, axis=0, keepdims=True).astype(np.float32)
    std = np.nanstd(y_train, axis=0, keepdims=True).astype(np.float32)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    std = np.where((std > 1e-8) & np.isfinite(std), std, 1.0).astype(np.float32)
    return mean, std


def regression_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    columns: list[str],
    primary_columns: list[str],
) -> dict[str, float]:
    abs_err = np.abs(y_true - y_pred)
    sq_err = np.square(y_true - y_pred)
    denom = np.maximum(np.abs(y_true), 1.0)
    ape = abs_err / denom * 100.0

    out: dict[str, float] = {
        "mae_mean": float(np.mean(abs_err)),
        "rmse_mean": float(np.mean(np.sqrt(np.mean(sq_err, axis=0)))),
        "mape_mean": float(np.mean(ape)),
    }
    primary_idx = [columns.index(c) for c in primary_columns if c in columns]
    if primary_idx:
        p_abs = abs_err[:, primary_idx]
        p_sq = sq_err[:, primary_idx]
        p_ape = ape[:, primary_idx]
        out["primary_mae_mean"] = float(np.mean(p_abs))
        out["primary_rmse_mean"] = float(np.mean(np.sqrt(np.mean(p_sq, axis=0))))
        out["primary_mape_mean"] = float(np.mean(p_ape))
    return out
