from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TIME_COL = "Time"
CELL_COL = "Cell"
TIMESTAMP_COL = "_timestamp"

ALARM_SIGNATURES = (
    "_RAN_WL_",
    "DEFAULT_RAN_WL",
    "TRANS_RAN_WL",
    "POWER_RAN_WL",
    "OTHER_RAN_WL",
    "S1_RAN_WL",
    "X2_RAN_WL",
)


@dataclass(frozen=True)
class ColumnGroups:
    meta_cols: list[str]
    kpi_cols: list[str]
    alarm_cols: list[str]
    active_alarm_cols: list[str]


def find_csv(root: Path | str = ".") -> Path:
    """Find the canonical-looking 510957.csv copy.

    The workspace contains duplicated copies. Prefer the largest file so that a
    smaller intermediate copy does not silently become the experiment source.
    """

    root = Path(root)
    candidates = list(root.rglob("510957.csv"))
    if not candidates:
        raise FileNotFoundError(f"No 510957.csv found under {root.resolve()}")
    return max(candidates, key=lambda p: (p.stat().st_size, -len(str(p))))


def load_panel(csv_path: Path | str) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    if TIME_COL not in raw.columns or CELL_COL not in raw.columns:
        raise ValueError(f"Expected columns {TIME_COL!r} and {CELL_COL!r}")
    timestamp = pd.to_datetime(raw[TIME_COL].astype(str), format="%Y%m%d%H%M%S").rename(TIMESTAMP_COL)
    df = pd.concat([raw, timestamp], axis=1)
    df = df.sort_values([CELL_COL, TIMESTAMP_COL]).reset_index(drop=True)
    return df


def split_columns(df: pd.DataFrame) -> ColumnGroups:
    meta_cols = [TIME_COL, CELL_COL, TIMESTAMP_COL]
    alarm_cols = [
        c for c in df.columns
        if c not in meta_cols and any(sig in c for sig in ALARM_SIGNATURES)
    ]
    kpi_cols = [c for c in df.columns if c not in meta_cols and c not in alarm_cols]
    active_alarm_cols = [
        c for c in alarm_cols
        if pd.to_numeric(df[c], errors="coerce").fillna(0).ne(0).any()
    ]
    return ColumnGroups(meta_cols, kpi_cols, alarm_cols, active_alarm_cols)


def chronological_cutoffs(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    unique_ts = pd.Series(df[TIMESTAMP_COL].unique()).sort_values().reset_index(drop=True)
    if unique_ts.empty:
        raise ValueError("No timestamps available")
    train_idx = max(0, min(len(unique_ts) - 1, int(len(unique_ts) * train_ratio) - 1))
    val_idx = max(
        train_idx,
        min(len(unique_ts) - 1, int(len(unique_ts) * (train_ratio + val_ratio)) - 1),
    )
    return pd.Timestamp(unique_ts.iloc[train_idx]), pd.Timestamp(unique_ts.iloc[val_idx])


def split_by_time(
    df: pd.DataFrame,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    train_mask = df[TIMESTAMP_COL] <= train_end
    val_mask = (df[TIMESTAMP_COL] > train_end) & (df[TIMESTAMP_COL] <= val_end)
    test_mask = df[TIMESTAMP_COL] > val_end
    return train_mask, val_mask, test_mask


def summarize_panel(df: pd.DataFrame, groups: ColumnGroups) -> dict[str, object]:
    unique_ts = pd.Series(df[TIMESTAMP_COL].unique()).sort_values()
    diffs = unique_ts.diff().value_counts().head(5)
    rows_per_cell = df.groupby(CELL_COL).size()
    any_alarm = df[groups.active_alarm_cols].fillna(0).ne(0).any(axis=1) if groups.active_alarm_cols else pd.Series(False, index=df.index)
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns) - 1),  # exclude parsed timestamp
        "time_min": str(unique_ts.min()),
        "time_max": str(unique_ts.max()),
        "unique_times": int(unique_ts.size),
        "top_time_diffs": {str(k): int(v) for k, v in diffs.items()},
        "cells": int(df[CELL_COL].nunique()),
        "rows_per_cell_min": int(rows_per_cell.min()),
        "rows_per_cell_max": int(rows_per_cell.max()),
        "kpi_cols": int(len(groups.kpi_cols)),
        "alarm_cols": int(len(groups.alarm_cols)),
        "active_alarm_cols": int(len(groups.active_alarm_cols)),
        "rows_with_active_alarm": int(any_alarm.sum()),
        "rows_with_active_alarm_rate": float(any_alarm.mean()),
        "missing_values": int(df.isna().sum().sum()),
    }


def select_existing(columns: Iterable[str], df: pd.DataFrame) -> list[str]:
    return [c for c in columns if c in df.columns]


def infer_cell_graph_by_correlation(
    df: pd.DataFrame,
    target_col: str,
    top_k: int = 5,
) -> pd.DataFrame:
    """Infer a weak cross-cell graph from train-only KPI correlation.

    This is not physical topology. It is a data-derived graph prior for
    ablation and smoke experiments.
    """

    pivot = df.pivot(index=TIMESTAMP_COL, columns=CELL_COL, values=target_col)
    corr = pivot.corr().fillna(0.0)
    rows: list[dict[str, object]] = []
    for src in corr.index:
        neighbors = corr.loc[src].drop(index=src, errors="ignore").abs().sort_values(ascending=False).head(top_k)
        for dst, weight in neighbors.items():
            rows.append({"src_cell": src, "dst_cell": dst, "weight": float(weight)})
    return pd.DataFrame(rows)


def safe_numeric_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
