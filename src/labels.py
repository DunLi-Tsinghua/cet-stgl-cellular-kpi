from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_utils import CELL_COL


@dataclass(frozen=True)
class DegradationThresholds:
    global_thresholds: dict[str, float]
    cell_thresholds: dict[object, dict[str, float]]


def fit_degradation_thresholds(train_df: pd.DataFrame) -> DegradationThresholds:
    """Fit weak degradation thresholds on train data only."""

    rules = {}
    if "Call Drop Rate(%)" in train_df:
        rules["call_drop_high"] = ("Call Drop Rate(%)", "high", max(1.0, float(train_df["Call Drop Rate(%)"].quantile(0.99))))
    if "RRC Setup Success Rate(%)" in train_df:
        rules["rrc_low"] = ("RRC Setup Success Rate(%)", "low", min(98.0, float(train_df["RRC Setup Success Rate(%)"].quantile(0.05))))
    if "ERAB Setup Success Rate(%)" in train_df:
        rules["erab_low"] = ("ERAB Setup Success Rate(%)", "low", min(98.0, float(train_df["ERAB Setup Success Rate(%)"].quantile(0.05))))
    if "LTE_User DL Average Throughput(Mbps)" in train_df:
        rules["dl_thr_low"] = (
            "LTE_User DL Average Throughput(Mbps)",
            "low",
            float(train_df["LTE_User DL Average Throughput(Mbps)"].quantile(0.10)),
        )
    if "CQI AVG" in train_df:
        rules["cqi_low"] = ("CQI AVG", "low", float(train_df["CQI AVG"].quantile(0.10)))

    global_thresholds = {name: value for name, (_, _, value) in rules.items()}
    cell_thresholds: dict[object, dict[str, float]] = {}
    for cell, g in train_df.groupby(CELL_COL):
        cell_thresholds[cell] = {}
        for name, (col, direction, fallback) in rules.items():
            if len(g[col].dropna()) < 24:
                cell_thresholds[cell][name] = fallback
                continue
            if name == "call_drop_high":
                value = max(1.0, float(g[col].quantile(0.99)))
            elif direction == "low" and col in ("RRC Setup Success Rate(%)", "ERAB Setup Success Rate(%)"):
                value = min(98.0, float(g[col].quantile(0.05)))
            elif direction == "low":
                value = float(g[col].quantile(0.10))
            else:
                value = fallback
            if not np.isfinite(value):
                value = fallback
            cell_thresholds[cell][name] = value

    return DegradationThresholds(global_thresholds, cell_thresholds)


def apply_degradation_label(
    df: pd.DataFrame,
    thresholds: DegradationThresholds,
    use_cell_thresholds: bool = True,
) -> pd.Series:
    """Return a weak KPI degradation label for each row."""

    y = pd.Series(False, index=df.index)

    def threshold_for(row_cell: object, name: str) -> float | None:
        if use_cell_thresholds and row_cell in thresholds.cell_thresholds:
            return thresholds.cell_thresholds[row_cell].get(name)
        return thresholds.global_thresholds.get(name)

    if "Call Drop Rate(%)" in df and "call_drop_high" in thresholds.global_thresholds:
        vals = df["Call Drop Rate(%)"].astype(float)
        th = df[CELL_COL].map(lambda c: threshold_for(c, "call_drop_high"))
        y |= vals.gt(th.astype(float))
    if "RRC Setup Success Rate(%)" in df and "rrc_low" in thresholds.global_thresholds:
        vals = df["RRC Setup Success Rate(%)"].astype(float)
        th = df[CELL_COL].map(lambda c: threshold_for(c, "rrc_low"))
        y |= vals.lt(th.astype(float))
    if "ERAB Setup Success Rate(%)" in df and "erab_low" in thresholds.global_thresholds:
        vals = df["ERAB Setup Success Rate(%)"].astype(float)
        th = df[CELL_COL].map(lambda c: threshold_for(c, "erab_low"))
        y |= vals.lt(th.astype(float))
    if "LTE_User DL Average Throughput(Mbps)" in df and "dl_thr_low" in thresholds.global_thresholds:
        vals = df["LTE_User DL Average Throughput(Mbps)"].astype(float)
        th = df[CELL_COL].map(lambda c: threshold_for(c, "dl_thr_low"))
        y |= vals.lt(th.astype(float))
    if "CQI AVG" in df and "cqi_low" in thresholds.global_thresholds:
        vals = df["CQI AVG"].astype(float)
        th = df[CELL_COL].map(lambda c: threshold_for(c, "cqi_low"))
        y |= vals.lt(th.astype(float))

    return y.astype(int)
