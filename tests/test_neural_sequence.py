from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_sequence import make_sequence_mask, regression_summary  # noqa: E402


def test_sequence_mask_rejects_windows_crossing_time_gap() -> None:
    df = pd.DataFrame(
        {
                "Cell": ["a"] * 7,
            "_timestamp": pd.to_datetime(
                [
                    "2024-01-01 00:00:00",
                    "2024-01-01 01:00:00",
                    "2024-01-01 02:00:00",
                    "2024-01-03 00:00:00",
                    "2024-01-03 01:00:00",
                    "2024-01-03 02:00:00",
                    "2024-01-03 03:00:00",
                ]
            ),
        }
    )

    mask = make_sequence_mask(df, horizon=1, lookback=3)

    assert mask.tolist() == [False, False, False, False, False, True, False]


def test_sequence_mask_accepts_exact_lookback_and_future_horizon() -> None:
    df = pd.DataFrame(
        {
            "Cell": ["a"] * 5,
            "_timestamp": pd.date_range("2024-01-01 00:00:00", periods=5, freq="h"),
        }
    )

    mask = make_sequence_mask(df, horizon=2, lookback=3)

    assert mask.tolist() == [False, False, True, False, False]


def test_regression_summary_uses_unit_floor_for_mape() -> None:
    summary = regression_summary(
        y_true=pd.DataFrame({"kpi": [0.0, 2.0]}).to_numpy(dtype="float32"),
        y_pred=pd.DataFrame({"kpi": [1.0, 4.0]}).to_numpy(dtype="float32"),
        columns=["kpi"],
        primary_columns=["kpi"],
    )

    assert summary["mape_mean"] == 100.0
    assert summary["primary_mape_mean"] == 100.0
