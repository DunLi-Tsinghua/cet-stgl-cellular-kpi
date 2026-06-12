from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

from baselines import (  # noqa: E402
    classification_metrics,
    fit_logistic_classifier,
    fit_ridge_forecaster,
    print_metric_block,
    regression_metrics,
)
from data_utils import (  # noqa: E402
    TIMESTAMP_COL,
    chronological_cutoffs,
    find_csv,
    load_panel,
    select_existing,
    split_by_time,
    split_columns,
    summarize_panel,
)
from features import build_lagged_dataset  # noqa: E402
from labels import apply_degradation_label, fit_degradation_thresholds  # noqa: E402
from weak_ranking import estimate_alarm_degradation_lift  # noqa: E402


DEFAULT_TARGETS = [
    "Call Drop Rate(%)",
    "RRC Setup Success Rate(%)",
    "ERAB Setup Success Rate(%)",
    "LTE_User DL Average Throughput(Mbps)",
    "CQI AVG",
]


def parse_horizons(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def maybe_limit(X, y_reg, y_cls, meta, max_samples: int | None):
    if not max_samples or len(X) <= max_samples:
        return X, y_reg, y_cls, meta
    return X.iloc[:max_samples], y_reg.iloc[:max_samples], y_cls.iloc[:max_samples], meta.iloc[:max_samples]


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the alarm-driven KPI degradation pipeline.")
    parser.add_argument("--csv", type=Path, default=None, help="Path to 510957.csv. If omitted, auto-discover under cwd.")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "configs" / "default.json")
    parser.add_argument("--horizons", default="1,3", help="Comma-separated horizons for the smoke run.")
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=20000)
    parser.add_argument("--max-test-samples", type=int, default=8000)
    parser.add_argument("--max-feature-kpis", type=int, default=12, help="Limit KPI features for quick local smoke tests.")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8")) if args.config.exists() else {}
    lookback = args.lookback or int(cfg.get("lookback", 24))
    horizons = parse_horizons(args.horizons)

    csv_path = args.csv or find_csv(Path.cwd())
    print(f"CSV: {csv_path}")
    df = load_panel(csv_path)
    groups = split_columns(df)
    print(json.dumps(summarize_panel(df, groups), ensure_ascii=False, indent=2))

    train_end, val_end = chronological_cutoffs(df, cfg.get("train_ratio", 0.7), cfg.get("val_ratio", 0.15))
    print(f"train_end={train_end} val_end={val_end}")
    train_raw_mask, _, _ = split_by_time(df, train_end, val_end)
    thresholds = fit_degradation_thresholds(df.loc[train_raw_mask])
    train_degradation = apply_degradation_label(df.loc[train_raw_mask], thresholds)

    lifts = estimate_alarm_degradation_lift(
        df.loc[train_raw_mask],
        groups.active_alarm_cols,
        train_degradation,
        lookback=lookback,
    )
    print("\n[Top train-only alarm -> degradation log-lift]")
    print(lifts.head(10).to_string() if not lifts.empty else "No active alarm lift estimated.")

    target_kpis = select_existing(cfg.get("target_kpis", DEFAULT_TARGETS), df)
    if not target_kpis:
        target_kpis = select_existing(DEFAULT_TARGETS, df)
    print(f"\nTargets: {target_kpis}")

    for horizon in horizons:
        print(f"\n===== Horizon {horizon}h =====")
        X, y_reg, y_cls, meta = build_lagged_dataset(
            df=df,
            kpi_cols=groups.kpi_cols,
            alarm_cols=groups.active_alarm_cols,
            target_kpis=target_kpis,
            thresholds=thresholds,
            horizon=horizon,
            lookback=lookback,
            max_feature_kpis=args.max_feature_kpis,
        )
        train_mask, val_mask, test_mask = split_by_time(meta, train_end, val_end)

        X_train, yreg_train, ycls_train, meta_train = maybe_limit(
            X.loc[train_mask], y_reg.loc[train_mask], y_cls.loc[train_mask], meta.loc[train_mask], args.max_train_samples
        )
        X_test, yreg_test, ycls_test, meta_test = maybe_limit(
            X.loc[test_mask], y_reg.loc[test_mask], y_cls.loc[test_mask], meta.loc[test_mask], args.max_test_samples
        )
        print(f"samples train={len(X_train)} test={len(X_test)} features={X_train.shape[1]}")
        print(f"test_time_min={meta_test[TIMESTAMP_COL].min()} test_time_max={meta_test[TIMESTAMP_COL].max()}")

        reg = fit_ridge_forecaster(X_train, yreg_train)
        yreg_pred = reg.predict(X_test)
        print_metric_block("Ridge forecasting", regression_metrics(yreg_test, yreg_pred), max_items=14)

        clf = fit_logistic_classifier(X_train, ycls_train)
        y_score = clf.predict_proba(X_test)
        print_metric_block("Logistic degradation", classification_metrics(ycls_test, y_score))


if __name__ == "__main__":
    main()
