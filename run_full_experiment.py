from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

from baselines import (  # noqa: E402
    classification_metrics,
    fit_logistic_classifier,
    fit_ridge_forecaster,
    regression_metrics,
)
from data_utils import (  # noqa: E402
    CELL_COL,
    TIMESTAMP_COL,
    chronological_cutoffs,
    find_csv,
    load_panel,
    split_by_time,
    split_columns,
    summarize_panel,
)
from full_features import (  # noqa: E402
    FeatureSpec,
    add_time_position_features,
    cross_cell_features,
    decayed_alarm_features,
    kpi_patch_token_features,
    kpi_raw_lag_features,
    rolling_alarm_features,
)
from labels import apply_degradation_label, fit_degradation_thresholds  # noqa: E402
from weak_ranking import estimate_alarm_degradation_lift  # noqa: E402


def parse_horizons(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def feature_specs() -> list[FeatureSpec]:
    return [
        FeatureSpec(
            name="ridge_logistic",
            use_raw_kpi_lags=True,
            use_patch_tokens=False,
            use_alarm_raw=True,
            use_alarm_decay=False,
            use_alarm_lift=False,
            use_cross_cell_graph=False,
        ),
        FeatureSpec(
            name="ridge_logistic_no_alarm",
            use_raw_kpi_lags=True,
            use_patch_tokens=False,
            use_alarm_raw=False,
            use_alarm_decay=False,
            use_alarm_lift=False,
            use_cross_cell_graph=False,
        ),
        FeatureSpec(
            name="hawkes_only",
            use_raw_kpi_lags=False,
            use_patch_tokens=False,
            use_alarm_raw=False,
            use_alarm_decay=True,
            use_alarm_lift=False,
            use_cross_cell_graph=False,
        ),
        FeatureSpec(
            name="event_attention_only",
            use_raw_kpi_lags=False,
            use_patch_tokens=False,
            use_alarm_raw=True,
            use_alarm_decay=False,
            use_alarm_lift=True,
            use_cross_cell_graph=False,
        ),
        FeatureSpec(
            name="cet_stgl_sklearn",
            use_raw_kpi_lags=False,
            use_patch_tokens=True,
            use_alarm_raw=True,
            use_alarm_decay=True,
            use_alarm_lift=True,
            use_cross_cell_graph=True,
        ),
        FeatureSpec(
            name="cet_ablation_no_alarm",
            use_raw_kpi_lags=False,
            use_patch_tokens=True,
            use_alarm_raw=False,
            use_alarm_decay=False,
            use_alarm_lift=False,
            use_cross_cell_graph=True,
        ),
        FeatureSpec(
            name="cet_ablation_no_dynamic_graph",
            use_raw_kpi_lags=False,
            use_patch_tokens=True,
            use_alarm_raw=True,
            use_alarm_decay=True,
            use_alarm_lift=False,
            use_cross_cell_graph=True,
        ),
        FeatureSpec(
            name="cet_ablation_no_hawkes_intensity",
            use_raw_kpi_lags=False,
            use_patch_tokens=True,
            use_alarm_raw=True,
            use_alarm_decay=False,
            use_alarm_lift=True,
            use_cross_cell_graph=True,
        ),
        FeatureSpec(
            name="cet_ablation_no_tokenization",
            use_raw_kpi_lags=True,
            use_patch_tokens=False,
            use_alarm_raw=True,
            use_alarm_decay=True,
            use_alarm_lift=True,
            use_cross_cell_graph=True,
        ),
        FeatureSpec(
            name="cet_ablation_no_cross_cell_graph",
            use_raw_kpi_lags=False,
            use_patch_tokens=True,
            use_alarm_raw=True,
            use_alarm_decay=True,
            use_alarm_lift=True,
            use_cross_cell_graph=False,
        ),
    ]


def backend_status_rows() -> list[dict[str, str]]:
    try:
        import torch  # noqa: F401

        torch_status = "available"
        reason = ""
    except Exception:
        torch_status = "not_run_backend_missing"
        reason = "PyTorch is not installed in the current runtime; sklearn proxy and interfaces are runnable."
    return [
        {"model": "lstm", "status": torch_status, "reason": reason},
        {"model": "tcn", "status": torch_status, "reason": reason},
        {"model": "informer", "status": torch_status, "reason": reason},
        {"model": "autoformer", "status": torch_status, "reason": reason},
        {"model": "stgcn", "status": torch_status, "reason": reason},
        {"model": "gat", "status": torch_status, "reason": reason},
    ]


def write_forecast_predictions(
    out_dir: Path,
    model_name: str,
    horizon: int,
    meta: pd.DataFrame,
    y_true: pd.DataFrame,
    y_pred: np.ndarray,
) -> Path:
    pred = pd.DataFrame(y_pred, columns=[f"pred__{c}" for c in y_true.columns])
    truth = y_true.rename(columns={c: f"true__{c}" for c in y_true.columns}).reset_index(drop=True)
    out = pd.concat([meta.reset_index(drop=True), truth, pred], axis=1)
    out.insert(0, "model", model_name)
    out.insert(1, "horizon", horizon)
    path = out_dir / f"forecast_{model_name}_h{horizon}.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def write_degradation_predictions(
    out_dir: Path,
    model_name: str,
    horizon: int,
    meta: pd.DataFrame,
    y_true: pd.Series,
    y_score: np.ndarray,
) -> Path:
    if y_score.ndim == 2:
        y_score = y_score[:, 1]
    out = meta.reset_index(drop=True).copy()
    out.insert(0, "model", model_name)
    out.insert(1, "horizon", horizon)
    out["y_true_degradation"] = y_true.reset_index(drop=True).astype(int)
    out["y_score_degradation"] = y_score
    out["y_pred_degradation_0_5"] = (y_score >= 0.5).astype(int)
    path = out_dir / f"degradation_{model_name}_h{horizon}.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def precompute_feature_blocks(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    kpi_cols: list[str],
    alarm_cols: list[str],
    target_kpis: list[str],
    alarm_lift: pd.Series,
    lookback: int,
    eta_hours: float,
) -> dict[str, pd.DataFrame]:
    print("[features] precomputing raw KPI lags")
    raw_kpi = kpi_raw_lag_features(df, kpi_cols, lookback)
    print("[features] precomputing KPI patch tokens")
    patch = kpi_patch_token_features(df, kpi_cols, patch_len=3, lookback=lookback)
    print("[features] precomputing alarm rolling sums")
    alarm_sum = rolling_alarm_features(df, alarm_cols, lookback)
    print("[features] precomputing alarm decayed intensities")
    alarm_decay = decayed_alarm_features(df, alarm_cols, eta_hours)
    print("[features] precomputing alarm lift graph features")
    lift_features = pd.DataFrame(index=df.index)
    source = alarm_decay if not alarm_decay.empty else alarm_sum
    for alarm in alarm_cols:
        source_cols = [c for c in source.columns if c.endswith(f"__{alarm}")]
        if not source_cols:
            continue
        lift_features[f"alarm_lift__{alarm}"] = source[source_cols[0]] * float(alarm_lift.get(alarm, 0.0))
    if not lift_features.empty:
        lift_features["alarm_lift_total_positive"] = lift_features.clip(lower=0).sum(axis=1)
    print("[features] precomputing cross-cell graph features")
    graph_kpis = target_kpis[: min(8, len(target_kpis))]
    cross_cell = cross_cell_features(df, train_df, graph_kpis, top_k=5)
    print("[features] precomputing time/cell positional features")
    time_cell = add_time_position_features(df)
    return {
        "raw_kpi": raw_kpi,
        "patch": patch,
        "alarm_sum": alarm_sum,
        "alarm_decay": alarm_decay,
        "alarm_lift": lift_features,
        "cross_cell": cross_cell,
        "time_cell": time_cell,
    }


def assemble_features(blocks: dict[str, pd.DataFrame], spec: FeatureSpec) -> pd.DataFrame:
    pieces = []
    if spec.use_raw_kpi_lags:
        pieces.append(blocks["raw_kpi"])
    if spec.use_patch_tokens:
        pieces.append(blocks["patch"])
    if spec.use_alarm_raw:
        pieces.append(blocks["alarm_sum"])
    if spec.use_alarm_decay:
        pieces.append(blocks["alarm_decay"])
    if spec.use_alarm_lift:
        pieces.append(blocks["alarm_lift"])
    if spec.use_cross_cell_graph:
        pieces.append(blocks["cross_cell"])
    if spec.use_time_cell_positional:
        pieces.append(blocks["time_cell"])
    if not pieces:
        return pd.DataFrame(index=blocks["time_cell"].index)
    return pd.concat(pieces, axis=1).replace([np.inf, -np.inf], np.nan)


def valid_temporal_mask(df: pd.DataFrame, horizon: int, lookback: int) -> pd.Series:
    masks = []
    max_lag = lookback - 1
    for _, g in df.groupby(CELL_COL, sort=False):
        current = g[TIMESTAMP_COL]
        future = current.shift(-horizon)
        past = current.shift(max_lag)
        masks.append(
            future.sub(current).eq(pd.Timedelta(hours=horizon))
            & current.sub(past).eq(pd.Timedelta(hours=max_lag))
        )
    return pd.concat(masks).sort_index()


def build_targets(
    df: pd.DataFrame,
    target_kpis: list[str],
    thresholds,
    horizon: int,
    lookback: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    y_reg = df[target_kpis].apply(pd.to_numeric, errors="coerce").groupby(df[CELL_COL]).shift(-horizon)
    y_cls_now = apply_degradation_label(df, thresholds)
    y_cls = y_cls_now.groupby(df[CELL_COL]).shift(-horizon)
    valid = valid_temporal_mask(df, horizon=horizon, lookback=lookback)
    valid &= y_reg.notna().all(axis=1) & y_cls.notna()
    meta = df.loc[valid, [CELL_COL, TIMESTAMP_COL]].copy()
    return (
        y_reg.loc[valid].reset_index(drop=True),
        y_cls.loc[valid].astype(int).reset_index(drop=True),
        meta.reset_index(drop=True),
        valid,
    )


def ranking_scores_for_row(
    row: pd.Series,
    alarm_cols: list[str],
    alarm_lift: pd.Series,
    spec: FeatureSpec,
    lookback: int,
) -> tuple[dict[str, float], dict[str, float]]:
    weak: dict[str, float] = {}
    pred: dict[str, float] = {}
    for alarm in alarm_cols:
        decay_col = f"alarm_decay__{alarm}"
        sum_col = f"alarm_sum{lookback}__{alarm}"
        lift_col = f"alarm_lift__{alarm}"
        base = 0.0
        if decay_col in row.index:
            base = float(row.get(decay_col, 0.0))
        elif sum_col in row.index:
            base = float(row.get(sum_col, 0.0))
        if base <= 0:
            continue
        lift = max(0.0, float(alarm_lift.get(alarm, 0.0)))
        weak_score = base * lift
        if weak_score > 0:
            weak[alarm] = weak_score
        if lift_col in row.index and spec.use_alarm_lift:
            score = max(0.0, float(row.get(lift_col, 0.0)))
        elif decay_col in row.index and spec.use_alarm_decay:
            score = max(0.0, float(row.get(decay_col, 0.0)))
        elif sum_col in row.index and spec.use_alarm_raw:
            score = max(0.0, float(row.get(sum_col, 0.0)))
        else:
            score = 0.0
        if score > 0:
            pred[alarm] = score
    return weak, pred


def ranking_metrics_and_rows(
    model_name: str,
    horizon: int,
    X_test: pd.DataFrame,
    meta_test: pd.DataFrame,
    y_cls_test: pd.Series,
    alarm_cols: list[str],
    alarm_lift: pd.Series,
    spec: FeatureSpec,
    lookback: int,
    max_rows: int | None = None,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    hits1 = []
    hits3 = []
    hits5 = []
    mrr = []
    ndcg5 = []
    rows: list[dict[str, object]] = []
    positives = np.where(y_cls_test.to_numpy(dtype=int) == 1)[0]
    if max_rows:
        positives = positives[:max_rows]
    for idx in positives:
        weak, pred = ranking_scores_for_row(X_test.iloc[idx], alarm_cols, alarm_lift, spec, lookback)
        if not weak or not pred:
            continue
        weak_alarm = max(weak.items(), key=lambda kv: kv[1])[0]
        ordered = sorted(pred.items(), key=lambda kv: kv[1], reverse=True)
        rank = None
        for pos, (alarm, _) in enumerate(ordered, start=1):
            if alarm == weak_alarm:
                rank = pos
                break
        hits1.append(1.0 if rank == 1 else 0.0)
        hits3.append(1.0 if rank is not None and rank <= 3 else 0.0)
        hits5.append(1.0 if rank is not None and rank <= 5 else 0.0)
        mrr.append(1.0 / rank if rank else 0.0)
        ndcg5.append(1.0 / np.log2(rank + 1) if rank is not None and rank <= 5 else 0.0)
        for pred_rank, (alarm, score) in enumerate(ordered[:5], start=1):
            rows.append(
                {
                    "model": model_name,
                    "horizon": horizon,
                    CELL_COL: meta_test.iloc[idx][CELL_COL],
                    TIMESTAMP_COL: meta_test.iloc[idx][TIMESTAMP_COL],
                    "rank": pred_rank,
                    "alarm": alarm,
                    "score": score,
                    "weak_score": weak.get(alarm, 0.0),
                    "weak_top_alarm": weak_alarm,
                    "is_weak_top": int(alarm == weak_alarm),
                }
            )
    if not hits1:
        return {
            "ranking_samples": 0,
            "hit_at_1": np.nan,
            "hit_at_3": np.nan,
            "hit_at_5": np.nan,
            "mrr": np.nan,
            "ndcg_at_5": np.nan,
        }, rows
    return {
        "ranking_samples": float(len(hits1)),
        "hit_at_1": float(np.mean(hits1)),
        "hit_at_3": float(np.mean(hits3)),
        "hit_at_5": float(np.mean(hits5)),
        "mrr": float(np.mean(mrr)),
        "ndcg_at_5": float(np.mean(ndcg5)),
    }, rows


def simple_markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows._\n"
    df = df[columns].copy()
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in columns:
            val = row[col]
            if isinstance(val, float):
                vals.append("" if np.isnan(val) else f"{val:.6f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def write_results_markdown(
    root: Path,
    out_dir: Path,
    metrics: pd.DataFrame,
    model_status: pd.DataFrame,
    audit: dict[str, object],
) -> None:
    forecasting = metrics[metrics["task"] == "forecasting"].copy()
    classification = metrics[metrics["task"] == "classification"].copy()
    ranking = metrics[metrics["task"] == "ranking"].copy()

    md = [
        "# EXPERIMENT_RESULTS",
        "",
        "Paper line: **Causal Event-Token Spatio-Temporal Graph Learning for Alarm-Driven KPI Degradation Prediction in Cellular Networks**.",
        "",
        "All reported runnable metrics are computed from the current private `510957.csv` data or weak labels derived from it. This file does not claim a telecom foundation model and does not claim strict causality.",
        "",
        "## Data Audit Snapshot",
        "",
        simple_markdown_table(pd.DataFrame([audit]), ["rows", "columns", "unique_times", "cells", "kpi_cols", "alarm_cols", "active_alarm_cols", "rows_with_active_alarm_rate"]),
        "## Runnable Forecasting Metrics",
        "",
        simple_markdown_table(
            forecasting.sort_values(["horizon", "model"]),
            ["model", "horizon", "mae_mean", "rmse_mean"],
        ),
        "## Runnable Degradation Classification Metrics",
        "",
        simple_markdown_table(
            classification.sort_values(["horizon", "model"]),
            ["model", "horizon", "positive_rate", "f1_at_0_5", "auroc", "auprc"],
        ),
        "## Weak Root-Cause Ranking Metrics",
        "",
        "Root-cause ranking uses weak labels only. The weak label is the top recent alarm candidate under train-only lagged alarm-to-degradation lift and recent event intensity.",
        "",
        simple_markdown_table(
            ranking.sort_values(["horizon", "model"]),
            ["model", "horizon", "ranking_samples", "hit_at_1", "hit_at_3", "hit_at_5", "mrr", "ndcg_at_5"],
        ),
        "## Neural Baseline Status",
        "",
        simple_markdown_table(model_status, ["model", "status", "reason"]),
        "## Output Files",
        "",
        f"- Metrics CSV: `{out_dir / 'metrics_all.csv'}`",
        f"- Model status CSV: `{out_dir / 'model_status.csv'}`",
        f"- Prediction CSV files: `{out_dir}`",
        "",
    ]
    (root / "EXPERIMENT_RESULTS.md").write_text("\n".join(md), encoding="utf-8")


def write_figures_markdown(root: Path, out_dir: Path) -> None:
    text = f"""# EXPERIMENT_FIGURES

Figure-ready CSV files generated from the current `510957.csv` experiments:

- Forecasting metrics: `{out_dir / 'figure_forecasting_metrics.csv'}`
- Degradation classification metrics: `{out_dir / 'figure_classification_metrics.csv'}`
- Weak root-cause ranking metrics: `{out_dir / 'figure_ranking_metrics.csv'}`
- Ablation summary: `{out_dir / 'figure_ablation_summary.csv'}`

Suggested paper figures:

1. Multi-horizon KPI forecasting: line/bar plot of `mae_mean` and `rmse_mean` by model and horizon.
2. KPI degradation prediction: AUPRC/AUROC by model and horizon.
3. Weak root-cause ranking: Hit@K and MRR by model and horizon.
4. Ablation: compare `cet_stgl_sklearn` against each `cet_ablation_*` variant.

Interpretation boundary: ranking labels are weak labels, not expert-verified root causes.
"""
    (root / "EXPERIMENT_FIGURES.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full runnable CET-STGL sklearn experiments.")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "configs" / "default.json")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--horizons", default="1,3,6,12")
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--ranking-max-rows", type=int, default=None)
    parser.add_argument("--models", default="all", help="Comma-separated model names, or all.")
    args = parser.parse_args()

    project_root = Path.cwd()
    cfg = json.loads(args.config.read_text(encoding="utf-8")) if args.config.exists() else {}
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    lookback = args.lookback or int(cfg.get("lookback", 24))
    horizons = parse_horizons(args.horizons)
    eta_hours = float(cfg.get("alarm_decay_eta_hours", 6.0))

    csv_path = args.csv or find_csv(project_root)
    df = load_panel(csv_path)
    groups = split_columns(df)
    audit = summarize_panel(df, groups)
    train_end, val_end = chronological_cutoffs(df, cfg.get("train_ratio", 0.7), cfg.get("val_ratio", 0.15))
    train_raw_mask, _, _ = split_by_time(df, train_end, val_end)
    train_df = df.loc[train_raw_mask].copy()

    thresholds = fit_degradation_thresholds(train_df)
    train_degradation = apply_degradation_label(train_df, thresholds)
    alarm_lift = estimate_alarm_degradation_lift(train_df, groups.active_alarm_cols, train_degradation, lookback=lookback)

    all_specs = feature_specs()
    if args.models != "all":
        wanted = {x.strip() for x in args.models.split(",") if x.strip()}
        all_specs = [s for s in all_specs if s.name in wanted]
    if not all_specs:
        raise ValueError("No runnable model specs selected")

    target_kpis = groups.kpi_cols
    metrics_rows: list[dict[str, object]] = []
    blocks = precompute_feature_blocks(
        df=df,
        train_df=train_df,
        kpi_cols=groups.kpi_cols,
        alarm_cols=groups.active_alarm_cols,
        target_kpis=target_kpis,
        alarm_lift=alarm_lift,
        lookback=lookback,
        eta_hours=eta_hours,
    )
    metadata = {
        "csv_path": str(csv_path),
        "lookback": lookback,
        "horizons": horizons,
        "train_end": str(train_end),
        "val_end": str(val_end),
        "target_kpis": target_kpis,
        "active_alarm_cols": groups.active_alarm_cols,
        "feature_specs": [asdict(s) for s in all_specs],
        "data_audit": audit,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    for horizon in horizons:
        print(f"\n===== Horizon {horizon}h =====")
        y_reg, y_cls, meta, valid = build_targets(
            df=df,
            target_kpis=target_kpis,
            thresholds=thresholds,
            horizon=horizon,
            lookback=lookback,
        )
        train_mask, val_mask, test_mask = split_by_time(meta, train_end, val_end)
        for spec in all_specs:
            print(f"[{spec.name}] assembling horizon {horizon}")
            X = assemble_features(blocks, spec).loc[valid].reset_index(drop=True)
            X_train, yreg_train, ycls_train = X.loc[train_mask], y_reg.loc[train_mask], y_cls.loc[train_mask]
            X_test, yreg_test, ycls_test, meta_test = X.loc[test_mask], y_reg.loc[test_mask], y_cls.loc[test_mask], meta.loc[test_mask]
            print(f"[{spec.name}] train={len(X_train)} test={len(X_test)} features={X_train.shape[1]}")

            reg = fit_ridge_forecaster(X_train, yreg_train)
            yreg_pred = reg.predict(X_test)
            reg_metrics = regression_metrics(yreg_test, yreg_pred)
            write_forecast_predictions(out_dir, spec.name, horizon, meta_test, yreg_test, yreg_pred)
            metrics_rows.append({"model": spec.name, "horizon": horizon, "task": "forecasting", **reg_metrics})

            clf = fit_logistic_classifier(X_train, ycls_train)
            y_score = clf.predict_proba(X_test)
            cls_metrics = classification_metrics(ycls_test, y_score)
            write_degradation_predictions(out_dir, spec.name, horizon, meta_test, ycls_test, y_score)
            metrics_rows.append({"model": spec.name, "horizon": horizon, "task": "classification", **cls_metrics})

            rank_metrics, rank_rows = ranking_metrics_and_rows(
                model_name=spec.name,
                horizon=horizon,
                X_test=X_test,
                meta_test=meta_test,
                y_cls_test=ycls_test,
                alarm_cols=groups.active_alarm_cols,
                alarm_lift=alarm_lift,
                spec=spec,
                lookback=lookback,
                max_rows=args.ranking_max_rows,
            )
            metrics_rows.append({"model": spec.name, "horizon": horizon, "task": "ranking", **rank_metrics})
            pd.DataFrame(rank_rows).to_csv(out_dir / f"root_cause_ranking_{spec.name}_h{horizon}.csv", index=False, encoding="utf-8-sig")

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(out_dir / "metrics_all.csv", index=False, encoding="utf-8-sig")

    model_status = pd.DataFrame(backend_status_rows())
    model_status.to_csv(out_dir / "model_status.csv", index=False, encoding="utf-8-sig")

    forecasting = metrics[metrics["task"] == "forecasting"].copy()
    classification = metrics[metrics["task"] == "classification"].copy()
    ranking = metrics[metrics["task"] == "ranking"].copy()
    forecasting.to_csv(out_dir / "figure_forecasting_metrics.csv", index=False, encoding="utf-8-sig")
    classification.to_csv(out_dir / "figure_classification_metrics.csv", index=False, encoding="utf-8-sig")
    ranking.to_csv(out_dir / "figure_ranking_metrics.csv", index=False, encoding="utf-8-sig")

    cet = metrics[metrics["model"].str.startswith("cet_")].copy()
    cet.to_csv(out_dir / "figure_ablation_summary.csv", index=False, encoding="utf-8-sig")

    write_results_markdown(project_root, out_dir, metrics, model_status, audit)
    write_figures_markdown(project_root, out_dir)

    print(f"\nWrote results to {out_dir}")


if __name__ == "__main__":
    main()
