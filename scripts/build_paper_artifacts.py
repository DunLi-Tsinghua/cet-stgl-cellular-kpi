from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score


ROOT = Path.cwd()
RESULTS = ROOT / "experiments" / "causal_event_token_stgl" / "results"
ARTIFACTS = RESULTS / "paper_artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

PRIMARY_KPIS = [
    "Call Drop Rate(%)",
    "RRC Setup Success Rate(%)",
    "ERAB Setup Success Rate(%)",
    "LTE_User DL Average Throughput(Mbps)",
    "LTE_User UL Average Throughput(Mbps)",
    "LTE_DL PRB Utilizing Rate(%)",
    "LTE_UL PRB Utilizing Rate(%)",
    "CQI AVG",
]

MODEL_LABELS = {
    "ridge_logistic": "Ridge/Logistic",
    "ridge_logistic_no_alarm": "Ridge/Logistic w/o alarms",
    "hawkes_only": "Hawkes-only",
    "event_attention_only": "Event-attention-only",
    "cet_stgl_sklearn": "CET-STGL-sklearn",
    "cet_ablation_no_alarm": "CET w/o alarm tokens",
    "cet_ablation_no_dynamic_graph": "CET w/o dynamic event graph",
    "cet_ablation_no_hawkes_intensity": "CET w/o Hawkes intensity",
    "cet_ablation_no_tokenization": "CET w/o tokenization",
    "cet_ablation_no_cross_cell_graph": "CET w/o cross-cell graph",
}

ABLATION_LABELS = {
    "ridge_logistic": "baseline",
    "ridge_logistic_no_alarm": "baseline_no_alarm",
    "hawkes_only": "event_only_baseline",
    "event_attention_only": "event_only_baseline",
    "cet_stgl_sklearn": "full_proxy",
    "cet_ablation_no_alarm": "no_alarm",
    "cet_ablation_no_dynamic_graph": "no_dynamic_graph",
    "cet_ablation_no_hawkes_intensity": "no_hawkes_intensity",
    "cet_ablation_no_tokenization": "no_tokenization",
    "cet_ablation_no_cross_cell_graph": "no_cross_cell_graph",
}


def model_and_horizon(path: Path, prefix: str) -> tuple[str, int]:
    name = path.stem
    body = name[len(prefix):]
    model, h = body.rsplit("_h", 1)
    return model, int(h)


def mape_floor(y_true: pd.DataFrame, y_pred: pd.DataFrame, floor: float = 1.0) -> float:
    denom = np.maximum(np.abs(y_true.to_numpy(dtype=float)), floor)
    values = np.abs(y_pred.to_numpy(dtype=float) - y_true.to_numpy(dtype=float)) / denom
    return float(np.nanmean(values) * 100.0)


def forecasting_tables(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    per_kpi_rows = []
    metrics_f = metrics[metrics["task"] == "forecasting"].copy()
    metrics_idx = metrics_f.set_index(["model", "horizon"])
    for path in sorted(RESULTS.glob("forecast_*.csv")):
        model, horizon = model_and_horizon(path, "forecast_")
        df = pd.read_csv(path)
        true_cols = [c for c in df.columns if c.startswith("true__")]
        pred_cols = [c for c in df.columns if c.startswith("pred__")]
        kpis = [c[len("true__"):] for c in true_cols]
        y_true = df[true_cols].rename(columns=lambda c: c[len("true__"):])
        y_pred = df[pred_cols].rename(columns=lambda c: c[len("pred__"):])

        primary = [c for c in PRIMARY_KPIS if c in y_true.columns]
        primary_mape = mape_floor(y_true[primary], y_pred[primary]) if primary else np.nan
        all_mape = mape_floor(y_true, y_pred)
        mrow = metrics_idx.loc[(model, horizon)]
        rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS.get(model, model),
                "ablation": ABLATION_LABELS.get(model, "unknown"),
                "horizon": horizon,
                "task": "forecasting",
                "primary_mae": float(np.mean([mrow[f"mae__{c}"] for c in primary if f"mae__{c}" in mrow.index])),
                "primary_rmse": float(np.mean([mrow[f"rmse__{c}"] for c in primary if f"rmse__{c}" in mrow.index])),
                "primary_mape_floor1_pct": primary_mape,
                "all_kpi_mae": float(mrow["mae_mean"]),
                "all_kpi_rmse": float(mrow["rmse_mean"]),
                "all_kpi_mape_floor1_pct": all_mape,
                "runnable_status": "run",
                "note": "MAPE uses denominator floor=1.0 because true KPI values include zeros or near-zeros.",
            }
        )
        for kpi in kpis:
            mae = float(np.mean(np.abs(y_pred[kpi] - y_true[kpi])))
            rmse = float(np.sqrt(np.mean((y_pred[kpi] - y_true[kpi]) ** 2)))
            mape = mape_floor(y_true[[kpi]], y_pred[[kpi]])
            per_kpi_rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS.get(model, model),
                    "ablation": ABLATION_LABELS.get(model, "unknown"),
                    "horizon": horizon,
                    "task": "forecasting",
                    "kpi": kpi,
                    "is_primary_kpi": int(kpi in PRIMARY_KPIS),
                    "mae": mae,
                    "rmse": rmse,
                    "mape_floor1_pct": mape,
                    "runnable_status": "run",
                }
            )
    return pd.DataFrame(rows).sort_values(["horizon", "model"]), pd.DataFrame(per_kpi_rows).sort_values(["horizon", "model", "kpi"])


def classification_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics_c = metrics[metrics["task"] == "classification"].copy().set_index(["model", "horizon"])
    for path in sorted(RESULTS.glob("degradation_*.csv")):
        model, horizon = model_and_horizon(path, "degradation_")
        df = pd.read_csv(path)
        y_true = df["y_true_degradation"].astype(int)
        y_pred = df["y_pred_degradation_0_5"].astype(int)
        mrow = metrics_c.loc[(model, horizon)]
        rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS.get(model, model),
                "ablation": ABLATION_LABELS.get(model, "unknown"),
                "horizon": horizon,
                "task": "classification",
                "auprc": float(mrow["auprc"]),
                "auroc": float(mrow["auroc"]),
                "f1": float(mrow["f1_at_0_5"]),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "positive_rate": float(mrow["positive_rate"]),
                "threshold": 0.5,
                "runnable_status": "run",
                "note": "Degradation labels are weak labels derived from train-only KPI thresholds.",
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon", "model"])


def ranking_table(metrics: pd.DataFrame) -> pd.DataFrame:
    cols = ["model", "horizon", "ranking_samples", "hit_at_1", "hit_at_3", "hit_at_5", "mrr", "ndcg_at_5"]
    df = metrics[metrics["task"] == "ranking"][cols].copy()
    df["model_label"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
    df["ablation"] = df["model"].map(ABLATION_LABELS).fillna("unknown")
    df["task"] = "ranking"
    df["runnable_status"] = "run"
    df["note"] = "Weak root-cause ranking uses train-only alarm-lift weak labels; not expert root-cause truth."
    return df[["model", "model_label", "ablation", "horizon", "task", "ranking_samples", "hit_at_1", "hit_at_3", "hit_at_5", "mrr", "ndcg_at_5", "runnable_status", "note"]].sort_values(["horizon", "model"])


def figure_long(forecast: pd.DataFrame, cls: pd.DataFrame, ranking: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in forecast.iterrows():
        for metric in ["primary_mae", "primary_rmse", "primary_mape_floor1_pct", "all_kpi_mae", "all_kpi_rmse", "all_kpi_mape_floor1_pct"]:
            rows.append(
                {
                    "model": r["model"],
                    "model_label": r["model_label"],
                    "ablation": r["ablation"],
                    "horizon": int(r["horizon"]),
                    "task": "forecasting",
                    "metric": metric,
                    "value": r[metric],
                    "label_type": "real_kpi",
                    "runnable_status": r["runnable_status"],
                }
            )
    for _, r in cls.iterrows():
        for metric in ["auprc", "auroc", "f1", "precision", "recall", "positive_rate"]:
            rows.append(
                {
                    "model": r["model"],
                    "model_label": r["model_label"],
                    "ablation": r["ablation"],
                    "horizon": int(r["horizon"]),
                    "task": "classification",
                    "metric": metric,
                    "value": r[metric],
                    "label_type": "weak_kpi_degradation",
                    "runnable_status": r["runnable_status"],
                }
            )
    for _, r in ranking.iterrows():
        for metric in ["hit_at_1", "hit_at_3", "hit_at_5", "mrr", "ndcg_at_5"]:
            rows.append(
                {
                    "model": r["model"],
                    "model_label": r["model_label"],
                    "ablation": r["ablation"],
                    "horizon": int(r["horizon"]),
                    "task": "ranking",
                    "metric": metric,
                    "value": r[metric],
                    "label_type": "weak_root_cause",
                    "runnable_status": r["runnable_status"],
                }
            )
    return pd.DataFrame(rows)


def ablation_delta(fig: pd.DataFrame) -> pd.DataFrame:
    full = fig[fig["model"] == "cet_stgl_sklearn"][["horizon", "task", "metric", "value"]].rename(columns={"value": "full_value"})
    abl = fig[fig["model"].str.startswith("cet_ablation")].copy()
    out = abl.merge(full, on=["horizon", "task", "metric"], how="left")
    out["delta_vs_full"] = out["value"] - out["full_value"]
    out["relative_delta_vs_full_pct"] = (out["delta_vs_full"] / out["full_value"].replace(0, np.nan)) * 100.0
    return out


def md_table(df: pd.DataFrame, cols: list[str], float_digits: int = 4, max_rows: int | None = None) -> str:
    part = df[cols].copy()
    if max_rows is not None:
        part = part.head(max_rows)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in part.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                vals.append("" if pd.isna(v) else f"{v:.{float_digits}f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def best_rows(df: pd.DataFrame, metric: str, lower_is_better: bool) -> pd.DataFrame:
    rows = []
    for horizon, g in df.groupby("horizon"):
        idx = g[metric].idxmin() if lower_is_better else g[metric].idxmax()
        rows.append(g.loc[idx])
    return pd.DataFrame(rows)


def write_markdowns(forecast: pd.DataFrame, cls: pd.DataFrame, ranking: pd.DataFrame, model_status: pd.DataFrame, fig: pd.DataFrame, delta: pd.DataFrame) -> None:
    primary_cols = ["model_label", "horizon", "primary_mae", "primary_rmse", "primary_mape_floor1_pct", "all_kpi_mae", "all_kpi_rmse"]
    cls_cols = ["model_label", "horizon", "auprc", "f1", "precision", "recall", "auroc"]
    rank_cols = ["model_label", "horizon", "ranking_samples", "hit_at_1", "hit_at_3", "hit_at_5", "mrr", "ndcg_at_5"]
    status_cols = ["model", "status", "reason"]

    text = [
        "# PAPER_TABLES",
        "",
        "All runnable rows are computed from the private `510957.csv` experiment outputs. Classification and ranking use weak labels defined from this dataset. Neural baselines marked `not_run_backend_missing` were not executed because the current runtime lacks PyTorch.",
        "",
        "MAPE is reported as `MAPE_floor1_pct`, using denominator `max(abs(y_true), 1.0)` because several KPI/business columns contain zero or near-zero true values.",
        "",
        "## Table 1. KPI Forecasting",
        "",
        md_table(forecast.sort_values(["horizon", "model"]), primary_cols),
        "## Table 2. KPI Degradation Classification",
        "",
        md_table(cls.sort_values(["horizon", "model"]), cls_cols),
        "## Table 3. Weak Root-Cause Ranking",
        "",
        md_table(ranking.sort_values(["horizon", "model"]), rank_cols),
        "## Table 4. Neural Baseline Status",
        "",
        md_table(model_status, status_cols, float_digits=4),
        "## Best Runnable Rows by Horizon",
        "",
        "Forecasting, lower `primary_mae` is better:",
        "",
        md_table(best_rows(forecast, "primary_mae", True), primary_cols),
        "Classification, higher AUPRC is better:",
        "",
        md_table(best_rows(cls, "auprc", False), cls_cols),
        "Ranking, higher MRR is better; weak-label consistency only:",
        "",
        md_table(best_rows(ranking.fillna({"mrr": -1}), "mrr", False), rank_cols),
        "## Footnotes",
        "",
        "- `CET-STGL-sklearn` is a runnable proxy implementation over the proposed feature blocks, not the final neural CET-STGL graph encoder.",
        "- LSTM, TCN, Informer, Autoformer, STGCN and GAT are listed as non-run neural backends in `model_status.csv`; no metrics are fabricated for them.",
        "- Root-cause ranking uses weak alarm-lift labels and should be described as weak-label consistency, not expert-verified root-cause localization.",
        "- The proposed graph is causal-inspired / causal discovery-based; the observational data do not support strict causal claims.",
    ]
    (ROOT / "PAPER_TABLES.md").write_text("\n".join(text), encoding="utf-8")

    figure_text = [
        "# PAPER_FIGURE_DATA",
        "",
        "Figure-ready files are stored under `experiments/causal_event_token_stgl/results/paper_artifacts/`.",
        "",
        "## Main CSV/JSON Files",
        "",
        "- `paper_figure_long.csv` / `paper_figure_long.json`: long format with `model, model_label, ablation, horizon, task, metric, value, label_type, runnable_status`.",
        "- `paper_ablation_delta.csv` / `paper_ablation_delta.json`: CET ablation deltas relative to `cet_stgl_sklearn`.",
        "- `paper_tables_forecasting.csv`: Model × horizon forecasting metrics.",
        "- `paper_tables_forecasting_per_kpi.csv`: per-KPI forecasting metrics.",
        "- `paper_tables_classification.csv`: weak degradation classification metrics.",
        "- `paper_tables_ranking.csv`: weak root-cause ranking metrics.",
        "",
        "## Recommended Figure Mappings",
        "",
        "1. KPI forecasting vs. horizon: filter `task == forecasting` and `metric in {primary_mae, primary_rmse, primary_mape_floor1_pct}`.",
        "2. Classification AUPRC/F1 vs. model: filter `task == classification` and `metric in {auprc, f1, precision, recall}`.",
        "3. Weak root-cause ranking: filter `task == ranking` and `metric in {hit_at_1, hit_at_3, hit_at_5, mrr, ndcg_at_5}`.",
        "4. Ablation plot: use `paper_ablation_delta.csv`, grouped by `ablation`, `horizon`, and `metric`.",
        "",
        "## Long-Format Preview",
        "",
        md_table(fig.head(24), ["model_label", "ablation", "horizon", "task", "metric", "value", "label_type"], max_rows=24),
        "## Ablation Delta Preview",
        "",
        md_table(delta.head(24), ["model_label", "ablation", "horizon", "task", "metric", "value", "full_value", "delta_vs_full"], max_rows=24),
        "## Interpretation Boundary",
        "",
        "All figure data are computed from real `510957.csv` outputs or weak labels. Do not describe these plots as evidence for telecom foundation-model training or strict causal identification.",
    ]
    (ROOT / "PAPER_FIGURE_DATA.md").write_text("\n".join(figure_text), encoding="utf-8")


def write_analysis(forecast: pd.DataFrame, cls: pd.DataFrame, ranking: pd.DataFrame, model_status: pd.DataFrame) -> None:
    # Compact facts for generated prose.
    f1 = forecast[forecast["horizon"] == 1].set_index("model")
    c1 = cls[cls["horizon"] == 1].set_index("model")
    h12c = cls[cls["horizon"] == 12].set_index("model")
    best_f = best_rows(forecast, "primary_mae", True)
    best_c = best_rows(cls, "auprc", False)

    text = f"""# PAPER_ANALYSIS

## Experimental Setting

All experiments are based on the real `510957.csv` cellular panel in the project workspace. The dataset contains 33 cells, 3,151 hourly timestamps, 48 KPI/business columns and 137 alarm/fault columns, of which 11 alarm columns are active in this slice. Windows crossing the detected long time gap are removed. Forecasting uses real future KPI values; degradation classification uses weak KPI-threshold labels fitted from the training split; root-cause ranking uses weak alarm-lift labels and is not expert-verified.

## Forecasting Results

For the 1-hour horizon, `CET-STGL-sklearn` obtains primary-KPI MAE/RMSE of {f1.loc['cet_stgl_sklearn','primary_mae']:.3f}/{f1.loc['cet_stgl_sklearn','primary_rmse']:.3f}, compared with {f1.loc['ridge_logistic','primary_mae']:.3f}/{f1.loc['ridge_logistic','primary_rmse']:.3f} for the Ridge/Logistic feature baseline. Event-only methods are substantially weaker for forecasting: `Hawkes-only` has 1-hour primary MAE/RMSE of {f1.loc['hawkes_only','primary_mae']:.3f}/{f1.loc['hawkes_only','primary_rmse']:.3f}. This suggests that sparse alarm events alone do not contain enough information to reconstruct continuous KPI trajectories; KPI temporal context remains the dominant signal.

Best forecasting rows by primary MAE are:

{md_table(best_f, ['model_label','horizon','primary_mae','primary_rmse','primary_mape_floor1_pct'])}

The ablation results should be interpreted carefully. In several horizons, removing alarm tokens or using the no-alarm baseline does not degrade forecasting and sometimes improves averaged forecasting error. This is plausible because only about 1.62% of rows contain active alarm events, whereas many forecasted KPI/business values are governed by regular temporal patterns and load dynamics. Therefore, the paper should emphasize event-conditioned analyses and degradation/ranking tasks rather than claiming universal forecasting gains from alarms.

## Degradation Classification

At the 1-hour horizon, `CET-STGL-sklearn` reaches AUPRC/F1 of {c1.loc['cet_stgl_sklearn','auprc']:.3f}/{c1.loc['cet_stgl_sklearn','f1']:.3f}. The Ridge/Logistic baseline obtains AUPRC/F1 of {c1.loc['ridge_logistic','auprc']:.3f}/{c1.loc['ridge_logistic','f1']:.3f}, while the no-alarm Ridge/Logistic variant obtains {c1.loc['ridge_logistic_no_alarm','auprc']:.3f}/{c1.loc['ridge_logistic_no_alarm','f1']:.3f}. Thus, under the current weak degradation labels, alarm tokens do not provide a uniformly significant classification gain. This is an important negative/nuanced result: the weak labels are primarily defined by future KPI thresholds, so lagged KPI dynamics can already be highly predictive.

Best classification rows by AUPRC are:

{md_table(best_c, ['model_label','horizon','auprc','f1','precision','recall','auroc'])}

Hawkes-only and event-attention-only baselines underperform KPI-history baselines in AUPRC and AUROC. Their limitation is structural: active alarms are sparse and many KPI degradation events are weakly labeled from KPI thresholds rather than confirmed operational incidents. Event-only models are therefore useful as root-cause candidate generators and event sensitivity probes, but not sufficient as standalone KPI degradation predictors in this data slice.

## CET-STGL Proxy and Ablation Interpretation

`CET-STGL-sklearn` should be described as a runnable proxy implementation of the proposed feature blocks: KPI patch tokens, alarm event features, train-only alarm-lift event influence features, time/cell encodings and an inferred cross-cell graph. It is not the final neural spatio-temporal graph encoder. The current result supports the feasibility of the data pipeline and ablation framework. It does not yet prove that the full neural CET-STGL architecture outperforms all baselines.

The ablation trends indicate that tokenization and cross-cell context can affect classification and forecasting, but their gains depend on horizon and metric. Removing Hawkes/event intensity has little effect in several rows, implying that the present decayed-intensity signal is either redundant with rolling alarm counts or too sparse to dominate. This motivates a stronger event-influence learning module in the next neural implementation.

## Weak Root-Cause Ranking

Root-cause ranking results are based on weak labels derived from train-only alarm-to-degradation lift and recent event intensity. Some models score highly because they use the same alarm-lift signal used to define the weak labels. These results should be framed as weak-label consistency and sanity checking, not as expert-verified root-cause localization. The correct paper wording is: "the ranking head recovers weak root-cause candidates induced by temporal alarm-degradation association." Avoid the phrase "ground-truth root cause" unless expert annotations are later added.

## Neural Baseline Status

The current runtime does not include PyTorch, so LSTM, TCN, Informer, Autoformer, STGCN and GAT were not executed. They are recorded in `model_status.csv` as `not_run_backend_missing`; no results are fabricated for them. The sklearn baselines and CET-STGL proxy are fully runnable and their outputs are saved under `experiments/causal_event_token_stgl/results/`.

## Limitations for the Paper

1. The dataset is a private project slice, not a public benchmark.
2. Degradation and root-cause labels are weak labels derived from observed KPI/alarm patterns.
3. The event influence graph is causal-inspired / causal discovery-based; the observational setup does not justify strict causal claims.
4. The current CET-STGL result is a sklearn proxy over the proposed representation blocks, not the final neural graph encoder.
5. Alarm sparsity is severe: only 11 of 137 alarm/fault columns are active and only about 1.62% of rows contain active alarms.
6. Do not claim telecom foundation-model training.

## Suggested Experimental Paragraph for the Paper

"We evaluate alarm-driven KPI degradation prediction on a private cellular cell-time panel extracted from `510957.csv`, containing 33 cells, 3,151 hourly timestamps, 48 KPI/business indicators and 137 alarm/fault indicators. Since no expert root-cause annotations are available, degradation labels are weakly derived from training-split KPI thresholds, and root-cause ranking labels are weak candidates induced by lagged alarm-to-degradation lift. Across 1/3/6/12-hour horizons, KPI-history baselines remain strong, while event-only baselines are weaker due to sparse alarms. The CET-STGL proxy validates the proposed representation pipeline and enables ablation over alarm tokens, event intensity, tokenization and inferred cross-cell graph features. The current results should be interpreted as evidence for the feasibility of causal-inspired event-token modeling under weak supervision, not as proof of strict causal discovery or telecom foundation-model training."
"""
    (ROOT / "PAPER_ANALYSIS.md").write_text(text, encoding="utf-8")


def main() -> None:
    metrics = pd.read_csv(RESULTS / "metrics_all.csv")
    model_status = pd.read_csv(RESULTS / "model_status.csv")

    forecast, per_kpi = forecasting_tables(metrics)
    cls = classification_table(metrics)
    ranking = ranking_table(metrics)
    fig = figure_long(forecast, cls, ranking)
    delta = ablation_delta(fig)

    forecast.to_csv(ARTIFACTS / "paper_tables_forecasting.csv", index=False, encoding="utf-8-sig")
    per_kpi.to_csv(ARTIFACTS / "paper_tables_forecasting_per_kpi.csv", index=False, encoding="utf-8-sig")
    cls.to_csv(ARTIFACTS / "paper_tables_classification.csv", index=False, encoding="utf-8-sig")
    ranking.to_csv(ARTIFACTS / "paper_tables_ranking.csv", index=False, encoding="utf-8-sig")
    fig.to_csv(ARTIFACTS / "paper_figure_long.csv", index=False, encoding="utf-8-sig")
    delta.to_csv(ARTIFACTS / "paper_ablation_delta.csv", index=False, encoding="utf-8-sig")

    (ARTIFACTS / "paper_tables.json").write_text(
        json.dumps(
            {
                "forecasting": forecast.to_dict(orient="records"),
                "forecasting_per_kpi": per_kpi.to_dict(orient="records"),
                "classification": cls.to_dict(orient="records"),
                "ranking": ranking.to_dict(orient="records"),
                "model_status": model_status.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (ARTIFACTS / "paper_figure_long.json").write_text(fig.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    (ARTIFACTS / "paper_ablation_delta.json").write_text(delta.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

    write_markdowns(forecast, cls, ranking, model_status, fig, delta)
    write_analysis(forecast, cls, ranking, model_status)

    print("Wrote paper artifacts to", ARTIFACTS)


if __name__ == "__main__":
    main()
