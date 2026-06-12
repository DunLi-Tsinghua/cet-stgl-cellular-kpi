from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_ridge_forecaster(X_train: pd.DataFrame, y_train: pd.DataFrame):
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(with_mean=False),
        MultiOutputRegressor(Ridge(alpha=1.0)),
    )
    return model.fit(X_train, y_train)


def fit_logistic_classifier(X_train: pd.DataFrame, y_train: pd.Series):
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(with_mean=False),
        SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            max_iter=1000,
            tol=1e-3,
            class_weight="balanced",
            random_state=7,
        ),
    )
    return model.fit(X_train, y_train)


def regression_metrics(y_true: pd.DataFrame, y_pred: np.ndarray) -> dict[str, float]:
    y_pred_df = pd.DataFrame(y_pred, columns=y_true.columns, index=y_true.index)
    metrics: dict[str, float] = {}
    for col in y_true.columns:
        metrics[f"mae__{col}"] = float(mean_absolute_error(y_true[col], y_pred_df[col]))
        metrics[f"rmse__{col}"] = float(np.sqrt(mean_squared_error(y_true[col], y_pred_df[col])))
    metrics["mae_mean"] = float(np.mean([v for k, v in metrics.items() if k.startswith("mae__")]))
    metrics["rmse_mean"] = float(np.mean([v for k, v in metrics.items() if k.startswith("rmse__")]))
    return metrics


def classification_metrics(y_true: pd.Series, y_score: np.ndarray) -> dict[str, float]:
    y_score = np.asarray(y_score)
    if y_score.ndim == 2:
        y_score = y_score[:, 1]
    y_pred = (y_score >= 0.5).astype(int)
    out = {
        "positive_rate": float(np.mean(y_true)),
        "f1_at_0_5": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        out["auroc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        out["auroc"] = float("nan")
    try:
        out["auprc"] = float(average_precision_score(y_true, y_score))
    except ValueError:
        out["auprc"] = float("nan")
    return out


def print_metric_block(title: str, metrics: dict[str, float], max_items: int = 12) -> None:
    print(f"\n[{title}]")
    for i, (key, value) in enumerate(metrics.items()):
        if i >= max_items:
            print(f"... {len(metrics) - max_items} more")
            break
        print(f"{key}: {value:.6f}")
