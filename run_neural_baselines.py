from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from scipy.special import expit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

from data_utils import (  # noqa: E402
    CELL_COL,
    TIMESTAMP_COL,
    chronological_cutoffs,
    find_csv,
    load_panel,
    split_by_time,
    split_columns,
)
from full_features import add_time_position_features  # noqa: E402
from labels import apply_degradation_label, fit_degradation_thresholds  # noqa: E402
from neural_sequence import (  # noqa: E402
    apply_sequence_standardizer,
    build_sequence_arrays,
    fit_sequence_standardizer,
    fit_target_standardizer,
    regression_summary,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return hidden[-1]


class TCNEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x.transpose(1, 2))
        return z[:, :, -1]


class SequenceRegressor(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class SequenceClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x)).squeeze(-1)


def make_encoder(name: str, input_dim: int, hidden_dim: int) -> nn.Module:
    if name == "lstm":
        return LSTMEncoder(input_dim, hidden_dim)
    if name == "tcn":
        return TCNEncoder(input_dim, hidden_dim)
    raise ValueError(f"Unsupported neural baseline: {name}")


def train_torch_model(
    model: nn.Module,
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    loss_fn,
    *,
    batch_size: int,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
) -> tuple[nn.Module, dict[str, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    bad_epochs = 0
    completed_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                val_losses.append(float(loss_fn(model(xb), yb).item()))
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        completed_epochs = epoch
        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return model, {"epochs": float(completed_epochs), "best_val_loss": float(best_val)}


def predict(model: nn.Module, X: np.ndarray, batch_size: int) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=batch_size, shuffle=False)
    outputs = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            outputs.append(model(xb).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def classification_summary(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_pred = (y_score >= 0.5).astype(int)
    out = {
        "positive_rate": float(np.mean(y_true)),
        "f1_at_0_5": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision_at_0_5": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_0_5": float(recall_score(y_true, y_pred, zero_division=0)),
        "auprc": float(average_precision_score(y_true, y_score)),
    }
    try:
        out["auroc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        out["auroc"] = float("nan")
    return out


def add_input_context(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    context = add_time_position_features(df)
    out = pd.concat([df.reset_index(drop=True), context.reset_index(drop=True)], axis=1)
    return out, list(context.columns)


def write_forecast_predictions(
    out_dir: Path,
    model_name: str,
    horizon: int,
    meta: pd.DataFrame,
    target_cols: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    truth = pd.DataFrame(y_true, columns=[f"true__{c}" for c in target_cols])
    pred = pd.DataFrame(y_pred, columns=[f"pred__{c}" for c in target_cols])
    out = pd.concat([meta.reset_index(drop=True), truth, pred], axis=1)
    out.insert(0, "model", model_name)
    out.insert(1, "horizon", horizon)
    out.to_csv(out_dir / f"forecast_{model_name}_h{horizon}.csv", index=False, encoding="utf-8-sig")


def write_degradation_predictions(
    out_dir: Path,
    model_name: str,
    horizon: int,
    meta: pd.DataFrame,
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> None:
    out = meta.reset_index(drop=True).copy()
    out.insert(0, "model", model_name)
    out.insert(1, "horizon", horizon)
    out["y_true_degradation"] = y_true.astype(int)
    out["y_score_degradation"] = y_score
    out["y_pred_degradation_0_5"] = (y_score >= 0.5).astype(int)
    out.to_csv(out_dir / f"degradation_{model_name}_h{horizon}.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight LSTM/TCN neural baselines.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "configs" / "default.json")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "results_neural")
    parser.add_argument("--models", default="lstm,tcn")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    lookback = int(cfg.get("lookback", 24))
    horizons = [int(h) for h in cfg.get("horizons", [1, 3, 6, 12])]
    primary_kpis = list(cfg.get("target_kpis", []))
    model_names = [m.strip().lower() for m in args.models.split(",") if m.strip()]

    csv_path = args.csv or find_csv(args.project_root)
    df_raw = load_panel(csv_path)
    groups = split_columns(df_raw)
    df, time_cols = add_input_context(df_raw)
    train_end, val_end = chronological_cutoffs(df, cfg.get("train_ratio", 0.7), cfg.get("val_ratio", 0.15))
    train_row_mask, _, _ = split_by_time(df, train_end, val_end)
    thresholds = fit_degradation_thresholds(df.loc[train_row_mask].copy())
    y_cls_now = apply_degradation_label(df, thresholds)

    target_cols = groups.kpi_cols
    input_cols = groups.kpi_cols + groups.active_alarm_cols + time_cols
    metrics_rows: list[dict[str, object]] = []

    run_info = {
        "csv_path": str(csv_path),
        "lookback": lookback,
        "horizons": horizons,
        "train_end": str(train_end),
        "val_end": str(val_end),
        "target_cols": target_cols,
        "primary_kpis": primary_kpis,
        "input_cols": input_cols,
        "active_alarm_cols": groups.active_alarm_cols,
        "models": model_names,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "torch_version": torch.__version__,
        "note": "Fixed lightweight LSTM/TCN baselines; no hyperparameter search.",
    }
    (args.out_dir / "neural_run_metadata.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for horizon in horizons:
        print(f"\n===== Neural horizon {horizon}h =====")
        X_raw, y_reg_raw, y_cls, meta, valid = build_sequence_arrays(
            df=df,
            input_cols=input_cols,
            reg_target_cols=target_cols,
            cls_target=y_cls_now,
            horizon=horizon,
            lookback=lookback,
        )
        train_mask, val_mask, test_mask = split_by_time(meta, train_end, val_end)
        train_idx = np.flatnonzero(train_mask.to_numpy())
        val_idx = np.flatnonzero(val_mask.to_numpy())
        test_idx = np.flatnonzero(test_mask.to_numpy())

        x_mean, x_std = fit_sequence_standardizer(X_raw[train_idx])
        y_mean, y_std = fit_target_standardizer(y_reg_raw[train_idx])
        X = apply_sequence_standardizer(X_raw, x_mean, x_std)
        y_reg = ((y_reg_raw - y_mean) / y_std).astype(np.float32)

        X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
        yreg_train, yreg_val = y_reg[train_idx], y_reg[val_idx]
        ycls_train, ycls_val, ycls_test = y_cls[train_idx], y_cls[val_idx], y_cls[test_idx]
        yreg_test_raw = y_reg_raw[test_idx]
        meta_test = meta.iloc[test_idx].reset_index(drop=True)

        print(
            f"samples train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
            f"features={X.shape[-1]} targets={len(target_cols)}"
        )

        for model_name in model_names:
            input_dim = X.shape[-1]
            hidden_dim = args.hidden_dim

            reg_model = SequenceRegressor(
                make_encoder(model_name, input_dim, hidden_dim),
                hidden_dim=hidden_dim,
                output_dim=len(target_cols),
            )
            reg_train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(yreg_train))
            reg_val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(yreg_val))
            reg_model, reg_stats = train_torch_model(
                reg_model,
                reg_train_ds,
                reg_val_ds,
                nn.MSELoss(),
                batch_size=args.batch_size,
                epochs=args.epochs,
                patience=args.patience,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            yreg_pred_scaled = predict(reg_model, X_test, args.batch_size)
            yreg_pred = yreg_pred_scaled * y_std + y_mean
            reg_metrics = regression_summary(yreg_test_raw, yreg_pred, target_cols, primary_kpis)
            reg_row = {
                "model": model_name.upper(),
                "horizon": horizon,
                "task": "forecasting",
                "train_samples": len(train_idx),
                "val_samples": len(val_idx),
                "test_samples": len(test_idx),
                **reg_stats,
                **reg_metrics,
            }
            metrics_rows.append(reg_row)
            write_forecast_predictions(args.out_dir, model_name.upper(), horizon, meta_test, target_cols, yreg_test_raw, yreg_pred)
            print(
                f"{model_name.upper()} forecast "
                f"P-MAE={reg_metrics.get('primary_mae_mean', float('nan')):.4f} "
                f"P-RMSE={reg_metrics.get('primary_rmse_mean', float('nan')):.4f}"
            )

            pos = float(ycls_train.sum())
            neg = float(len(ycls_train) - pos)
            pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
            cls_model = SequenceClassifier(make_encoder(model_name, input_dim, hidden_dim), hidden_dim=hidden_dim)
            cls_train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(ycls_train.astype(np.float32)))
            cls_val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(ycls_val.astype(np.float32)))
            cls_model, cls_stats = train_torch_model(
                cls_model,
                cls_train_ds,
                cls_val_ds,
                nn.BCEWithLogitsLoss(pos_weight=pos_weight),
                batch_size=args.batch_size,
                epochs=args.epochs,
                patience=args.patience,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            logits = predict(cls_model, X_test, args.batch_size).reshape(-1)
            scores = expit(logits)
            cls_metrics = classification_summary(ycls_test, scores)
            cls_row = {
                "model": model_name.upper(),
                "horizon": horizon,
                "task": "classification",
                "train_samples": len(train_idx),
                "val_samples": len(val_idx),
                "test_samples": len(test_idx),
                **cls_stats,
                **cls_metrics,
            }
            metrics_rows.append(cls_row)
            write_degradation_predictions(args.out_dir, model_name.upper(), horizon, meta_test, ycls_test, scores)
            print(
                f"{model_name.upper()} cls "
                f"AUPRC={cls_metrics['auprc']:.4f} F1={cls_metrics['f1_at_0_5']:.4f}"
            )

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(args.out_dir / "neural_metrics_all.csv", index=False, encoding="utf-8-sig")
    metrics[metrics["task"] == "forecasting"].to_csv(
        args.out_dir / "neural_forecasting_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics[metrics["task"] == "classification"].to_csv(
        args.out_dir / "neural_classification_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"\nWrote neural baselines to {args.out_dir}")


if __name__ == "__main__":
    main()
