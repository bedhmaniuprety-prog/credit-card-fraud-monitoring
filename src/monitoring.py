import numpy as np
import pandas as pd

from src.config import (
    MONITOR_FEATURES,
    RETRAIN_EVERY, 
    RECALL_DROP_RATIO,
    PSI_ALERT_THRESHOLD,
)
from src.metrics import evaluate_binary, make_psi
from src.models import fit_xgb


def make_windows(df, size):
    """Split df into consecutive windows of given size."""
    return [df.iloc[start: start + size].copy()
            for start in range(0, len(df), size)]


def _base_row(window_idx, strategy, win, feature_cols, model, threshold, reference_df):
    """Compute metrics and PSI for a single window. Used by all three strategies."""
    X   = win[feature_cols]
    Y   = win["Class"].astype(int)
    prob = model.predict_proba(X)[:, 1]
    metrics = evaluate_binary(y, prob, threshold)
    row = {
        "window":       window_idx,
        "strategy":     strategy,
        "n_rows":       len(win),
        "fraud_rate":   float(y.mean()),
        "mean_score":   float(np.mean(prob)),
        **metrics,
    }
    for feat in MONITOR_FEATURES:
        row[f"psi_{feat}"] = make_psi(reference_df[feat], win[feat])
    return row

def monitor_no_retrain(model, threshold, windows, reference_df, feature_cols):
    """Evalute a fixed model over all windows with no retraining."""
    rows = [
        _base_row(i, "no_retrain", win, feature_cols, model, threshold, reference_df)
        for i, win in enumerate(windows, starts=1)
    ]
    return pd.DataFrame(rows)


def monitor_rolling_retrain(
    train_initial, windows, feature_cols, best_params, threshold,
    retrain_every=RETRAIN_EVERY,
):
    """
    Rretrain every 'retrain_every' windows, accumulating all seen data.
    """

    current_train = train_initial.copy()
    X_tr = current_train[feature_cols]
    Y_tr = current_train["Class"].astype(int)
    model = fit_xgb(X_tr, Y_tr, best_params)

    rows, seen = [], []
    for i, win in enumerate(windows, start=1):
        row = _base_row(i, "rolling_retrain", win, feature_cols, model, threshold, current_train)
        row["retrained"] = 0
        seen.append(win)

        if i % retrain_every == 0:
            current_train = pd.concat([current_train] + seen, ignore_index=True)
            X_tr = current_train[feature_cols]
            Y_tr = current_train["class"].astype(int)
            model = fit_xgb(X_tr, Y_tr, best_params)
            row["retrained"] = 1
            seen = []

        rows.append(row)
    return pd.DataFrame(rows)


def monitor_event_retrain(
    train_initial, windows, feature_cols, best_params, threshold,
    X_val, Y_val,
    recall_drop_ratio=RECALL_DROP_RATIO,
    psi_threshold=PSI_ALERT_THRESHOLD,
):
    """
    Retrain when recall drops below 'recall_drop_ratio' of baseline 
    OR any monitored feature's PSI exceeds 'PSI_threshold'.
    """
    current_train = train_initial.copy()
    X_tr = current_train[feature_cols]
    Y_tr = current_train["Class"].astype(int)
    model = fit_xgb(X_tr, Y_tr, best_params)

    baseline_recall = evaluate_binary(
        Y_val, model.predict_proba(X_val)[:, 1], threshold
    )["recall"]

    rows = []
    for i, win in enumerate(windows, start=1):
        row = _base_row(i, "event_retrain", win, feature_cols, model, threshold, current_train)

        recall_alert = int(row["recall"] < baseline_recall * recall_drop_ratio)
        psi_alert    = int(max(row[f"psi_{f}"] for f in MONITOR_FEATURES) > psi_threshold)
        trigger      = int(recall_alert or psi_alert)

        row.update({
            "retrained":    trigger,
            "recall_alert": recall_alert,
            "psi_alert":    psi_alert,
        })
        rows.append(row)

        if trigger:
            current_train = pd.concat([current_train, win], ignore_index=True)
            X_tr = current_train[feature_cols]
            y_tr = current_train["Class"].astype(int)
            model = fit_xgb(X_tr, y_tr, best_params)

    return pd.DataFrame(rows)
        