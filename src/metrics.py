import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix,precision_score,recall_score,f1_score, roc_auc_score,average_precision_score,)
def evaluate_binary(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold) .astype(int)
    out = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   np. nan if len(np.unique(y_true)) < 2 else roc_auc_score(y_true, y_prob),
        "pr_auc":    np.nan if len(np.unique(y_true)) < 2 else average_precision_score(y_true,y_prob),  
}
tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
out.update({"tn": tn, "fn": fn, "tp": tp, "threshold": threshold})
return out
def threshold_search(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    rows = [evaluate_binary(y_true, y_prob, t) for t in thresholds]
    res = pd.DataFrame(rows)
    return res.sort_values(["f1", "recall", "precision"], ascending=False)

def make_psi(reference, current, bins=10):
    breakpoints = np.linspace(
        min(reference.min(), current.min()),
        max(reference.max(), current.max()),
        bins + 1,
    )
    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    cur_counts = np.histogram(current,   bins=breakpoints)[0]
    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)
    # Avoid log(0)
    ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
    cur_pct = np.where(cur_pct == 0, 1e-6, cur_pct)
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)

    