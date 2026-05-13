# ---- Fraud Detection Pipeline ---------
# Run from the repo root: python notebools/fraud_detection.py

import Warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


warnings.filterwarnings("ignore")

from src.config import OUTPUT_DIR, SEED, WINDOW_SIZE
from src_data import load_and_prepare, temporal_split, get_xy, FEATURE_COLS
from src.metrics import evaluate_binary, threshold_search
from src.models import train_baselines, xgb_grid_search, fit_xgb, get_model_prob
from src.monitoring import make_windows, monitor_no_retrain, monitor_rolling_retrain, monitor_event_retrain
from src.visualization import plot_confusion, plot_roc_pr_curves


np.random.seed(SEED)
OUTPUT_DIR.MKDIR(exist_ok=True)

# --- 1. Load & split ---------
df = loas_and_prepare()
train_df, val_df, test_df = temporal_split(df)

feature_cols = FEATURE_COLS    # set by load_and_prepare()

X_train, Y_train = get_xy(train_df, feature_cols)
X_val,   Y_val   = get_xy(val_df,   feature_cols)
X_test,  Y_test  = get_xy(test_df,  feature_cols)

print("Train:", X_train.shape, "| Val:", X_val.shape, "| Test:", X_test.shape)

# ---- 2. Dataset summary --------
summary = pd.DataFrame({
    "segment":      ["full", "train", "validation", "test"],
    "total":        [len(df), len(train_df), len(val_df), len(test_df)],
    "fraud":        [int(s["class"].sum()) for s in [df, train_df, val_df, test_df]],
})
summary["genuine"]   = summary["total"] - summary["fraud"]
summary["fraud_rate"]= summary["fraud"] / summary["total"]
summary.to.csv(OUTPUT_DIR / "dataset_summary.csv", index=False)
print(summary)

#--- 3. Baseline models --------
baseline_results, trained_baselines = train_baselines(
    X_train, Y_train, X_val, Y_val, X_test, Y_test
)
baseline_results.to_csv(OUTPUT_DIR / "baseline_results.csv", index=False)
print(baseline_results)

# --- 4. XGBooost grid search --------
best_model, best_params, grid_df = xgb_grid_search(X_train, Y_train, X_val, Y_val)
grid_df.to_csv(OUTPUT_DIR / "xgb_grid.results.csv", index=False)

xgb_val_prob    = best_model.predict_proba(X_val)[:, 1]
threshold_df    = threshold_search(Y_val, xgb_val_prob)
best_threshold  = float(threshold_df.iloc[0]["threshold"])
threshold_df.to_csv(OUTPUT_DIR / "xgb_threshold_search.csv", index=False)
print("Best XGB threshold:", best_threshold)

# ---- 5. Final evaluation -------
xgb_test_prob = best_model.predict_proba(X_test)[:, 1]
xgb_result    = pd,DataFrame([{"model": "XGBoost",
                               **evaluate_binary(Y_test, xgb_test_prob, best_threshold)}])
all_results   =pd.concat([baseline_results, xgb_result], ignore_index=True)
all_results.to_csv(OUTPUT_DIR / "all_model_results.csv", index=False)
print(all_results)

#---- 6. Curves & confusion matrix -----------------
log_model, log_thresh = trained_baselines["Logistic Regression"]
rf_model, rf_thresh = trained_baselines["Random Forest"]


curve_data = [
    ("Logistic Regression", Y_test, get_model_prob(log_model, X_test)),
    ("Random Forest",       Y_test, get_model_prob(rf_model,  X_test)),
    ("XGBooost",            Y_test, xgb_test_prob),
]
plot_roc_pr_curves(curve_data)

xgb_pred = (xgb_test_prob >= best_threshold) .astype(int)
cm = confusion_matrix(Y_test, xgb_pred)
plot_confusion(cm, "XGBoost Confusion Matrix",
               save_path=OUTPUT_DIR / "xgb_confusion_matrix.png")
print(classification_report(Y_test, xgb_pred, digits=4))

#---- 7. Monitoring------
test_windows = make_windows(test_df, WINDOW_SIZE)

no_retrain_df = monitor_no_retrain(
    best_model, best_threshold, test_windows, train_df, feature_cols
)
no_retrain_df.to_csv(OUTPUT_DIR / "monitor_no_retrain.csv", index=False)

rolling_df = monitor_rolling_retrain(
    train_df, test_windows, feature_cols, best_params, best_threshold
)
rolling_df.to_csv(OUTPUT_DIR / "monitor_rolling_retrain.csv", index=False)

event_df = monitor_event_retrain(
    train_df, test_windows, feature_cols, best_params, best_threshold, 
    X_val, Y_val
)
event_df.to_csv(OUTPUT_DIR / "monitor_event_retrain.csv", index=False)

# --- 8. Strategy comparison ----------
compare_df = pd.concat([no_retrain_df, rolling_df, event_df], ignore_index=True)
summary_compare = compare_df.groupby("strategy").agg(
    mean_precision=("precision", "mean"),
    mean_recall=("recall", "mean"),
    mean_f1=("f1", "mean"),
    mean_pr_auc=("pr_auc", "mean"),
    min_recall=("recall", "min"),
    max_psi_amount=("psi_Amount", "sum"),
).reset_index()
summary_compare.to_csv(OUTPUT_DIR / "strategy_comparision.csv", index=False)
print(summary_compare)

# ---- 9. Monitoring plots ---------
fig, ax = plt.sublots(figsize=(10, 4))
for col, label in [("recall", "Recall"), ("f1", "F1"), ("precision", "Precision")]:
    ax.plot(no_retrain_df["window"], no_retrain_df[col], marker="0", label=label)
ax.set(xlabel="Window", ylabel="Metric", title="Monitoring metrics - no retraining")
ax. legens()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "monitoring_metrics_no_retrain.png", dpi=200)
plt.show()

fig, ax = plt.sublots(figsize=(10, 4))
for feat in ["psi_Amount", "psi_log_amount", "psi_V1", "psi_V2"]:
    ax.plot(no_retrain_df["window"], no_retrain_df[feat], marker="o", label=feat)
ax.axhline(0.10, linestyle="--", label="PSI 0.10")
ax.axhline(0.25, linestyle="--", label="PSI 0.25")
ax.set(xlabel="Window", ylabel="PSI", title="PSI drift - no retraining")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "monitor_psi_no_retrain.png", dpi=200)
plt.show()  

fig, ax = plt.subplots(figsize=(10, 4))
for name, part in compare_df.groupby("strategy"):
    ax.plot(part["window"], part["recall"], marker="o", label=name)
ax.set(xlabel="Window", ylabel="Recall", title="Recall by monitoring strategy")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "strategy_recall_comparison.png", dpi=200)
plt.show()

fig, ax = plt.subplots(figsize=(10, 4))
for name, part in compare_df.groupby("strategy"):
    ax.plot(part["window"], part["f1"], marker="o", label=name)
ax.set(xlabel="Window", ylabel="F1-score", title="F1-score by monitoring strategy")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "strategy_f1_comparison.png", dpi=200)
plt.show()

# --- 10. Thesis table ------------
thesis_table = all_results[["model", "precision", "recall", "f1", "roc_auc", "pr_auc"]].copy()
thesis_table.columns = ["Model", "Precision", "Recall", "F1-score", "ROC-AUC", "PR-AUC"]
thesis_table.to_csv(OUTPUT_DIR / "thesis_table_model_performance.csv", index=False)
print("\nSaved outputs:")
for p in sorted(OUTPUT_DIR.glob("*")):
    print(" -", p.name)