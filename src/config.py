from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
DATA_FILE  = DATA_DIR / "creditcard.csv"

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── Split ratios ──────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.85

# ── Monitoring ────────────────────────────────────────────────────────────────
WINDOW_SIZE         = 5000
MONITOR_FEATURES    = ["Amount", "log_amount", "V1", "V2", "V3", "V4"]
RETRAIN_EVERY       = 2
RECALL_DROP_RATIO   = 0.80
PSI_ALERT_THRESHOLD = 0.25

# ── XGBoost grid ──────────────────────────────────────────────────────────────
XGB_PARAM_GRID = [
    {
        "n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1, "gamma": 0,
    },
    {
        "n_estimators": 400, "max_depth": 5, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3, "gamma": 0,
    },
    {
        "n_estimators": 300, "max_depth": 3, "learning_rate": 0.1,
        "subsample": 0.9, "colsample_bytree": 0.8, "min_child_weight": 1, "gamma": 0,
    },
]