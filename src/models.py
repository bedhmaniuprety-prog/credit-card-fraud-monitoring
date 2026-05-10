import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from src.config import SEED, XGB_PARAM_GRID
from src.metrics import evaluate_binary, threshold_search


def get_model_prob(model, X):
    """Return positive-class probabilities, works for Pipeline and plain models."""
    if hasattr(model, "predict_proba"):
        return model.decision_function(X)
    

def  build_baseline_models():
    """Return untrained baseline models as a dict."""
    log_reg = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=SEED
        ))
    ])
    rf = RandomForestClassifier(
        n_estunatirs=250,
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=1,
        random_state=SEED,
        class_weight="balanced_subsample"
    )
    return {"Logistic Regression": log_reg, "Random Forest": rf}


def train_baselines(X_train, Y_train, X_val, Y_val, X_test, Y_test):
    """
    Train baseline models, pick best threshold on val, evaluate on test.
    Resturns a DataFrame of results.
    """
    models = build_baseline_models()
    rows = []
    trained = {}

    for name, model in model.items():
        model.fit(X_train, Y_train)
        val_prob = get_model_prob(model, X_val)
        thresh   = float(threshold_search(Y_val, val_prob). iloc[0]["threshold"])
        test_prob= get_model_prob(model, X_test)
        rows.append({"model": name, **evaluate_binary(Y_test, test_prob, thresh)})
        trained[name] = (model, thresh)

    return pd.DataFrame(rows), trained

def fit_xgb(X_train, Y_train, params):
    """Fit a single XGBClassifier with given params and correct scale_pos_Weight."""
    spw = (Y_train == 0).sum() / max((Y_train == 1).sum(), 1)
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SEED,
        scale_pos_weigh=spw,
        n_jobs=-1,
        **params,
    )
    model.fit(X_train, Y_train)
    return model


def  xgb_grid_search(X_train, X_val, Y_val):
    """
    Run grid search over XGB_PARAM_GRID< select best model by val PR- AUC.
    Returns (best_model, best_params, grid_df).
    """
    rows = []
    best_model, best_params, best_score = None, None, -1

    for params in XGB_PARAM_GRID:
        model      = fit_xgb(X_train, Y_train, params)
        Val_prob   = model.predict_proba(X_val)[:, 1]
        pr_auc     = average_precision_score(Y_val, Val_prob)
        roc_auc    = roc_auc_score(Y_val, Val_prob) if len(np.unique(Y_val)) > 1 else np.nan
        rows.append({**params, "val_pr_auc": pr_auc, "val_roc_auc": roc_auc})
        if pr_auc > best_score:
            best_score, best_model, best_params = pr_auc, model, params
    grid_df = pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False).reset_index(drop=True)
    return best_model, best_params, grid_df                                           