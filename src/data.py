import numpy as np
import pandas as pd

from src.config import DATA_FILE, SEED, TRAIN_RATIO, VAL_RATIO


FEATURE_COLS = None # set after load; accessed as data.FEATURE_COLS


def load_and_prepare(path=None):
    """
    Load creditcard.csv, engineer features, and return a time-sorted DataFrame.
    """
    global FEATURE_COLS

    path = path or DATA_FILE
    df = pd.read_csv(path)

    #Temporal ordering - critical for a deployment-realistic split
    df = df.sort_values("Time").reset_index(drop=True)

    # Feature engineering
    df["log_amount"] = np.log1p(df["Amount"])
    df["hour"]       = (df["Time"] / 3600.0) % 24
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24)

    FEATURE_COLS = [c for c in df.columns if c != "Class"]
    return df


def temporal_split(df):
    """
    Split df into train / val / test using temporal cutpoints.
    Returns (train_df, val_df, test_df).
    """
    n         = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * VAL_RATIO)

    train_df = df.iloc[:train_end].copy()
    val_df   = df.iloc[train_end:val_end].copy()
    test_df  = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def get_xy(df, feature_cols):
    """Return feature matrix X and integer label vector y for a split."""
    X = df[feature_cols]
    y = df["Class"].astype(int)
    return X, y