"""
Section 1: Data Preprocessing
- Outlier analysis via the IQR method (cap values to threshold, no rows dropped
  so the 100k sample size and class balance are preserved).
- Feature scaling is NOT done here: MinMaxScaler is fit inside each CV fold's
  pipeline (see src/models.py) to avoid leaking test-fold statistics into training.
"""
import numpy as np
import pandas as pd


def cap_outliers_iqr(df: pd.DataFrame, feature_cols, k: float = 1.5) -> pd.DataFrame:
    """Cap values outside [Q1 - k*IQR, Q3 + k*IQR] to the threshold (winsorizing)."""
    df = df.copy()
    report = []
    for col in feature_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        n_low = (df[col] < lower).sum()
        n_high = (df[col] > upper).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        report.append({"feature": col, "lower": lower, "upper": upper,
                        "n_capped_low": int(n_low), "n_capped_high": int(n_high)})
    return df, pd.DataFrame(report)


def load_sample(path: str):
    df = pd.read_csv(path)
    label_col = "label"
    feature_cols = [c for c in df.columns if c != label_col]
    return df, label_col, feature_cols


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Report any NaN/missing values per column. Raises if any are found,
    since the HIGGS dataset is expected to be complete and a silent
    imputation would mask a real data issue."""
    counts = df.isna().sum()
    report = counts[counts > 0]
    if not report.empty:
        raise ValueError(f"Missing values found, preprocessing must handle them:\n{report}")
    return pd.DataFrame({"column": df.columns, "n_missing": counts.values})
