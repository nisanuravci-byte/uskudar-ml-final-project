"""
End-to-end pipeline for the ML final project:
  1. Load 100k HIGGS sample
  2. Outlier capping (IQR) + (scaling happens inside each CV fold's pipeline)
  3. Filter-based feature selection (ANOVA F-score, top 15) -- inside pipeline
  4. Nested CV (outer 5-fold / inner 3-fold) for KNN, SVM, MLP, XGBoost
  5. Metrics table + ROC curves + selected-features report -> results/
"""
import os
import pandas as pd

from src.preprocessing import load_sample, cap_outliers_iqr, check_missing_values
from src.models import get_model_grid
from src.nested_cv import run_nested_cv
from src.evaluate import build_metrics_table, plot_roc_curves, selected_features_report

DATA_PATH = os.path.join("data", "higgs_sample_100k.csv")
RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")


def _readable(value):
    """Render sklearn score_func callables (and functools.partial wrappers) as
    plain strings so they serialize legibly into CSV."""
    if callable(value):
        return getattr(value, "func", value).__name__
    return value


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Loading data...")
    df, label_col, feature_cols = load_sample(DATA_PATH)
    print(f"  shape={df.shape}, class balance:\n{df[label_col].value_counts(normalize=True)}")

    print("Checking for missing values...")
    missing_report = check_missing_values(df)
    missing_report.to_csv(os.path.join(RESULTS_DIR, "missing_value_report.csv"), index=False)

    print("Capping outliers (IQR method)...")
    df, outlier_report = cap_outliers_iqr(df, feature_cols)
    outlier_report.to_csv(os.path.join(RESULTS_DIR, "outlier_report.csv"), index=False)

    X = df[feature_cols].values
    y = df[label_col].values.astype(int)

    print("Running nested CV (this will take a while)...", flush=True)
    model_grids = get_model_grid()
    # SVM (libsvm) scales poorly with sample size; cap its training fold so
    # grid search stays tractable. Other models use the full ~80k-row fold.
    train_caps = {"SVM": 15000}
    results = run_nested_cv(
        X, y, model_grids, outer_splits=5, inner_splits=3, seed=42, train_caps=train_caps
    )

    print("Building metrics table...")
    per_fold_df, summary_df = build_metrics_table(results)
    per_fold_df.to_csv(os.path.join(RESULTS_DIR, "per_fold_metrics.csv"), index=False)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "metrics_summary.csv"))
    print(summary_df)

    print("Plotting ROC curves...")
    plot_roc_curves(results, os.path.join(PLOTS_DIR, "roc_curves.png"))

    print("Reporting selected features per fold...")
    feat_df = selected_features_report(results, feature_cols)
    feat_df.to_csv(os.path.join(RESULTS_DIR, "selected_features.csv"), index=False)

    best_params_rows = []
    for model_name, fold_records in results.items():
        for r in fold_records:
            readable_params = {k: _readable(v) for k, v in r["best_params"].items()}
            best_params_rows.append({"model": model_name, "fold": r["fold"], **readable_params})
    pd.DataFrame(best_params_rows).to_csv(os.path.join(RESULTS_DIR, "best_hyperparameters.csv"), index=False)

    print("\nDone. Results written to ./results/")


if __name__ == "__main__":
    main()
