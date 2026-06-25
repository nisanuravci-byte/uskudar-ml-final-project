"""
Performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC) and ROC curve plotting.
Binary classification (HIGGS signal=1 vs background=0), so "OVA" reduces to a
single ROC curve per model; we additionally overlay all models' fold-averaged
curves on one figure for comparison.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)


def fold_metrics(record):
    y_test, y_pred, y_proba = record["y_test"], record["y_pred"], record["y_proba"]
    return {
        "fold": record["fold"],
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def build_metrics_table(results: dict) -> pd.DataFrame:
    rows = []
    for model_name, fold_records in results.items():
        for record in fold_records:
            m = fold_metrics(record)
            m["model"] = model_name
            rows.append(m)
    df = pd.DataFrame(rows)
    summary = df.groupby("model")[["accuracy", "precision", "recall", "f1", "roc_auc"]].agg(["mean", "std"])
    return df, summary


def plot_roc_curves(results: dict, out_path: str):
    plt.figure(figsize=(7, 6))
    for model_name, fold_records in results.items():
        y_test_all = np.concatenate([r["y_test"] for r in fold_records])
        y_proba_all = np.concatenate([r["y_proba"] for r in fold_records])
        fpr, tpr, _ = roc_curve(y_test_all, y_proba_all)
        auc = roc_auc_score(y_test_all, y_proba_all)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (pooled outer-fold predictions)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def selected_features_report(results: dict, feature_names):
    rows = []
    for model_name, fold_records in results.items():
        for record in fold_records:
            mask = record["selected_mask"]
            feats = [f for f, m in zip(feature_names, mask) if m]
            rows.append({"model": model_name, "fold": record["fold"], "selected_features": feats})
    return pd.DataFrame(rows)
