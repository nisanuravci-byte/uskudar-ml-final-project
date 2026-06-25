"""
Section 3: Nested cross-validation.

Outer loop: 5-fold CV -> unbiased estimate of test performance.
Inner loop: 3-fold CV (via GridSearchCV) -> selects best hyperparameters
            for the fixed top-15 ANOVA feature selection (Flowchart B),
            with feature selection itself refit per fold (Flowchart A is
            folded into the pipeline since k=15 is fixed by the spec).

For each model and each outer fold we store: best hyperparameters, the
fold's held-out predictions/probabilities, and the features selected by
SelectKBest on that fold's training data.

NOTE on runtime: kernel SVM (rbf/linear via libsvm) scales roughly
quadratically-to-cubically with training set size. On the full ~80k-row
outer training fold this made grid search intractable (10+ hours, no
convergence in practice). `train_cap` lets us subsample just the SVM
training fold (stratified) to keep it tractable; all other models still
see the full outer training fold. This is documented in the final report.
"""
import sys
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split


def _maybe_cap(X_train, y_train, cap, seed):
    if cap is None or len(X_train) <= cap:
        return X_train, y_train
    X_capped, _, y_capped, _ = train_test_split(
        X_train, y_train, train_size=cap, stratify=y_train, random_state=seed
    )
    return X_capped, y_capped


def run_nested_cv(X, y, model_grids, outer_splits=5, inner_splits=3, seed=42,
                   verbose=True, train_caps=None):
    """
    train_caps: optional dict {model_name: max_training_rows}. If a model's
    outer-fold training set exceeds the cap, it is stratified-subsampled
    down to the cap before the inner grid search (keeps slow models like
    SVM tractable on a 100k-row dataset).
    """
    train_caps = train_caps or {}
    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)

    results = {}
    for name, (pipeline, param_grid) in model_grids.items():
        if verbose:
            print(f"\n=== {name} ===", flush=True)
        fold_records = []
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
            t0 = time.time()
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            cap = train_caps.get(name)
            X_fit, y_fit = _maybe_cap(X_train, y_train, cap, seed)
            if verbose and cap is not None and len(X_fit) < len(X_train):
                print(f"  fold {fold_idx}: capped training rows {len(X_train)} -> {len(X_fit)}",
                      flush=True)

            search = GridSearchCV(
                pipeline, param_grid, cv=inner_cv, scoring="roc_auc", n_jobs=-1, verbose=0
            )
            search.fit(X_fit, y_fit)
            best_model = search.best_estimator_

            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)[:, 1]

            selected_mask = best_model.named_steps["select"].get_support()

            fold_records.append({
                "fold": fold_idx,
                "best_params": search.best_params_,
                "inner_best_score": search.best_score_,
                "y_test": y_test,
                "y_pred": y_pred,
                "y_proba": y_proba,
                "selected_mask": selected_mask,
            })
            if verbose:
                elapsed = time.time() - t0
                print(f"  fold {fold_idx}/{outer_splits}: best_params={search.best_params_} "
                      f"inner_auc={search.best_score_:.4f} ({elapsed:.1f}s)", flush=True)

        results[name] = fold_records
    return results
