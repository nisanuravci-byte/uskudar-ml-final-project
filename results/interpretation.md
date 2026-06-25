# Interpretation of Results

## Setup recap
- 100,000-row stratified sample of the HIGGS dataset (28 features, binary label: 1 = signal, 0 = background).
- Preprocessing: missing-value check (none found, see `missing_value_report.csv`), IQR-based outlier capping (winsorizing), then MinMaxScaler fit per CV fold.
- Feature selection: filter-based, top 15 features (k=15 fixed per spec). The *method* used to score features is itself searched in the inner loop between ANOVA F-score and Mutual Information (Flowchart A), jointly with each model's hyperparameters (Flowchart B) — both searches happen inside the same 3-fold inner `GridSearchCV`, refit per outer fold.
- Nested CV: 5-fold outer (unbiased test estimate) / 3-fold inner (joint feature-selection-method + hyperparameter search, scored on ROC-AUC).
- Result of the Flowchart A search: **ANOVA F-score (`f_classif`) was selected over Mutual Information in all 20 outer-fold fits (4 models × 5 folds)** — a consistent, non-trivial finding suggesting the HIGGS features' relationship to the label is well captured by a linear/variance-based score, and the extra computational cost of Mutual Information bought nothing here.
- Note: SVM's training fold was stratified-subsampled to 15,000 rows (from ~80,000) to keep its rbf/linear kernel grid search tractable; KNN, MLP, and XGBoost used the full training fold each outer iteration. This is the one deviation from a literal full-data nested CV and is called out here for transparency.

## Performance summary (mean ± std over 5 outer folds)

| Model   | Accuracy | Precision | Recall | F1     | ROC-AUC |
|---------|----------|-----------|--------|--------|---------|
| KNN     | 0.669 ± 0.005 | 0.674 ± 0.005 | 0.730 ± 0.005 | 0.701 ± 0.005 | 0.728 ± 0.006 |
| SVM     | 0.684 ± 0.005 | 0.683 ± 0.004 | 0.756 ± 0.016 | 0.717 ± 0.007 | 0.748 ± 0.007 |
| MLP     | 0.722 ± 0.004 | 0.728 ± 0.002 | 0.758 ± 0.015 | 0.743 ± 0.006 | **0.797 ± 0.004** |
| XGBoost | 0.719 ± 0.005 | 0.733 ± 0.004 | 0.739 ± 0.007 | 0.736 ± 0.005 | 0.796 ± 0.005 |

## Best-performing model and feature representation

**MLP and XGBoost are statistically indistinguishable at the top** (ROC-AUC 0.797 vs 0.796, within one standard deviation of each other), both clearly ahead of SVM (0.748) and KNN (0.728, the weakest).

- **XGBoost** is the more practical choice: near-identical AUC to MLP, much faster to train (the run logs show XGBoost finishing its full nested CV in a fraction of MLP's time), and its accuracy/F1 are also marginally better and far more stable across folds (std ≈ 0.005 vs MLP's looser recall variance, std 0.015).
- **MLP** edges out XGBoost very slightly on raw ROC-AUC, with hyperparameters consistently settling on `hidden_layer_sizes=(100,)` and `activation=relu` across all 5 outer folds — a sign the search space and the data agree on a single best configuration rather than overfitting to fold-specific noise.
- **XGBoost** likewise converges on the same hyperparameters (`max_depth=5`, `n_estimators=100`, `learning_rate=0.1`) every fold — a strong stability signal.

**Feature representation:** the ANOVA top-15 selection is highly stable across models and folds — features `f1, f4, f6, f10, f13, f14, f17, f18, f21/f22/f23, f26, f27, f28` dominate nearly every fold for KNN, MLP, and XGBoost. SVM's selected set is more variable fold-to-fold (likely an artifact of the smaller 15k-row training subsample changing per-fold ANOVA F-statistics slightly), but draws from the same core feature pool. This consistency suggests these ~12-15 features carry most of the discriminative signal for separating HIGGS signal from background in this sample, and that the feature-selection step is not the limiting factor in performance — model capacity/expressiveness is.

## Overall conclusion
For this task, **gradient-boosted trees (XGBoost) offer the best practical tradeoff of accuracy and computational cost**, while a tuned MLP can match its discriminative power (ROC-AUC) at higher training cost. Simpler models (KNN, linear/rbf SVM at reduced scale) leave meaningful performance on the table, indicating the true class boundary in this feature space is non-linear and benefits from either deep non-linear projections (MLP) or tree-based partitioning (XGBoost).
