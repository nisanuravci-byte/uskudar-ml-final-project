"""
Section 2 & 3: Feature selection + model/hyperparameter definitions.

Each model is wrapped in a Pipeline:
    MinMaxScaler -> SelectKBest(k=15, score_func searched) -> estimator
so that scaling and feature selection are refit on the training portion of
every CV fold (no leakage from validation/test folds).

Flowchart A (feature-selection search) and Flowchart B (hyperparameter
search) are combined into a single inner-loop GridSearchCV per the
"combined nested CV" design: each candidate jointly varies the filter-based
feature-selection method (ANOVA F-score vs Mutual Information, per the
spec's "ANOVA F-score or Mutual Information") AND the model's
hyperparameters. k is held fixed at 15 features, as the spec specifies.
"""
from functools import partial

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

N_FEATURES = 15
MI_SCORE_FUNC = partial(mutual_info_classif, random_state=42)

# Flowchart A search space: which filter method selects the top-15 features.
FEATURE_SELECTION_METHODS = [f_classif, MI_SCORE_FUNC]


def make_pipeline(estimator):
    return Pipeline([
        ("scaler", MinMaxScaler()),
        ("select", SelectKBest(score_func=f_classif, k=N_FEATURES)),
        ("model", estimator),
    ])


def get_model_grid():
    """Returns dict: name -> (pipeline, param_grid) for GridSearchCV (inner loop)."""
    grids = {
        "KNN": (
            make_pipeline(KNeighborsClassifier()),
            {
                "select__score_func": FEATURE_SELECTION_METHODS,
                "model__n_neighbors": list(range(3, 12)),  # 3..11 inclusive, per spec
            },
        ),
        "SVM": (
            make_pipeline(SVC(probability=True, random_state=42)),
            {
                "select__score_func": FEATURE_SELECTION_METHODS,
                "model__C": [0.1, 1, 10],
                "model__kernel": ["linear", "rbf"],
            },
        ),
        "MLP": (
            make_pipeline(MLPClassifier(max_iter=300, random_state=42)),
            {
                "select__score_func": FEATURE_SELECTION_METHODS,
                "model__hidden_layer_sizes": [(50,), (100,)],
                "model__activation": ["relu", "tanh"],
            },
        ),
        "XGBoost": (
            make_pipeline(XGBClassifier(
                eval_metric="logloss", random_state=42, n_jobs=1
            )),
            {
                "select__score_func": FEATURE_SELECTION_METHODS,
                "model__n_estimators": [50, 100],
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.05, 0.1],
            },
        ),
    }
    return grids
