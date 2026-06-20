"""
Logistic Regression — training and hyperparameter tuning.

Validation strategy: 5-fold StratifiedKFold entirely inside X_train.
The test set is never used here — only in the notebook for final evaluation.
Primary optimisation criterion: Recall on is_risky=1.
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, recall_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import SEED, MODELS_DIR

# --------------------------------------------------------
# Hyperparameter grid
# penalty is deprecated in sklearn 1.8+; use l1_ratio instead.
# l1_ratio=0 → L2 regularization, l1_ratio=1 → L1 regularization.
# saga is the solver that supports l1_ratio.
# --------------------------------------------------------
PARAM_GRID = {
    'C':         [0.001, 0.01, 0.1, 1, 10, 100],
    'l1_ratio':  [0, 1],
    'solver':    ['saga'],
    'max_iter':  [2000],
}

RECALL_SCORER = make_scorer(recall_score, pos_label=1, zero_division=0)

DEFAULT_MODEL_PATH = MODELS_DIR / 'logistic_regression.pkl'


def train_baseline(X_train: np.ndarray, y_train) -> LogisticRegression:
    """
    Fit a default LogisticRegression (C=1, L2) with no tuning.
    Used as a reference point before hyperparameter search.
    """
    model = LogisticRegression(solver='saga', l1_ratio=0, max_iter=2000, random_state=SEED)
    model.fit(X_train, y_train)
    return model


def tune(X_train: np.ndarray, y_train,
         cv_folds: int = 5) -> tuple:
    """
    GridSearchCV over PARAM_GRID using StratifiedKFold on X_train only.
    Returns (best_model, cv_results_dataframe).

    Stratified folds preserve the ~78/22 class ratio in every fold,
    which matters here because is_risky=1 is the majority but recall on
    it is still our primary concern.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)

    grid = GridSearchCV(
        LogisticRegression(random_state=SEED, solver='saga'),
        PARAM_GRID,
        scoring=RECALL_SCORER,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,         # refit best params on full X_train
    )
    grid.fit(X_train, y_train)

    print(f'\nBest params : {grid.best_params_}')
    print(f'Best CV recall (is_risky=1): {grid.best_score_:.4f}')

    import pandas as pd
    cv_df = pd.DataFrame(grid.cv_results_)
    return grid.best_estimator_, cv_df


def cross_validate_model(model: LogisticRegression,
                         X_train: np.ndarray, y_train,
                         cv_folds: int = 5) -> dict:
    """
    Run cross_validate on a fitted (or unfitted) model configuration
    and return per-fold scores for recall, precision, F1, and ROC-AUC.
    Useful for reporting fold-level variance in the final report.
    """
    from sklearn.metrics import make_scorer, precision_score, f1_score, accuracy_score

    scoring = {
        'accuracy':    make_scorer(accuracy_score),
        'recall_1':    make_scorer(recall_score,    pos_label=1, zero_division=0),
        'precision_1': make_scorer(precision_score, pos_label=1, zero_division=0),
        'f1_1':        make_scorer(f1_score,        pos_label=1, zero_division=0),
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    results = cross_validate(model, X_train, y_train, cv=cv,
                             scoring=scoring, n_jobs=-1)

    summary = {}
    for metric, values in results.items():
        if metric.startswith('test_'):
            name = metric[5:]
            summary[name] = {'mean': values.mean(), 'std': values.std(), 'folds': values.tolist()}

    return summary


def save_model(model: LogisticRegression, path=DEFAULT_MODEL_PATH):
    joblib.dump(model, path)
    print(f'Model saved to {path}')


def load_model(path=DEFAULT_MODEL_PATH) -> LogisticRegression:
    return joblib.load(path)
