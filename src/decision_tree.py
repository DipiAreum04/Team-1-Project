"""
Decision Tree — training and hyperparameter tuning.

Validation strategy: 5-fold StratifiedKFold entirely inside X_train.
The test set is never used here — only in the notebook for final evaluation.
Primary optimisation criterion: Recall on is_risky=1.
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, recall_score, precision_score, f1_score, accuracy_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import SEED, MODELS_DIR

# --------------------------------------------------------
# Hyperparameter grid
# max_depth controls tree size (None = unlimited, prone to overfitting)
# min_samples_split / min_samples_leaf control how early splits stop
# criterion is the impurity measure used to choose splits
# --------------------------------------------------------
PARAM_GRID = {
    'max_depth':        [3, 5, 10, 15, None],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf':  [1, 5, 10],
    'criterion':         ['gini', 'entropy'],
}

RECALL_SCORER = make_scorer(recall_score, pos_label=1, zero_division=0)

DEFAULT_MODEL_PATH = MODELS_DIR / 'decision_tree.pkl'


def train_baseline(X_train: np.ndarray, y_train) -> DecisionTreeClassifier:
    """
    Fit a default DecisionTreeClassifier with no tuning.
    Used as a reference point before hyperparameter search.
    """
    model = DecisionTreeClassifier(random_state=SEED)
    model.fit(X_train, y_train)
    return model


def tune(X_train: np.ndarray, y_train, cv_folds: int = 5) -> tuple:
    """
    GridSearchCV over PARAM_GRID using StratifiedKFold on X_train only.
    Returns (best_model, cv_results_dataframe).
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)

    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=SEED),
        PARAM_GRID,
        scoring=RECALL_SCORER,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    grid.fit(X_train, y_train)

    print(f'\nBest params : {grid.best_params_}')
    print(f'Best CV recall (is_risky=1): {grid.best_score_:.4f}')

    import pandas as pd
    return grid.best_estimator_, pd.DataFrame(grid.cv_results_)


def cross_validate_model(model: DecisionTreeClassifier,
                         X_train: np.ndarray, y_train,
                         cv_folds: int = 5) -> dict:
    """
    Run cross_validate and return per-fold scores for all four metrics.
    """
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


def save_model(model: DecisionTreeClassifier, path=DEFAULT_MODEL_PATH):
    joblib.dump(model, path)
    print(f'Model saved to {path}')


def load_model(path=DEFAULT_MODEL_PATH) -> DecisionTreeClassifier:
    return joblib.load(path)
