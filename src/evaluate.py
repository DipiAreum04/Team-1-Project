"""
Evaluation helpers for the Loan Approval Prediction project.

Out-of-fold threshold tuning (leakage-safe), test-set scoring, and the plotting /
error-analysis routines used by the 04_evaluation notebook. The notebook loads the
saved pipelines and wires these together, so all reusable logic lives here.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix, precision_recall_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import MODELS_DIR, RESULTS_DIR, PLOTS_DIR, SEED

# ========================================================
# Config
# ========================================================
SELECTION_METRIC = 'F1'
METRIC_COLS = ['Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'F1', 'PR-AUC']

MODEL_FILES = {
    'Logistic Regression':  'LogisticRegression.pkl',
    'SVM (RBF)':            'SVM_RBF.pkl',
    'Gaussian Naive Bayes': 'GaussianNB.pkl',
    'Random Forest':        'RandomForest.pkl',
    'Gradient Boosting':    'GradientBoosting.pkl',
}


def get_cv(n_splits: int = 5) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)


CV_SPLITS = get_cv()


def set_plot_style():
    """Apply the notebook's plotting theme and ensure the plots dir exists."""
    sns.set_theme(style='whitegrid', palette='muted')
    plt.rcParams['figure.dpi'] = 120
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _slug(name: str) -> str:
    return name.lower().replace(' ', '_').replace('(', '').replace(')', '')


# ========================================================
# Loading
# ========================================================
def load_models(model_files=None, models_dir=MODELS_DIR, verbose: bool = True) -> dict:
    """Load the saved pipelines named in `model_files` (default: MODEL_FILES)."""
    model_files = model_files or MODEL_FILES
    models = {}
    for name, fname in model_files.items():
        path = models_dir / fname
        if not path.exists():
            raise FileNotFoundError(f'{path} not found - run 03_model_training first.')
        models[name] = joblib.load(path)
        if verbose:
            print(f'Loaded: {name}')
    return models


def load_cv_results(results_dir=RESULTS_DIR) -> pd.DataFrame:
    cv_df = pd.read_csv(results_dir / 'cv_results_primary.csv')
    return cv_df.drop_duplicates(subset='model').reset_index(drop=True)


def format_cv_results(cv_df, display_cols=('model', 'pr_auc', 'accuracy', 'precision', 'recall', 'f1'),
                      sort_by: str = 'pr_auc') -> pd.DataFrame:
    """Format CV metrics as 'mean ± std' strings. Sorts numerically BEFORE formatting."""
    display_cols = list(display_cols)
    out = cv_df.sort_values(sort_by, ascending=False).reset_index(drop=True).copy()
    for col in display_cols[1:]:
        std = f'{col}_std'
        if std in out.columns:
            out[col] = out.apply(lambda r: f"{r[col]:.4f} ± {r[std]:.4f}", axis=1)
    return out[display_cols]


# ========================================================
# Scoring and  threshold tuning
# ========================================================
def get_scores(pipeline, X):
    """Probability-like scores for the positive class (approved=1).
    SVMs lack predict_proba, so decision_function is used; average_precision handles either."""
    clf = pipeline.named_steps['clf']
    if hasattr(clf, 'predict_proba'):
        return pipeline.predict_proba(X)[:, 1]
    return pipeline.decision_function(X)


def get_oof_scores(pipeline, X, y, cv=None):
    """Out-of-fold scores on the training set.

    Each score comes from a fold-clone of pipeline fit WITHOUT the row it scores
    (preprocessor + classifier both refit per fold). Avoids tuning the threshold on in-sample predictions.
    """
    cv = cv or CV_SPLITS
    clf = pipeline.named_steps['clf']
    if hasattr(clf, 'predict_proba'):
        scores = cross_val_predict(clone(pipeline), X, y, cv=cv, method='predict_proba', n_jobs=-1)
        return scores[:, 1]
    return cross_val_predict(clone(pipeline), X, y, cv=cv, method='decision_function', n_jobs=-1)


def tune_threshold(y_true, scores, plot=True, title='', save_path=None):
    """Find the threshold maximising positive-class F1 on precomputed scores.

    Pass out-of-fold scores (from `get_oof_scores`), not in-sample, or the threshold will be optimistic.
    """
    grid = np.linspace(scores.min(), scores.max(), 300)

    precisions, recalls, f1s = [], [], []
    for t in grid:
        preds = (scores >= t).astype(int)
        precisions.append(precision_score(y_true, preds, pos_label=1, zero_division=0))
        recalls.append(recall_score(y_true, preds, pos_label=1, zero_division=0))
        f1s.append(f1_score(y_true, preds, pos_label=1, zero_division=0))

    best_idx = int(np.argmax(f1s))
    best_t = float(grid[best_idx])

    if plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(grid, recalls,    lw=2, color='tomato',          label='Recall')
        ax.plot(grid, precisions, lw=2, color='steelblue',       label='Precision')
        ax.plot(grid, f1s,        lw=2, color='mediumseagreen',  label='F1')
        ax.axvline(best_t, color='gray', linestyle='--', label=f'Best threshold = {best_t:.3f}')
        ax.set_xlabel('Threshold'); ax.set_ylabel('Score')
        ax.set_title(f'Threshold Tuning (out-of-fold) - {title}')
        ax.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    return best_t


def tune_all_thresholds(models, X_train, y_train, cv=None, plot=True,
                        plots_dir=PLOTS_DIR, verbose=True) -> dict:
    """Out-of-fold F1-optimal threshold per model. Returns {name: threshold}."""
    best = {}
    for name, pipeline in models.items():
        oof = get_oof_scores(pipeline, X_train, y_train, cv=cv)
        save = str(plots_dir / f'threshold_{_slug(name)}.png') if plot else None
        best[name] = tune_threshold(y_train, oof, plot=plot, title=name, save_path=save)
        if verbose:
            print(f'{name}: best threshold (out-of-fold) = {best[name]:.4f}\n')
    return best


# ========================================================
# Test-set evaluation
# ========================================================
def evaluate_model(pipeline, X_test, y_test, threshold) -> dict:
    scores = get_scores(pipeline, X_test)
    preds = (scores >= threshold).astype(int)
    return {
        'Threshold':         round(float(threshold), 4),
        'Accuracy':          accuracy_score(y_test, preds),
        'Balanced Accuracy': balanced_accuracy_score(y_test, preds),
        'Precision':         precision_score(y_test, preds, pos_label=1, zero_division=0),
        'Recall':            recall_score(y_test, preds, pos_label=1, zero_division=0),
        'F1':                f1_score(y_test, preds, pos_label=1, zero_division=0),
        'PR-AUC':            average_precision_score(y_test, scores),
    }


def evaluate_models(models, X_test, y_test, thresholds, sort_by=SELECTION_METRIC,
                    save_path=None) -> pd.DataFrame:
    """Per-model test-set metrics at each model's tuned threshold, sorted by `sort_by`."""
    rows = []
    for name, pipeline in models.items():
        rows.append({'Model': name, **evaluate_model(pipeline, X_test, y_test, thresholds[name])})
    results_df = pd.DataFrame(rows).sort_values(sort_by, ascending=False).reset_index(drop=True)
    if save_path is not None:
        results_df.to_csv(save_path, index=False)
    return results_df


def style_results(results_df, metric_cols=METRIC_COLS):
    """Jupyter Styler: 4dp formatting + best/worst highlighting per metric column."""
    fmt = {c: '{:.4f}' for c in ['Threshold'] + metric_cols}
    return (results_df.style.format(fmt)
            .highlight_max(subset=metric_cols, color='lightgreen')
            .highlight_min(subset=metric_cols, color='#ffcccc'))


# ========================================================
# Plots
# ========================================================
def plot_confusion_matrices(models, X_test, y_test, thresholds, results_df=None,
                            ncols=3, save_path=None):
    n = len(models)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.3, nrows * 5))
    axes_flat = np.atleast_1d(axes).flatten()

    for i, (name, pipeline) in enumerate(models.items()):
        scores = get_scores(pipeline, X_test)
        preds = (scores >= thresholds[name]).astype(int)
        cm = confusion_matrix(y_test, preds)

        subtitle = ''
        if results_df is not None:
            row = results_df[results_df['Model'] == name].iloc[0]
            subtitle = f"\nP={row['Precision']:.3f}  R={row['Recall']:.3f}  F1={row['F1']:.3f}"

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes_flat[i],
                    xticklabels=['Rejected (0)', 'Approved (1)'],
                    yticklabels=['Rejected (0)', 'Approved (1)'])
        axes_flat[i].set_title(f'{name}{subtitle}', fontsize=10)
        axes_flat[i].set_xlabel('Predicted'); axes_flat[i].set_ylabel('Actual')

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)  # hide unused subplots

    plt.suptitle('Confusion Matrices - Test Set (all models)', fontsize=14, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_pr_curves(models, X_test, y_test, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.tab10

    for i, (name, pipeline) in enumerate(models.items()):
        scores = get_scores(pipeline, X_test)
        prec, rec, _ = precision_recall_curve(y_test, scores)
        ap = average_precision_score(y_test, scores)
        ax.plot(rec, prec, lw=2, color=cmap(i % 10), label=f'{name} (AP={ap:.3f})')

    ax.axhline(y_test.mean(), color='gray', linestyle='--', lw=1,
               label=f'No-skill baseline ({y_test.mean():.3f})')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves - Test Set')
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_importances(pipeline, model_name, feature_names, X_eval=None, y_eval=None,
                     top_n=15, save_path=None):
    """Top feature importances for a fitted pipeline.

    Native `feature_importances_` (Random Forest) or `coef_` (Logistic Regression) when
    available. HistGradientBoosting exposes neither, so it falls back to permutation
    importance on `X_eval`/`y_eval` (reported on the original pre-one-hot columns, since
    the preprocessor lives inside the pipeline).
    """
    clf = pipeline.named_steps['clf']

    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
        names = feature_names
    elif hasattr(clf, 'coef_'):
        importances = np.abs(clf.coef_[0])
        names = feature_names
    else:
        if X_eval is None or y_eval is None:
            print(f'{model_name}: no native feature importances, and no evaluation data given for permutation importance.')
            return
        perm = permutation_importance(
            pipeline, X_eval, y_eval,
            scoring='average_precision', n_repeats=10,
            random_state=SEED, n_jobs=-1,
        )
        importances = perm.importances_mean
        names = list(X_eval.columns)  # raw columns -- preprocessing lives inside the pipeline

    imp_df = (pd.DataFrame({'feature': names, 'importance': importances})
              .sort_values('importance', ascending=False).head(top_n))

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    ax.barh(imp_df['feature'][::-1], imp_df['importance'][::-1], color='steelblue', edgecolor='white')
    ax.set_title(f'{model_name} - Top {top_n} Feature Importances')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ========================================================
# Error analysis
# ========================================================
def outcome_masks(y_true, preds) -> dict:
    """Boolean masks for TP / TN / FP / FN. Inputs are aligned positionally."""
    y_true = np.asarray(y_true)
    preds = np.asarray(preds)
    return {
        'TP': (preds == 1) & (y_true == 1),
        'TN': (preds == 0) & (y_true == 0),
        'FP': (preds == 1) & (y_true == 0),
        'FN': (preds == 0) & (y_true == 1),
    }


def outcome_profile(X, masks, numeric_cols) -> pd.DataFrame:
    """Mean of `numeric_cols` for each outcome group, columns ordered TP/FP/TN/FN."""
    cols = [c for c in numeric_cols if c in X.columns]
    label = {'TP': 'True Positive', 'FP': 'False Positive',
             'TN': 'True Negative', 'FN': 'False Negative'}
    series = [X[masks[k]][cols].mean().rename(label[k]) for k in ['TP', 'FP', 'TN', 'FN']]
    return pd.concat(series, axis=1).round(2)
