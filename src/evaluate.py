"""
Shared evaluation utilities reused by every model notebook.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, recall_score, precision_score, f1_score,
)


def evaluate(model, X, y_true, threshold: float = 0.5, label: str = '') -> dict:
    """
    Evaluate a fitted classifier at a given threshold.
    Prints a classification report and returns a metrics dict.
    """
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        'threshold':    threshold,
        'accuracy':     accuracy_score(y_true, y_pred),
        'recall_1':     recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        'precision_1':  precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'f1_1':         f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }

    header = f'--- {label} ---' if label else '---'
    print(header)
    print(f'Threshold: {threshold:.2f}')
    print(classification_report(y_true, y_pred, target_names=['Safe (0)', 'Risky (1)']))

    return metrics


def plot_confusion_matrix(model, X, y_true, threshold: float = 0.5,
                          title: str = 'Confusion Matrix', save_path: str = None):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Safe (0)', 'Risky (1)'],
                yticklabels=['Safe (0)', 'Risky (1)'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'{title}\n(threshold={threshold:.2f})')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()



def tune_threshold(model, X, y_true, min_recall: float = 0.85,
                   save_path: str = None) -> float:
    """
    Sweep thresholds 0.01→0.99 and pick the lowest one that achieves
    at least `min_recall` on is_risky=1 while maximising precision.
    Returns the chosen threshold.
    """
    y_prob = model.predict_proba(X)[:, 1]
    thresholds = np.linspace(0.01, 0.99, 200)

    recalls    = [recall_score(y_true, (y_prob >= t).astype(int),
                               pos_label=1, zero_division=0) for t in thresholds]
    precisions = [precision_score(y_true, (y_prob >= t).astype(int),
                                  pos_label=1, zero_division=0) for t in thresholds]

    candidates = [(t, p) for t, r, p in zip(thresholds, recalls, precisions)
                  if r >= min_recall]
    best_threshold = max(candidates, key=lambda x: x[1])[0] if candidates \
                     else thresholds[int(np.argmax(recalls))]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, recalls,    lw=2, color='tomato',    label='Recall (is_risky=1)')
    ax.plot(thresholds, precisions, lw=2, color='steelblue', label='Precision (is_risky=1)')
    ax.axvline(best_threshold, color='gray', linestyle='--',
               label=f'Chosen = {best_threshold:.2f}')
    ax.axhline(min_recall, color='tomato', linestyle=':', alpha=0.5,
               label=f'Recall floor = {min_recall}')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Precision & Recall vs. Threshold')
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

    print(f'Chosen threshold: {best_threshold:.2f}')
    return best_threshold


def save_metrics(metrics: dict, path: str):
    """Append one metrics row to the shared CSV comparison table."""
    row = pd.DataFrame([metrics])
    if not os.path.exists(path):
        row.to_csv(path, index=False)
    else:
        pd.concat([pd.read_csv(path), row], ignore_index=True).to_csv(path, index=False)
    print(f'Metrics appended to {path}')
