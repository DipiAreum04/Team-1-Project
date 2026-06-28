"""
Training pipeline for the Loan Approval Prediction project.

Leakage-safe sklearn pipelines (preprocessor inside), multi-metric grid search cross-validation 
tuning, and cross-validation evaluation helper functions used by the model-training notebook.
The notebook only wires up estimators and hyperparameter grids.
"""

import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import build_preprocessor, load_data, split_data
from utils import SEED, MODELS_DIR, RESULTS_DIR

# ========================================================
# CV config and scoring
# ========================================================
SCORING = {
    'f1':           'f1',
    'pr_auc':       'average_precision',
    'accuracy':     'accuracy',
    'balanced_acc': 'balanced_accuracy',
    'precision':    'precision',
    'recall':       'recall',
}
PRIMARY = 'f1'
METRIC_ORDER = ['f1', 'pr_auc', 'precision', 'recall', 'accuracy', 'balanced_acc']


def get_cv(n_splits: int = 5) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)


# ========================================================
# Pipeline helpers
# ========================================================
def make_pipe(estimator, drop_prev_defaults: bool = False, add_engineered: bool = False) -> Pipeline:
    return Pipeline([
        ('pre', build_preprocessor(drop_prev_defaults=drop_prev_defaults,
                                   add_engineered=add_engineered)),
        ('clf', estimator),
    ])


def _summarise(folds_by_metric):
    out = {}
    for m, folds in folds_by_metric.items():
        folds = np.asarray(folds, dtype=float)
        out[m] = {'mean': float(folds.mean()), 'std': float(folds.std()), 'folds': folds.tolist()}
    return out


def cv_eval(pipe, X, y, cv=None):
    """Cross-validate one fixed config and return per-fold mean/std/folds per metric."""
    cv = cv or get_cv()
    res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=-1, error_score='raise')
    return _summarise({m: res[f'test_{m}'] for m in SCORING})


def _per_fold_from_gs(gs, cv):
    i = gs.best_index_
    n = cv.get_n_splits()
    return _summarise({m: [gs.cv_results_[f'split{k}_test_{m}'][i] for k in range(n)]
                       for m in SCORING})


def _strip(params):
    return {k.replace('clf__', ''): v for k, v in params.items()}


def print_cv_summary(title, summary, show_folds=False):
    print(f'  {title}')
    for m in METRIC_ORDER:
        s = summary[m]
        if show_folds:
            folds = ' '.join(f'{v:.3f}' for v in s['folds'])
            print(f'    {m:13s} mean={s["mean"]:.4f}  std={s["std"]:.4f}  folds=[{folds}]')
        else:
            print(f'    {m:13s} mean={s["mean"]:.4f}  std={s["std"]:.4f}')


def evaluate_baseline(pipe, X, y, title, cv=None):
    summary = cv_eval(pipe, X, y, cv=cv)
    print_cv_summary(title, summary, show_folds=False)
    return summary


# ========================================================
# Training session (accumulates fitted pipelines and CV rows)
# ========================================================
class TrainingSession:
    """
    Holds training data and accumulates tuned pipelines and CV rows across model runs.
    Re-running run_model() for the same name overwrites prior results.
    """

    def __init__(self, X_train, y_train, drop_prev_defaults: bool = False,
                 add_engineered: bool = False, cv=None):
        self.X_train = X_train
        self.y_train = y_train
        self.drop_prev_defaults = drop_prev_defaults
        self.add_engineered = add_engineered
        self.cv = cv or get_cv()
        self.best_pipes = {}
        self.result_rows = {}

    def _make_pipe(self, estimator):
        return make_pipe(estimator, self.drop_prev_defaults, self.add_engineered)

    def _tune(self, estimator, grid):
        pipe = self._make_pipe(estimator)
        pgrid = {f'clf__{k}': v for k, v in grid.items()}
        gs = GridSearchCV(pipe, pgrid, scoring=SCORING, refit=PRIMARY,
                          cv=self.cv, n_jobs=-1, error_score='raise')
        gs.fit(self.X_train, self.y_train)
        return gs

    def run_model(self, name, estimator, grid):
        print('=' * 60)
        print(name)
        print('=' * 60)
        t0 = time.time()

        base = cv_eval(self._make_pipe(clone(estimator)), self.X_train, self.y_train, cv=self.cv)
        print_cv_summary('Baseline  : 5-fold CV (means/std):', base, show_folds=False)

        gs = self._tune(estimator, grid)
        print(f'  Best params: {_strip(gs.best_params_)}')

        tuned = _per_fold_from_gs(gs, self.cv)
        print_cv_summary('Tuned     : 5-fold CV (per fold):', tuned, show_folds=True)

        for m in [PRIMARY, 'pr_auc']:
            b, t = base[m]['mean'], tuned[m]['mean']
            print(f'  {m:6s}: baseline={b:.4f} -> tuned={t:.4f}  (delta {t - b:+.4f})')
        print(f'  ({time.time() - t0:.1f}s)')

        self.best_pipes[name] = gs.best_estimator_
        row = {'model': name, 'best_params': _strip(gs.best_params_)}
        row.update({m: tuned[m]['mean'] for m in SCORING})
        row.update({f'{m}_std': tuned[m]['std'] for m in SCORING})
        self.result_rows[name] = row
        return gs

    def results_df(self) -> pd.DataFrame:
        return (
            pd.DataFrame(self.result_rows.values())
              .sort_values(PRIMARY, ascending=False)
              .reset_index(drop=True)
        )

    def save(self, models_dir=MODELS_DIR, results_dir=RESULTS_DIR):
        models_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        for name, pipe in self.best_pipes.items():
            joblib.dump(pipe, models_dir / f'{name}.pkl')
        self.results_df().to_csv(results_dir / 'cv_results_primary.csv', index=False)


# ========================================================
# Optional feature-set experiments
# ========================================================
def _feature_set_specs():
    """
    Four models for section 9 feature-set experiments (SVM excluded since it is too slow).
    """
    return {
        'LogisticRegression': (
            LogisticRegression(solver='saga', max_iter=2000, random_state=SEED),
            {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'l1_ratio': [0, 0.5, 1],
                'class_weight': [None, 'balanced'],
            },
        ),
        'RandomForest': (
            RandomForestClassifier(random_state=SEED, n_jobs=1),
            {
                'n_estimators': [200, 400],
                'max_depth': [None, 10, 20],
                'class_weight': [None, 'balanced'],
            },
        ),
        'GradientBoosting': (
            HistGradientBoostingClassifier(random_state=SEED),
            {
                'learning_rate': [0.05, 0.1],
                'max_depth': [None, 6],
                'max_iter': [200, 400],
                'class_weight': [None, 'balanced'],
            },
        ),
        'GaussianNB': (
            GaussianNB(),
            {'var_smoothing': [1e-9, 1e-8, 1e-7]},
        ),
    }


def run_feature_set(data_path, drop_prev_defaults, add_engineered, tag, save=False,
                    cv=None, models_dir=MODELS_DIR):
    """Tune a subset of models on an alternate feature set; return a comparison DataFrame."""
    cv = cv or get_cv()
    df_ = load_data(data_path, add_engineered=add_engineered)
    Xtr, _, ytr, _ = split_data(df_)

    def mp(est):
        return make_pipe(est, drop_prev_defaults=drop_prev_defaults,
                         add_engineered=add_engineered)

    out = []
    specs = _feature_set_specs()
    print(f'Feature set "{tag}": tuning {len(specs)} models ({", ".join(specs)})')
    for name, (est, grid) in specs.items():
        print(f'  -> {name} ...', flush=True)
        gs = GridSearchCV(
            mp(est), {f'clf__{k}': v for k, v in grid.items()},
            scoring=SCORING, refit=PRIMARY, cv=cv, n_jobs=-1, error_score='raise',
        ).fit(Xtr, ytr)
        out.append({'feature_set': tag, 'model': name, PRIMARY: gs.best_score_})
        print(f'     done  F1={gs.best_score_:.4f}', flush=True)
        if save:
            joblib.dump(gs.best_estimator_, models_dir / f'{name}_{tag}.pkl')
    return pd.DataFrame(out)


def save_best_demo_models(session, res_without, res_eng_with=None, res_eng_without=None,
                          models_dir=MODELS_DIR):
    """
    Export demo pipelines:
    - best WITH-prevdef from main training (section 7)
    - best WITHOUT-prevdef from section 8.1
    - best engineered-feature variant from section 8.2 (WITH vs WITHOUT prevdef)
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    def _copy_best(row, dest_name):
        name = str(row['model'])
        tag = str(row['feature_set'])
        src = models_dir / f'{name}_{tag}.pkl'
        dest = models_dir / dest_name
        if not src.exists():
            raise FileNotFoundError(f'Expected trained model not found: {src}')
        joblib.dump(joblib.load(src), dest)
        return src, dest

    with_df = session.results_df()
    best_with = with_df.iloc[0]
    with_name = str(best_with['model'])
    with_src = models_dir / f'{with_name}.pkl'
    with_dest = models_dir / 'best_demo_model_with_prevdef.pkl'
    if not with_src.exists():
        raise FileNotFoundError(f'Expected trained model not found: {with_src}')
    joblib.dump(joblib.load(with_src), with_dest)

    best_wo = res_without.loc[res_without[PRIMARY].idxmax()]
    wo_src, wo_dest = _copy_best(best_wo, 'best_demo_model_without_prevdef.pkl')

    metadata = {
        'with_prevdef': {
            'model_name': with_name,
            'feature_set': 'WITH_prevdef',
            f'cross_validation_{PRIMARY}': float(best_with[PRIMARY]),
            'source_file': with_src.name,
            'demo_file': with_dest.name,
        },
        'without_prevdef': {
            'model_name': str(best_wo['model']),
            'feature_set': str(best_wo['feature_set']),
            f'cross_validation_{PRIMARY}': float(best_wo[PRIMARY]),
            'source_file': wo_src.name,
            'demo_file': wo_dest.name,
        },
    }

    if res_eng_with is not None and res_eng_without is not None:
        res_eng = pd.concat([res_eng_with, res_eng_without], ignore_index=True)
        best_eng = res_eng.loc[res_eng[PRIMARY].idxmax()]
        eng_src, eng_dest = _copy_best(best_eng, 'best_demo_model_engineered.pkl')
        metadata['engineered'] = {
            'model_name': str(best_eng['model']),
            'feature_set': str(best_eng['feature_set']),
            f'cross_validation_{PRIMARY}': float(best_eng[PRIMARY]),
            'source_file': eng_src.name,
            'demo_file': eng_dest.name,
        }

    metadata_path = models_dir / 'best_demo_model_metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)

    print(f'Best WITH prevdef   : {with_name}  (F1={best_with[PRIMARY]:.4f})')
    print(f'  -> {with_dest}')
    print(f'Best WITHOUT prevdef: {best_wo["model"]}  (F1={best_wo[PRIMARY]:.4f})')
    print(f'  -> {wo_dest}')
    if 'engineered' in metadata:
        eng = metadata['engineered']
        print(f'Best engineered     : {eng["model_name"]}  ({eng["feature_set"]}, F1={eng[f"cross_validation_{PRIMARY}"]:.4f})')
        print(f'  -> {models_dir / eng["demo_file"]}')
    print(f'Metadata -> {metadata_path}')
    return metadata
