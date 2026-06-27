"""
Preprocessing pipeline for the Loan Approval Prediction project.
All transformations are fit on the training set only and applied to test set.
Validation is done using k-fold cross-validation inside the training set, so the test set remains untouched.
Target:
    loan_status (1 = approved, 0 = rejected) - predicted directly.
    The approved class (1) is the minority (~22%).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

# Make utils.py importable whether this runs from src/, the project root, or a notebook
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import SEED, MODELS_DIR 

# ========================================================
# Column groups
# ========================================================
EDUCATION_ORDER = [['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate']] # order of education levels

# person_age values above this are considered implausible for loans because 
# average life expectancy in Canada is 82.5 years. This caps outliers without removing rows.
AGE_CAP = 80  

# numerical columns
NUMERICAL_COLS = [ 
    'person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
    'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score'
]

# Feature Engineered numerical columns. Only present when add_engineered=True is used.
# Treated exactly like the raw numerical columns (scaled with StandardScaler).
ENGINEERED_COLS = ['employment_experience_ratio', 'credit_history_ratio']


# ordinal column
ORDINAL_COLS = ['person_education'] 

# nominal columns
NOMINAL_COLS = [ 
    'person_gender', 'person_home_ownership', 'loan_intent',
    'previous_loan_defaults_on_file'
]

# loan_status is the prediction target (y), so it is held out of the
# feature matrix X as it is the label, not a feature.
TARGET_COL = 'loan_status'


DEFAULT_PREPROCESSOR_PATH = MODELS_DIR / 'preprocessor.pkl'

# ========================================================
# FUNCTIONS
# ========================================================

# ========================================================
# Optional feature engineering
# ========================================================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add two ratio features. Purely row-wise arithmetic (no cross-row statistics),
    so it is leakage-free and can be applied before the train/test split.

    adult_years = (person_age - 18), floored at 1, to anchor ratios to working/adult
    years rather than raw age.

    NOTE: employment_experience_ratio can exceed 1 when stated experience is implausibly
    high for the age, so those rows are noise resulting from data quality issues.
    """
    df = df.copy()
    adult_years = (df['person_age'] - 18).clip(lower=1)
    df['employment_experience_ratio'] = df['person_emp_exp'] / adult_years
    df['credit_history_ratio'] = df['cb_person_cred_hist_length'] / adult_years
    return df


# ========================================================
# Load
# ========================================================
def load_data(path: str, add_engineered: bool = False) -> pd.DataFrame:
    """
    Load the CSV. Target is loan_status (1 = approved, 0 = rejected).
    add_engineered=True appends the two ratio features from add_features().
    """
    df = pd.read_csv(path)
    if add_engineered:
        df = add_features(df)
    return df


# ========================================================
# Numerical cleaning: Handle outlier ages and log-transform skewed features
# ========================================================
def _clean_numerical(X):
    """
    Cap implausible ages and log-transform the two heavy right-skewed features
    (income and employment experience). clip(lower=0) guards against any negative
    values producing NaN under log1p.
    """
    X = X.copy()
    X['person_age'] = X['person_age'].clip(upper=AGE_CAP)
    X['person_income'] = np.log1p(X['person_income'].clip(lower=0))
    X['person_emp_exp'] = np.log1p(X['person_emp_exp'].clip(lower=0))
    return X


# ========================================================
# Train/test split (80/20), stratified on loan_status.
# Cross-validation with 5-fold StratifiedKFold runs INSIDE the training
# set during tuning. Test set stays untouched until final evaluation.
# ========================================================
def split_data(df: pd.DataFrame, test_size: float = 0.20):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=SEED
    )

    return X_train, X_test, y_train, y_test


# ========================================================
# The preprocessor
# ========================================================
def build_preprocessor(drop_prev_defaults: bool = False,
                       add_engineered: bool = False) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
    - Cleans then scales numerical features (cap age, log skewed features, StandardScaler)
    - Ordinally encodes person_education (HS < Associate < Bachelor < Master < Doctorate)
    - One-hot encodes the nominal categoricals

    drop_prev_defaults: if True, excludes 'previous_loan_defaults_on_file' (the
        near-perfect shortcut predictor) to produce the WITHOUT feature set.
    add_engineered: if True, includes the two ratio features from add_features()
        in the numerical group.
    """
    numerical_pipeline = Pipeline([
        ('clean', FunctionTransformer(_clean_numerical, validate=False,
                                      feature_names_out='one-to-one')),
        ('scaler', StandardScaler()),
    ])

    ordinal_pipeline = Pipeline([
        ('ordinal', OrdinalEncoder(categories=EDUCATION_ORDER))
    ])

    # handle_unknown='ignore' keeps the demo robust if an unseen category ever appears (maps it to all-zeros).
    # sparse_output=False makes it return a dense array, which is easier to work with.
    nominal_pipeline = Pipeline([
        ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
    ])

    # numerical group: optionally extend with engineered ratio columns
    numerical_cols = NUMERICAL_COLS + ENGINEERED_COLS if add_engineered else NUMERICAL_COLS

    # feature-set switch: drop the loan default column from the nominal group only if flag is true
    nominal_cols = [c for c in NOMINAL_COLS if c != 'previous_loan_defaults_on_file'] \
        if drop_prev_defaults else NOMINAL_COLS

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, numerical_cols),   # local variable, not the constant
        ('ord', ordinal_pipeline, ORDINAL_COLS),
        ('nom', nominal_pipeline, nominal_cols),       # local variable, not the constant
    ], remainder='drop')

    return preprocessor


def fit_and_save_preprocessor(X_train: pd.DataFrame, path=DEFAULT_PREPROCESSOR_PATH,
                              drop_prev_defaults: bool = False,
                              add_engineered: bool = False) -> ColumnTransformer:
    """
    Fit a preprocessor on X_train and save it. Defaults to the WITH feature set.
    Pass drop_prev_defaults=True / add_engineered=True for variants, and give each
    variant a distinct `path` so it doesn't overwrite the demo pkl.
    """
    preprocessor = build_preprocessor(drop_prev_defaults=drop_prev_defaults,
                                      add_engineered=add_engineered)
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, path)
    print(f'Preprocessor saved to {path}')
    return preprocessor


def transform(preprocessor: ColumnTransformer, X: pd.DataFrame) -> np.ndarray:
    return preprocessor.transform(X)


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """
    Return the column names of the encoded matrix in output order.
    """
    return [name.split('__', 1)[-1] for name in preprocessor.get_feature_names_out()]



# ========================================================
# One-call data preparation function, reused by the model-training notebook.
# Deterministic with optional feature sets, and saves the preprocessor for reuse at inference time.
# ========================================================
def prepare(data_path: str, drop_prev_defaults: bool = False,
            add_engineered: bool = False, save_path=None):
    """
    Load -> (optional FE) -> split -> fit preprocessor on TRAIN ONLY -> transform both.
    Returns: X_train_enc, X_test_enc, y_train, y_test, feature_names, preprocessor.
    """
    df = load_data(data_path, add_engineered=add_engineered)
    X_train, X_test, y_train, y_test = split_data(df)

    pre = build_preprocessor(drop_prev_defaults=drop_prev_defaults,
                             add_engineered=add_engineered)
    pre.fit(X_train)
    if save_path is not None:
        joblib.dump(pre, save_path)
        print(f'Preprocessor saved to {save_path}')

    X_train_enc = transform(pre, X_train)
    X_test_enc = transform(pre, X_test)
    feature_names = get_feature_names(pre)
    return X_train_enc, X_test_enc, y_train, y_test, feature_names, pre


# ========================================================
# End-to-end convenience wrapper for the preprocessing notebook
# ========================================================
def run_full_pipeline(data_path: str):
    """End-to-end convenience function used by the preprocessing notebook."""
    df = load_data(data_path)

    # Class balance of the loan approval target (approved = 1 is the minority).
    ratio = df[TARGET_COL].value_counts(normalize=True).round(3).to_dict()
    print(f'loan_status class balance: {ratio}')

    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = fit_and_save_preprocessor(X_train)

    X_train_enc = transform(preprocessor, X_train)
    X_test_enc  = transform(preprocessor, X_test)

    feature_names = get_feature_names(preprocessor)

    print(f'\nSplit sizes - Train: {len(X_train)}, Test: {len(X_test)}')
    print(f'Encoded feature count: {X_train_enc.shape[1]}')
    print(f'Features: {feature_names}')

    return (X_train_enc, X_test_enc, y_train, y_test, feature_names, preprocessor)
