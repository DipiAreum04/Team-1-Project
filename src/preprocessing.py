"""
Preprocessing pipeline for the Loan Default Risk project..
All transformations are fit on the training set only and applied to val/test.
Target:
    is_risky = 1 - loan_status
    (loan_status: 1 = approved, 0 = rejected  ->  risky = rejected applicant)
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

# Make utils.py importable whether this runs from src/, the project root, or a notebook.
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

# ordinal column
ORDINAL_COLS = ['person_education'] 

# nominal columns
NOMINAL_COLS = [ 
    'person_gender', 'person_home_ownership', 'loan_intent',
    'previous_loan_defaults_on_file'
]

# loan_status is dropped before modeling because
# it is the inverse of our target variable (is_risky)
DROP_COLS = ['loan_status']


DEFAULT_PREPROCESSOR_PATH = MODELS_DIR / 'preprocessor.pkl'

# ========================================================
# FUNCTIONS
# ========================================================

# ========================================================
# Target creation
# ========================================================
def load_and_create_target(path: str) -> pd.DataFrame:
    """Load the CSV and build the binary target is_risky = 1 - loan_status."""
    df = pd.read_csv(path)
    df['is_risky'] = 1 - df['loan_status']
    return df

# ========================================================
# Numerical cleaning: Handle outlier ages and log-transform income
# ========================================================
def _clean_numerical(X):
    """
    Cap implausible ages and log-transform income.
    """
    X = X.copy()
    X['person_age'] = X['person_age'].clip(upper=AGE_CAP)
    X['person_income'] = np.log1p(X['person_income'])
    return X


# ========================================================
# Train/test split (80/20), stratified on is_risky.
# Cross-validation with 5-fold StratifiedKFold runs INSIDE the training
# set during tuning. Test set stays untouched until final evaluation.
# ========================================================
def split_data(df: pd.DataFrame, test_size: float = 0.20):
    X = df.drop(columns=DROP_COLS + ['is_risky'])
    y = df['is_risky']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=SEED
    )

    return X_train, X_test, y_train, y_test


# ========================================================
# The preprocessor
# ========================================================
def build_preprocessor(drop_prev_defaults: bool = False) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
    - Cleans then scales numerical features (cap age, log income, StandardScaler)
    - Ordinally encodes person_education (HS < Associate < Bachelor < Master < Doctorate)
    - One-hot encodes the nominal categoricals
    
    drop_prev_defaults: if True, excludes 'previous_loan_defaults_on_file' (the
    near-perfect shortcut predictor) to produce the WITHOUT feature set.
    """
    numerical_pipeline = Pipeline([
        ('clean', FunctionTransformer(_clean_numerical, validate=False)),
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

    # feature-set switch: drop the shortcut column from the nominal group only
    nominal_cols = [c for c in NOMINAL_COLS if c != 'previous_loan_defaults_on_file'] \
        if drop_prev_defaults else NOMINAL_COLS


    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, NUMERICAL_COLS),
        ('ord', ordinal_pipeline, ORDINAL_COLS),
        ('nom', nominal_pipeline, nominal_cols), # local variable, not the constant
    ], remainder='drop')

    return preprocessor


def fit_and_save_preprocessor(X_train: pd.DataFrame, path=DEFAULT_PREPROCESSOR_PATH) -> ColumnTransformer:
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, path)
    print(f'Preprocessor saved to {path}')
    return preprocessor


def transform(preprocessor: ColumnTransformer, X: pd.DataFrame) -> np.ndarray:
    return preprocessor.transform(X)


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Return the column names of the encoded matrix, in output order."""
    ohe = preprocessor.named_transformers_['nom']['ohe']
    # ohe.feature_names_in_ = the nominal cols this preprocessor was actually fit on
    # (3 cols for the WITHOUT set, 4 for WITH)
    nom_names = ohe.get_feature_names_out(ohe.feature_names_in_).tolist()
    return list(NUMERICAL_COLS) + list(ORDINAL_COLS) + nom_names


# ========================================================
# End-to-end convenience wrapper for the notebook
# ========================================================
def run_full_pipeline(data_path: str):
    """End-to-end convenience function used by the preprocessing notebook."""
    df = load_and_create_target(data_path)

    # Confirms which class "risky" actually is.
    ratio = df['is_risky'].value_counts(normalize=True).round(3).to_dict()
    print(f'is_risky class balance: {ratio}')

    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = fit_and_save_preprocessor(X_train)

    X_train_enc = transform(preprocessor, X_train)
    X_test_enc  = transform(preprocessor, X_test)

    feature_names = get_feature_names(preprocessor)

    print(f'\nSplit sizes - Train: {len(X_train)}, Test: {len(X_test)}')
    print(f'Encoded feature count: {X_train_enc.shape[1]}')
    print(f'Features: {feature_names}')

    return (X_train_enc, X_test_enc,
            y_train, y_test,
            feature_names, preprocessor)