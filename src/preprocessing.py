"""
Preprocessing pipeline.
All transformations are fit on the training set only and applied to val/test.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

# ========================================================
# global constants
# ========================================================
SEED = 42 # random seed for reproducibility

EDUCATION_ORDER = [['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate']] # order of education levels

AGE_CAP = 80  # person_age values above this are biologically invalid

NUMERICAL_COLS = [ # numerical columns
    'person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
    'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score'
]

ORDINAL_COLS = ['person_education'] # ordinal column

NOMINAL_COLS = [ # nominal columns
    'person_gender', 'person_home_ownership', 'loan_intent',
    'previous_loan_defaults_on_file'
]

# loan_status is dropped — it is the inverse of our target variable (is_risky)
DROP_COLS = ['loan_status']

# ========================================================
# FUNCTIONS
# ========================================================

# ========================================================
# load and create target our target variable is_risky
# ========================================================
def load_and_create_target(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['is_risky'] = 1 - df['loan_status']
    return df


# ========================================================
# handle outliers in person_age by capping at AGE_CAP
# ========================================================
def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['person_age'] = df['person_age'].clip(upper=AGE_CAP)
    return df


# ========================================================
# apply log1p transform to person_income to reduce right skew
# ========================================================
def apply_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['person_income'] = np.log1p(df['person_income'])
    return df


# ========================================================
# split data into train, val, and test sets (70/15/15 split)
# stratify on is_risky to preserve the ~78/22 class ratio in each split
# ========================================================
def split_data(df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15):
    X = df.drop(columns=DROP_COLS + ['is_risky'])
    y = df['is_risky']

    # First split off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=SEED
    )

    # Then split the remaining data into train and val
    relative_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val, stratify=y_temp, random_state=SEED
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
    - Scales numerical features with StandardScaler
    - Ordinally encodes person_education (High School < Associate < Bachelor < Master < Doctorate)
    - One-hot encodes nominal categoricals
    """
    numerical_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    ordinal_pipeline = Pipeline([
        ('ordinal', OrdinalEncoder(categories=EDUCATION_ORDER))
    ])

    # Use get_dummies style via OneHotEncoder; drop='first' removes one dummy per feature
    from sklearn.preprocessing import OneHotEncoder
    nominal_pipeline = Pipeline([
        ('ohe', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, NUMERICAL_COLS),
        ('ord', ordinal_pipeline, ORDINAL_COLS),
        ('nom', nominal_pipeline, NOMINAL_COLS),
    ], remainder='drop')

    return preprocessor


def fit_and_save_preprocessor(X_train: pd.DataFrame, path: str = '../models/preprocessor.pkl') -> ColumnTransformer:
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, path)
    print(f'Preprocessor saved to {path}')
    return preprocessor


def transform(preprocessor: ColumnTransformer, X: pd.DataFrame) -> np.ndarray:
    return preprocessor.transform(X)


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    num_names = NUMERICAL_COLS
    ord_names = ORDINAL_COLS
    ohe = preprocessor.named_transformers_['nom']['ohe']
    nom_names = ohe.get_feature_names_out(NOMINAL_COLS).tolist()
    return num_names + ord_names + nom_names


def run_full_pipeline(data_path: str):
    """End-to-end convenience function used by the preprocessing notebook."""
    df = load_and_create_target(data_path)
    df = handle_outliers(df)
    df = apply_log_transform(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    preprocessor = fit_and_save_preprocessor(X_train)

    X_train_enc = transform(preprocessor, X_train)
    X_val_enc   = transform(preprocessor, X_val)
    X_test_enc  = transform(preprocessor, X_test)

    feature_names = get_feature_names(preprocessor)

    print(f'\nSplit sizes — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')
    print(f'Encoded feature count: {X_train_enc.shape[1]}')
    print(f'Features: {feature_names}')

    return (X_train_enc, X_val_enc, X_test_enc,
            y_train, y_val, y_test,
            feature_names, preprocessor)
