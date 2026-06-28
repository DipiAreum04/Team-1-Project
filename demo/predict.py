#!/usr/bin/env python3
r"""
Command-line demo for the Loan Approval Prediction project.

Loads the trained Gradient Boosting pipeline and predicts approval for new applicants.
The pipeline carries its own preprocessing, so applicants are passed as raw field dicts /
rows and no manual feature engineering is needed here.

Example usage:
    python demo\predict.py                          # predicts using the built-in sample applicants
    python demo\predict.py --json applicant.json    # predicts for one applicant object, or a JSON array
    python demo\predict.py --csv applicants.csv     # predicts for a batch of applicants and writes <file>_predictions.csv
    python demo\predict.py --without-prevdef        # use the model trained without prior-default history
    python demo\predict.py --threshold 0.30         # override the decision threshold
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

def _find_src():
    for d in Path(__file__).resolve().parents:
        if (d / "src" / "utils.py").exists():
            return d / "src"
    raise FileNotFoundError("Could not locate src/ above predict.py")

SRC_DIR = _find_src()
sys.path.insert(0, str(SRC_DIR))

import preprocessing 
from utils import MODELS_DIR, RESULTS_DIR

MODEL_FILES = {
    "with_prevdef": "GradientBoosting.pkl",
    "without_prevdef": "GradientBoosting_WITHOUT_prevdef.pkl",
}
PREV_DEFAULT_COLUMN = "previous_loan_defaults_on_file"
FALLBACK_THRESHOLD = 0.50

SAMPLE_APPLICANTS = [
    {
        "person_age": 29, 
        "person_gender": "male", 
        "person_education": "Bachelor",
        "person_income": 55000, 
        "person_emp_exp": 4, 
        "person_home_ownership": "RENT",
        "loan_amnt": 12000, 
        "loan_intent": "EDUCATION", 
        "loan_int_rate": 11.5,
        "cb_person_cred_hist_length": 5, 
        "credit_score": 640,
        "previous_loan_defaults_on_file": "No",
    },
    {
        "person_age": 27, 
        "person_gender": "female", 
        "person_education": "Doctorate",
        "person_income": 90000, 
        "person_emp_exp": 8, 
        "person_home_ownership": "RENT",
        "loan_amnt": 5000, 
        "loan_intent": "MEDICAL", 
        "loan_int_rate": 7.5,
        "cb_person_cred_hist_length": 9, 
        "credit_score": 810,
        "previous_loan_defaults_on_file": "Yes",
    },
]


def load_threshold(model_key: str) -> float:
    """F1-tuned threshold from 04_evaluation's result CSVs; defaults to 0.50 if not found."""
    try:
        if model_key == "with_prevdef":
            df = pd.read_csv(RESULTS_DIR / "test_results.csv")
            df = df[df["Model"] == "Gradient Boosting"]
        else:
            df = pd.read_csv(RESULTS_DIR / "feature_set_test_results.csv")
            df = df[(df["Model"] == "Gradient Boosting") &
                    (df["Feature Set"] == "WITHOUT_prevdef")]
        return float(df["Threshold"].iloc[0])
    except (FileNotFoundError, IndexError, KeyError):
        return FALLBACK_THRESHOLD


def predict(df: pd.DataFrame, model_key: str, threshold: float) -> pd.DataFrame:
    """Score applicants with the saved pipeline and apply the decision threshold."""
    pipeline = joblib.load(MODELS_DIR / MODEL_FILES[model_key])
    df = df.copy()

    # Match the training convention if the applicant didn't supply the ratio.
    if "loan_percent_income" not in df.columns:
        df["loan_percent_income"] = df["loan_amnt"] / df["person_income"]

    if model_key == "without_prevdef":
        df = df.drop(columns=[PREV_DEFAULT_COLUMN], errors="ignore")

    proba = pipeline.predict_proba(df)[:, 1]
    df["approval_probability"] = proba
    df["decision"] = ["APPROVED" if p >= threshold else "REJECTED" for p in proba]
    return df


def load_input(args) -> pd.DataFrame:
    if args.json:
        data = json.loads(Path(args.json).read_text())
        records = data if isinstance(data, list) else [data]
        return pd.DataFrame(records)
    if args.csv:
        return pd.read_csv(args.csv)
    print("No input given - using built-in sample applicants.\n")
    return pd.DataFrame(SAMPLE_APPLICANTS)


def main():
    p = argparse.ArgumentParser(description="Predict loan approval for new applicants.")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--json", help="JSON file: one applicant object or an array of them")
    source.add_argument("--csv", help="CSV file of applicants (batch mode)")
    p.add_argument("--without-prevdef", action="store_true",
                   help="use the model trained without prior-default history")
    p.add_argument("--threshold", type=float, default=None,
                   help="override the decision threshold (default: the F1-tuned value)")
    args = p.parse_args()

    model_key = "without_prevdef" if args.without_prevdef else "with_prevdef"
    threshold = args.threshold if args.threshold is not None else load_threshold(model_key)

    result = predict(load_input(args), model_key, threshold)

    print(f"Model    : {MODEL_FILES[model_key]}")
    print(f"Threshold: {threshold:.4f}\n")

    n = len(result)
    for i, row in result.head(20).iterrows():
        print(f"  Applicant {i + 1}: {row['decision']:9s}  P(approved) = {row['approval_probability']:.1%}")
    if n > 20:
        print(f"  ... ({n} applicants total)")
        print("\n" + result["decision"].value_counts().to_string())

    if args.csv:
        out_path = Path(args.csv).with_name(Path(args.csv).stem + "_predictions.csv")
        result.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
