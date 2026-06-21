"""
Demo — Loan Approval Prediction
COEN 330 — Applied Machine Learning

Loads the best trained model (Gradient Boosting) and predicts the loan approval
outcome for a sample applicant. Run from the project root:

    python demo/demo.py
"""

import sys
import json
from pathlib import Path

import pandas as pd
import joblib

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / 'models'
MODEL_PATH   = MODELS_DIR / 'GradientBoosting.pkl'

# ── Sample applicant ─────────────────────────────────────────────────────────
# A borderline case — not an obvious approve or reject.
SAMPLE = {
    'person_age':                      29,
    'person_gender':                   'male',
    'person_education':                'Bachelor',
    'person_income':                   55000,
    'person_emp_exp':                  4,
    'person_home_ownership':           'RENT',
    'loan_amnt':                       12000,
    'loan_intent':                     'EDUCATION',
    'loan_int_rate':                   11.5,
    'loan_percent_income':             0.22,
    'cb_person_cred_hist_length':      5,
    'credit_score':                    640,
    'previous_loan_defaults_on_file':  'No',
}


def load_model(path: Path):
    if not path.exists():
        print(f'[ERROR] Model not found at {path}')
        print('        Run 03_model_training.ipynb first to generate the model file.')
        sys.exit(1)
    return joblib.load(path)


def predict(pipeline, applicant: dict) -> tuple[str, float]:
    """
    Run one applicant through the pipeline.
    Returns (decision, confidence_score).
    The pipeline accepts a raw DataFrame — no manual preprocessing needed.
    """
    X = pd.DataFrame([applicant])

    # Use predict_proba if available (all models except SVM by default)
    clf = pipeline.named_steps['clf']
    if hasattr(clf, 'predict_proba'):
        prob = pipeline.predict_proba(X)[0, 1]   # P(approved)
    else:
        score = pipeline.decision_function(X)[0]
        # Normalise decision function to a [0,1] range for display only
        prob = 1 / (1 + pow(2.718, -score))

    decision = 'APPROVED' if prob >= 0.5 else 'REJECTED'
    return decision, prob


def print_report(applicant: dict, decision: str, confidence: float):
    width = 52
    bar   = '─' * width

    print()
    print('┌' + bar + '┐')
    print('│{:^52}│'.format('LOAN APPLICATION — PREDICTION RESULT'))
    print('├' + bar + '┤')

    print('│  {:<30} {:<18}│'.format('Feature', 'Value'))
    print('│  ' + '─' * 48 + '  │')
    labels = {
        'person_age':                     'Age',
        'person_gender':                  'Gender',
        'person_education':               'Education',
        'person_income':                  'Annual Income',
        'person_emp_exp':                 'Employment Experience',
        'person_home_ownership':          'Home Ownership',
        'loan_amnt':                      'Loan Amount',
        'loan_intent':                    'Loan Intent',
        'loan_int_rate':                  'Interest Rate (%)',
        'loan_percent_income':            'Loan / Income Ratio',
        'cb_person_cred_hist_length':     'Credit History (yrs)',
        'credit_score':                   'Credit Score',
        'previous_loan_defaults_on_file': 'Previous Defaults',
    }
    for key, label in labels.items():
        value = applicant[key]
        if key == 'person_income':
            value = f'${value:,.0f}'
        elif key == 'loan_amnt':
            value = f'${value:,.0f}'
        elif key == 'loan_int_rate':
            value = f'{value}%'
        elif key == 'loan_percent_income':
            value = f'{value:.0%}'
        print('│  {:<30} {:<18}│'.format(label, str(value)))

    print('├' + bar + '┤')

    decision_str = f'  Decision   : {decision}'
    conf_str     = f'  Confidence : {confidence:.1%} probability of approval'
    print(f'│{decision_str:<52}│')
    print(f'│{conf_str:<52}│')

    print('├' + bar + '┤')
    if decision == 'APPROVED':
        note = '  The model predicts this applicant is likely'
        print(f'│{note:<52}│')
        note = '  to repay the loan.'
    else:
        note = '  The model predicts this applicant poses a'
        print(f'│{note:<52}│')
        note = '  higher risk of default.'
    print(f'│{note:<52}│')

    print('└' + bar + '┘')
    print()


def main():
    print(f'Loading model from: {MODEL_PATH}')
    pipeline = load_model(MODEL_PATH)

    decision, confidence = predict(pipeline, SAMPLE)
    print_report(SAMPLE, decision, confidence)


if __name__ == '__main__':
    main()
