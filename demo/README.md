# Loan Approval Prediction - Demo

This `/demo` folder contains two ways to run live predictions on sample loan applicants using the trained **Gradient Boosting** model:

| File | Description |
|------|-------------|
| `demo.ipynb` | An interactive notebook that compares predictions **with** and **without** `previous_loan_defaults_on_file` on sample profiles |
| `predict.py` | A command-line script that scores one applicant, a JSON batch, or a CSV file |

Both tools load saved sklearn pipelines from `models/`.

For more information on the project, see the main [README.md](../README.md) file located at the root of the project.

## Prerequisites

Run the training and evaluation notebooks first using the `Run All` command found at the top of each notebook.

| Step | Notebook | Required outputs |
|------|----------|------------------|
| Training | `notebooks/03_model_training.ipynb` (sections 1–7 and feature-set section 8) | `models/GradientBoosting.pkl`, `models/GradientBoosting_WITHOUT_prevdef.pkl` |
| Evaluation | `notebooks/04_evaluation.ipynb` | `results/test_results.csv`, `results/feature_set_test_results.csv` (F1-tuned thresholds) |

---

If these steps are not complete, errors will most likely follow through when running the jupyter notebook.

## Environment setup

From the **project root** `./`:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

For the notebook demo, ensure that Jupyter notebook is installed (included in `requirements.txt`):

```bash
jupyter lab
```

Open `demo/demo.ipynb` from the project root or launch Jupyter and navigate to the `demo/` folder.

---

## Interactive Notebook Demo (`demo.ipynb`)

The notebook loads two Gradient Boosting pipelines trained on the same algorithm but different feature sets:

1. **With** `previous_loan_defaults_on_file` → `models/GradientBoosting.pkl`
2. **Without** that feature → `models/GradientBoosting_WITHOUT_prevdef.pkl`

Summary: Execute all cells top-to-bottom.

| Step | Description |
|------|--------------|
| 1 | Load both pipelines and print their steps |
| 2 | Define three sample applicant profiles (A, B, C), their specific values can be modified |
| 3 | Score each profile with both models and compare probabilities and predictions |
| 4 | Bar chart shows  approval probability with vs without the previous-default feature |
| 5 | Counterfactual test flips `previous_loan_defaults_on_file` from `"No"` to `"Yes"` and observes which model changes |

**CSV and PNG outputs saved under `results/`:**

- `results/demo_model_feature_comparison.csv` - side-by-side comparison table
- `results/plots/demo_with_vs_without_previous_defaults.png` - visual comparison
- `results/demo_previous_default_counterfactual.csv` - counterfactual flip results

The comparison shows whether removing the previous-default feature changes estimated approval probability or the final approve/reject decision for the same applicant.

---

## Command-line demo (`predict.py`)

Score new applicants from the terminal. The default model is `GradientBoosting.pkl` (with `previous_loan_defaults_on_file`).

**Run from the project root** with the virtual environment activated:

```bash
# Built-in sample applicants (two profiles)
python demo/predict.py

# Single applicant or JSON array
python demo/predict.py --json path/to/applicant.json

# Batch CSV (writes <file>_predictions.csv next to the input)
python demo/predict.py --csv path/to/applicants.csv

# Model trained without previous_loan_defaults_on_file
python demo/predict.py --without-prevdef

# Override the F1-tuned decision threshold
python demo/predict.py --threshold 0.30
```

**Windows** (same commands, backslashes also work):

```bash
python demo\predict.py --json example.json
```

### Sample input

`example.json` at the project root is a ready-to-run test case:

```bash
python demo/predict.py --json example.json
```

### Input columns

Required fields (same schema as `data/raw/dataset.csv`):

`person_age`, `person_gender`, `person_education`, `person_income`, `person_emp_exp`, `person_home_ownership`, `loan_amnt`, `loan_intent`, `loan_int_rate`, `cb_person_cred_hist_length`, `credit_score`, `previous_loan_defaults_on_file`

Optional:

- `loan_percent_income` - if omitted, computed as `loan_amnt / person_income`

When using `--without-prevdef`, the script drops `previous_loan_defaults_on_file` before prediction to match the reduced feature set.

### Output

For each applicant, the script prints:

- `APPROVED` or `REJECTED`
- `P(approved)` - estimated probability of approval

In CSV batch mode, the output file includes the original columns plus `approval_probability` and `decision`.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `FileNotFoundError: *.pkl` | Run `notebooks/03_model_training.ipynb` through section 7 and feature-set section 8 |
| Threshold or model errors in `predict.py` | Run `notebooks/04_evaluation.ipynb`; confirm `GradientBoosting.pkl` exists in `models/` |
| Notebook cannot find `src/` | Run Jupyter from the project root, or ensure the first code cell (which adds `../src` to the path) has been executed |
| Unexpected decisions | Check which model is loaded (`--without-prevdef` vs default) and the threshold printed at runtime |

---