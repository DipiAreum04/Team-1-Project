# Loan Approval Prediction Model

**Team:** 1
**Course:** COEN 330 - Applied Machine Learning (Summer 2026)  
**Task:** Binary classification: predict whether a loan application will be **approved** (`loan_status = 1`) or **rejected** (`loan_status = 0`).

This README explains how to set up the environment, prepare the data, run the pipeline, train the models and reproduce the results.

---

## Project overview

We train and compare **five** classifiers on historical loan-approval decisions:

1. Logistic Regression (baseline model)  
2. SVM (RBF-kernel)  
3. Gaussian Naive Bayes  
4. Random Forest  
5. Gradient Boosting (`HistGradientBoostingClassifier`)

**Primary tuning metric:** F1-score (minority class = approved, ~22%).  
**Validation:** 5-fold stratified Cross-Validation on the **training set only**; a separated **20% test set** is used once in evaluation.  
**Leakage control:** preprocessing (scaling, encoding) lives **inside** each sklearn `Pipeline`, so it refits on each Cross-Validation fold.
**feature-set experiments** compare models with/without `previous_loan_defaults_on_file` feature and with two engineered ratio features.

---

## Repository structure

```text
COEN330-Machine-Learning-Project/
├── README.md
├── requirements.txt
├── example.json                # sample applicant for demo/predict.py
├── data/
│   ├── data_link.txt           # dataset URL and access notes
│   └── raw/
│       └── dataset.csv         # raw data
├── notebooks/
│   ├── 01_eda.ipynb            # exploratory data analysis
│   ├── 02_preprocessing.ipynb  # split, fit preprocessor, save encoded features
│   ├── 03_model_training.ipynb # train/tune 5 models + feature-set experiments
│   └── 04_evaluation.ipynb     # test-set metrics, thresholds, error analysis
├── src/
│   ├── utils.py                # contains global paths and seed used across the repository
│   ├── preprocessing.py        # Helper File: load, split, encode, feature engineering
│   ├── train.py                # Helper File: CV tuning, Training Session, feature-set helpers
│   └── evaluate.py             # Helper File: metrics, plots, error analysis helpers
├── models/                     # saved pipelines (.pkl) generated from training
├── results/
│   ├── cv_results_primary.csv
│   ├── demo_model_feature_comparison.csv
│   ├── demo_previous_default_counterfactual.csv
│   ├── test_results.csv
│   ├── feature_set_comparison.csv
│   ├── feature_set_test_results.csv
│   └── plots/                 # figures, graphs and plots
├── demo/
│   ├── demo.ipynb             # interactive prediction demo (with vs without prev. default)
│   └── predict.py             # command-line prediction script
└── report/
    └── final_report.pdf       # final report
```

---

## 1. Environment setup

**Requirements:** 
- Python 3.10+ recommended
- Jupyter

*If Python is not installed/version is too old, install Python from https://www.python.org/downloads/*

```bash
# Check if you have Python installed and its installed version
# enter the following script in shell/terminal
python --version

# from the project root
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
jupyter lab
```

Open notebooks from the `notebooks/` folder or launch Jupyter from the project root.

---

## 2. Dataset

| Field | Value |
|-------|--------|
| **File** | `data/raw/dataset.csv` |
| **Rows** | 45,000 |
| **Columns** | 14 (13 features + target `loan_status`) |
| **Target** | `loan_status`: 1 = approved, 0 = rejected |
| **Class balance** | ~78% rejected, ~22% approved |

**Features (summary):** demographics (`person_age`, `person_gender`, `person_education`, …), income, employment years, loan terms (`loan_amnt`, `loan_int_rate`, `loan_intent`, …), credit history (`credit_score`, `cb_person_cred_hist_length`), and `previous_loan_defaults_on_file`.

**How to obtain the data**

1. See **`data/data_link.txt`** for the dataset URL and source.  

2. Download the CSV and save it as:

   ```text
   data/raw/dataset.csv
   ```

3. Do **not** rename columns. The preprocessing code expects the schema shown in `01_eda.ipynb`.

---

## 3. How to run the pipeline (reproduce main results)

Run the notebooks **in order**. Each notebook’s first code cell adds `src/` to the path.

| Step | Notebook | What it produces |
|------|----------|------------------|
| 1 | `notebooks/01_eda.ipynb` | EDA plots in `results/plots/` (distributions, correlations, approval rates) |
| 2 | `notebooks/02_preprocessing.ipynb` | Encoded train/test arrays, `models/preprocessor.pkl` |
| 3 | `notebooks/03_model_training.ipynb` | Tuned pipelines in `models/*.pkl`, `results/cv_results_primary.csv` |
| 4 | `notebooks/04_evaluation.ipynb` | `results/test_results.csv`, threshold plots, confusion matrices, error analysis |

### 3.1 Model training (`03_model_training.ipynb`)

- **Sections 1–5:** train and tune each of the five models (expect 10-15 minutes in total; SVM is the slowest ~ 5-7 mins).  
- **Section 6:** comparison table sorted by Cross-Validation F1.  
- **Section 7:** `session.save()` → writes `models/LogisticRegression.pkl`, `SVM_RBF.pkl`, …  
- **Section 8:** Feature-set Experiments  
  - **8.1** — without `previous_loan_defaults_on_file`  
  - **8.2** — with engineered ratio features (with & without prev. default)  
  - **8.3** — copies best demo pipelines to:
    - `models/best_demo_model_with_prevdef.pkl`
    - `models/best_demo_model_without_prevdef.pkl`
    - `models/best_demo_model_engineered.pkl`
    - `models/best_demo_model_metadata.json`

**Tip:** After editing `src/train.py`, re-run the **Setup** cell (it reloads the module).

### 3.2 Evaluation (`04_evaluation.ipynb`)

Requires the five `.pkl` files from step 3 and the same train/test split (fixed seed `42` in `src/utils.py`).

Main outputs:

- **`results/test_results.csv`**: test accuracy, precision, recall, F1, PR-AUC per model (after threshold tuning)  
- **`results/plots/confusion_matrices_all.png`**: all models  
- **`results/plots/confusion_matrix_best_model.png`**: primary model (highest performance on test-set F1-score)  
- **`results/plots/pr_curves_all.png`**: precision–recall curves  
- Per-model threshold and importance plots under `results/plots/`

### 3.3 Demo (`demo/demo.ipynb`)

Compares predictions **with** vs **without** the `previous_loan_defaults_on_file` feature on sample applicant profiles.

**Prerequisites** (after 03_model_training notebook section 8.3):

- `best_demo_model_with_prevdef.pkl`
- `best_demo_model_without_prevdef.pkl`

Run all cells top-to-bottom. Summary plots are saved under `results/plots/`.

### 3.4 Command-line demo (`demo/predict.py`)

A CLI alternative to the notebook demo. Loads the saved **Gradient Boosting** pipeline and predicts approval for new applicants. The pipeline includes preprocessing, so only pass **raw applicant fields** (no manual encoding or scaling).

**Prerequisites** (after training + evaluation):

| File | Purpose |
|------|---------|
| `models/GradientBoosting.pkl` | Default model (with `previous_loan_defaults_on_file`) |
| `models/GradientBoosting_WITHOUT_prevdef.pkl` | Optional; use with `--without-prevdef` |
| `results/test_results.csv` | F1-tuned threshold for the default model |
| `results/feature_set_test_results.csv` | Threshold for the without-prevdef model |

If a threshold CSV is missing, the script uses default threshold = `0.50`.

**Usage** (from the project root, with the virtual environment activated):

```bash
# Predicts on built-in sample applicants (two profiles)
python demo/predict.py

# Predicts on a single applicant given as a JSON array
python demo/predict.py --json path/to/applicant.json

# Predicts for a batch of applicants and writes the result as <file>_predictions.csv in the mentioned path
python demo/predict.py --csv path/to/applicants.csv

# Uses the model trained without previous_loan_defaults_on_file
python demo/predict.py --without-prevdef

# Overrides the decision threshold
python demo/predict.py --threshold 0.30
```

**Input columns**:

`person_age`, `person_gender`, `person_education`, `person_income`, `person_emp_exp`, `person_home_ownership`, `loan_amnt`, `loan_intent`, `loan_int_rate`, `cb_person_cred_hist_length`, `credit_score`, `previous_loan_defaults_on_file`

`loan_percent_income` is optional; if omitted, it is computed as `loan_amnt / person_income`.

**Sample file:** `example.json` at the project root is a ready-to-run test case (25-year-old applicant, doctorate, no prior defaults). Same field layout as the training data, so open that file or copy its structure for your own inputs.

To run the sample file from project root:
```bash
python demo\predict.py --json example.json
```

**Output:** for each applicant, prints `APPROVED` or `REJECTED` and `P(approved)`. CSV batch mode adds `approval_probability` and `decision` columns to the output file.

---

## 4. Key results (Reference)

After a full run, typical **test-set F1** ranking:

| Model | Notes |
|-------|--------|
| Gradient Boosting | Usually strongest overall F1 / PR-AUC |
| Random Forest | Close second; good precision |
| SVM (RBF) | Strong but slower to train |
| Logistic Regression | Interpretable baseline |
| Gaussian Naive Bayes | High recall, lower precision |

**Primary model selection:** highest **test-set F1-score** after out-of-fold threshold tuning (`04_evaluation.ipynb`, section 4).

Confusion-matrix breakdown (TP / TN / FP / FN) for the primary model is in **section 8** of the evaluation notebook.

---

## 5. Reproducibility

- **Random seed:** `SEED = 42` in `src/utils.py` (split, CV, and models).  
- **Train/test split:** 80/20, stratified on `loan_status`, same split in training and evaluation notebooks.  
- **Paths:** resolved from `src/utils.py`, so notebooks work from `notebooks/` or project root.

---

## 6. Troubleshooting

| Issue | Fix |
|-------|-----|
| `FileNotFoundError: dataset.csv` | Place CSV in `data/raw/dataset.csv` |
| `FileNotFoundError: *.pkl` | Run `03_model_training.ipynb` through section 7 and 8 |
| `predict.py` threshold / model errors | Run `04_evaluation.ipynb` first; ensure `GradientBoosting.pkl` exists in `models/` |
| Stale imports after editing `src/` | Re-run Setup cell (`importlib.reload(train)` in training notebook) |
| SVM / feature-set cells very slow | Normal on ~36k rows, wait for completion or reduce grid in notebook |
