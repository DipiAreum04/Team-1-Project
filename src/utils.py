"""
Shared constants and paths for the Loan Approval Prediction project.
Defined ONCE here so the seed and folder locations do not drift apart
between teammates or between files.
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd

# One common seed used for reproducibility across the project
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Paths resolve relative to this file, so they work from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

# The dataset filename 
DATASET_FILE = "dataset.csv"

# Make sure output folders exist.
for _d in (DATA_PROCESSED, MODELS_DIR, RESULTS_DIR, PLOTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)