"""
SentinelAI Configuration

This module stores all project-wide constants and paths.
Every other module imports configuration from here.
"""

from pathlib import Path

# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

ASSETS_DIR = PROJECT_ROOT / "assets"

# ==========================================================
# DATASET
# ==========================================================

DATASET_NAME = "creditcard.csv"

RAW_DATA_PATH = RAW_DATA_DIR / DATASET_NAME

# ==========================================================
# MODEL FILES
# ==========================================================

MODEL_FILENAME = "fraud_detector.pkl"

SCALER_FILENAME = "scaler.pkl"

MODEL_PATH = MODELS_DIR / MODEL_FILENAME

SCALER_PATH = MODELS_DIR / SCALER_FILENAME

# ==========================================================
# TARGET COLUMN
# ==========================================================

TARGET_COLUMN = "Class"

# ==========================================================
# FEATURES
# ==========================================================

FEATURE_COLUMNS = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount",
]

SCALE_COLUMNS = [
    "Time",
    "Amount",
]

# ==========================================================
# TRAINING
# ==========================================================

TEST_SIZE = 0.20

RANDOM_STATE = 42

# ==========================================================
# RANDOM FOREST
# ==========================================================

RF_ESTIMATORS = 200

# ==========================================================
# XGBOOST
# ==========================================================

XGB_ESTIMATORS = 300

XGB_MAX_DEPTH = 6

XGB_LEARNING_RATE = 0.05

XGB_EVAL_METRIC = "logloss"