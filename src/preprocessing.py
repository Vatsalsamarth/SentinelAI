"""
SentinelAI Preprocessing Module

Handles:
- Loading the dataset
- Splitting train/test sets
- Scaling features
- Applying SMOTE
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    RAW_DATA_PATH,
    TEST_SIZE,
    RANDOM_STATE,
)


def load_dataset(path=RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the fraud detection dataset.

    Args:
        path: Path to the CSV file.

    Returns:
        Loaded pandas DataFrame.
    """
    return pd.read_csv(path)


def split_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split dataset into train and test sets.
    """

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
):
    """
    Scale Time and Amount using StandardScaler.
    """

    scaler = StandardScaler()

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[["Time", "Amount"]] = scaler.fit_transform(
        X_train[["Time", "Amount"]]
    )

    X_test[["Time", "Amount"]] = scaler.transform(
        X_test[["Time", "Amount"]]
    )

    return X_train, X_test, scaler


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
):
    """
    Balance the training data using SMOTE.
    """

    smote = SMOTE(random_state=RANDOM_STATE)

    X_resampled, y_resampled = smote.fit_resample(
        X_train,
        y_train,
    )

    return X_resampled, y_resampled