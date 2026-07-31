"""
SentinelAI Prediction Module

Provides a reusable prediction interface for:
- FastAPI
- Streamlit
- Batch prediction
- Single transaction prediction
"""

from __future__ import annotations

import joblib
import pandas as pd

from src.config import (
    MODEL_PATH,
    SCALER_PATH,
    FEATURE_COLUMNS,
)


class FraudPredictor:
    """
    Wrapper around the trained fraud detection model.
    """

    def __init__(self):

        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)

    def preprocess(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Prepare incoming data before prediction.
        """

        df = df.copy()

        # Remove target column if supplied
        if "Class" in df.columns:
            df = df.drop(columns=["Class"])

        # Ensure feature order
        df = df[FEATURE_COLUMNS]

        # Scale Time and Amount
        df[["Time", "Amount"]] = self.scaler.transform(
            df[["Time", "Amount"]]
        )

        return df

    def predict(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Predict fraud for one or more transactions.
        """

        processed = self.preprocess(df)

        predictions = self.model.predict(processed)

        probabilities = self.model.predict_proba(processed)[:, 1]

        result = df.copy()

        result["Prediction"] = predictions

        result["Fraud Probability"] = probabilities

        result["Prediction Label"] = result["Prediction"].map(
            {
                0: "Legitimate",
                1: "Fraud",
            }
        )

        return result

    def predict_single(
        self,
        transaction: dict,
    ) -> dict:
        """
        Predict a single transaction.
        """

        df = pd.DataFrame([transaction])

        result = self.predict(df)

        return {
            "prediction": int(result.iloc[0]["Prediction"]),
            "label": result.iloc[0]["Prediction Label"],
            "fraud_probability": float(
                result.iloc[0]["Fraud Probability"]
            ),
        }