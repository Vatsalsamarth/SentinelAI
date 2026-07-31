"""
SentinelAI FastAPI Application
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd

from src.predictor import FraudPredictor
from src.config import FEATURE_COLUMNS

app = FastAPI(
    title="SentinelAI Fraud Detection API",
    version="2.0.0",
    description="Production-ready Credit Card Fraud Detection API",
)

# ----------------------------------------------------
# Load predictor once at startup
# ----------------------------------------------------

predictor = FraudPredictor()


# ----------------------------------------------------
# Pydantic Model
# ----------------------------------------------------

class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


# ----------------------------------------------------
# Routes
# ----------------------------------------------------

@app.get("/")
def home():
    return {
        "project": "SentinelAI",
        "version": "2.0.0",
        "status": "Running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.get("/model_info")
def model_info():
    return {
        "model": predictor.model.__class__.__name__,
        "features": FEATURE_COLUMNS,
        "feature_count": len(FEATURE_COLUMNS),
    }


@app.post("/predict")
def predict(transaction: Transaction):

    df = pd.DataFrame([transaction.model_dump()])

    result = predictor.predict(df)

    return {
        "prediction": int(result.iloc[0]["Prediction"]),
        "label": result.iloc[0]["Prediction Label"],
        "fraud_probability": float(
            result.iloc[0]["Fraud Probability"]
        ),
    }