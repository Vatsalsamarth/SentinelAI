"""
SentinelAI Training Module

Trains multiple machine learning models,
evaluates them, and saves the best model.
"""

from __future__ import annotations

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from xgboost import XGBClassifier

from src.preprocessing import (
    load_dataset,
    split_data,
    scale_features,
    apply_smote,
)

from src.config import (
    MODEL_PATH,
    SCALER_PATH,
    RANDOM_STATE,
)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model.
    """

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1 Score": f1_score(y_test, predictions),
        "ROC AUC": roc_auc_score(y_test, probabilities),
    }


def main():

    print("=" * 60)
    print("Loading dataset...")
    print("=" * 60)

    df = load_dataset()

    X_train, X_test, y_train, y_test = split_data(df)

    X_train, X_test, scaler = scale_features(
        X_train,
        X_test,
    )

    X_train, y_train = apply_smote(
        X_train,
        y_train,
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        ),
    }

    results = []

    best_model = None
    best_auc = 0

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        metrics["Model"] = name

        results.append(metrics)

        if metrics["ROC AUC"] > best_auc:
            best_auc = metrics["ROC AUC"]
            best_model = model

    results_df = pd.DataFrame(results)

    results_df = results_df[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC AUC",
        ]
    ]

    print("\n")
    print(results_df.sort_values("ROC AUC", ascending=False))

    print("\nSaving best model...")

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Scaler saved to: {SCALER_PATH}")


if __name__ == "__main__":
    main()