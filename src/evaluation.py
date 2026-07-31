"""
SentinelAI Evaluation Module

Provides reusable functions for evaluating trained models,
printing metrics, and generating plots.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Compute evaluation metrics for a trained model.
    """

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1 Score": f1_score(y_test, predictions),
        "ROC AUC": roc_auc_score(y_test, probabilities),
    }

    return metrics


def print_classification_report(model, X_test, y_test):
    """
    Print the sklearn classification report.
    """

    predictions = model.predict(X_test)

    print("\nClassification Report\n")
    print(classification_report(y_test, predictions))


def save_metrics(metrics: dict, output_path: str | Path):
    """
    Save metrics as a CSV file.
    """

    df = pd.DataFrame([metrics])
    df.to_csv(output_path, index=False)


def plot_confusion_matrix(model, X_test, y_test):
    """
    Display confusion matrix.
    """

    predictions = model.predict(X_test)

    cm = confusion_matrix(y_test, predictions)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot()

    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def plot_roc_curve(model, X_test, y_test):
    """
    Display ROC Curve.
    """

    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test,
    )

    plt.title("ROC Curve")
    plt.tight_layout()
    plt.show()


def plot_precision_recall(model, X_test, y_test):
    """
    Display Precision-Recall Curve.
    """

    PrecisionRecallDisplay.from_estimator(
        model,
        X_test,
        y_test,
    )

    plt.title("Precision