import streamlit as st


def display_metrics(total, fraud, legitimate, fraud_rate):
    """
    Display dashboard KPI cards.
    """

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            label="📄 Total Transactions",
            value=f"{total:,}"
        )

    with c2:
        st.metric(
            label="🚨 Fraud Detected",
            value=f"{fraud:,}"
        )

    with c3:
        st.metric(
            label="✅ Legitimate",
            value=f"{legitimate:,}"
        )

    with c4:
        st.metric(
            label="📊 Fraud Rate",
            value=f"{fraud_rate:.4f}%"
        )


def display_model_metrics(
    accuracy=None,
    precision=None,
    recall=None,
    f1_score=None,
    roc_auc=None,
):
    """
    Display model evaluation metrics.
    """

    cols = st.columns(5)

    metrics = [
        ("Accuracy", accuracy),
        ("Precision", precision),
        ("Recall", recall),
        ("F1 Score", f1_score),
        ("ROC-AUC", roc_auc),
    ]

    for col, (label, value) in zip(cols, metrics):
        with col:
            if value is None:
                st.metric(label, "--")
            else:
                st.metric(label, f"{value:.4f}")