from pathlib import Path
import sys

# ---------------------------------------------------
# Add Project Root
# ---------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------
# Imports
# ---------------------------------------------------
import streamlit as st
import pandas as pd

from src.predictor import FraudPredictor

from dashboard.components.sidebar import render_sidebar
from dashboard.components.metrics import display_metrics

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="Batch Prediction",
    page_icon="📂",
    layout="wide",
)

render_sidebar()

st.title("📂 Batch Fraud Prediction")

st.markdown(
    """
Upload a CSV containing transactions.

SentinelAI will analyze every transaction and generate fraud predictions.
"""
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"],
)

if uploaded_file is None:

    st.info("👆 Upload a CSV file to begin.")

    st.stop()

# ---------------------------------------------------
# Read CSV
# ---------------------------------------------------
try:

    df = pd.read_csv(uploaded_file)

except Exception as e:

    st.error(f"Unable to read file.\n\n{e}")

    st.stop()

st.subheader("Uploaded Dataset")

st.dataframe(df.head(), use_container_width=True)

st.write(f"Rows: **{len(df):,}**")

st.divider()

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------
predictor = FraudPredictor()

if st.button("🚀 Run Prediction", use_container_width=True):

    progress = st.progress(0)

    status = st.empty()

    status.info("Loading model...")
    progress.progress(20)

    status.info("Running inference...")
    progress.progress(60)

    results = predictor.predict(df)

    progress.progress(90)

    status.info("Generating analytics...")

    total = len(results)
    fraud = (results["Prediction"] == 1).sum()
    legitimate = (results["Prediction"] == 0).sum()
    fraud_rate = fraud / total * 100

    progress.progress(100)

    status.success("Prediction completed successfully!")

    st.divider()

    display_metrics(
        total,
        fraud,
        legitimate,
        fraud_rate,
    )

    st.divider()

    st.subheader("Prediction Results")

    st.dataframe(
        results,
        use_container_width=True,
    )

    st.divider()

    st.subheader("🚨 High Risk Transactions")

    risky = results.sort_values(
        "Fraud Probability",
        ascending=False,
    ).head(10)

    st.dataframe(
        risky,
        use_container_width=True,
    )

    csv = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Predictions",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )