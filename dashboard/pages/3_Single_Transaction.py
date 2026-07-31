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

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="Single Transaction",
    page_icon="🔍",
    layout="wide"
)

render_sidebar()

st.title("🔍 Single Transaction Prediction")

st.markdown(
    """
Predict whether an individual transaction is fraudulent using the trained
SentinelAI XGBoost model.
"""
)

st.divider()

predictor = FraudPredictor()

# ---------------------------------------------------
# Transaction Input
# ---------------------------------------------------
st.subheader("Transaction Features")

values = {}

tab1, tab2, tab3 = st.tabs(
    [
        "Basic Information",
        "Principal Components (V1-V14)",
        "Principal Components (V15-V28)"
    ]
)

# ---------------- Basic ----------------

with tab1:

    values["Time"] = st.number_input(
        "Time",
        value=0.0
    )

    values["Amount"] = st.number_input(
        "Amount",
        value=0.0,
        min_value=0.0
    )

# ---------------- V1-V14 ----------------

with tab2:

    c1, c2