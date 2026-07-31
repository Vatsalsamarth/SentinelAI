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

from dashboard.components.sidebar import render_sidebar

st.set_page_config(
    page_title="Model Information",
    page_icon="🤖",
    layout="wide"
)

render_sidebar()

st.title("🤖 Model Information")

st.markdown(
    """
This page summarizes the machine learning model,
training configuration, and deployment architecture
used in SentinelAI.
"""
)

st.divider()

# ---------------------------------------------------
# Model Summary
# ---------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    st.subheader("Model Details")

    st.write("**Algorithm:** XGBoost Classifier")
    st.write("**Problem Type:** Binary Classification")
    st.write("**Target Variable:** Class")
    st.write("**Features:** 30")
    st.write("**Training Dataset:** Credit Card Fraud Detection")

with col2:

    st.subheader("Deployment")

    st.write("Frontend : Streamlit")
    st.write("Backend : FastAPI")
    st.write("Model Storage : Joblib")
    st.write("API : REST")
    st.write("Prediction : Real-Time")

st.divider()

# ---------------------------------------------------
# Performance
# ---------------------------------------------------

st.subheader("Model Performance")

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric("Accuracy", "99.95%")
m2.metric("Precision", "97.80%")
m3.metric("Recall", "91.60%")
m4.metric("F1 Score", "94.60%")
m5.metric("ROC-AUC", "