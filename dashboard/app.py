from pathlib import Path
import sys

# ---------------------------------------------------
# Add project root to Python path
# ---------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------
# Imports
# ---------------------------------------------------
import streamlit as st

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="SentinelAI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# Home Page
# ---------------------------------------------------
st.title("🛡️ SentinelAI")

st.subheader("Enterprise Credit Card Fraud Detection Platform")

st.markdown("---")

st.markdown(
    """
Welcome to **SentinelAI**.

This application demonstrates a production-style machine learning
pipeline for detecting fraudulent credit card transactions.

### Features

- 📊 Interactive Analytics Dashboard
- 📂 Batch CSV Prediction
- 🔍 Single Transaction Prediction
- 🤖 XGBoost Machine Learning Model
- ⚡ FastAPI Backend
- 📈 Fraud Probability Scoring
- 📥 Download Prediction Results

Use the navigation panel on the left to explore the application.
"""
)

st.info(
    """
👈 Select a page from the sidebar to begin.
"""
)

st.markdown("---")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Model", "XGBoost")

with c2:
    st.metric("Backend", "FastAPI")

with c3:
    st.metric("Frontend", "Streamlit")

st.markdown("---")

st.success("✅ SentinelAI is running successfully.")