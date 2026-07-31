import streamlit as st


def render_sidebar():
    """Render the common sidebar for all pages."""

    with st.sidebar:

        st.title("🛡️ SentinelAI")

        st.caption("Enterprise Fraud Detection Platform")

        st.divider()

        st.subheader("🤖 Model")

        st.write("**Algorithm:** XGBoost")
        st.write("**Task:** Binary Classification")
        st.write("**Version:** 2.0")

        st.divider()

        st.subheader("⚙️ Technology Stack")

        st.markdown(
            """
- Python 3
- Streamlit
- FastAPI
- XGBoost
- Scikit-Learn
- Pandas
- Plotly
"""
        )

        st.divider()

        st.subheader("📈 Performance")

        st.metric("ROC-AUC", "0.99+")

        st.metric("Status", "Online")

        st.divider()

        st.subheader("👨‍💻 Developer")

        st.write("Samarth Vatsal")

        st.caption("AI/ML Engineer")

        st.divider()

        st.success("✅ System Healthy")