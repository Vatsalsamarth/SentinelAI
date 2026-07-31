import streamlit as st
import plotly.express as px


def fraud_distribution_chart(df):
    fraud = (df["Prediction"] == 1).sum()
    legitimate = (df["Prediction"] == 0).sum()

    fig = px.pie(
        names=["Legitimate", "Fraud"],
        values=[legitimate, fraud],
        hole=0.65,
        title="Fraud Distribution",
    )

    st.plotly_chart(fig, use_container_width=True)


def fraud_probability_histogram(df):
    fig = px.histogram(
        df,
        x="Fraud Probability",
        nbins=40,
        title="Fraud Probability Distribution",
    )

    st.plotly_chart(fig, use_container_width=True)


def amount_vs_probability(df):
    fig = px.scatter(
        df,
        x="Amount",
        y="Fraud Probability",
        color="Prediction Label",
        title="Transaction Amount vs Fraud Probability",
    )

    st.plotly_chart(fig, use_container_width=True)


def top_risky_transactions(df, top_n=20):
    st.subheader(f"🚨 Top {top_n} Highest Risk Transactions")

    top = (
        df.sort_values(
            "Fraud Probability",
            ascending=False,
        )
        .head(top_n)
    )

    st.dataframe(top, use_container_width=True)