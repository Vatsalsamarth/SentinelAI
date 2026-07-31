# 🛡️ SentinelAI

> AI Powered Credit Card Fraud Detection Platform

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Production-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

# Overview

SentinelAI is an end-to-end Machine Learning platform that detects fraudulent credit card transactions in real time.

The project demonstrates the complete ML lifecycle:

- Exploratory Data Analysis
- Feature Engineering
- Data Preprocessing
- Handling Imbalanced Data (SMOTE)
- XGBoost Model Training
- Explainable AI using SHAP
- FastAPI REST API
- Streamlit Dashboard
- Docker Deployment

---

# Dataset

Dataset:

Credit Card Fraud Detection Dataset

Rows:

284,807

Columns:

31

Fraud Cases:

492

Fraud Percentage:

0.1727%

---

# Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- SHAP
- FastAPI
- Streamlit
- Docker
- Joblib

---

# Project Structure

```text
SentinelAI/

dashboard/
models/
src/
notebooks/
tests/

README.md
Dockerfile
requirements.txt
```

---

# Machine Learning Pipeline

1. Load Dataset

2. Data Cleaning

3. Feature Engineering

4. Standard Scaling

5. SMOTE

6. XGBoost Training

7. Model Evaluation

8. SHAP Explainability

9. Save Model

10. Deploy API

11. Dashboard

---

# Run Locally

Clone

```bash
git clone https://github.com/Vatsalsamarth/SentinelAI.git
```

Install

```bash
pip install -r requirements.txt
```

Run API

```bash
uvicorn src.api.main:app --reload
```

Run Dashboard

```bash
streamlit run dashboard/app.py
```

---

# Future Improvements

- Kafka Streaming
- MLflow Tracking
- CI/CD
- Kubernetes
- Model Monitoring

---

# Author

Samarth Vatsal

GitHub

https://github.com/Vatsalsamarth