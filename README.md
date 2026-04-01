# Churn Analysis & Prediction - MLOps Pipeline

An end-to-end ML pipeline for predicting customer churn using **XGBoost**, orchestrated with **Apache Airflow**, containerized with **Docker Compose**, and backed by **PostgreSQL**.

---

### Key Insights from Top 5 Features

| Feature | Insight |
|---------|---------|
| **tenure** | New customers (low tenure) are significantly more likely to churn |
| **Contract_Two year** | Long-term contracts strongly reduce churn probability |
| **InternetService_Fiber optic** | Fiber customers churn slightly more, likely due to higher costs |
| **MonthlyCharges** | Higher monthly charges correlate with increased churn risk |
| **TotalCharges** | Reflects cumulative relationship value — lower total = higher churn risk |
---

## Architecture

```
Docker Compose
├── PostgreSQL              ← raw data storage + model metrics
├── Apache Airflow          ← pipeline orchestration
│   ├── DAG 1 — Ingest
│   │   ├── t1: Download dataset from Kaggle API
│   │   └── t2: Load raw data into PostgreSQL
│   └── DAG 2 — Train & Analyze
│       └── t1: Feature Engineering → XGBoost Training → Metrics + Feature Importance → Save to DB
└── Jupyter / Python        ← EDA, model comparison, metrics visualization
```

---

## Getting Started

### Prerequisites

- [Docker](https://www.docker.com/) & Docker Compose
- Kaggle API credentials (`~/.kaggle/kaggle.json`)

### Run

```bash
git clone https://github.com/your-username/Churn-Analysis-Prediction.git
cd Churn-Analysis-Prediction

# Set environment variables
cp .env.example .env

chmod -R 777 ./models ./scripts ./dags

# Start all services
docker compose up -d #--build 

# Access Airflow UI
open http://localhost:8080
```

### Trigger Pipelines

1. In the Airflow UI, trigger **`dag_01_ingest_churn`** — downloads and loads data into PostgreSQL
2. Trigger **`dag_02_train_and_report`** — trains the model and saves metrics

IMPORTANT:

-   You will not be able to run correctly the code blocks in **metrics_eval.ipynb** if you don't follow the steps (triggering the DAGs in the correct order)

---

## ML Pipeline (`pipeline.py`)

| Step | Description |
|------|-------------|
| **Data Fetch** | Reads `raw_churn` table from PostgreSQL via SQLAlchemy |
| **Feature Engineering** | Creates `tenure_group` (new / medium / old) using `pd.cut` |
| **Encoding** | One-hot encodes all categorical features with `get_dummies` |
| **Scaling** | Normalizes numeric columns (`tenure`, `MonthlyCharges`, `TotalCharges`) with MinMaxScaler |
| **Training** | XGBoost classifier (1000 estimators, lr=0.01, max_depth=6) |
| **Evaluation** | Accuracy, F1, Precision, Recall saved to `model_metrics` table |
| **Feature Importance** | Top 5 features by gain saved as JSON in `model_metrics` |

---

## Exploratory Data Analysis

The EDA notebook (`scripts/EDA.ipynb`) covers:

- Class distribution (churn vs. no churn)
- Correlation analysis and feature distributions
- Comparison of 3 ML models before selecting XGBoost
- SHAP-based feature importance interpretation

## Database Schema

### `raw_churn`
Raw customer data ingested from Kaggle (Telco Customer Churn dataset).

### `model_metrics`
Populated after each DAG 2 run:

| Column | Type | Description |
|--------|------|-------------|
| `run_date` | timestamp | When the pipeline ran |
| `accuracy` | float | Test set accuracy |
| `f1_score` | float | F1 score |
| `precision` | float | Precision |
| `recall` | float | Recall |
| `model_path` | text | Path to saved `.joblib` model |
| `top_features` | json | Top 5 features with importance scores |

---

## Project Structure

```
.
├── README.md
├── airflow
│   └── dags
│       ├── dag_ingest.py
│       └── dag_train.py
├── data
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── docker-compose.yaml
├── models
│   ├── logistic_model.pkl
│   ├── model_churn.joblib # saved from pipeline
│   ├── random_forest_model.pkl
│   └── xgb_model.pkl
├── postgres
│   └── data  [error opening dir]
├── postgres_conn.sh
├── scripts
│   ├── EDA.ipynb
│   ├── __pycache__
│   │   └── pipeline.cpython-312.pyc
│   ├── metrics_eval.ipynb
│   ├── pipeline.py
│   └── requirements.txt
├── secrets
│   └── kaggle.json
└── test_db.sh
```

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-orange)
![Airflow](https://img.shields.io/badge/Apache%20Airflow-2.x-017CEE?logo=apacheairflow)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-336791?logo=postgresql)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)

---
