# Churn-Analysis-Prediction

```
Docker Compose
├── PostgreSQL          ← raw data + feature store
├── Apache Airflow      ← orchestrate pipeline
│   ├── DAG 1: ingest & clean raw data
│   ├── DAG 2: feature engineering
│   └── DAG 3: retrain model + save metrics
├── FastAPI             ← model as REST API
│   └── /predict endpoint
└── Jupyter / Python    ← EDA + SHAP analysis
```
