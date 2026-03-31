import datetime
import pandas as pd
import joblib
from xgboost import XGBClassifier
import shap
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import json
from airflow.providers.postgres.hooks.postgres import PostgresHook
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

MODEL_PATH = "/opt/airflow/scripts/model_churn.joblib"

def train_and_analyze():
    """
        Model Pipeline that does:
        1. Fetches data from PostgreSQL
        2. Tranforms data (Feature Engineering, Encoding, NAs, Scaling, etc.)
            - OBS: This is based on the jupyter notebook in scripts/EDA.ipynb
        3. Trains a XGBoost model on the resulting data from 2nd step
        4. Saves the model in MODEL_PATH
        5. Calculates metrics like:
            - "run_date": datetime.now()
            - "accuracy"
            - "f1_score"
            - "precision"
            - "recall"
        6. SHAP Processing for getting the "Top 5 Feature Importance"
        7. Saves the results (metrics and SHAP) for further investigation
    """
    
    hook = PostgresHook(postgres_conn_id="postgres_default")
    engine = hook.get_sqlalchemy_engine()
    
    try:
        df = pd.read_sql("SELECT * FROM raw_churn", engine)
        if df.empty:
            raise ValueError("raw_churn is empty")
    except Exception as e:
        print(f"Error reading from postgres: {e}")
        raise
    
    try:
        df['tenure_group'] = pd.cut(df['tenure'],
                                    bins=[0, 10, 50, 72],
                                    labels=['new', 'medium', 'old'])
        
        numeric_col = ['tenure', 'MonthlyCharges', 'TotalCharges']
        categorical_col = [col for col in df.columns if col not in numeric_col + ["Churn"] + ["customerID"]]
        
        df_encoded = pd.get_dummies(df, columns=categorical_col, drop_first=True)
        
        df_encoded['TotalCharges'] = pd.to_numeric(df_encoded['TotalCharges'], errors="coerce")
        df_encoded.dropna(subset=['TotalCharges'], inplace=True)

        scaler = MinMaxScaler()
        df_encoded[numeric_col] = scaler.fit_transform(df_encoded[numeric_col])
        df_encoded["Churn"] = df_encoded["Churn"].map({"Yes": 1, "No": 0})
        
        X = df_encoded.drop(["Churn", "customerID"], axis=1)
        y = df_encoded["Churn"]
    except KeyError as e:
        print(f"Column missing: {e}")
        raise
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7, stratify=y)
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    
    xgb = XGBClassifier(
        random_state=7,
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss'
    )

    xgb.fit(X_train, y_train)
    joblib.dump(xgb, MODEL_PATH)
    print(f"Model saved at: {MODEL_PATH}")
    
    try:
        explainer = shap.TreeExplainer(xgb)
        shap_values = explainer.shap_values(X)
    except Exception as e:
        print(f"Error SHAP: {e}")
        shap_values = None
    
    y_pred = xgb.predict(X_test)
    
    metrics = {
        "run_date": datetime.now(),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "model_path": MODEL_PATH
    }

    ## SHAP Processing
    # Top 5 Feature Importance
    feature_names = X.columns
    
    if isinstance(shap_values, list):
        vals = np.abs(shap_values[1]).mean(0)
    else:
        vals = np.abs(shap_values).mean(0)
    
    importance_df = pd.DataFrame(list(zip(feature_names, vals)), columns=['feature', 'importance'])
    top_features = importance_df.sort_values(by='importance', ascending=False).head(5)
    metrics["top_features"] = json.dumps(top_features.set_index('feature')['importance'].to_dict())

    df_metrics = pd.DataFrame([metrics])
    
    df_metrics.to_sql(
        name="model_metrics",
        con=engine,
        if_exists="append",
        index=False
    )
    
    print("Metrics and SHAP analysis have been saved in 'model_metrics'")
    print("SHAP finished")
