from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

app = FastAPI(title="Churn Prediction API")

model = joblib.load("/models/model_churn.joblib")

class CustomerData(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    Contract_One_year: int
    Contract_Two_year: int
    InternetService_Fiber_optic: int
    InternetService_No: int
    PaymentMethod_Electronic_check: int
    tenure_group_new: int
    tenure_group_old: int

@app.get("/")
def root():
    return {"status": "Churn Prediction API is running"}

@app.post("/predict")
def predict(customer: CustomerData):
    data = pd.DataFrame([customer.model_dump()])
    
    scaler = MinMaxScaler()
    numeric = ['tenure', 'MonthlyCharges', 'TotalCharges']
    data[numeric] = scaler.fit_transform(data[numeric])
    
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    return {
        "churn": bool(prediction),
        "churn_probability": round(float(probability), 4),
        "risk_level": "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"
    }
