from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(title="Churn Prediction API")

model = joblib.load("/models/model_churn.joblib")
scaler = joblib.load("/models/scaler.joblib")

class CustomerData(BaseModel):
    # Numeric
    tenure: float
    MonthlyCharges: float
    TotalCharges: float

    # Binary encoded
    gender_Male: int
    SeniorCitizen_1: int
    Partner_Yes: int
    Dependents_Yes: int
    PhoneService_Yes: int
    PaperlessBilling_Yes: int

    # MultipleLines
    MultipleLines_No_phone_service: int = Field(alias="MultipleLines_No phone service")
    MultipleLines_Yes: int

    # InternetService
    InternetService_Fiber_optic: int = Field(alias="InternetService_Fiber optic")
    InternetService_No: int

    # OnlineSecurity
    OnlineSecurity_No_internet_service: int = Field(alias="OnlineSecurity_No internet service")
    OnlineSecurity_Yes: int

    # OnlineBackup
    OnlineBackup_No_internet_service: int = Field(alias="OnlineBackup_No internet service")
    OnlineBackup_Yes: int

    # DeviceProtection
    DeviceProtection_No_internet_service: int = Field(alias="DeviceProtection_No internet service")
    DeviceProtection_Yes: int

    # TechSupport
    TechSupport_No_internet_service: int = Field(alias="TechSupport_No internet service")
    TechSupport_Yes: int

    # StreamingTV
    StreamingTV_No_internet_service: int = Field(alias="StreamingTV_No internet service")
    StreamingTV_Yes: int

    # StreamingMovies
    StreamingMovies_No_internet_service: int = Field(alias="StreamingMovies_No internet service")
    StreamingMovies_Yes: int

    # Contract
    Contract_One_year: int = Field(alias="Contract_One year")
    Contract_Two_year: int = Field(alias="Contract_Two year")

    # PaymentMethod
    PaymentMethod_Credit_card_automatic: int = Field(alias="PaymentMethod_Credit card (automatic)")
    PaymentMethod_Electronic_check: int = Field(alias="PaymentMethod_Electronic check")
    PaymentMethod_Mailed_check: int = Field(alias="PaymentMethod_Mailed check")

    # tenure_group
    tenure_group_medium: int
    tenure_group_old: int

    model_config = {"populate_by_name": True} 


@app.get("/")
def root():
    return {"status": "Churn Prediction API is running"}


@app.post("/predict")
def predict(customer: CustomerData):
    data = pd.DataFrame([{
        "tenure": customer.tenure,
        "MonthlyCharges": customer.MonthlyCharges,
        "TotalCharges": customer.TotalCharges,
        "gender_Male": customer.gender_Male,
        "SeniorCitizen_1": customer.SeniorCitizen_1,
        "Partner_Yes": customer.Partner_Yes,
        "Dependents_Yes": customer.Dependents_Yes,
        "PhoneService_Yes": customer.PhoneService_Yes,
        "MultipleLines_No phone service": customer.MultipleLines_No_phone_service,
        "MultipleLines_Yes": customer.MultipleLines_Yes,
        "InternetService_Fiber optic": customer.InternetService_Fiber_optic,
        "InternetService_No": customer.InternetService_No,
        "OnlineSecurity_No internet service": customer.OnlineSecurity_No_internet_service,
        "OnlineSecurity_Yes": customer.OnlineSecurity_Yes,
        "OnlineBackup_No internet service": customer.OnlineBackup_No_internet_service,
        "OnlineBackup_Yes": customer.OnlineBackup_Yes,
        "DeviceProtection_No internet service": customer.DeviceProtection_No_internet_service,
        "DeviceProtection_Yes": customer.DeviceProtection_Yes,
        "TechSupport_No internet service": customer.TechSupport_No_internet_service,
        "TechSupport_Yes": customer.TechSupport_Yes,
        "StreamingTV_No internet service": customer.StreamingTV_No_internet_service,
        "StreamingTV_Yes": customer.StreamingTV_Yes,
        "StreamingMovies_No internet service": customer.StreamingMovies_No_internet_service,
        "StreamingMovies_Yes": customer.StreamingMovies_Yes,
        "Contract_One year": customer.Contract_One_year,
        "Contract_Two year": customer.Contract_Two_year,
        "PaperlessBilling_Yes": customer.PaperlessBilling_Yes,
        "PaymentMethod_Credit card (automatic)": customer.PaymentMethod_Credit_card_automatic,
        "PaymentMethod_Electronic check": customer.PaymentMethod_Electronic_check,
        "PaymentMethod_Mailed check": customer.PaymentMethod_Mailed_check,
        "tenure_group_medium": customer.tenure_group_medium,
        "tenure_group_old": customer.tenure_group_old,
    }])

    numeric = ['tenure', 'MonthlyCharges', 'TotalCharges']
    data[numeric] = scaler.transform(data[numeric])

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    return {
        "churn": bool(prediction),
        "churn_probability": round(float(probability), 4),
        "risk_level": "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"
    }