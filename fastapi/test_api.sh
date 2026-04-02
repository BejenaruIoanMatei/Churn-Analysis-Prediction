#!/bin/bash

BASE_URL="http://localhost:8001"
PASS=0
FAIL=0

check_risk() {
    local response=$1
    local expected=$2
    local test_name=$3

    risk=$(echo $response | grep -o '"risk_level":"[^"]*"' | cut -d'"' -f4)
    churn=$(echo $response | grep -o '"churn":[a-z]*' | cut -d':' -f2)
    prob=$(echo $response | grep -o '"churn_probability":[0-9.]*' | cut -d':' -f2)

    if [ "$risk" == "$expected" ]; then
        echo "PASS — $test_name"
        echo "churn=$churn | probability=$prob | risk=$risk"
        PASS=$((PASS + 1))
    else
        echo "FAIL — $test_name"
        echo "Expected risk=$expected | Got risk=$risk"
        echo "churn=$churn | probability=$prob"
        FAIL=$((FAIL + 1))
    fi
}

echo "========================================"
echo "  Churn Prediction API — Test Suite"
echo "  $BASE_URL"
echo "========================================"
echo ""

echo "--- Health Check ---"
response=$(curl -s "$BASE_URL/")
echo "Response: $response"
echo ""

echo "--- Predict Tests ---"
echo ""

echo "[1] New customer, month-to-month, fiber optic"
response=$(curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 2,
    "MonthlyCharges": 95.0,
    "TotalCharges": 190.0,
    "gender_Male": 0,
    "SeniorCitizen_1": 0,
    "Partner_Yes": 0,
    "Dependents_Yes": 0,
    "PhoneService_Yes": 1,
    "PaperlessBilling_Yes": 1,
    "MultipleLines_No phone service": 0,
    "MultipleLines_Yes": 0,
    "InternetService_Fiber optic": 1,
    "InternetService_No": 0,
    "OnlineSecurity_No internet service": 0,
    "OnlineSecurity_Yes": 0,
    "OnlineBackup_No internet service": 0,
    "OnlineBackup_Yes": 0,
    "DeviceProtection_No internet service": 0,
    "DeviceProtection_Yes": 0,
    "TechSupport_No internet service": 0,
    "TechSupport_Yes": 0,
    "StreamingTV_No internet service": 0,
    "StreamingTV_Yes": 0,
    "StreamingMovies_No internet service": 0,
    "StreamingMovies_Yes": 0,
    "Contract_One year": 0,
    "Contract_Two year": 0,
    "PaymentMethod_Credit card (automatic)": 0,
    "PaymentMethod_Electronic check": 1,
    "PaymentMethod_Mailed check": 0,
    "tenure_group_medium": 0,
    "tenure_group_old": 0
  }')
check_risk "$response" "High" "New + month-to-month + fiber → High risk"
echo ""

echo "[2] Old customer, two-year contract, no internet"
response=$(curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 68,
    "MonthlyCharges": 25.0,
    "TotalCharges": 1700.0,
    "gender_Male": 1,
    "SeniorCitizen_1": 0,
    "Partner_Yes": 1,
    "Dependents_Yes": 1,
    "PhoneService_Yes": 1,
    "PaperlessBilling_Yes": 0,
    "MultipleLines_No phone service": 0,
    "MultipleLines_Yes": 0,
    "InternetService_Fiber optic": 0,
    "InternetService_No": 1,
    "OnlineSecurity_No internet service": 1,
    "OnlineSecurity_Yes": 0,
    "OnlineBackup_No internet service": 1,
    "OnlineBackup_Yes": 0,
    "DeviceProtection_No internet service": 1,
    "DeviceProtection_Yes": 0,
    "TechSupport_No internet service": 1,
    "TechSupport_Yes": 0,
    "StreamingTV_No internet service": 1,
    "StreamingTV_Yes": 0,
    "StreamingMovies_No internet service": 1,
    "StreamingMovies_Yes": 0,
    "Contract_One year": 0,
    "Contract_Two year": 1,
    "PaymentMethod_Credit card (automatic)": 1,
    "PaymentMethod_Electronic check": 0,
    "PaymentMethod_Mailed check": 0,
    "tenure_group_medium": 0,
    "tenure_group_old": 1
  }')
check_risk "$response" "Low" "Old + two-year + no internet → Low risk"
echo ""

echo "[3] Medium tenure, one-year contract, DSL"
response=$(curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 24,
    "MonthlyCharges": 55.0,
    "TotalCharges": 1320.0,
    "gender_Male": 1,
    "SeniorCitizen_1": 0,
    "Partner_Yes": 0,
    "Dependents_Yes": 0,
    "PhoneService_Yes": 1,
    "PaperlessBilling_Yes": 1,
    "MultipleLines_No phone service": 0,
    "MultipleLines_Yes": 1,
    "InternetService_Fiber optic": 0,
    "InternetService_No": 0,
    "OnlineSecurity_No internet service": 0,
    "OnlineSecurity_Yes": 0,
    "OnlineBackup_No internet service": 0,
    "OnlineBackup_Yes": 1,
    "DeviceProtection_No internet service": 0,
    "DeviceProtection_Yes": 0,
    "TechSupport_No internet service": 0,
    "TechSupport_Yes": 0,
    "StreamingTV_No internet service": 0,
    "StreamingTV_Yes": 1,
    "StreamingMovies_No internet service": 0,
    "StreamingMovies_Yes": 0,
    "Contract_One year": 1,
    "Contract_Two year": 0,
    "PaymentMethod_Credit card (automatic)": 0,
    "PaymentMethod_Electronic check": 0,
    "PaymentMethod_Mailed check": 1,
    "tenure_group_medium": 1,
    "tenure_group_old": 0
  }')
check_risk "$response" "Medium" "Medium tenure + one-year → Medium risk"
echo ""

echo "========================================"
echo "  Results: $PASS passed | $FAIL failed"
echo "========================================"