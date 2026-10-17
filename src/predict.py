import joblib
import pandas as pd


# ==========================================
# 1. Load Saved Model
# ==========================================

print("Loading trained model...")

model = joblib.load("models/churn_model.pkl")

print("Model loaded successfully!")


# ==========================================
# 2. New Customer Data
# ==========================================

new_customer = pd.DataFrame([
    {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.50,
        "TotalCharges": 427.50
    }
])


# ==========================================
# 3. Make Prediction
# ==========================================

prediction = model.predict(new_customer)

probability = model.predict_proba(new_customer)


# ==========================================
# 4. Display Prediction
# ==========================================

if prediction[0] == 1:
    result = "CHURN"
else:
    result = "NOT CHURN"


print("\n==============================")
print("Customer Churn Prediction")
print("==============================")

print("Prediction:", result)

print(
    "Churn Probability:",
    round(probability[0][1] * 100, 2),
    "%"
)

print(
    "No Churn Probability:",
    round(probability[0][0] * 100, 2),
    "%"
)

print("====")