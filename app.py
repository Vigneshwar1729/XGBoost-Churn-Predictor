from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import xgboost as xgb
import os

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API to predict bank customer churn using an XGBoost model.",
    version="1.0.0"
)

# Enable CORS for the React/Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (update for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model on startup
MODEL_PATH = "xgboost_churn_model.json"
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"Warning: {MODEL_PATH} not found. Predictions will fail.")

# Define the input schema
class ChurnPredictionRequest(BaseModel):
    CreditScore: int
    Gender: int       # 0 for Female, 1 for Male
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int    # 0 or 1
    IsActiveMember: int # 0 or 1

@app.post("/predict")
def predict_churn(request: ChurnPredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded on the server.")
    
    # Create DataFrame for prediction
    data = {
        "CreditScore": [request.CreditScore],
        "Gender": [request.Gender],
        "Age": [request.Age],
        "Tenure": [request.Tenure],
        "Balance": [request.Balance],
        "NumOfProducts": [request.NumOfProducts],
        "HasCrCard": [request.HasCrCard],
        "IsActiveMember": [request.IsActiveMember]
    }
    df = pd.DataFrame(data)
    
    # Predict
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    
    return {
        "churn_prediction": int(prediction),
        "churn_probability": float(probability)
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}
