import pickle
import numpy as np
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import os

app = FastAPI(title="FraudShield AI API", description="API para predicción de fraude usando XGBoost")

# Paths to models
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, '..', 'app', 'models', 'fraud_model.pkl')
ENCODERS_PATH = os.path.join(BASE_DIR, '..', 'app', 'models', 'label_encoders.pkl')
SCALER_PATH = os.path.join(BASE_DIR, '..', 'app', 'models', 'scaler.pkl')
SELECTOR_PATH = os.path.join(BASE_DIR, '..', 'app', 'models', 'feature_selector.pkl')

model = None
encoders = None
scaler = None
selector = None

@app.on_event("startup")
def load_models():
    global model, encoders, scaler, selector
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(ENCODERS_PATH, 'rb') as f:
            encoders = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        with open(SELECTOR_PATH, 'rb') as f:
            selector = pickle.load(f)
        print("Model, encoders, scaler and selector loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")

class TransactionRequest(BaseModel):
    transaction_amount: float
    customer_id: str
    payment_method: str
    product_category: str
    quantity: int
    customer_age: int
    customer_location: str
    device_used: str
    account_age_days: int
    transaction_hour: int

class PredictionResponse(BaseModel):
    fraud_probability: float
    risk_level: str

def safe_encode(encoder, value):
    """
    Intenta codificar el valor. Si no existe en las clases aprendidas,
    retorna 0 (o la clase más común) para evitar errores.
    """
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        return 0

@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(req: TransactionRequest):
    if model is None or encoders is None or scaler is None or selector is None:
        raise HTTPException(status_code=503, detail="Models are not loaded.")

    try:
        # Encode categorical variables
        customer_id_encoded = safe_encode(encoders['Customer ID'], req.customer_id)
        payment_encoded = safe_encode(encoders['Payment Method'], req.payment_method)
        category_encoded = safe_encode(encoders['Product Category'], req.product_category)
        location_encoded = safe_encode(encoders['Customer Location'], req.customer_location)
        device_encoded = safe_encode(encoders['Device Used'], req.device_used)

        # Date related features (simulated from current time)
        now = datetime.now()
        transaction_year = now.year
        transaction_month = now.month
        transaction_dayofweek = now.weekday()
        transaction_is_weekend = 1 if transaction_dayofweek >= 5 else 0

        # Construct the 14-feature array in the exact order of `all_features`
        # 1. Customer ID
        # 2. Transaction Amount
        # 3. Payment Method
        # 4. Product Category
        # 5. Quantity
        # 6. Customer Age
        # 7. Customer Location
        # 8. Device Used
        # 9. Account Age Days
        # 10. Transaction Hour
        # 11. transaction_year
        # 12. transaction_month
        # 13. transaction_dayofweek
        # 14. transaction_is_weekend

        features = np.array([[
            customer_id_encoded,
            req.transaction_amount,
            payment_encoded,
            category_encoded,
            req.quantity,
            req.customer_age,
            location_encoded,
            device_encoded,
            req.account_age_days,
            req.transaction_hour,
            transaction_year,
            transaction_month,
            transaction_dayofweek,
            transaction_is_weekend
        ]])

        # 1. Scale all 14 features
        scaled_features = scaler.transform(features)
        
        # 2. Select the 6 features
        selected_features = selector.transform(scaled_features)

        # 3. Predict probability
        proba = model.predict_proba(selected_features)[0]
        
        fraud_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
        
        if fraud_prob > 0.75:
            risk_level = "HIGH"
        elif fraud_prob > 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return PredictionResponse(
            fraud_probability=fraud_prob,
            risk_level=risk_level
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": model is not None}
