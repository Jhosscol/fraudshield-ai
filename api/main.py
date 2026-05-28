import pickle
import numpy as np
import traceback
import gdown
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime

app = FastAPI(title="FraudShield AI API", description="API para predicción de fraude usando XGBoost")

# Paths to models
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, '..', 'app', 'models')

MODEL_PATH    = os.path.join(MODELS_DIR, 'fraud_model.pkl')
ENCODERS_PATH = os.path.join(MODELS_DIR, 'label_encoders.pkl')
SCALER_PATH   = os.path.join(MODELS_DIR, 'scaler.pkl')

# Google Drive IDs
DRIVE_FILES = {
    MODEL_PATH:    "1uDbQAxBF7rtoJjM4lQ2VrSXEKMVcgeHL",
    ENCODERS_PATH: "1n-ubf_Hp_Qz-rRJROOwUhQ-cSsh_xG6p",
    SCALER_PATH:   "1xLSh-SADeGZQqigho1e6tRFNp_TOKQZc",
}

# Features seleccionadas (hardcodeadas para evitar cargar feature_selector.pkl de 1GB)
# Customer ID, Transaction Amount, Customer Age, Customer Location, Account Age Days, Transaction Hour
SELECTED_INDICES = [0, 1, 5, 6, 8, 9]

model    = None
encoders = None
scaler   = None


def download_if_missing(path: str, file_id: str):
    """Descarga el archivo desde Google Drive si no existe localmente."""
    if not os.path.exists(path):
        print(f"Descargando {os.path.basename(path)} desde Google Drive...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, path, quiet=False)
        print(f"✓ {os.path.basename(path)} descargado.")
    else:
        print(f"✓ {os.path.basename(path)} ya existe, omitiendo descarga.")


@app.on_event("startup")
def load_models():
    global model, encoders, scaler
    try:
        for path, file_id in DRIVE_FILES.items():
            download_if_missing(path, file_id)

        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(ENCODERS_PATH, 'rb') as f:
            encoders = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)

        print("✓ Todos los modelos cargados correctamente.")
    except Exception as e:
        print(f"Error cargando modelos: {e}")
        traceback.print_exc()


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
    """Codifica el valor; retorna 0 si no existe en las clases aprendidas."""
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    return 0


@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(req: TransactionRequest):
    if model is None or encoders is None or scaler is None:
        raise HTTPException(status_code=503, detail="Los modelos no están cargados aún.")

    try:
        customer_id_encoded = safe_encode(encoders['Customer ID'],       req.customer_id)
        payment_encoded     = safe_encode(encoders['Payment Method'],    req.payment_method)
        category_encoded    = safe_encode(encoders['Product Category'],  req.product_category)
        location_encoded    = safe_encode(encoders['Customer Location'], req.customer_location)
        device_encoded      = safe_encode(encoders['Device Used'],       req.device_used)

        now = datetime.now()
        transaction_year       = now.year
        transaction_month      = now.month
        transaction_dayofweek  = now.weekday()
        transaction_is_weekend = 1 if transaction_dayofweek >= 5 else 0

        features = np.array([[
            customer_id_encoded,       # 0 - Customer ID
            req.transaction_amount,    # 1 - Transaction Amount
            payment_encoded,           # 2 - Payment Method
            category_encoded,          # 3 - Product Category
            req.quantity,              # 4 - Quantity
            req.customer_age,          # 5 - Customer Age
            location_encoded,          # 6 - Customer Location
            device_encoded,            # 7 - Device Used
            req.account_age_days,      # 8 - Account Age Days
            req.transaction_hour,      # 9 - Transaction Hour
            transaction_year,          # 10
            transaction_month,         # 11
            transaction_dayofweek,     # 12
            transaction_is_weekend     # 13
        ]])

        scaled_features   = scaler.transform(features)
        selected_features = scaled_features[:, SELECTED_INDICES]
        proba             = model.predict_proba(selected_features)[0]
        fraud_prob        = float(proba[1]) if len(proba) > 1 else float(proba[0])

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
    return {
        "status": "ok",
        "models_loaded": model is not None
    }