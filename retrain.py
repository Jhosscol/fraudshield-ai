"""
retrain.py — Re-entrenamiento automático de FraudShield AI
Triggers:
  1. Drift detectado (F1/AUC bajo umbral)
  2. Nuevo dataset subido al repo
Los modelos se guardan directo en el repo (sin Drive).
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier

# ─── Configuración ───────────────────────────────────────────────────────────
DATA_PATH    = "app/data/dataset.csv"
MODELS_DIR   = "app/models"
METRICS_FILE = "metrics/latest_metrics.json"

F1_THRESHOLD  = 0.25
AUC_THRESHOLD = 0.75

CATEGORICAL_COLS = [
    "Customer ID", "Payment Method", "Product Category",
    "Customer Location", "Device Used"
]
TARGET_COL = "Is Fraudulent"

# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_metrics() -> dict:
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE) as f:
            return json.load(f)
    return {"f1": 0.0, "auc": 0.0}


def save_metrics(f1: float, auc: float):
    os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
    data = {
        "f1": round(f1, 4),
        "auc": round(auc, 4),
        "updated_at": datetime.now().isoformat()
    }
    with open(METRICS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Métricas guardadas: F1={data['f1']} | AUC={data['auc']}")


def drift_detected() -> bool:
    metrics = load_metrics()
    f1  = metrics.get("f1", 0.0)
    auc = metrics.get("auc", 0.0)
    print(f"Métricas actuales → F1: {f1} | AUC: {auc}")
    if f1 < F1_THRESHOLD or auc < AUC_THRESHOLD:
        print(f"Drift detectado. Re-entrenamiento necesario.")
        return True
    print("Modelo dentro de umbrales. No hay drift.")
    return False


# ─── Pipeline de entrenamiento ───────────────────────────────────────────────

def train():
    print("=" * 50)
    print(f"Iniciando re-entrenamiento: {datetime.now()}")
    print("=" * 50)

    if not os.path.exists(DATA_PATH):
        print(f"Dataset no encontrado en {DATA_PATH}. Abortando.")
        return False

    df = pd.read_csv(DATA_PATH)
    print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

    df = df.dropna()

    encoders = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    if "Transaction Date" in df.columns:
        df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")
        df["transaction_year"]       = df["Transaction Date"].dt.year
        df["transaction_month"]      = df["Transaction Date"].dt.month
        df["transaction_dayofweek"]  = df["Transaction Date"].dt.dayofweek
        df["transaction_is_weekend"] = (df["transaction_dayofweek"] >= 5).astype(int)
        df = df.drop(columns=["Transaction Date"])

    all_features = [
        "Customer ID", "Transaction Amount", "Payment Method",
        "Product Category", "Quantity", "Customer Age",
        "Customer Location", "Device Used", "Account Age Days",
        "Transaction Hour", "transaction_year", "transaction_month",
        "transaction_dayofweek", "transaction_is_weekend"
    ]
    available = [f for f in all_features if f in df.columns]
    X = df[available].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    selector = SelectKBest(f_classif, k=6)
    X_train_sel = selector.fit_transform(X_train_scaled, y_train)
    X_test_sel  = selector.transform(X_test_scaled)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train_sel, y_train)

    y_pred  = model.predict(X_test_sel)
    y_proba = model.predict_proba(X_test_sel)[:, 1]
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"Resultado → F1: {f1:.4f} | AUC: {auc:.4f}")

    if f1 < F1_THRESHOLD:
        print(f"F1 ({f1:.4f}) bajo umbral ({F1_THRESHOLD}). Modelo no guardado.")
        return False

    # Guardar modelos localmente (el workflow hará commit de ellos)
    os.makedirs(MODELS_DIR, exist_ok=True)
    artifacts = {
        "fraud_model.pkl":      model,
        "label_encoders.pkl":   encoders,
        "scaler.pkl":           scaler,
        "feature_selector.pkl": selector,
    }
    for filename, obj in artifacts.items():
        path = os.path.join(MODELS_DIR, filename)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        print(f"Guardado: {path}")

    save_metrics(f1, auc)
    print("Re-entrenamiento completado exitosamente.")
    return True


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    trigger = sys.argv[1] if len(sys.argv) > 1 else "data"
    print(f"Trigger: {trigger}")

    if trigger == "drift":
        if not drift_detected():
            print("Sin drift. Saliendo sin re-entrenar.")
            sys.exit(0)

    success = train()
    sys.exit(0 if success else 1)