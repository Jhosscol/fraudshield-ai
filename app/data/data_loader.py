import pandas as pd
import numpy as np
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'data', 'dataset.csv')
MODEL_INFO_PATH = os.path.join(BASE_DIR, 'models', 'model_info.json')

def load_real_transactions(n=20000):
    """
    Carga las últimas n transacciones del dataset real para mostrar en el dashboard.
    Lee todo el CSV (puede tardar unos segundos) y retorna el tail.
    En un entorno de producción real esto se haría con una consulta SQL.
    """
    if not os.path.exists(DATASET_PATH):
        return pd.DataFrame()
        
    # Read the whole CSV and take the last n rows to keep the UI fast.
    # To save memory, we can use usecols if we only need specific columns, but we will load all for now.
    df = pd.read_csv(DATASET_PATH)
    
    # Rename columns to match the ones expected by the UI if necessary
    # The UI expects 'Date', 'Is Fraud', 'Fraud Probability'
    # The real dataset has 'Transaction Date', 'Is Fraudulent'
    
    df = df.tail(n).copy()
    
    if 'Transaction Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Transaction Date'])
    
    if 'Is Fraudulent' in df.columns:
        df['Is Fraud'] = df['Is Fraudulent']
        
    return df

def get_model_metrics():
    """
    Lee las métricas directamente del archivo model_info.json
    """
    if not os.path.exists(MODEL_INFO_PATH):
        return {
            'Accuracy': 0.0,
            'Precision': 0.0,
            'Recall': 0.0,
            'F1 Score': 0.0,
            'ROC-AUC': 0.0,
            'Last Trained': 'Unknown'
        }
        
    with open(MODEL_INFO_PATH, 'r') as f:
        info = json.load(f)
        
    metrics = info.get('metrics', {})
    
    return {
        'Accuracy': metrics.get('accuracy', 0.0),
        'Precision': metrics.get('precision', 0.0),
        'Recall': metrics.get('recall', 0.0),
        'F1 Score': metrics.get('f1_score', 0.0),
        'ROC-AUC': metrics.get('auc_roc', 0.0),
        'Last Trained': info.get('trained_at', 'Unknown')
    }

def get_confusion_matrix_data():
    """
    Genera una matriz de confusión matemática aproximada basada en las métricas reales
    y el número de casos de prueba, ya que no tenemos las predicciones exactas del test set.
    """
    if not os.path.exists(MODEL_INFO_PATH):
        return np.array([[0, 0], [0, 0]])
        
    with open(MODEL_INFO_PATH, 'r') as f:
        info = json.load(f)
        
    metrics = info.get('metrics', {})
    test_rows = info.get('test_rows', 10000)
    fraud_rate = info.get('fraud_rate_train', 0.05)
    
    accuracy = metrics.get('accuracy', 0.9)
    precision = metrics.get('precision', 0.8)
    recall = metrics.get('recall', 0.8)
    
    actual_fraud = int(test_rows * fraud_rate)
    actual_legit = test_rows - actual_fraud
    
    # TP = True Positives (Fraudulent predicted as Fraudulent)
    tp = int(actual_fraud * recall)
    # FN = False Negatives (Fraudulent predicted as Legit)
    fn = actual_fraud - tp
    # FP = False Positives (Legit predicted as Fraudulent)
    # Precision = TP / (TP + FP) -> FP = (TP / Precision) - TP
    if precision > 0:
        fp = int((tp / precision) - tp)
    else:
        fp = 0
    # TN = True Negatives (Legit predicted as Legit)
    tn = actual_legit - fp
    
    return np.array([[tn, fp], [fn, tp]])
