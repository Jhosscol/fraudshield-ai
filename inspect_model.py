import pickle
import sys

try:
    with open('app/models/fraud_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model type:", type(model))
    
    if hasattr(model, 'feature_names_in_'):
        print("Features expected:", model.feature_names_in_)
    elif hasattr(model, 'get_booster'):
        print("Features expected:", model.get_booster().feature_names)
except Exception as e:
    print("Error loading model:", e)
