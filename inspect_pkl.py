import pickle
import traceback

def inspect():
    try:
        with open('app/models/fraud_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        print("Model loaded successfully. Type:", type(model))
        
        if hasattr(model, 'feature_names_in_'):
            print("Features (sklearn API):", list(model.feature_names_in_))
        elif hasattr(model, 'get_booster'):
            print("Features (xgboost API):", model.get_booster().feature_names)
        else:
            print("Could not automatically determine feature names from the model object.")
            
    except Exception as e:
        print("Error reading model:")
        traceback.print_exc()

    print("\n----------------\n")
    
    try:
        with open('app/models/label_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        
        print("Encoders loaded successfully. Type:", type(encoders))
        if isinstance(encoders, dict):
            print("Encoder keys (features):", list(encoders.keys()))
            for k, v in encoders.items():
                print(f"  {k} -> {type(v)}")
        else:
            print("Encoders object is not a dict. Content:", encoders)
    except Exception as e:
        print("Error reading encoders:")
        traceback.print_exc()

if __name__ == '__main__':
    inspect()
