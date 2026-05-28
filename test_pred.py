import pickle
import pandas as pd
with open('app/models/fraud_model.pkl', 'rb') as f:
    model = pickle.load(f)

booster = model.get_booster()
df = booster.trees_to_dataframe()
print("Features used in splits:", df['Feature'].unique())
