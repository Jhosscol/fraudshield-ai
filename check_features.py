import pickle

with open("app/models/feature_selector.pkl", "rb") as f:
    selector = pickle.load(f)

all_features = [
    "Customer ID", "Transaction Amount", "Payment Method",
    "Product Category", "Quantity", "Customer Age",
    "Customer Location", "Device Used", "Account Age Days",
    "Transaction Hour", "transaction_year", "transaction_month",
    "transaction_dayofweek", "transaction_is_weekend"
]

selected = [all_features[i] for i in selector.get_support(indices=True)]
print("Features seleccionadas:", selected)