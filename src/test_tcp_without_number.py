import pandas as pd
import numpy as np
import joblib

MODEL_PATH = "models/cyberattack_model_without_number.pkl"
DATA_PATH = "dataset/DDoS-TCP_Flood.pcap.csv"

print("Loading model without Number...")

model = joblib.load(MODEL_PATH)

print("Loading TCP Flood dataset...")

data = pd.read_csv(DATA_PATH, nrows=50000)

# Get model feature names
features = list(model.feature_names_in_)

# Select matching columns
X = data[features].copy()

# Convert values to numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Handle invalid values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.clip(lower=-1e10, upper=1e10)
X = X.fillna(0)

print("\nTesting TCP Flood without Number...")

predictions = model.predict(X)

normal_count = int((predictions == 0).sum())
attack_count = int((predictions == 1).sum())

total = len(predictions)

print("\n===== TCP FLOOD RESULTS =====")
print("Total records:", total)
print("Predicted Normal:", normal_count)
print("Predicted Attack:", attack_count)

print(
    "Predicted Attack Percentage:",
    round(attack_count / total * 100, 2),
    "%"
)