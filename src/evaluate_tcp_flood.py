import pandas as pd
import numpy as np
import joblib

# File paths
data_path = "dataset/DDoS-TCP_Flood.pcap.csv"
model_path = "models/cyberattack_model.pkl"

print("Loading model...")
model = joblib.load(model_path)

print("Loading TCP Flood dataset...")
data = pd.read_csv(data_path, nrows=50000)

# Keep the same feature columns used by the model
if hasattr(model, "feature_names_in_"):
    feature_columns = list(model.feature_names_in_)
    data = data[feature_columns]

# Clean numeric values
data = data.apply(pd.to_numeric, errors="coerce")
data = data.replace([np.inf, -np.inf], np.nan)
data = data.fillna(0)

print("\nEvaluating TCP Flood traffic...")

predictions = model.predict(data)

# Change these labels if your model uses different names
normal_label = 0
attack_label = 1

total = len(predictions)
predicted_attack = (predictions == attack_label).sum()
predicted_normal = (predictions == normal_label).sum()

detection_rate = (predicted_attack / total) * 100
miss_rate = (predicted_normal / total) * 100

print("\n===== TCP FLOOD EVALUATION =====")
print("Total records:", total)
print("Predicted Attack:", predicted_attack)
print("Predicted Normal:", predicted_normal)
print(f"Attack Detection Rate: {detection_rate:.2f}%")
print(f"Miss Rate: {miss_rate:.2f}%")

print("\nEvaluation completed!")