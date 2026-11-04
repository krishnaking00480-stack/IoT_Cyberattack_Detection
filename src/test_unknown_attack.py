
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load(
    "models/cyberattack_model.pkl"
)

# Change this to your actual downloaded filename
file_path = "dataset/DDoS-TCP_Flood.pcap.csv"

print("Loading unseen attack dataset...")

# Load a sample of the dataset
data = pd.read_csv(
    file_path,
    nrows=50000
)

print("Records loaded:", len(data))

# Prepare features
features = list(model.feature_names_in_)

# Check for missing columns
missing_columns = [
    col for col in features
    if col not in data.columns
]

if missing_columns:
    print("Missing features:", missing_columns)
    raise ValueError(
        "Dataset features do not match the model."
    )

# Select features
X = data[features].copy()

# Convert to numeric
X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

# Handle invalid values
X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.clip(
    lower=-1e10,
    upper=1e10
)

X = X.fillna(0)

# Predict
print("\nTesting unseen attack...")

predictions = model.predict(X)

# Count predictions
normal_count = int(
    (predictions == 0).sum()
)

attack_count = int(
    (predictions == 1).sum()
)

total = len(predictions)

attack_percentage = (
    attack_count / total * 100
)

print("\n===== UNKNOWN ATTACK RESULTS =====")

print("Total records:", total)

print("Predicted Normal:", normal_count)

print("Predicted Attack:", attack_count)

print(
    "Predicted Attack Percentage:",
    round(attack_percentage, 2),
    "%"
)

print("\nTesting completed!")
