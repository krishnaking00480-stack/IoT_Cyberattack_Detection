import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "cyberattack_model.pkl"
DATASET_PATH = BASE_DIR / "dataset" / "BenignTraffic1.pcap.csv"

print("Loading trained model...")
model = joblib.load(MODEL_PATH)

print("Loading benign dataset...")
df = pd.read_csv(DATASET_PATH, nrows=50000)

# Clean infinite and missing values
df = df.replace([float("inf"), float("-inf")], pd.NA)
df = df.dropna()

# Remove target-related columns if present
X = df.drop(columns=["Label"], errors="ignore")

# Match the model's expected feature columns
expected_features = model.feature_names_in_

X = X.reindex(columns=expected_features, fill_value=0)

print("Testing unseen benign traffic...")

predictions = model.predict(X)

normal_count = (predictions == 0).sum()
attack_count = (predictions == 1).sum()

total = len(predictions)

print("\n===== UNSEEN BENIGN RESULTS =====")
print("Total records:", total)
print("Predicted Normal:", normal_count)
print("Predicted Attack:", attack_count)
print(
    "False Alarm Percentage:",
    round((attack_count / total) * 100, 2),
    "%"
)

print("\nTesting completed!")
