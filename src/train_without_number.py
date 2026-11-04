
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "dataset" / "prepared_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "cyberattack_model_without_number.pkl"

print("Loading prepared dataset...")

df = pd.read_csv(DATA_PATH)

# Remove invalid values
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# Separate features and target
X = df.drop(columns=["Label", "Number"], errors="ignore")
y = df["Label"]

print("Dataset shape:", X.shape)
print("Number feature excluded.")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

print("\n===== MODEL EVALUATION =====")
print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Normal Traffic", "UDP Flood Attack"]
    )
)

# Save model
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully:")
print(MODEL_PATH)
