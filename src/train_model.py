
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("Loading prepared dataset...")

# Load prepared data
data = pd.read_csv("dataset/prepared_dataset.csv")

# Separate features and labels
X = data.drop("Label", axis=1)
y = data["Label"]

# Replace invalid values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

print("Dataset shape:", X.shape)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

print("\nTraining AI model...")

# Train model
model.fit(X_train, y_train)

print("Training completed!")

# Predict test data
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Normal Traffic", "UDP Flood Attack"]
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save trained model
joblib.dump(model, "models/cyberattack_model.pkl")

print("\nModel saved successfully!")
