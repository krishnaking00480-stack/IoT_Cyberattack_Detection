import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("Loading prepared dataset...")

data = pd.read_csv("dataset/prepared_dataset.csv")

# Remove the known packet-count feature
X = data.drop(columns=["Label", "Number"])
y = data["Label"]

# Remove invalid values
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

# Combine features and labels, then remove exact duplicate records
clean_data = X.copy()
clean_data["Label"] = y

print("Records before deduplication:", len(clean_data))

clean_data = clean_data.drop_duplicates()

print("Records after deduplication:", len(clean_data))

# Separate features and labels
X = clean_data.drop(columns=["Label"])
y = clean_data["Label"]

print("\nClass counts:")
print(y.value_counts())

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

print("\nTraining model...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Normal Traffic", "UDP Flood Attack"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))