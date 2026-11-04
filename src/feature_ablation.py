
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Loading dataset...")

data = pd.read_csv("dataset/prepared_dataset.csv")

# Remove Number because of the packet-count mismatch
X = data.drop(columns=["Label", "Number"])
y = data["Label"]

# Handle invalid values
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

# Remove exact duplicate feature rows
combined = X.copy()
combined["Label"] = y
combined = combined.drop_duplicates()

X = combined.drop(columns=["Label"])
y = combined["Label"]

print("Records after deduplication:", len(X))
print("Features:", X.shape[1])

# Keep the same train/test split for every experiment
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

def train_and_evaluate(features, name):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(X_train[features], y_train)
    predictions = model.predict(X_test[features])

    accuracy = accuracy_score(y_test, predictions)

    print(f"{name}: {accuracy * 100:.4f}%")

print("\nBaseline: all features")
train_and_evaluate(list(X.columns), "All features")

# Remove one feature at a time
print("\nFeature ablation results:")

for feature in X.columns:
    remaining_features = [
        column for column in X.columns
        if column != feature
    ]

    train_and_evaluate(
        remaining_features,
        f"Without {feature}"
    )
