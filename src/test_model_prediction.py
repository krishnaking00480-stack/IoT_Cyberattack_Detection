import pandas as pd
import joblib

# Load dataset and trained model
data = pd.read_csv("dataset/prepared_dataset.csv")
model = joblib.load("models/cyberattack_model.pkl")

# Select one normal record and one attack record
normal = data[data["Label"] == 0].drop(columns=["Label"]).iloc[[0]]
attack = data[data["Label"] == 1].drop(columns=["Label"]).iloc[[0]]

# Predict
normal_prediction = model.predict(normal)[0]
attack_prediction = model.predict(attack)[0]

print("Normal record prediction:", normal_prediction)
print("Attack record prediction:", attack_prediction)

print("\nExpected:")
print("Normal = 0")
print("Attack = 1")