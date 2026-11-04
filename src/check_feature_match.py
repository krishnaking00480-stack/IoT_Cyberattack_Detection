import pandas as pd
import joblib

dataset_path = "dataset/prepared_dataset.csv"
model_path = "models/cyberattack_model.pkl"

data = pd.read_csv(dataset_path, nrows=5)
model = joblib.load(model_path)

dataset_features = list(data.drop(columns=["Label"]).columns)
model_features = list(model.feature_names_in_)

print("Dataset feature count:", len(dataset_features))
print("Model feature count:", len(model_features))

print("\nDataset and model feature order match:",
      dataset_features == model_features)

if dataset_features != model_features:
    print("\nFeatures missing from dataset:")
    print(set(model_features) - set(dataset_features))

    print("\nExtra dataset features:")
    print(set(dataset_features) - set(model_features))