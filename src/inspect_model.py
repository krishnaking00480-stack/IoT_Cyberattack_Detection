import joblib

model_path = "models/cyberattack_model.pkl"

print("Loading trained model...")

model = joblib.load(model_path)

print("\nModel loaded successfully!")
print("Model type:", type(model).__name__)
print("Number of features:", model.n_features_in_)
print("Classes:", model.classes_)

if hasattr(model, "feature_names_in_"):
    print("\nModel feature names:")
    for feature in model.feature_names_in_:
        print(feature)
else:
    print("\nFeature names are not available in the model.")