import joblib

model = joblib.load("models/cyberattack_model.pkl")

print("===== MODEL FEATURE CHECK =====")
print("Number of features:", len(model.feature_names_in_))

print("\nFeatures:")
for feature in model.feature_names_in_:
    print(feature)

print("\nDoes model use Number?")
print("Number" in model.feature_names_in_)