
import pandas as pd
import numpy as np

# Dataset paths
benign_path = "dataset/BenignTraffic1.pcap.csv"
attack_path = "dataset/DDoS-UDP_Flood.pcap.csv"

print("Loading datasets...")

# Read a manageable amount of data initially
benign = pd.read_csv(benign_path, nrows=50000)
attack = pd.read_csv(attack_path, nrows=50000)

# Create labels
benign["Label"] = 0
attack["Label"] = 1

# Combine datasets
data = pd.concat(
    [benign, attack],
    ignore_index=True
)

print("\nCombined dataset shape:", data.shape)

# Remove invalid numeric values
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

# Separate features and target
X = data.drop("Label", axis=1)
y = data["Label"]

print("\nNormal traffic:", (y == 0).sum())
print("Attack traffic:", (y == 1).sum())

print("\nFeatures:", X.shape[1])
print("Total records:", len(data))

# Save prepared dataset
data.to_csv(
    "dataset/prepared_dataset.csv",
    index=False
)

print("\nPrepared dataset saved successfully!")
