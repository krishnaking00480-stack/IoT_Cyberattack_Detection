import pandas as pd
import numpy as np

file_path = "dataset/DDoS-TCP_Flood.pcap.csv"

print("Loading TCP Flood dataset...")

data = pd.read_csv(file_path, nrows=50000)

print("\n===== DATASET CHECK =====")
print("Records:", len(data))
print("Columns:", len(data.columns))

print("\nMissing values:", data.isnull().sum().sum())

data = data.replace([np.inf, -np.inf], np.nan)

print("Infinite values replaced.")
print("TCP protocol values:")
print(data["TCP"].value_counts(dropna=False))

print("\nProtocol Type values:")
print(data["Protocol Type"].value_counts(dropna=False))

print("\nDataset check completed!")