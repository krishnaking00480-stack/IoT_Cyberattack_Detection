import pandas as pd

dataset_path = "dataset/BenignTraffic1.pcap.csv"

print("Reading dataset columns...")

data = pd.read_csv(dataset_path, nrows=5)

print("\nDataset columns:")
for column in data.columns:
    print(column)

print("\nTotal columns:", len(data.columns))

print("\nFirst 5 rows:")
print(data.head())