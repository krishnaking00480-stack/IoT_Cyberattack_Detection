import pandas as pd

file_path = "dataset/DDoS-TCP_Flood.pcap.csv"

print("Loading TCP Flood dataset...")

df = pd.read_csv(file_path, nrows=50000)

print("\n===== NUMBER FEATURE CHECK =====")

print("\nNumber statistics:")
print(df["Number"].describe())

print("\nMost common Number values:")
print(df["Number"].value_counts().head(20))

print("\nTCP feature statistics:")
print(df["TCP"].describe())