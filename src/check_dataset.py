
import pandas as pd
import os

# Correct dataset paths
benign_path = "dataset/BenignTraffic1.pcap.csv"
attack_path = "dataset/DDoS-UDP_Flood.pcap.csv"

# Check files
print("Benign file exists:", os.path.exists(benign_path))
print("Attack file exists:", os.path.exists(attack_path))

# Load only the first 5 rows
benign = pd.read_csv(benign_path, nrows=5)
attack = pd.read_csv(attack_path, nrows=5)

print("\nBENIGN DATASET")
print(benign.head())

print("\nATTACK DATASET")
print(attack.head())

print("\nBENIGN COLUMNS")
print(benign.columns.tolist())

print("\nATTACK COLUMNS")
print(attack.columns.tolist())
