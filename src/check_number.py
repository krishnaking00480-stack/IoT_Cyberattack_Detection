import pandas as pd

df = pd.read_csv("dataset/BenignTraffic1.pcap.csv")

print("Benign Number value counts:")
print(df["Number"].value_counts().head(20))

print("\nFirst 10 values:")
print(df["Number"].head(10).to_list())