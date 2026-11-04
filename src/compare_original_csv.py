import pandas as pd

files = {
    "Benign": "dataset/BenignTraffic1.pcap.csv",
    "UDP Attack": "dataset/DDoS-UDP_Flood.pcap.csv",
}

features = [
    "Protocol Type",
    "Rate",
    "ack_count",
    "syn_count",
    "Tot sum",
    "Min",
    "Max",
    "AVG",
    "IAT",
    "Number",
]

for name, path in files.items():
    print(f"\n========== {name} ==========")

    data = pd.read_csv(path)

    print("Rows:", len(data))
    print("Columns:", len(data.columns))

    print("\nFeature statistics:")
    print(
        data[features]
        .describe()
        .T[["min", "50%", "mean", "max"]]
        .to_string()
    )

    print("\nFirst 5 rows:")
    print(data[features].head().to_string(index=False))