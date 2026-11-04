import pandas as pd

data = pd.read_csv("dataset/prepared_dataset.csv")

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

for label, name in [(0, "Benign"), (1, "Attack")]:
    subset = data[data["Label"] == label]

    print(f"\n{name} traffic")
    print(f"Rows: {len(subset)}")
    print(subset[features].describe().T[["min", "50%", "mean", "max"]])