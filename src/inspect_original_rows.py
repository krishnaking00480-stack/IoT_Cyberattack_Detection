import pandas as pd

data = pd.read_csv("dataset/prepared_dataset.csv")

features = [
    "Header_Length",
    "Protocol Type",
    "Time_To_Live",
    "Rate",
    "fin_flag_number",
    "syn_flag_number",
    "ack_flag_number",
    "ack_count",
    "Tot sum",
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot size",
    "IAT",
    "Number",
    "Variance",
    "Label",
]

print("First 10 benign rows:")
print(data[data["Label"] == 0][features].head(10).to_string(index=False))

print("\nFirst 10 attack rows:")
print(data[data["Label"] == 1][features].head(10).to_string(index=False))