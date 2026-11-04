import pandas as pd

data = pd.read_csv("dataset/prepared_dataset.csv")

features = [
    "Header_Length",
    "Protocol Type",
    "Time_To_Live",
    "Rate",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "ack_count",
    "syn_count",
    "fin_count",
    "rst_count",
    "Tot sum",
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot size",
    "IAT",
    "Number",
    "Variance",
]

print("Training feature ranges:\n")
print(data[features].describe().T[["min", "mean", "max"]])