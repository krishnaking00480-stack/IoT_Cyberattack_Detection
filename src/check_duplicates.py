import pandas as pd

data = pd.read_csv("dataset/prepared_dataset.csv")

print("Total records:", len(data))

# Check duplicates including the label
print(
    "Exact duplicate rows:",
    data.duplicated().sum()
)

# Check duplicates using features only
X = data.drop(columns=["Label"])

print(
    "Duplicate feature rows:",
    X.duplicated().sum()
)

# Check whether identical feature rows have different labels
label_counts = data.groupby(
    list(X.columns),
    dropna=False
)["Label"].nunique()

print(
    "Identical feature rows with conflicting labels:",
    (label_counts > 1).sum()
)