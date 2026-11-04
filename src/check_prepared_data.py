import pandas as pd

df = pd.read_csv("dataset/prepared_dataset.csv")

print("Dataset shape:", df.shape)

print("\nLabel counts:")
print(df["Label"].value_counts())

print("\nNumber values by label:")
print(
    df.groupby("Label")["Number"]
    .value_counts()
    .groupby(level=0)
    .head(10)
)

print("\nNumber summary by label:")
print(df.groupby("Label")["Number"].describe())