import pandas as pd

df = pd.read_csv("data/customer_churn.csv")

print("Dataset Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nChurn Distribution:")
print(df["Churn"].value_counts())