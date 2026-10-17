import pandas as pd

df = pd.read_csv("data/customer_churn.csv")

print("Missing Values:")
print(df.isnull().sum())

print("\nData Types:")
print(df.dtypes)