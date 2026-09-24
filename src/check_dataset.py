import pandas as pd
import config

df=pd.read_csv(config.PROCESSED_DATA)

print(df.head())

print("\nShape: ",df.shape)

print("\nColumns: ")
print(df.columns.tolist())

print("\nMissing Values: ")
print(df.isnull().sum())

print("\nDuplicate Rows Before:")
print(df.duplicated().sum())

print("\nLabel Distribution: ")
print(df["label"].value_counts())

