import pandas as pd
import config

fake=pd.read_csv(config.FAKE_DATA)
true=pd.read_csv(config.TRUE_DATA)

print("Fake Dataset shape:",fake.shape)
print("True Dataset shape:",true.shape)

# print("\nFake Dataset")
# print(fake.head())

# print("\nTrue Dataset")
# print(true.head())

# add labels
fake["label"]=0
true["label"]=1

# merge datasets
df=pd.concat([fake,true],ignore_index=True)

print("\nMerged Dataset Shape :",df.shape)

# Shuffle dataset
df=df.sample(
    frac=1,
    random_state=config.RANDOM_STATE
).reset_index(drop=True)

print("\nFirst 5 Rows")
print(df.head())



print("\nProcessed dataset saved successfully.")
print(config.PROCESSED_DATA)

# Removing Duplicate values
print("\nDuplicate Rows Before:")
print(df.duplicated().sum())

df=df.drop_duplicates()

print("\nDuplicate Rows After:")
print(df.duplicated().sum())

print("\nFinal Dataset Shape:",df.shape)


# Save processed dataset
df.to_csv(
    config.PROCESSED_DATA,
    index=False
)