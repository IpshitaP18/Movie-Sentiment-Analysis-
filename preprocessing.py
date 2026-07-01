import pandas as pd
from utils import preprocess_text

df = pd.read_csv("dataset/IMDB_Dataset.csv")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

print("\nDataset Shape After Cleaning:")
print(df.shape)

print("\nPreprocessing reviews...")
df["clean_review"] = df["review"].apply(preprocess_text)

print("\nPreprocessing Completed!")
print(df[["review", "clean_review"]].head())

df.to_csv("dataset/cleaned_IMDB_Dataset.csv", index=False)
print("\nCleaned dataset saved as:")
print("dataset/cleaned_IMDB_Dataset.csv")