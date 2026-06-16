import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/CarPriceOps_starter_dataset_preview.csv")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())