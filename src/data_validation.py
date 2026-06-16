import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/CarPriceOps_starter_dataset_preview.csv")

REQUIRED_COLUMNS = [
    "brand",
    "model",
    "year",
    "car_age",
    "km_driven",
    "fuel",
    "transmission",
    "seller_type",
    "owner",
    "condition",
    "listed_price_cad",
]


def validate_data():
    print("Starting data validation...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH.resolve()}")

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset loaded. Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # 1. Check required columns
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    print("Column check passed.")

    # 2. Check empty dataset
    if df.empty:
        raise ValueError("Dataset is empty.")

    print("Empty dataset check passed.")

    # 3. Check missing values
    missing_values = df.isnull().sum()

    if missing_values.sum() > 0:
        print("Missing values found:")
        print(missing_values[missing_values > 0])
        raise ValueError("Dataset contains missing values.")

    print("Missing value check passed.")

    # 4. Check duplicate rows
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        print(f"Warning: Found {duplicate_count} duplicate rows.")
    else:
        print("Duplicate check passed.")

    # 5. Check invalid year values
    if (df["year"] < 1990).any() or (df["year"] > 2026).any():
        raise ValueError("Invalid year values found.")

    print("Year check passed.")

    # 6. Check invalid car age
    if (df["car_age"] < 0).any():
        raise ValueError("Invalid car_age values found.")

    print("Car age check passed.")

    # 7. Check negative kilometers
    if (df["km_driven"] < 0).any():
        raise ValueError("Negative km_driven values found.")

    print("Kilometer check passed.")

    # 8. Check invalid prices
    if (df["listed_price_cad"] <= 0).any():
        raise ValueError("Invalid listed_price_cad values found.")

    print("Price check passed.")

    # 9. Show unique values for categorical columns
    categorical_columns = [
        "brand",
        "model",
        "fuel",
        "transmission",
        "seller_type",
        "owner",
        "condition",
    ]

    print("\nUnique values in categorical columns:")

    for col in categorical_columns:
        print(f"{col}: {df[col].nunique()} unique values")

    print("\nData validation completed successfully.")


if __name__ == "__main__":
    validate_data()