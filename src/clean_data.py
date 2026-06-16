import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/CarPriceOps_starter_dataset_preview.csv")

PROCESSED_DATA_PATH = Path("data/processed/cleaned_car_data.csv")

CURRENT_YEAR = 2026

def clean_data():
    print("starting data cleaning...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at: {DATA_PATH.resolve()}"
        )
    
    df = pd.read_csv(DATA_PATH)

    print(f"Original datset shape: {df.shape}")

    # 1. Remove exact duplicate rows
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        print(f"Removing {duplicate_count} duplicate rows.")
        df = df.drop_duplicates()
    else:
        print("No duplicate rows found.")

    # 2. Define text columns
    text_columns = [
        "brand",
        "model",
        "fuel",
        "transmission",
        "seller_type",
        "owner",
        "condition",
    ]

    # 3. Standardize text values
    for column in text_columns:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

    print("Text columns standardized.")

    # 4. Convert numeric columns
    numeric_columns = [
        "year",
        "km_driven",
        "listed_price_cad",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    print("Numeric columns converted.")

    # 5. Remove rows with missing required values
    required_columns = (
        text_columns
        + numeric_columns
    )

    rows_before_missing_removal = len(df)

    df = df.dropna(
        subset=required_columns
    )

    removed_missing_rows = (
        rows_before_missing_removal - len(df)
    )

    print(
        f"Rows removed because of missing values: "
        f"{removed_missing_rows}"
    )

    # 6. Remove invalid rows
    rows_before_filtering = len(df)

    df = df[
        (df["year"] >= 1990)
        & (df["year"] <= CURRENT_YEAR)
        & (df["km_driven"] >= 0)
        & (df["listed_price_cad"] > 0)
    ]

    removed_invalid_rows = (
        rows_before_filtering - len(df)
    )

    print(
        f"Rows removed because of invalid values: "
        f"{removed_invalid_rows}"
    )

    # 7. Recalculate car age
    df["car_age"] = CURRENT_YEAR - df["year"]

    print("car_age recalculated.")

    # 8. Arrange columns
    final_columns = [
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

    df = df[final_columns]

    # 9. Create processed directory if needed
    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 10. Save cleaned dataset
    df.to_csv(
        PROCESSED_DATA_PATH,
        index=False,
    )

    print(f"Cleaned dataset shape: {df.shape}")
    print(
        "Cleaned dataset saved to:",
        PROCESSED_DATA_PATH.resolve(),
    )


if __name__ == "__main__":
    clean_data()