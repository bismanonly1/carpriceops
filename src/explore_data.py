import pandas as pd
from pathlib import Path


DATA_PATH = Path(
    "data/processed/cleaned_car_data.csv"
)


def explore_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at: {DATA_PATH.resolve()}"
        )

    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:")
    print(df.shape)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumn data types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nNumerical summary:")
    print(
        df[
            [
                "year",
                "car_age",
                "km_driven",
                "listed_price_cad",
            ]
        ].describe()
    )

    categorical_columns = [
        "brand",
        "model",
        "fuel",
        "transmission",
        "seller_type",
        "owner",
        "condition",
    ]

    print("\nCategorical value counts:")

    for column in categorical_columns:
        print(f"\n--- {column} ---")
        print(df[column].value_counts())

    print("\nCorrelation between numerical columns:")
    print(
        df[
            [
                "year",
                "car_age",
                "km_driven",
                "listed_price_cad",
            ]
        ].corr()
    )


if __name__ == "__main__":
    explore_data()