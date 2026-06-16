import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/processed/cleaned_car_data.csv")

TARGET_COLUMN = "listed_price_cad"

CATEGORICAL_FEATURES = [
    "brand",
    "model",
    "fuel",
    "transmission",
    "seller_type",
    "owner",
    "condition",
]

NUMERICAL_FEATURES = [
    "car_age",
    "km_driven",
]

def load_clean_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at: {DATA_PATH.resolve()}"
        )
    
    df = pd.read_csv(DATA_PATH)

    print("Cleaned dataset loaded.")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    return df

def split_features_and_target(df):
    feature_columns = (
        CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    )

    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    print("\nFeatures selected:")
    print(feature_columns)

    print("\nFeature shape:", X.shape)
    print("Target shape", y.shape)

    return X, y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    print("\nTrain-test split completed.")
    print("Training rows:", X_train.shape[0])
    print("Testing rows:", X_test.shape[0])

    return X_train, X_test, y_train, y_test


def create_preprocessor():
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore"
    )

    numerical_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_transformer,
                CATEGORICAL_FEATURES,
            ),
            (
                "numerical",
                numerical_transformer,
                NUMERICAL_FEATURES,
            ),
        ]
    )

    return preprocessor


def test_preprocessing():
    df = load_clean_data()

    X, y = split_features_and_target(df)

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
    )

    preprocessor = create_preprocessor()

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

    print("\nPreprocessing completed.")
    print(
        "Original training shape:",
        X_train.shape,
    )
    print(
        "Processed training shape:",
        X_train_processed.shape,
    )
    print(
        "Processed testing shape:",
        X_test_processed.shape,
    )


if __name__ == "__main__":
    test_preprocessing()