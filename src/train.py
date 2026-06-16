import json
from pathlib import Path

import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EXPERIMENT_NAME = "CarPriceOps"

DATA_PATH = Path(
    "data/processed/cleaned_car_data.csv"
)

MODEL_PATH = Path(
    "models/car_price_model.pkl"
)

METRICS_PATH = Path(
    "models/model_metrics.json"
)

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


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at: {DATA_PATH.resolve()}"
        )

    df = pd.read_csv(DATA_PATH)

    print(
        f"Dataset loaded: {df.shape[0]} rows, "
        f"{df.shape[1]} columns"
    )

    return df


def prepare_data(df):
    feature_columns = (
        CATEGORICAL_FEATURES
        + NUMERICAL_FEATURES
    )

    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

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


def create_pipeline(model):
    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return pipeline


def evaluate_model(
    model_name,
    pipeline,
    X_test,
    y_test,
):
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    mse = mean_squared_error(
        y_test,
        predictions,
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        predictions,
    )

    metrics = {
        "model_name": model_name,
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "r2": round(float(r2), 4),
    }

    print(f"\n{model_name}")
    print("-" * len(model_name))
    print(f"MAE:  ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R²:   {r2:.4f}")

    comparison = pd.DataFrame(
        {
            "actual_price": y_test.values,
            "predicted_price": predictions,
        }
    )

    comparison["absolute_error"] = (
        comparison["actual_price"]
        - comparison["predicted_price"]
    ).abs()

    print("\nActual vs predicted:")
    print(comparison.round(2))

    return metrics


def train_and_compare_models():
    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    df = load_data()

    X_train, X_test, y_train, y_test = prepare_data(
        df
    )

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    candidate_models = {
        "Linear Regression": {
            "model": LinearRegression(),
            "params": {},
        },

        "Random Forest": {
            "model": RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=2,
                random_state=42,
            ),
            "params": {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_split": 2,
                "random_state": 42,
            },
        },
    }

    all_metrics = []
    trained_pipelines = {}

    for model_name, model_config in candidate_models.items():
        print(f"\nTraining {model_name}...")

        model = model_config["model"]
        model_params = model_config["params"]

        pipeline = create_pipeline(model)

        pipeline.fit(
            X_train,
            y_train,
        )

        metrics = evaluate_model(
            model_name,
            pipeline,
            X_test,
            y_test,
        )

        all_metrics.append(metrics)
        trained_pipelines[model_name] = pipeline

        with mlflow.start_run(
            run_name=model_name
        ):
            mlflow.log_param(
                "model_name",
                model_name,
            )

            mlflow.log_param(
                "training_rows",
                len(X_train),
            )

            mlflow.log_param(
                "testing_rows",
                len(X_test),
            )

            mlflow.log_params(
                model_params
            )

            mlflow.log_metric(
                "mae",
                metrics["mae"],
            )

            mlflow.log_metric(
                "rmse",
                metrics["rmse"],
            )

            mlflow.log_metric(
                "r2",
                metrics["r2"],
            )

            mlflow.set_tag(
                "project",
                "CarPriceOps",
            )

            mlflow.set_tag(
                "model_type",
                model_name,
            )

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
            )

            print(
                f"{model_name} logged to MLflow."
            )

    best_result = min(
        all_metrics,
        key=lambda result: result["rmse"],
    )

    best_model_name = best_result["model_name"]
    best_pipeline = trained_pipelines[
        best_model_name
    ]

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        best_pipeline,
        MODEL_PATH,
    )

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as metrics_file:
        json.dump(
            {
                "best_model": best_model_name,
                "best_metrics": best_result,
                "all_models": all_metrics,
            },
            metrics_file,
            indent=4,
        )

    print("\nModel comparison completed.")
    print(f"Best model: {best_model_name}")
    print(f"Best RMSE: ${best_result['rmse']:,.2f}")
    print(f"Model saved to: {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    train_and_compare_models()