import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

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


DATA_PATH = Path(
    "data/processed/cleaned_car_data.csv"
)

CURRENT_MODEL_PATH = Path(
    "models/car_price_model.pkl"
)

CURRENT_METRICS_PATH = Path(
    "models/model_metrics.json"
)

RETRAINING_REPORT_PATH = Path(
    "models/retraining_report.json"
)

TARGET_COLUMN = "listed_price_cad"

EXPERIMENT_NAME = "CarPriceOps-Retraining"

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

MINIMUM_IMPROVEMENT_PERCENT = 5.0

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at: "
            f"{DATA_PATH.resolve()}"
        )

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError(
            "Processed dataset is empty."
        )

    print(
        f"Loaded dataset: "
        f"{df.shape[0]} rows, "
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
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numerical",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
        ]
    )

def create_pipeline(model):
    return Pipeline(
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

def evaluate_model(
    pipeline,
    X_test,
    y_test,
):
    predictions = pipeline.predict(
        X_test
    )

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

    return {
        "mae": round(
            float(mae),
            2,
        ),
        "rmse": round(
            float(rmse),
            2,
        ),
        "r2": round(
            float(r2),
            4,
        ),
    }

def load_current_metrics():
    if not CURRENT_METRICS_PATH.exists():
        print(
            "Current metrics file not found."
        )

        return None

    with CURRENT_METRICS_PATH.open(
        "r",
        encoding="utf-8",
    ) as metrics_file:
        metrics_data = json.load(
            metrics_file
        )

    return metrics_data["best_metrics"]

def train_candidate_models(
    X_train,
    X_test,
    y_train,
    y_test,
):
    candidate_models = {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=2,
            random_state=42,
        ),
    }

    trained_models = {}
    results = []

    for model_name, model in (
        candidate_models.items()
    ):
        print(
            f"\nRetraining {model_name}..."
        )

        pipeline = create_pipeline(
            model
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        metrics = evaluate_model(
            pipeline,
            X_test,
            y_test,
        )

        result = {
            "model_name": model_name,
            **metrics,
        }

        results.append(
            result
        )

        trained_models[
            model_name
        ] = pipeline

        print(result)

    return trained_models, results

def select_best_model(
    trained_models,
    results,
):
    best_result = min(
        results,
        key=lambda result: result["rmse"],
    )

    best_model_name = (
        best_result["model_name"]
    )

    best_pipeline = trained_models[
        best_model_name
    ]

    return (
        best_model_name,
        best_pipeline,
        best_result,
    )

def should_replace_model(
    current_metrics,
    new_metrics,
):
    if current_metrics is None:
        return True

    current_rmse = float(
        current_metrics["rmse"]
    )

    new_rmse = float(
        new_metrics["rmse"]
    )

    if current_rmse <= 0:
        return False

    improvement_percentage = (
        (current_rmse - new_rmse)
        / current_rmse
    ) * 100

    return (
        improvement_percentage
        >= MINIMUM_IMPROVEMENT_PERCENT
    )

def save_retraining_report(
    current_metrics,
    new_metrics,
    replaced,
):
    report = {
        "current_model_metrics": (
            current_metrics
        ),
        "new_model_metrics": (
            new_metrics
        ),
        "model_replaced": replaced,
    }

    RETRAINING_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RETRAINING_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as report_file:
        json.dump(
            report,
            report_file,
            indent=4,
        )

def retrain():
    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    df = load_data()

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = prepare_data(df)

    trained_models, results = (
        train_candidate_models(
            X_train,
            X_test,
            y_train,
            y_test,
        )
    )

    (
        best_model_name,
        best_pipeline,
        best_metrics,
    ) = select_best_model(
        trained_models,
        results,
    )

    current_metrics = (
        load_current_metrics()
    )

    replace_model = should_replace_model(
        current_metrics,
        best_metrics,
    )

    with mlflow.start_run(
        run_name="Retraining Run"
    ):
        mlflow.log_param(
            "best_model_name",
            best_model_name,
        )

        mlflow.log_param(
            "training_rows",
            len(X_train),
        )

        mlflow.log_param(
            "testing_rows",
            len(X_test),
        )

        mlflow.log_metric(
            "new_mae",
            best_metrics["mae"],
        )

        mlflow.log_metric(
            "new_rmse",
            best_metrics["rmse"],
        )

        mlflow.log_metric(
            "new_r2",
            best_metrics["r2"],
        )

        if current_metrics is not None:
            mlflow.log_metric(
                "current_rmse",
                current_metrics["rmse"],
            )

        mlflow.log_param(
            "model_replaced",
            replace_model,
        )

        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            name="retrained_model",
        )

    if replace_model:
        joblib.dump(
            best_pipeline,
            CURRENT_MODEL_PATH,
        )

        updated_metrics = {
            "best_model": best_model_name,
            "best_metrics": best_metrics,
            "all_models": results,
        }

        with CURRENT_METRICS_PATH.open(
            "w",
            encoding="utf-8",
        ) as metrics_file:
            json.dump(
                updated_metrics,
                metrics_file,
                indent=4,
            )

        print(
            "\nCurrent model replaced."
        )

    else:
        print(
            "\nCurrent model retained."
        )

    save_retraining_report(
        current_metrics=current_metrics,
        new_metrics=best_metrics,
        replaced=replace_model,
    )

    print(
        f"Best retrained model: "
        f"{best_model_name}"
    )

    print(
        f"New RMSE: "
        f"${best_metrics['rmse']:,.2f}"
    )

    print(
        "Retraining report saved to:",
        RETRAINING_REPORT_PATH.resolve(),
    )

if __name__ == "__main__":
    retrain()