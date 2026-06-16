import json
from pathlib import Path

import pandas as pd


TRAINING_DATA_PATH = Path(
    "data/processed/cleaned_car_data.csv"
)

PREDICTION_LOG_PATH = Path(
    "logs/predictions.csv"
)

DRIFT_REPORT_PATH = Path(
    "logs/drift_report.json"
)

MINIMUM_PREDICTIONS = 5

NUMERICAL_FEATURES = [
    "car_age",
    "km_driven",
]

CATEGORICAL_FEATURES = [
    "brand",
    "fuel",
]


def load_training_data():
    if not TRAINING_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found at: "
            f"{TRAINING_DATA_PATH.resolve()}"
        )

    return pd.read_csv(TRAINING_DATA_PATH)


def load_prediction_logs():
    if not PREDICTION_LOG_PATH.exists():
        raise FileNotFoundError(
            f"Prediction log not found at: "
            f"{PREDICTION_LOG_PATH.resolve()}"
        )

    prediction_df = pd.read_csv(
        PREDICTION_LOG_PATH
    )

    if prediction_df.empty:
        raise ValueError(
            "Prediction log is empty."
        )

    return prediction_df


def calculate_mean_drift(
    training_mean,
    prediction_mean,
):
    if training_mean == 0:
        return 0.0

    return (
        abs(prediction_mean - training_mean)
        / abs(training_mean)
    ) * 100


def classify_drift(
    drift_percentage,
):
    if drift_percentage > 25:
        return "HIGH DRIFT"

    if drift_percentage > 10:
        return "MODERATE DRIFT"

    return "LOW DRIFT"


def calculate_category_distribution(
    dataframe,
    feature,
):
    return (
        dataframe[feature]
        .value_counts(normalize=True)
        .to_dict()
    )


def calculate_categorical_drift(
    training_distribution,
    prediction_distribution,
):
    all_categories = set(
        training_distribution.keys()
    ) | set(
        prediction_distribution.keys()
    )

    total_difference = 0.0

    for category in all_categories:
        training_share = (
            training_distribution.get(
                category,
                0.0,
            )
        )

        prediction_share = (
            prediction_distribution.get(
                category,
                0.0,
            )
        )

        total_difference += abs(
            prediction_share - training_share
        )

    return (total_difference / 2) * 100


def build_drift_report():
    training_df = load_training_data()
    prediction_df = load_prediction_logs()

    prediction_count = len(
        prediction_df
    )

    report = {
        "prediction_count": prediction_count,
        "minimum_predictions_required": (
            MINIMUM_PREDICTIONS
        ),
        "sample_size_status": "SUFFICIENT",
        "numerical_drift": {},
        "categorical_drift": {},
    }

    if prediction_count < MINIMUM_PREDICTIONS:
        report["sample_size_status"] = (
            "INSUFFICIENT"
        )

        report["message"] = (
            "Not enough prediction records "
            "for a reliable drift assessment."
        )

    for feature in NUMERICAL_FEATURES:
        training_mean = float(
            training_df[feature].mean()
        )

        prediction_mean = float(
            prediction_df[feature].mean()
        )

        drift_percentage = (
            calculate_mean_drift(
                training_mean,
                prediction_mean,
            )
        )

        report["numerical_drift"][
            feature
        ] = {
            "training_mean": round(
                training_mean,
                2,
            ),
            "prediction_mean": round(
                prediction_mean,
                2,
            ),
            "drift_percentage": round(
                drift_percentage,
                2,
            ),
            "status": classify_drift(
                drift_percentage
            ),
        }

    for feature in CATEGORICAL_FEATURES:
        training_distribution = (
            calculate_category_distribution(
                training_df,
                feature,
            )
        )

        prediction_distribution = (
            calculate_category_distribution(
                prediction_df,
                feature,
            )
        )

        drift_percentage = (
            calculate_categorical_drift(
                training_distribution,
                prediction_distribution,
            )
        )

        report["categorical_drift"][
            feature
        ] = {
            "training_distribution": (
                training_distribution
            ),
            "prediction_distribution": (
                prediction_distribution
            ),
            "drift_percentage": round(
                drift_percentage,
                2,
            ),
            "status": classify_drift(
                drift_percentage
            ),
        }

    return report


def save_drift_report(
    report,
):
    DRIFT_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with DRIFT_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as report_file:
        json.dump(
            report,
            report_file,
            indent=4,
        )


def print_drift_report(
    report,
):
    print("Drift monitoring report")
    print("-----------------------")

    print(
        f"Prediction records: "
        f"{report['prediction_count']}"
    )

    print(
        f"Sample-size status: "
        f"{report['sample_size_status']}"
    )

    if "message" in report:
        print(report["message"])

    print("\nNumerical drift:")

    for feature, result in (
        report["numerical_drift"].items()
    ):
        print(f"\nFeature: {feature}")
        print(
            f"Training mean: "
            f"{result['training_mean']}"
        )
        print(
            f"Prediction mean: "
            f"{result['prediction_mean']}"
        )
        print(
            f"Drift: "
            f"{result['drift_percentage']}%"
        )
        print(
            f"Status: {result['status']}"
        )

    print("\nCategorical drift:")

    for feature, result in (
        report["categorical_drift"].items()
    ):
        print(f"\nFeature: {feature}")
        print(
            f"Drift: "
            f"{result['drift_percentage']}%"
        )
        print(
            f"Status: {result['status']}"
        )


def main():
    report = build_drift_report()

    print_drift_report(
        report
    )

    save_drift_report(
        report
    )

    print(
        "\nDrift report saved to:",
        DRIFT_REPORT_PATH.resolve(),
    )


if __name__ == "__main__":
    main()