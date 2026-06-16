from pathlib import Path

import pandas as pd

from src.drift_monitor import (
    build_drift_report,
    calculate_categorical_drift,
    calculate_mean_drift,
    classify_drift,
)


def test_calculate_mean_drift():
    result = calculate_mean_drift(
        training_mean=100,
        prediction_mean=120,
    )

    assert result == 20


def test_classify_low_drift():
    assert classify_drift(5) == "LOW DRIFT"


def test_classify_moderate_drift():
    assert classify_drift(15) == "MODERATE DRIFT"


def test_classify_high_drift():
    assert classify_drift(30) == "HIGH DRIFT"


def test_calculate_categorical_drift():
    training_distribution = {
        "Gasoline": 0.8,
        "Hybrid": 0.2,
    }

    prediction_distribution = {
        "Gasoline": 0.5,
        "Hybrid": 0.5,
    }

    result = calculate_categorical_drift(
        training_distribution,
        prediction_distribution,
    )

    assert round(result, 2) == 30.00

def test_build_drift_report_with_insufficient_samples(
    tmp_path,
    monkeypatch,
):
    training_data = pd.DataFrame(
        {
            "car_age": [5, 6, 7, 8, 9],
            "km_driven": [
                50000,
                60000,
                70000,
                80000,
                90000,
            ],
            "brand": [
                "Honda",
                "Honda",
                "Toyota",
                "Toyota",
                "Ford",
            ],
            "fuel": [
                "Gasoline",
                "Gasoline",
                "Gasoline",
                "Hybrid",
                "Gasoline",
            ],
        }
    )

    prediction_data = pd.DataFrame(
        {
            "car_age": [12, 13],
            "km_driven": [
                150000,
                160000,
            ],
            "brand": [
                "Honda",
                "Honda",
            ],
            "fuel": [
                "Electric",
                "Electric",
            ],
        }
    )

    training_path = (
        tmp_path / "training.csv"
    )

    prediction_path = (
        tmp_path / "predictions.csv"
    )

    training_data.to_csv(
        training_path,
        index=False,
    )

    prediction_data.to_csv(
        prediction_path,
        index=False,
    )

    monkeypatch.setattr(
        "src.drift_monitor.TRAINING_DATA_PATH",
        training_path,
    )

    monkeypatch.setattr(
        "src.drift_monitor.PREDICTION_LOG_PATH",
        prediction_path,
    )

    report = build_drift_report()

    assert report["prediction_count"] == 2

    assert (
        report["sample_size_status"]
        == "INSUFFICIENT"
    )

    assert "numerical_drift" in report
    assert "categorical_drift" in report