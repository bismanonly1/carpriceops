from fastapi.testclient import TestClient
from pathlib import Path

from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    response_data = response.json()

    assert response_data["status"] == "healthy"
    assert response_data["model_loaded"] is True

def test_predict_endpoint_with_valid_input():
    request_data = {
        "brand": "Honda",
        "model": "Civic",
        "year": 2019,
        "km_driven": 85000,
        "fuel": "Gasoline",
        "transmission": "Automatic",
        "seller_type": "Dealer",
        "owner": "First Owner",
        "condition": "Good",
        "listed_price": 19000,
    }

    response = client.post(
        "/predict",
        json=request_data,
    )

    assert response.status_code == 200

    response_data = response.json()

    assert "estimated_fair_price_cad" in response_data
    assert "listed_price_cad" in response_data
    assert "price_difference_cad" in response_data
    assert "difference_percentage" in response_data
    assert "deal_status" in response_data

    assert response_data["estimated_fair_price_cad"] >= 0
    assert response_data["listed_price_cad"] == 19000

    assert response_data["deal_status"] in [
        "Good Deal",
        "Fair Price",
        "Overpriced",
    ]

def test_predict_rejects_future_year():
    request_data = {
        "brand": "Honda",
        "model": "Civic",
        "year": 2030,
        "km_driven": 85000,
        "fuel": "Gasoline",
        "transmission": "Automatic",
        "seller_type": "Dealer",
        "owner": "First Owner",
        "condition": "Good",
        "listed_price": 19000,
    }

    response = client.post(
        "/predict",
        json=request_data,
    )

    assert response.status_code == 422

def test_predict_rejects_negative_kilometers():
    request_data = {
        "brand": "Honda",
        "model": "Civic",
        "year": 2019,
        "km_driven": -5000,
        "fuel": "Gasoline",
        "transmission": "Automatic",
        "seller_type": "Dealer",
        "owner": "First Owner",
        "condition": "Good",
        "listed_price": 19000,
    }

    response = client.post(
        "/predict",
        json=request_data,
    )

    assert response.status_code == 422

def test_predict_rejects_missing_required_field():
    request_data = {
        "brand": "Honda",
        "year": 2019,
        "km_driven": 85000,
        "fuel": "Gasoline",
        "transmission": "Automatic",
        "seller_type": "Dealer",
        "owner": "First Owner",
        "condition": "Good",
        "listed_price": 19000,
    }

    response = client.post(
        "/predict",
        json=request_data,
    )

    assert response.status_code == 422

def test_predict_rejects_negative_listed_price():
    request_data = {
        "brand": "Honda",
        "model": "Civic",
        "year": 2019,
        "km_driven": 85000,
        "fuel": "Gasoline",
        "transmission": "Automatic",
        "seller_type": "Dealer",
        "owner": "First Owner",
        "condition": "Good",
        "listed_price": -19000,
    }

    response = client.post(
        "/predict",
        json=request_data,
    )

    assert response.status_code == 422

def test_prediction_is_logged(
    tmp_path,
    monkeypatch,
):
    temporary_log = (
        tmp_path / "predictions.csv"
    )

    monkeypatch.setattr(
        "app.main.LOG_PATH",
        temporary_log,
    )

    request_data = {
        "brand": "Honda",
        "model": "Civic",
        "year": 2019,
        "km_driven": 85000,
        "fuel": "Gasoline",
        "transmission": "Automatic",
        "seller_type": "Dealer",
        "owner": "First Owner",
        "condition": "Good",
        "listed_price": 19000,
    }

    response = client.post(
        "/predict",
        json=request_data,
    )

    assert response.status_code == 200
    assert temporary_log.exists()

    log_content = temporary_log.read_text(
        encoding="utf-8"
    )

    assert "Honda" in log_content
    assert "Civic" in log_content
    assert "predicted_price_cad" in log_content