from pathlib import Path
import csv
from datetime import datetime, timezone

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

LOG_PATH = Path("logs/predictions.csv")
MODEL_VERSION = "1.0.0"

MODEL_PATH = Path("models/car_price_model.pkl")
CURRENT_YEAR = 2026


app = FastAPI(
    title="CarPriceOps API",
    description="Used-car fair price prediction API",
    version=MODEL_VERSION,
)


class CarInput(BaseModel):
    brand: str = Field(min_length=1)
    model: str = Field(min_length=1)

    year: int = Field(
        ge=1990,
        le=CURRENT_YEAR,
    )

    km_driven: float = Field(
        ge=0,
    )

    fuel: str = Field(min_length=1)
    transmission: str = Field(min_length=1)
    seller_type: str = Field(min_length=1)
    owner: str = Field(min_length=1)
    condition: str = Field(min_length=1)

    listed_price: float | None = Field(
        default=None,
        gt=0,
    )


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH.resolve()}"
        )

    return joblib.load(MODEL_PATH)


model = load_model()


def classify_deal(
    predicted_price: float,
    listed_price: float,
):
    difference = listed_price - predicted_price

    difference_percentage = (
        difference / predicted_price
    ) * 100

    if difference_percentage > 10:
        deal_status = "Overpriced"
    elif difference_percentage < -10:
        deal_status = "Good Deal"
    else:
        deal_status = "Fair Price"

    return (
        difference,
        difference_percentage,
        deal_status,
    )

def log_prediction(
    car: CarInput,
    predicted_price: float,
    deal_status: str | None,
):
    LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = LOG_PATH.exists()

    log_row = {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "model_version": MODEL_VERSION,
        "brand": car.brand.strip(),
        "model": car.model.strip(),
        "year": car.year,
        "car_age": CURRENT_YEAR - car.year,
        "km_driven": car.km_driven,
        "fuel": car.fuel.strip(),
        "transmission": car.transmission.strip(),
        "seller_type": car.seller_type.strip(),
        "owner": car.owner.strip(),
        "condition": car.condition.strip(),
        "predicted_price_cad": round(
            predicted_price,
            2,
        ),
        "listed_price_cad": (
            round(car.listed_price, 2)
            if car.listed_price is not None
            else None
        ),
        "deal_status": deal_status,
    }

    with LOG_PATH.open(
        mode="a",
        newline="",
        encoding="utf-8",
    ) as log_file:
        writer = csv.DictWriter(
            log_file,
            fieldnames=log_row.keys(),
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(log_row)

@app.get("/")
def root():
    return {
        "message": "CarPriceOps API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict_price(car: CarInput):
    try:
        car_age = CURRENT_YEAR - car.year

        input_data = pd.DataFrame(
            [
                {
                    "brand": car.brand.strip(),
                    "model": car.model.strip(),
                    "fuel": car.fuel.strip(),
                    "transmission": car.transmission.strip(),
                    "seller_type": car.seller_type.strip(),
                    "owner": car.owner.strip(),
                    "condition": car.condition.strip(),
                    "car_age": car_age,
                    "km_driven": car.km_driven,
                }
            ]
        )

        prediction = float(
            model.predict(input_data)[0]
        )

        prediction = max(
            prediction,
            0,
        )

        response = {
            "estimated_fair_price_cad": round(
                prediction,
                2,
            )
        }

        deal_status = None

        if car.listed_price is not None:
            (
                difference,
                difference_percentage,
                deal_status,
            ) = classify_deal(
                prediction,
                car.listed_price,
            )

            response.update(
                {
                    "listed_price_cad": round(
                        car.listed_price,
                        2,
                    ),
                    "price_difference_cad": round(
                        difference,
                        2,
                    ),
                    "difference_percentage": round(
                        difference_percentage,
                        2,
                    ),
                    "deal_status": deal_status,
                }
            )

        log_prediction(car=car,
                       predicted_price=prediction,
                       deal_status=deal_status,
                       )
        
        return response

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error