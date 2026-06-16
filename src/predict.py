from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path(
    "models/car_price_model.pkl"
)

CURRENT_YEAR = 2026


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Saved model not found. Run src/train.py first."
        )

    model = joblib.load(MODEL_PATH)

    return model


def predict_car_price(car_details):
    model = load_model()

    input_data = pd.DataFrame(
        [car_details]
    )

    prediction = model.predict(
        input_data
    )[0]

    return round(float(prediction), 2)


if __name__ == "__main__":
    sample_car = {
        "brand": "Honda",
        "model": "Civic",
        "fuel": "Gasoline",
        "transmission": "Automatic",
        "seller_type": "Dealer",
        "owner": "First Owner",
        "condition": "Good",
        "car_age": CURRENT_YEAR - 2019,
        "km_driven": 85000,
    }

    estimated_price = predict_car_price(
        sample_car
    )

    print(
        f"Estimated fair price: "
        f"${estimated_price:,.2f} CAD"
    )