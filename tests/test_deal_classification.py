from app.main import classify_deal


def test_classify_deal_as_overpriced():
    predicted_price = 10000
    listed_price = 12000

    difference, percentage, status = classify_deal(
        predicted_price,
        listed_price,
    )

    assert difference == 2000
    assert percentage == 20
    assert status == "Overpriced"


def test_classify_deal_as_good_deal():
    predicted_price = 10000
    listed_price = 8000

    difference, percentage, status = classify_deal(
        predicted_price,
        listed_price,
    )

    assert difference == -2000
    assert percentage == -20
    assert status == "Good Deal"


def test_classify_deal_as_fair_price():
    predicted_price = 10000
    listed_price = 10500

    difference, percentage, status = classify_deal(
        predicted_price,
        listed_price,
    )

    assert difference == 500
    assert percentage == 5
    assert status == "Fair Price"

def test_exactly_ten_percent_above_is_fair():
    predicted_price = 10000
    listed_price = 11000

    _, percentage, status = classify_deal(
        predicted_price,
        listed_price,
    )

    assert percentage == 10
    assert status == "Fair Price"


def test_exactly_ten_percent_below_is_fair():
    predicted_price = 10000
    listed_price = 9000

    _, percentage, status = classify_deal(
        predicted_price,
        listed_price,
    )

    assert percentage == -10
    assert status == "Fair Price"