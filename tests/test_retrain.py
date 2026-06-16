from src.retrain import should_replace_model


def test_replace_when_current_metrics_do_not_exist():
    assert should_replace_model(
        None,
        {"rmse": 900.0},
    ) is True


def test_replace_when_improvement_exceeds_threshold():
    current_metrics = {
        "rmse": 1000.0
    }

    new_metrics = {
        "rmse": 900.0
    }

    assert should_replace_model(
        current_metrics,
        new_metrics,
    ) is True


def test_replace_at_exact_threshold():
    current_metrics = {
        "rmse": 1000.0
    }

    new_metrics = {
        "rmse": 950.0
    }

    assert should_replace_model(
        current_metrics,
        new_metrics,
    ) is True


def test_retain_when_improvement_is_too_small():
    current_metrics = {
        "rmse": 1000.0
    }

    new_metrics = {
        "rmse": 970.0
    }

    assert should_replace_model(
        current_metrics,
        new_metrics,
    ) is False


def test_retain_when_metrics_are_equal():
    assert should_replace_model(
        {"rmse": 800.0},
        {"rmse": 800.0},
    ) is False


def test_retain_when_new_model_is_worse():
    assert should_replace_model(
        {"rmse": 800.0},
        {"rmse": 950.0},
    ) is False