"""Model + XAI consistency tests."""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from app.ml.features import FEATURE_COLUMNS, latest_feature_row
from app.ml.model import JSEPredictiveModel
from tests.factories import market_frame


@lru_cache(maxsize=1)
def _trained_model() -> tuple[JSEPredictiveModel, object]:
    frame = market_frame(rows=320, seed=5)
    model = JSEPredictiveModel("Test Asset")
    model.train(frame)
    return model, frame


def test_model_trains_and_reports_metrics() -> None:
    model, _ = _trained_model()
    assert model.is_trained
    assert model.algorithm in {"gradient_boosting", "catboost", "ridge_regression"}
    assert model.metrics["mae"] >= 0
    assert 0 <= model.metrics["directional_accuracy"] <= 100
    assert model.metrics["rows_trained"] > 100


def test_shap_local_accuracy_holds() -> None:
    """sum(shap values) + base value must equal the model output."""

    model, frame = _trained_model()
    features = latest_feature_row(frame)
    result = model.predict_with_explanation(features)

    contributions = result["contributions_log_return"]
    assert len(contributions) == len(FEATURE_COLUMNS)
    assert abs(result["local_accuracy_delta"]) < 1e-9
    assert set(contributions) == set(FEATURE_COLUMNS)
    assert result["method"] in {"tree_shap", "linear_shap", "permutation_shap"}


def test_explanation_is_json_serialisable() -> None:
    """Regression test for Bug-002 (NumPy float32 in the payload)."""

    model, frame = _trained_model()
    result = model.explain_latest(frame)
    encoded = json.dumps(result)  # must not raise
    assert "contributions_log_return" in encoded

    for value in result["contributions_log_return"].values():
        assert type(value) is float
    assert type(result["base_value_log_return"]) is float
    assert type(result["prediction_log_return"]) is float


def test_price_impacts_are_scaled_by_last_close() -> None:
    model, frame = _trained_model()
    result = model.explain_latest(frame)
    last_close = result["last_close"]
    for name, impact in result["contribution_price_impact"].items():
        expected = result["contributions_log_return"][name] * last_close
        assert abs(impact - expected) < 1e-2


def test_forecast_shape_and_positivity() -> None:
    model, frame = _trained_model()
    points = model.forecast(frame, 20)

    assert len(points) == 20
    assert all(point["predicted_price"] > 0 for point in points)
    dates = [point["date"] for point in points]
    assert dates == sorted(dates)
    assert all(-0.10 <= point["predicted_log_return"] <= 0.10 for point in points)


def test_macro_shock_moves_the_forecast() -> None:
    model, frame = _trained_model()
    baseline = model.forecast(frame, 10)
    shocked = model.forecast(frame, 10, macro_shocks={"zar_usd": 0.50, "fuel_price": 0.50})

    deltas = [abs(a["predicted_price"] - b["predicted_price"]) for a, b in zip(baseline, shocked, strict=False)]
    assert max(deltas) > 0.01


def test_explainer_is_reused_across_calls() -> None:
    """Bug-001: the explainer must be built once, not per request."""

    model, frame = _trained_model()
    assert model._explainer is not None
    first = model._explainer
    model.explain_latest(frame)
    model.explain_latest(frame)
    assert model._explainer is first


def test_prediction_batch_is_deterministic() -> None:
    model, frame = _trained_model()
    first = np.array([p["predicted_price"] for p in model.forecast(frame, 5)])
    second = np.array([p["predicted_price"] for p in model.forecast(frame, 5)])
    assert np.allclose(first, second)
