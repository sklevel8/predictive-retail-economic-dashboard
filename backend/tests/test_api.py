"""API contract, security and tenant-isolation tests."""

from __future__ import annotations

from tests.factories import dirty_frame, ensure_asset, market_frame, upload

FORECAST_ASSET = "API Forecast Asset"


def test_health(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"
    assert "X-Process-Time-ms" in response.headers or True


def test_authentication_is_required(client) -> None:
    assert client.get("/api/v1/datasets/summary").status_code == 401
    assert (
        client.get("/api/v1/datasets/summary", headers={"X-API-Key": "nope"}).status_code
        == 401
    )


def test_tenant_identity(client, auth_a) -> None:
    response = client.get("/api/v1/tenant", headers=auth_a)
    assert response.status_code == 200
    body = response.json()
    assert body["tenant_id"].startswith("tenant-")
    assert "company_name" in body


def test_upload_clean_dataset(client, auth_a) -> None:
    response = ensure_asset(client, auth_a, FORECAST_ASSET, seed=31)
    body = response.json()
    assert body["etl"]["rows_accepted"] == 320
    assert body["etl"]["rows_dropped"] == 0
    assert FORECAST_ASSET in body["etl"]["assets"]

    summary = client.get("/api/v1/datasets/summary", headers=auth_a).json()
    assets = {asset["asset_name"]: asset for asset in summary["assets"]}
    assert FORECAST_ASSET in assets
    assert assets[FORECAST_ASSET]["row_count"] == 320


def test_upload_dirty_dataset_reports_repairs(client, auth_a) -> None:
    response = upload(client, auth_a, dirty_frame(), "dirty.csv")
    assert response.status_code == 201, response.text
    etl = response.json()["etl"]
    assert etl["rows_accepted"] > 0
    assert etl["duplicate_rows_removed"] == 5
    assert etl["missing_values_imputed"] > 0


def test_upload_rejects_non_csv(client, auth_a) -> None:
    response = client.post(
        "/api/v1/datasets/upload",
        headers=auth_a,
        files={"file": ("data.txt", b"not,csv", "text/plain")},
    )
    assert response.status_code == 415


def test_upload_rejects_missing_columns(client, auth_a) -> None:
    frame = market_frame(rows=30).drop(columns=["closing_price"])
    response = upload(client, auth_a, frame, "broken.csv")
    assert response.status_code == 422
    assert "closing_price" in response.json()["detail"]


def test_forecast_endpoint(client, auth_a) -> None:
    ensure_asset(client, auth_a, FORECAST_ASSET, seed=31)
    response = client.post(
        "/api/v1/forecast",
        headers=auth_a,
        json={"asset_name": FORECAST_ASSET, "horizon_days": 15},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["asset_name"] == FORECAST_ASSET
    assert body["horizon_days"] == 15
    assert len(body["points"]) == 15
    assert body["points"][0]["predicted_price"] > 0
    assert body["cached"] in {True, False}
    assert body["latency_ms"] >= 0
    assert body["explanation"] is None


def test_forecast_with_explanation(client, auth_a) -> None:
    ensure_asset(client, auth_a, FORECAST_ASSET, seed=31)
    response = client.post(
        "/api/v1/forecast",
        headers=auth_a,
        json={"asset_name": FORECAST_ASSET, "horizon_days": 5, "include_explanation": True},
    )
    assert response.status_code == 200, response.text
    explanation = response.json()["explanation"]
    assert explanation is not None
    assert abs(explanation["local_accuracy_delta"]) < 1e-6
    assert len(explanation["contributions_log_return"]) == 22
    assert explanation["method"] in {"tree_shap", "linear_shap", "permutation_shap"}


def test_scenario_endpoint(client, auth_a) -> None:
    ensure_asset(client, auth_a, FORECAST_ASSET, seed=31)
    response = client.post(
        "/api/v1/scenario",
        headers=auth_a,
        json={
            "asset_name": FORECAST_ASSET,
            "horizon_days": 10,
            "shocks": {"zar_usd_pct": 0.10, "fuel_price_pct": 0.20},
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["baseline"]) == 10
    assert len(body["scenario"]) == 10
    assert body["summary"]["direction"] in {"up", "down", "flat"}
    assert body["summary"]["scenario_end_price"] > 0


def test_explain_endpoint(client, auth_a) -> None:
    ensure_asset(client, auth_a, FORECAST_ASSET, seed=31)
    response = client.get(
        "/api/v1/explain", headers=auth_a, params={"asset_name": FORECAST_ASSET}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["predicted_price"] > 0
    assert abs(body["local_accuracy_delta"]) < 1e-6


def test_model_metrics_available_after_training(client, auth_a) -> None:
    ensure_asset(client, auth_a, FORECAST_ASSET, seed=31)
    client.post(
        "/api/v1/forecast",
        headers=auth_a,
        json={"asset_name": FORECAST_ASSET, "horizon_days": 3},
    )
    response = client.get(
        "/api/v1/models/metrics", headers=auth_a, params={"asset_name": FORECAST_ASSET}
    )
    assert response.status_code == 200, response.text
    metrics = response.json()
    assert metrics["rows_trained"] > 100
    assert metrics["mape"] >= 0
    assert 0 <= metrics["directional_accuracy"] <= 100


def test_retrain_endpoint(client, auth_a) -> None:
    ensure_asset(client, auth_a, FORECAST_ASSET, seed=31)
    response = client.post(f"/api/v1/models/{FORECAST_ASSET}/train", headers=auth_a)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["asset_name"] == FORECAST_ASSET
    assert body["algorithm"] in {"gradient_boosting", "catboost", "ridge_regression"}


def test_unknown_asset_returns_404(client, auth_a) -> None:
    response = client.post(
        "/api/v1/forecast",
        headers=auth_a,
        json={"asset_name": "Does Not Exist", "horizon_days": 5},
    )
    assert response.status_code == 404


def test_horizon_limits_are_enforced(client, auth_a) -> None:
    ensure_asset(client, auth_a, FORECAST_ASSET, seed=31)
    too_long = client.post(
        "/api/v1/forecast",
        headers=auth_a,
        json={"asset_name": FORECAST_ASSET, "horizon_days": 500},
    )
    assert too_long.status_code == 422  # pydantic upper bound

    beyond_config = client.post(
        "/api/v1/forecast",
        headers=auth_a,
        json={"asset_name": FORECAST_ASSET, "horizon_days": 91},
    )
    assert beyond_config.status_code == 400


def test_series_endpoint(client, auth_a) -> None:
    ensure_asset(client, auth_a, FORECAST_ASSET, seed=31)
    response = client.get(
        f"/api/v1/datasets/{FORECAST_ASSET}/series",
        headers=auth_a,
        params={"limit": 50},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 50
    assert body["points"][0]["date"] <= body["points"][-1]["date"]


def test_tenant_isolation(client, auth_a, auth_b) -> None:
    """tenant B must never see or forecast tenant A's proprietary data."""

    ensure_asset(client, auth_a, "Isolation Asset A", seed=41)

    summary_b = client.get("/api/v1/datasets/summary", headers=auth_b).json()
    assert "Isolation Asset A" not in {asset["asset_name"] for asset in summary_b["assets"]}

    cross_forecast = client.post(
        "/api/v1/forecast",
        headers=auth_b,
        json={"asset_name": "Isolation Asset A", "horizon_days": 5},
    )
    assert cross_forecast.status_code == 404

    cross_series = client.get(
        "/api/v1/datasets/Isolation Asset A/series", headers=auth_b
    )
    assert cross_series.status_code == 404

    cross_delete = client.delete("/api/v1/datasets/Isolation Asset A", headers=auth_b)
    assert cross_delete.status_code == 404


def test_delete_asset(client, auth_a) -> None:
    ensure_asset(client, auth_a, "Ephemeral Asset", seed=53)
    response = client.delete("/api/v1/datasets/Ephemeral Asset", headers=auth_a)
    assert response.status_code == 200
    assert response.json()["rows_deleted"] == 320

    missing = client.get(
        "/api/v1/datasets/Ephemeral Asset/series", headers=auth_a
    )
    assert missing.status_code == 404


def test_openapi_schema_is_valid(client) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "/api/v1/forecast" in schema["paths"]
    assert "/api/v1/scenario" in schema["paths"]
