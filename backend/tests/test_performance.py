"""Non-functional tests: inference latency against the 3-second SLA."""

from __future__ import annotations

import time

import pytest

from tests.factories import ensure_asset

ASSET = "Performance Asset"
SLA_SECONDS = 3.0


@pytest.fixture(scope="module")
def warmed(client, auth_a) -> None:
    ensure_asset(client, auth_a, ASSET, seed=61)
    # First request trains the model; keep that cost outside the SLA window.
    response = client.post(
        "/api/v1/forecast",
        headers=auth_a,
        json={"asset_name": ASSET, "horizon_days": 30},
    )
    assert response.status_code == 200, response.text


def _timed_call(client, headers, payload):
    started = time.perf_counter()
    response = client.post("/api/v1/scenario", headers=headers, json=payload)
    elapsed = time.perf_counter() - started
    assert response.status_code == 200, response.text
    return elapsed, response.json()


def test_scenario_latency_under_three_seconds(client, auth_a, warmed) -> None:
    durations = []
    for index in range(5):
        payload = {
            "asset_name": ASSET,
            "horizon_days": 30,
            "shocks": {"zar_usd_pct": -0.05 + index * 0.01, "fuel_price_pct": 0.02},
        }
        elapsed, body = _timed_call(client, auth_a, payload)
        durations.append(elapsed)
        assert body["latency_ms"] < SLA_SECONDS * 1000

    worst = max(durations)
    assert worst < SLA_SECONDS, f"worst scenario latency was {worst:.2f}s"


def test_cached_forecast_is_fast(client, auth_a, warmed) -> None:
    payload = {"asset_name": ASSET, "horizon_days": 30}
    client.post("/api/v1/forecast", headers=auth_a, json=payload)

    started = time.perf_counter()
    response = client.post("/api/v1/forecast", headers=auth_a, json=payload)
    elapsed = time.perf_counter() - started

    assert response.status_code == 200
    assert response.json()["cached"] is True
    assert elapsed < 1.0
