"""Forecast orchestration: registry + cache + response shaping."""

from __future__ import annotations

import hashlib
import json
import time
from functools import lru_cache
from typing import Any

from app.config import get_settings
from app.core.errors import DomainError
from app.db.repository import Repository, get_repository
from app.ml.model import JSEPredictiveModel, to_native
from app.ml.registry import ModelRegistry
from app.schemas.api import (
    Explanation,
    ForecastPoint,
    ForecastRequest,
    ScenarioRequest,
    ScenarioSummary,
)
from app.services.cache import get_cache


@lru_cache(maxsize=1)
def get_registry() -> ModelRegistry:
    settings = get_settings()
    return ModelRegistry(get_repository(), settings.model_dir, settings.min_training_rows)


def reset_registry() -> None:
    get_registry.cache_clear()


def _key(tenant_id: str, kind: str, asset: str, version: str, payload: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    return f"p10:{kind}:{tenant_id}:{asset}:{version}:{digest}"


def _shock_dict(request: ScenarioRequest) -> dict[str, float]:
    shocks = {
        "zar_usd": request.shocks.zar_usd_pct,
        "cpi": request.shocks.cpi_pct,
        "fuel_price": request.shocks.fuel_price_pct,
        "repo_rate": request.shocks.repo_rate_pct,
        "volume": request.shocks.volume_pct,
    }
    return {key: value for key, value in shocks.items() if abs(value) > 1e-12}


def _points(points: list[dict[str, Any]]) -> list[ForecastPoint]:
    return [ForecastPoint(**point) for point in points]


def _explanation(model: JSEPredictiveModel, frame) -> Explanation:
    payload = to_native(model.explain_latest(frame))
    return Explanation(**payload)


def _validate_horizon(horizon: int) -> None:
    maximum = get_settings().max_horizon
    if horizon > maximum:
        raise DomainError(f"horizon_days must be between 1 and {maximum}.")


def run_forecast(
    tenant_id: str, repository: Repository, request: ForecastRequest
) -> dict[str, Any]:
    _validate_horizon(request.horizon_days)
    started = time.perf_counter()

    registry = get_registry()
    version = registry.version(tenant_id, request.asset_name)
    cache = get_cache()
    cache_key = _key(
        tenant_id,
        "forecast",
        request.asset_name,
        version,
        {"horizon": request.horizon_days, "explain": request.include_explanation},
    )

    cached = cache.get(cache_key)
    if cached is not None:
        cached["cached"] = True
        cached["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return cached

    model = registry.get(tenant_id, request.asset_name)
    frame = registry.frame_for(tenant_id, request.asset_name)

    points = model.forecast(frame, request.horizon_days)
    explanation = (
        _explanation(model, frame).model_dump() if request.include_explanation else None
    )

    last = frame.sort_values("date_logged").iloc[-1]
    payload = to_native(
        {
            "asset_name": request.asset_name,
            "algorithm": model.algorithm,
            "horizon_days": request.horizon_days,
            "last_observed_date": last["date_logged"].strftime("%Y-%m-%d"),
            "last_close": float(last["closing_price"]),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cached": False,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "points": points,
            "explanation": explanation,
        }
    )
    cache.set(cache_key, payload)
    return payload


def run_scenario(
    tenant_id: str, repository: Repository, request: ScenarioRequest
) -> dict[str, Any]:
    _validate_horizon(request.horizon_days)
    started = time.perf_counter()

    registry = get_registry()
    version = registry.version(tenant_id, request.asset_name)
    cache = get_cache()
    shocks = _shock_dict(request)
    cache_key = _key(
        tenant_id,
        "scenario",
        request.asset_name,
        version,
        {
            "horizon": request.horizon_days,
            "shocks": shocks,
            "explain": request.include_explanation,
        },
    )

    cached = cache.get(cache_key)
    if cached is not None:
        cached["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return cached

    model = registry.get(tenant_id, request.asset_name)
    frame = registry.frame_for(tenant_id, request.asset_name)

    baseline = model.forecast(frame, request.horizon_days)
    scenario = model.forecast(frame, request.horizon_days, macro_shocks=shocks)

    baseline_end = baseline[-1]["predicted_price"] if baseline else 0.0
    scenario_end = scenario[-1]["predicted_price"] if scenario else 0.0
    delta = scenario_end - baseline_end
    percent = (delta / baseline_end * 100.0) if baseline_end else 0.0
    per_step_deltas = [
        item["predicted_price"] - base["predicted_price"]
        for item, base in zip(scenario, baseline, strict=False)
    ]

    if abs(percent) < 0.01:
        direction = "flat"
    elif percent > 0:
        direction = "up"
    else:
        direction = "down"

    summary = ScenarioSummary(
        baseline_end_price=round(baseline_end, 4),
        scenario_end_price=round(scenario_end, 4),
        absolute_change=round(delta, 4),
        percent_change=round(percent, 4),
        average_price_delta=round(sum(per_step_deltas) / len(per_step_deltas), 4)
        if per_step_deltas
        else 0.0,
        max_price_delta=round(max(per_step_deltas, key=abs), 4) if per_step_deltas else 0.0,
        direction=direction,
    )

    last = frame.sort_values("date_logged").iloc[-1]
    explanation = (
        _explanation(model, frame).model_dump() if request.include_explanation else None
    )

    payload = to_native(
        {
            "asset_name": request.asset_name,
            "algorithm": model.algorithm,
            "horizon_days": request.horizon_days,
            "last_observed_date": last["date_logged"].strftime("%Y-%m-%d"),
            "last_close": float(last["closing_price"]),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "shocks": request.shocks.model_dump(),
            "baseline": baseline,
            "scenario": scenario,
            "summary": summary.model_dump(),
            "explanation": explanation,
        }
    )
    cache.set(cache_key, payload)
    return payload
