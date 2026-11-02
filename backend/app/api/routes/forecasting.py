"""Forecasting, scenario testing and model lifecycle endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.security import require_tenant
from app.db.repository import Repository, Tenant, get_repository
from app.schemas.api import (
    Explanation,
    ForecastRequest,
    ForecastResponse,
    ModelMetricsResponse,
    ScenarioRequest,
    ScenarioResponse,
    TrainResponse,
)
from app.services.forecasting import get_registry, run_forecast, run_scenario

router = APIRouter(tags=["forecasting"])


@router.post("/forecast", response_model=ForecastResponse)
def forecast(
    request: ForecastRequest,
    tenant: Tenant = Depends(require_tenant),
    repository: Repository = Depends(get_repository),
) -> ForecastResponse:
    payload = run_forecast(tenant.tenant_id, repository, request)
    return ForecastResponse(**payload)


@router.post("/scenario", response_model=ScenarioResponse)
def scenario(
    request: ScenarioRequest,
    tenant: Tenant = Depends(require_tenant),
    repository: Repository = Depends(get_repository),
) -> ScenarioResponse:
    payload = run_scenario(tenant.tenant_id, repository, request)
    return ScenarioResponse(**payload)


@router.get("/explain", response_model=Explanation)
def explain(
    asset_name: str = Query(..., min_length=1, max_length=120),
    tenant: Tenant = Depends(require_tenant),
    repository: Repository = Depends(get_repository),
) -> Explanation:
    registry = get_registry()
    model = registry.get(tenant.tenant_id, asset_name)
    frame = registry.frame_for(tenant.tenant_id, asset_name)
    return Explanation(**model.explain_latest(frame))


@router.post("/models/{asset_name}/train", response_model=TrainResponse)
def train_model(
    asset_name: str,
    tenant: Tenant = Depends(require_tenant),
    repository: Repository = Depends(get_repository),
) -> TrainResponse:
    registry = get_registry()
    result = registry.retrain(tenant.tenant_id, asset_name)

    from app.services.cache import get_cache

    get_cache().invalidate(f"p10:forecast:{tenant.tenant_id}:{asset_name}:")
    get_cache().invalidate(f"p10:scenario:{tenant.tenant_id}:{asset_name}:")
    return TrainResponse(**result.as_dict())


@router.get("/models/metrics", response_model=ModelMetricsResponse)
def model_metrics(
    asset_name: str | None = Query(None, max_length=120),
    tenant: Tenant = Depends(require_tenant),
    repository: Repository = Depends(get_repository),
) -> ModelMetricsResponse:
    metrics = repository.latest_metrics(tenant.tenant_id, asset_name)
    if metrics is None:
        raise HTTPException(
            status_code=404,
            detail="No model has been trained for this tenant yet. Request a forecast first.",
        )
    return ModelMetricsResponse(
        asset_name=metrics["asset_name"],
        algorithm=metrics["algorithm"],
        trained_at=metrics["trained_at"],
        rows_trained=int(metrics["rows_trained"]),
        mae=float(metrics["mae"]),
        rmse=float(metrics["rmse"]),
        mape=float(metrics["mape"]),
        directional_accuracy=float(metrics["directional_accuracy"]),
    )


@router.get("/models", response_model=list[str])
def list_assets(
    tenant: Tenant = Depends(require_tenant),
    repository: Repository = Depends(get_repository),
) -> list[str]:
    return [asset.asset_name for asset in repository.list_assets(tenant.tenant_id)]
