"""Health and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.db.repository import Repository, get_repository
from app.ml.model import CATBOOST_AVAILABLE, SHAP_AVAILABLE
from app.schemas.api import HealthResponse
from app.services.cache import get_cache

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(
    settings: Settings = Depends(get_settings),
    repository: Repository = Depends(get_repository),
) -> HealthResponse:
    database_status = "ok"
    try:
        repository.ensure_schema()
    except Exception as exc:  # pragma: no cover - defensive
        database_status = f"error: {exc.__class__.__name__}"

    status = "ok" if database_status == "ok" else "degraded"
    return HealthResponse(
        status=status,
        version=settings.version,
        environment=settings.environment,
        cache=get_cache().stats(),
        ml={
            "catboost": CATBOOST_AVAILABLE,
            "shap": SHAP_AVAILABLE,
            "explainability": "tree_shap" if SHAP_AVAILABLE else "fallback",
        },
        database=database_status,
    )
