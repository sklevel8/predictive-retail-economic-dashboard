"""Dataset ingestion and inspection endpoints (tenant scoped)."""

from __future__ import annotations

import io
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from app.config import Settings, get_settings
from app.core.errors import DomainError
from app.core.security import require_tenant
from app.db.repository import AssetSummary, Repository, Tenant, get_repository
from app.etl.pipeline import clean_market_frame, frame_to_records
from app.schemas.api import (
    AssetSummaryModel,
    DatasetSummaryResponse,
    SeriesPoint,
    SeriesResponse,
    UploadResponse,
)
from app.services.cache import get_cache
from app.services.forecasting import get_registry

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _summary(tenant: Tenant, repository: Repository, assets: list[AssetSummary]) -> DatasetSummaryResponse:
    return DatasetSummaryResponse(
        tenant_id=tenant.tenant_id,
        company_name=tenant.company_name,
        asset_count=len(assets),
        total_rows=sum(asset.row_count for asset in assets),
        assets=[
            AssetSummaryModel(
                asset_name=asset.asset_name,
                row_count=asset.row_count,
                first_date=asset.first_date,
                last_date=asset.last_date,
                last_close=asset.last_close,
            )
            for asset in assets
        ],
    )


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: UploadFile = File(..., description="CSV export of historical market data"),
    tenant: Tenant = Depends(require_tenant),
    repository: Repository = Depends(get_repository),
    settings: Settings = Depends(get_settings),
) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Only .csv files are supported.")

    raw_bytes = await file.read()
    if len(raw_bytes) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Uploaded file is too large.")
    if not raw_bytes.strip():
        raise DomainError("The uploaded file is empty.")

    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw_bytes.decode("latin-1", errors="replace")

    try:
        raw_frame = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    except Exception as exc:
        raise DomainError(f"Could not parse the CSV file: {exc}") from exc

    cleaned, report = clean_market_frame(raw_frame, max_rows=settings.max_upload_rows)

    repository.upsert_market_rows(tenant.tenant_id, frame_to_records(cleaned), source=file.filename)
    repository.log_audit(
        tenant.tenant_id,
        "dataset_uploaded",
        f"{file.filename}: {report.rows_accepted} rows across {len(report.assets)} asset(s)",
    )

    registry = get_registry()
    for asset_name in report.assets:
        registry.invalidate(tenant.tenant_id, asset_name)
    get_cache().invalidate(f"p10:forecast:{tenant.tenant_id}:")
    get_cache().invalidate(f"p10:scenario:{tenant.tenant_id}:")

    assets = repository.list_assets(tenant.tenant_id)
    return UploadResponse(
        asset_count=len(report.assets),
        rows_written=report.rows_accepted,
        etl=report.as_dict(),
        dataset=_summary(tenant, repository, assets),
    )


@router.get("/summary", response_model=DatasetSummaryResponse)
def dataset_summary(
    tenant: Tenant = Depends(require_tenant),
    repository: Repository = Depends(get_repository),
) -> DatasetSummaryResponse:
    return _summary(tenant, repository, repository.list_assets(tenant.tenant_id))


@router.get("/{asset_name}/series", response_model=SeriesResponse)
def asset_series(
    asset_name: str,
    limit: int = Query(500, ge=10, le=5000),
    tenant: Tenant = Depends(require_tenant),
    repository: Repository = Depends(get_repository),
) -> SeriesResponse:
    rows = repository.get_series(tenant.tenant_id, asset_name, limit=limit)
    if not rows:
        raise HTTPException(status_code=404, detail=f"No stored data for asset '{asset_name}'.")

    points: list[SeriesPoint] = []
    for row in rows:
        points.append(
            SeriesPoint(
                date=str(row["date_logged"]),
                closing_price=float(row["closing_price"]),
                volume=float(row["volume"] or 0.0),
                zar_usd=_optional_float(row.get("zar_usd")),
                cpi=_optional_float(row.get("cpi")),
                fuel_price=_optional_float(row.get("fuel_price")),
                repo_rate=_optional_float(row.get("repo_rate")),
            )
        )

    return SeriesResponse(asset_name=asset_name, count=len(points), points=points)


@router.delete("/{asset_name}", status_code=status.HTTP_200_OK)
def delete_asset(
    asset_name: str,
    tenant: Tenant = Depends(require_tenant),
    repository: Repository = Depends(get_repository),
) -> dict[str, Any]:
    deleted = repository.delete_asset(tenant.tenant_id, asset_name)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"No stored data for asset '{asset_name}'.")

    get_registry().invalidate(tenant.tenant_id, asset_name)
    get_cache().invalidate(f"p10:forecast:{tenant.tenant_id}:{asset_name}:")
    get_cache().invalidate(f"p10:scenario:{tenant.tenant_id}:{asset_name}:")
    repository.log_audit(tenant.tenant_id, "asset_deleted", f"{asset_name}: {deleted} rows")
    return {"asset_name": asset_name, "rows_deleted": deleted}


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if value != value:  # NaN
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
