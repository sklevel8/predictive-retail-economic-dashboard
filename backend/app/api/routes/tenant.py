"""Caller identity endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.security import require_tenant
from app.db.repository import Repository, Tenant, get_repository

router = APIRouter(tags=["tenant"])


@router.get("/tenant")
def current_tenant(
    tenant: Tenant = Depends(require_tenant),
    repository: Repository = Depends(get_repository),
) -> dict[str, object]:
    assets = repository.list_assets(tenant.tenant_id)
    return {
        "tenant_id": tenant.tenant_id,
        "company_name": tenant.company_name,
        "asset_count": len(assets),
        "total_rows": sum(asset.row_count for asset in assets),
        "assets": [asset.asset_name for asset in assets],
    }
