"""API-key authentication and tenant resolution.

API keys are never stored in plain text: only the SHA-256 digest is persisted.
The digest is looked up to resolve the caller's ``tenant_id``, which is then
passed explicitly into every repository call so that no query can ever run
without a tenant predicate.
"""

from __future__ import annotations

import hashlib

from fastapi import Depends, Header, HTTPException, status

from app.db.repository import Repository, Tenant, get_repository

API_KEY_HEADER = "X-API-Key"


def hash_api_key(api_key: str) -> str:
    """Return the hex SHA-256 digest used as the stored credential."""

    return hashlib.sha256(api_key.strip().encode("utf-8")).hexdigest()


def require_tenant(
    x_api_key: str | None = Header(default=None, alias=API_KEY_HEADER),
    repo: Repository = Depends(get_repository),
) -> Tenant:
    """FastAPI dependency that resolves (or rejects) the calling tenant."""

    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    tenant = repo.get_tenant_by_api_key_hash(hash_api_key(x_api_key))
    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return tenant
