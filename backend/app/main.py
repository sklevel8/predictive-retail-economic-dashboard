"""FastAPI application factory and lifecycle wiring."""

from __future__ import annotations

import hashlib
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import datasets, forecasting, health, tenant
from app.config import Settings, get_settings
from app.core.errors import DomainError
from app.core.security import hash_api_key
from app.db.repository import get_repository


def seed_demo_tenants(settings: Settings) -> None:
    """Create the two sandbox tenants used by the demo dashboard and tests."""

    if not settings.demo_mode:
        return

    repository = get_repository()
    for api_key, company_name in settings.demo_api_keys:
        tenant_id = "tenant-" + hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]
        repository.create_tenant(tenant_id, company_name, hash_api_key(api_key))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    get_repository().ensure_schema()
    seed_demo_tenants(settings)
    yield


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description=(
            "Predictive Retail & Economic Data Dashboard API. "
            "Tenant isolation is enforced on every query; authenticate with X-API-Key."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Process-Time-ms"],
    )

    @app.middleware("http")
    async def add_process_time(request: Request, call_next):  # type: ignore[no-untyped-def]
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Process-Time-ms"] = f"{elapsed_ms:.2f}"
        return response

    @app.exception_handler(DomainError)
    async def domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message, "error": exc.__class__.__name__},
        )

    app.include_router(health.router)
    app.include_router(health.router, prefix=settings.api_prefix)
    app.include_router(tenant.router, prefix=settings.api_prefix)
    app.include_router(datasets.router, prefix=settings.api_prefix)
    app.include_router(forecasting.router, prefix=settings.api_prefix)

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {
            "service": settings.app_name,
            "version": settings.version,
            "docs": "/docs",
            "health": "/health",
            "api": settings.api_prefix,
        }

    @app.get("/ready", include_in_schema=False)
    def ready() -> dict[str, str]:
        return {"status": "ready"}

    return app


app = create_app()
