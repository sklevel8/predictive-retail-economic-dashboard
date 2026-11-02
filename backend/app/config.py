"""Application configuration.

All values are read from the environment once and cached.  Nothing in this
module talks to a database or the network, so it is safe to import anywhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the forecasting API."""

    app_name: str = "Predictive Retail & Economic Data Dashboard"
    version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    environment: str = "development"
    database_path: Path = field(default_factory=lambda: BACKEND_DIR / "data" / "app.db")
    model_dir: Path = field(default_factory=lambda: BACKEND_DIR / "data" / "models")
    redis_url: str | None = None
    cache_ttl_seconds: int = 900
    max_upload_rows: int = 100_000
    max_upload_bytes: int = 50 * 1024 * 1024
    min_training_rows: int = 120
    default_horizon: int = 30
    max_horizon: int = 90
    demo_mode: bool = True
    demo_api_keys: tuple[tuple[str, str], ...] = ()
    cors_origins: tuple[str, ...] = ("*",)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    database_path = Path(
        os.getenv("P10_DB_PATH", str(BACKEND_DIR / "data" / "app.db"))
    ).expanduser()
    model_dir = Path(
        os.getenv("P10_MODEL_DIR", str(BACKEND_DIR / "data" / "models"))
    ).expanduser()
    demo_mode = _env_bool("P10_DEMO_MODE", True)

    demo_keys: list[tuple[str, str]] = []
    if demo_mode:
        demo_keys = [
            (
                os.getenv("P10_DEMO_KEY_A", "p10-demo-tenant-a"),
                "Ubuntu Retail Group (demo)",
            ),
            (
                os.getenv("P10_DEMO_KEY_B", "p10-demo-tenant-b"),
                "Karoo Financial Services (demo)",
            ),
        ]

    origins = tuple(
        origin.strip()
        for origin in os.getenv("P10_CORS_ORIGINS", "*").split(",")
        if origin.strip()
    ) or ("*",)

    return Settings(
        app_name=os.getenv("P10_APP_NAME", Settings.app_name),
        version=os.getenv("P10_VERSION", Settings.version),
        api_prefix=os.getenv("P10_API_PREFIX", "/api/v1"),
        environment=os.getenv("P10_ENV", "development"),
        database_path=database_path,
        model_dir=model_dir,
        redis_url=os.getenv("P10_REDIS_URL") or None,
        cache_ttl_seconds=_env_int("P10_CACHE_TTL_SECONDS", 900),
        max_upload_rows=_env_int("P10_MAX_UPLOAD_ROWS", 100_000),
        max_upload_bytes=_env_int("P10_MAX_UPLOAD_BYTES", 50 * 1024 * 1024),
        min_training_rows=_env_int("P10_MIN_TRAINING_ROWS", 120),
        default_horizon=_env_int("P10_DEFAULT_HORIZON", 30),
        max_horizon=_env_int("P10_MAX_HORIZON", 90),
        demo_mode=demo_mode,
        demo_api_keys=tuple(demo_keys),
        cors_origins=origins,
    )
