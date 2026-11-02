"""Prediction cache backed by Redis when available, otherwise in-process TTL.

Only JSON-serialisable payloads are stored, which keeps the Redis path and the
local path behaviourally identical.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

try:  # pragma: no cover - optional dependency
    import redis

    REDIS_AVAILABLE = True
except Exception:  # pragma: no cover
    redis = None  # type: ignore[assignment]
    REDIS_AVAILABLE = False


class PredictionCache:
    def __init__(self, ttl_seconds: int = 900, redis_url: str | None = None) -> None:
        self.ttl_seconds = ttl_seconds
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()
        self._redis = None
        self.backend = "memory"

        if redis_url and REDIS_AVAILABLE:
            try:  # pragma: no cover - requires a live Redis
                client = redis.Redis.from_url(redis_url, decode_responses=True)
                client.ping()
                self._redis = client
                self.backend = "redis"
            except Exception:
                self._redis = None

    def get(self, key: str) -> Any | None:
        if self._redis is not None:  # pragma: no cover
            try:
                raw = self._redis.get(key)
                return json.loads(raw) if raw is not None else None
            except Exception:
                return None

        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at < time.time():
                self._store.pop(key, None)
                return None
            return value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        ttl = ttl or self.ttl_seconds
        if self._redis is not None:  # pragma: no cover
            try:
                self._redis.set(key, json.dumps(value), ex=ttl)
                return
            except Exception:
                pass

        with self._lock:
            self._store[key] = (time.time() + ttl, value)

    def invalidate(self, prefix: str) -> None:
        if self._redis is not None:  # pragma: no cover
            try:
                for key in self._redis.scan_iter(match=f"{prefix}*"):
                    self._redis.delete(key)
            except Exception:
                pass

        with self._lock:
            for key in [key for key in self._store if key.startswith(prefix)]:
                self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            size = len(self._store)
        return {"backend": self.backend, "entries": size, "ttl_seconds": self.ttl_seconds}


_cache: PredictionCache | None = None


def get_cache() -> PredictionCache:
    global _cache
    if _cache is None:
        from app.config import get_settings

        settings = get_settings()
        _cache = PredictionCache(settings.cache_ttl_seconds, settings.redis_url)
    return _cache


def reset_cache() -> None:
    global _cache
    _cache = None
