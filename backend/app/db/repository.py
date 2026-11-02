"""Persistence layer.

Local development and the automated test-suite use SQLite (stdlib) so the
project has zero infrastructure dependencies out of the box.  Production uses
Supabase/PostgreSQL, where the same schema is created by
``supabase/migrations/0001_init.sql`` and Row-Level Security enforces isolation
at the database engine level.

Every public method requires an explicit ``tenant_id`` argument: there is no
API surface that can read across tenants.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import get_settings

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tenants (
    tenant_id     TEXT PRIMARY KEY,
    company_name  TEXT NOT NULL,
    api_key_hash  TEXT NOT NULL UNIQUE,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS economic_data (
    record_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id     TEXT NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    asset_name    TEXT NOT NULL,
    date_logged   TEXT NOT NULL,
    closing_price REAL NOT NULL,
    volume        REAL NOT NULL DEFAULT 0,
    zar_usd       REAL,
    cpi           REAL,
    fuel_price    REAL,
    repo_rate     REAL,
    source        TEXT NOT NULL DEFAULT 'upload',
    ingested_at   TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_economic_data_key
    ON economic_data (tenant_id, asset_name, date_logged);

CREATE INDEX IF NOT EXISTS ix_economic_data_lookup
    ON economic_data (tenant_id, asset_name, date_logged);

CREATE TABLE IF NOT EXISTS model_metrics (
    metric_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id     TEXT NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    asset_name    TEXT NOT NULL,
    algorithm     TEXT NOT NULL,
    trained_at    TEXT NOT NULL,
    rows_trained  INTEGER NOT NULL,
    mae           REAL NOT NULL,
    rmse          REAL NOT NULL,
    mape          REAL NOT NULL,
    directional_accuracy REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    audit_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id   TEXT NOT NULL,
    action      TEXT NOT NULL,
    detail      TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
"""

MARKET_COLUMNS: tuple[str, ...] = (
    "asset_name",
    "date_logged",
    "closing_price",
    "volume",
    "zar_usd",
    "cpi",
    "fuel_price",
    "repo_rate",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Tenant:
    tenant_id: str
    company_name: str


@dataclass(frozen=True)
class AssetSummary:
    asset_name: str
    row_count: int
    first_date: str
    last_date: str
    last_close: float


class Repository:
    """Tenant-scoped SQLite repository."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ensure_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def ensure_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA_SQL)

    # ------------------------------------------------------------------
    # Tenants
    # ------------------------------------------------------------------
    def create_tenant(self, tenant_id: str, company_name: str, api_key_hash: str) -> Tenant:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO tenants (tenant_id, company_name, api_key_hash, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(tenant_id) DO UPDATE SET company_name = excluded.company_name
                """,
                (tenant_id, company_name, api_key_hash, utc_now()),
            )
        return Tenant(tenant_id=tenant_id, company_name=company_name)

    def get_tenant_by_api_key_hash(self, api_key_hash: str) -> Tenant | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT tenant_id, company_name FROM tenants WHERE api_key_hash = ?",
                (api_key_hash,),
            ).fetchone()
        if row is None:
            return None
        return Tenant(tenant_id=row["tenant_id"], company_name=row["company_name"])

    def get_tenant(self, tenant_id: str) -> Tenant | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT tenant_id, company_name FROM tenants WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchone()
        if row is None:
            return None
        return Tenant(tenant_id=row["tenant_id"], company_name=row["company_name"])

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------
    def upsert_market_rows(
        self, tenant_id: str, rows: Sequence[dict[str, Any]], source: str = "upload"
    ) -> int:
        """Insert or refresh rows for a tenant. Returns the number of rows written."""

        if not rows:
            return 0

        ingested_at = utc_now()
        payload = [
            (
                tenant_id,
                str(row["asset_name"]),
                str(row["date_logged"]),
                float(row["closing_price"]),
                float(row.get("volume") or 0.0),
                _maybe_float(row.get("zar_usd")),
                _maybe_float(row.get("cpi")),
                _maybe_float(row.get("fuel_price")),
                _maybe_float(row.get("repo_rate")),
                source,
                ingested_at,
            )
            for row in rows
        ]

        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO economic_data (
                    tenant_id, asset_name, date_logged, closing_price, volume,
                    zar_usd, cpi, fuel_price, repo_rate, source, ingested_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, asset_name, date_logged) DO UPDATE SET
                    closing_price = excluded.closing_price,
                    volume        = excluded.volume,
                    zar_usd       = COALESCE(excluded.zar_usd, economic_data.zar_usd),
                    cpi           = COALESCE(excluded.cpi, economic_data.cpi),
                    fuel_price    = COALESCE(excluded.fuel_price, economic_data.fuel_price),
                    repo_rate     = COALESCE(excluded.repo_rate, economic_data.repo_rate),
                    source        = excluded.source,
                    ingested_at   = excluded.ingested_at
                """,
                payload,
            )
        return len(payload)

    def list_assets(self, tenant_id: str) -> list[AssetSummary]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT asset_name,
                       COUNT(*)        AS row_count,
                       MIN(date_logged) AS first_date,
                       MAX(date_logged) AS last_date
                FROM economic_data
                WHERE tenant_id = ?
                GROUP BY asset_name
                ORDER BY asset_name
                """,
                (tenant_id,),
            ).fetchall()

            summaries: list[AssetSummary] = []
            for row in rows:
                last = conn.execute(
                    """
                    SELECT closing_price FROM economic_data
                    WHERE tenant_id = ? AND asset_name = ?
                    ORDER BY date_logged DESC LIMIT 1
                    """,
                    (tenant_id, row["asset_name"]),
                ).fetchone()
                summaries.append(
                    AssetSummary(
                        asset_name=row["asset_name"],
                        row_count=int(row["row_count"]),
                        first_date=row["first_date"],
                        last_date=row["last_date"],
                        last_close=float(last["closing_price"]) if last else 0.0,
                    )
                )
        return summaries

    def get_series(
        self,
        tenant_id: str,
        asset_name: str,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["tenant_id = ?", "asset_name = ?"]
        params: list[Any] = [tenant_id, asset_name]

        if start_date is not None:
            clauses.append("date_logged >= ?")
            params.append(start_date.isoformat())
        if end_date is not None:
            clauses.append("date_logged <= ?")
            params.append(end_date.isoformat())

        query = (
            "SELECT asset_name, date_logged, closing_price, volume, zar_usd, "
            "cpi, fuel_price, repo_rate FROM economic_data "
            f"WHERE {' AND '.join(clauses)} ORDER BY date_logged ASC"
        )
        if limit is not None:
            # Keep the most recent `limit` observations, still ascending.
            query = (
                "SELECT * FROM ("
                + query.replace("ORDER BY date_logged ASC", "ORDER BY date_logged DESC")
                + f" LIMIT {int(limit)}) ORDER BY date_logged ASC"
            )

        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def row_count(self, tenant_id: str, asset_name: str | None = None) -> int:
        query = "SELECT COUNT(*) AS n FROM economic_data WHERE tenant_id = ?"
        params: list[Any] = [tenant_id]
        if asset_name:
            query += " AND asset_name = ?"
            params.append(asset_name)
        with self.connect() as conn:
            row = conn.execute(query, params).fetchone()
        return int(row["n"]) if row else 0

    def asset_exists(self, tenant_id: str, asset_name: str) -> bool:
        return self.row_count(tenant_id, asset_name) > 0

    def delete_asset(self, tenant_id: str, asset_name: str) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                "DELETE FROM economic_data WHERE tenant_id = ? AND asset_name = ?",
                (tenant_id, asset_name),
            )
            return cur.rowcount

    # ------------------------------------------------------------------
    # Model telemetry
    # ------------------------------------------------------------------
    def save_model_metrics(self, tenant_id: str, metrics: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO model_metrics (
                    tenant_id, asset_name, algorithm, trained_at, rows_trained,
                    mae, rmse, mape, directional_accuracy
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tenant_id,
                    metrics["asset_name"],
                    metrics["algorithm"],
                    metrics["trained_at"],
                    int(metrics["rows_trained"]),
                    float(metrics["mae"]),
                    float(metrics["rmse"]),
                    float(metrics["mape"]),
                    float(metrics["directional_accuracy"]),
                ),
            )

    def latest_metrics(self, tenant_id: str, asset_name: str | None = None) -> dict[str, Any] | None:
        clauses = ["tenant_id = ?"]
        params: list[Any] = [tenant_id]
        if asset_name:
            clauses.append("asset_name = ?")
            params.append(asset_name)

        with self.connect() as conn:
            row = conn.execute(
                f"""
                SELECT * FROM model_metrics
                WHERE {' AND '.join(clauses)}
                ORDER BY trained_at DESC, metric_id DESC
                LIMIT 1
                """,
                params,
            ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data.pop("tenant_id", None)
        return data

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------
    def log_audit(self, tenant_id: str, action: str, detail: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO audit_log (tenant_id, action, detail, created_at) VALUES (?, ?, ?, ?)",
                (tenant_id, action, detail, utc_now()),
            )


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if value != value:  # NaN check without importing numpy
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=1)
def get_repository() -> Repository:
    settings = get_settings()
    return Repository(settings.database_path)


def reset_repository_cache() -> None:
    """Used by tests after swapping the configured database path."""

    get_repository.cache_clear()
