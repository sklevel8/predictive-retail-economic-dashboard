"""Pydantic request/response contracts for the public API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

MAX_HORIZON = 366


class ForecastRequest(BaseModel):
    asset_name: str = Field(..., min_length=1, max_length=120, examples=["Standard Bank"])
    horizon_days: int = Field(30, ge=1, le=MAX_HORIZON, description="Business days to project.")
    include_explanation: bool = Field(
        False, description="Attach a SHAP attribution for the first forecast step."
    )

    @field_validator("asset_name")
    @classmethod
    def _strip_asset(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("asset_name must not be blank")
        return value


class ScenarioShocks(BaseModel):
    """Proportional shocks applied to macro inputs for the whole horizon."""

    zar_usd_pct: float = Field(0.0, ge=-0.95, le=5.0, description="-0.05 = rand strengthens 5%")
    cpi_pct: float = Field(0.0, ge=-0.95, le=5.0)
    fuel_price_pct: float = Field(0.0, ge=-0.95, le=5.0)
    repo_rate_pct: float = Field(0.0, ge=-0.95, le=5.0)
    volume_pct: float = Field(0.0, ge=-0.95, le=5.0)


class ScenarioRequest(BaseModel):
    asset_name: str = Field(..., min_length=1, max_length=120, examples=["Naspers"])
    horizon_days: int = Field(30, ge=1, le=MAX_HORIZON)
    shocks: ScenarioShocks = Field(default_factory=ScenarioShocks)
    include_explanation: bool = False

    @field_validator("asset_name")
    @classmethod
    def _strip_asset(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("asset_name must not be blank")
        return value


class ForecastPoint(BaseModel):
    step: int
    date: str
    predicted_price: float
    predicted_log_return: float


class Explanation(BaseModel):
    asset_name: str
    algorithm: str
    method: str
    last_close: float
    predicted_price: float
    prediction_log_return: float
    base_value_log_return: float
    reconstructed_log_return: float
    local_accuracy_delta: float
    contributions_log_return: dict[str, float]
    contribution_labels: dict[str, str]
    contribution_price_impact: dict[str, float]


class ForecastResponse(BaseModel):
    asset_name: str
    algorithm: str
    horizon_days: int
    last_observed_date: str
    last_close: float
    generated_at: str
    cached: bool
    latency_ms: float
    points: list[ForecastPoint]
    explanation: Explanation | None = None


class ScenarioSummary(BaseModel):
    baseline_end_price: float
    scenario_end_price: float
    absolute_change: float
    percent_change: float
    average_price_delta: float
    max_price_delta: float
    direction: Literal["up", "down", "flat"]


class ScenarioResponse(BaseModel):
    asset_name: str
    algorithm: str
    horizon_days: int
    last_observed_date: str
    last_close: float
    generated_at: str
    latency_ms: float
    shocks: ScenarioShocks
    baseline: list[ForecastPoint]
    scenario: list[ForecastPoint]
    summary: ScenarioSummary
    explanation: Explanation | None = None


class AssetSummaryModel(BaseModel):
    asset_name: str
    row_count: int
    first_date: str
    last_date: str
    last_close: float


class DatasetSummaryResponse(BaseModel):
    tenant_id: str
    company_name: str
    asset_count: int
    total_rows: int
    assets: list[AssetSummaryModel]


class SeriesPoint(BaseModel):
    date: str
    closing_price: float
    volume: float = 0.0
    zar_usd: float | None = None
    cpi: float | None = None
    fuel_price: float | None = None
    repo_rate: float | None = None


class SeriesResponse(BaseModel):
    asset_name: str
    count: int
    points: list[SeriesPoint]


class UploadResponse(BaseModel):
    asset_count: int
    rows_written: int
    etl: dict[str, Any]
    dataset: DatasetSummaryResponse


class ModelMetricsResponse(BaseModel):
    asset_name: str
    algorithm: str
    trained_at: str
    rows_trained: int
    mae: float
    rmse: float
    mape: float
    directional_accuracy: float


class TrainResponse(BaseModel):
    asset_name: str
    algorithm: str
    rows_trained: int
    mae: float
    rmse: float
    mape: float
    directional_accuracy: float
    trained_at: str
    feature_importance: dict[str, float]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    environment: str
    cache: dict[str, Any]
    ml: dict[str, Any]
    database: str
