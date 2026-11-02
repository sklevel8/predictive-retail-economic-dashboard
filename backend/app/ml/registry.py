"""In-process registry of trained models.

One estimator is kept per (tenant, asset).  Training is expensive, so the
registry caches the fitted object and only retrains when the caller asks for it
or when the underlying dataset has changed.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.errors import AssetNotFoundError, InsufficientDataError
from app.db.repository import Repository
from app.ml.model import JSEPredictiveModel, TrainResult


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-")
    return cleaned.lower() or "asset"


class ModelRegistry:
    def __init__(self, repository: Repository, model_dir: str | Path, min_training_rows: int) -> None:
        self.repository = repository
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.min_training_rows = min_training_rows
        self._models: dict[tuple[str, str], JSEPredictiveModel] = {}
        self._versions: dict[tuple[str, str], str] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    def frame_for(self, tenant_id: str, asset_name: str) -> pd.DataFrame:
        rows = self.repository.get_series(tenant_id, asset_name)
        if not rows:
            raise AssetNotFoundError(asset_name)
        frame = pd.DataFrame(rows)
        frame["date_logged"] = pd.to_datetime(frame["date_logged"])
        for column in ("closing_price", "volume", "zar_usd", "cpi", "fuel_price", "repo_rate"):
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame

    def version(self, tenant_id: str, asset_name: str) -> str:
        summaries = self.repository.list_assets(tenant_id)
        for summary in summaries:
            if summary.asset_name == asset_name:
                return f"{summary.row_count}:{summary.last_date}"
        raise AssetNotFoundError(asset_name)

    def get(self, tenant_id: str, asset_name: str, force_retrain: bool = False) -> JSEPredictiveModel:
        key = (tenant_id, asset_name)
        version = self.version(tenant_id, asset_name)

        with self._lock:
            cached = self._models.get(key)
            if cached is not None and not force_retrain and self._versions.get(key) == version:
                return cached

        frame = self.frame_for(tenant_id, asset_name)
        if len(frame) < self.min_training_rows:
            raise InsufficientDataError(len(frame), self.min_training_rows)

        model = JSEPredictiveModel(asset_name)
        result = model.train(frame)

        self._persist(tenant_id, asset_name, model, result)

        with self._lock:
            self._models[key] = model
            self._versions[key] = version
        return model

    def retrain(self, tenant_id: str, asset_name: str) -> TrainResult:
        model = self.get(tenant_id, asset_name, force_retrain=True)
        assert model.metrics
        return TrainResult(**{k: model.metrics[k] for k in TrainResult.__dataclass_fields__})

    def invalidate(self, tenant_id: str, asset_name: str | None = None) -> None:
        with self._lock:
            if asset_name is None:
                for key in [k for k in self._models if k[0] == tenant_id]:
                    self._models.pop(key, None)
                    self._versions.pop(key, None)
            else:
                key = (tenant_id, asset_name)
                self._models.pop(key, None)
                self._versions.pop(key, None)

    def warm(self, tenant_id: str, asset_name: str) -> dict[str, Any]:
        model = self.get(tenant_id, asset_name)
        return {"asset_name": asset_name, "algorithm": model.algorithm, "metrics": model.metrics}

    # ------------------------------------------------------------------
    def _persist(
        self, tenant_id: str, asset_name: str, model: JSEPredictiveModel, result: TrainResult
    ) -> None:
        try:
            directory = self.model_dir / _slug(tenant_id)
            directory.mkdir(parents=True, exist_ok=True)
            model.save(str(directory / f"{_slug(asset_name)}.joblib"))
        except Exception:  # pragma: no cover - persistence is best effort
            pass

        self.repository.save_model_metrics(
            tenant_id,
            {
                "asset_name": result.asset_name,
                "algorithm": result.algorithm,
                "trained_at": result.trained_at,
                "rows_trained": result.rows_trained,
                "mae": result.mae,
                "rmse": result.rmse,
                "mape": result.mape,
                "directional_accuracy": result.directional_accuracy,
            },
        )
        self.repository.log_audit(
            tenant_id,
            "model_trained",
            f"{asset_name} trained with {result.algorithm} on {result.rows_trained} rows",
        )
