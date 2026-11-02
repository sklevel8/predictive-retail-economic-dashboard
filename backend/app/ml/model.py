"""Predictive modelling engine with an explainability (XAI) layer.

Design notes
------------
* CatBoost is used when the optional dependency is installed and there is
  enough data; otherwise a scikit-learn Random Forest is used, and a regularised
  linear model is the last-resort fallback.  The API contract is identical.
* The SHAP explainer is built **once** during ``train()`` and cached on the
  instance.  Re-creating ``TreeExplainer`` per request caused the latency spike
  documented as Bug-001 in the test plan.
* All numeric outputs are cast to native Python floats before leaving this
  module (Bug-002: NumPy float32 broke JSON serialisation).
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.core.errors import InsufficientDataError
from app.ml.features import (
    DATE_COLUMN,
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    PRICE_COLUMN,
    TARGET_COLUMN,
    latest_feature_row,
    training_matrix,
)

try:  # pragma: no cover - optional dependency
    from catboost import CatBoostRegressor

    CATBOOST_AVAILABLE = True
except Exception:  # pragma: no cover
    CatBoostRegressor = None  # type: ignore[assignment]
    CATBOOST_AVAILABLE = False

try:  # pragma: no cover - optional dependency
    import shap

    SHAP_AVAILABLE = True
except Exception:  # pragma: no cover
    shap = None  # type: ignore[assignment]
    SHAP_AVAILABLE = False


import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="shap")

MIN_TRAIN_ROWS = 40
MAX_STEP_RETURN = 0.10  # clip pathological daily moves to +/-10%
MAX_HISTORY_WINDOW = 120


@dataclass
class TrainResult:
    asset_name: str
    algorithm: str
    rows_trained: int
    mae: float
    rmse: float
    mape: float
    directional_accuracy: float
    trained_at: str
    feature_importance: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "asset_name": self.asset_name,
            "algorithm": self.algorithm,
            "rows_trained": self.rows_trained,
            "mae": self.mae,
            "rmse": self.rmse,
            "mape": self.mape,
            "directional_accuracy": self.directional_accuracy,
            "trained_at": self.trained_at,
            "feature_importance": self.feature_importance,
        }


class JSEPredictiveModel:
    """One model instance per (tenant, asset) pair."""

    def __init__(self, asset_name: str) -> None:
        self.asset_name = asset_name
        self.model: Any | None = None
        self.algorithm: str = "untrained"
        self.feature_names: list[str] = list(FEATURE_COLUMNS)
        self.metrics: dict[str, Any] = {}
        self._explainer: Any | None = None
        self._explainer_kind: str = "none"
        self._base_value: float | None = None
        self._background: np.ndarray | None = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, frame: pd.DataFrame) -> TrainResult:
        matrix = training_matrix(frame)
        if len(matrix) < MIN_TRAIN_ROWS:
            raise InsufficientDataError(len(matrix), MIN_TRAIN_ROWS)

        split = max(int(len(matrix) * 0.8), len(matrix) - 15)
        split = min(split, len(matrix) - 1)
        train_rows = matrix.iloc[:split]
        valid_rows = matrix.iloc[split:]

        x_train = train_rows[FEATURE_COLUMNS].to_numpy(dtype=float)
        y_train = train_rows[TARGET_COLUMN].to_numpy(dtype=float)

        estimator, algorithm = _build_estimator(len(train_rows))
        estimator.fit(x_train, y_train)

        with self._lock:
            self.model = estimator
            self.algorithm = algorithm
            # Bug-001 fix: explainer is created once, here, and reused.
            self._explainer, self._explainer_kind, self._base_value = _build_explainer(
                estimator, x_train
            )
            sample_size = min(200, len(x_train))
            rng = np.random.default_rng(42)
            indices = rng.choice(len(x_train), size=sample_size, replace=False)
            self._background = x_train[indices]

        metrics = self._evaluate(valid_rows if not valid_rows.empty else train_rows)
        metrics["feature_importance"] = self._feature_importance()

        result = TrainResult(
            asset_name=self.asset_name,
            algorithm=algorithm,
            rows_trained=len(train_rows),
            mae=metrics["mae"],
            rmse=metrics["rmse"],
            mape=metrics["mape"],
            directional_accuracy=metrics["directional_accuracy"],
            trained_at=datetime.now(UTC).isoformat(timespec="seconds"),
            feature_importance=metrics["feature_importance"],
        )
        self.metrics = result.as_dict()
        return result

    def _evaluate(self, rows: pd.DataFrame) -> dict[str, float]:
        if rows.empty or self.model is None:
            return {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "directional_accuracy": 0.0}

        x = rows[FEATURE_COLUMNS].to_numpy(dtype=float)
        y_true = rows[TARGET_COLUMN].to_numpy(dtype=float)
        y_pred = np.asarray(self.model.predict(x), dtype=float)

        last_close = rows[PRICE_COLUMN].to_numpy(dtype=float)
        actual_price = last_close * np.exp(y_true)
        predicted_price = last_close * np.exp(y_pred)

        errors = predicted_price - actual_price
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors**2)))
        non_zero = np.abs(actual_price) > 1e-9
        mape = float(np.mean(np.abs(errors[non_zero] / actual_price[non_zero])) * 100.0) if non_zero.any() else 0.0
        direction = float(np.mean(np.sign(y_true) == np.sign(y_pred)) * 100.0)

        return {"mae": mae, "rmse": rmse, "mape": mape, "directional_accuracy": direction}

    def _feature_importance(self) -> dict[str, float]:
        if self.model is None:
            return {}
        estimator = self.model
        if isinstance(estimator, Pipeline):
            estimator = estimator[-1]
        importances = getattr(estimator, "feature_importances_", None)
        if importances is None:
            return {}
        total = float(np.sum(importances)) or 1.0
        ranked = sorted(
            (
                (name, float(value / total))
                for name, value in zip(FEATURE_COLUMNS, importances, strict=False)
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        return {name: round(value, 6) for name, value in ranked}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @property
    def is_trained(self) -> bool:
        return self.model is not None

    def predict_return(self, feature_row: pd.DataFrame) -> float:
        if self.model is None:
            raise RuntimeError("Model has not been trained.")
        values = feature_row[FEATURE_COLUMNS].to_numpy(dtype=float)
        return float(np.asarray(self.model.predict(values), dtype=float)[0])

    def predict_with_explanation(self, feature_row: pd.DataFrame) -> dict[str, Any]:
        """Predict the next log return and attribute it to each feature."""

        if self.model is None:
            raise RuntimeError("Model has not been trained.")

        values = feature_row[FEATURE_COLUMNS].to_numpy(dtype=float)
        prediction = float(np.asarray(self.model.predict(values), dtype=float)[0])

        with self._lock:
            contributions, base_value, method = self._shap_values(values)

        contributions = [float(value) for value in contributions]
        base_value = float(base_value)
        reconstructed = base_value + float(sum(contributions))

        pairs = sorted(
            zip(FEATURE_COLUMNS, contributions, strict=False),
            key=lambda item: abs(item[1]),
            reverse=True,
        )

        return {
            "prediction_log_return": prediction,
            "base_value_log_return": base_value,
            "contributions_log_return": {name: value for name, value in pairs},
            "contribution_labels": {name: FEATURE_LABELS.get(name, name) for name, _ in pairs},
            "reconstructed_log_return": reconstructed,
            "local_accuracy_delta": reconstructed - prediction,
            "method": method,
        }

    def _shap_values(self, values: np.ndarray) -> tuple[np.ndarray, float, str]:
        if self._explainer is not None:
            try:
                raw = self._explainer.shap_values(values)
                vector = _normalise_shap_output(raw)
                base = _normalise_base_value(self._explainer.expected_value)
                if vector is not None and base is not None:
                    return vector, base, self._explainer_kind
            except Exception:  # pragma: no cover - defensive
                pass

        return self._permutation_shap(values)

    def _permutation_shap(self, values: np.ndarray, samples: int = 40) -> tuple[np.ndarray, float, str]:
        """Deterministic permutation-sampling Shapley approximation (fallback)."""

        assert self.model is not None
        background = self._background
        if background is None:
            background = values
        n_features = len(FEATURE_COLUMNS)
        rng = np.random.default_rng(7)
        totals = np.zeros(n_features, dtype=float)

        baseline = float(np.mean(self.model.predict(background)))

        for _ in range(samples):
            order = rng.permutation(n_features)
            coalition = background[np.newaxis, :].copy()
            previous = float(np.mean(self.model.predict(coalition)))
            for index in order:
                coalition[:, index] = values[0, index]
                current = float(np.mean(self.model.predict(coalition)))
                totals[index] += current - previous
                previous = current

        return totals / samples, baseline, "permutation_shap"

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------
    def forecast(
        self,
        frame: pd.DataFrame,
        horizon: int,
        macro_shocks: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """Recursive multi-step forecast.

        ``macro_shocks`` maps a macro column to a proportional shock, e.g.
        ``{"zar_usd": -0.05}`` means "the rand strengthens 5% for the whole
        forecast horizon".
        """

        if self.model is None:
            raise RuntimeError("Model has not been trained.")
        if horizon < 1:
            return []

        shocks = {key: float(value) for key, value in (macro_shocks or {}).items()}
        history = frame.copy()
        history[DATE_COLUMN] = pd.to_datetime(history[DATE_COLUMN])
        history = history.sort_values(DATE_COLUMN).reset_index(drop=True)
        if len(history) < 30:
            raise InsufficientDataError(len(history), 30)

        volume_shock = shocks.get("volume", 0.0)
        macro_shocks = {key: value for key, value in shocks.items() if key != "volume"}

        last_date = pd.to_datetime(history[DATE_COLUMN].iloc[-1])
        business_days = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=horizon)
        last_row = history.iloc[-1]
        base_macro = {
            column: float(last_row.get(column) or 0.0)
            for column in ("zar_usd", "cpi", "fuel_price", "repo_rate")
        }
        base_volume = float(last_row.get("volume") or 0.0)

        projections: list[dict[str, Any]] = []

        for step, next_date in enumerate(business_days):
            features = latest_feature_row(history.tail(MAX_HISTORY_WINDOW))
            if features.empty:
                break

            predicted_return = self.predict_return(features)
            predicted_return = float(np.clip(predicted_return, -MAX_STEP_RETURN, MAX_STEP_RETURN))

            previous_close = float(history[PRICE_COLUMN].iloc[-1])
            next_close = max(0.01, previous_close * math.exp(predicted_return))

            new_row = {
                "asset_name": self.asset_name,
                DATE_COLUMN: next_date.normalize(),
                PRICE_COLUMN: next_close,
                "volume": max(0.0, base_volume * (1.0 + volume_shock)),
            }
            for column, value in base_macro.items():
                new_row[column] = value * (1.0 + macro_shocks.get(column, 0.0))

            history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

            projections.append(
                {
                    "step": step + 1,
                    "date": next_date.strftime("%Y-%m-%d"),
                    "predicted_price": float(round(next_close, 4)),
                    "predicted_log_return": float(round(predicted_return, 8)),
                }
            )

        return projections

    def explain_latest(self, frame: pd.DataFrame) -> dict[str, Any]:
        features = latest_feature_row(frame.tail(MAX_HISTORY_WINDOW))
        if features.empty:
            raise InsufficientDataError(len(frame), 30)

        explanation = self.predict_with_explanation(features)
        last_close = float(frame.sort_values(DATE_COLUMN)[PRICE_COLUMN].iloc[-1])
        price_impacts = {
            name: round(value * last_close, 4)
            for name, value in explanation["contributions_log_return"].items()
        }
        explanation["last_close"] = last_close
        explanation["contribution_price_impact"] = price_impacts
        explanation["predicted_price"] = round(last_close * math.exp(explanation["prediction_log_return"]), 4)
        explanation["asset_name"] = self.asset_name
        explanation["algorithm"] = self.algorithm
        return explanation

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        payload = {
            "asset_name": self.asset_name,
            "algorithm": self.algorithm,
            "metrics": self.metrics,
            "explainer_kind": self._explainer_kind,
            "model": self.model,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str) -> JSEPredictiveModel:
        payload = joblib.load(path)
        instance = cls(payload["asset_name"])
        instance.model = payload["model"]
        instance.algorithm = payload["algorithm"]
        instance.metrics = payload.get("metrics", {})
        instance._explainer_kind = payload.get("explainer_kind", "none")
        instance._explainer, instance._explainer_kind, instance._base_value = _build_explainer(
            instance.model, None
        )
        return instance


def _build_estimator(n_rows: int) -> tuple[Any, str]:
    if CATBOOST_AVAILABLE and n_rows >= 250:  # pragma: no cover - optional
        return (
            CatBoostRegressor(
                iterations=400,
                depth=6,
                learning_rate=0.03,
                loss_function="RMSE",
                random_seed=42,
                verbose=False,
                allow_writing_files=False,
            ),
            "catboost",
        )

    if n_rows >= 200:
        return (
            GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.9,
                random_state=42,
            ),
            "gradient_boosting",
        )

    return (
        Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "ridge_regression",
    )


def _build_explainer(model: Any, background: np.ndarray | None) -> tuple[Any | None, str, float | None]:
    """Create the SHAP explainer once per trained model (Bug-001)."""

    if not SHAP_AVAILABLE or model is None:
        return None, "none", None

    try:
        explainer = shap.TreeExplainer(model)
        base = _normalise_base_value(explainer.expected_value)
        return explainer, "tree_shap", base
    except Exception:  # pragma: no cover - non-tree models
        pass

    try:
        if background is None:
            return None, "none", None
        explainer = shap.LinearExplainer(model, background)
        base = _normalise_base_value(explainer.expected_value)
        return explainer, "linear_shap", base
    except Exception:  # pragma: no cover
        return None, "none", None


def _normalise_shap_output(raw: Any) -> np.ndarray | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        if not raw:
            return None
        raw = raw[0]
    array = np.asarray(raw, dtype=float)
    if array.ndim == 3:
        array = array[0]
    if array.ndim == 2:
        array = array[0]
    if array.ndim != 1:
        return None
    return array


def _normalise_base_value(raw: Any) -> float | None:
    if raw is None:
        return None
    array = np.asarray(raw, dtype=float).reshape(-1)
    if array.size == 0:
        return None
    return float(array[0])


def to_native(value: Any) -> Any:
    """Recursively cast NumPy scalars to JSON-serialisable Python types (Bug-002)."""

    if isinstance(value, dict):
        return {str(key): to_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_native(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
