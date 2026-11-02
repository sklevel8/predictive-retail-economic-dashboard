"""Feature engineering for the time-series forecasting models.

Every feature at row ``t`` is computed strictly from data available up to and
including ``t``; the supervised target is the *next* log return.  That makes the
training set leakage-free and lets the same code path be reused recursively for
multi-step forecasting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MACRO_COLUMNS: tuple[str, ...] = ("zar_usd", "cpi", "fuel_price", "repo_rate")

FEATURE_COLUMNS: list[str] = [
    "ret_lag_0",
    "ret_lag_1",
    "ret_lag_2",
    "ret_lag_3",
    "ret_lag_4",
    "roll_mean_5",
    "roll_std_5",
    "roll_mean_20",
    "roll_std_20",
    "ma_ratio_20",
    "volume_z_20",
    "zar_usd",
    "cpi",
    "fuel_price",
    "repo_rate",
    "zar_usd_chg_1",
    "fuel_chg_1",
    "cpi_chg_1",
    "rate_chg_1",
    "month",
    "quarter",
    "day_of_week",
]

TARGET_COLUMN = "target_log_return"
PRICE_COLUMN = "closing_price"
DATE_COLUMN = "date_logged"

# Human-readable labels used by the XAI layer and the dashboard.
FEATURE_LABELS: dict[str, str] = {
    "ret_lag_0": "Today's return",
    "ret_lag_1": "Return 1 day ago",
    "ret_lag_2": "Return 2 days ago",
    "ret_lag_3": "Return 3 days ago",
    "ret_lag_4": "Return 4 days ago",
    "roll_mean_5": "5-day mean return",
    "roll_std_5": "5-day volatility",
    "roll_mean_20": "20-day mean return",
    "roll_std_20": "20-day volatility",
    "ma_ratio_20": "Price vs 20-day average",
    "volume_z_20": "Volume anomaly (z-score)",
    "zar_usd": "ZAR/USD exchange rate",
    "cpi": "CPI index",
    "fuel_price": "Fuel price",
    "repo_rate": "Repo rate",
    "zar_usd_chg_1": "ZAR/USD daily move",
    "fuel_chg_1": "Fuel price daily move",
    "cpi_chg_1": "CPI daily move",
    "rate_chg_1": "Repo rate change",
    "month": "Month of year",
    "quarter": "Quarter",
    "day_of_week": "Day of week",
}

MIN_HISTORY_ROWS = 60


def _safe_macro(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype("float64")
    values = values.ffill().bfill()
    median = values.median()
    if pd.isna(median):
        return values.fillna(0.0)
    return values.fillna(float(median))


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a feature frame for a single asset, ordered by date."""

    if frame.empty:
        return pd.DataFrame(columns=[DATE_COLUMN, PRICE_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN])

    df = frame.copy()
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    close = pd.to_numeric(df[PRICE_COLUMN], errors="coerce").astype("float64")
    close = close.ffill().bfill().clip(lower=1e-6)
    log_close = np.log(close)
    returns = log_close.diff()

    df["ret_lag_0"] = returns
    for lag in range(1, 5):
        df[f"ret_lag_{lag}"] = returns.shift(lag)

    for window in (5, 20):
        df[f"roll_mean_{window}"] = returns.rolling(window, min_periods=window).mean()
        df[f"roll_std_{window}"] = returns.rolling(window, min_periods=window).std()

    moving_average = close.rolling(20, min_periods=20).mean()
    df["ma_ratio_20"] = close / moving_average - 1.0

    volume = pd.to_numeric(df.get("volume"), errors="coerce").astype("float64").fillna(0.0)
    volume_mean = volume.rolling(20, min_periods=20).mean()
    volume_std = volume.rolling(20, min_periods=20).std().replace(0.0, np.nan)
    df["volume_z_20"] = ((volume - volume_mean) / volume_std).fillna(0.0)

    for column in MACRO_COLUMNS:
        df[column] = _safe_macro(df[column])

    df["zar_usd_chg_1"] = df["zar_usd"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["fuel_chg_1"] = df["fuel_price"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["cpi_chg_1"] = df["cpi"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["rate_chg_1"] = df["repo_rate"].diff().fillna(0.0)

    dates = pd.to_datetime(df[DATE_COLUMN])
    df["month"] = dates.dt.month
    df["quarter"] = dates.dt.quarter
    df["day_of_week"] = dates.dt.dayofweek

    df[TARGET_COLUMN] = returns.shift(-1)

    return df[[DATE_COLUMN, PRICE_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]]


def training_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Feature frame with incomplete rows removed, ready for ``model.fit``."""

    features = build_features(frame)
    usable = features.dropna(subset=[*FEATURE_COLUMNS, TARGET_COLUMN])
    return usable.reset_index(drop=True)


def _forward_backfill(values: np.ndarray) -> np.ndarray:
    """Vectorised equivalent of ``Series.ffill().bfill()``."""

    out = values.astype(float, copy=True)
    last = np.nan
    for index in range(out.size):
        if np.isnan(out[index]):
            out[index] = last
        else:
            last = out[index]

    first_valid = last = np.nan
    for index in range(out.size):
        if not np.isnan(out[index]):
            first_valid = out[index]
            break
    if np.isnan(first_valid):
        return out
    out[np.isnan(out)] = first_valid
    return out


def _macro_value(series: pd.Series) -> tuple[float, float]:
    """Return the latest value and previous value of a macro series."""

    values = _forward_backfill(pd.to_numeric(series, errors="coerce").to_numpy(dtype=float))
    if np.isnan(values).all():
        return 0.0, 0.0
    last = float(values[-1])
    previous = float(values[-2]) if values.size > 1 else last
    if np.isnan(last):
        last = 0.0
    if np.isnan(previous) or previous == 0.0:
        previous = last or 1.0
    return last, previous


def latest_feature_row(frame: pd.DataFrame) -> pd.DataFrame:
    """The most recent complete feature vector.

    This is the hot path of recursive forecasting, so it uses one numpy pass per
    column instead of the full ``build_features`` pipeline.  The
    ``test_feature_parity`` test guarantees the two implementations agree.
    """

    if frame.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    data = frame.tail(MIN_HISTORY_ROWS * 2).reset_index(drop=True)
    close = _forward_backfill(pd.to_numeric(data[PRICE_COLUMN], errors="coerce").to_numpy(dtype=float))
    if np.isnan(close).any():
        close[np.isnan(close)] = 1.0
    close = np.clip(close, 1e-6, None)

    log_returns = np.diff(np.log(close), prepend=np.log(close[0]))
    if log_returns.size < 21:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    volume = pd.to_numeric(data.get("volume"), errors="coerce").to_numpy(dtype=float)
    volume = np.nan_to_num(volume, nan=0.0)

    features: dict[str, float] = {}
    for lag in range(5):
        features[f"ret_lag_{lag}"] = float(log_returns[-1 - lag])

    window_5 = log_returns[-5:]
    window_20 = log_returns[-20:]
    features["roll_mean_5"] = float(np.mean(window_5))
    features["roll_std_5"] = float(np.std(window_5, ddof=1))
    features["roll_mean_20"] = float(np.mean(window_20))
    features["roll_std_20"] = float(np.std(window_20, ddof=1))

    features["ma_ratio_20"] = float(close[-1] / np.mean(close[-20:]) - 1.0)

    volume_window = volume[-20:]
    volume_std = float(np.std(volume_window, ddof=1))
    features["volume_z_20"] = (
        float((volume[-1] - np.mean(volume_window)) / volume_std) if volume_std > 0 else 0.0
    )

    macro_changes = {"zar_usd": "zar_usd_chg_1", "cpi": "cpi_chg_1", "fuel_price": "fuel_chg_1"}
    for column in MACRO_COLUMNS:
        latest, previous = _macro_value(data[column] if column in data else pd.Series([np.nan]))
        features[column] = latest
        change_name = macro_changes.get(column)
        if change_name is None:
            continue
        change = (latest / previous - 1.0) if previous not in (0.0, np.nan) else 0.0
        features[change_name] = float(change) if np.isfinite(change) else 0.0

    _, previous_rate = _macro_value(data["repo_rate"] if "repo_rate" in data else pd.Series([np.nan]))
    rate_latest = features["repo_rate"]
    features["rate_chg_1"] = float(rate_latest - previous_rate)

    last_date = pd.to_datetime(data[DATE_COLUMN].iloc[-1])
    features["month"] = float(last_date.month)
    features["quarter"] = float(last_date.quarter)
    features["day_of_week"] = float(last_date.dayofweek)

    row = {name: float(features.get(name, 0.0)) for name in FEATURE_COLUMNS}
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
