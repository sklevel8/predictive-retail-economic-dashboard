"""CSV ingestion and cleaning (the ETL pipeline).

The pipeline is deliberately defensive: a malformed upload must never reach the
model trainer or raise an unhandled 500.  Every rejected row is counted and
reported back to the caller so analysts can fix their source extract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.core.errors import UploadValidationError
from app.db.repository import MARKET_COLUMNS

REQUIRED_COLUMNS: tuple[str, ...] = ("asset_name", "date_logged", "closing_price")
NUMERIC_COLUMNS: tuple[str, ...] = (
    "closing_price",
    "volume",
    "zar_usd",
    "cpi",
    "fuel_price",
    "repo_rate",
)
MACRO_COLUMNS: list[str] = ["zar_usd", "cpi", "fuel_price", "repo_rate"]

# Column aliases accepted from messy enterprise extracts.
COLUMN_ALIASES: dict[str, str] = {
    "asset": "asset_name",
    "ticker": "asset_name",
    "symbol": "asset_name",
    "date": "date_logged",
    "trade_date": "date_logged",
    "close": "closing_price",
    "close_price": "closing_price",
    "price": "closing_price",
    "vol": "volume",
    "usd_zar": "zar_usd",
    "zar_us_d": "zar_usd",
    "exchange_rate": "zar_usd",
    "inflation": "cpi",
    "cpi_index": "cpi",
    "petrol_price": "fuel_price",
    "fuel": "fuel_price",
    "interest_rate": "repo_rate",
    "repo": "repo_rate",
}

MIN_CLOSE = 0.01
MAX_CLOSE = 1_000_000.0


@dataclass
class ETLReport:
    rows_received: int = 0
    rows_accepted: int = 0
    rows_dropped: int = 0
    missing_values_imputed: int = 0
    duplicate_rows_removed: int = 0
    invalid_values_corrected: int = 0
    assets: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows_received": self.rows_received,
            "rows_accepted": self.rows_accepted,
            "rows_dropped": self.rows_dropped,
            "missing_values_imputed": self.missing_values_imputed,
            "duplicate_rows_removed": self.duplicate_rows_removed,
            "invalid_values_corrected": self.invalid_values_corrected,
            "assets": self.assets,
            "warnings": self.warnings,
        }


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed: dict[str, str] = {}
    for column in df.columns:
        key = str(column).strip().lower().replace(" ", "_").replace("-", "_")
        renamed[column] = COLUMN_ALIASES.get(key, key)
    return df.rename(columns=renamed)


def _coerce_numeric(df: pd.DataFrame, report: ETLReport) -> pd.DataFrame:
    for column in NUMERIC_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
            continue
        original_missing = df[column].isna().sum()
        coerced = pd.to_numeric(
            df[column].astype("string").str.replace(r"[,\s]", "", regex=True),
            errors="coerce",
        )
        newly_missing = int(coerced.isna().sum() - original_missing)
        if newly_missing > 0:
            report.invalid_values_corrected += newly_missing
            report.warnings.append(
                f"{newly_missing} non-numeric value(s) in column '{column}' were discarded."
            )
        df[column] = coerced.astype("float64")
    return df


def _sanity_check(df: pd.DataFrame, report: ETLReport) -> pd.DataFrame:
    """Drop or repair physically impossible observations."""

    bad_price = (df["closing_price"] < MIN_CLOSE) | (df["closing_price"] > MAX_CLOSE)
    report.invalid_values_corrected += int(bad_price.sum())
    if bad_price.any():
        report.warnings.append(
            f"{int(bad_price.sum())} row(s) had a non-positive or implausible closing price and were dropped."
        )
    df = df.loc[~bad_price].copy()

    negative_volume = df["volume"] < 0
    if negative_volume.any():
        report.invalid_values_corrected += int(negative_volume.sum())
        report.warnings.append(
            f"{int(negative_volume.sum())} negative volume value(s) were treated as missing."
        )
        df.loc[negative_volume, "volume"] = np.nan

    # Negative CPI / fuel / exchange-rate observations are data-entry errors.
    for column in MACRO_COLUMNS:
        negative = df[column] <= 0
        if negative.any():
            report.invalid_values_corrected += int(negative.sum())
            report.warnings.append(
                f"{int(negative.sum())} non-positive value(s) in '{column}' were treated as missing."
            )
            df.loc[negative, column] = np.nan

    return df


def _impute(df: pd.DataFrame, report: ETLReport) -> pd.DataFrame:
    """Forward-fill macro series per asset, then fall back to the global median."""

    df = df.sort_values(["asset_name", "date_logged"]).reset_index(drop=True)
    missing_before = int(df[MACRO_COLUMNS].isna().sum().sum())

    grouped = df.groupby("asset_name", group_keys=False)
    for column in MACRO_COLUMNS:
        df[column] = grouped[column].transform(lambda series: series.ffill().bfill())
    df[MACRO_COLUMNS] = df[MACRO_COLUMNS].fillna(df[MACRO_COLUMNS].median(numeric_only=True))
    df[MACRO_COLUMNS] = df[MACRO_COLUMNS].bfill().ffill()

    missing_after = int(df[MACRO_COLUMNS].isna().sum().sum())
    report.missing_values_imputed += max(0, missing_before - missing_after)

    if df["volume"].isna().any():
        volume_missing = int(df["volume"].isna().sum())
        df["volume"] = df["volume"].fillna(0.0)
        report.missing_values_imputed += volume_missing

    return df


def clean_market_frame(raw: pd.DataFrame, max_rows: int = 100_000) -> tuple[pd.DataFrame, ETLReport]:
    """Validate, clean and normalise a raw market-data frame."""

    if raw is None or raw.empty:
        raise UploadValidationError("The uploaded file contains no data rows.")

    report = ETLReport(rows_received=len(raw))
    if len(raw) > max_rows:
        raise UploadValidationError(
            f"Upload has {len(raw)} rows which exceeds the {max_rows} row limit."
        )

    df = _normalise_columns(raw.copy())
    missing_required = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_required:
        raise UploadValidationError(
            "Missing required column(s): " + ", ".join(missing_required)
        )

    df = df.loc[:, [c for c in MARKET_COLUMNS if c in df.columns]].copy()
    for column in MARKET_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

    df["asset_name"] = df["asset_name"].astype("string").str.strip()
    blank_asset = df["asset_name"].isna() | (df["asset_name"].str.len() == 0)
    if blank_asset.any():
        report.warnings.append(f"{int(blank_asset.sum())} row(s) had no asset name and were dropped.")
        df = df.loc[~blank_asset].copy()

    df["date_logged"] = pd.to_datetime(
        df["date_logged"].astype("string"), errors="coerce", format="mixed", utc=False
    )
    bad_dates = df["date_logged"].isna()
    if bad_dates.any():
        report.warnings.append(
            f"{int(bad_dates.sum())} row(s) had an unparseable date and were dropped."
        )
    df = df.loc[~bad_dates].copy()

    df = _coerce_numeric(df, report)

    missing_price = df["closing_price"].isna()
    if missing_price.any():
        report.warnings.append(
            f"{int(missing_price.sum())} row(s) had no closing price and were dropped."
        )
    df = df.loc[~missing_price].copy()

    before_dedupe = len(df)
    df["date_logged"] = df["date_logged"].dt.normalize()
    df = df.sort_values(["asset_name", "date_logged"])
    df = df.drop_duplicates(subset=["asset_name", "date_logged"], keep="last")
    report.duplicate_rows_removed = before_dedupe - len(df)

    df = _sanity_check(df, report)
    if df.empty:
        raise UploadValidationError(
            "No rows survived validation. Check dates, asset names and closing prices."
        )

    df = _impute(df, report)

    if df[MACRO_COLUMNS].isna().all(axis=None):
        report.warnings.append(
            "No macro-economic indicators were supplied; forecasts will rely on price dynamics only."
        )

    df["asset_name"] = df["asset_name"].astype(str)
    df["date_logged"] = df["date_logged"].dt.strftime("%Y-%m-%d")
    df = df.sort_values(["asset_name", "date_logged"]).reset_index(drop=True)

    report.rows_accepted = len(df)
    report.rows_dropped = report.rows_received - report.rows_accepted
    report.assets = sorted(df["asset_name"].unique().tolist())
    return df, report


def frame_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a cleaned frame into JSON-safe dictionaries for persistence."""

    records: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        records.append(
            {
                "asset_name": row.asset_name,
                "date_logged": row.date_logged,
                "closing_price": float(row.closing_price),
                "volume": float(row.volume) if pd.notna(row.volume) else 0.0,
                "zar_usd": None if pd.isna(row.zar_usd) else float(row.zar_usd),
                "cpi": None if pd.isna(row.cpi) else float(row.cpi),
                "fuel_price": None if pd.isna(row.fuel_price) else float(row.fuel_price),
                "repo_rate": None if pd.isna(row.repo_rate) else float(row.repo_rate),
            }
        )
    return records
