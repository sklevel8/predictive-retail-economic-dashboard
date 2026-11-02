"""ETL data-quality tests (integration)."""

from __future__ import annotations

import pandas as pd

from app.etl.pipeline import clean_market_frame
from tests.factories import dirty_frame, market_frame


def test_clean_frame_survives_round_trip() -> None:
    frame = market_frame(rows=120)
    cleaned, report = clean_market_frame(frame)

    assert report.rows_received == 120
    assert report.rows_accepted == 120
    assert report.rows_dropped == 0
    assert cleaned["closing_price"].gt(0).all()
    assert cleaned.notna().all().all()
    assert cleaned["date_logged"].is_monotonic_increasing


def test_dirty_frame_is_repaired_without_crashing() -> None:
    dirty = dirty_frame(rows=300)
    cleaned, report = clean_market_frame(dirty)

    # 300 originals + 5 duplicates - 5 duplicates - 1 bad date - 1 missing price
    # - 1 negative price = 297
    assert report.rows_received == 305
    assert report.duplicate_rows_removed == 5
    assert report.rows_accepted == 297
    assert report.rows_dropped == 8
    assert report.missing_values_imputed > 0
    assert report.invalid_values_corrected >= 6  # bad date, prices, negative CPI/volume
    assert any("non-positive" in warning for warning in report.warnings)

    # Imputation guarantees a complete feature frame for the model.
    assert cleaned["zar_usd"].notna().all()
    assert cleaned["cpi"].notna().all()
    assert (cleaned["cpi"] > 0).all()
    assert (cleaned["volume"] >= 0).all()
    assert (cleaned["closing_price"] > 0).all()
    assert cleaned["date_logged"].str.match(r"^\d{4}-\d{2}-\d{2}$").all()


def test_column_aliases_are_accepted() -> None:
    frame = market_frame(rows=80).rename(
        columns={
            "asset_name": "ticker",
            "date_logged": "date",
            "closing_price": "close",
            "volume": "vol",
            "zar_usd": "usd_zar",
            "cpi": "inflation",
            "fuel_price": "petrol_price",
            "repo_rate": "repo",
        }
    )
    cleaned, report = clean_market_frame(frame)
    assert report.rows_accepted == 80
    assert {"asset_name", "date_logged", "closing_price"}.issubset(cleaned.columns)


def test_empty_upload_is_rejected() -> None:
    from app.core.errors import UploadValidationError

    try:
        clean_market_frame(pd.DataFrame())
    except UploadValidationError as exc:
        assert "no data" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("empty frame should raise")


def test_missing_required_column_is_rejected() -> None:
    from app.core.errors import UploadValidationError

    frame = market_frame(rows=20).drop(columns=["closing_price"])
    try:
        clean_market_frame(frame)
    except UploadValidationError as exc:
        assert "closing_price" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing column should raise")
