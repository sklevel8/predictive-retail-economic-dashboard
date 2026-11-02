"""Feature-engineering tests, including the fast-path parity check."""

from __future__ import annotations

import numpy as np

from app.ml.features import (
    FEATURE_COLUMNS,
    build_features,
    latest_feature_row,
    training_matrix,
)
from tests.factories import market_frame


def test_training_matrix_has_no_nulls_and_no_leakage() -> None:
    frame = market_frame(rows=200)
    matrix = training_matrix(frame)

    assert not matrix.empty
    assert matrix[FEATURE_COLUMNS].notna().all().all()
    assert matrix["target_log_return"].notna().all()

    # The target at row t must equal the return realised at row t+1.
    full = build_features(frame)
    usable = full.dropna(subset=[*FEATURE_COLUMNS, "target_log_return"])
    first_row = usable.index[0]
    assert np.isclose(full.loc[first_row, "target_log_return"], full.loc[first_row + 1, "ret_lag_0"])


def test_fast_path_matches_batch_feature_builder() -> None:
    """latest_feature_row must reproduce the last build_features() row."""

    frame = market_frame(rows=240, seed=17)
    batch = build_features(frame).dropna(subset=FEATURE_COLUMNS).tail(1)
    fast = latest_feature_row(frame)

    for column in FEATURE_COLUMNS:
        assert np.isclose(
            float(batch.iloc[0][column]),
            float(fast.iloc[0][column]),
            rtol=1e-9,
            atol=1e-12,
        ), column


def test_fast_path_survives_missing_macro_data() -> None:
    frame = market_frame(rows=120).drop(columns=["zar_usd", "cpi"])
    row = latest_feature_row(frame)
    assert not row.empty
    assert row.notna().all().all()
    assert float(row.iloc[0]["zar_usd"]) == 0.0


def test_fast_path_requires_enough_history() -> None:
    frame = market_frame(rows=10)
    assert latest_feature_row(frame).empty
