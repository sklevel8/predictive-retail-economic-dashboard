"""Deterministic synthetic market data used by the test-suite."""

from __future__ import annotations

import numpy as np
import pandas as pd


def market_frame(
    asset: str = "Test Asset",
    rows: int = 320,
    seed: int = 11,
    start: str = "2024-01-02",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=rows)

    usd = 18.0 + np.cumsum(rng.normal(0, 0.03, rows))
    fuel = 22.0 + np.cumsum(rng.normal(0, 0.05, rows))
    cpi = 100.0 + np.cumsum(np.full(rows, 0.12) + rng.normal(0, 0.02, rows))
    repo = np.clip(8.0 + np.cumsum(rng.normal(0, 0.01, rows)), 5.0, 12.0)

    usd_ret = np.diff(np.log(usd), prepend=np.log(usd[0]))
    fuel_ret = np.diff(np.log(fuel), prepend=np.log(fuel[0]))

    returns = -0.9 * usd_ret + 0.6 * fuel_ret + rng.normal(0, 0.004, rows)
    price = 100.0 * np.exp(np.cumsum(returns))

    return pd.DataFrame(
        {
            "asset_name": asset,
            "date_logged": dates.strftime("%Y-%m-%d"),
            "closing_price": np.round(price, 2),
            "volume": np.round(np.abs(rng.normal(500_000, 90_000, rows))).astype(int),
            "zar_usd": np.round(usd, 4),
            "cpi": np.round(cpi, 2),
            "fuel_price": np.round(fuel, 3),
            "repo_rate": np.round(repo, 2),
        }
    )


def dirty_frame(rows: int = 300, seed: int = 23) -> pd.DataFrame:
    """A frame with duplicates, bad dates, missing and impossible values."""

    frame = market_frame(rows=rows, seed=seed)
    frame.loc[10:24, "zar_usd"] = np.nan
    frame.loc[40:43, "cpi"] = -1.0
    frame.loc[60, "date_logged"] = "31/31/2024"
    frame.loc[61, "closing_price"] = np.nan
    frame.loc[62, "closing_price"] = -12.5
    frame.loc[63, "volume"] = -500
    duplicates = frame.iloc[100:105].copy()
    frame = pd.concat([frame, duplicates], ignore_index=True)
    return frame.sample(frac=1.0, random_state=7).reset_index(drop=True)


def upload(client, headers: dict[str, str], frame: pd.DataFrame, name: str = "dataset.csv"):
    payload = frame.to_csv(index=False).encode("utf-8")
    return client.post(
        "/api/v1/datasets/upload",
        headers=headers,
        files={"file": (name, payload, "text/csv")},
    )


def ensure_asset(client, headers: dict[str, str], asset: str, seed: int = 11):
    """Upload (idempotently) a synthetic asset and return the response."""

    response = upload(client, headers, market_frame(asset=asset, seed=seed), f"{asset}.csv")
    assert response.status_code == 201, response.text
    return response
