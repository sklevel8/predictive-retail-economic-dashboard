"""Generate deterministic demo data for the dashboard.

Usage
-----
    python scripts/generate_demo_data.py --months 18
    python scripts/generate_demo_data.py --start 2016-10-03 --end 2026-08-31

The generator simulates correlated JSE price paths driven by a ZAR/USD
exchange rate, fuel price, CPI index and repo rate, so the models have genuine
signal to learn instead of pure noise.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ASSETS: dict[str, dict[str, float]] = {
    "Standard Bank": {"start": 145.0, "usd_beta": -0.55, "fuel_beta": -0.05, "drift": 0.00018, "vol": 0.013},
    "Anglo American": {"start": 178.0, "usd_beta": 0.45, "fuel_beta": 0.10, "drift": 0.00012, "vol": 0.019},
    "Naspers": {"start": 2_180.0, "usd_beta": -0.75, "fuel_beta": -0.02, "drift": 0.00022, "vol": 0.021},
}


def simulate(start: str, end: str, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)

    # ---- Macro drivers -------------------------------------------------
    usd = 14.2 * np.exp(np.cumsum(rng.normal(0.00025, 0.007, n)))
    fuel = 20.5 * np.exp(np.cumsum(rng.normal(0.0004, 0.009, n)))
    cpi = 100.0 * np.exp(np.cumsum(np.full(n, 0.0045) + rng.normal(0, 0.0012, n)))
    repo = np.clip(6.75 + np.cumsum(rng.normal(0, 0.02, n)), 3.5, 11.75)

    usd_ret = np.concatenate([[0.0], np.diff(np.log(usd))])
    fuel_ret = np.concatenate([[0.0], np.diff(np.log(fuel))])

    frames: list[pd.DataFrame] = []
    for asset, params in ASSETS.items():
        noise = rng.normal(0, params["vol"], n)
        returns = (
            params["drift"]
            + params["usd_beta"] * usd_ret
            + params["fuel_beta"] * fuel_ret
            + noise
        )
        # A couple of regime shifts so the series is not a pure random walk.
        for shock_day, magnitude in ((int(n * 0.35), -0.055), (int(n * 0.68), 0.045)):
            returns[shock_day : shock_day + 6] += magnitude

        price = params["start"] * np.exp(np.cumsum(returns))
        volume = np.round(np.abs(rng.normal(1_200_000, 350_000, n))).astype(int)

        frames.append(
            pd.DataFrame(
                {
                    "asset_name": asset,
                    "date_logged": dates.strftime("%Y-%m-%d"),
                    "closing_price": np.round(price, 2),
                    "volume": volume,
                    "zar_usd": np.round(usd, 4),
                    "cpi": np.round(cpi, 2),
                    "fuel_price": np.round(fuel, 3),
                    "repo_rate": np.round(repo, 2),
                }
            )
        )

    return pd.concat(frames, ignore_index=True).sort_values(
        ["asset_name", "date_logged"]
    ).reset_index(drop=True)


def make_dirty(frame: pd.DataFrame) -> pd.DataFrame:
    """Inject realistic data-quality defects for the ETL tests/demo."""

    dirty = frame.copy()
    dirty.loc[dirty.index[10:25], "zar_usd"] = np.nan
    dirty.loc[dirty.index[40:44], "cpi"] = -1.0
    dirty.loc[dirty.index[60], "date_logged"] = "not-a-date"
    dirty.loc[dirty.index[61], "closing_price"] = np.nan
    dirty.loc[dirty.index[62], "closing_price"] = -5.0
    dirty = pd.concat([dirty, dirty.iloc[100:105]], ignore_index=True)
    return dirty.sample(frac=1.0, random_state=7).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--months", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent / "data" / "sample" / "jse_economic_sample.csv"),
    )
    parser.add_argument("--dirty-out", default=None)
    args = parser.parse_args()

    end = pd.Timestamp(args.end) if args.end else pd.Timestamp("2026-08-31")
    start = pd.Timestamp(args.start) if args.start else end - pd.DateOffset(months=args.months)

    frame = simulate(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"Wrote {len(frame):,} rows to {out}")

    if args.dirty_out:
        dirty_path = Path(args.dirty_out)
        dirty_path.parent.mkdir(parents=True, exist_ok=True)
        make_dirty(frame).to_csv(dirty_path, index=False)
        print(f"Wrote dirty variant to {dirty_path}")


if __name__ == "__main__":
    main()
