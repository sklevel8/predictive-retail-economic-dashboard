"""Train and persist models for every asset a tenant has uploaded.

    python scripts/train.py --tenant-api-key p10-demo-tenant-a
    python scripts/train.py --api-key p10-demo-tenant-a --asset "Naspers"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.security import hash_api_key  # noqa: E402
from app.db.repository import get_repository  # noqa: E402
from app.services.forecasting import get_registry  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key", required=True, help="Tenant API key")
    parser.add_argument("--asset", default=None, help="Only train this asset")
    args = parser.parse_args()

    repository = get_repository()
    tenant = repository.get_tenant_by_api_key_hash(hash_api_key(args.api_key))
    if tenant is None:
        print("Unknown API key.", file=sys.stderr)
        return 1

    assets = [args.asset] if args.asset else [a.asset_name for a in repository.list_assets(tenant.tenant_id)]
    if not assets:
        print("No datasets uploaded for this tenant.", file=sys.stderr)
        return 2

    registry = get_registry()
    for asset in assets:
        result = registry.retrain(tenant.tenant_id, asset)
        print(
            f"{asset}: {result.algorithm} | MAE {result.mae:.2f} | "
            f"MAPE {result.mape:.2f}% | direction {result.directional_accuracy:.1f}% | "
            f"{result.rows_trained} rows"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
