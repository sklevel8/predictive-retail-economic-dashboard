"""Shared fixtures.

Environment variables are set *before* importing the application so the
cached Settings object points at a throw-away SQLite database and model folder.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

TEST_ROOT = Path(tempfile.mkdtemp(prefix="p10-tests-"))

os.environ["P10_DB_PATH"] = str(TEST_ROOT / "test.db")
os.environ["P10_MODEL_DIR"] = str(TEST_ROOT / "models")
os.environ["P10_DEMO_MODE"] = "true"
os.environ["P10_CACHE_TTL_SECONDS"] = "120"
os.environ["P10_MIN_TRAINING_ROWS"] = "60"
os.environ["P10_MAX_HORIZON"] = "90"
os.environ["P10_ENV"] = "test"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app  # noqa: E402

API_KEY_A = "p10-demo-tenant-a"
API_KEY_B = "p10-demo-tenant-b"


@pytest.fixture(scope="session")
def client() -> TestClient:
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def auth_a() -> dict[str, str]:
    return {"X-API-Key": API_KEY_A}


@pytest.fixture(scope="session")
def auth_b() -> dict[str, str]:
    return {"X-API-Key": API_KEY_B}
