# P10 — Predictive Retail & Economic Data Dashboard

Enterprise-grade forecasting and explainable AI for South African retail and
financial institutions. Upload historical market and macro-economic data,
project price and input-cost trajectories, stress-test macro scenarios, and see
exactly **why** the model produced each forecast.

[![CI](https://github.com/sklevel8/predictive-retail-economic-dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/sklevel8/predictive-retail-economic-dashboard/actions/workflows/ci.yml)

---

## What it does

| Capability | Detail |
| --- | --- |
| **ETL** | CSV ingestion with alias mapping, deduplication, macro imputation and a machine-readable quality report |
| **Forecasting** | Next-business-day log-return model, recursively projected up to 90 business days |
| **Explainable AI** | SHAP attributions per prediction, validated for exact local accuracy |
| **Scenario testing** | Shock ZAR/USD, CPI, fuel price, repo rate and volume; compare baseline vs shocked paths |
| **Multi-tenancy** | Every query is tenant-scoped; Supabase RLS policies ship with the repo |
| **Dashboard** | Streamlit + Plotly: forecasts, scenarios, SHAP waterfall, macro correlations |
| **Ops** | Docker Compose, health checks, Redis caching, GitHub Actions CI, 38 automated tests |

## Architecture

```
Streamlit dashboard ──HTTP──▶ FastAPI inference service
                                   │
              ┌────────────────────┼─────────────────────┐
              ▼                    ▼                     ▼
        ETL (pandas)        Model registry         Prediction cache
       cleaning, impute     CatBoost / GBR /      Redis, in-process
       feature engineering   Ridge + SHAP          TTL fallback
                                   │
                                   ▼
                     Tenant-scoped repository
                  SQLite (dev) · Supabase Postgres (prod)
```

Full design: [`docs/02-SDD.md`](docs/02-SDD.md)

---

## Quick start (5 minutes)

### 1. API

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Interactive docs: <http://localhost:8000/docs>

> CatBoost is optional (`pip install -r requirements-optional.txt`). Without it
> the service uses scikit-learn gradient boosting — identical API, slightly
> different accuracy profile.

### 2. Load demo data

```bash
# from the repository root
python backend/scripts/generate_demo_data.py --months 18
```

Then in PowerShell:

```powershell
$key = "p10-demo-tenant-a"
$form = @{ file = Get-Item backend/data/sample/jse_economic_sample.csv }
Invoke-RestMethod -Uri http://localhost:8000/api/v1/datasets/upload `
  -Method Post -Headers @{ "X-API-Key" = $key } -Form $form
```

or with curl:

```bash
curl -X POST http://localhost:8000/api/v1/datasets/upload \
  -H "X-API-Key: p10-demo-tenant-a" \
  -F "file=@backend/data/sample/jse_economic_sample.csv"
```

### 3. Dashboard

```bash
pip install -r frontend/requirements.txt
streamlit run frontend/app.py
```

Open <http://localhost:8501> and use API key `p10-demo-tenant-a`.

### 4. Docker (everything at once)

```bash
docker compose up --build -d
# API       http://localhost:8000/docs
# Dashboard http://localhost:8501
```

---

## Sandbox tenants

| Company | API key |
| --- | --- |
| Ubuntu Retail Group | `p10-demo-tenant-a` |
| Karoo Financial Services | `p10-demo-tenant-b` |

Demo tenants are created at startup when `P10_DEMO_MODE=true`. Set it to
`false` in production.

---

## API cheat-sheet

All tenant endpoints require `X-API-Key`.

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Service, database, cache and ML capability status |
| `GET` | `/api/v1/tenant` | Caller identity and dataset footprint |
| `POST` | `/api/v1/datasets/upload` | Upload a CSV (multipart `file`) |
| `GET` | `/api/v1/datasets/summary` | Assets, row counts, date ranges |
| `GET` | `/api/v1/datasets/{asset}/series?limit=400` | Historical series |
| `DELETE` | `/api/v1/datasets/{asset}` | Delete an asset for this tenant only |
| `POST` | `/api/v1/forecast` | Point forecast (+ optional SHAP) |
| `POST` | `/api/v1/scenario` | Baseline vs shocked forecast |
| `GET` | `/api/v1/explain?asset_name=...` | Attribution for the next close |
| `POST` | `/api/v1/models/{asset}/train` | Force a retrain |
| `GET` | `/api/v1/models/metrics` | Latest hold-out metrics |

### Forecast

```bash
curl -X POST http://localhost:8000/api/v1/forecast \
  -H "X-API-Key: p10-demo-tenant-a" \
  -H "Content-Type: application/json" \
  -d '{"asset_name":"Standard Bank","horizon_days":30,"include_explanation":true}'
```

```json
{
  "asset_name": "Standard Bank",
  "algorithm": "gradient_boosting",
  "horizon_days": 30,
  "last_close": 61.72,
  "points": [
    {"step": 1, "date": "2026-09-01", "predicted_price": 60.73, "predicted_log_return": 0.0016},
    {"step": 2, "date": "2026-09-02", "predicted_price": 60.41, "predicted_log_return": -0.0053}
  ],
  "explanation": {
    "method": "tree_shap",
    "local_accuracy_delta": -2.8e-18,
    "contribution_price_impact": {"ret_lag_0": -0.42, "zar_usd_chg_1": 0.31}
  }
}
```

### Scenario

```bash
curl -X POST http://localhost:8000/api/v1/scenario \
  -H "X-API-Key: p10-demo-tenant-a" \
  -H "Content-Type: application/json" \
  -d '{
        "asset_name": "Standard Bank",
        "horizon_days": 30,
        "shocks": {"zar_usd_pct": -0.05, "fuel_price_pct": 0.10}
      }'
```

The response contains `baseline[]`, `scenario[]` and a `summary` with
`percent_change`, `max_price_delta` and `direction`.

---

## CSV format

Required: `asset_name`, `date_logged`, `closing_price`

Optional: `volume`, `zar_usd`, `cpi`, `fuel_price`, `repo_rate`

Aliases are accepted (`ticker`, `date`, `close`, `vol`, `usd_zar`, `inflation`,
`petrol_price`, `interest_rate`). Sample files:

- `backend/data/sample/jse_economic_sample.csv` — 1,176 clean rows, 3 JSE assets
- `backend/data/sample/jse_economic_dirty.csv` — deliberately defective, for ETL demos

---

## Testing

```bash
cd backend
pip install -r requirements-dev.txt
python -m pytest -q
```

```
38 passed
```

Coverage includes ETL repairs, feature-leakage checks, fast-path parity, SHAP
local accuracy, JSON serialisation regressions, tenant isolation across read /
forecast / stream / delete, and the 3-second scenario SLA.

See [`docs/03-TEST-PLAN.md`](docs/03-TEST-PLAN.md) for the full matrix and five
resolved defect reports.

---

## Project structure

```
P10/
├── backend/
│   ├── app/
│   │   ├── api/routes/        health, tenant, datasets, forecasting
│   │   ├── core/              API-key security, domain errors
│   │   ├── db/repository.py   tenant-scoped SQLite repository
│   │   ├── etl/pipeline.py    cleaning, imputation, validation report
│   │   ├── ml/                features, model + SHAP, registry
│   │   ├── schemas/           Pydantic contracts
│   │   ├── services/          forecasting orchestration, cache
│   │   └── main.py            app factory, lifespan, middleware
│   ├── scripts/               demo-data generator, bulk trainer
│   ├── tests/                 pytest suite (38 tests)
│   ├── data/sample/           committed demo CSVs
│   └── Dockerfile, requirements*.txt, pyproject.toml
├── frontend/                  Streamlit dashboard
├── supabase/migrations/       PostgreSQL schema + RLS policies
├── docs/                      SRS, SDD, test plan, release notes, runbook
├── .github/workflows/ci.yml   lint + test + docker build
├── docker-compose.yml         api + redis + dashboard
└── Makefile
```

---

## SDLC deliverables

| Phase | Role | Document |
| --- | --- | --- |
| Requirements | Business Analyst | [`docs/01-SRS.md`](docs/01-SRS.md) |
| Design | System Architect | [`docs/02-SDD.md`](docs/02-SDD.md) |
| Implementation | Software Engineer | `backend/`, `frontend/`, `supabase/` |
| Testing | QA Tester | [`docs/03-TEST-PLAN.md`](docs/03-TEST-PLAN.md) |
| Deployment & Maintenance | Project Manager / Release Manager | [`docs/04-RELEASE-NOTES.md`](docs/04-RELEASE-NOTES.md), [`docs/05-MAINTENANCE-LOG.md`](docs/05-MAINTENANCE-LOG.md) |

---

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `P10_ENV` | `development` | Environment label reported by `/health` |
| `P10_DB_PATH` | `backend/data/app.db` | SQLite location (dev/test) |
| `P10_MODEL_DIR` | `backend/data/models` | Model artefact directory |
| `P10_REDIS_URL` | unset | Enables Redis prediction cache |
| `P10_CACHE_TTL_SECONDS` | `900` | Forecast/scenario cache TTL |
| `P10_MAX_UPLOAD_ROWS` | `100000` | Row limit per upload |
| `P10_MIN_TRAINING_ROWS` | `120` | Minimum history before training |
| `P10_MAX_HORIZON` | `90` | Maximum forecast horizon |
| `P10_DEMO_MODE` | `true` | Seed demo tenants |
| `P10_API_URL` | `http://localhost:8000/api/v1` | Dashboard → API base URL |

Copy `.env.example` to `.env` and adjust as needed.

---

## Design notes

- **Why log returns?** Forecasting returns rather than price levels keeps the
  target stationary and prevents unbounded extrapolation.
- **Why gradient boosting?** Recursive forecasting predicts one row at a time.
  Random Forest was measured at ~60–100 ms per single-row call here, which
  broke the 3-second SLA; gradient boosting is ~0.7 ms. CatBoost is used when a
  wheel is available.
- **Why a fast feature path?** The full pandas feature builder allocates ~30
  columns per call (~45 ms). The inference path uses one numpy pass (~4 ms) and
  is locked to the batch implementation by a parity test.
- **Why two isolation layers?** Code-level `tenant_id` predicates protect the
  service; PostgreSQL RLS protects the data even if a future query forgets one.

## Known limitations

1. API keys are the v1 authentication mechanism (SSO in v1.2.0).
2. Retraining is manual or on-demand (scheduled automation in v1.1.0).
3. Local development uses SQLite; apply `supabase/migrations/0001_init.sql` for production RLS.
4. CatBoost support depends on wheel availability for your Python version.

## License

MIT — see [`LICENSE`](LICENSE).
