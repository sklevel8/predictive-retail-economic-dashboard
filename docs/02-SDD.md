# System Design Document (SDD)

**Project:** Predictive Retail & Economic Data Dashboard (P10)
**Version:** 1.0.0
**Role:** System Architect
**Status:** Implemented

---

## 1. Architectural overview

```
┌───────────────────────────┐        HTTPS/JSON        ┌────────────────────────────┐
│  Streamlit dashboard      │ ───────────────────────▶ │  FastAPI inference service │
│  (Plotly charts, XAI UI)  │ ◀─────────────────────── │  app/main.py               │
└───────────────────────────┘                          └───────────┬────────────────┘
                                                                   │
                    ┌──────────────────────────────────────────────┼─────────────────────┐
                    │                                              │                     │
             ┌──────▼───────┐                            ┌─────────▼────────┐   ┌────────▼────────┐
             │ ETL pipeline │                            │ Model registry   │   │ Prediction cache│
             │ pandas/NumPy │                            │ + SHAP explainer │   │ Redis / in-proc │
             └──────┬───────┘                            └─────────┬────────┘   └─────────────────┘
                    │                                              │
             ┌──────▼──────────────────────────────────────────────▼───────┐
             │  Tenant-scoped repository (SQLite locally / Supabase Postgres) │
             └─────────────────────────────────────────────────────────────┘
```

The design is deliberately decoupled: the dashboard holds no business logic,
the API owns validation/ETL/inference, and the database enforces isolation a
second time via Row-Level Security in production.

## 2. Component responsibilities

| Component | Module | Responsibility |
| --- | --- | --- |
| HTTP layer | `app/api/routes/*` | Request validation, auth dependency, response models |
| Auth | `app/core/security.py` | SHA-256 API-key hashing, tenant resolution |
| ETL | `app/etl/pipeline.py` | Cleaning, alias mapping, imputation, sanity checks |
| Features | `app/ml/features.py` | Leakage-free features, plus a fast single-row path for inference |
| Model | `app/ml/model.py` | Estimator selection, training/evaluation, recursive forecast, SHAP |
| Registry | `app/ml/registry.py` | One model per (tenant, asset), version-aware caching, artefact persistence |
| Cache | `app/services/cache.py` | Redis with transparent in-process TTL fallback |
| Orchestration | `app/services/forecasting.py` | Forecast/scenario assembly, caching, timing |
| Persistence | `app/db/repository.py` | Tenant-scoped SQL, upserts, metrics, audit log |

## 3. Data model

```sql
tenants(tenant_id PK, company_name, api_key_hash UNIQUE, created_at)
economic_data(record_id PK, tenant_id FK, asset_name, date_logged,
              closing_price, volume, zar_usd, cpi, fuel_price, repo_rate,
              source, ingested_at,
              UNIQUE (tenant_id, asset_name, date_logged))
model_metrics(metric_id PK, tenant_id, asset_name, algorithm, trained_at,
              rows_trained, mae, rmse, mape, directional_accuracy)
audit_log(audit_id PK, tenant_id, action, detail, created_at)
tenant_members(user_id FK auth.users, tenant_id FK, role)   -- production only
```

Row-Level Security (production):

```sql
create policy economic_data_select on economic_data
  for select to authenticated
  using (tenant_id = public.current_tenant_id());
```

`current_tenant_id()` resolves `auth.uid()` through `tenant_members`. The
FastAPI service uses the service role but still passes `tenant_id` explicitly to
every repository method, so isolation does not depend on a single mechanism.

## 4. Machine-learning design

### 4.1 Supervised formulation

For each asset, the target is the **next business-day log return**:

```
target_t = ln(close_{t+1}) - ln(close_t)
```

Forecasting recursively:
`close_{t+1} = close_t * exp(clip(ŷ_t, -0.10, 0.10))`.

### 4.2 Feature set (22 features)

| Group | Features |
| --- | --- |
| Momentum | `ret_lag_0..4` |
| Volatility / trend | `roll_mean_5`, `roll_std_5`, `roll_mean_20`, `roll_std_20`, `ma_ratio_20` |
| Liquidity | `volume_z_20` |
| Macro levels | `zar_usd`, `cpi`, `fuel_price`, `repo_rate` |
| Macro dynamics | `zar_usd_chg_1`, `fuel_chg_1`, `cpi_chg_1`, `rate_chg_1` |
| Seasonality | `month`, `quarter`, `day_of_week` |

All features at time *t* use only information available at *t*. The training
matrix drops incomplete rows; the most recent row is kept for inference.

### 4.3 Estimator selection

| Condition | Estimator | Rationale |
| --- | --- | --- |
| CatBoost installed and ≥250 training rows | `CatBoostRegressor` | Best tabular accuracy, native SHAP |
| ≥200 training rows | `GradientBoostingRegressor` | Strong accuracy, ~0.7 ms single-row predict |
| Otherwise | `Ridge` pipeline | Stable with very short histories |

The recursive forecast calls the estimator once per simulated day, so
single-row inference cost dominates the SLA. A Random Forest was measured at
~60–100 ms per single-row prediction in this environment (batch overhead),
which pushed a 30-day scenario past 3 s; gradient boosting reduces the same
work to ~0.7 ms. Batch evaluation during training is unaffected.

### 4.4 Explainability

- `shap.TreeExplainer` is instantiated once inside `train()` and stored on the model.
- Each request calls `explainer.shap_values(...)` only.
- Local accuracy is verified: `base_value + Σ φ_i == prediction`.
- Contributions are also expressed in ZAR via `φ_i × close_t` (first-order linearisation).
- If SHAP is unavailable, a deterministic permutation-sampling Shapley fallback is used and labelled `permutation_shap`.

### 4.5 Scenario engine

`forecast(frame, horizon, macro_shocks)` appends simulated rows whose macro
inputs equal the last observed level × `(1 + shock)`, keeping the shocked
environment constant across the horizon. Baseline and scenario share the same
model instance and cached explainer.

## 5. API surface

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Liveness/readiness, dependency and ML capability report |
| GET | `/api/v1/tenant` | Caller identity and dataset footprint |
| POST | `/api/v1/datasets/upload` | CSV ingestion with ETL report |
| GET | `/api/v1/datasets/summary` | Assets, row counts, date ranges |
| GET | `/api/v1/datasets/{asset}/series` | Historical series for charting |
| DELETE | `/api/v1/datasets/{asset}` | Tenant-scoped deletion |
| POST | `/api/v1/forecast` | Point forecast, optional SHAP explanation |
| POST | `/api/v1/scenario` | Baseline vs shocked forecast |
| GET | `/api/v1/explain` | Attribution for the next close |
| POST | `/api/v1/models/{asset}/train` | Force retrain |
| GET | `/api/v1/models/metrics` | Latest hold-out metrics |

Authentication: `X-API-Key`. Errors use `{"detail": ..., "error": ...}` with
sensible 4xx codes (404 for unknown/foreign assets, 422 for invalid payloads).

## 6. Performance strategy

1. **Fast feature path** — one numpy pass instead of the 30-step pandas pipeline (≈45 ms → ≈4 ms per step).
2. **Single estimator per asset** cached in memory and keyed by a `(row_count, last_date)` version.
3. **Redis caching** of forecast/scenario payloads; cache keys include tenant, asset, data version, horizon and shocks.
4. **Cache invalidation** on upload, delete and retrain.
5. **Optional pre-warm** via `scripts/train.py` or a startup hook in deployment.

Measured on the reference environment:

| Operation | Latency |
| --- | --- |
| First forecast (includes training) | 0.8–2.9 s |
| Warm forecast (30 days, SHAP included) | 0.2–0.3 s |
| Scenario (30 days, two paths) | 0.5–0.9 s |
| Cached response | 10–40 ms |

## 7. Security design

- API keys hashed with SHA-256; plaintext never stored.
- Tenant id resolved per request; all SQL is parameterised.
- CSV payloads are parsed with `pandas.read_csv` from decoded text, never executed; numeric coercion prevents type confusion.
- Upload size (50 MB) and row count (100k) limits.
- CORS configurable; demo tenants can be disabled with `P10_DEMO_MODE=false`.
- Audit log records ingestion, deletion and training events.
- Production hardening backlog: rotate to Supabase Auth JWTs, rate limiting, per-tenant quotas.

## 8. Deployment

- Docker images for API and dashboard; `docker-compose.yml` adds Redis.
- Health checks gate dashboard start-up (`depends_on: condition: service_healthy`).
- CI (`.github/workflows/ci.yml`) runs lint + tests on Python 3.11/3.12 and builds both images.
- Model artefacts persist under `P10_MODEL_DIR`; metrics persist in the database.

## 9. Architecture decisions (ADR)

| ID | Decision | Alternatives | Consequence |
| --- | --- | --- | --- |
| ADR-1 | API-key auth for v1 | OAuth/OIDC | Fast delivery; documented upgrade path to Supabase Auth |
| ADR-2 | SQLite for local/dev, Supabase for prod | PostgreSQL only | Zero-friction onboarding; repository SQL kept portable |
| ADR-3 | Gradient boosting as sklearn default | Random Forest | Meets the 3 s SLA for recursive inference |
| ADR-4 | SHAP pre-built at train time | Per-request explainer | Removes 8 s p100 latency regression (Bug-001) |
| ADR-5 | Redis optional with TTL fallback | Mandatory Redis | Service runs and tests green without infrastructure |
| ADR-6 | Forecast next log return | Forecast price directly | Stationarity, avoids unbounded extrapolation |

## 10. Risk register

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Model drift after macro regime change | Medium | High | Weekly MAE monitor, monthly retrain (v1.1 automation) |
| Cold-start latency on first request | High | Low | Pre-warm script / startup training job |
| Supabase RLS misconfiguration | Low | Critical | Explicit tenant predicate in code + isolation tests |
| CSV schema drift from source systems | Medium | Medium | Alias map, ETL report, validation errors |
| CatBoost wheel unavailable on new Python | Medium | Low | Automatic estimator fallback |
