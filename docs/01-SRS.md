# Software Requirements Specification (SRS)

**Project:** Predictive Retail & Economic Data Dashboard (P10)
**Version:** 1.0.0
**Role:** Business Analyst
**Status:** Approved — implemented, tested and released

---

## 1. Purpose and scope

The system gives South African retail and financial institutions a single
analytical surface for local economic indicators (ZAR/USD, CPI, fuel price,
repo rate) and their own proprietary market data. It forecasts short-horizon
price and input-cost trajectories, explains every prediction, and allows
analysts to stress-test macro scenarios before committing to procurement or
pricing decisions.

In scope:

- CSV ingestion of historical market/economic series.
- Supervised forecasting of the next *n* business-day closing prices.
- Scenario analysis ("what if the rand strengthens by 5%?").
- Explainable AI (SHAP) attributions per forecast.
- Multi-tenant isolation and API-key authentication.

Out of scope (v1.0.0):

- Real-time streaming market feeds.
- Automated trade execution.
- Mobile-native clients.
- SSO/SAML (API keys only in this release).

## 2. Stakeholders

| Stakeholder | Interest |
| --- | --- |
| Strategic analyst | Scenario comparison, forecast accuracy, explainability |
| Executive / CFO | Directional cost and price outlook, confidence in the model |
| Data engineer | Reliable ETL, idempotent ingestion, audit trail |
| Compliance / risk | Tenant isolation, auditability, no cross-client data leakage |
| DevOps | Observable, containerised deployment with an SLA |

## 3. Functional requirements

### FR-1 — Data ingestion and ETL

The system shall accept a CSV upload containing `asset_name`, `date_logged`,
`closing_price` and optional `volume` plus macro indicators.

Acceptance criteria:

- Column aliases (`ticker`, `date`, `close`, `usd_zar`, `inflation`, …) are normalised.
- Unparseable dates, missing prices, non-positive prices and negative volumes are rejected and reported.
- Missing macro values are forward-filled per asset, then back-filled, then imputed with the median.
- Duplicate `(asset_name, date)` rows resolve to the last observation.
- Uploads are idempotent: re-uploading the same file does not duplicate rows.
- At most 100,000 rows per upload (configurable) and 50 MB per file.

### FR-2 — Predictive modelling engine

The system shall train a supervised model per `(tenant, asset)` on engineered
time-series features and forecast up to 90 business days ahead.

Acceptance criteria:

- Target variable is the next-period log return; forecasts are converted back to price.
- Recursive multi-step forecasting updates price, volume and macro state at each step.
- Model quality is reported as MAE, RMSE, MAPE and directional accuracy on a hold-out tail.
- Retraining is available on demand and is audited.

### FR-3 — Interactive visualisation

The dashboard shall render observed and forecast price lines, scenario
comparison charts, a macro-correlation heatmap, macro scatter plots and a SHAP
contribution chart, filterable by asset and horizon.

Acceptance criteria:

- Asset, horizon and scenario inputs update the charts without a page reload.
- Charts are interactive (zoom, hover, legend toggles).

### FR-4 — Scenario testing

The system shall recompute the forecast under proportional shocks to ZAR/USD,
CPI, fuel price, repo rate and volume, and report the delta versus baseline.

Acceptance criteria:

- Shocks are expressed as percentages in the UI and as fractions in the API.
- Response includes baseline path, scenario path, end-of-horizon delta, average delta and maximum delta.
- Scenario runs are cached by `(tenant, asset, data version, horizon, shocks)`.

### FR-5 — Explainability

Every forecast shall be accompanied by SHAP attributions that sum exactly to
the prediction minus the model's base value (local accuracy).

Acceptance criteria:

- Attribution is returned in log-return space and as an estimated ZAR price impact.
- Residual of the reconstruction is < 1e-6.
- The explainer object is created once per trained model, never per request.

### FR-6 — Tenant isolation

The system shall ensure that a tenant can only read, forecast, explain or
delete its own data.

Acceptance criteria:

- Every repository query carries an explicit `tenant_id` predicate.
- Production (Supabase/PostgreSQL) enforces Row-Level Security with the same predicate.
- Cross-tenant read attempts return HTTP 404 without revealing existence.

### FR-7 — Authentication and audit

Access requires an `X-API-Key` header. Keys are stored only as SHA-256
digests. Ingestions, deletions and trainings are written to an audit log.

### FR-8 — Operational telemetry

The API shall expose health/readiness endpoints, per-request latency headers,
model metrics and cache statistics.

## 4. Non-functional requirements

| ID | Requirement | Target | Result |
| --- | --- | --- | --- |
| NFR-1 | Scenario recalculation latency | p100 < 3 s | ~0.5–0.9 s server-side for 30 business days |
| NFR-2 | Cached forecast latency | < 500 ms | ~10–40 ms |
| NFR-3 | Dataset size | 100,000 rows | Enforced at upload; tested at 320-row and 1,176-row demo scale |
| NFR-4 | Availability | 99.5 % monthly | Container health checks + restart policy |
| NFR-5 | Recoverability | 24 h RPO | Supabase PITR + model artefacts in object storage |
| NFR-6 | Explainability fidelity | exact reconstruction | residual ≈ 1e-18 (float64) |

## 5. Assumptions and constraints

- Input data is end-of-day, not intraday.
- Macro indicators are supplied alongside price data; absent indicators degrade gracefully to price-only models.
- Python 3.11+ runtime; CatBoost is optional and the service degrades to scikit-learn gradient boosting.
- The local development database is SQLite; production targets Supabase/PostgreSQL.

## 6. Traceability

| Requirement | Implementation | Verification |
| --- | --- | --- |
| FR-1 | `backend/app/etl/pipeline.py` | `tests/test_etl.py` |
| FR-2 | `backend/app/ml/model.py`, `registry.py` | `tests/test_model.py` |
| FR-3 | `frontend/app.py` | Manual UI walkthrough |
| FR-4 | `app/services/forecasting.py` | `tests/test_api.py::test_scenario_endpoint` |
| FR-5 | `JSEPredictiveModel.predict_with_explanation` | `tests/test_model.py::test_shap_local_accuracy_holds` |
| FR-6 | `app/db/repository.py`, `supabase/migrations/0001_init.sql` | `tests/test_api.py::test_tenant_isolation` |
| FR-7 | `app/core/security.py`, `audit_log` | `tests/test_api.py::test_authentication_is_required` |
| FR-8 | `app/api/routes/health.py`, middleware | `tests/test_api.py::test_health` |
| NFR-1/2 | Feature fast path + prediction cache | `tests/test_performance.py` |
