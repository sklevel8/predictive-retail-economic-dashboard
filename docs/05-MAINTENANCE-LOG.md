# Maintenance Log & Operational Runbook

**Project:** Predictive Retail & Economic Data Dashboard (P10)
**Role:** Project Manager / Release Manager
**Current version:** 1.0.0

---

## 1. Service topology

| Service | Port | Health |
| --- | --- | --- |
| `p10-api` (FastAPI) | 8000 | `GET /health`, `GET /ready` |
| `p10-redis` | 6379 | `redis-cli ping` |
| `p10-dashboard` (Streamlit) | 8501 | `/_stcore/health` |

Bring everything up:

```bash
docker compose up --build -d
docker compose ps
```

## 2. Monitoring & alerting

| Signal | Source | Threshold | Action |
| --- | --- | --- | --- |
| Model drift | Weekly job comparing forecast vs realised closes; `model_metrics` table | MAPE degrades > 15 % vs baseline | Retrain, review macro coverage |
| API latency | Access logs / `X-Process-Time-ms` | p99 > 2.5 s on `/forecast` and `/scenario` | Inspect CPU, cache hit-rate, estimator |
| Error rate | 5xx count | > 1 % over 5 min | Roll back or scale out |
| Cache health | `/health` → `cache.backend` | `memory` in production | Verify `P10_REDIS_URL` |
| Disk | `P10_DB_PATH` volume | > 80 % | Archive old model artefacts |

Suggested alerting queries (if logs are shipped to Loki/CloudWatch):

```
rate(http_requests_total{path="/api/v1/scenario",status=~"5.."}[5m]) > 0.01
histogram_quantile(0.99, rate(http_request_duration_seconds_bucket{path="/api/v1/forecast"}[5m])) > 2.5
```

## 3. Routine maintenance

### Daily
- Confirm `/health` is `ok` on all replicas.
- Verify the previous night's backup completed (Supabase dashboard).
- Scan the audit log for unexpected deletions or uploads.

### Weekly
- Run drift check (`scripts/train.py --api-key <tenant>` and compare metrics).
- Review cache hit-rate; tune `P10_CACHE_TTL_SECONDS` if stale forecasts are reported.
- Check container image vulnerabilities.

### Monthly (first business day)
- Refresh macro datasets and re-run ETL.
- Retrain all tenant models; archive the new artefacts.
- Rotate tenant API keys where policy requires it (provision new hash, notify client, revoke old).

## 4. Runbook: common incidents

### 4.1 Forecast returns 404 for a known asset
1. `GET /api/v1/tenant` — confirm the caller's tenant.
2. `GET /api/v1/datasets/summary` — confirm the asset is registered for that tenant.
3. If missing, the upload either failed ETL or was written under another tenant.
   Check `audit_log` and the ETL report returned at upload time.

### 4.2 Latency spike on `/scenario`
1. Inspect `/health` for `cache.backend`.
2. If `memory`, Redis is unreachable — restore connectivity; the service remains correct but slower.
3. Check whether a retrain is thrashing the registry (repeated training logs).
4. Confirm no asset has an unusually long history causing repeated feature builds.

### 4.3 Model outputs look implausible
1. Check `model_metrics` for the asset (MAPE, directional accuracy).
2. Inspect `/api/v1/explain` — the largest SHAP contributor usually reveals a data problem
   (e.g. a stale CPI value or a 10× volume spike).
3. Re-upload corrected data, then `POST /api/v1/models/{asset}/train`.

### 4.4 Tenant reports seeing another tenant's data
1. **Treat as a security incident (P1).**
2. Immediately disable the affected API key (`tenants.api_key_hash` rotation).
3. Verify RLS is enabled in Supabase (`alter table ... enable row level security`).
4. Reproduce with the cross-tenant tests in `tests/test_api.py::test_tenant_isolation`.
5. Preserve logs; notify the data protection officer.

## 5. Backup & disaster recovery

| Asset | Frequency | Retention | Restore |
| --- | --- | --- | --- |
| PostgreSQL (Supabase) | Continuous PITR + daily snapshot | 7 days | Supabase restore to timestamp |
| Model artefacts (`P10_MODEL_DIR`) | On every training run | 30 days | Copy artefact back, restart API |
| Redis cache | Ephemeral | — | Rebuilt automatically on demand |
| Configuration / secrets | Change-controlled | — | Secret manager |

RPO 24 h, RTO 2 h. A quarterly restore drill is required.

## 6. Maintenance log

| Date | Version | Change | Owner | Notes |
| --- | --- | --- | --- | --- |
| 2026-11-01 | 1.0.0 | Initial production release | Release Manager | 38 tests passing; SLA verified |
| 2026-11-01 | 1.0.0 | Bug-001 explainer caching | Engineering | p100 8 s → 0.3 s |
| 2026-11-01 | 1.0.0 | Bug-002 JSON float casting | Engineering | 500 errors eliminated |
| 2026-11-01 | 1.0.0 | Bug-005 estimator + fast feature path | Engineering | Scenario SLA restored |

## 7. Roadmap — v1.1.0

1. **Automated retraining** — scheduled GitHub Actions or Airflow DAG on the 1st of each month: ETL → train → evaluate → promote/rollback.
2. **External load test** — Locust/k6 against containerised deployment with 100 concurrent analysts.
3. **Drift dashboard** — Grafana panel for MAPE and directional accuracy by asset.
4. **Rate limiting and quotas** — per-tenant token buckets on the inference endpoints.
5. **Auth upgrade** — Supabase Auth JWT verification alongside API keys.
6. **Object storage** — presigned uploads for very large CSV extracts (>50 MB).
