# Release Notes — v1.0.0 (Production)

**Project:** Predictive Retail & Economic Data Dashboard (P10)
**Release date:** 2026-11-01
**Role:** Project Manager / Release Manager
**Deployment target:** Containerised API + Redis + Streamlit dashboard, Supabase PostgreSQL for managed persistence

---

## 1. Highlights

### Live inference API

- `POST /api/v1/forecast` returns up to 90 business-day price projections per asset.
- `POST /api/v1/scenario` returns baseline and shocked trajectories with a delta summary.
- Tree-based ensemble inference with cached SHAP explainers serves warm requests in **200–300 ms** server-side; cached responses return in **10–40 ms**.
- First request per asset trains the model on demand (0.8–2.9 s) and persists the artefact.

### Explainable AI

- Every prediction ships with SHAP attributions per feature.
- Local accuracy is enforced: base value + attributions reconstruct the prediction (residual ≈ 1e-18).
- Contributions are exposed both in log-return space and as an estimated ZAR price impact, so analysts can answer "what moved this forecast?".

### Data platform

- CSV ingestion with alias mapping, deduplication, imputation and a machine-readable ETL report.
- Idempotent upserts keyed on `(tenant_id, asset_name, date_logged)`.
- Model metrics, audit log and dataset summaries persisted alongside the data.

### Interactive dashboard

- Forecast view with observed history overlay.
- Scenario studio with ZAR/USD, CPI, fuel and repo-rate sliders.
- XAI tab with a ranked SHAP bar chart and raw values.
- Data tab with CSV upload, asset inventory, correlation heatmap and macro scatter plots.

## 2. Security & compliance

| Control | Implementation |
| --- | --- |
| Tenant isolation | Explicit `tenant_id` predicate in every query; Supabase RLS policies in `supabase/migrations/0001_init.sql` |
| Credential storage | API keys stored as SHA-256 digests only |
| Transport | HTTPS terminated by the hosting platform |
| Input hardening | CSV size/row limits, numeric coercion, parameterised SQL, 415/422 on malformed uploads |
| Audit | `audit_log` entries for upload, delete and training |
| Cross-tenant leakage | Automated tests assert 404 on foreign assets |

## 3. Verified performance

| Operation | Result |
| --- | --- |
| Scenario, 30 business days | 0.5–0.9 s server time |
| Forecast with SHAP, warm | 0.2–0.3 s |
| Cached response | 10–40 ms |
| Test suite | 38 passed |

## 4. Upgrade / install notes

```bash
# API
cd backend
python -m venv .venv && . .venv/Scripts/activate     # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload

# Dashboard
pip install -r frontend/requirements.txt
streamlit run frontend/app.py
```

`P10_DEMO_MODE=true` creates two sandbox tenants:

| Tenant | API key |
| --- | --- |
| Ubuntu Retail Group | `p10-demo-tenant-a` |
| Karoo Financial Services | `p10-demo-tenant-b` |

**Disable demo mode in production** (`P10_DEMO_MODE=false`) and provision
tenants with hashed keys through the admin path.

## 5. Known limitations

1. Authentication is API-key based; SSO/OIDC is deferred to v1.2.0.
2. Model retraining is a manual script (`scripts/train.py`) or on-demand endpoint.
3. The local development database is SQLite; Supabase RLS is applied in production only.
4. CatBoost is optional and requires a compatible wheel; the service otherwise uses scikit-learn gradient boosting.
5. Load testing was performed in-process; an external load test is scheduled for v1.1.0.

## 6. Rollback plan

1. Re-deploy the previous container image tag.
2. Restore the database snapshot (Supabase PITR, 24 h window).
3. Restore the previous model artefacts from object storage (`P10_MODEL_DIR`).
4. Re-run `pytest` and a single scenario request as a smoke test.

## 7. Sign-off

| Role | Responsibility | Status |
| --- | --- | --- |
| Business Analyst | SRS approval | ✅ |
| System Architect | SDD approval | ✅ |
| Software Engineer | Implementation | ✅ |
| QA Tester | Test plan, defect closure | ✅ |
| Release Manager | Release notes, rollback plan | ✅ |
