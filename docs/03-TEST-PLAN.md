# Test Plan & Bug Reports

**Project:** Predictive Retail & Economic Data Dashboard (P10)
**Version under test:** 1.0.0
**Role:** QA Tester
**Execution date:** 2026-11-01
**Environment:** Windows 11, Python 3.14.7, pytest 8.x
**Result:** **38 passed, 0 failed** (`cd backend && python -m pytest -q`)

---

## 1. Strategy

| Layer | Focus | Tooling |
| --- | --- | --- |
| Unit | Feature engineering, model training, SHAP fidelity, JSON safety | pytest |
| Integration | CSV upload → ETL → persistence → forecast HTTP contracts | FastAPI `TestClient` |
| Security | Authentication, tenant isolation across read/forecast/delete | pytest |
| Non-functional | 3-second scenario SLA, cached response latency | pytest timing |
| Manual / exploratory | Dashboard rendering, chart interactivity | Streamlit walkthrough |

## 2. Test suites

### TS-01 — ETL data quality (`tests/test_etl.py`)

| Case | Expectation |
| --- | --- |
| Clean frame round-trip | 120/120 rows accepted, no nulls, ascending dates |
| Dirty frame (duplicates, bad date, missing/negative price, negative CPI, negative volume) | 297/305 rows accepted, duplicates removed, macro gaps imputed, warnings emitted |
| Column aliases (`ticker`, `close`, `usd_zar`, …) | Normalised successfully |
| Empty upload | `UploadValidationError` |
| Missing required column | `UploadValidationError` naming the column |

### TS-02 — Model & XAI consistency (`tests/test_model.py`)

| Case | Expectation |
| --- | --- |
| Training | Estimator selected, MAE/RMSE/MAPE/direction reported, >100 rows trained |
| SHAP local accuracy | `base + Σφ − prediction` < 1e-9 |
| JSON serialisation (Bug-002 regression) | `json.dumps(explanation)` succeeds; all values are native `float` |
| Price impacts | `φ_i × last_close` matches within 1e-2 |
| Forecast shape | 30 points, positive prices, ascending dates, clipped returns |
| Macro shock | A ±50 % macro shock changes the trajectory |
| Explainer reuse (Bug-001 regression) | The same explainer object survives repeated calls |
| Determinism | Two forecasts produce identical prices |

### TS-03 — Feature engineering (`tests/test_features.py`)

| Case | Expectation |
| --- | --- |
| Leakage check | `target_t == ret_lag_0_{t+1}` |
| Fast-path parity | `latest_feature_row` equals the last `build_features` row (rtol 1e-9) |
| Missing macro columns | Degrades to zeros without raising |
| Short history | Returns an empty frame instead of guessing |

### TS-04 — API contract & security (`tests/test_api.py`)

| Case | Expectation |
| --- | --- |
| Health | 200, `status=ok`, database ok |
| Missing / invalid API key | 401 |
| Tenant identity | `tenant_id` + `company_name` returned |
| Clean upload | 201, 320 rows, asset listed |
| Dirty upload | 201, repairs reported |
| Non-CSV / missing columns | 415 / 422 |
| Forecast | 15 points, positive prices, latency field present |
| Forecast + explanation | 22 attributions, residual < 1e-6 |
| Scenario | Baseline and scenario paths of equal length, summary populated |
| Explain endpoint | Price and local accuracy returned |
| Metrics / retrain | Metrics exposed after training; retrain returns the new algorithm |
| Unknown asset | 404 |
| Horizon bounds | 422 above the model bound, 400 above the configured bound |
| Series endpoint | 50 points, chronological |
| **Tenant isolation** | B cannot see, forecast, stream or delete A's asset (404 / absent from summary) |
| Delete | 320 rows deleted, subsequent series 404 |
| OpenAPI | Forecast and scenario paths present |

### TS-05 — Performance (`tests/test_performance.py`)

| Case | SLA | Measured |
| --- | --- | --- |
| 5 × scenario (30 days) after warm-up | < 3 000 ms each | 470 ms average (server-reported ~400–900 ms) |
| Cached forecast | < 1 000 ms | ~10–40 ms |

## 3. Bug reports

### Bug-001 — SHAP explainer rebuilt per request (High, resolved)

**Symptom:** p100 inference latency rose from 1.2 s to > 8 s under load; memory
grew with every request.
**Root cause:** `shap.TreeExplainer(self.model)` was constructed inside
`predict_with_explanation()` — an O(trees) build on every call.
**Resolution:** the explainer is now built once in `train()` and cached on the
model instance (`_explainer`). Inference only calls `.shap_values()`.
**Verification:** `test_explainer_is_reused_across_calls`; warm latency now 200–300 ms.
**Status:** Resolved & Verified.

### Bug-002 — `TypeError: Object of type float32 is not JSON serializable` (Medium, resolved)

**Symptom:** HTTP 500 when returning the attribution payload.
**Root cause:** NumPy scalars leaked into the response dictionary.
**Resolution:** `to_native()` recursively casts NumPy types to Python `float`;
Pydantic response models double-validate.
**Verification:** `test_explanation_is_json_serialisable` asserts `type(value) is float` and round-trips through `json.dumps`.
**Status:** Resolved & Verified.

### Bug-003 — Recursive forecast crashed on mixed date dtypes (High, resolved)

**Symptom:** `TypeError: '<' not supported between instances of 'Timestamp' and 'str'`
during the second forecast step.
**Root cause:** history frames from the database carry date strings, while
simulated rows were appended as `Timestamp` objects, so `sort_values` compared
incomparable types.
**Resolution:** `forecast()` normalises `date_logged` to datetime before the loop.
**Verification:** `test_forecast_shape_and_positivity`, `test_scenario_endpoint`.
**Status:** Resolved & Verified.

### Bug-004 — pandas 3 rejected tuple column selection (High, resolved)

**Symptom:** `KeyError: ('zar_usd', 'cpi', 'fuel_price', 'repo_rate')` on every upload.
**Root cause:** `MACRO_COLUMNS` was a tuple and was used directly as `df[MACRO_COLUMNS]`;
pandas treats a tuple as a single multi-index key.
**Resolution:** the constant is a `list[str]`, and group-wise imputation uses
per-column `transform` instead of multi-column `apply`.
**Verification:** `test_clean_frame_survives_round_trip`, `test_dirty_frame_is_repaired_without_crashing`.
**Status:** Resolved & Verified.

### Bug-005 — Random Forest single-row inference breached the 3 s SLA (High, resolved)

**Symptom:** a 30-day scenario took 9.6–20.4 s; profiling showed ~60–100 ms per
single-row `predict` call (batch and OpenMP overhead), multiplied by 60 calls.
**Resolution:** (a) added a numpy fast path for the latest feature vector
(45 ms → 4 ms per step); (b) selected `GradientBoostingRegressor` as the
scikit-learn default (0.7 ms per single-row prediction) while keeping CatBoost
as the preferred estimator. SHAP remains tree-based.
**Verification:** `tests/test_performance.py`, `test_fast_path_matches_batch_feature_builder`.
**Status:** Resolved & Verified.

## 4. Exit criteria

- All P1/P2 defects closed: **met**.
- No open security defects: **met**.
- 3-second SLA demonstrated under repeated scenario calls: **met**.
- Tenant isolation proven by automated cross-tenant tests: **met**.
- Dashboard manual walkthrough completed on demo data: **met**.

## 5. Residual risks

- Load testing used the in-process `TestClient`; a Locust/k6 run against a
  containerised deployment is scheduled before v1.1.0.
- SHAP values were validated for gradient-boosted models only (the CatBoost
  path shares the same TreeExplainer contract but was not executed in CI
  because no wheel exists for every Python version).
