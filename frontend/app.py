"""Streamlit dashboard for the Predictive Retail & Economic Data Dashboard.

Run with:
    streamlit run frontend/app.py

Environment:
    P10_API_URL  base URL of the FastAPI service (default http://localhost:8000/api/v1)
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE = os.getenv("P10_API_URL", "http://localhost:8000/api/v1").rstrip("/")
TIMEOUT = 30

st.set_page_config(
    page_title="P10 | Predictive Retail & Economic Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .block-container {padding-top: 1.6rem;}
    .kpi-card {
        background: linear-gradient(160deg, #0f172a, #1e293b);
        border: 1px solid #334155; border-radius: 14px;
        padding: 16px 18px; color: #e2e8f0;
    }
    .kpi-label {font-size: 0.74rem; letter-spacing: .09em; text-transform: uppercase; color: #94a3b8;}
    .kpi-value {font-size: 1.55rem; font-weight: 650; margin-top: 4px;}
    .kpi-delta {font-size: 0.82rem; margin-top: 2px;}
    .up {color: #34d399;} .down {color: #f87171;}
    div[data-testid="stMetricValue"] {font-size: 1.4rem;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def _headers(api_key: str) -> dict[str, str]:
    return {"X-API-Key": api_key.strip()} if api_key else {}


def api_get(api_key: str, path: str, **params: Any) -> tuple[dict | None, str | None]:
    try:
        response = requests.get(
            f"{API_BASE}{path}", headers=_headers(api_key), params=params, timeout=TIMEOUT
        )
    except requests.RequestException as exc:
        return None, f"Cannot reach the API at {API_BASE}: {exc}"
    return _handle(response)


def api_post(
    api_key: str, path: str, payload: dict | None = None, files: dict | None = None
) -> tuple[dict | None, str | None]:
    try:
        response = requests.post(
            f"{API_BASE}{path}",
            headers=_headers(api_key),
            json=payload if files is None else None,
            files=files,
            timeout=TIMEOUT,
        )
    except requests.RequestException as exc:
        return None, f"Cannot reach the API at {API_BASE}: {exc}"
    return _handle(response)


def _handle(response: requests.Response) -> tuple[dict | None, str | None]:
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        return None, f"{response.status_code}: {detail}"
    try:
        return response.json(), None
    except ValueError:
        return None, "The API returned an invalid JSON payload."


def money(value: float | None) -> str:
    if value is None:
        return "—"
    return f"R {value:,.2f}"


def pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:+.2f}%"


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def forecast_figure(history: pd.DataFrame, points: list[dict], title: str) -> go.Figure:
    figure = go.Figure()
    if not history.empty:
        figure.add_trace(
            go.Scatter(
                x=history["date"],
                y=history["closing_price"],
                mode="lines",
                name="Observed",
                line={"color": "#38bdf8", "width": 2},
            )
        )

    if points:
        dates = [point["date"] for point in points]
        values = [point["predicted_price"] for point in points]
        if not history.empty:
            dates = [history["date"].iloc[-1], *dates]
            values = [float(history["closing_price"].iloc[-1]), *values]
        figure.add_trace(
            go.Scatter(
                x=dates,
                y=values,
                mode="lines+markers",
                name="Forecast",
                line={"color": "#f59e0b", "width": 2, "dash": "dot"},
                marker={"size": 4},
            )
        )

    figure.update_layout(
        title=title,
        template="plotly_dark",
        height=430,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.08},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.35)",
    )
    figure.update_yaxes(title="Price (ZAR)", tickprefix="R ")
    return figure


def scenario_figure(baseline: list[dict], scenario: list[dict], title: str) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[point["date"] for point in baseline],
            y=[point["predicted_price"] for point in baseline],
            mode="lines",
            name="Baseline forecast",
            line={"color": "#38bdf8", "width": 2},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[point["date"] for point in scenario],
            y=[point["predicted_price"] for point in scenario],
            mode="lines",
            name="Scenario forecast",
            line={"color": "#f472b6", "width": 2, "dash": "dash"},
        )
    )
    figure.update_layout(
        title=title,
        template="plotly_dark",
        height=430,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.08},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.35)",
    )
    figure.update_yaxes(title="Price (ZAR)", tickprefix="R ")
    return figure


def shap_figure(explanation: dict) -> go.Figure:
    impacts: dict[str, float] = explanation["contribution_price_impact"]
    labels: dict[str, str] = explanation["contribution_labels"]

    ordered = sorted(impacts.items(), key=lambda item: abs(item[1]), reverse=True)[:12]
    ordered.reverse()

    names = [labels.get(key, key) for key, _ in ordered]
    values = [value for _, value in ordered]
    colours = ["#34d399" if value >= 0 else "#f87171" for value in values]

    figure = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=colours,
            hovertemplate="%{y}<br>%{x:,.2f} ZAR<extra></extra>",
        )
    )
    figure.update_layout(
        title="Why this forecast? (SHAP contribution to the next close, in ZAR)",
        template="plotly_dark",
        height=420,
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.35)",
        xaxis_title="Impact on predicted price",
    )
    return figure


def correlation_figure(frame: pd.DataFrame) -> go.Figure:
    columns = {
        "closing_price": "Close",
        "volume": "Volume",
        "zar_usd": "ZAR/USD",
        "cpi": "CPI",
        "fuel_price": "Fuel",
        "repo_rate": "Repo",
    }
    subset = frame[[column for column in columns if column in frame.columns]].rename(columns=columns)
    matrix = subset.corr(numeric_only=True).round(2)
    figure = px.imshow(
        matrix,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    figure.update_layout(
        title="Macro-indicator correlation matrix",
        template="plotly_dark",
        height=420,
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def macro_scatter(frame: pd.DataFrame, macro: str, label: str) -> go.Figure:
    figure = px.scatter(
        frame,
        x=macro,
        y="closing_price",
        trendline="ols" if len(frame) > 5 else None,
        opacity=0.65,
        labels={macro: label, "closing_price": "Closing price (ZAR)"},
    )
    figure.update_traces(marker={"color": "#38bdf8", "size": 6})
    figure.update_layout(
        title=f"Closing price vs {label}",
        template="plotly_dark",
        height=360,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.35)",
    )
    return figure


def kpi(label: str, value: str, delta: str | None = None, positive: bool | None = None) -> str:
    tone = ""
    if delta and positive is not None:
        tone = "up" if positive else "down"
    delta_html = f'<div class="kpi-delta {tone}">{delta}</div>' if delta else ""
    return (
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>{delta_html}</div>'
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("P10 Intelligence")
st.sidebar.caption("Predictive retail & economic analytics for South African markets")

api_key = st.sidebar.text_input(
    "Tenant API key",
    value=os.getenv("P10_API_KEY", ""),
    type="password",
    help="Demo sandbox: p10-demo-tenant-a / p10-demo-tenant-b",
)

if not api_key:
    st.title("Predictive Retail & Economic Data Dashboard")
    st.info(
        "Enter a tenant API key in the sidebar to begin. "
        "Sandbox keys: `p10-demo-tenant-a` and `p10-demo-tenant-b`."
    )
    st.stop()

summary, error = api_get(api_key, "/datasets/summary")
if error:
    st.error(error)
    st.stop()

assets = [asset["asset_name"] for asset in summary.get("assets", [])]
if not assets:
    st.warning("No datasets on this tenant yet. Upload a CSV in the **Data** tab to get started.")
    st.stop()

asset_name = st.sidebar.selectbox("Asset", assets, index=0)
horizon = st.sidebar.slider("Forecast horizon (business days)", 5, 90, 30, 5)

st.sidebar.divider()
st.sidebar.subheader("Scenario shocks")
zar_shock = st.sidebar.slider("ZAR/USD move (%)", -30, 30, 0, 1)
cpi_shock = st.sidebar.slider("CPI move (%)", -20, 20, 0, 1)
fuel_shock = st.sidebar.slider("Fuel price move (%)", -30, 30, 0, 1)
rate_shock = st.sidebar.slider("Repo rate move (%)", -30, 30, 0, 1)

run_forecast = st.sidebar.button("Run forecast", type="primary", use_container_width=True)
run_scenario = st.sidebar.button("Run scenario", use_container_width=True)
refresh = st.sidebar.button("Refresh data", use_container_width=True)

if refresh:
    st.cache_data.clear()
    st.rerun()

history_payload, history_error = api_get(api_key, f"/datasets/{asset_name}/series", limit=400)
history = (
    pd.DataFrame(history_payload["points"]).assign(date=lambda frame: pd.to_datetime(frame["date"]))
    if history_payload
    else pd.DataFrame()
)
if history_error:
    st.warning(history_error)

# ---------------------------------------------------------------------------
# Header + KPIs
# ---------------------------------------------------------------------------
st.title("Predictive Retail & Economic Data Dashboard")
st.caption(
    f"{summary['company_name']} · {len(assets)} asset(s) · {summary['total_rows']:,} stored observations"
)

asset_meta = next((item for item in summary["assets"] if item["asset_name"] == asset_name), None)
last_close = asset_meta["last_close"] if asset_meta else None

forecast_data: dict | None = st.session_state.get(f"forecast::{asset_name}::{horizon}")
scenario_data: dict | None = st.session_state.get(f"scenario::{asset_name}::{horizon}")

if run_forecast:
    with st.spinner("Scoring the ensemble model and computing SHAP attributions…"):
        payload, error = api_post(
            api_key,
            "/forecast",
            {
                "asset_name": asset_name,
                "horizon_days": horizon,
                "include_explanation": True,
            },
        )
    if error:
        st.error(error)
    else:
        st.session_state[f"forecast::{asset_name}::{horizon}"] = payload
        forecast_data = payload

if run_scenario:
    with st.spinner("Re-running the model under the scenario assumptions…"):
        payload, error = api_post(
            api_key,
            "/scenario",
            {
                "asset_name": asset_name,
                "horizon_days": horizon,
                "shocks": {
                    "zar_usd_pct": zar_shock / 100,
                    "cpi_pct": cpi_shock / 100,
                    "fuel_price_pct": fuel_shock / 100,
                    "repo_rate_pct": rate_shock / 100,
                },
                "include_explanation": True,
            },
        )
    if error:
        st.error(error)
    else:
        st.session_state[f"scenario::{asset_name}::{horizon}"] = payload
        scenario_data = payload

end_price = None
change_pct = None
if forecast_data and forecast_data.get("points"):
    end_price = forecast_data["points"][-1]["predicted_price"]
    if last_close:
        change_pct = (end_price / last_close - 1.0) * 100.0

columns = st.columns(4)
columns[0].markdown(
    kpi("Last observed close", money(last_close), asset_meta["last_date"] if asset_meta else None),
    unsafe_allow_html=True,
)
columns[1].markdown(
    kpi(
        f"Forecast close (+{horizon}d)",
        money(end_price),
        pct(change_pct) if change_pct is not None else "Run a forecast",
        positive=(change_pct or 0) >= 0,
    ),
    unsafe_allow_html=True,
)
columns[2].markdown(
    kpi(
        "Model",
        (forecast_data or {}).get("algorithm", "not loaded").replace("_", " ").title(),
        f"{(forecast_data or {}).get('latency_ms', 0):.0f} ms response",
    ),
    unsafe_allow_html=True,
)
columns[3].markdown(
    kpi(
        "Scenario impact",
        pct(scenario_data["summary"]["percent_change"]) if scenario_data else "—",
        money(scenario_data["summary"]["absolute_change"]) if scenario_data else "Configure in sidebar",
        positive=(scenario_data or {}).get("summary", {}).get("percent_change", 0) >= 0,
    ),
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
overview_tab, scenario_tab, explain_tab, data_tab = st.tabs(
    ["📈 Forecast", "🧪 Scenario testing", "🔍 Explainability (XAI)", "🗂 Data"]
)

with overview_tab:
    if forecast_data:
        figure = forecast_figure(
            history,
            forecast_data["points"],
            f"{asset_name} — {horizon}-day forecast ({forecast_data['algorithm']})",
        )
        st.plotly_chart(figure, use_container_width=True)

        table = pd.DataFrame(forecast_data["points"])
        table["predicted_price"] = table["predicted_price"].map(money)
        table["predicted_log_return"] = table["predicted_log_return"].map(lambda value: f"{value:.4%}")
        st.dataframe(table, use_container_width=True, height=240)
    else:
        st.info("Press **Run forecast** in the sidebar to generate a projection.")
        if not history.empty:
            st.plotly_chart(
                forecast_figure(history, [], f"{asset_name} — observed closing prices"),
                use_container_width=True,
            )

with scenario_tab:
    st.markdown(
        "Adjust the macro assumptions in the sidebar, then press **Run scenario**. "
        "The baseline and the shocked trajectory are scored by the same model."
    )
    if scenario_data:
        st.plotly_chart(
            scenario_figure(
                scenario_data["baseline"],
                scenario_data["scenario"],
                f"{asset_name} — baseline vs scenario ({horizon} business days)",
            ),
            use_container_width=True,
        )
        summary_block = scenario_data["summary"]
        left, middle, right = st.columns(3)
        left.metric("Baseline close", money(summary_block["baseline_end_price"]))
        middle.metric("Scenario close", money(summary_block["scenario_end_price"]))
        right.metric(
            "Delta",
            money(summary_block["absolute_change"]),
            pct(summary_block["percent_change"]),
        )
        st.caption(
            f"Largest per-step deviation: {money(summary_block['max_price_delta'])} · "
            f"Average deviation: {money(summary_block['average_price_delta'])}"
        )
    else:
        st.info("No scenario has been run for this asset yet.")

with explain_tab:
    if forecast_data and forecast_data.get("explanation"):
        explanation = forecast_data["explanation"]
        st.plotly_chart(shap_figure(explanation), use_container_width=True)
        st.caption(
            f"Attribution method: **{explanation['method']}** · "
            f"local-accuracy residual: {explanation['local_accuracy_delta']:.2e} "
            "(SHAP values plus the base value reconstruct the prediction exactly)."
        )
        with st.expander("Raw attribution values (log-return space)"):
            st.json(explanation["contributions_log_return"])
    else:
        st.info("Run a forecast with explanations enabled to see the SHAP breakdown.")

with data_tab:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Upload historical data")
        st.caption("CSV columns: asset_name, date_logged, closing_price, volume, zar_usd, cpi, fuel_price, repo_rate")
        upload = st.file_uploader("CSV export", type=["csv"])
        if upload is not None and st.button("Ingest dataset"):
            with st.spinner("Running ETL, validation and tenant-scoped upsert…"):
                payload, error = api_post(
                    api_key,
                    "/datasets/upload",
                    files={"file": (upload.name, upload.getvalue(), "text/csv")},
                )
            if error:
                st.error(error)
            else:
                st.success(f"Ingested {payload['rows_written']:,} rows for {payload['asset_count']} asset(s).")
                st.json(payload["etl"])

    with right:
        st.subheader("Stored assets")
        st.dataframe(
            pd.DataFrame(summary["assets"]),
            use_container_width=True,
            hide_index=True,
        )

    if not history.empty:
        st.subheader("Data quality & relationships")
        first, second = st.columns(2)
        with first:
            st.plotly_chart(correlation_figure(history), use_container_width=True)
        with second:
            macro_choice = st.selectbox(
                "Macro-indicator scatter",
                options=["zar_usd", "cpi", "fuel_price", "repo_rate"],
                format_func=lambda value: {
                    "zar_usd": "ZAR/USD",
                    "cpi": "CPI",
                    "fuel_price": "Fuel price",
                    "repo_rate": "Repo rate",
                }[value],
            )
            st.plotly_chart(
                macro_scatter(
                    history,
                    macro_choice,
                    {"zar_usd": "ZAR/USD", "cpi": "CPI", "fuel_price": "Fuel price", "repo_rate": "Repo rate"}[
                        macro_choice
                    ],
                ),
                use_container_width=True,
            )

st.sidebar.divider()
st.sidebar.caption(f"API: {API_BASE}")
if st.sidebar.button("Check API health"):
    root = API_BASE.removesuffix("/api/v1")
    try:
        health = requests.get(f"{root}/health", timeout=10).json()
    except (requests.RequestException, ValueError) as exc:
        st.sidebar.error(f"Health check failed: {exc}")
    else:
        st.sidebar.success(
            f"{health['status']} · v{health['version']} · cache {health['cache']['backend']}"
        )
