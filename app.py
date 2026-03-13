import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans

from phase2_core import (
    FORECAST_EXOGENOUS_COLUMNS,
    add_cluster_labels,
    add_dropoff_cluster_labels,
    build_hourly_cluster_demand,
    build_hourly_forecast_frame,
    build_single_pricing_features,
    dict_from_cluster_series,
    estimate_dropoff_transition_matrix,
    estimate_intercluster_travel_time_matrix,
    load_and_prepare_trip_data,
    normalize_weights,
    recommend_rebalancing_actions,
    run_dispatch_simulation,
    surge_multiplier,
)

st.set_page_config(
    page_title="Fleet Command Center",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Newsreader:opsz,wght@6..72,500;6..72,700&display=swap');

:root {
  --bg-1: #07111f;
  --bg-2: #0d1728;
  --card: #13233a;
  --card-2: #0f1c30;
  --ink: #e8eef7;
  --muted: #8ea2bf;
  --accent: #f59e0b;
  --accent-2: #38bdf8;
  --line: #223754;
  --success: #10b981;
}

html, body, [class*="css"] {
  font-family: "Space Grotesk", sans-serif;
  color: var(--ink) !important;
}

.stApp {
  background:
    radial-gradient(1200px 520px at -10% -15%, rgba(245, 158, 11, 0.08) 0%, transparent 55%),
    radial-gradient(1000px 520px at 110% -5%, rgba(56, 189, 248, 0.10) 0%, transparent 58%),
    linear-gradient(180deg, var(--bg-1) 0%, var(--bg-2) 100%);
}

.block-container {
  padding-top: 0.8rem;
  padding-bottom: 2rem;
}

h1, h2, h3 {
  font-family: "Newsreader", serif;
  color: #f8fbff !important;
}

[data-testid="stHeader"] {
  background: #070d18 !important;
  border-bottom: 1px solid #182338;
}

[data-testid="stToolbar"] {
  right: 0.7rem;
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0a1020 0%, #0f172a 100%) !important;
  border-right: 1px solid #182338;
}

[data-testid="stSidebar"] * {
  color: #d7e2f1 !important;
}

[data-testid="stSidebar"] .stRadio > div {
  gap: 0.15rem;
}

[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stRadio label {
  color: #aac0de !important;
}

div[data-testid="stMetric"] {
  background: linear-gradient(180deg, #152742 0%, #0f1d31 100%);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 0.8rem 0.9rem;
  box-shadow: 0 10px 24px rgba(3, 10, 22, 0.26);
}

div[data-testid="stMetricLabel"] {
  color: var(--muted) !important;
  opacity: 1 !important;
}

div[data-testid="stMetricValue"] {
  color: #f8fbff !important;
  font-size: 1.55rem;
  opacity: 1 !important;
}

.hero {
  background:
    radial-gradient(circle at top right, rgba(245, 158, 11, 0.18), transparent 24%),
    linear-gradient(120deg, #12263f 0%, #102c4a 45%, #0f4264 100%);
  border-radius: 22px;
  padding: 1.4rem 1.5rem;
  color: #f8fbff !important;
  border: 1px solid #29476b;
  box-shadow: 0 18px 44px rgba(2, 10, 24, 0.36);
  margin-bottom: 1rem;
}

.hero h1 {
  margin: 0;
  color: #ffffff;
  font-size: 2.3rem;
  line-height: 1;
}

.hero p {
  margin: 0.55rem 0 0;
  color: #d3e3f4;
  font-size: 1rem;
  max-width: 56rem;
}

.badge {
  display: inline-block;
  font-size: 0.75rem;
  margin-top: 0.85rem;
  margin-right: 0.45rem;
  padding: 0.32rem 0.72rem;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(255, 255, 255, 0.16);
  color: #f8fbff;
}

.sidebar-card {
  background: linear-gradient(180deg, #152742 0%, #0f1d31 100%);
  border: 1px solid #22314a;
  border-radius: 16px;
  padding: 0.9rem 1rem;
  margin-bottom: 0.7rem;
}

.caption {
  color: #aac0de !important;
}

.stMarkdown p, .stCaption, label {
  color: #d8e4f2 !important;
}

.section-kicker {
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 0.72rem;
  color: #7dd3fc;
  margin-bottom: 0.35rem;
}

.section-title {
  font-family: "Newsreader", serif;
  font-size: 2rem;
  line-height: 1;
  color: #f8fbff;
  margin: 0;
}

.section-copy {
  color: #9fb2cc;
  max-width: 52rem;
  margin-top: 0.55rem;
  margin-bottom: 1rem;
}

.feature-card {
  background: linear-gradient(180deg, #14253e 0%, #101d31 100%);
  border: 1px solid #233a58;
  border-radius: 18px;
  padding: 1rem 1rem 0.95rem 1rem;
  min-height: 148px;
  box-shadow: 0 14px 28px rgba(2, 10, 24, 0.22);
}

.feature-card .eyebrow {
  color: #7dd3fc;
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.feature-card .headline {
  color: #f8fbff;
  font-size: 1.15rem;
  font-weight: 700;
  margin-top: 0.5rem;
}

.feature-card .meta {
  color: #93a8c7;
  font-size: 0.92rem;
  margin-top: 0.45rem;
  line-height: 1.45;
}

.phase1-note {
  background: rgba(245, 158, 11, 0.08);
  border: 1px solid rgba(245, 158, 11, 0.2);
  border-radius: 14px;
  padding: 0.9rem 1rem;
  color: #f2ddb0;
  margin-bottom: 1rem;
}

button[role="tab"] {
  color: #8ca4c5 !important;
  font-weight: 600 !important;
  border-bottom: 2px solid transparent !important;
}

button[role="tab"][aria-selected="true"] {
  color: #e7effa !important;
  border-bottom: 2px solid #38bdf8 !important;
}

[data-testid="stDataFrame"] {
  border: 1px solid #22314a;
  border-radius: 14px;
  overflow: hidden;
}

[data-testid="stExpander"] {
  border: 1px solid #22314a !important;
  border-radius: 14px !important;
  background: #0f1b2f !important;
}

div[data-testid="stAlert"] {
  border: 1px solid #2b4467;
}

/* Prevent dimmed stale screen on reruns */
[data-stale="true"] {
  opacity: 1 !important;
}
[data-stale="true"] * {
  opacity: 1 !important;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data(filepath: str):
    try:
        return load_and_prepare_trip_data(filepath, nrows=50000)
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner=False)
def load_kmeans_model(model_path: str = "kmeans_fleet_model.pkl"):
    try:
        model = joblib.load(model_path)
        return model, "Online"
    except Exception:
        return None, "Missing"


def _read_json_file(path: Path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_resource(show_spinner=False)
def load_phase2_artifacts(models_dir: str = "phase2_models"):
    models_path = Path(models_dir)
    artifacts = {
        "ready": False,
        "forecast_model": None,
        "forecast_model_type": None,
        "forecast_meta": {},
        "forecast_validation": {},
        "forecast_cluster_metrics": {},
        "pricing_model": None,
        "pricing_model_type": None,
        "pricing_features": [],
        "pricing_importance": {},
        "pricing_holdout": {},
        "training_report": {},
        "baseline": {},
        "errors": [],
    }

    if not models_path.exists():
        artifacts["errors"].append("phase2_models directory not found.")
        return artifacts

    artifacts["forecast_meta"] = _read_json_file(models_path / "forecast_metadata.json")
    artifacts["forecast_validation"] = _read_json_file(models_path / "forecast_validation_metrics.json")
    artifacts["forecast_cluster_metrics"] = _read_json_file(models_path / "forecast_cluster_metrics.json")
    artifacts["pricing_importance"] = _read_json_file(models_path / "pricing_feature_importance.json")
    artifacts["pricing_holdout"] = _read_json_file(models_path / "pricing_holdout_metrics.json")
    artifacts["training_report"] = _read_json_file(models_path / "training_report.json")
    artifacts["baseline"] = _read_json_file(models_path / "demand_baseline.json")

    model_type = artifacts["forecast_meta"].get("model_type")
    artifacts["forecast_model_type"] = model_type
    if model_type == "keras":
        try:
            import tensorflow as tf  # type: ignore

            artifacts["forecast_model"] = tf.keras.models.load_model(
                models_path / "lstm_demand_forecaster.keras"
            )
        except Exception as exc:
            artifacts["errors"].append(f"Forecast model load failed (keras): {exc}")
    elif model_type == "sklearn_flat":
        try:
            artifacts["forecast_model"] = joblib.load(models_path / "lstm_demand_forecaster.pkl")
        except Exception as exc:
            artifacts["errors"].append(f"Forecast model load failed (sklearn): {exc}")
    else:
        artifacts["errors"].append("Unknown forecast model type in metadata.")

    xgb_path = models_path / "xgb_pricing_model.json"
    skl_path = models_path / "skl_pricing_model.pkl"
    if xgb_path.exists():
        try:
            from xgboost import XGBRegressor  # type: ignore

            mdl = XGBRegressor()
            mdl.load_model(str(xgb_path))
            artifacts["pricing_model"] = mdl
            artifacts["pricing_model_type"] = "xgboost"
        except Exception as exc:
            artifacts["errors"].append(f"Pricing model load failed (xgboost): {exc}")

    if artifacts["pricing_model"] is None and skl_path.exists():
        try:
            artifacts["pricing_model"] = joblib.load(skl_path)
            artifacts["pricing_model_type"] = "sklearn_gbr"
        except Exception as exc:
            artifacts["errors"].append(f"Pricing model load failed (sklearn): {exc}")

    artifacts["pricing_features"] = _read_json_file(models_path / "xgb_features.json")
    if not isinstance(artifacts["pricing_features"], list):
        artifacts["pricing_features"] = []

    artifacts["ready"] = (
        artifacts["forecast_model"] is not None and artifacts["pricing_model"] is not None
    )
    return artifacts


def infer_next_hour_demand(
    hourly_forecast_features: pd.DataFrame,
    artifacts: dict,
    n_clusters: int,
):
    model = artifacts.get("forecast_model")
    model_type = artifacts.get("forecast_model_type")
    meta = artifacts.get("forecast_meta", {})
    lookback = int(meta.get("lookback_hours", 24))
    expected_features = int(meta.get("input_feature_count", n_clusters))

    if model is None or len(hourly_forecast_features) < lookback:
        return None

    recent = hourly_forecast_features.tail(lookback).values.astype(np.float32)
    if recent.shape[1] > expected_features:
        recent = recent[:, :expected_features]
    elif recent.shape[1] < expected_features:
        pad = np.zeros((lookback, expected_features - recent.shape[1]), dtype=np.float32)
        recent = np.hstack([recent, pad])

    try:
        if model_type == "keras":
            pred = model.predict(recent[np.newaxis, :, :], verbose=0)[0]
        else:
            pred = model.predict(recent.reshape(1, -1))[0]
        pred = np.asarray(pred, dtype=float).reshape(-1)
        if len(pred) > n_clusters:
            pred = pred[:n_clusters]
        elif len(pred) < n_clusters:
            pred = np.pad(pred, (0, n_clusters - len(pred)), mode="constant")
        return np.maximum(pred, 0.0)
    except Exception:
        return None


def _standard_layout(fig, title: str, height: int = 430):
    fig.update_layout(
        title=title,
        template="plotly_dark",
        font=dict(color="#d7e2f1"),
        title_font=dict(color="#f8fbff"),
        height=height,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _page_intro(kicker: str, title: str, copy: str):
    st.markdown(
        f"""
<div class="section-kicker">{kicker}</div>
<div class="section-title">{title}</div>
<div class="section-copy">{copy}</div>
""",
        unsafe_allow_html=True,
    )


def _feature_card(eyebrow: str, headline: str, meta: str):
    st.markdown(
        f"""
<div class="feature-card">
  <div class="eyebrow">{eyebrow}</div>
  <div class="headline">{headline}</div>
  <div class="meta">{meta}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def build_phase2_state(df_clustered: pd.DataFrame, kmeans_model, artifacts: dict) -> dict:
    n_clusters = int(
        getattr(kmeans_model, "n_clusters", artifacts.get("forecast_meta", {}).get("n_clusters", 8))
    )
    hourly_demand = build_hourly_cluster_demand(df_clustered, n_clusters=n_clusters)
    hourly_forecast_features = build_hourly_forecast_frame(df_clustered, n_clusters=n_clusters)
    forecast = infer_next_hour_demand(hourly_forecast_features, artifacts, n_clusters=n_clusters)

    if forecast is None and not hourly_demand.empty:
        forecast = hourly_demand.tail(1).values[0]

    baseline = artifacts.get("baseline", {})
    if not baseline and not hourly_demand.empty:
        baseline = {str(i): float(hourly_demand[i].mean()) for i in range(n_clusters)}

    current_supply = (
        hourly_demand.tail(1).values[0]
        if not hourly_demand.empty
        else np.ones(n_clusters, dtype=float)
    )

    pred_df = None
    rebalancing = None
    if forecast is not None:
        cluster_labels = [f"Cluster {i}" for i in range(n_clusters)]
        pred_df = pd.DataFrame(
            {"cluster": cluster_labels, "predicted_demand": forecast, "current_supply": current_supply}
        )
        pred_df["supply_gap"] = pred_df["predicted_demand"] - pred_df["current_supply"]
        pred_df = pred_df.sort_values("predicted_demand", ascending=False).reset_index(drop=True)
        pred_df["rank"] = np.arange(1, len(pred_df) + 1)
        rebalancing = recommend_rebalancing_actions(
            predicted_demand=forecast,
            current_supply=current_supply,
            top_k=3,
        )

    return {
        "n_clusters": n_clusters,
        "hourly_demand": hourly_demand,
        "hourly_forecast_features": hourly_forecast_features,
        "forecast": forecast,
        "baseline": baseline,
        "current_supply": current_supply,
        "pred_df": pred_df,
        "rebalancing": rebalancing,
    }


def run_live_operations(df: pd.DataFrame, model: KMeans | None):
    st.subheader("Live Operations")
    selected_hour = st.sidebar.slider("Live hour", 0, 23, 18)
    df_view = df[df["hour"] == selected_hour].copy()

    if df_view.empty:
        st.warning("No records available for this hour in the loaded sample.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active pickups", f"{len(df_view):,}")
    col2.metric("Avg duration", f"{df_view['trip_duration'].mean() / 60:.1f} min")
    col3.metric("Avg speed", f"{df_view['speed_kmh'].mean():.1f} km/h")
    peak_status = "Critical" if len(df_view) > 2500 else "High" if len(df_view) > 1500 else "Normal"
    col4.metric("Ops status", peak_status)

    if model is not None:
        coords = df_view[["pickup_latitude", "pickup_longitude"]]
        df_view["cluster"] = model.predict(coords)
        fig = px.scatter_mapbox(
            df_view,
            lat="pickup_latitude",
            lon="pickup_longitude",
            color="cluster",
            size="passenger_count",
            zoom=11,
            height=610,
            mapbox_style="carto-darkmatter",
            color_continuous_scale="Teal",
        )
        _standard_layout(fig, f"AI Demand Zones at {selected_hour:02d}:00", height=610)
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.density_mapbox(
            df_view,
            lat="pickup_latitude",
            lon="pickup_longitude",
            z="passenger_count",
            radius=10,
            zoom=11,
            mapbox_style="carto-darkmatter",
            height=610,
        )
        _standard_layout(fig, f"Demand Density at {selected_hour:02d}:00", height=610)
        st.plotly_chart(fig, use_container_width=True)


def run_research_lab(df: pd.DataFrame):
    st.subheader("Research Lab")
    st.caption("Compare centroid-based and density-based spatial segmentation.")

    sample_limit = max(1000, min(12000, len(df)))
    sample_size = st.slider("Sample size", 1000, sample_limit, min(6000, sample_limit))
    sampled = df.sample(sample_size, random_state=42)

    c1, c2 = st.columns(2)
    with c1:
        k = st.slider("K-means clusters", 3, 15, 8)
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(
            sampled[["pickup_latitude", "pickup_longitude"]]
        )
        sampled["kmeans"] = km.labels_.astype(str)
        fig_k = px.scatter_mapbox(
            sampled,
            lat="pickup_latitude",
            lon="pickup_longitude",
            color="kmeans",
            zoom=10,
            height=500,
            mapbox_style="carto-darkmatter",
        )
        _standard_layout(fig_k, "K-Means Partitions", height=500)
        st.plotly_chart(fig_k, use_container_width=True)

    with c2:
        try:
            import hdbscan  # type: ignore

            min_cluster_size = st.slider("HDBSCAN min cluster size", 10, 120, 35)
            hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
            sampled["hdbscan"] = hdb.fit_predict(sampled[["pickup_latitude", "pickup_longitude"]])
            noise_points = int((sampled["hdbscan"] == -1).sum())
            fig_h = px.scatter_mapbox(
                sampled,
                lat="pickup_latitude",
                lon="pickup_longitude",
                color="hdbscan",
                zoom=10,
                height=500,
                mapbox_style="carto-darkmatter",
            )
            _standard_layout(fig_h, f"HDBSCAN Hotspots (Noise={noise_points})", height=500)
            st.plotly_chart(fig_h, use_container_width=True)
        except ImportError:
            st.info("Install `hdbscan` to run this comparison panel.")


def run_efficiency_analysis(df: pd.DataFrame):
    st.subheader("Efficiency Analysis")
    tab1, tab2 = st.tabs(["Fleet velocity", "Trip durations"])

    with tab1:
        avg_speed = df.groupby("hour")["speed_kmh"].mean().reset_index()
        fig_speed = px.line(
            avg_speed,
            x="hour",
            y="speed_kmh",
            markers=True,
            line_shape="spline",
            color_discrete_sequence=["#0f766e"],
        )
        slowest_idx = int(avg_speed["speed_kmh"].idxmin())
        fig_speed.add_annotation(
            x=int(avg_speed.loc[slowest_idx, "hour"]),
            y=float(avg_speed.loc[slowest_idx, "speed_kmh"]),
            text="Slowest hour",
            showarrow=True,
            arrowhead=2,
        )
        _standard_layout(fig_speed, "Average Fleet Speed by Hour")
        st.plotly_chart(fig_speed, use_container_width=True)

    with tab2:
        fig_hist = px.histogram(
            df,
            x="trip_duration",
            nbins=90,
            range_x=[0, 3600],
            color_discrete_sequence=["#0284c7"],
        )
        _standard_layout(fig_hist, "Trip Duration Distribution")
        st.plotly_chart(fig_hist, use_container_width=True)


def _render_forecast_evaluation_panel(artifacts: dict):
    eval_cols = st.columns(3)
    best = artifacts.get("forecast_meta", {}).get("best_validation", {})
    eval_cols[0].metric("Best val MAE", f"{float(best.get('val_mae', 0.0)):.3f}")
    eval_cols[1].metric("Best val RMSE", f"{float(best.get('val_rmse', 0.0)):.3f}")
    eval_cols[2].metric(
        "Sparse-hour weighting",
        "Enabled" if artifacts.get("forecast_meta", {}).get("sparse_hour_weighting", {}).get("enabled") else "Off",
    )

    val_log = artifacts.get("forecast_validation", {}).get("epochs", [])
    if val_log:
        val_df = pd.DataFrame(val_log)
        if "epoch" in val_df.columns:
            fig_val = px.line(
                val_df,
                x="epoch",
                y=[c for c in ["val_mae", "val_rmse"] if c in val_df.columns],
                markers=True,
                color_discrete_sequence=["#1d4ed8", "#0f766e"],
            )
            _standard_layout(fig_val, "Validation Learning Curve", height=360)
            st.plotly_chart(fig_val, use_container_width=True)


def render_phase2_overview(state: dict, artifacts: dict):
    _page_intro(
        "Phase 2 Command",
        "Predictive operations, pricing, and dispatch in one place.",
        "This dashboard now treats Phase 2 as the main product. Forecasting, pricing, rebalancing, and evaluation are split into dedicated views so the system feels like a real command center instead of a nested demo.",
    )

    pred_df = state["pred_df"]
    forecast = state["forecast"]
    top_cluster = pred_df.iloc[0]["cluster"] if pred_df is not None else "Unavailable"
    total_pred = float(pred_df["predicted_demand"].sum()) if pred_df is not None else 0.0
    largest_gap = float(pred_df["supply_gap"].max()) if pred_df is not None else 0.0

    c1, c2, c3 = st.columns(3)
    with c1:
        _feature_card("Demand Focus", str(top_cluster), "Highest projected next-hour pickup demand.")
    with c2:
        _feature_card("Predicted Volume", f"{total_pred:.2f}", "Aggregate next-hour forecast across all clusters.")
    with c3:
        _feature_card("Rebalancing Gap", f"{largest_gap:.2f}", "Largest supply shortfall to act on immediately.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Forecast model", artifacts.get("forecast_model_type", "missing"))
    m2.metric("Pricing model", artifacts.get("pricing_model_type", "missing"))
    m3.metric("Clusters", state["n_clusters"])
    m4.metric("Exogenous signals", len(FORECAST_EXOGENOUS_COLUMNS))

    if pred_df is not None:
        left, right = st.columns([1.4, 1])

        with left:
            fig_pred = px.bar(
                pred_df.head(8),
                x="cluster",
                y="predicted_demand",
                color="supply_gap",
                color_continuous_scale="Tealgrn",
            )
            _standard_layout(fig_pred, "Forecast Snapshot by Cluster", height=420)
            st.plotly_chart(fig_pred, use_container_width=True)

        with right:
            st.markdown("### Immediate action")
            top3 = pred_df.head(3)
            cols = st.columns(3)
            for idx, (_, row) in enumerate(top3.iterrows()):
                with cols[idx]:
                    _feature_card(
                        f"Rank {int(row['rank'])}",
                        row["cluster"],
                        f"Demand {row['predicted_demand']:.2f} | Gap {row['supply_gap']:.2f}",
                    )

    fc_metrics = artifacts.get("forecast_cluster_metrics", {}).get("overall", {})
    hold = artifacts.get("pricing_holdout", {})
    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        st.markdown("### Forecast quality")
        st.metric("Hold-out RMSE", f"{float(fc_metrics.get('rmse', 0.0)):.3f}")
        st.metric("Hold-out MAE", f"{float(fc_metrics.get('mae', 0.0)):.3f}")
    with bottom_right:
        st.markdown("### Pricing quality")
        st.metric("Hold-out R2", f"{float(hold.get('r2', 0.0)):.3f}")
        st.metric("Hold-out RMSE", f"{float(hold.get('rmse', 0.0)):.2f}")


def render_phase2_demand(state: dict):
    _page_intro(
        "Phase 2 / Forecasting",
        "Demand Forecasting",
        f"One-hour-ahead demand forecast with exogenous signals: {', '.join(FORECAST_EXOGENOUS_COLUMNS)}.",
    )

    pred_df = state["pred_df"]
    hourly_demand = state["hourly_demand"]
    if pred_df is None:
        st.warning("Forecast model is unavailable. Run `python train_phase2_models.py`.")
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Top cluster", pred_df.loc[0, "cluster"])
    k2.metric("Top predicted trips", f"{pred_df.loc[0, 'predicted_demand']:.2f}")
    k3.metric("Total predicted demand", f"{pred_df['predicted_demand'].sum():.2f}")
    k4.metric("Largest gap", f"{pred_df['supply_gap'].max():.2f}")

    chart_col, side_col = st.columns([1.55, 1])
    with chart_col:
        fig_pred = px.bar(
            pred_df,
            x="cluster",
            y="predicted_demand",
            color="supply_gap",
            color_continuous_scale="Tealgrn",
        )
        _standard_layout(fig_pred, "Next-Hour Demand Forecast by Cluster", height=440)
        st.plotly_chart(fig_pred, use_container_width=True)

    with side_col:
        st.markdown("### Top 3 clusters")
        for _, row in pred_df.head(3).iterrows():
            _feature_card(
                f"Rank {int(row['rank'])}",
                row["cluster"],
                f"Predicted demand {row['predicted_demand']:.2f}<br>Current supply {row['current_supply']:.2f}<br>Gap {row['supply_gap']:.2f}",
            )

    recent = hourly_demand.tail(24)
    if not recent.empty:
        heat = recent.T.reindex(range(state["n_clusters"])).fillna(0.0)
        heat.index = [f"Cluster {i}" for i in range(state["n_clusters"])]
        fig_heat = go.Figure(
            data=go.Heatmap(
                z=heat.values,
                x=[x.strftime("%d %b %H:%M") for x in recent.index],
                y=heat.index.tolist(),
                colorscale="YlGnBu",
                zmin=0,
                zmax=max(1.0, float(np.nanmax(heat.values))),
                colorbar=dict(title="Trips"),
                hovertemplate="Cluster: %{y}<br>Hour: %{x}<br>Trips: %{z}<extra></extra>",
            )
        )
        _standard_layout(fig_heat, "Last 24 Hours Demand Heatmap", height=450)
        fig_heat.update_xaxes(tickangle=-30)
        st.plotly_chart(fig_heat, use_container_width=True)

    with st.expander("Open full cluster forecast table"):
        st.dataframe(
            pred_df[["rank", "cluster", "predicted_demand", "current_supply", "supply_gap"]],
            use_container_width=True,
            hide_index=True,
        )


def render_phase2_pricing(df_clustered: pd.DataFrame, kmeans_model, artifacts: dict, state: dict):
    _page_intro(
        "Phase 2 / Pricing",
        "Dynamic Pricing",
        "Estimate trip duration, demand ratio, and surge from route geometry, time-of-day, and external demand signals.",
    )

    if artifacts.get("pricing_model") is None:
        st.warning("Pricing model artifacts are missing. Run `python train_phase2_models.py`.")
        return

    default_pick_lat = float(df_clustered["pickup_latitude"].median())
    default_pick_lon = float(df_clustered["pickup_longitude"].median())
    default_drop_lat = float(df_clustered["dropoff_latitude"].median())
    default_drop_lon = float(df_clustered["dropoff_longitude"].median())

    c1, c2 = st.columns(2)
    with c1:
        pickup_lat = st.number_input("Pickup latitude", value=default_pick_lat, format="%.6f")
        pickup_lon = st.number_input("Pickup longitude", value=default_pick_lon, format="%.6f")
        passenger_count = st.slider("Passenger count", 1, 6, 1)
        selected_hour = st.slider("Trip hour", 0, 23, 18, key="phase2_pricing_hour")
    with c2:
        dropoff_lat = st.number_input("Dropoff latitude", value=default_drop_lat, format="%.6f")
        dropoff_lon = st.number_input("Dropoff longitude", value=default_drop_lon, format="%.6f")
        weather_temp = st.slider("Weather temp (C)", -5.0, 42.0, 24.0)
        event_intensity = st.slider("Event intensity", 0.0, 3.0, 0.8, 0.1)

    ex1, ex2, ex3 = st.columns(3)
    precip = ex1.slider("Precipitation (mm)", 0.0, 12.0, 1.2, 0.1)
    holiday = ex2.selectbox("Holiday flag", [0.0, 1.0], format_func=lambda x: "Yes" if x else "No")
    use_forecast_ratio = ex3.checkbox("Use forecast demand ratio", value=True)
    manual_ratio = st.slider("Manual demand ratio", 0.5, 3.0, 1.2, 0.1)

    speed_by_hour = df_clustered.groupby("hour")["speed_kmh"].mean().to_dict()
    assumed_speed = float(speed_by_hour.get(selected_hour, df_clustered["speed_kmh"].mean()))

    if st.button("Estimate duration and surge", use_container_width=True):
        base_date = pd.Timestamp("2026-01-05") + pd.Timedelta(hours=int(selected_hour))
        feature_row = build_single_pricing_features(
            pickup_dt=base_date,
            pickup_longitude=pickup_lon,
            pickup_latitude=pickup_lat,
            dropoff_longitude=dropoff_lon,
            dropoff_latitude=dropoff_lat,
            passenger_count=passenger_count,
            assumed_speed_kmh=assumed_speed,
            weather_temp_c=weather_temp,
            precipitation_mm=precip,
            is_holiday=holiday,
            event_intensity=event_intensity,
        )

        expected_columns = artifacts.get("pricing_features") or list(feature_row.columns)
        features_used = feature_row.reindex(columns=expected_columns, fill_value=0.0)
        pred_duration = float(artifacts["pricing_model"].predict(features_used)[0])
        pred_duration = float(np.clip(pred_duration, 60.0, 7200.0))

        pickup_cluster = int(kmeans_model.predict(np.array([[pickup_lat, pickup_lon]], dtype=float))[0])
        demand_ratio = manual_ratio
        if use_forecast_ratio and state["forecast"] is not None:
            base = float(state["baseline"].get(str(pickup_cluster), 1.0))
            demand_ratio = float(state["forecast"][pickup_cluster] / max(base, 1.0))

        surge = surge_multiplier(demand_ratio=demand_ratio, base=1.0, alpha=0.85, max_surge=3.0)
        fare_proxy = (pred_duration / 60.0) * 1.9 * surge

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted duration", f"{pred_duration / 60:.1f} min")
        m2.metric("Demand ratio", f"{demand_ratio:.2f}")
        m3.metric("Surge multiplier", f"{surge:.2f}x")
        m4.metric("Fare proxy", f"${fare_proxy:.2f}")
        st.caption(f"Pickup cluster: {pickup_cluster} | Pricing model: {artifacts.get('pricing_model_type')}")
        st.dataframe(features_used, use_container_width=True, hide_index=True)

    importance_rows = artifacts.get("pricing_importance", {}).get("feature_importance", [])
    if importance_rows:
        imp_df = pd.DataFrame(importance_rows).head(10)
        fig_imp = px.bar(
            imp_df,
            x="importance_normalized",
            y="feature",
            orientation="h",
            color="importance_normalized",
            color_continuous_scale="Blues",
        )
        _standard_layout(fig_imp, "Pricing Driver Importance", height=410)
        fig_imp.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig_imp, use_container_width=True)


def render_phase2_dispatch(df_clustered: pd.DataFrame, state: dict):
    _page_intro(
        "Phase 2 / Dispatch",
        "Dispatch and Rebalancing",
        "Translate forecast into repositioning decisions, then validate the strategy with a more realistic simulation based on travel time, service time, and driver availability.",
    )

    forecast = state["forecast"]
    rebalancing = state["rebalancing"]

    requests = st.slider("Simulated requests", 500, 9000, 2500, step=250)
    drivers = st.slider("Available drivers", 80, 1000, 420, step=20)
    service_mean = st.slider("Mean service time (sec)", 240, 1500, 600, step=30)
    rebalance_every = st.slider("AI rebalance frequency (requests)", 20, 250, 80, step=10)

    demand_profile = st.radio(
        "Request demand profile",
        ["Historical distribution", "Forecast-adjusted distribution"],
        horizontal=True,
    )

    counts = dict_from_cluster_series(df_clustered["cluster"], n_clusters=state["n_clusters"])
    historical_weights = normalize_weights([counts[str(i)] for i in range(state["n_clusters"])])
    if demand_profile == "Forecast-adjusted distribution" and forecast is not None:
        demand_weights = normalize_weights(forecast)
    else:
        demand_weights = historical_weights

    travel_matrix = estimate_intercluster_travel_time_matrix(df_clustered, n_clusters=state["n_clusters"])
    transition_matrix = estimate_dropoff_transition_matrix(df_clustered, n_clusters=state["n_clusters"])
    ai_weights = rebalancing.get("target_weights") if rebalancing else None
    result = run_dispatch_simulation(
        demand_weights=demand_weights,
        num_requests=int(requests),
        seed=42,
        travel_time_matrix=travel_matrix,
        transition_matrix=transition_matrix,
        num_drivers=int(drivers),
        service_time_mean_sec=float(service_mean),
        ai_supply_weights=ai_weights,
        rebalance_every_n_requests=int(rebalance_every),
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Random mean wait", f"{result.random_mean_wait:.1f}s")
    m2.metric("AI mean wait", f"{result.ai_mean_wait:.1f}s")
    m3.metric("Mean improvement", f"{result.mean_improvement_pct:.1f}%")
    m4.metric("Service level gain", f"{result.ai_service_level_300 - result.random_service_level_300:.1f} pp")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Random P95 wait", f"{result.random_p95_wait:.1f}s")
    m6.metric("AI P95 wait", f"{result.ai_p95_wait:.1f}s")
    m7.metric("Random utilization", f"{result.random_driver_utilization:.1f}%")
    m8.metric("AI utilization", f"{result.ai_driver_utilization:.1f}%")

    left, right = st.columns([1.35, 1])
    with left:
        hist_df = pd.DataFrame(
            {
                "wait_time_sec": np.concatenate([result.random_wait_times, result.ai_wait_times]),
                "strategy": ["Random baseline"] * len(result.random_wait_times)
                + ["AI-guided"] * len(result.ai_wait_times),
            }
        )
        fig_hist = px.histogram(
            hist_df,
            x="wait_time_sec",
            color="strategy",
            barmode="overlay",
            nbins=36,
            opacity=0.7,
            color_discrete_map={"Random baseline": "#f97316", "AI-guided": "#0ea5a6"},
        )
        _standard_layout(fig_hist, "Dispatch Wait-Time Distribution", height=420)
        st.plotly_chart(fig_hist, use_container_width=True)

    with right:
        st.markdown("### Rebalancing moves")
        if rebalancing:
            top_df = pd.DataFrame(rebalancing.get("top_clusters", []))
            if not top_df.empty:
                top_df["cluster"] = top_df["cluster"].apply(lambda x: f"Cluster {x}")
                st.dataframe(top_df, use_container_width=True, hide_index=True)

            actions_df = pd.DataFrame(rebalancing.get("actions", []))
            if not actions_df.empty:
                actions_df["from_cluster"] = actions_df["from_cluster"].apply(lambda x: f"Cluster {x}")
                actions_df["to_cluster"] = actions_df["to_cluster"].apply(lambda x: f"Cluster {x}")
                st.dataframe(actions_df, use_container_width=True, hide_index=True)
            else:
                st.info("No strong relocation needed for this forecast window.")

    travel_df = pd.DataFrame(travel_matrix)
    travel_df.index = [f"From C{i}" for i in range(state["n_clusters"])]
    travel_df.columns = [f"To C{i}" for i in range(state["n_clusters"])]
    fig_t = go.Figure(
        data=go.Heatmap(
            z=travel_df.values,
            x=travel_df.columns.tolist(),
            y=travel_df.index.tolist(),
            colorscale="Blues",
            colorbar=dict(title="Sec"),
        )
    )
    _standard_layout(fig_t, "Inter-Cluster Travel Time Matrix", height=420)
    st.plotly_chart(fig_t, use_container_width=True)


def render_phase2_evaluation(artifacts: dict):
    _page_intro(
        "Phase 2 / Evaluation",
        "Model Evaluation",
        "This page collects the validation story for forecasting and pricing, including hold-out performance and k-fold stability checks.",
    )

    _render_forecast_evaluation_panel(artifacts)

    fc_metrics = artifacts.get("forecast_cluster_metrics", {}).get("clusters", [])
    if fc_metrics:
        st.markdown("### Forecast errors by cluster")
        mdf = pd.DataFrame(fc_metrics)
        mdf["cluster"] = mdf["cluster"].apply(lambda x: f"Cluster {x}")
        st.dataframe(mdf, use_container_width=True, hide_index=True)

    st.markdown("### Pricing hold-out metrics")
    hold = artifacts.get("pricing_holdout", {})
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("MAE", f"{float(hold.get('mae', 0.0)):.2f}")
    p2.metric("RMSE", f"{float(hold.get('rmse', 0.0)):.2f}")
    p3.metric("R2", f"{float(hold.get('r2', 0.0)):.3f}")
    p4.metric("P90 |error|", f"{float(hold.get('abs_error_p90', 0.0)):.2f}")

    cv_payload = artifacts.get("training_report", {}).get("pricing", {}).get("cross_validation", {})
    fold_rows = cv_payload.get("fold_metrics", []) if isinstance(cv_payload, dict) else []
    if fold_rows:
        cv_df = pd.DataFrame(fold_rows)
        fig_cv = px.bar(
            cv_df,
            x="fold",
            y=["mae", "rmse"],
            barmode="group",
            color_discrete_sequence=["#f59e0b", "#38bdf8"],
        )
        _standard_layout(fig_cv, "Pricing K-Fold Stability", height=360)
        st.plotly_chart(fig_cv, use_container_width=True)


def run_phase1_tools(df: pd.DataFrame, model: KMeans | None):
    _page_intro(
        "Phase 1 Support",
        "Phase 1 Tools",
        "Phase 1 is still available for supporting visuals and baseline analysis, but it now sits behind the main Phase 2 workflow.",
    )
    st.markdown(
        '<div class="phase1-note">Phase 1 is kept as a supporting module for hotspot visualization, clustering comparison, and fleet-efficiency diagnostics. The primary project focus remains Phase 2.</div>',
        unsafe_allow_html=True,
    )
    tabs = st.tabs(["Live Operations", "Research Lab", "Efficiency Analysis"])
    with tabs[0]:
        run_live_operations(df, model)
    with tabs[1]:
        run_research_lab(df)
    with tabs[2]:
        run_efficiency_analysis(df)


def main():
    st.markdown(
        """
<div class="hero">
  <h1>Intelligent Fleet Allocation System</h1>
  <p>Phase 2 command dashboard for demand forecasting, dynamic pricing, dispatch simulation, and model evaluation. Phase 1 tools remain available as supporting diagnostics.</p>
  <span class="badge">NSUT B.Tech Project</span>
  <span class="badge">Phase 2 Focus</span>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown(
            '<div class="sidebar-card"><b>Phase 2 Command</b><br><span class="caption">Forecasting, pricing, simulation, and evaluation</span></div>',
            unsafe_allow_html=True,
        )
        view_mode = st.radio(
            "Workspace",
            [
                "Phase 2 Overview",
                "Demand Forecasting",
                "Dynamic Pricing",
                "Dispatch & Rebalancing",
                "Model Evaluation",
                "Phase 1 Tools",
            ],
        )

    with st.spinner("Loading data and models..."):
        df = load_data("train_small.csv")
        kmeans_model, kmeans_status = load_kmeans_model("kmeans_fleet_model.pkl")
        phase2_artifacts = load_phase2_artifacts("phase2_models")

    if df is None:
        st.error("`train_small.csv` is missing. Add it to the project root and rerun.")
        st.stop()

    if kmeans_model is not None:
        df_clustered = add_cluster_labels(df, kmeans_model)
        df_clustered = add_dropoff_cluster_labels(df_clustered, kmeans_model)
    else:
        df_clustered = df.copy()
        df_clustered["cluster"] = 0
        df_clustered["dropoff_cluster"] = 0

    state = None
    if kmeans_model is not None:
        state = build_phase2_state(df_clustered, kmeans_model, phase2_artifacts)

    if view_mode != "Phase 1 Tools":
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Records", f"{len(df):,}")
        k2.metric("KMeans", kmeans_status)
        k3.metric("Phase 2 artifacts", "Ready" if phase2_artifacts.get("ready") else "Partial")
        k4.metric("External signals", ", ".join(FORECAST_EXOGENOUS_COLUMNS[:2]) + " +3")

    if phase2_artifacts.get("errors"):
        with st.expander("Artifact diagnostics"):
            for err in phase2_artifacts["errors"]:
                st.write(f"- {err}")
            st.code("python train_phase2_models.py")

    st.markdown("---")

    if view_mode == "Phase 1 Tools":
        run_phase1_tools(df, kmeans_model)
    else:
        if kmeans_model is None:
            st.error("Phase 2 requires `kmeans_fleet_model.pkl`.")
        else:
            if view_mode == "Phase 2 Overview":
                render_phase2_overview(state, phase2_artifacts)
            elif view_mode == "Demand Forecasting":
                render_phase2_demand(state)
            elif view_mode == "Dynamic Pricing":
                render_phase2_pricing(df_clustered, kmeans_model, phase2_artifacts, state)
            elif view_mode == "Dispatch & Rebalancing":
                render_phase2_dispatch(df_clustered, state)
            else:
                render_phase2_evaluation(phase2_artifacts)


if __name__ == "__main__":
    main()
