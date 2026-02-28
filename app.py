import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans

from phase2_core import (
    add_cluster_labels,
    build_hourly_cluster_demand,
    build_single_pricing_features,
    dict_from_cluster_series,
    load_and_prepare_trip_data,
    run_dispatch_simulation,
    surge_multiplier,
)

st.set_page_config(
    page_title="Fleet Command Center (Phase 1 + Phase 2)",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp { background-color: #0E1117; color: white; }
    div.stButton > button { background-color: #4B5563; color: white; border-radius: 5px; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #00CC96; }
    h1, h2, h3 { color: #E5E7EB; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_data(filepath: str):
    try:
        return load_and_prepare_trip_data(filepath, nrows=50000)
    except FileNotFoundError:
        return None


@st.cache_resource
def load_kmeans_model(model_path: str = "kmeans_fleet_model.pkl"):
    try:
        model = joblib.load(model_path)
        return model, "✅ Online (Pre-Trained)"
    except Exception:
        return None, "⚠️ Offline (File Missing)"


@st.cache_resource
def load_phase2_artifacts(models_dir: str = "phase2_models"):
    models_path = Path(models_dir)
    artifacts = {
        "ready": False,
        "forecast_model": None,
        "forecast_model_type": None,
        "forecast_meta": {},
        "pricing_model": None,
        "pricing_model_type": None,
        "baseline": {},
        "pricing_features": [],
        "errors": [],
    }

    if not models_path.exists():
        artifacts["errors"].append("phase2_models/ directory not found.")
        return artifacts

    # Forecasting metadata + model
    forecast_meta_path = models_path / "forecast_metadata.json"
    if forecast_meta_path.exists():
        try:
            artifacts["forecast_meta"] = json.loads(forecast_meta_path.read_text(encoding="utf-8"))
            model_type = artifacts["forecast_meta"].get("model_type")
            artifacts["forecast_model_type"] = model_type

            if model_type == "keras":
                try:
                    import tensorflow as tf  # type: ignore

                    artifacts["forecast_model"] = tf.keras.models.load_model(
                        models_path / "lstm_demand_forecaster.keras"
                    )
                except Exception as exc:
                    artifacts["errors"].append(
                        f"Could not load TensorFlow LSTM model: {exc}"
                    )
            elif model_type == "sklearn_flat":
                pkl_path = models_path / "lstm_demand_forecaster.pkl"
                if pkl_path.exists():
                    artifacts["forecast_model"] = joblib.load(pkl_path)
                else:
                    artifacts["errors"].append("Fallback forecasting model .pkl is missing.")
        except Exception as exc:
            artifacts["errors"].append(f"Invalid forecast metadata: {exc}")
    else:
        artifacts["errors"].append("forecast_metadata.json missing.")

    # Pricing model
    xgb_path = models_path / "xgb_pricing_model.json"
    skl_path = models_path / "skl_pricing_model.pkl"
    if xgb_path.exists():
        try:
            from xgboost import XGBRegressor  # type: ignore

            xgb_model = XGBRegressor()
            xgb_model.load_model(str(xgb_path))
            artifacts["pricing_model"] = xgb_model
            artifacts["pricing_model_type"] = "xgboost"
        except Exception as exc:
            artifacts["errors"].append(f"Could not load XGBoost model: {exc}")

    if artifacts["pricing_model"] is None and skl_path.exists():
        try:
            artifacts["pricing_model"] = joblib.load(skl_path)
            artifacts["pricing_model_type"] = "sklearn_gbr"
        except Exception as exc:
            artifacts["errors"].append(f"Could not load sklearn pricing model: {exc}")

    feature_path = models_path / "xgb_features.json"
    if feature_path.exists():
        try:
            artifacts["pricing_features"] = json.loads(feature_path.read_text(encoding="utf-8"))
        except Exception as exc:
            artifacts["errors"].append(f"Could not parse xgb_features.json: {exc}")

    baseline_path = models_path / "demand_baseline.json"
    if baseline_path.exists():
        try:
            artifacts["baseline"] = json.loads(baseline_path.read_text(encoding="utf-8"))
        except Exception as exc:
            artifacts["errors"].append(f"Could not parse demand_baseline.json: {exc}")

    artifacts["ready"] = (
        artifacts["forecast_model"] is not None and artifacts["pricing_model"] is not None
    )
    return artifacts


def infer_next_hour_demand(hourly_demand: pd.DataFrame, artifacts: dict):
    model = artifacts.get("forecast_model")
    model_type = artifacts.get("forecast_model_type")
    lookback = int(artifacts.get("forecast_meta", {}).get("lookback_hours", 24))

    if model is None or len(hourly_demand) < lookback:
        return None

    recent_window = hourly_demand.tail(lookback).values.astype(np.float32)
    try:
        if model_type == "keras":
            pred = model.predict(recent_window[np.newaxis, :, :], verbose=0)[0]
        else:
            pred = model.predict(recent_window.reshape(1, -1))[0]
        return np.maximum(pred, 0.0)
    except Exception:
        return None


def run_live_operations(df: pd.DataFrame, model: KMeans | None, map_config: dict):
    st.sidebar.markdown("### ⚙️ Dispatch Controls")
    selected_hour = st.sidebar.slider("Select Time Window (Hour):", 0, 23, 18)

    df_view = df[df["hour"] == selected_hour].copy()
    if df_view.empty:
        st.warning("No records available for this hour in the sample.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Pickups", len(df_view), delta="Live")
    col2.metric("Avg Duration", f"{df_view['trip_duration'].mean() / 60:.1f} min")
    col3.metric("Avg Speed", f"{df_view['speed_kmh'].mean():.1f} km/h")

    if len(df_view) > 2500:
        status = "🔥 Critical Demand"
    elif len(df_view) > 1500:
        status = "🟠 High Demand"
    else:
        status = "🟢 Normal Operations"
    col4.metric("Status", status)

    if model:
        try:
            coords = df_view[["pickup_latitude", "pickup_longitude"]]
            df_view["cluster"] = model.predict(coords)

            st.subheader(f"AI-Optimized Driver Allocation Zones ({selected_hour}:00)")
            fig = px.scatter_mapbox(
                df_view,
                lat="pickup_latitude",
                lon="pickup_longitude",
                color="cluster",
                size="passenger_count",
                zoom=11,
                height=600,
                mapbox_style="carto-darkmatter",
                color_continuous_scale=px.colors.qualitative.Bold,
                title="Predicted Hotspots (Clusters)",
            )
            st.plotly_chart(fig, use_container_width=True, config=map_config)
            st.info(
                "Strategy: The AI has segmented the city into optimal zones. "
                "Reposition drivers toward clusters with higher live demand."
            )
        except Exception as exc:
            st.warning(f"Model prediction error: {exc}")
    else:
        st.subheader("Raw Demand Density")
        fig = px.density_mapbox(
            df_view,
            lat="pickup_latitude",
            lon="pickup_longitude",
            z="passenger_count",
            radius=10,
            zoom=11,
            mapbox_style="carto-darkmatter",
        )
        st.plotly_chart(fig, use_container_width=True, config=map_config)


def run_research_lab(df: pd.DataFrame, map_config: dict):
    st.subheader("🧪 Comparative Algorithm Analysis")
    st.markdown(
        """
**Objective:** Show density-based clustering can better filter outliers than centroid-only approaches for city demand topology.
"""
    )

    max_sample = max(1000, min(10000, len(df)))
    default_sample = min(5000, max_sample)
    sample_size = st.slider("Experiment Sample Size:", 1000, max_sample, default_sample)
    df_sample = df.sample(sample_size, random_state=42).copy()

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Method A: K-Means")
        k = st.slider("K (Centroids):", 3, 15, 8, key="k_slider")
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(
            df_sample[["pickup_latitude", "pickup_longitude"]]
        )
        df_sample["kmeans"] = km.labels_.astype(str)

        fig_k = px.scatter_mapbox(
            df_sample,
            lat="pickup_latitude",
            lon="pickup_longitude",
            color="kmeans",
            zoom=10,
            height=450,
            mapbox_style="carto-positron",
            title="Result: Spherical Zones",
        )
        st.plotly_chart(fig_k, use_container_width=True, config=map_config)
        st.error("Observation: K-Means forces outliers into clusters.")

    with col_right:
        st.markdown("#### Method B: HDBSCAN (Proposed)")
        try:
            import hdbscan  # type: ignore

            min_c = st.slider("Min Cluster Size:", 10, 100, 30, key="h_slider")
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_c)
            df_sample["hdbscan"] = clusterer.fit_predict(
                df_sample[["pickup_latitude", "pickup_longitude"]]
            )
            noise_count = int((df_sample["hdbscan"] == -1).sum())

            fig_h = px.scatter_mapbox(
                df_sample,
                lat="pickup_latitude",
                lon="pickup_longitude",
                color="hdbscan",
                zoom=10,
                height=450,
                mapbox_style="carto-positron",
                title="Result: High-Density Hotspots",
            )
            st.plotly_chart(fig_h, use_container_width=True, config=map_config)
            st.success(f"Observation: Detected {noise_count} noise points to ignore.")
        except ImportError:
            st.warning("HDBSCAN library is not installed.")


def run_efficiency_analysis(df: pd.DataFrame):
    st.subheader("🚦 Fleet Efficiency & Congestion Metrics")
    st.markdown("Analyzing trip duration and speed to identify low-efficiency time windows.")

    tab1, tab2 = st.tabs(["Velocity Profile", "Trip Durations"])

    with tab1:
        st.markdown("**Average Fleet Speed by Hour**")
        avg_speed = df.groupby("hour")["speed_kmh"].mean().reset_index()

        fig_speed = px.line(
            avg_speed,
            x="hour",
            y="speed_kmh",
            markers=True,
            line_shape="spline",
            color_discrete_sequence=["#00CC96"],
        )

        slowest_idx = int(avg_speed["speed_kmh"].idxmin())
        slowest_hour = int(avg_speed.loc[slowest_idx, "hour"])
        slowest_speed = float(avg_speed.loc[slowest_idx, "speed_kmh"])
        fig_speed.add_annotation(
            x=slowest_hour,
            y=slowest_speed,
            text=f"Slowest hour ({slowest_hour}:00)",
            showarrow=True,
            arrowhead=1,
        )

        st.plotly_chart(fig_speed, use_container_width=True)
        st.caption("Insight: This curve helps identify congestion-driven low-productivity periods.")

    with tab2:
        st.markdown("**Trip Duration Distribution**")
        fig_hist = px.histogram(
            df,
            x="trip_duration",
            nbins=100,
            range_x=[0, 3600],
            title="Frequency of Trip Lengths (Seconds)",
            color_discrete_sequence=["#AB63FA"],
        )
        st.plotly_chart(fig_hist, use_container_width=True)


def run_phase2_module(df_clustered: pd.DataFrame, kmeans_model, artifacts: dict):
    st.subheader("🤖 Phase 2: Predictive Intelligence")

    n_clusters = int(
        getattr(kmeans_model, "n_clusters", artifacts.get("forecast_meta", {}).get("n_clusters", 8))
    )
    hourly_demand = build_hourly_cluster_demand(df_clustered, n_clusters=n_clusters)
    forecast = infer_next_hour_demand(hourly_demand, artifacts)

    if forecast is None and not hourly_demand.empty:
        forecast = hourly_demand.tail(1).values[0]

    baseline = artifacts.get("baseline", {})
    if not baseline and not hourly_demand.empty:
        baseline = {str(i): float(hourly_demand[i].mean()) for i in range(n_clusters)}

    tabs = st.tabs([
        "📈 Demand Forecasting",
        "💸 Dynamic Pricing",
        "🕹️ Dispatch Simulation",
    ])

    with tabs[0]:
        st.markdown("One-hour-ahead cluster demand from the Phase 2 forecasting pipeline.")

        if forecast is None:
            st.warning(
                "Forecast model artifact not available yet. Run `python train_phase2_models.py` first."
            )
        else:
            cluster_labels = [f"Cluster {i}" for i in range(n_clusters)]
            pred_df = pd.DataFrame(
                {
                    "cluster": cluster_labels,
                    "predicted_demand": forecast,
                }
            ).sort_values("predicted_demand", ascending=False)
            pred_df["rank"] = np.arange(1, len(pred_df) + 1)

            col1, col2, col3 = st.columns(3)
            top_cluster = pred_df.iloc[0]
            col1.metric("Top Demand Cluster", top_cluster["cluster"])
            col2.metric("Predicted Trips (Top)", f"{top_cluster['predicted_demand']:.1f}")
            col3.metric("Total Predicted Demand", f"{pred_df['predicted_demand'].sum():.1f}")

            fig_pred = px.bar(
                pred_df,
                x="cluster",
                y="predicted_demand",
                color="predicted_demand",
                color_continuous_scale="Turbo",
                title="Next-Hour Demand Forecast by Cluster",
            )
            st.plotly_chart(fig_pred, use_container_width=True)

            recent = hourly_demand.tail(24)
            if not recent.empty:
                heat_df = recent.T
                heat_df.index = [f"Cluster {idx}" for idx in heat_df.index]
                fig_heat = px.imshow(
                    heat_df,
                    labels={"x": "Recent Hour", "y": "Cluster", "color": "Trips"},
                    x=[ts.strftime("%m-%d %H:%M") for ts in recent.index],
                    color_continuous_scale="Viridis",
                    aspect="auto",
                    title="Last 24 Hours: Historical Demand Matrix",
                )
                st.plotly_chart(fig_heat, use_container_width=True)

            st.markdown("**Top 3 Rebalancing Recommendations**")
            top_reco = pred_df.head(3)[["rank", "cluster", "predicted_demand"]]
            st.dataframe(top_reco, use_container_width=True, hide_index=True)

            st.markdown("**All Cluster Forecasts (not just top 3)**")
            all_forecasts = pred_df[["rank", "cluster", "predicted_demand"]]
            st.dataframe(all_forecasts, use_container_width=True, hide_index=True)

    with tabs[1]:
        st.markdown("Trip-duration regression + demand-ratio surge pricing simulation.")

        if artifacts.get("pricing_model") is None:
            st.warning("Pricing model artifact missing. Run `python train_phase2_models.py` first.")
        else:
            default_pick_lat = float(df_clustered["pickup_latitude"].median())
            default_pick_lon = float(df_clustered["pickup_longitude"].median())
            default_drop_lat = float(df_clustered["dropoff_latitude"].median())
            default_drop_lon = float(df_clustered["dropoff_longitude"].median())

            c1, c2 = st.columns(2)
            with c1:
                pickup_lat = st.number_input("Pickup Latitude", value=default_pick_lat, format="%.6f")
                pickup_lon = st.number_input("Pickup Longitude", value=default_pick_lon, format="%.6f")
                passenger_count = st.slider("Passenger Count", 1, 6, 1)
            with c2:
                dropoff_lat = st.number_input("Dropoff Latitude", value=default_drop_lat, format="%.6f")
                dropoff_lon = st.number_input("Dropoff Longitude", value=default_drop_lon, format="%.6f")
                selected_hour = st.slider("Trip Hour", 0, 23, 18, key="phase2_pricing_hour")

            speed_by_hour = df_clustered.groupby("hour")["speed_kmh"].mean().to_dict()
            assumed_speed = float(speed_by_hour.get(selected_hour, df_clustered["speed_kmh"].mean()))

            use_forecast_ratio = st.checkbox("Use forecast-based demand ratio", value=True)
            manual_ratio = st.slider("Manual Demand Ratio", 0.5, 3.0, 1.2, 0.1)

            if st.button("Estimate Duration + Surge Fare Proxy"):
                base_date = pd.Timestamp("2026-01-05")  # Monday anchor
                pickup_dt = base_date + pd.Timedelta(hours=int(selected_hour))

                features_df = build_single_pricing_features(
                    pickup_dt=pickup_dt,
                    pickup_longitude=pickup_lon,
                    pickup_latitude=pickup_lat,
                    dropoff_longitude=dropoff_lon,
                    dropoff_latitude=dropoff_lat,
                    passenger_count=passenger_count,
                    assumed_speed_kmh=assumed_speed,
                    weather_temp_c=24.0,
                )

                model = artifacts["pricing_model"]
                expected_columns = artifacts.get("pricing_features") or list(features_df.columns)
                features_used = features_df.reindex(columns=expected_columns)

                pred_duration = float(model.predict(features_used)[0])
                pred_duration = float(np.clip(pred_duration, 60.0, 7200.0))

                pickup_cluster = int(
                    kmeans_model.predict(np.array([[pickup_lat, pickup_lon]], dtype=float))[0]
                )

                demand_ratio = manual_ratio
                if use_forecast_ratio and forecast is not None:
                    base = float(baseline.get(str(pickup_cluster), 1.0))
                    demand_ratio = float(forecast[pickup_cluster] / max(base, 1.0))

                surge = surge_multiplier(demand_ratio=demand_ratio, base=1.0, alpha=0.8, max_surge=3.0)
                fare_proxy = (pred_duration / 60.0) * 1.8 * surge

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Predicted Duration", f"{pred_duration / 60:.1f} min")
                m2.metric("Demand Ratio", f"{demand_ratio:.2f}")
                m3.metric("Surge Multiplier", f"{surge:.2f}x")
                m4.metric("Fare Proxy", f"${fare_proxy:.2f}")

                st.caption(
                    f"Pickup mapped to Cluster {pickup_cluster}; pricing model type: {artifacts.get('pricing_model_type')}"
                )
                st.dataframe(features_used, use_container_width=True)

    with tabs[2]:
        st.markdown("Discrete-event simulation: random baseline vs AI-driven positioning.")

        requests = st.slider("Number of Simulated Requests", 500, 8000, 2000, step=250)
        seed = st.number_input("Random Seed", value=42, step=1)
        demand_profile = st.radio(
            "Demand Profile",
            ["Historical Distribution", "Forecast-Adjusted Distribution"],
        )

        cluster_counts = dict_from_cluster_series(df_clustered["cluster"], n_clusters=n_clusters)
        weights = np.array([cluster_counts[str(i)] for i in range(n_clusters)], dtype=float)
        weights = weights / max(weights.sum(), 1.0)

        if demand_profile == "Forecast-Adjusted Distribution" and forecast is not None:
            base_vec = np.array([float(baseline.get(str(i), 1.0)) for i in range(n_clusters)], dtype=float)
            ratio_vec = forecast / np.maximum(base_vec, 1.0)
            weights = weights * np.maximum(ratio_vec, 0.1)
            weights = weights / max(weights.sum(), 1e-9)

        result = run_dispatch_simulation(
            demand_weights=weights,
            num_requests=int(requests),
            seed=int(seed),
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Random Mean Wait", f"{result.random_mean_wait:.1f} s")
        c2.metric("AI Mean Wait", f"{result.ai_mean_wait:.1f} s")
        c3.metric("Mean Improvement", f"{result.mean_improvement_pct:.1f}%")
        c4.metric("Median Improvement", f"{result.median_improvement_pct:.1f}%")

        hist_df = pd.DataFrame(
            {
                "wait_time_sec": np.concatenate([result.random_wait_times, result.ai_wait_times]),
                "strategy": ["Random Baseline"] * len(result.random_wait_times)
                + ["AI-Driven"] * len(result.ai_wait_times),
            }
        )
        fig_hist = px.histogram(
            hist_df,
            x="wait_time_sec",
            color="strategy",
            barmode="overlay",
            nbins=30,
            opacity=0.7,
            title="Dispatch Simulation Wait-Time Distribution",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        weight_df = pd.DataFrame(
            {
                "cluster": [f"Cluster {i}" for i in range(n_clusters)],
                "weight": weights,
            }
        )
        fig_w = px.bar(weight_df, x="cluster", y="weight", title="Simulation Demand Weights")
        st.plotly_chart(fig_w, use_container_width=True)


def main():
    st.sidebar.title("🚖 Fleet Command")
    st.sidebar.caption("NSUT B.Tech Project | Phase 1 + Phase 2")
    st.sidebar.markdown("---")

    view_mode = st.sidebar.radio(
        "Select System Module:",
        [
            "📍 Live Operations",
            "🔬 Research Lab",
            "📉 Efficiency Analysis",
            "🤖 Phase 2 Intelligence",
        ],
    )

    with st.spinner("Initializing Fleet Systems..."):
        df = load_data("train_small.csv")
        kmeans_model, model_status = load_kmeans_model("kmeans_fleet_model.pkl")
        phase2_artifacts = load_phase2_artifacts("phase2_models")

    if df is None:
        st.error(
            "🚨 Critical Error: 'train_small.csv' not found. Please put the dataset in the project folder."
        )
        st.stop()

    if kmeans_model is not None:
        df_clustered = add_cluster_labels(df, kmeans_model)
    else:
        df_clustered = df.copy()
        df_clustered["cluster"] = -1

    st.markdown("### Intelligent Fleet Allocation System")
    st.markdown(
        f"**AI Engine:** `{model_status}` | "
        f"**Active Records:** `{len(df):,}` | "
        f"**Phase 2 Artifacts:** `{'Ready' if phase2_artifacts.get('ready') else 'Partial / Missing'}`"
    )

    if phase2_artifacts.get("errors"):
        with st.expander("Phase 2 Artifact Diagnostics"):
            for item in phase2_artifacts["errors"]:
                st.write(f"- {item}")
            st.code("python train_phase2_models.py")

    st.markdown("---")
    map_config = {"scrollZoom": True, "displayModeBar": True}

    if view_mode == "📍 Live Operations":
        run_live_operations(df, kmeans_model, map_config)
    elif view_mode == "🔬 Research Lab":
        run_research_lab(df, map_config)
    elif view_mode == "📉 Efficiency Analysis":
        run_efficiency_analysis(df)
    else:
        if kmeans_model is None:
            st.error("Phase 2 requires the pre-trained `kmeans_fleet_model.pkl` model file.")
        else:
            run_phase2_module(df_clustered, kmeans_model, phase2_artifacts)


if __name__ == "__main__":
    main()
