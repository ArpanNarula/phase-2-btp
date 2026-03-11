"""Core utilities for Phase 2 fleet intelligence modules.

This module keeps the data pipeline, pricing features, forecasting inputs,
and simulation logic separate from the Streamlit UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    from pandas.tseries.holiday import USFederalHolidayCalendar
except Exception:  # pragma: no cover - fallback when pandas holiday module is unavailable
    USFederalHolidayCalendar = None


FORECAST_EXOGENOUS_COLUMNS: List[str] = [
    "weather_temp_c",
    "precipitation_mm",
    "is_holiday",
    "event_intensity",
    "is_weekend",
]

PRICING_FEATURE_COLUMNS: List[str] = [
    "hour",
    "weekday",
    "hour_sin",
    "hour_cos",
    "distance_km",
    "passenger_count",
    "traffic_proxy",
    "weather_temp_c",
    "precipitation_mm",
    "is_holiday",
    "event_intensity",
    "is_weekend",
]


@dataclass
class DispatchSimulationResult:
    random_wait_times: np.ndarray
    ai_wait_times: np.ndarray
    random_mean_wait: float
    random_median_wait: float
    ai_mean_wait: float
    ai_median_wait: float
    mean_improvement_pct: float
    median_improvement_pct: float
    random_p95_wait: float
    ai_p95_wait: float
    random_service_level_300: float
    ai_service_level_300: float
    random_driver_utilization: float
    ai_driver_utilization: float


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Compute Haversine distance in kilometers."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return 6371.0 * c


def normalize_weights(values: Iterable[float], floor: float = 1e-6) -> np.ndarray:
    """Convert any non-negative vector into a probability distribution."""
    arr = np.asarray(list(values), dtype=float)
    arr = np.clip(arr, a_min=0.0, a_max=None)
    if len(arr) == 0:
        return arr
    arr = np.maximum(arr, floor)
    denom = float(arr.sum())
    if denom <= 0:
        return np.ones_like(arr) / len(arr)
    return arr / denom


def derive_exogenous_signals(
    timestamps: pd.Series | pd.DatetimeIndex,
) -> pd.DataFrame:
    """Generate weather/holiday/event signals when external feeds are unavailable."""
    if isinstance(timestamps, pd.Series):
        ts = pd.to_datetime(timestamps, errors="coerce")
        out_index = timestamps.index
    else:
        ts = pd.Series(pd.to_datetime(timestamps, errors="coerce"), index=timestamps)
        out_index = timestamps

    ts = ts.fillna(pd.Timestamp("2016-01-01 00:00:00"))
    hour = ts.dt.hour.astype(int)
    weekday = ts.dt.dayofweek.astype(int)
    month = ts.dt.month.astype(int)
    day = ts.dt.day.astype(int)

    # Synthetic but deterministic proxies for weather + events.
    seasonal_temp = 16.0 + 11.0 * np.sin(2.0 * np.pi * (month - 1) / 12.0)
    diurnal_temp = 4.0 * np.sin(2.0 * np.pi * (hour - 7) / 24.0)
    weather_temp_c = seasonal_temp + diurnal_temp
    precipitation_mm = np.clip(
        3.2
        - 0.11 * weather_temp_c
        + 1.1 * np.cos(2.0 * np.pi * (month - 1) / 12.0),
        a_min=0.0,
        a_max=12.0,
    )

    if USFederalHolidayCalendar is not None:
        calendar = USFederalHolidayCalendar()
        start = ts.min().floor("D")
        end = ts.max().ceil("D")
        holiday_dates = set(calendar.holidays(start=start, end=end))
        is_holiday = ts.dt.normalize().isin(holiday_dates).astype(float)
    else:
        is_holiday = (
            ((month == 1) & (day == 1))
            | ((month == 7) & (day == 4))
            | ((month == 12) & (day == 25))
        ).astype(float)

    is_weekend = (weekday >= 5).astype(float)
    evening_peak = ((hour >= 17) & (hour <= 22)).astype(float)
    event_intensity = np.clip(
        0.20 + 0.55 * is_weekend + 0.40 * evening_peak + 0.45 * is_holiday,
        a_min=0.0,
        a_max=2.5,
    )

    exog = pd.DataFrame(
        {
            "weather_temp_c": weather_temp_c.astype(float),
            "precipitation_mm": precipitation_mm.astype(float),
            "is_holiday": is_holiday.astype(float),
            "event_intensity": event_intensity.astype(float),
            "is_weekend": is_weekend.astype(float),
        },
        index=out_index,
    )
    return exog


def load_and_prepare_trip_data(
    csv_path: str,
    nrows: int | None = 50000,
) -> pd.DataFrame:
    """Load and clean the taxi dataset for Phase 1 + Phase 2 use."""
    required_cols = [
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "trip_duration",
    ]
    optional_cols = [
        "weather_temp_c",
        "precipitation_mm",
        "is_holiday",
        "event_intensity",
        "is_weekend",
        "event_flag",
    ]
    allowed = set(required_cols + optional_cols)
    df = pd.read_csv(csv_path, usecols=lambda c: c in allowed, nrows=nrows)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df = df.dropna(subset=["pickup_datetime"]).copy()
    df["hour"] = df["pickup_datetime"].dt.hour
    df["weekday"] = df["pickup_datetime"].dt.dayofweek
    df["weekday_name"] = df["pickup_datetime"].dt.day_name()

    # NYC bounds filter to remove extreme GPS outliers.
    df = df[
        (df["pickup_latitude"] > 40.60)
        & (df["pickup_latitude"] < 40.90)
        & (df["pickup_longitude"] > -74.05)
        & (df["pickup_longitude"] < -73.70)
        & (df["dropoff_latitude"] > 40.60)
        & (df["dropoff_latitude"] < 40.90)
        & (df["dropoff_longitude"] > -74.05)
        & (df["dropoff_longitude"] < -73.70)
    ].copy()

    df["distance_km"] = df.apply(
        lambda row: haversine_km(
            row["pickup_longitude"],
            row["pickup_latitude"],
            row["dropoff_longitude"],
            row["dropoff_latitude"],
        ),
        axis=1,
    )

    df = df[df["trip_duration"] > 60].copy()
    df["speed_kmh"] = df["distance_km"] / (df["trip_duration"] / 3600.0)
    df = df[(df["speed_kmh"] > 1.0) & (df["speed_kmh"] < 120.0)].copy()

    derived_exog = derive_exogenous_signals(df["pickup_datetime"])
    if "event_flag" in df.columns and "event_intensity" not in df.columns:
        df["event_intensity"] = pd.to_numeric(df["event_flag"], errors="coerce")

    for col in FORECAST_EXOGENOUS_COLUMNS:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            df[col] = numeric.fillna(derived_exog[col])
        else:
            df[col] = derived_exog[col]

    df["is_holiday"] = df["is_holiday"].clip(0.0, 1.0)
    df["is_weekend"] = df["is_weekend"].clip(0.0, 1.0)
    df["event_intensity"] = df["event_intensity"].clip(0.0, 5.0)
    df["precipitation_mm"] = df["precipitation_mm"].clip(lower=0.0)
    return df.reset_index(drop=True)


def add_cluster_labels(df: pd.DataFrame, kmeans_model) -> pd.DataFrame:
    """Assign pickup cluster labels using pre-trained KMeans model."""
    df_clustered = df.copy()
    coords = df_clustered[["pickup_latitude", "pickup_longitude"]]
    df_clustered["cluster"] = kmeans_model.predict(coords)
    return df_clustered


def add_dropoff_cluster_labels(df: pd.DataFrame, kmeans_model) -> pd.DataFrame:
    """Assign dropoff cluster labels for OD travel-time calibration."""
    df_clustered = df.copy()
    coords = df_clustered[["dropoff_latitude", "dropoff_longitude"]]
    df_clustered["dropoff_cluster"] = kmeans_model.predict(coords)
    return df_clustered


def build_hourly_cluster_demand(
    df_clustered: pd.DataFrame,
    n_clusters: int,
) -> pd.DataFrame:
    """Build an hourly demand matrix with one column per cluster."""
    if df_clustered.empty:
        return pd.DataFrame(columns=list(range(n_clusters)), dtype=float)

    hourly = (
        df_clustered.set_index("pickup_datetime")
        .groupby([pd.Grouper(freq="h"), "cluster"])
        .size()
        .unstack(fill_value=0)
    )

    for cluster_id in range(n_clusters):
        if cluster_id not in hourly.columns:
            hourly[cluster_id] = 0
    hourly = hourly.reindex(sorted(hourly.columns), axis=1)

    full_idx = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
    hourly = hourly.reindex(full_idx, fill_value=0)
    return hourly.astype(float)


def build_hourly_exogenous_features(df_clustered: pd.DataFrame) -> pd.DataFrame:
    """Aggregate exogenous weather/holiday/event signals to hourly resolution."""
    if df_clustered.empty:
        return pd.DataFrame(columns=FORECAST_EXOGENOUS_COLUMNS, dtype=float)

    existing_cols = [c for c in FORECAST_EXOGENOUS_COLUMNS if c in df_clustered.columns]
    if existing_cols:
        hourly = (
            df_clustered.set_index("pickup_datetime")[existing_cols]
            .resample("h")
            .mean()
        )
    else:
        idx = pd.to_datetime(df_clustered["pickup_datetime"])
        hourly = pd.DataFrame(index=pd.date_range(idx.min().floor("h"), idx.max().ceil("h"), freq="h"))

    full_idx = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
    hourly = hourly.reindex(full_idx)
    derived = derive_exogenous_signals(hourly.index)
    for col in FORECAST_EXOGENOUS_COLUMNS:
        if col in hourly.columns:
            hourly[col] = pd.to_numeric(hourly[col], errors="coerce").fillna(derived[col])
        else:
            hourly[col] = derived[col]

    hourly["is_holiday"] = hourly["is_holiday"].clip(0.0, 1.0)
    hourly["is_weekend"] = hourly["is_weekend"].clip(0.0, 1.0)
    hourly["event_intensity"] = hourly["event_intensity"].clip(0.0, 5.0)
    hourly["precipitation_mm"] = hourly["precipitation_mm"].clip(lower=0.0)
    return hourly[FORECAST_EXOGENOUS_COLUMNS].astype(float)


def build_hourly_forecast_frame(
    df_clustered: pd.DataFrame,
    n_clusters: int,
) -> pd.DataFrame:
    """Combine cluster demand + exogenous hourly features for forecasting inputs."""
    hourly_demand = build_hourly_cluster_demand(df_clustered, n_clusters=n_clusters)
    if hourly_demand.empty:
        return hourly_demand
    hourly_exog = build_hourly_exogenous_features(df_clustered).reindex(hourly_demand.index)
    return pd.concat([hourly_demand, hourly_exog], axis=1).astype(float)


def make_lstm_sequences(
    hourly_feature_frame: pd.DataFrame,
    lookback_hours: int = 24,
    target_cluster_count: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding windows: X in R^(lookback x features), y in R^(target_clusters)."""
    values = hourly_feature_frame.values.astype(np.float32)
    feature_count = values.shape[1] if values.ndim == 2 else 0
    if target_cluster_count is None:
        target_dim = feature_count
    else:
        target_dim = int(max(1, min(target_cluster_count, feature_count)))

    if len(values) <= lookback_hours or feature_count == 0:
        return np.empty((0, lookback_hours, feature_count)), np.empty((0, target_dim))

    x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    for idx in range(lookback_hours, len(values)):
        x_list.append(values[idx - lookback_hours : idx])
        y_list.append(values[idx, :target_dim])

    return np.asarray(x_list, dtype=np.float32), np.asarray(y_list, dtype=np.float32)


def build_pricing_feature_frame(
    df: pd.DataFrame,
    weather_temp_c: float | None = None,
) -> pd.DataFrame:
    """Construct the feature matrix used by the pricing regressor."""
    feature_df = pd.DataFrame(index=df.index)
    feature_df["hour"] = df["hour"].astype(int)
    feature_df["weekday"] = df["weekday"].astype(int)
    feature_df["hour_sin"] = np.sin(2.0 * np.pi * feature_df["hour"] / 24.0)
    feature_df["hour_cos"] = np.cos(2.0 * np.pi * feature_df["hour"] / 24.0)
    feature_df["distance_km"] = df["distance_km"].astype(float)
    feature_df["passenger_count"] = df["passenger_count"].fillna(1).astype(float)

    speed = df["speed_kmh"].clip(lower=5.0, upper=80.0)
    normalized_speed = speed / 80.0
    feature_df["traffic_proxy"] = 1.0 / normalized_speed

    derived = derive_exogenous_signals(df["pickup_datetime"])
    if weather_temp_c is None:
        feature_df["weather_temp_c"] = (
            pd.to_numeric(df.get("weather_temp_c"), errors="coerce")
            if "weather_temp_c" in df.columns
            else pd.Series(np.nan, index=df.index)
        ).fillna(derived["weather_temp_c"])
    else:
        feature_df["weather_temp_c"] = float(weather_temp_c)

    feature_df["precipitation_mm"] = (
        pd.to_numeric(df.get("precipitation_mm"), errors="coerce")
        if "precipitation_mm" in df.columns
        else pd.Series(np.nan, index=df.index)
    ).fillna(derived["precipitation_mm"])
    feature_df["is_holiday"] = (
        pd.to_numeric(df.get("is_holiday"), errors="coerce")
        if "is_holiday" in df.columns
        else pd.Series(np.nan, index=df.index)
    ).fillna(derived["is_holiday"])
    feature_df["event_intensity"] = (
        pd.to_numeric(df.get("event_intensity"), errors="coerce")
        if "event_intensity" in df.columns
        else pd.Series(np.nan, index=df.index)
    ).fillna(derived["event_intensity"])
    feature_df["is_weekend"] = (
        pd.to_numeric(df.get("is_weekend"), errors="coerce")
        if "is_weekend" in df.columns
        else pd.Series(np.nan, index=df.index)
    ).fillna(derived["is_weekend"])

    feature_df["is_holiday"] = feature_df["is_holiday"].clip(0.0, 1.0)
    feature_df["is_weekend"] = feature_df["is_weekend"].clip(0.0, 1.0)
    feature_df["event_intensity"] = feature_df["event_intensity"].clip(0.0, 5.0)
    feature_df["precipitation_mm"] = feature_df["precipitation_mm"].clip(lower=0.0)
    return feature_df[PRICING_FEATURE_COLUMNS]


def build_single_pricing_features(
    pickup_dt: pd.Timestamp,
    pickup_longitude: float,
    pickup_latitude: float,
    dropoff_longitude: float,
    dropoff_latitude: float,
    passenger_count: int,
    assumed_speed_kmh: float,
    weather_temp_c: float | None = None,
    precipitation_mm: float | None = None,
    is_holiday: float | None = None,
    event_intensity: float | None = None,
) -> pd.DataFrame:
    """Build one-row pricing features for dashboard inference."""
    hour = int(pickup_dt.hour)
    weekday = int(pickup_dt.dayofweek)
    distance_km = haversine_km(
        pickup_longitude,
        pickup_latitude,
        dropoff_longitude,
        dropoff_latitude,
    )
    speed_kmh = float(np.clip(assumed_speed_kmh, 5.0, 80.0))
    traffic_proxy = 1.0 / (speed_kmh / 80.0)

    derived = derive_exogenous_signals(pd.Series([pickup_dt]))
    row = {
        "hour": hour,
        "weekday": weekday,
        "hour_sin": np.sin(2.0 * np.pi * hour / 24.0),
        "hour_cos": np.cos(2.0 * np.pi * hour / 24.0),
        "distance_km": distance_km,
        "passenger_count": float(passenger_count),
        "traffic_proxy": traffic_proxy,
        "weather_temp_c": float(derived.iloc[0]["weather_temp_c"] if weather_temp_c is None else weather_temp_c),
        "precipitation_mm": float(
            derived.iloc[0]["precipitation_mm"] if precipitation_mm is None else precipitation_mm
        ),
        "is_holiday": float(derived.iloc[0]["is_holiday"] if is_holiday is None else is_holiday),
        "event_intensity": float(
            derived.iloc[0]["event_intensity"] if event_intensity is None else event_intensity
        ),
        "is_weekend": float(derived.iloc[0]["is_weekend"]),
    }
    return pd.DataFrame([row], columns=PRICING_FEATURE_COLUMNS)


def surge_multiplier(
    demand_ratio: float,
    base: float = 1.0,
    alpha: float = 0.8,
    max_surge: float = 3.0,
) -> float:
    """Compute demand-based surge multiplier."""
    surge = base + alpha * max(0.0, demand_ratio - 1.0)
    return float(min(max_surge, surge))


def estimate_intercluster_travel_time_matrix(
    df_clustered: pd.DataFrame,
    n_clusters: int,
    min_pair_samples: int = 20,
) -> np.ndarray:
    """Estimate inter-cluster travel-time matrix from historical OD statistics."""
    matrix = np.full((n_clusters, n_clusters), np.nan, dtype=float)

    if "dropoff_cluster" in df_clustered.columns:
        pair_stats = (
            df_clustered.groupby(["cluster", "dropoff_cluster"])["trip_duration"]
            .agg(["median", "count"])
            .reset_index()
        )
        for _, row in pair_stats.iterrows():
            i = int(row["cluster"])
            j = int(row["dropoff_cluster"])
            if int(row["count"]) >= min_pair_samples:
                matrix[i, j] = float(row["median"])

    centroid_df = (
        df_clustered.groupby("cluster")[["pickup_latitude", "pickup_longitude"]]
        .mean()
        .reindex(range(n_clusters))
    )
    overall_lat = float(df_clustered["pickup_latitude"].median())
    overall_lon = float(df_clustered["pickup_longitude"].median())
    centroid_df["pickup_latitude"] = centroid_df["pickup_latitude"].fillna(overall_lat)
    centroid_df["pickup_longitude"] = centroid_df["pickup_longitude"].fillna(overall_lon)

    fallback_speed = float(np.clip(df_clustered["speed_kmh"].median(), 10.0, 45.0))
    for i in range(n_clusters):
        for j in range(n_clusters):
            if not np.isnan(matrix[i, j]):
                continue
            lat1 = float(centroid_df.iloc[i]["pickup_latitude"])
            lon1 = float(centroid_df.iloc[i]["pickup_longitude"])
            lat2 = float(centroid_df.iloc[j]["pickup_latitude"])
            lon2 = float(centroid_df.iloc[j]["pickup_longitude"])
            dist_km = haversine_km(lon1, lat1, lon2, lat2)
            fallback_sec = 70.0 + (dist_km / fallback_speed) * 3600.0
            if i == j:
                fallback_sec = min(fallback_sec, 120.0)
            matrix[i, j] = fallback_sec

    matrix = np.clip(matrix, a_min=45.0, a_max=3600.0)
    return matrix


def estimate_dropoff_transition_matrix(
    df_clustered: pd.DataFrame,
    n_clusters: int,
    smoothing: float = 0.2,
) -> np.ndarray:
    """Estimate probability of ending trip in each dropoff cluster."""
    trans = np.full((n_clusters, n_clusters), float(max(smoothing, 1e-6)), dtype=float)

    if "dropoff_cluster" in df_clustered.columns:
        counts = (
            df_clustered.groupby(["cluster", "dropoff_cluster"])
            .size()
            .reset_index(name="count")
        )
        for _, row in counts.iterrows():
            i = int(row["cluster"])
            j = int(row["dropoff_cluster"])
            trans[i, j] += float(row["count"])
    else:
        base = normalize_weights(np.ones(n_clusters))
        return np.tile(base, (n_clusters, 1))

    trans = trans / np.maximum(trans.sum(axis=1, keepdims=True), 1e-9)
    return trans


def recommend_rebalancing_actions(
    predicted_demand: Iterable[float],
    current_supply: Iterable[float],
    top_k: int = 3,
) -> Dict[str, object]:
    """Convert forecast into actionable top-k cluster rebalancing recommendations."""
    demand = np.asarray(list(predicted_demand), dtype=float)
    supply = np.asarray(list(current_supply), dtype=float)
    if len(demand) != len(supply):
        raise ValueError("predicted_demand and current_supply must have equal length.")
    if len(demand) == 0:
        return {
            "ranking": [],
            "top_clusters": [],
            "actions": [],
            "target_supply": [],
            "target_weights": [],
            "total_relocations": 0,
        }

    demand = np.clip(demand, a_min=0.0, a_max=None)
    supply = np.clip(supply, a_min=0.0, a_max=None)
    total_supply = float(max(supply.sum(), 1.0))
    target_weights = normalize_weights(demand)
    target_supply = target_weights * total_supply
    gap = target_supply - supply

    ranking = [
        {
            "cluster": int(i),
            "predicted_demand": float(demand[i]),
            "current_supply": float(supply[i]),
            "target_supply": float(target_supply[i]),
            "supply_gap": float(gap[i]),
        }
        for i in range(len(demand))
    ]
    ranking.sort(key=lambda item: item["supply_gap"], reverse=True)
    top_clusters = ranking[: max(1, min(top_k, len(ranking)))]

    deficit = gap.copy()
    actions: List[Dict[str, float | int]] = []
    deficit_clusters = list(np.argsort(-deficit))
    surplus_clusters = list(np.argsort(deficit))
    for to_cluster in deficit_clusters:
        need = int(np.ceil(max(deficit[to_cluster], 0.0)))
        if need <= 0:
            continue
        for from_cluster in surplus_clusters:
            if from_cluster == to_cluster:
                continue
            available = int(np.floor(max(-deficit[from_cluster], 0.0)))
            if available <= 0:
                continue
            moved = min(need, available)
            if moved <= 0:
                continue
            actions.append(
                {
                    "from_cluster": int(from_cluster),
                    "to_cluster": int(to_cluster),
                    "drivers_to_move": int(moved),
                }
            )
            deficit[from_cluster] += moved
            deficit[to_cluster] -= moved
            need -= moved
            if need <= 0:
                break

    return {
        "ranking": ranking,
        "top_clusters": top_clusters,
        "actions": actions,
        "target_supply": [float(x) for x in target_supply],
        "target_weights": [float(x) for x in target_weights],
        "total_relocations": int(sum(int(a["drivers_to_move"]) for a in actions)),
    }


def _default_travel_time_matrix(n_clusters: int) -> np.ndarray:
    ids = np.arange(n_clusters, dtype=float)
    return 90.0 + 55.0 * np.abs(ids[:, None] - ids[None, :])


def _simulate_dispatch_strategy(
    strategy: str,
    request_clusters: np.ndarray,
    interarrival_seconds: np.ndarray,
    travel_time_matrix: np.ndarray,
    transition_matrix: np.ndarray,
    initial_supply_weights: np.ndarray,
    dispatch_weights: np.ndarray,
    num_drivers: int,
    service_time_mean_sec: float,
    rebalance_every_n_requests: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    n_clusters = len(dispatch_weights)
    driver_cluster = rng.choice(np.arange(n_clusters), size=num_drivers, p=initial_supply_weights)
    driver_available_at = np.zeros(num_drivers, dtype=float)
    driver_busy_time = np.zeros(num_drivers, dtype=float)
    wait_times = np.zeros(len(request_clusters), dtype=float)

    current_time = 0.0
    for idx, req_cluster in enumerate(request_clusters):
        current_time += float(interarrival_seconds[idx])

        if (
            strategy == "ai"
            and rebalance_every_n_requests > 0
            and idx > 0
            and idx % rebalance_every_n_requests == 0
        ):
            idle_driver_idx = np.where(driver_available_at <= current_time)[0]
            if len(idle_driver_idx) > 0:
                for drv in idle_driver_idx:
                    if rng.random() < 0.30:
                        target_cluster = int(rng.choice(np.arange(n_clusters), p=dispatch_weights))
                        source_cluster = int(driver_cluster[drv])
                        if source_cluster != target_cluster:
                            reposition_time = 0.70 * float(
                                travel_time_matrix[source_cluster, target_cluster]
                            )
                            driver_available_at[drv] = current_time + reposition_time
                            driver_busy_time[drv] += reposition_time
                            driver_cluster[drv] = target_cluster

        if strategy == "random":
            chosen_driver = int(rng.integers(0, num_drivers))
        else:
            eta = np.maximum(driver_available_at - current_time, 0.0) + travel_time_matrix[
                driver_cluster, req_cluster
            ]
            chosen_driver = int(np.argmin(eta))

        pre_wait = float(max(driver_available_at[chosen_driver] - current_time, 0.0))
        pickup_travel = float(travel_time_matrix[int(driver_cluster[chosen_driver]), int(req_cluster)])
        wait = pre_wait + pickup_travel

        drop_cluster = int(rng.choice(np.arange(n_clusters), p=transition_matrix[int(req_cluster)]))
        in_service_drive = float(travel_time_matrix[int(req_cluster), int(drop_cluster)])
        stochastic_service = float(
            max(60.0, rng.gamma(shape=2.0, scale=max(service_time_mean_sec, 120.0) / 2.0))
        )
        service_time = max(in_service_drive, stochastic_service)
        total_busy = wait + service_time

        driver_available_at[chosen_driver] = current_time + total_busy
        driver_busy_time[chosen_driver] += total_busy
        driver_cluster[chosen_driver] = drop_cluster
        wait_times[idx] = wait

    horizon = max(current_time + service_time_mean_sec, 1.0)
    utilization = float(np.clip(driver_busy_time.sum() / (num_drivers * horizon), 0.0, 1.5))
    return wait_times, utilization


def run_dispatch_simulation(
    demand_weights: Iterable[float],
    num_requests: int = 2000,
    seed: int = 42,
    travel_time_matrix: np.ndarray | None = None,
    transition_matrix: np.ndarray | None = None,
    num_drivers: int = 450,
    service_time_mean_sec: float = 600.0,
    ai_supply_weights: Iterable[float] | None = None,
    rebalance_every_n_requests: int = 80,
) -> DispatchSimulationResult:
    """Compare random baseline vs AI-guided dispatch with realistic constraints."""
    rng = np.random.default_rng(seed)
    request_weights = normalize_weights(demand_weights)
    if len(request_weights) == 0:
        request_weights = np.ones(1, dtype=float)

    n_clusters = len(request_weights)
    ai_weights = normalize_weights(ai_supply_weights if ai_supply_weights is not None else request_weights)
    if len(ai_weights) != n_clusters:
        ai_weights = request_weights.copy()

    if travel_time_matrix is None:
        travel_time_matrix = _default_travel_time_matrix(n_clusters)
    if transition_matrix is None:
        transition_matrix = np.tile(request_weights, (n_clusters, 1))
    if transition_matrix.shape != (n_clusters, n_clusters):
        transition_matrix = np.tile(request_weights, (n_clusters, 1))

    request_clusters = rng.choice(np.arange(n_clusters), size=int(num_requests), p=request_weights)
    avg_gap = 3600.0 / max(float(num_requests), 1.0)
    interarrival_seconds = rng.exponential(scale=avg_gap, size=int(num_requests))

    random_wait, random_util = _simulate_dispatch_strategy(
        strategy="random",
        request_clusters=request_clusters,
        interarrival_seconds=interarrival_seconds,
        travel_time_matrix=travel_time_matrix,
        transition_matrix=transition_matrix,
        initial_supply_weights=normalize_weights(np.ones(n_clusters)),
        dispatch_weights=normalize_weights(np.ones(n_clusters)),
        num_drivers=int(max(num_drivers, 20)),
        service_time_mean_sec=float(max(service_time_mean_sec, 120.0)),
        rebalance_every_n_requests=0,
        rng=rng,
    )
    ai_wait, ai_util = _simulate_dispatch_strategy(
        strategy="ai",
        request_clusters=request_clusters,
        interarrival_seconds=interarrival_seconds,
        travel_time_matrix=travel_time_matrix,
        transition_matrix=transition_matrix,
        initial_supply_weights=ai_weights,
        dispatch_weights=ai_weights,
        num_drivers=int(max(num_drivers, 20)),
        service_time_mean_sec=float(max(service_time_mean_sec, 120.0)),
        rebalance_every_n_requests=int(max(rebalance_every_n_requests, 0)),
        rng=rng,
    )

    random_mean = float(np.mean(random_wait))
    random_median = float(np.median(random_wait))
    ai_mean = float(np.mean(ai_wait))
    ai_median = float(np.median(ai_wait))
    mean_improvement = ((random_mean - ai_mean) / max(random_mean, 1e-9)) * 100.0
    median_improvement = ((random_median - ai_median) / max(random_median, 1e-9)) * 100.0

    return DispatchSimulationResult(
        random_wait_times=random_wait,
        ai_wait_times=ai_wait,
        random_mean_wait=random_mean,
        random_median_wait=random_median,
        ai_mean_wait=ai_mean,
        ai_median_wait=ai_median,
        mean_improvement_pct=float(mean_improvement),
        median_improvement_pct=float(median_improvement),
        random_p95_wait=float(np.percentile(random_wait, 95)),
        ai_p95_wait=float(np.percentile(ai_wait, 95)),
        random_service_level_300=float(np.mean(random_wait <= 300.0) * 100.0),
        ai_service_level_300=float(np.mean(ai_wait <= 300.0) * 100.0),
        random_driver_utilization=float(random_util * 100.0),
        ai_driver_utilization=float(ai_util * 100.0),
    )


def dict_from_cluster_series(series: pd.Series, n_clusters: int) -> Dict[str, float]:
    """Return a complete cluster-count dictionary with zero-filled missing keys."""
    counts = series.value_counts().to_dict()
    return {str(cluster_id): float(counts.get(cluster_id, 0.0)) for cluster_id in range(n_clusters)}
