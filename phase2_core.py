"""Core utilities for Phase 2 fleet intelligence modules.

This module keeps the data pipeline, pricing features, and simulator logic
separate from the Streamlit UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


PRICING_FEATURE_COLUMNS: List[str] = [
    "hour",
    "weekday",
    "hour_sin",
    "hour_cos",
    "distance_km",
    "passenger_count",
    "traffic_proxy",
    "weather_temp_c",
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


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Compute Haversine distance in kilometers."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return 6371.0 * c


def load_and_prepare_trip_data(
    csv_path: str,
    nrows: int | None = 50000,
) -> pd.DataFrame:
    """Load and clean the taxi dataset for Phase 1 + Phase 2 use."""
    cols = [
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "trip_duration",
    ]
    df = pd.read_csv(csv_path, usecols=cols, nrows=nrows)

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

    return df.reset_index(drop=True)


def add_cluster_labels(df: pd.DataFrame, kmeans_model) -> pd.DataFrame:
    """Assign pickup cluster labels using pre-trained KMeans model."""
    df_clustered = df.copy()
    coords = df_clustered[["pickup_latitude", "pickup_longitude"]]
    df_clustered["cluster"] = kmeans_model.predict(coords)
    return df_clustered


def build_hourly_cluster_demand(
    df_clustered: pd.DataFrame,
    n_clusters: int,
) -> pd.DataFrame:
    """Build an hourly demand matrix with one column per cluster."""
    hourly = (
        df_clustered.set_index("pickup_datetime")
        .groupby([pd.Grouper(freq="h"), "cluster"])
        .size()
        .unstack(fill_value=0)
    )

    # Ensure all cluster columns exist and ordered.
    for cluster_id in range(n_clusters):
        if cluster_id not in hourly.columns:
            hourly[cluster_id] = 0
    hourly = hourly.reindex(sorted(hourly.columns), axis=1)

    return hourly.astype(float)


def make_lstm_sequences(
    hourly_cluster_demand: pd.DataFrame,
    lookback_hours: int = 24,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding windows: X in R^(lookback x clusters), y in R^(clusters)."""
    values = hourly_cluster_demand.values
    if len(values) <= lookback_hours:
        return np.empty((0, lookback_hours, values.shape[1])), np.empty((0, values.shape[1]))

    x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    for idx in range(lookback_hours, len(values)):
        x_list.append(values[idx - lookback_hours : idx])
        y_list.append(values[idx])

    return np.asarray(x_list, dtype=np.float32), np.asarray(y_list, dtype=np.float32)


def build_pricing_feature_frame(
    df: pd.DataFrame,
    weather_temp_c: float = 24.0,
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

    feature_df["weather_temp_c"] = float(weather_temp_c)
    return feature_df[PRICING_FEATURE_COLUMNS]


def build_single_pricing_features(
    pickup_dt: pd.Timestamp,
    pickup_longitude: float,
    pickup_latitude: float,
    dropoff_longitude: float,
    dropoff_latitude: float,
    passenger_count: int,
    assumed_speed_kmh: float,
    weather_temp_c: float = 24.0,
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

    row = {
        "hour": hour,
        "weekday": weekday,
        "hour_sin": np.sin(2.0 * np.pi * hour / 24.0),
        "hour_cos": np.cos(2.0 * np.pi * hour / 24.0),
        "distance_km": distance_km,
        "passenger_count": float(passenger_count),
        "traffic_proxy": traffic_proxy,
        "weather_temp_c": float(weather_temp_c),
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


def run_dispatch_simulation(
    demand_weights: Iterable[float],
    num_requests: int = 2000,
    seed: int = 42,
) -> DispatchSimulationResult:
    """Compare random baseline vs AI-driven positioning with synthetic requests."""
    rng = np.random.default_rng(seed)
    weights = np.asarray(list(demand_weights), dtype=float)
    weights = np.clip(weights, a_min=0.0, a_max=None)
    if weights.sum() == 0:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / weights.sum()

    n_clusters = len(weights)
    request_clusters = rng.choice(np.arange(n_clusters), size=num_requests, p=weights)
    random_driver_clusters = rng.integers(0, n_clusters, size=num_requests)
    ai_driver_clusters = request_clusters

    # wait(i, j) = 120 + 60 * |i - j|
    random_wait = 120.0 + 60.0 * np.abs(random_driver_clusters - request_clusters)
    ai_wait = 120.0 + 60.0 * np.abs(ai_driver_clusters - request_clusters)

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
    )


def dict_from_cluster_series(series: pd.Series, n_clusters: int) -> Dict[str, float]:
    """Return a complete cluster-count dictionary with zero-filled missing keys."""
    counts = series.value_counts().to_dict()
    return {str(cluster_id): float(counts.get(cluster_id, 0.0)) for cluster_id in range(n_clusters)}
