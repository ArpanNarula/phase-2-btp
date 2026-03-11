"""Offline trainer for Phase 2 models.

Generates:
- phase2_models/forecast_metadata.json
- phase2_models/lstm_demand_forecaster.keras OR .pkl (fallback)
- phase2_models/forecast_validation_metrics.json
- phase2_models/forecast_cluster_metrics.json
- phase2_models/demand_baseline.json
- phase2_models/xgb_pricing_model.json OR skl_pricing_model.pkl
- phase2_models/xgb_features.json
- phase2_models/pricing_feature_importance.json
- phase2_models/pricing_holdout_metrics.json
- phase2_models/training_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor

from phase2_core import (
    FORECAST_EXOGENOUS_COLUMNS,
    PRICING_FEATURE_COLUMNS,
    add_cluster_labels,
    build_hourly_cluster_demand,
    build_hourly_forecast_frame,
    build_pricing_feature_frame,
    load_and_prepare_trip_data,
    make_lstm_sequences,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phase 2 models offline.")
    parser.add_argument("--data", default="train_small.csv", help="Input CSV file.")
    parser.add_argument("--kmeans", default="kmeans_fleet_model.pkl", help="Phase 1 KMeans model path.")
    parser.add_argument("--models-dir", default="phase2_models", help="Directory to save artifacts.")
    parser.add_argument("--nrows", type=int, default=50000, help="Number of rows to load.")
    parser.add_argument("--lookback", type=int, default=24, help="LSTM lookback hours.")
    parser.add_argument("--epochs", type=int, default=20, help="Keras epochs when TensorFlow is available.")
    parser.add_argument("--batch-size", type=int, default=64, help="Keras batch size.")
    parser.add_argument(
        "--sparse-weight-power",
        type=float,
        default=1.0,
        help="Inverse-demand weighting power for sparse-hour emphasis in forecasting training.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="K-fold splits for pricing-model stability diagnostics (set <2 to disable).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--weather-temp",
        type=float,
        default=None,
        help="Optional fixed weather temperature override for pricing features.",
    )
    return parser.parse_args()


def _split_time_ordered(
    x: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split_idx = int(len(x) * train_frac)
    split_idx = min(max(split_idx, 1), len(x) - 1)
    return x[:split_idx], x[split_idx:], y[:split_idx], y[split_idx:]


def _build_sparse_hour_weights(
    y_samples: np.ndarray,
    power: float,
) -> np.ndarray:
    """Give larger weight to low-demand hours for imbalanced forecasting targets."""
    if len(y_samples) == 0:
        return np.asarray([], dtype=np.float32)

    total_hourly_demand = np.sum(y_samples, axis=1).astype(float)
    inv_density = 1.0 / np.power(np.maximum(total_hourly_demand, 1.0), max(power, 0.0))
    # Clamp extremes to avoid unstable training from very rare outliers.
    lo = float(np.quantile(inv_density, 0.05))
    hi = float(np.quantile(inv_density, 0.95))
    clipped = np.clip(inv_density, lo, hi)
    normalized = clipped / max(float(np.mean(clipped)), 1e-9)
    return normalized.astype(np.float32)


def _extract_epoch_validation_log(history) -> List[Dict[str, float]]:
    val_mse = [float(v) for v in history.history.get("val_loss", [])]
    val_mae = [float(v) for v in history.history.get("val_mae", [])]
    epoch_count = max(len(val_mse), len(val_mae))
    rows: List[Dict[str, float]] = []
    for idx in range(epoch_count):
        mse_val = val_mse[idx] if idx < len(val_mse) else (val_mse[-1] if val_mse else 0.0)
        mae_val = val_mae[idx] if idx < len(val_mae) else (val_mae[-1] if val_mae else 0.0)
        rows.append(
            {
                "epoch": idx + 1,
                "val_mse": float(mse_val),
                "val_rmse": float(np.sqrt(max(mse_val, 0.0))),
                "val_mae": float(mae_val),
            }
        )
    return rows


def _feature_importance_rows(
    feature_names: List[str],
    importances: np.ndarray,
) -> List[Dict[str, float]]:
    values = np.asarray(importances, dtype=float).reshape(-1)
    if len(values) != len(feature_names):
        return []
    denom = float(values.sum())
    normalized = values / denom if denom > 0 else values
    rows = [
        {
            "feature": feature_names[idx],
            "importance": float(values[idx]),
            "importance_normalized": float(normalized[idx]),
        }
        for idx in range(len(feature_names))
    ]
    rows.sort(key=lambda item: item["importance"], reverse=True)
    return rows


def _run_pricing_cross_validation(
    base_estimator,
    features: np.ndarray,
    target: np.ndarray,
    cv_folds: int,
    seed: int,
) -> Dict[str, Any]:
    if cv_folds < 2:
        return {"enabled": False, "reason": "cv_folds < 2"}
    n_splits = min(cv_folds, int(len(features)))
    if n_splits < 2:
        return {"enabled": False, "reason": "Insufficient samples for k-fold CV"}

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_metrics: List[Dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(features), start=1):
        est = clone(base_estimator)
        est.fit(features[train_idx], target[train_idx])
        pred = est.predict(features[val_idx])
        fold_metrics.append(
            {
                "fold": int(fold_idx),
                "mae": float(mean_absolute_error(target[val_idx], pred)),
                "rmse": float(np.sqrt(mean_squared_error(target[val_idx], pred))),
                "r2": float(r2_score(target[val_idx], pred)),
            }
        )

    mae_vals = np.asarray([m["mae"] for m in fold_metrics], dtype=float)
    rmse_vals = np.asarray([m["rmse"] for m in fold_metrics], dtype=float)
    r2_vals = np.asarray([m["r2"] for m in fold_metrics], dtype=float)
    return {
        "enabled": True,
        "n_splits": n_splits,
        "fold_metrics": fold_metrics,
        "mean": {
            "mae": float(mae_vals.mean()),
            "rmse": float(rmse_vals.mean()),
            "r2": float(r2_vals.mean()),
        },
        "std": {
            "mae": float(mae_vals.std()),
            "rmse": float(rmse_vals.std()),
            "r2": float(r2_vals.std()),
        },
    }


def _forecast_cluster_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_clusters: int,
) -> Dict[str, Any]:
    rows: List[Dict[str, float]] = []
    for cluster_id in range(n_clusters):
        c_true = y_true[:, cluster_id]
        c_pred = y_pred[:, cluster_id]
        rows.append(
            {
                "cluster": int(cluster_id),
                "mae": float(mean_absolute_error(c_true, c_pred)),
                "rmse": float(np.sqrt(mean_squared_error(c_true, c_pred))),
            }
        )
    rows.sort(key=lambda item: item["rmse"], reverse=True)
    return {
        "clusters": rows,
        "overall": {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        },
    }


def _pricing_holdout_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    abs_err = np.abs(y_true - y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "abs_error_p50": float(np.percentile(abs_err, 50)),
        "abs_error_p90": float(np.percentile(abs_err, 90)),
        "abs_error_p95": float(np.percentile(abs_err, 95)),
    }


def train_forecasting_model(
    x_seq: np.ndarray,
    y_seq: np.ndarray,
    models_dir: Path,
    lookback: int,
    n_clusters: int,
    input_feature_names: List[str],
    seed: int,
    epochs: int,
    batch_size: int,
    sparse_weight_power: float,
) -> Dict[str, Any]:
    x_train_full, x_test, y_train_full, y_test = _split_time_ordered(x_seq, y_seq)
    x_train, x_val, y_train, y_val = _split_time_ordered(x_train_full, y_train_full, train_frac=0.8)
    forecast_info: Dict[str, Any] = {
        "lookback_hours": lookback,
        "n_clusters": n_clusters,
        "cluster_columns": [str(i) for i in range(n_clusters)],
        "input_feature_count": int(x_seq.shape[-1]),
        "input_feature_names": input_feature_names,
        "exogenous_features": FORECAST_EXOGENOUS_COLUMNS,
    }
    validation_log_path = models_dir / "forecast_validation_metrics.json"
    per_cluster_metrics_path = models_dir / "forecast_cluster_metrics.json"

    train_sample_weights = _build_sparse_hour_weights(y_train, power=sparse_weight_power)
    forecast_info["sparse_hour_weighting"] = {
        "enabled": True,
        "power": float(sparse_weight_power),
        "min_weight": float(train_sample_weights.min()),
        "max_weight": float(train_sample_weights.max()),
        "mean_weight": float(train_sample_weights.mean()),
    }

    try:
        import tensorflow as tf  # type: ignore

        tf.keras.utils.set_random_seed(seed)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(lookback, x_seq.shape[-1])),
                tf.keras.layers.LSTM(32),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(n_clusters, activation="relu"),
            ]
        )
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        callbacks = [tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)]
        history = model.fit(
            x_train,
            y_train,
            sample_weight=train_sample_weights,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=callbacks,
        )
        epoch_log = _extract_epoch_validation_log(history)
        best_row = min(epoch_log, key=lambda row: row["val_rmse"]) if epoch_log else None

        pred = model.predict(x_test, verbose=0)
        pred = np.maximum(pred, 0.0)
        model.save(models_dir / "lstm_demand_forecaster.keras")

        forecast_info["model_type"] = "keras"
        forecast_info["validation_samples"] = int(len(x_val))
        if best_row:
            forecast_info["best_validation"] = {
                "epoch": int(best_row["epoch"]),
                "val_mae": float(best_row["val_mae"]),
                "val_rmse": float(best_row["val_rmse"]),
            }
        validation_payload = {
            "model_type": "keras",
            "epochs": epoch_log,
        }
        validation_log_path.write_text(json.dumps(validation_payload, indent=2), encoding="utf-8")
    except Exception as exc:
        # sklearn fallback keeps Phase 2 functional without TensorFlow.
        fallback_model = MultiOutputRegressor(GradientBoostingRegressor(random_state=seed))
        x_train_flat = x_train.reshape((x_train.shape[0], -1))
        x_val_flat = x_val.reshape((x_val.shape[0], -1))
        x_test_flat = x_test.reshape((x_test.shape[0], -1))
        fallback_model.fit(x_train_flat, y_train)
        val_pred = np.maximum(fallback_model.predict(x_val_flat), 0.0)
        pred = fallback_model.predict(x_test_flat)
        pred = np.maximum(pred, 0.0)
        joblib.dump(fallback_model, models_dir / "lstm_demand_forecaster.pkl")

        forecast_info["model_type"] = "sklearn_flat"
        forecast_info["fallback_reason"] = str(exc)
        forecast_info["validation_samples"] = int(len(x_val))
        forecast_info["best_validation"] = {
            "epoch": 1,
            "val_mae": float(mean_absolute_error(y_val, val_pred)),
            "val_rmse": float(np.sqrt(mean_squared_error(y_val, val_pred))),
        }
        validation_payload = {
            "model_type": "sklearn_flat",
            "note": "Per-epoch validation logging is unavailable for sklearn fallback.",
            "epochs": [
                {
                    "epoch": 1,
                    "val_mae": float(mean_absolute_error(y_val, val_pred)),
                    "val_rmse": float(np.sqrt(mean_squared_error(y_val, val_pred))),
                }
            ],
        }
        validation_log_path.write_text(json.dumps(validation_payload, indent=2), encoding="utf-8")

    forecast_cluster_metrics = _forecast_cluster_metrics(y_test, pred, n_clusters=n_clusters)
    per_cluster_metrics_path.write_text(
        json.dumps(forecast_cluster_metrics, indent=2),
        encoding="utf-8",
    )

    forecast_info["eval_mae"] = float(forecast_cluster_metrics["overall"]["mae"])
    forecast_info["eval_rmse"] = float(forecast_cluster_metrics["overall"]["rmse"])
    forecast_info["worst_cluster_rmse"] = float(
        forecast_cluster_metrics["clusters"][0]["rmse"]
        if forecast_cluster_metrics["clusters"]
        else 0.0
    )
    forecast_info["train_samples"] = int(len(x_train))
    forecast_info["test_samples"] = int(len(x_test))
    forecast_info["cluster_metrics_file"] = per_cluster_metrics_path.name
    return forecast_info


def train_pricing_model(
    features: np.ndarray,
    target: np.ndarray,
    models_dir: Path,
    seed: int,
    feature_names: List[str],
    cv_folds: int,
) -> Dict[str, Any]:
    x_train, x_test, y_train, y_test = _split_time_ordered(features, target)

    pricing_info: Dict[str, Any] = {}
    base_estimator = None
    try:
        from xgboost import XGBRegressor  # type: ignore

        base_estimator = XGBRegressor(
            n_estimators=350,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=4,
        )
        model = clone(base_estimator)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        model.save_model(str(models_dir / "xgb_pricing_model.json"))
        pricing_info["model_type"] = "xgboost"
    except Exception as exc:
        base_estimator = GradientBoostingRegressor(random_state=seed)
        model = clone(base_estimator)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        joblib.dump(model, models_dir / "skl_pricing_model.pkl")
        pricing_info["model_type"] = "sklearn_gbr"
        pricing_info["fallback_reason"] = str(exc)

    feature_importance_rows: List[Dict[str, float]] = []
    if hasattr(model, "feature_importances_"):
        feature_importance_rows = _feature_importance_rows(
            feature_names=feature_names,
            importances=np.asarray(model.feature_importances_),
        )

    importance_payload = {
        "model_type": pricing_info["model_type"],
        "feature_importance": feature_importance_rows,
    }
    (models_dir / "pricing_feature_importance.json").write_text(
        json.dumps(importance_payload, indent=2),
        encoding="utf-8",
    )
    pricing_info["top_features"] = feature_importance_rows[:5]

    cv_results = _run_pricing_cross_validation(
        base_estimator=base_estimator,
        features=features,
        target=target,
        cv_folds=cv_folds,
        seed=seed,
    )
    pricing_info["cross_validation"] = cv_results

    holdout_metrics = _pricing_holdout_metrics(y_test, pred)
    (models_dir / "pricing_holdout_metrics.json").write_text(
        json.dumps(holdout_metrics, indent=2),
        encoding="utf-8",
    )

    pricing_info["eval_mae"] = float(holdout_metrics["mae"])
    pricing_info["eval_rmse"] = float(holdout_metrics["rmse"])
    pricing_info["eval_r2"] = float(holdout_metrics["r2"])
    pricing_info["holdout_metrics_file"] = "pricing_holdout_metrics.json"
    pricing_info["train_samples"] = int(len(x_train))
    pricing_info["test_samples"] = int(len(x_test))
    return pricing_info


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Phase2] Loading data from: {args.data}")
    df = load_and_prepare_trip_data(args.data, nrows=args.nrows)
    print(f"[Phase2] Clean records: {len(df):,}")

    print(f"[Phase2] Loading KMeans model: {args.kmeans}")
    kmeans = joblib.load(args.kmeans)
    n_clusters = int(getattr(kmeans, "n_clusters", 8))
    print(f"[Phase2] Using clusters: {n_clusters}")

    df_clustered = add_cluster_labels(df, kmeans)

    print("[Phase2] Building hourly demand + exogenous forecasting inputs...")
    hourly = build_hourly_cluster_demand(df_clustered, n_clusters=n_clusters)
    hourly_forecast_features = build_hourly_forecast_frame(df_clustered, n_clusters=n_clusters)
    forecast_feature_names = [f"cluster_{i}" for i in range(n_clusters)] + FORECAST_EXOGENOUS_COLUMNS
    x_seq, y_seq = make_lstm_sequences(
        hourly_forecast_features,
        lookback_hours=args.lookback,
        target_cluster_count=n_clusters,
    )
    if len(x_seq) < 10:
        raise RuntimeError(
            "Not enough sequence samples for forecasting training. "
            "Try increasing --nrows or use a larger dataset."
        )

    baseline = hourly.mean(axis=0).to_dict()
    baseline_json = {str(int(k)): float(v) for k, v in baseline.items()}
    (models_dir / "demand_baseline.json").write_text(
        json.dumps(baseline_json, indent=2),
        encoding="utf-8",
    )

    print("[Phase2] Training forecasting model...")
    forecast_info = train_forecasting_model(
        x_seq=x_seq,
        y_seq=y_seq,
        models_dir=models_dir,
        lookback=args.lookback,
        n_clusters=n_clusters,
        input_feature_names=forecast_feature_names,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sparse_weight_power=args.sparse_weight_power,
    )
    (models_dir / "forecast_metadata.json").write_text(
        json.dumps(forecast_info, indent=2),
        encoding="utf-8",
    )

    print("[Phase2] Building pricing features...")
    pricing_df = build_pricing_feature_frame(df_clustered, weather_temp_c=args.weather_temp)
    x_price = pricing_df.values.astype(np.float32)
    y_price = df_clustered["trip_duration"].values.astype(np.float32)

    print("[Phase2] Training pricing model...")
    pricing_info = train_pricing_model(
        features=x_price,
        target=y_price,
        models_dir=models_dir,
        seed=args.seed,
        feature_names=PRICING_FEATURE_COLUMNS,
        cv_folds=args.cv_folds,
    )
    (models_dir / "xgb_features.json").write_text(
        json.dumps(PRICING_FEATURE_COLUMNS, indent=2),
        encoding="utf-8",
    )

    report = {
        "forecasting": forecast_info,
        "pricing": pricing_info,
        "forecast_exogenous_features": FORECAST_EXOGENOUS_COLUMNS,
        "nrows_used": int(args.nrows),
        "records_after_cleaning": int(len(df_clustered)),
    }
    (models_dir / "training_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("[Phase2] Training complete. Artifacts saved under:", models_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
