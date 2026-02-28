# Intelligent Fleet Allocation System (Phase 1 + Phase 2)

This project is a Streamlit-based **Fleet Command Center** for taxi demand intelligence.

## Phase 1
- Live hotspot visualization with pre-trained KMeans clusters.
- Research lab comparison: KMeans vs HDBSCAN.
- Fleet efficiency analytics: speed profile + trip-duration distribution.

## Phase 2
Implemented from your mid-sem specification:
- **Demand forecasting** (one-hour ahead, cluster-wise) using a 24-hour sliding window.
- **Dynamic pricing simulation** with trip-duration regression + demand-ratio surge multiplier.
- **Dispatch simulation** (discrete-event style) comparing random baseline vs AI-driven positioning.
- **Integrated dashboard** with low-latency inference from offline-trained artifacts.

## Project Files
- `app.py` -> Phase 1 + Phase 2 dashboard.
- `phase2_core.py` -> Shared pipeline/utilities for Phase 2.
- `train_phase2_models.py` -> Offline trainer for forecasting + pricing models.
- `phase2_models/` -> Generated model artifacts (created after training).

## Setup
```bash
pip install -r requirements.txt
```

Optional (recommended for full Phase 2 behavior):
```bash
pip install tensorflow xgboost
```

## Run Phase 2 Training (Offline)
This creates all artifacts used by the dashboard:
```bash
python train_phase2_models.py
```

Artifacts generated under `phase2_models/`:
- `forecast_metadata.json`
- `lstm_demand_forecaster.keras` OR `lstm_demand_forecaster.pkl` (fallback)
- `forecast_validation_metrics.json` (epoch-wise val MAE/RMSE log)
- `demand_baseline.json`
- `xgb_pricing_model.json` OR `skl_pricing_model.pkl` (fallback)
- `xgb_features.json`
- `pricing_feature_importance.json` (pricing driver importance ranking)
- `training_report.json`

Example with tuning flags:
```bash
python train_phase2_models.py --cv-folds 5 --sparse-weight-power 1.0
```

## Run Dashboard
```bash
streamlit run app.py
```

In the sidebar:
- `📍 Live Operations`
- `🔬 Research Lab`
- `📉 Efficiency Analysis`
- `🤖 Phase 2 Intelligence`

## Notes
- If TensorFlow is unavailable, forecasting training automatically falls back to a sklearn multi-output model.
- If XGBoost is unavailable, pricing training automatically falls back to sklearn GradientBoostingRegressor.
- Forecasting now upweights sparse-demand hours during training via inverse-demand sample weighting.
- Pricing training now logs k-fold cross-validation stability metrics into `training_report.json`.
- `kmeans_fleet_model.pkl` is required for both Phase 1 and Phase 2.
