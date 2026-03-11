# Intelligent Fleet Allocation System (Phase 1 + Phase 2)

Streamlit-based command center for fleet operations, demand forecasting, pricing, and dispatch simulation.

## Implemented Scope

### Phase 1
- Live hotspot view with pre-trained KMeans zones.
- Research comparison: KMeans vs HDBSCAN.
- Efficiency diagnostics: hourly speed profile and duration distribution.

### Phase 2
- **Demand forecasting (LSTM / fallback model)** with one-hour-ahead cluster prediction.
- **Validation MAE/RMSE logging** across training epochs (when Keras is used).
- **Forecast hold-out error by cluster** (`MAE`, `RMSE`) for comprehensive evaluation.
- **Dynamic pricing regression** with exogenous inputs (weather, holiday, event intensity).
- **Feature importance export** for pricing drivers.
- **K-fold cross-validation** for pricing model stability.
- **Imbalanced demand handling** by sparse-hour weighted forecasting loss.
- **Realistic dispatch simulation** with travel-time matrix calibration, driver availability, and service-time effects.
- **End-to-end rebalancing logic** (top-k deficit clusters + relocation recommendations), validated in simulation.
- **Professional dashboard UI overhaul** with clean information architecture.

## Key Files
- `app.py` - Streamlit dashboard (Phase 1 + Phase 2).
- `phase2_core.py` - data pipeline, exogenous feature logic, simulation + rebalancing engine.
- `train_phase2_models.py` - offline model training and metric export.
- `phase2_models/` - generated model artifacts.

## Setup
```bash
pip install -r requirements.txt
```

Optional (recommended for full capability):
```bash
pip install tensorflow xgboost
```

## Train Phase 2 Models
```bash
python train_phase2_models.py
```

Useful flags:
```bash
python train_phase2_models.py --cv-folds 5 --sparse-weight-power 1.2 --epochs 25
```

## Artifacts Generated (`phase2_models/`)
- `forecast_metadata.json`
- `forecast_validation_metrics.json`
- `forecast_cluster_metrics.json`
- `demand_baseline.json`
- `lstm_demand_forecaster.keras` or `lstm_demand_forecaster.pkl`
- `xgb_pricing_model.json` or `skl_pricing_model.pkl`
- `xgb_features.json`
- `pricing_feature_importance.json`
- `pricing_holdout_metrics.json`
- `training_report.json`

## Run Dashboard
```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Select repository + branch `main`.
4. Set **Main file path** = `app.py`.
5. Deploy.

If deployment shows old UI, click **Manage app -> Reboot app** after confirming latest commit is on `main`.
