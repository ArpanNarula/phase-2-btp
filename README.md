# Intelligent Fleet Allocation System (Phase 1 + Phase 2)

## Group Project Details
- **Institution / Program:** NSUT CSIOT Branch 2026
- **Team Members:**
1. Arpan Narula (ROLL NO - 2022UCI8004)
2. Sania Gupta (ROLL NO - 2022UCI8026)
3. Ravi Pandey (ROLL NO - 2022UCI8068)

## Project Repository
- **GitHub:** [https://github.com/ArpanNarula/phase-2-btp](https://github.com/ArpanNarula/phase-2-btp)

## Project Overview
This is our intelligent fleet allocation and decision-support system built as a two-phase project. The dashboard integrates live operations monitoring, demand forecasting, pricing intelligence, and dispatch simulation.

## Implemented Scope

### Phase 1
- Live hotspot visualization with pre-trained KMeans clusters.
- Research comparison module: KMeans vs HDBSCAN.
- Fleet efficiency diagnostics: speed profile and trip-duration analysis.

### Phase 2
- One-hour-ahead cluster-level demand forecasting (LSTM pipeline with fallback support).
- Validation MAE/RMSE tracking for forecasting training.
- Forecast hold-out metrics at cluster level (MAE/RMSE per cluster).
- Dynamic pricing regression with exogenous signals (weather, holiday, event intensity).
- Feature importance export for pricing-driver analysis.
- K-fold cross-validation for pricing model stability checks.
- Imbalanced-demand handling via sparse-hour weighted forecasting loss.
- Dispatch simulation with calibrated inter-cluster travel time, driver availability, and service-time effects.
- End-to-end rebalancing logic (top-k deficit clusters + suggested relocations).

## Key Files
- `app.py` - Main Streamlit dashboard (Phase 1 + Phase 2 modules).
- `phase2_core.py` - Shared Phase 2 pipeline, features, simulation, and recommendation logic.
- `train_phase2_models.py` - Offline training script for forecasting and pricing models.
- `phase2_models/` - Saved artifacts (models, metrics, and reports).

## Setup
```bash
pip install -r requirements.txt
```

Optional (for full model stack):
```bash
pip install tensorflow xgboost
```

## Train Phase 2 Models
```bash
python train_phase2_models.py
```

Example with tuning flags:
```bash
python train_phase2_models.py --cv-folds 5 --sparse-weight-power 1.2 --epochs 25
```

## Generated Artifacts (`phase2_models/`)
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

## Deployment (Streamlit Community Cloud)
1. Push latest code to GitHub (`main` branch).
2. Open [https://share.streamlit.io](https://share.streamlit.io/).
3. Select repository: `ArpanNarula/phase-2-btp`.
4. Set main file path: `app.py`.
5. Deploy/Reboot app.
