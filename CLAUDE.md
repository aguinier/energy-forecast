# CLAUDE.md

This file provides guidance to Claude Code when working with the energy forecasting module.

## Module Overview

D+2 energy forecasting module for European electricity markets. Generates 24-hour forecasts for the day after tomorrow, running daily at 18:00.

**Forecast Types:**
- **Load** - Electricity demand (MW)
- **Price** - Day-ahead prices (EUR/MWh)
- **Renewable** - Total renewable generation (MW)
- **Individual Renewable Types:**
  - Solar - Solar PV generation (MW)
  - Wind Onshore - Onshore wind generation (MW)
  - Wind Offshore - Offshore wind generation (MW)
  - Hydro Total - Combined run-of-river and reservoir hydro (MW)
  - Biomass - Biomass generation (MW)
- **Net Position** - Cross-border import/export balance (MW) [Chronos-2 only]

**Coverage:** 24 European countries with complete data

## Architecture

```
energy_forecast/
├── config.py           # Configuration (paths, countries, model params)
├── requirements.txt    # Python dependencies
├── src/
│   ├── db.py               # Database operations
│   ├── features.py         # Feature engineering (incl. holiday features)
│   ├── metrics.py          # Evaluation metrics
│   ├── forecaster.py       # Forecaster class (XGBoost/LightGBM/CatBoost)
│   ├── hyperopt.py         # Optuna Bayesian hyperparameter optimization
│   ├── feature_selection.py # Automated feature selection
│   ├── validation.py       # Walk-forward validation
│   ├── baselines.py        # Baseline models (persistence, seasonal naive)
│   ├── model_registry.py   # Model versioning and registry
│   ├── deployment.py       # Model deployment management
│   └── chronos2/           # Chronos-2 foundation model (ported from netpredict2)
│       ├── engine.py           # Chronos-2 pipeline wrapper (forecast, batch)
│       ├── input_builder.py    # DB loading + covariate alignment
│       ├── finetuner.py        # Fine-tuning pipeline (5000 steps, cosine LR)
│       └── covariate_mapper.py # Country→covariate mapping (ENTSO-E + weather)
├── scripts/
│   ├── train.py              # Training script (enhanced)
│   ├── train_chronos2.py     # Chronos-2 fine-tuning script
│   ├── forecast_daily.py     # Daily forecast job
│   ├── forecast_chronos2.py  # Chronos-2 forecast generation
│   ├── compare_experiments.py # Cross-experiment backtest comparison
│   └── scheduler_setup.sh    # Cron setup
├── experiments/        # Versioned experiment configs (V001-Vnnn)
│   ├── registry.json       # Master experiment index
│   └── V00N/config.json    # Per-experiment configuration
├── models/             # Saved model artifacts
└── logs/               # Execution logs
```

## Database

Uses the shared database at `../data_gathering/energy_dashboard.db`.

**New Table:** `forecasts`
```sql
CREATE TABLE forecasts (
    id INTEGER PRIMARY KEY,
    country_code TEXT NOT NULL,
    forecast_type TEXT NOT NULL,      -- 'load', 'price', 'renewable', or individual types
    renewable_type TEXT,              -- For individual renewable types (solar, wind_onshore, etc.)
    target_timestamp_utc TIMESTAMP,   -- When forecast is FOR
    generated_at TIMESTAMP,           -- When forecast was MADE
    horizon_hours INTEGER,            -- Hours ahead (30-54 for D+2)
    forecast_value REAL,
    model_name TEXT,                  -- 'xgboost'
    model_version TEXT
);
```

## Model Storage

Models are stored in a filesystem-based structure with embedded metadata:

```
models/
├── {country_code}/
│   ├── {forecast_type}/
│   │   └── model.joblib    # XGBoost model + metadata
```

**Example:**
```
models/
├── DE/
│   ├── load/model.joblib
│   ├── solar/model.joblib
│   └── wind_onshore/model.joblib
└── FR/
    ├── load/model.joblib
    ├── price/model.joblib
    └── renewable/model.joblib
```

**Metadata Structure:**
Each `.joblib` file contains a Python dictionary with:
- `model`: Trained XGBRegressor instance
- `feature_columns`: List of feature names used during training
- `country_code`: ISO 2-letter country code
- `forecast_type`: Type of forecast (load, price, renewable, solar, etc.)
- `model_version`: Timestamp of training (YYYYMMDD_HHMMSS)
- `training_metrics`: Dict with MAE, MAPE, RMSE, SMAPE
- `saved_at`: ISO timestamp of when model was saved

**Key Points:**
- No separate JSON metadata file - all metadata embedded in joblib
- Only latest model version kept per country/type (no historical versions)
- Models discovered via filesystem traversal
- Typical model size: 2-6 MB depending on number of features

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train models for all countries
python scripts/train.py --countries all --types all

# Generate D+2 forecasts
python scripts/forecast_daily.py

# Setup daily cron job (18:00)
bash scripts/scheduler_setup.sh
```

## Key Commands

### Training

```bash
# Train all models (includes load, price, renewable, and individual renewable types)
python scripts/train.py --countries all --types all

# Train specific country/type
python scripts/train.py --countries DE --types load

# Train individual renewable types for a country
python scripts/train.py --countries FR --types solar,wind_onshore,wind_offshore,hydro_total,biomass

# Train with custom date range
python scripts/train.py --countries DE --types load --start 2023-01-01

# Train with different algorithm (xgboost, lightgbm, catboost)
python scripts/train.py --countries DE --types load --algorithm lightgbm

# Train with Optuna hyperparameter optimization (50 trials)
python scripts/train.py --countries DE --types load --optuna --n-trials 50

# Train with walk-forward validation (6 folds)
python scripts/train.py --countries DE --types load --walk-forward --n-folds 6

# Compare multiple algorithms
python scripts/train.py --countries DE --types load --algorithms xgboost,lightgbm,catboost

# Train with automated feature selection
python scripts/train.py --countries DE --types load --feature-selection

# Full optimization pipeline
python scripts/train.py --countries DE --types load --walk-forward --optuna --feature-selection

# Train with backtest week exclusion (for fair Chronos-2 comparison)
python scripts/train.py --countries all --types all --exclude-backtest
```

### Chronos-2 (ported from netpredict2)

```bash
# Zero-shot forecast (no fine-tuning, uses pretrained Chronos-2)
python scripts/forecast_chronos2.py --experiment V002 --countries DE --types load --target-date 2024-01-15

# Fine-tune Chronos-2 (requires GPU + chronos venv)
python scripts/train_chronos2.py --experiment V003 --device cuda

# Fine-tune with overrides
python scripts/train_chronos2.py --experiment V003 --countries DE --types load --steps 100 --device cuda

# Generate fine-tuned forecasts
python scripts/forecast_chronos2.py --experiment V003 --countries DE,FR --types load,price --save-to-db

# Compare experiments (XGBoost vs Chronos-2 across backtest weeks)
python scripts/compare_experiments.py --experiments V001,V003 --weeks all --countries DE --types load

# Net position forecasting (V010+)
python scripts/forecast_chronos2.py --experiment V010 --countries DE --types net_position --target-date 2024-01-15
python scripts/compare_experiments.py --experiments persistence,V010 --weeks W01 --countries DE --types net_position
```

### Forecasting

```bash
# Generate D+2 forecasts for all countries
python scripts/forecast_daily.py

# Dry run (no database write)
python scripts/forecast_daily.py --dry-run

# Specific countries
python scripts/forecast_daily.py --countries DE,FR
```

## Model Details

### Features

**Time Features:**
- hour, day_of_week, month, is_weekend
- Cyclical encoding: hour_sin/cos, day_sin/cos, month_sin/cos

**Lag Features (same hour):**
- D-1 (24h ago)
- D-7 (1 week ago)
- D-14 (2 weeks ago)

**Rolling Statistics:**
- 24h and 168h (1 week) rolling mean, std, min, max

**Holiday Features:**
- is_holiday - Binary flag for public holidays
- days_to_holiday - Days until next holiday (capped at 14)
- days_from_holiday - Days since last holiday (capped at 14)
- is_bridge_day - Workday between holiday and weekend

**Weather Features:**
- Load: temperature, heating/cooling degree days
- Price: temperature, wind speed, solar radiation
- Renewable (total): solar radiation, wind speeds
- Solar: shortwave/direct/diffuse radiation
- Wind (onshore/offshore): wind speed at 10m and 100m
- Hydro: temperature, precipitation
- Biomass: temperature

### Supported Algorithms

| Algorithm | Description |
|-----------|-------------|
| XGBoost | Default. Gradient boosting with regularization |
| LightGBM | Fast gradient boosting with histogram-based learning |
| CatBoost | Gradient boosting with built-in categorical handling |
| Chronos-2 | Foundation model (120M params). Requires GPU + separate venv |

### XGBoost Configuration

```python
{
    'n_estimators': 500,
    'max_depth': 8,
    'learning_rate': 0.05,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
```

### Chronos-2 Configuration (ported from netpredict2)

```python
{
    'model': 'amazon/chronos-2',       # 120M param foundation model
    'context_length': 672,             # 4 weeks of hourly data
    'prediction_length': 24,           # 1 day ahead
    'fine_tune_steps': 5000,           # Cosine LR schedule
    'learning_rate': 1e-5,
    'quantiles': [0.1, 0.2, ..., 0.9] # 9 quantile levels
}
```

**Covariates (suffix convention from netpredict2):**
- **Suffix-0** (future-known, through D+2): Weather (Open-Meteo), time features, holidays
- **Suffix-1** (past-only, through D+1): TSO load/generation forecasts, DA prices, neighbor features

**Key dependencies:** `torch>=2.1`, `transformers>=4.40`, `chronos-forecasting>=2.0` (separate venv)

### Experiment System

Experiments are versioned V001-Vnnn with configs in `experiments/`. Both XGBoost and Chronos-2 run in parallel — forecasts stored with distinct `model_name` values in the `forecasts` table.

```bash
experiments/
├── registry.json           # Master index of all experiments
├── V001/config.json        # XGBoost baseline
├── V002/config.json        # Chronos-2 zero-shot
└── V003/config.json        # Chronos-2 fine-tuned (5000 steps)
```

**Backtest evaluation:** 12 held-out weeks (W01-W12) spanning 2024-2026, NaN-masked during training of ALL models. Use `--exclude-backtest` flag for XGBoost, automatic for Chronos-2.

### Expected Performance

| Type | Typical MAPE |
|------|-------------|
| Load | 2-5% |
| Price | 10-20% |
| Renewable | 15-30% |

## Evaluation Queries

```sql
-- Compare forecasts vs actuals for load
SELECT
    f.target_timestamp_utc,
    f.forecast_value AS predicted,
    l.load_mw AS actual,
    ABS(f.forecast_value - l.load_mw) AS error
FROM forecasts f
JOIN energy_load l
    ON f.country_code = l.country_code
    AND f.target_timestamp_utc = l.timestamp_utc
WHERE f.forecast_type = 'load'
    AND f.country_code = 'DE'
ORDER BY f.target_timestamp_utc DESC
LIMIT 24;

-- Forecast accuracy summary by country
SELECT
    f.country_code,
    f.forecast_type,
    COUNT(*) as forecasts,
    AVG(ABS(f.forecast_value - l.load_mw)) as avg_mae
FROM forecasts f
JOIN energy_load l
    ON f.country_code = l.country_code
    AND f.target_timestamp_utc = l.timestamp_utc
WHERE f.forecast_type = 'load'
GROUP BY f.country_code, f.forecast_type;
```

## Supported Countries

AT, BE, BG, CH, CZ, DE, EE, ES, FI, FR, GR, HR, HU, IT, LT, LV, NL, NO, PL, PT, RO, SE, SI, SK

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENERGY_DB_PATH` | Override database path | `../data_gathering/energy_dashboard.db` |

## Troubleshooting

**"Model not found"**
- Run training first: `python scripts/train.py --countries <code> --types <type>`

**"Database error"**
- Check database path exists
- Set `ENERGY_DB_PATH` environment variable if needed

**Low accuracy**
- Ensure sufficient training data (minimum 1 year recommended)
- Check for data quality issues in source tables
- Consider retraining with more recent data

## Maintenance

**Weekly:** Retrain models with latest data
```bash
python scripts/train.py --countries all --types all
```

**Monitor logs:**
```bash
tail -f logs/daily_*.log
```

**Check cron job:**
```bash
crontab -l | grep forecast
```
