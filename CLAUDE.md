# CLAUDE.md

This file provides guidance to Claude Code when working with the energy forecasting module.

## Module Overview

D+2 energy forecasting module for European electricity markets. Generates 24-hour forecasts for the day after tomorrow.

`scripts/scheduler_setup.sh` installs `forecast_daily.py` at 18:00, but that is
not the only job: every Chronos-2 net-position run in the database was generated
at **~06:00 UTC** (8 runs at 06:00, 1 at 07:00 as of 2026-08-04), scheduled
elsewhere. `RUN_HOUR` in `compare_experiments.py` tracks that measured time,
since backtest `as_of` bounds depend on it — check it against real `generated_at`
values before trusting a backtest, rather than against this file.

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

Two files, and pointing at the wrong one is the trap this section exists to
prevent (ABL-73). Neither path is hardcoded — both come from the environment:

| role | path | env var |
|---|---|---|
| **replica** (read) | `C:\Code\able\data\energy_dashboard.db` | `ENERGY_DB_PATH` |
| **sidecar** (write) | `C:\Code\able\data\forecasts_local.db` | `FORECAST_OUTPUT_DB` |

`scripts/workstation/run-net-position.ps1:10-11` is what sets them for the
scheduled job, and `reports/net_position_eval/latest.json` → `meta.replica_db` /
`meta.sidecar_db` records which pair a stored evaluation actually ran against.
The replica is refreshed at 07:00 by the `able-db-sync` job; the forecast runs
at 08:00 behind it. **All writes go to the sidecar** — the replica is a
read-only mirror of prod and nothing here may write to it.

Local runs read `.env` (via `python-dotenv`, `config.py:11`). It is gitignored
and must stay untracked — it carries a machine-specific absolute path.

> **There is a decoy.** `../energy-data-gathering/energy_dashboard.db` (3.0 GB)
> is a **stale partial snapshot**, not the replica, and it is the nearest real
> file to every wrong path this module has been pointed at. Measured 2026-08-07:
> its `net_position` holds 10,968 rows ending **2024-01-15** (the replica has
> 645,618, current to the hour); **AT and DE have zero rows**, BE/NL/FR stop in
> 2023-24; `energy_generation` does not exist as a table; and every `fetched_at`
> falls in one 52-minute import session on 2026-04-01. A per-country training or
> backtest run against it yields a 19-country program with the priority majors
> (BE, NL, AT, FR — the net-position program plan's §7.2, recorded on ABL-73)
> silently missing and numbers that look fine.
> Do not delete it — `energy-data-gathering` may own it.

`validate_config()` (`config.py`) now catches exactly that: it checks the
database is not merely *present* but *current*, requiring `net_position` rows
within `DB_STALE_AFTER_HOURS` (48) for `DB_CURRENCY_PROBE_COUNTRIES`
(BE, NL, AT, FR, DE) and failing with a per-country reason otherwise. A stale
timestamp is disqualifying; a *future* one is not — `net_position` is day-ahead,
so a healthy replica reaches the end of tomorrow's market day. `ALLOW_STALE_DB=1`
downgrades the failure to a warning for a deliberate run against a partial
database; do not bake it into a script. `python config.py` prints the verdict.

Note this runs in `validate_config()`, which is called by `scripts/train.py`,
`train_all.py`, `train_baselines.py` and `forecast_daily.py` — **not** by
`scripts/forecast_chronos2.py`, so the scheduled 08:00 net-position job is
unaffected by it.

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

`prediction_length: 24` is the *published* horizon, not the horizon the model
is asked for. See below.

**Covariates (suffix convention from netpredict2):**
- **Suffix-0** (future-known, through D+2): Weather (Open-Meteo), time features, holidays
- **Suffix-1** (past-only, through D+1): TSO load/generation forecasts, DA prices, neighbor features

### The context ends where the data ends, not where the schedule says

A D+2 run fires around 06:00 UTC on day D for the whole of day D+2. The
schedule's nominal context cutoff is D+1 23:00 — roughly **42 hours after the
run actually happens**. No observation exists for that span.

`build_for_country` therefore measures the last real observation
(`_last_available_timestamp`) and ends the context there, then forecasts across
the gap *and* the target day, and the caller publishes the **last 24** points.
`future_index` names their timestamps; `forecast_chronos2.py` asserts that tail
is exactly the target day rather than trusting the arithmetic. When observations
do reach the nominal cutoff there is no gap and the horizon collapses to a
plain 24.

**How long the horizon actually comes back depends on how the target is
published, and `net_position` is the exception.** Where actuals stop near real
time (`load`, `price`), a 06:00Z run is ~42h short of the nominal cutoff and
`prediction_length` is ~66. But `net_position` is **day-ahead** published —
day D's values appear around 12:45 CET on D−1 — so a 06:00Z run on D
legitimately holds actuals through **D 21:00**, the gap is 26h, and
`prediction_length` comes back as **50**. Measured 2026-08-06 across all 16
stored vintages: 26h staleness and a 50h horizon for all 19 live countries,
without exception (ABL-28).

This is also a trap for `as_of`. `as_of` bounds on *target* timestamp, not on
ingest time, so setting it to the run instant (`RUN_HOUR` on D) cuts a
day-ahead target's context 15h shorter than the live run really had, and
understates the pipeline. For `net_position` the serve-faithful bound is
**D 22:00**, not D 06:00 — verified by reproducing the live 2026-08-06 vintage
**bit-exactly** (max |diff| 0.0 MW over 480 points; `predict_quantiles` is
deterministic, so an exact match really does mean an identical input).

Before this, the context was built out to the nominal cutoff regardless, where
`_align_to_index` forward-filled 6h and wrote **0.0** into the remaining ~36.
The model's most recent context was a block of zeros. Net position is signed and
centred near zero, so nothing downstream looked wrong — but measured forecasts
came out at 6% of actual for FR and sign-flipped for DE, and the dashboard
showed an 8 GW discontinuity at each day boundary where one run's recovered tail
met the next run's near-zero start.

**This is why offline experiment scores did not catch it.** `compare_experiments.py`
read the database as it stands *today*, so its context ran right up to D+1 23:00
with real data — the harness was scoring a model that never existed in
production. Both it and any new backtest must pass `as_of` (the moment the run
would have fired: D+2 minus two days at `RUN_HOUR`), which bounds every query
including the weather-forecast run time. Without it, offline numbers are
measuring leaked information.

Interior gaps shorter than the ffill limit are still filled, and anything longer
still becomes `0.0` via `_align_to_index`. It is the same failure mode, so
prefer leaving a genuine hole NaN over inventing a zero. Two things this still
bites, both measured 2026-08-06 (ABL-28):

- **The target, for a country that has stopped publishing.** For the 19 live
  net-position countries the 672h context is 672/672 real observations with
  zero fill, so "coverage is near-complete in practice" holds — but GR is
  **24 real hours and 648 zero-filled**, and its 24 real hours are themselves
  exactly `0.0` upstream. A constant-zero context yields a constant-zero
  forecast (1e-10..4.6e-7 MW), which the pipeline still publishes and pushes;
  the dashboard withholds it at render time (ABL-25). GR's horizon reaches
  **362h**. Filed as its own issue: refuse rather than forecast when the
  context is degenerate or stale.
- **Covariates, which the context-cutoff fix does not cover.** Each is aligned
  to the *target's* index, so a covariate whose source stops earlier is ffilled
  6h and then zeroed. `weather_data` `data_quality='actual'` is retained on a
  rolling 672h window, so on a current vintage `weather__temperature_2m_k`
  reads **297 K for 656 hours and then 0 K for the last 16**. Measured cost:
  under 1% of MAE, so it is filed rather than fixed in flight.

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
