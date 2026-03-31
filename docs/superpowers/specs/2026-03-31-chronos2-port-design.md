# Chronos-2 Port from NetPredict2 to Energy Dashboard

**Date:** 2026-03-31
**Status:** Draft
**Scope:** Phase 1 of netpredict2 knowledge transfer

## Context

The netpredict2 project (Coreso/JAO EUPHEMIA competition) achieved +0.141 skill score forecasting ~70 European grid zone net positions using Chronos-2 foundation model (120M params) with ENTSO-E covariates and Ridge post-correction.

This design ports the Chronos-2 engine into the energy-forecast dashboard to improve load, price, and renewable forecasting across 24 European countries. This is Phase 1 of a multi-phase effort; later phases will add net position forecasting and apply additional netpredict2 techniques (Ridge correction, hierarchical reconciliation).

**Constraints:**
- No Meteologica data (commercial, unavailable for dashboard)
- No VPS/commercial flow data
- Available: ENTSO-E data (load, price, renewables, TSO forecasts, generation forecasts), Open-Meteo weather, realized actuals

## Architecture

### New module structure

```
src/chronos2/
  engine.py             # Chronos-2 pipeline wrapper (ported from netpredict2)
  finetuner.py          # Fine-tuning pipeline (5000 steps, cosine LR)
  input_builder.py      # Covariate assembly + context windowing
  covariate_mapper.py   # Country -> covariate mapping (ENTSO-E + weather)

scripts/
  train_chronos2.py     # Fine-tune Chronos-2
  forecast_chronos2.py  # Generate forecasts
  compare_experiments.py # Cross-experiment evaluation

experiments/
  registry.json         # Master experiment index
  V001/config.json      # XGBoost baseline (existing)
  V002/config.json      # Chronos-2 zero-shot
  V003/config.json      # Chronos-2 fine-tuned
  ...
```

### Coexistence with XGBoost

Both pipelines run in parallel. Each stores forecasts with distinct `model_name` values in the `forecasts` table:
- `"xgboost"` (or `"xgboost-V001"`) for existing pipeline
- `"chronos-2-V003"` for Chronos-2 experiments

The dashboard frontend can filter and compare by model_name.

## Experiment Versioning

Inspired by netpredict2's V001-V066 progression. Each experiment gets:

```
experiments/{id}/
  config.json    # Full configuration (model, training, covariates, countries)
  results.json   # Metrics per country/type/backtest week
  notes.md       # What changed, why, outcome
```

### registry.json

```json
{
  "experiments": [
    {"id": "V001", "model": "xgboost", "parent": null, "description": "XGBoost baseline (existing pipeline)"},
    {"id": "V002", "model": "chronos-2", "parent": null, "description": "Pretrained Chronos-2, no fine-tuning"},
    {"id": "V003", "model": "chronos-2", "parent": "V002", "description": "Fine-tuned 5000 steps, all covariates"},
    {"id": "V004", "model": "chronos-2", "parent": "V003", "description": "+ geographic neighbor features"},
    {"id": "V005", "model": "chronos-2", "parent": "V003", "description": "+ extended context (1008h vs 672h)"}
  ]
}
```

### config.json per experiment

```json
{
  "id": "V003",
  "model": {
    "type": "chronos-2",
    "base": "amazon/chronos-2",
    "context_length": 672,
    "prediction_length": 24,
    "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  },
  "training": {
    "fine_tune_steps": 5000,
    "learning_rate": 1e-5,
    "batch_size": 32,
    "lr_scheduler": "cosine",
    "warmup_ratio": 0.1,
    "gradient_accumulation_steps": 4,
    "val_fraction": 0.1
  },
  "covariates": {
    "suffix_0": ["weather_temperature", "weather_wind_100m", "weather_radiation", "hour", "dayofweek", "month", "is_holiday"],
    "suffix_1": ["tso_load_forecast", "tso_solar_forecast", "tso_wind_forecast", "da_prices"]
  },
  "countries": ["all"],
  "forecast_types": ["load", "price", "renewable", "solar", "wind_onshore", "wind_offshore", "hydro_total", "biomass"],
  "training_data": {
    "start": "2023-01-01",
    "end": "2026-03-01"
  }
}
```

## Chronos-2 Engine

### engine.py

Ported from netpredict2's `src/model/chronos_engine.py`. Key interface:

```python
class ChronosEngine:
    def __init__(self, model_path, device="cuda", context_length=672, prediction_length=24):
        # Loads Chronos2Pipeline.from_pretrained()

    def forecast(self, target, past_covariates, future_covariates) -> dict:
        # Returns {"median": array(24), "mean": array(24), "quantiles": array(9, 24)}

    def forecast_batch(self, inputs: list[dict]) -> list[dict]:
        # Batch inference for multiple series
```

**Key parameters (from netpredict2 research):**
- Context length: 672 hours (4 weeks) -- proven optimal
- Prediction length: 24 hours (1 day)
- 9 quantiles: [0.1, 0.2, ..., 0.9]
- NaN handling: replace with 0.0, pad/truncate to context_length

### finetuner.py

Ported from netpredict2's `src/model/finetuner.py`. Key interface:

```python
class ChronosFinetuner:
    def prepare_training_data(self, countries, forecast_types, start_date, end_date) -> list[dict]:
        # One training input per (country, forecast_type) pair
        # NaN-masks backtest weeks

    def train(self, training_inputs, output_dir, num_steps=5000, lr=1e-5, scheduler="cosine"):
        # Fine-tunes from pretrained amazon/chronos-2
        # Saves checkpoint to output_dir
```

**Training approach:**
- Single model trained on ALL series (all countries x all forecast types together)
- Chronos's tokenization handles scale differences between load (thousands MW), price (tens EUR), and renewables
- Multi-series training helps generalization (netpredict2 key finding)
- 5000 steps with cosine LR (sweet spot from netpredict2, 200 steps in current Bolt code is far too few)

### input_builder.py

Loads target + covariates from DB, aligns to hourly grid:

```python
class InputBuilder:
    def build_for_country(self, country_code, forecast_type, target_date) -> dict:
        # Returns {"target": array, "past_covariates": dict, "future_covariates": dict}

    def build_training_input(self, country_code, forecast_type, start, end) -> dict:
        # For training: full history with NaN-masked backtest weeks
```

**Time logic (following netpredict2 exactly):**
- For forecast date T (D+2):
  - Past cutoff: T - 1 day + 23 hours (D+1 23:00)
  - Context start: past_cutoff - 671 hours
  - Target window: T 00:00 to T 23:00
- Suffix-0 covariates available through D+2 (weather, time features)
- Suffix-1 covariates available through D+1 only (TSO forecasts, DA prices)
- Alignment: reindex to hourly grid with ffill(limit=6h).bfill(limit=6h).fillna(0.0)

### covariate_mapper.py

Maps (country, forecast_type) to available covariates:

| Forecast Type | Suffix-0 (future-known) | Suffix-1 (past-only) |
|---|---|---|
| Load | weather (temp, wind), time, holidays | TSO load forecast, DA prices, neighbor loads |
| Price | weather (all), time, holidays | TSO load + gen forecast, DA prices (own + neighbors) |
| Renewable | weather (radiation, wind), time | TSO gen forecast, DA prices |
| Solar | weather (radiation), time | TSO solar forecast |
| Wind onshore/offshore | weather (wind speeds), time | TSO wind forecast |
| Hydro | weather (temp, precip proxy), time | TSO load forecast |
| Biomass | weather (temp), time | TSO load forecast |

### Geographic neighbors

Simplified from netpredict2's PTDF electrical distance matrix:

```python
COUNTRY_NEIGHBORS = {
    "AT": ["DE", "CZ", "HU", "SI", "CH", "IT"],
    "BE": ["FR", "NL", "DE"],
    "BG": ["RO", "GR"],
    "CH": ["DE", "FR", "AT", "IT"],
    "CZ": ["DE", "PL", "SK", "AT"],
    "DE": ["FR", "NL", "PL", "CZ", "AT", "CH"],
    "EE": ["LT", "LV", "FI"],
    "ES": ["FR", "PT"],
    "FI": ["SE", "EE", "NO"],
    "FR": ["DE", "BE", "ES", "IT", "CH"],
    "GR": ["BG", "IT"],
    "HR": ["SI", "HU"],
    "HU": ["AT", "SK", "RO", "HR", "SI"],
    "IT": ["FR", "AT", "SI", "GR", "CH"],
    "LT": ["LV", "PL", "EE"],
    "LV": ["LT", "EE"],
    "NL": ["DE", "BE"],
    "NO": ["SE", "FI"],
    "PL": ["DE", "CZ", "SK", "LT"],
    "PT": ["ES"],
    "RO": ["BG", "HU"],
    "SE": ["NO", "FI"],
    "SI": ["AT", "IT", "HR", "HU"],
    "SK": ["CZ", "PL", "HU", "AT"],
}
```

Top-3 neighbors' load and price values injected as suffix-1 past covariates.

## Quantile Storage

New table for probabilistic forecasts:

```sql
CREATE TABLE forecast_quantiles (
    id INTEGER PRIMARY KEY,
    country_code TEXT NOT NULL,
    forecast_type TEXT NOT NULL,
    target_timestamp_utc TIMESTAMP,
    generated_at TIMESTAMP,
    quantile REAL,
    forecast_value REAL,
    model_name TEXT,
    UNIQUE(country_code, forecast_type, target_timestamp_utc, quantile, model_name, generated_at)
);
```

## Backtest Framework

### 12 held-out weeks (NaN-masked during training of ALL models):

| Week | Dates | Season |
|------|-------|--------|
| W01 | 2024-01-15 to 2024-01-21 | Winter Y1 |
| W02 | 2024-03-11 to 2024-03-17 | Early spring Y1 |
| W03 | 2024-04-22 to 2024-04-28 | Spring Y1 |
| W04 | 2024-07-15 to 2024-07-21 | Summer Y1 |
| W05 | 2024-09-09 to 2024-09-15 | Late summer Y1 |
| W06 | 2024-11-11 to 2024-11-17 | Autumn Y1 |
| W07 | 2025-01-13 to 2025-01-19 | Winter Y2 |
| W08 | 2025-04-07 to 2025-04-13 | Spring Y2 |
| W09 | 2025-06-16 to 2025-06-22 | Summer Y2 |
| W10 | 2025-10-06 to 2025-10-12 | Autumn Y2 |
| W11 | 2026-01-12 to 2026-01-18 | Winter Y3 |
| W12 | 2026-02-16 to 2026-02-22 | Late winter Y3 |

Total masked: 2,016 hours (~7.7% of data). None overlap with major holidays.

### Evaluation protocol

For each backtest week (7 days):
1. For each day D: build input with context ending D-1 23:00
2. Generate D+2 forecast (24 hours)
3. Compare to actuals from DB
4. Compute: MAE, MAPE, RMSE, SMAPE per (country, type, week)
5. Skill score: `skill = 1 - (MAE_model / MAE_baseline)` where baseline = V001 (XGBoost)

### Comparison script

```bash
python scripts/compare_experiments.py \
    --experiments V001,V003 \
    --weeks all \
    --countries DE,FR,BE \
    --types load,price,renewable
```

**XGBoost baseline must be retrained** with the same 12 weeks excluded for fair comparison.

## Dependencies

### New (Chronos-2 venv):
- torch>=2.1
- transformers>=4.40
- chronos-forecasting>=2.0

### Existing (unchanged):
- xgboost, lightgbm, catboost, pandas, numpy, scikit-learn, joblib, holidays

### Infrastructure:
- GPU with 8GB+ VRAM (training + inference)
- Separate virtual environment for Chronos-2 dependencies
- `forecast_daily.py --model all` invokes XGBoost natively and shells out to Chronos-2 venv via subprocess (same pattern as existing `chronos_forecaster.py`)

## Experiment Roadmap (Phase 1)

| Exp | Name | Description | Parent |
|-----|------|-------------|--------|
| V001 | XGBoost baseline | Existing pipeline, retrained with backtest exclusion | -- |
| V002 | Chronos-2 zero-shot | Pretrained, no fine-tuning, weather + time covariates | -- |
| V003 | Chronos-2 fine-tuned | 5000 steps, cosine LR, all covariates | V002 |
| V004 | + Geographic neighbors | Top-3 neighbor load/price as covariates | V003 |
| V005 | + Extended context | 1008h vs 672h context window | V003 |

### Future phases (not in scope):
- **Phase 2:** Add ENTSO-E cross-border flows + realized net position data collection
- **Phase 3:** Net position forecasting with Chronos-2 (V010+)
- **Phase 4:** Ridge post-correction on Chronos-2 forecasts
- **Phase 5:** Frontend: experiment comparison views, quantile bands, net position dashboard

## Verification

### End-to-end test:
1. Fine-tune Chronos-2 on DE load data (single country, single type) with backtest exclusion
2. Generate forecast for one backtest week
3. Compare MAE/MAPE to XGBoost V001 on the same week
4. Verify forecasts are stored correctly in DB with proper model_name
5. Verify quantiles are stored in forecast_quantiles table

### Smoke test:
```bash
# Train on small subset (fast)
python scripts/train_chronos2.py --experiment V002 --countries DE --types load --steps 0

# Generate forecast
python scripts/forecast_chronos2.py --experiment V002 --countries DE --types load --target-date 2026-01-15

# Compare
python scripts/compare_experiments.py --experiments V001,V002 --weeks W01 --countries DE --types load
```
