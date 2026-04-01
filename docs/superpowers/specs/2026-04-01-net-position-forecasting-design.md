# Phase 3: Net Position Forecasting with Chronos-2

**Date:** 2026-04-01
**Status:** Draft
**Scope:** Extend energy-forecast to forecast country-level net position using a dedicated Chronos-2 model
**Target repo:** `C:\Code\energy-forecast\`
**Depends on:** Phase 1 (Chronos-2 engine), Phase 2 (crossborder flow + net position data)

## Context

Phases 1-2 built the Chronos-2 forecasting engine and the cross-border data pipeline. Phase 3 connects them: a dedicated Chronos-2 model fine-tuned to forecast the aggregate net position (MW) for each of the 24 European countries.

**What we're forecasting:**
- Net position per country — the aggregate import/export balance (positive = net exporter, negative = net importer)
- 24 hourly values for D+2 (same as load/price/renewable forecasts)
- 24 series total (one per supported country)

**What we're NOT doing in Phase 3:**
- No bilateral flow forecasting (individual DE→FR, DE→NL flows)
- No Ridge post-correction (deferred to Phase 4)
- No XGBoost net position model (Chronos-2 only)

## Architecture

This is an **extension** of the existing Phase 1 infrastructure, not a new system. The Chronos-2 engine, finetuner, experiment framework, and comparison scripts are already generic. We extend the input builder and covariate mapper to handle net position as a target and crossborder flows as covariates.

### Changes overview

| File | Change | Size |
|------|--------|------|
| `config.py` | Register `net_position` as forecast type | Small |
| `src/chronos2/input_builder.py` | Add net position target loader + crossborder flow covariate loader | Medium |
| `src/chronos2/covariate_mapper.py` | Add `net_position` covariate mapping with per-border flows | Small |
| `scripts/compare_experiments.py` | Add persistence baseline for non-XGBoost types | Small |
| `experiments/registry.json` | Add V010, V011 entries | Small |
| `experiments/V010/config.json` | **CREATE** — zero-shot config | Small |
| `experiments/V011/config.json` | **CREATE** — fine-tuned config | Small |

No changes to `engine.py`, `finetuner.py`, `train_chronos2.py`, `forecast_chronos2.py`.

## Config Registration

Add `net_position` to the existing configuration:

```python
# FORECAST_TYPES — add to list
FORECAST_TYPES = ['load', 'price', 'renewable', 'net_position']

# FORECAST_HORIZONS
FORECAST_HORIZONS['net_position'] = [1, 2]

# WEATHER_FEATURES
WEATHER_FEATURES['net_position'] = [
    'temperature_2m_k',
    'wind_speed_100m_ms',
    'shortwave_radiation_wm2',
]
```

The `TARGET_TABLE_MAP` in `input_builder.py`:
```python
"net_position": ("net_position", "net_position_mw"),
```

## Input Builder Extensions

### Target loader

New function `_load_net_position_series()` to load from the `net_position` table:

```python
def _load_net_position_series(country_code, start, end) -> pd.Series:
    query = """
        SELECT timestamp_utc, net_position_mw as target_value
        FROM net_position
        WHERE country_code = ?
          AND timestamp_utc >= ? AND timestamp_utc < ?
        ORDER BY timestamp_utc
    """
    # Same resample-to-hourly pattern as other loaders
```

Integrated into `_load_target_series()` via the `TARGET_TABLE_MAP` lookup — no special-casing needed if the map entry is correct.

### Crossborder flow covariate loader

New function to load per-border flows as individual covariates:

```python
def _load_crossborder_flow_covariates(country_code, start, end) -> dict[str, pd.Series]:
    """Load per-border flow series as individual suffix-1 covariates.

    For country DE with neighbors FR, NL, PL, CZ, AT, CH, returns:
        {
            "flow__FR": pd.Series (hourly MW),
            "flow__NL": pd.Series (hourly MW),
            "flow__PL": pd.Series (hourly MW),
            ...
        }
    """
    query = """
        SELECT country_to, timestamp_utc, flow_mw
        FROM crossborder_flows
        WHERE country_from = ?
          AND timestamp_utc >= ? AND timestamp_utc < ?
        ORDER BY timestamp_utc
    """
    # Group by country_to, resample each to hourly, return as dict
```

Each border becomes a separate suffix-1 (past-only) covariate since we don't have flow forecasts for D+2.

### Neighbor net position covariate loader

```python
def _load_neighbor_net_position(neighbor_code, start, end) -> pd.Series:
    """Load a neighbor country's net position as a suffix-1 covariate."""
    # Same query as target loader but for a different country
```

## Covariate Mapper Extension

Add `net_position` entry to the mapper:

```python
WEATHER_COVARIATES["net_position"] = [
    "temperature_2m_k",
    "wind_speed_100m_ms",
    "shortwave_radiation_wm2",
]

TSO_COVARIATES["net_position"] = ["tso_load_forecast"]
```

The `build_covariate_map()` function gets a new branch for `net_position`:

| Suffix | Source | Covariates |
|--------|--------|------------|
| suffix-0 (future-known) | Weather | temperature, wind speed, radiation |
| suffix-0 | Calendar | hour, dayofweek, month, is_holiday |
| suffix-1 (past-only) | Crossborder flows | flow\_\_FR, flow\_\_NL, flow\_\_PL, ... (per border, dynamic) |
| suffix-1 | TSO forecasts | tso\_\_load\_forecast |
| suffix-1 | DA prices | da\_\_price |
| suffix-1 | Neighbor net positions | neighbor\_np\_\_FR, neighbor\_np\_\_NL, ... (top 3) |

The crossborder flow covariates are **dynamic** — the mapper queries the database to discover which neighbors exist for a given country, then creates one covariate entry per border.

## Experiment Configs

Net position experiments start at V010 (clear separation from load/price/renewable V001-V005):

| Exp | Name | Description |
|-----|------|-------------|
| V010 | Chronos-2 net position zero-shot | Pretrained, no fine-tuning. All covariates. |
| V011 | Chronos-2 net position fine-tuned | 5000 steps, cosine LR. Dedicated to net position. |
| V012 | + Neighbor net positions | Add top-3 neighbor net positions as covariates |

### V011 config example

```json
{
  "id": "V011",
  "model": {
    "type": "chronos-2",
    "base": "amazon/chronos-2",
    "context_length": 672,
    "prediction_length": 24,
    "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  },
  "training": {
    "fine_tune": true,
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
    "suffix_1": ["crossborder_flows", "tso_load_forecast", "da_prices"]
  },
  "countries": ["all"],
  "forecast_types": ["net_position"],
  "training_data": {
    "start": "2023-01-01",
    "end": "2026-03-01",
    "backtest_exclusion": true
  }
}
```

## Evaluation

### Baseline

**Persistence baseline:** "D+2 net position = D net position at the same hour." Implemented via the existing `PersistenceBaseline` class in `src/baselines.py`.

The `compare_experiments.py` script gets a new path for `persistence` as a special experiment ID:
```python
if experiment_id == "persistence":
    return _run_persistence_backtest(country_code, forecast_type, week_start, week_end)
```

### Metrics

| Metric | Role | Notes |
|--------|------|-------|
| MAE (MW) | **Primary** | Average absolute error |
| RMSE (MW) | Secondary | Penalizes large errors |
| Skill score | **Primary** | `1 - (MAE_model / MAE_persistence)`, positive = better |
| SMAPE (%) | Secondary | Handles near-zero values better than MAPE |

MAPE is NOT used for net position because values cross zero (country switches between importing/exporting), causing division-by-zero artifacts.

### Backtest

Same 12 weeks (W01-W12) as Phase 1. The comparison script already supports arbitrary forecast types:

```bash
python scripts/compare_experiments.py \
    --experiments persistence,V010,V011 \
    --weeks all \
    --countries DE,FR,BE \
    --types net_position
```

## Verification

1. Register net_position type, verify config loads:
   ```bash
   python -c "import config; print('net_position' in config.FORECAST_TYPES)"
   ```
2. Build a training input for DE net position:
   ```bash
   python -c "
   from src.chronos2.input_builder import InputBuilder
   ib = InputBuilder()
   inp = ib.build_training_input('DE', 'net_position', '2023-01-01', '2024-01-01')
   print(f'Target: {len(inp[\"target\"])} points, Past covs: {len(inp[\"past_covariates\"])}')
   "
   ```
3. Run zero-shot forecast for DE:
   ```bash
   python scripts/forecast_chronos2.py --experiment V010 --countries DE --types net_position --target-date 2024-01-15
   ```
4. Compare V010 vs persistence:
   ```bash
   python scripts/compare_experiments.py --experiments persistence,V010 --weeks W01 --countries DE --types net_position
   ```
