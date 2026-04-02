# Net Position Forecasting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the Chronos-2 pipeline to forecast country-level net position using crossborder flows as covariates.

**Architecture:** Register `net_position` as a forecast type, add data loaders for the `net_position` and `crossborder_flows` tables, extend the covariate mapper with per-border flow features, add a persistence baseline to the comparison script, and create V010/V011 experiment configs.

**Tech Stack:** Python, pandas, SQLite, Chronos-2 (existing engine unchanged)

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `config.py` | MODIFY | Register net_position as forecast type |
| `src/chronos2/input_builder.py` | MODIFY | Add crossborder flow covariate loader, add net_position to TARGET_TABLE_MAP |
| `src/chronos2/covariate_mapper.py` | MODIFY | Add net_position covariate mapping with per-border flows |
| `scripts/compare_experiments.py` | MODIFY | Add persistence baseline path |
| `experiments/registry.json` | MODIFY | Add V010, V011 entries |
| `experiments/V010/config.json` | CREATE | Zero-shot net position config |
| `experiments/V011/config.json` | CREATE | Fine-tuned net position config |

---

### Task 1: Register net_position in config.py

**Files:**
- Modify: `C:\Code\energy-forecast\config.py`

- [ ] **Step 1: Add net_position to FORECAST_TYPES**

Find the line `FORECAST_TYPES = ['load', 'price', 'renewable']` and change to:

```python
FORECAST_TYPES = ['load', 'price', 'renewable', 'net_position']
```

- [ ] **Step 2: Add net_position to FORECAST_HORIZONS**

Find the `FORECAST_HORIZONS` dict and add:

```python
    'net_position': [1, 2],   # D+1 and D+2
```

- [ ] **Step 3: Add net_position to DEFAULT_HORIZONS**

Find the `DEFAULT_HORIZONS` dict and add:

```python
    'net_position': 2,    # D+2 - net position forecast
```

- [ ] **Step 4: Add net_position to WEATHER_FEATURES**

Find the `WEATHER_FEATURES` dict and add after the `biomass` entry:

```python
    'net_position': [
        'temperature_2m_k',
        'wind_speed_100m_ms',
        'shortwave_radiation_wm2',
    ],
```

- [ ] **Step 5: Verify config loads**

```bash
cd C:/Code/energy-forecast
python -c "import config; print('net_position' in config.FORECAST_TYPES, config.FORECAST_HORIZONS.get('net_position'), config.WEATHER_FEATURES.get('net_position'))"
```
Expected: `True [1, 2] ['temperature_2m_k', 'wind_speed_100m_ms', 'shortwave_radiation_wm2']`

- [ ] **Step 6: Commit**

```bash
git add config.py
git commit -m "feat: register net_position as forecast type in config"
```

---

### Task 2: Add net_position to TARGET_TABLE_MAP in input_builder.py

**Files:**
- Modify: `C:\Code\energy-forecast\src\chronos2\input_builder.py`

- [ ] **Step 1: Add net_position entry to TARGET_TABLE_MAP**

Find the `TARGET_TABLE_MAP` dict (around line 37) and add after the `biomass` entry:

```python
    "net_position": ("net_position", "net_position_mw"),
```

- [ ] **Step 2: Handle the net_position table's lack of data_quality column**

The `_load_target_series()` function adds a `data_quality = 'actual'` filter for `energy_load` and `energy_price` tables. The `net_position` table doesn't have a `data_quality` column used in the same way. The current code already handles this correctly — it only adds the filter for `energy_load` and `energy_price`:

```python
    quality_filter = ""
    if table in ("energy_load", "energy_price"):
        quality_filter = "AND data_quality = 'actual'"
```

No change needed here — `net_position` table will NOT get the filter. Verify by reading the code.

- [ ] **Step 3: Verify target loading works**

```bash
python -c "
from src.chronos2.input_builder import TARGET_TABLE_MAP
print('net_position' in TARGET_TABLE_MAP)
print(TARGET_TABLE_MAP['net_position'])
"
```
Expected: `True` and `('net_position', 'net_position_mw')`

- [ ] **Step 4: Commit**

```bash
git add src/chronos2/input_builder.py
git commit -m "feat: add net_position to input builder target table map"
```

---

### Task 3: Add crossborder flow covariate loader to input_builder.py

**Files:**
- Modify: `C:\Code\energy-forecast\src\chronos2\input_builder.py`

- [ ] **Step 1: Add `_load_crossborder_flow_covariates()` function**

Add this function after the existing `_load_load_series()` function (around line 275):

```python
def _load_crossborder_flow_covariates(
    country_code: str,
    start_date: str,
    end_date: str,
) -> dict[str, pd.Series]:
    """Load per-border flow series as individual covariates.

    For country DE with neighbors FR, NL, PL, CZ, AT, CH, returns:
        {"flow__FR": pd.Series, "flow__NL": pd.Series, ...}

    Each series is hourly MW indexed by datetime.
    """
    query = """
        SELECT country_to, timestamp_utc, flow_mw
        FROM crossborder_flows
        WHERE country_from = ?
          AND timestamp_utc >= ?
          AND timestamp_utc < ?
        ORDER BY country_to, timestamp_utc
    """
    conn = _get_connection()
    try:
        df = pd.read_sql_query(query, conn, params=(country_code, start_date, end_date))
    finally:
        conn.close()

    if df.empty:
        return {}

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], format="mixed", utc=True).dt.tz_localize(None)

    result = {}
    for neighbor, group in df.groupby("country_to"):
        series = group.set_index("timestamp_utc")["flow_mw"].resample("h").mean()
        result[f"flow__{neighbor}"] = series

    return result
```

- [ ] **Step 2: Add `_load_neighbor_net_position()` function**

Add right after `_load_crossborder_flow_covariates()`:

```python
def _load_neighbor_net_position(
    country_code: str,
    start_date: str,
    end_date: str,
) -> pd.Series:
    """Load a country's net position as a covariate (for neighbor features)."""
    query = """
        SELECT timestamp_utc, net_position_mw as value
        FROM net_position
        WHERE country_code = ?
          AND timestamp_utc >= ?
          AND timestamp_utc < ?
        ORDER BY timestamp_utc
    """
    conn = _get_connection()
    try:
        df = pd.read_sql_query(query, conn, params=(country_code, start_date, end_date))
    finally:
        conn.close()

    if df.empty:
        return pd.Series(dtype=float)

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], format="mixed", utc=True).dt.tz_localize(None)
    return df.set_index("timestamp_utc")["value"].resample("h").mean()
```

- [ ] **Step 3: Verify syntax**

```bash
python -c "import ast; ast.parse(open('src/chronos2/input_builder.py').read()); print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/chronos2/input_builder.py
git commit -m "feat: add crossborder flow and neighbor net position loaders"
```

---

### Task 4: Extend covariate mapper for net_position

**Files:**
- Modify: `C:\Code\energy-forecast\src\chronos2\covariate_mapper.py`

- [ ] **Step 1: Add net_position to WEATHER_COVARIATES**

Add after the `"biomass"` entry in the `WEATHER_COVARIATES` dict:

```python
    "net_position": [
        "temperature_2m_k",
        "wind_speed_100m_ms",
        "shortwave_radiation_wm2",
    ],
```

- [ ] **Step 2: Add net_position to TSO_COVARIATES**

Add after the `"biomass"` entry in the `TSO_COVARIATES` dict:

```python
    "net_position": ["tso_load_forecast"],
```

- [ ] **Step 3: Add crossborder flows and neighbor net position to `build_covariate_map()`**

Find the section `# --- Suffix-1: Day-ahead prices ---` (around line 159). Replace the `if forecast_type in ("load", "price", "renewable"):` block and everything after it (through the end of the function before the `return`) with:

```python
    # --- Suffix-1: Day-ahead prices ---
    if forecast_type in ("load", "price", "renewable", "net_position"):
        suffix_1.append({
            "source": "energy_price",
            "column": "price_eur_mwh",
            "cov_name": "da__price",
        })

    # --- Suffix-1: Crossborder flows (net_position only) ---
    if forecast_type == "net_position":
        suffix_1.append({
            "source": "crossborder_flows",
            "column": "flow_mw",
            "cov_name": "crossborder_flows",  # Placeholder — input builder handles dynamic expansion
        })

    # --- Suffix-1: Neighbor features ---
    if include_neighbors:
        neighbors = config.COUNTRY_NEIGHBORS.get(country_code, [])[:top_n_neighbors]
        for neighbor in neighbors:
            if forecast_type == "net_position":
                # For net position: use neighbor net positions
                suffix_1.append({
                    "source": "net_position",
                    "column": "net_position_mw",
                    "country_override": neighbor,
                    "cov_name": f"neighbor_np__{neighbor}",
                })
            else:
                # For other types: use neighbor load and price
                suffix_1.append({
                    "source": "energy_load",
                    "column": "load_mw",
                    "country_override": neighbor,
                    "cov_name": f"neighbor__{neighbor}_load",
                })
                suffix_1.append({
                    "source": "energy_price",
                    "column": "price_eur_mwh",
                    "country_override": neighbor,
                    "cov_name": f"neighbor__{neighbor}_price",
                })
```

- [ ] **Step 4: Verify the mapper works for net_position**

```bash
python -c "
from src.chronos2.covariate_mapper import build_covariate_map
cov = build_covariate_map('DE', 'net_position', include_neighbors=True)
print('Suffix-0:', [c['cov_name'] for c in cov['suffix_0']])
print('Suffix-1:', [c['cov_name'] for c in cov['suffix_1']])
"
```
Expected: suffix-0 has weather + calendar, suffix-1 has tso_load_forecast, da_price, crossborder_flows, and neighbor_np__FR/NL/PL.

- [ ] **Step 5: Commit**

```bash
git add src/chronos2/covariate_mapper.py
git commit -m "feat: add net_position covariate mapping with crossborder flows"
```

---

### Task 5: Wire crossborder flow loading into InputBuilder

**Files:**
- Modify: `C:\Code\energy-forecast\src\chronos2\input_builder.py`

The covariate mapper now emits `"source": "crossborder_flows"` entries and `"source": "net_position"` entries. The InputBuilder's `build_for_country()` and `build_training_input()` methods need to handle these new source types in their suffix-1 processing loops.

- [ ] **Step 1: Add crossborder_flows and net_position handling to `build_for_country()`**

Find the suffix-1 processing loop in `build_for_country()` (the `for cov_spec in cov_map["suffix_1"]:` block). Add two new `elif` branches after the existing `energy_load` handler:

```python
            elif source == "crossborder_flows":
                # Load all per-border flows as individual covariates
                flow_dict = _load_crossborder_flow_covariates(
                    country_code, context_start_str, past_cutoff_str
                )
                for flow_name, flow_series in flow_dict.items():
                    past_covariates[flow_name] = _align_to_index(flow_series, past_index) if not flow_series.empty else np.zeros(len(past_index), dtype=np.float32)

            elif source == "net_position":
                series = _load_neighbor_net_position(cc, context_start_str, past_cutoff_str)
                past_covariates[cov_name] = _align_to_index(series, past_index) if not series.empty else np.zeros(len(past_index), dtype=np.float32)
```

- [ ] **Step 2: Add crossborder_flows and net_position handling to `build_training_input()`**

Find the suffix-1 processing loop in `build_training_input()` (the `for cov_spec in cov_map["suffix_1"]:` block). Add two new branches:

```python
            elif source == "crossborder_flows":
                flow_dict = _load_crossborder_flow_covariates(
                    country_code, start_date, end_date
                )
                for flow_name, flow_series in flow_dict.items():
                    if not flow_series.empty:
                        aligned = _align_to_index(flow_series, series_index)
                        if len(aligned) == len(target):
                            past_covariates[flow_name] = aligned

            elif source == "net_position":
                series_data = _load_neighbor_net_position(cc, start_date, end_date)
```

Note: the `net_position` branch should follow the same pattern as the existing `energy_load` branch — check `if not series_data.empty`, align, check length match, add to past_covariates.

- [ ] **Step 3: Verify syntax**

```bash
python -c "import ast; ast.parse(open('src/chronos2/input_builder.py').read()); print('OK')"
```

- [ ] **Step 4: Quick integration test**

```bash
python -c "
from src.chronos2.covariate_mapper import build_covariate_map
cov = build_covariate_map('FR', 'net_position')
print('FR net_position covariates:')
for c in cov['suffix_1']:
    print(f'  {c[\"source\"]}: {c[\"cov_name\"]}')
"
```

- [ ] **Step 5: Commit**

```bash
git add src/chronos2/input_builder.py
git commit -m "feat: wire crossborder flow and neighbor net position loading into InputBuilder"
```

---

### Task 6: Add persistence baseline to compare_experiments.py

**Files:**
- Modify: `C:\Code\energy-forecast\scripts\compare_experiments.py`

- [ ] **Step 1: Add persistence baseline backtest function**

Add after the `_run_xgboost_backtest()` function:

```python
def _run_persistence_backtest(
    country_code: str,
    forecast_type: str,
    week_start: str,
    week_end: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Run persistence baseline backtest (value at same hour 48h ago)."""
    all_actuals = []
    all_forecasts = []

    start_dt = pd.Timestamp(week_start)
    end_dt = pd.Timestamp(week_end)
    current = start_dt

    while current <= end_dt:
        target_date = current.strftime("%Y-%m-%d")
        next_day = (current + timedelta(days=1)).strftime("%Y-%m-%d")

        # Load actuals for this day
        actuals_series = load_actuals(country_code, forecast_type, target_date, next_day)

        if not actuals_series.empty:
            # Persistence: value 48h ago (D+2 baseline)
            history_start = (current - timedelta(days=3)).strftime("%Y-%m-%d")
            history_end = target_date
            history = load_actuals(country_code, forecast_type, history_start, history_end)

            if not history.empty:
                target_index = pd.date_range(current, periods=24, freq="h")
                persist_index = target_index - pd.Timedelta(hours=48)

                actuals_aligned = actuals_series.reindex(target_index)
                persist_values = history.reindex(persist_index)

                # Only use hours where both actual and persistence exist
                valid = ~actuals_aligned.isna() & ~persist_values.isna()
                if valid.sum() > 0:
                    all_actuals.extend(actuals_aligned[valid].values)
                    all_forecasts.extend(persist_values[valid].values)

        current += timedelta(days=1)

    if not all_actuals:
        return None

    return np.array(all_actuals), np.array(all_forecasts)
```

- [ ] **Step 2: Add dispatch for "persistence" experiment in `run_backtest_for_experiment()`**

Find the `run_backtest_for_experiment()` function. Add this check at the very beginning (before loading the experiment config):

```python
    # Handle persistence baseline (no config file needed)
    if experiment_id == "persistence":
        return _run_persistence_backtest(
            country_code, forecast_type, week_start, week_end,
        )
```

- [ ] **Step 3: Verify syntax**

```bash
python -c "import ast; ast.parse(open('scripts/compare_experiments.py').read()); print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add scripts/compare_experiments.py
git commit -m "feat: add persistence baseline to experiment comparison script"
```

---

### Task 7: Create experiment configs V010 and V011

**Files:**
- Create: `C:\Code\energy-forecast\experiments\V010\config.json`
- Create: `C:\Code\energy-forecast\experiments\V011\config.json`
- Modify: `C:\Code\energy-forecast\experiments\registry.json`

- [ ] **Step 1: Create V010 directory and config**

```bash
mkdir -p experiments/V010
```

Write `experiments/V010/config.json`:

```json
{
  "id": "V010",
  "model": {
    "type": "chronos-2",
    "base": "amazon/chronos-2",
    "context_length": 672,
    "prediction_length": 24,
    "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  },
  "training": {
    "fine_tune": false,
    "fine_tune_steps": 0
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

- [ ] **Step 2: Create V011 directory and config**

```bash
mkdir -p experiments/V011
```

Write `experiments/V011/config.json`:

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

- [ ] **Step 3: Update registry.json**

Add V010 and V011 entries to the `"experiments"` array in `experiments/registry.json`:

```json
    {
      "id": "V010",
      "name": "Chronos-2 net position zero-shot",
      "model": "chronos-2",
      "status": "pending",
      "created_at": "2026-04-02",
      "parent": null,
      "description": "Pretrained Chronos-2 for net position, no fine-tuning. Weather + flows + TSO covariates."
    },
    {
      "id": "V011",
      "name": "Chronos-2 net position fine-tuned",
      "model": "chronos-2",
      "status": "pending",
      "created_at": "2026-04-02",
      "parent": "V010",
      "description": "Fine-tuned 5000 steps, cosine LR. Dedicated net position model with crossborder flow covariates."
    }
```

- [ ] **Step 4: Validate JSON files**

```bash
python -c "
import json
for f in ['experiments/registry.json', 'experiments/V010/config.json', 'experiments/V011/config.json']:
    json.load(open(f)); print(f'  OK: {f}')
"
```

- [ ] **Step 5: Commit**

```bash
git add experiments/
git commit -m "feat: add V010 (zero-shot) and V011 (fine-tuned) net position experiment configs"
```

---

### Task 8: Update CLAUDE.md documentation

**Files:**
- Modify: `C:\Code\energy-forecast\CLAUDE.md`

- [ ] **Step 1: Add net_position to forecast types documentation**

Find the `**Forecast Types:**` section at the top and add:

```
- **Net Position** - Cross-border import/export balance (MW) [Chronos-2 only]
```

- [ ] **Step 2: Add net position commands to the Chronos-2 section**

Find the `### Chronos-2 (ported from netpredict2)` section and add these examples:

```bash
# Net position forecasting (V010+)
python scripts/forecast_chronos2.py --experiment V010 --countries DE --types net_position --target-date 2024-01-15
python scripts/compare_experiments.py --experiments persistence,V010 --weeks W01 --countries DE --types net_position
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add net position forecasting to CLAUDE.md"
```

---

### Task 9: End-to-end verification

This task verifies the pipeline works with whatever data we have in the database.

- [ ] **Step 1: Verify config registration**

```bash
python -c "
import config
assert 'net_position' in config.FORECAST_TYPES
assert config.FORECAST_HORIZONS.get('net_position') == [1, 2]
assert config.WEATHER_FEATURES.get('net_position') is not None
print('Config: OK')
"
```

- [ ] **Step 2: Verify covariate mapper**

```bash
python -c "
from src.chronos2.covariate_mapper import build_covariate_map
cov = build_covariate_map('DE', 'net_position', include_neighbors=True)
s0 = [c['cov_name'] for c in cov['suffix_0']]
s1 = [c['cov_name'] for c in cov['suffix_1']]
assert any('weather' in n for n in s0), 'Missing weather covariates'
assert any('cal__' in n for n in s0), 'Missing calendar covariates'
assert any('crossborder' in n for n in s1), 'Missing crossborder flows'
assert any('da__price' in n for n in s1), 'Missing DA price'
print(f'Covariate mapper: OK ({len(s0)} suffix-0, {len(s1)} suffix-1)')
"
```

- [ ] **Step 3: Test input builder with actual data (if available)**

```bash
python -c "
from src.chronos2.input_builder import InputBuilder
ib = InputBuilder()
try:
    inp = ib.build_training_input('FR', 'net_position', '2023-01-01', '2023-02-01')
    if inp:
        print(f'Training input: {len(inp[\"target\"])} points, {len(inp.get(\"past_covariates\", {}))} past covs')
    else:
        print('Training input: None (insufficient data)')
except Exception as e:
    print(f'Training input: error - {e}')
"
```

- [ ] **Step 4: Verify experiment configs load**

```bash
python -c "
import json
v010 = json.load(open('experiments/V010/config.json'))
assert v010['forecast_types'] == ['net_position']
print(f'V010: {v010[\"model\"][\"type\"]}, fine_tune={v010[\"training\"][\"fine_tune\"]}')

v011 = json.load(open('experiments/V011/config.json'))
assert v011['training']['fine_tune'] == True
print(f'V011: {v011[\"model\"][\"type\"]}, steps={v011[\"training\"][\"fine_tune_steps\"]}')
print('Experiments: OK')
"
```

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "feat: net position forecasting with Chronos-2 — Phase 3 complete"
```
