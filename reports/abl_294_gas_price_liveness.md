# ABL-294 — Is `gas_prices` a live ML feature?

**Author:** Forecasting Scientist
**Date:** 2026-08-12
**Origin:** incidental finding from the ABL-286 provenance audit, routed by the Founding Engineer
**Verdict:** **No — and it is not worth reviving on current evidence.**

---

## 1. Answer in three lines

1. `gas_prices` is **not** a feature of any model artifact in this module — 247/247 artifacts,
   zero gas features. **There is no model-input staleness problem.** Nothing has been training
   or predicting against the frozen feed.
2. It is also **not obviously worth reviving**: against the feature set the price models already
   carry, TTF adds **no measurable out-of-sample accuracy** (pooled +1.34% RMSE, 95% CI
   ±1.61 — includes zero, helps in 9/20 folds).
3. Recommend: **do not revive the feed, do not build a freshness signal for it, do not delete
   the table.** Revisit only against a specific price-model gap (see §6).

---

## 2. Table state — reproduces the FE's measurement exactly

Measured on the **live replica** `C:\Code\able\data\energy_dashboard.db`
(9.4 GB, refreshed by `able-db-sync` 2026-08-12 07:34), opened read-only
(`file:...?mode=ro`, `uri=True`). *Not* the 3.0 GB `energy-data-gathering` decoy.

| property | value |
|---|---|
| rows | 788 |
| `date` range | 2023-01-03 .. 2026-02-20 |
| `created_at` | 2026-02-21 18:49:53 — **`COUNT(DISTINCT created_at) = 1`** |
| `source` | `ttf`, all 788 rows (no second source) |
| level range | 22.9 .. 74.3 EUR/MWh |
| staleness | **173 days** as of 2026-08-12 |

Corroborating the FE's "no freshness signal anywhere":
`data_ingestion_log.pipeline_type` holds 10 values —
`crossborder_flows, load, load_forecast_day_ahead, load_forecast_week_ahead, net_position,
price, renewable, weather_forecast, weather_update, wind_solar_forecast` —
**none of them gas.** `gas_prices` is also the only gas/commodity/carbon table in the schema.

So: one manual one-shot load in February, never wired to a pipeline, never updated.

---

## 3. Liveness — conclusive, 247/247 artifacts

The question "is it a live feature" is only answerable from **artifacts**, not from source, since a
deployed model carries its own feature list. All of `models/` and `experiments/`:

| check | result |
|---|---|
| model artifacts scanned (`*.joblib`, `*.pkl`) | **247** |
| exposed a feature list directly | 243 |
| remaining 4 resolved by object-graph walk | 4 → **247/247 accounted for** |
| artifacts with **any** gas/commodity feature | **0** |
| distinct feature names across all artifacts | 205 — none gas-related |
| JSON configs / metadata mentioning gas | **0** of 44 |
| text references in `src/`, `scripts/`, `experiments/`, `config.py` | **0** |

Match keys: `gas, ttf, commod, fuel, coal, carbon, co2, eua, api2`.

The 4 initially-unreadable artifacts (`models/BE/price_cascade`, `models/tso_correction/BE/{solar,
wind_offshore,wind_onshore}`) are nested dicts, not flat estimators; walking the object graph
recovered their `feature_cols` / `*_feature_columns` lists. All clean.

**One false positive worth recording** so the next auditor does not re-derive it: a binary grep
hits `experiments/ABL195/artifacts/FR/wind_offshore/model.joblib`. That is the byte sequence
`ttf` occurring 6× inside the compressed pickle stream — **not** a feature name. That artifact's
actual feature list was read cleanly and contains no gas feature.

**Timing note.** `models/BE/price_cascade` carries `model_version: 20260221_201435` — retrained
**85 minutes after** the 18:49:53 gas load, and gas still did not enter its 32-feature list. The
most likely reading is that gas was trialled that evening and dropped. §5 is consistent with why.

---

## 4. Is the feed worth reviving? Two screens that disagree — the second is the right one

### Screen A — weak baseline (the misleading one)

Daily-mean day-ahead price, OLS, baseline = price lags 3d & 7d only, 6 expanding-origin folds
per country, 5 countries, window 2023-01-03..2026-02-20.

| | pooled result |
|---|---|
| mean ΔRMSE | **−4.65%** (gas helps) |
| median | −4.28% |
| folds helped | 24/30 |
| corr(gas, daily price) | 0.44 – 0.56 |

On this evidence gas looks clearly valuable. **It is not.** A 2-lag baseline is a straw man: it
has no weather, no load forecast, no rolling price level. Gas is proxying for information the
real models already hold.

### Screen B — realistic baseline (the decisive one)

Hourly, LightGBM (300 trees, lr 0.05, 63 leaves), 4 rolling-origin folds per country, refit each
fold. Baseline = **28 features** mirroring the deployed price stack:

- calendar + cyclical: `hour, dow, month, is_weekend` + sin/cos of each
- **serve-faithful** price history — nothing newer than 72h before target:
  `p_lag{72,96,168,336}` and 8 rolling stats (24h/168h × mean/std/min/max) computed off a
  72h-lagged base
- weather: `t2m`, `wind_speed_100m`, `shortwave_radiation`
- TSO day-ahead load forecast

Gas added as `gas[t−3d]` (business-day forward-fill, strictly past-dated for a D+2 forecast) plus
its 7-day rolling mean.

| country | n (hourly) | base RMSE | +gas RMSE | mean ΔRMSE% | folds helped |
|---|---:|---:|---:|---:|---:|
| BE | 26,517 | 30.06 | 29.88 | **−0.31** | 3/4 |
| NL | 25,423 | 33.91 | 34.35 | **+1.55** | 1/4 |
| FR | 25,652 | 27.76 | 27.95 | **+0.88** | 2/4 |
| DE | 24,018 | 35.99 | 36.23 | **+1.84** | 2/4 |
| AT | 24,187 | 32.76 | 33.53 | **+2.76** | 1/4 |

**Pooled over 20 country-folds: mean +1.34%, median +1.08%, helped 9/20.
95% CI +1.34 ± 1.61 — includes zero.** (Folds share training data, so that CI is *optimistic*;
it fails to exclude zero even so.) Negative = gas helps; the point estimate is mildly **positive**,
i.e. gas is if anything a small net dilution.

Out-of-sample, against features already in the stack, **TTF gas price buys nothing.**

---

## 5. Why importance does not rescue it

Fit on the full window with gas included, gas takes a large share of splits:

| country | gas share of split importance | ranks (of 28 features) |
|---|---:|---|
| BE | **13.40%** | `gas_roll7` #2, `gas` #4 |
| DE | **11.23%** | `gas_roll7` #3, `gas` #4 |

The model uses gas heavily and still does not get better. That is the textbook signature of a
**smooth, slow-moving, high-cardinality variable**: trees find it convenient to split on, and it
substitutes for price-level information already carried by `p_roll168_*`. Split importance
measures *usage*, not *value*.

**Flagging this explicitly** because an importance table is the most likely thing to be cited later
as evidence that gas matters. It is not evidence. Only §4 Screen B is.

---

## 6. Caveats — what this does *not* establish

- **Window is the gas table's own coverage, 2023-01-03..2026-02-20.** It therefore **excludes the
  2022 crisis**, when gas–power coupling was strongest. This evidence says nothing about a
  crisis/regime-shift regime. Reviving gas *for that purpose* would need a backfill through
  2021–22, which the table does not have.
- Single model family (LightGBM) and single target (day-ahead price). Gas was not tested as a
  volatility/regime feature, nor against the other eight forecast types.
- Folds are expanding-origin and share training data — not independent.
- Screen B's baseline is a faithful *reconstruction* of the deployed price feature set, not the
  deployed cascade itself (`cascade_*` features need their own upstream models). The redundancy it
  measures would, if anything, be **stronger** against the true cascade, which carries more signal.

### Contamination status for this window

| issue | touches this window? |
|---|---|
| **ABL-71** prod ingest stale, fixes undeployed | **No** — window ends 2026-02-20, ~6 months before today; the stale recent tail is outside it |
| **ABL-67** fabricated `net_position` rows | **No** — `net_position` not used here |
| **ABL-111 / ABL-109** zero-as-missing actual-load rows | **No** — used `energy_load_forecast` (TSO day-ahead), not `energy_load` actuals |

Two further data observations, **not** handled as contamination:

- `energy_price` carries **0.45%–1.50% exact-zero hourly prices** in-window (BE 244/49,523;
  NL 195/31,835; FR 486/32,485; DE 174/29,789; AT 138/30,343). **Not filtered** — zero and
  negative prices are legitimate outcomes in European day-ahead markets.
- `energy_price.timestamp_utc` holds **mixed string formats and mixed timezone offsets**; parsing
  requires `format='mixed', utc=True`. Despite the column name, not all values are UTC-naive.
  Worth a look on the ingest side — raised separately to the FE.

---

## 7. Recommendation

| | |
|---|---|
| **Is it a live feature?** | No. 247/247 artifacts, zero gas features. No model-input staleness problem exists. |
| **Revive the feed?** | **No.** No measurable price-forecast gain over features already in the stack. |
| **Build the dashboard freshness signal the FE offered?** | **No** — it would monitor a feed nothing consumes. |
| **Delete the table?** | **No.** 788 rows is negligible to keep, and it is the only commodity series present. Deleting it destroys the only backfill seed if a regime feature is ever wanted. |
| **Owner needed for gas ingest?** | **No** — nothing depends on it. |

**Revisit if and only if** one of: a price-model experiment shows a residual gap plausibly
commodity-driven; a crisis/regime feature is wanted (requires 2021–22 backfill first); or gas
becomes an input to a forecast type other than price.

Recorded here so the next provenance audit does not re-derive any of it.

---

## Reproduction

Read-only against the live replica; no writes to any database. Scripts are analysis-only and were
run from scratch under `.venv\Scripts\python.exe` (Python 3.14.3 — the rail interpreter; a bare
`python` here would be the wrong xgboost).

```
ENERGY_DB_PATH=C:\Code\able\data\energy_dashboard.db   # read-only, file:...?mode=ro
```

Screens A and B, the artifact scan, and the importance fit are described in full above; each is a
standalone script of <80 lines with no dependency on module state.
