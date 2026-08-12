# ABL-191 — extend the serve-faithful feature builder to solar

**Disposition:** implemented and tested; no deploy, no production DB write,
no registry/serving-config change, no model promotion, no retraining
performed. All reads used the rail interpreter (`.venv`, Python 3.14.3)
against local joblib artifacts only — no database write of any kind.

## What changed

`src/wind_features.py` (ABL-183's shared builder) now also serves solar:

- `_WEATHER_RAW_COLUMNS` gained `shortwave_radiation_wm2`,
  `direct_radiation_wm2`, `diffuse_radiation_wm2` alongside wind's
  `temperature_2m_k`/wind-speed columns — solar's `config.WEATHER_FEATURES`
  entry.
- `SUPPORTED_FORECAST_TYPES` gained `"solar"`. `Forecaster.predict_d2`
  (`src/forecaster.py:624`) branches on `self.forecast_type in
  SERVE_FAITHFUL_FORECAST_TYPES` — imported directly from
  `SUPPORTED_FORECAST_TYPES` — so this one-line addition is the entire
  serving change; `_predict_d2_serve_faithful` is already generic across
  `forecast_type` and needed no edits.
- No other code path changed. `RenewableFeatureBuilder`'s lag/rolling/
  calendar logic was already generic; nothing forked.

## Artifact shape, confirmed against the real frozen models (2026-08-11)

All four solar artifacts (`models/{AT,BE,DE,FR}/solar/model.joblib`) report
identical `feature_columns`, 24 names in the same order:

```
hour, day_of_week, month, is_weekend, hour_sin, hour_cos, day_sin, day_cos,
month_sin, month_cos, target_value_lag_1d, target_value_lag_7d,
target_value_lag_14d, target_value_roll_24h_{mean,std,min,max},
target_value_roll_168h_{mean,std,min,max}, shortwave_radiation_wm2,
direct_radiation_wm2, diffuse_radiation_wm2, temperature_c
```

Same shape as wind's 24 (10 calendar + 3 lags + 8 rolling + temperature_c),
with the two wind-speed columns swapped for the three radiation columns —
exactly what the ABL-183 code comment predicted. Pinned in
`tests/test_solar_features.py::REAL_ARTIFACT_FEATURE_COLUMNS` and
`test_to_vector_produces_exactly_the_real_artifacts_24_columns_in_order`.

## The two things the issue asked to confirm, not assume

**`temperature_c` under the shared builder.** ABL-185 finding #3: solar's
generic weather-inference block already recomputed `temperature_c` from
forecast temperature pre-ABL-183, unlike wind (whose weather allow-list has
no temperature entry, so it was never overridden). Under this builder,
`_weather_features` resolves `temperature_c` unconditionally for every
`forecast_type` — not gated by `config.WEATHER_FEATURES` — so the same code
path already serving wind's `temperature_c` serves solar's. Confirmed, not
assumed:
`tests/test_solar_features.py::test_temperature_c_is_always_populated_from_the_same_weather_row`.

**The ABL-188 DE-solar-zero invariant.** `_load_actuals_series` calls
`load_renewable_type_data`, which already runs `exclude_suspect_constant_runs`
before this builder ever sees the data — the same call wind's actuals go
through. No solar-specific code was needed for the 6,408-quarter-hour DE
zero-fill window to be excluded from a lag or rolling-window feature.
Confirmed with a dedicated fixture that reproduces the shape of that defect
(a 30-hour bit-identical run inside a lag's lookback window) and asserts the
resulting feature is `NaN`, not `0.0`:
`tests/test_solar_features.py::test_a_suspect_constant_actuals_run_is_excluded_from_lags_and_rolling`.

## Tests

New: `tests/test_solar_features.py` (16 tests) mirrors
`tests/test_wind_features.py`'s golden-vector standard — pins *meaning*
(which `as_of` each lag/rolling stat resolves against, weather publication
cutoff honoured, no post-`observation_as_of` leakage) via a poisoned-future
fixture, not just vector shape. Two solar-specific wiring tests were added
to `tests/test_forecaster_wind_serving.py` exercising the real
`Forecaster.predict_d2` entrypoint for `forecast_type="solar"` end to end,
mirroring the existing wind wiring tests.

Existing fixtures in `tests/test_wind_features.py` and
`tests/test_forecaster_wind_serving.py` needed their in-memory `weather_data`
schema extended with the three new radiation columns (`_WEATHER_RAW_COLUMNS`
is fetched for every forecast_type, wind included, since the query has no
per-type branch) — otherwise wind's own tests would fail with "no such
column" against the widened SELECT. No behavioral change to wind; confirmed
by full-suite pass below.

```
$ .venv/Scripts/python.exe -m pytest tests/ -q
213 passed in 11.25s
```

(Pre-change baseline was ~166 tests across the two touched files plus the
rest of the suite untouched; the delta is entirely new solar/wiring tests
plus the four wind fixtures widened for schema, none removed.)

## Explicitly out of scope, unchanged from ABL-183

No retraining, no backtest, no gate read, no registry/promotion action, no
deploy. Production solar WAPE (48.4% pooled, per
`reports/abl_185_solar_diagnosis.md`) is not expected to move from this
change alone — ABL-185 found the DE artifact's level failure (62.4% country
WAPE, -61.4% bias, fit on 3,208 rows with a zero-filled prehistory) as the
dominant cause, which this issue does not touch. A fresh serve-faithful
solar retrain, evaluated against a pre-registered gate per ABL-195's
template, is the Forecasting Scientist's next step and will be filed as its
own issue once this lands.

## Owner / next step

Founding Engineer (this issue) → branch
`fix/abl-191-serve-faithful-solar-features`, pushed for review, same pattern
as ABL-183/ABL-195. Forecasting Scientist retrain issue to follow after
merge.
