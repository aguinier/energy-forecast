# ABL-332 — the renewable feature builder was hourly over a series that is not

**Status: implemented on `ABL-332-renewable-resolution`, deliberately not landed.**
Deliverable 4 of the issue says to stop if the currently-serving DE pairs change.
They do, materially. See §5.

Database: `C:\Code\able\data\energy_dashboard.db` (read-only; the `.env`
`ENERGY_DB_PATH` points at a path that does not exist and was overridden
explicitly). Measurements 2026-08-12.
Regenerate with `python scripts/audit_renewable_resolution.py`.

---

## 1. The finding is confirmed, and it is larger than the issue states

The issue described a DE/NL problem in `energy_generation`, gating the ABL-322
offshore pilot. Two things are worse than that.

**It is not two countries.** Over `config.SUPPORTED_COUNTRIES`, counting only
rows whose target column is non-NULL:

| source table | countries with sub-hourly rows | countries hourly throughout |
|---|---|---|
| `energy_renewable` | **22 of 24** | BE, BG, CH, LV, PT |
| `energy_generation` | **20 of 24** | BE, BG, CH, LV, PT (+ EE/GR/SI/SK mostly hourly) |

**It is not a future problem.** `RENEWABLE_TYPE_SOURCE_TABLE` is still
`energy_renewable` (`src/db.py:350`) — ABL-321's switch to `energy_generation`
was withheld. `energy_renewable` is quarter-hourly too, for a *larger* share of
its rows. The defect is live in production today, not waiting on ABL-321.

Resolution is also not a stable property of a country. Most of these have an
hourly backbone for the early years and switch to quarter-hourly partway
through, so the same country is both. `DE` in `energy_generation`: 196,756 rows,
49,189 on `:00` — 75.0% unreachable by a `floor("h")` lookup. `SI`: 11.0%.

Per-country, `energy_renewable`, solar (`%drop` = rows an exact hourly lookup
can never reach):

| CC | rows | on `:00` | %drop | | CC | rows | on `:00` | %drop |
|---|---|---|---|---|---|---|---|---|
| AT | 27,731 | 6,933 | 75.0% | | IT | 30,641 | 7,663 | 75.0% |
| BE | 23,157 | 23,157 | 0.0% | | LT | 26,644 | 6,661 | 75.0% |
| BG | 49,318 | 49,318 | 0.0% | | LV | 6,700 | 6,700 | 0.0% |
| CH | 11,086 | 11,086 | 0.0% | | NL | 27,485 | 6,873 | 75.0% |
| CZ | 26,724 | 6,681 | 75.0% | | NO | 27,413 | 6,854 | 75.0% |
| DE | 33,503 | 8,377 | 75.0% | | PL | 30,856 | 7,715 | 75.0% |
| EE | 23,587 | 6,501 | 72.4% | | PT | 6,896 | 6,896 | 0.0% |
| ES | 27,631 | 6,907 | 75.0% | | RO | 29,975 | 7,494 | 75.0% |
| FI | 30,819 | 7,704 | 75.0% | | SE | 24,766 | 7,648 | 69.1% |
| FR | 76,999 | 32,171 | 58.2% | | SI | 12,608 | 6,514 | 48.3% |
| GR | 13,870 | 6,721 | 51.5% | | SK | 14,097 | 6,504 | 53.9% |
| HR | 26,921 | 6,731 | 75.0% | | HU | 26,055 | 6,515 | 75.0% |

`wind_onshore` and `wind_offshore` match within a few hundred rows.
The full table for both sources and all three streams is in the script output.

## 2. What the builder actually did — reproduced, not read

`scripts/audit_renewable_resolution.py` builds one target day of features twice
through the real `RenewableFeatureBuilder`: arm A as serving did it, arm B with
the actuals replaced by their hourly means. DE, `energy_renewable`, target day
2026-08-01, D+2 shape:

```
DE/solar         3,072 native rows in the 31d span -> 768 hourly; minutes [0,15,30,45]
  target_value_lag_1d          max|Δ| = 4,816.28 MW   max 14,054.6%   17/24 hours
  target_value_lag_7d          max|Δ| = 5,025.64 MW   max  9,701.8%   19/24 hours
  target_value_lag_14d         max|Δ| = 3,518.51 MW   max  4,734.1%   19/24 hours
  target_value_roll_24h_mean   max|Δ| =   543.61 MW                   24/24 hours
  target_value_roll_24h_max    max|Δ| =   318.89 MW                   24/24 hours
  (+ 4 more rolling stats)

DE/wind_onshore  same span
  target_value_lag_1d          max|Δ| = 1,381.89 MW   max     21.1%   24/24 hours
  target_value_roll_24h_max    max|Δ| = 1,142.50 MW   max      6.7%   24/24 hours
  (+ 9 more)
```

The finding reproduces. It does not merely fail to raise — it produced a
complete, plausible, fully-populated 24-name feature row from a quarter of the
data, every run, for every affected country.

**Two distinct defects, not one.** The issue named the lags. The rolling windows
have the opposite bug:

- **Lags** (`_lookup_hour`, `wind_features.py:308`) floor to the hour, so they
  read the `:00` sub-sample and discard `:15`/`:30`/`:45`.
- **Rolling windows** (`_rolling_features`) slice the *raw* index by time, so
  they averaged all ~96 samples a day — no subsampling, a different definition
  again. That is why `roll_*_max`/`_min` move most: sub-hourly extremes are more
  extreme than hourly ones.

One feature row, three resolutions.

## 3. The decisive fact: training was already hourly

This is what settles the design question, and neither the issue nor ABL-321
mentions it.

- `db.load_training_data` (`src/db.py:798`) does `resample('h').mean()` on the
  energy frame before features are built.
- `features.py:create_lag_features` then does `df[target].shift(days * 24)` — a
  **positional** shift, which only means one day on an hourly frame.
  `create_rolling_features` likewise uses `rolling(window=24)` rows.

So every frozen artifact now serving was **fitted on hourly means**, and serving
was feeding it `:00` sub-samples under the same column names. This is not "which
definition should we adopt"; it is a train/serve skew, and serving is the arm
that was wrong.

## 4. Decision: aggregate at the source read

**Chosen: aggregate to the hourly mean inside `load_renewable_type_data`, at
`aggregate_renewable_to_hourly` (`src/db.py:377`).** Rejected: making the
builder resolution-aware.

Why:

1. **It is the only place both consumers pass through.** Training and serving
   share exactly one read. Fixing the builder would leave the loader still
   emitting mixed resolutions to every other caller.
2. **It converges on the definition the artifacts were fitted with**, rather
   than inventing a third. `load_training_data`'s own resample becomes a no-op
   for the renewable types, so the training path is byte-unchanged — the fix
   moves only the arm that was wrong.
3. **It is uniform by construction.** There is no per-country branch and no
   list to maintain: an hourly country's rows floor to themselves and the
   function returns early. That matters because resolution changes *within* a
   country's history — any per-country table would already be stale.

Rule: the hourly mean of whatever sub-samples the hour has. Deliberate details:

- **`mean`, never `sum`.** `groupby().sum()` collapses an all-NaN hour to `0.0`
  without `min_count=1`. An hour with no live sub-sample stays `NaN`.
  Pinned by `test_an_all_null_hour_stays_nan_and_does_not_become_zero`.
- **After `exclude_suspect_constant_runs`, not before.** That guard infers a
  series' cadence from its own median step and measures runs in hours (ABL-188's
  real instance is 6,408 *quarter*-hours). Averaging first would blur a
  zero-fill run's edges into non-constant values and hide them from it.
- **A partial hour is the mean of the sub-samples present**, counted and logged.
  This is precisely what `resample('h').mean()` has always given training, so it
  introduces no new behaviour on that path. See §7 for the alternative.

## 5. The quiet failure is now loud

Two guards, both uniform:

1. `aggregate_renewable_to_hourly` emits a `logger.warning` naming the row
   counts, the inferred cadence and the partial-hour count every time it
   actually aggregates. An already-hourly series returns early and says nothing.
2. `src/wind_features.py` raises `SubHourlyResolutionError` (`:142`) from
   `_assert_hourly` (`:284`) if it is ever handed an off-the-hour series again —
   naming how many observations are off the hour and the first offender.
   Subsampling is not an acceptable degraded mode.

## 6. STOP — this changes the currently-serving DE pairs

Per deliverable 4 of the issue. **It does, and by a lot.**

`:00` sub-sample vs its hour's mean, 2026-01-01 → 2026-08-12, `energy_renewable`
(the serving source). This is the size of the correction to every lag feature:

| pair | hours | median abs Δ | p90 abs Δ | max abs Δ | median rel | mean bias |
|---|---|---|---|---|---|---|
| **DE solar** (serving) | 5,339 | **373.6 MW** | 3,211.3 MW | 5,500.1 MW | 19.06% | +3.05 MW |
| **DE wind_onshore** (serving) | 5,339 | **236.6 MW** | 803.9 MW | 4,179.6 MW | 2.81% | +1.22 MW |
| DE wind_offshore | 5,339 | 78.1 MW | 271.8 MW | 1,250.2 MW | 3.88% | +1.06 MW |
| NL wind_offshore (ABL-322) | 5,338 | 47.2 MW | 190.9 MW | 959.3 MW | 4.79% | +0.29 MW |
| NL solar | 5,338 | 0.8 MW | 18.0 MW | 124.4 MW | 20.41% | −0.06 MW |

Note the shape of the DE solar row: a **mean bias of +3 MW** against a **median
absolute error of 374 MW**. The subsampling is near-unbiased in aggregate —
sunrise hours read low, sunset hours read high, and they cancel — while being
wrong in nearly every individual hour. That is exactly why nothing caught it:
any aggregate check passes.

`energy_generation` gives the same picture (DE solar median 368.5 MW), so the
conclusion does not depend on which table ABL-321 eventually selects.

**So this is a live behaviour change and I have not landed it.** The branch is
`ABL-332-renewable-resolution`, tests green, not merged. What is needed before
it lands is a decision, not more code:

- The change is a *correction toward* what the artifacts were trained on, so
  leaving it out is not the safe option — it is the option that keeps a known
  train/serve skew in production.
- But it will move served DE solar and DE wind_onshore numbers, and this issue
  is explicitly scoped "no retraining, no promotion". Quantifying the effect on
  the *forecasts* (rather than the features) means a backtest, which is the
  Forecasting Scientist's step and a separate issue.

## 7. Open, deliberately not decided here

- **Partial hours.** An hour with 1 of 4 sub-samples present becomes the mean of
  the 1 — a real measurement, weaker support, labelled as the hour. A stricter
  rule (require full support, else NaN) is defensible and would change training
  data for every sub-hourly country. That is a larger decision than this issue,
  so the existing training behaviour was preserved and the partial-hour count is
  logged instead.
- **ABL-321's recorded A/B numbers were produced by the pre-ABL-332 builder**
  (`experiments/ABL321/results_w*.json`). Its truth series are read straight
  from the tables and are unaffected, but both prediction arms subsampled. The
  §6 verdict there — AT solar +4.3%, DE wind_onshore +3.6%, BE wind_onshore
  +2.7% — is therefore a comparison of two subsampled arms. Whether the withheld
  switch looks different once both arms are hourly is unknown and worth a re-run.
- **`load`, `price` and `renewable`** still read their own tables at native
  resolution and are resampled by `load_training_data`. They were not in scope
  here; whether their serving paths have the same skew was not checked.

## 8. Verification

```
$ python -m pytest tests/ -q
270 passed in 13.63s          (baseline on origin/main: 256 passed)
```

14 new tests in `tests/test_renewable_resolution.py`. Post-fix, the reproduction
in §2 reports `minutes [0]` and **0 features differing** for DE solar,
DE wind_onshore and NL wind_offshore — the two arms are now the same series.
