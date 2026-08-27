> **Archived from `CLAUDE.md` on 2026-08-27** (companion to the ABL-536
> energy-dashboard-frontend trim). Historical narrative, incident forensics
> and dated measurements; `file:line` references are frozen as of the archive
> date. The durable rules distilled from this material live in the repo-root
> `CLAUDE.md`; where they conflict, the root file wins.
# Key Commands

## Key Commands

### Training

Every command below needs a sidecar target. `scripts/train.py` exits `2` without
writing anything when neither `FORECAST_OUTPUT_DB` nor `--sidecar-db` resolves
(ABL-346) — see the Database section for why the fallthrough it replaces aimed at
the replica.

```bash
# Train all models (includes load, price, renewable, and individual renewable types)
python scripts/train.py --countries all --types all

# Explicit sidecar, no environment dependency
python scripts/train.py --countries DE --types renewable --sidecar-db C:\Code\able\data\forecasts_local.db

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

### Solar is clamped to physical reality on the way out (ABL-337)

`save_forecasts()` (`src/db.py`) is the choke point every serving write goes
through, and solar rows do not pass it unchanged. `src/solar_clamp.py` zeroes any
hour whose sun stays below `NIGHT_ELEVATION_THRESHOLD_DEG` (-8 deg, geometric)
for the whole hour, and floors the rest at zero. `renewable_type='solar'` only,
**new rows only** — stored history is never rewritten, and no `UPDATE` is issued,
so the vintage archive stays a faithful record of what the models said.

**The night zero is per-country, and ES is exempt (ABL-425).** The premise "a
solar fleet cannot generate at night" is false for Spain: it runs ~2.3 GW of
concentrated solar power with molten-salt storage, and ABL-411 checked Red
Eléctrica's own `solar fotovoltaica` / `solar térmica` split against the replica
over 3,196 night hours — the two account for **98.55%** of the MW we book for ES
when the sun is down, at a **263.5 MW** mean night level, 80.1% of it CSP. So
the physical fact is registered per country in
`solar_geometry.NIGHT_GENERATION_POSSIBLE` and the clamp reads it. Three things
follow and all three are load-bearing:

- **There is no default.** A country reaching the clamp undeclared raises
  `UndeclaredNightGenerationError` and the save writes nothing. The silent
  direction is the destructive one — an unregistered ES-like country would
  inherit "cannot generate at night" and have real MW deleted, logged as a
  correction. Add a country to `config.SUPPORTED_COUNTRIES` and you add it to
  that table in the same commit; `tests/test_night_generation_registration.py`
  fails otherwise.
- **The `max(0, prediction)` floor is not per-country** — but not because
  negative solar is impossible. It is not: `energy_generation` is the A75
  document *net of consumption*, and NL books a structural overnight floor of
  about −1.1 MW (100% of instants 20Z–02Z, min −1.62 MW — the deepest anywhere
  in the fleet over ABL-348's registered window). The floor erases that
  reported MW, and is justified by the size of the excursion, not by physics.
  `src/solar_geometry.py`'s "Why the non-negativity floor is fleet-wide"
  carries the measurement, the window the bound holds over, the five
  full-history instants that exceed it, and the tripwire if NL is ever served.
  Two premises of the same class have now failed here: ES generates when the
  code said it could not, NL books a negative when the code said it could not.
  Both were physical absolutes the A75 semantics never supported.
- **`energy_renewable` cannot arbitrate the sign of solar, so do not use it to
  check the floor.** Over ABL-348's gate window it is the *zero-clipped copy*
  of `energy_generation`: `ren == max(0, gen)` to 1e-9 at 100.0% of instants in
  28 of 32 countries and 99.0% for NL, NL flipping into that regime between
  2026-07-01 (41.7%) and 2026-07-02 (99.0%) — which is the same fact `db.py`
  records as "the gate truth is byte-identical between the two tables for 9 of
  10 pairs" (ABL-321). So `ren − gen` in that window is `max(0, −gen)`, the
  floor's own correction, and reads as a clean non-negative "Actual Consumption
  series" no matter what the data says. Outside the window it is not a
  consumption series either — it goes to −185.84 MW at NL midday over fit+gate,
  with only 305 of 8,668 excursions attributable to ABL-188 zero-fill. This
  retired an ABL-425 finding that had been reported from both sides; see
  `solar_geometry.py` for the full reproduction.
- **The registered thing is the *fact*, not the policy.** The clamp and
  ABL-376's `exclude_impossible_night` fit rule both read this one table and
  apply their own policy on top, so they cannot come to disagree about which
  hours are dark-but-real. A single shared *value* would not work: BG's
  overnight floor is genuine contamination (clamp on) yet ABL-403 measured the
  fit-side rule costing it 1.4-1.9pp of gate WAPE (rule off).

This is a guard, not a fix. ABL-335 measured what the models emit: 22,718 of
131,356 stored solar rows negative, DE holding a 155-268 MW floor straight
through local midnight. The fit defect underneath is ABL-338's. **So the clamp
reports itself**: every run appends one row per country and model to
`forecast_clamp_log`, in the same database the clamped rows went into —

```sql
SELECT clamped_at, country_code, model_name,
       night_generation_possible, night_mask_applied,
       night_hours, hours_zeroed_night,
       hours_raised_floor, mw_removed_night, mw_removed_total
FROM forecast_clamp_log ORDER BY clamped_at DESC;
```

A retrain that fixes the fit drives `hours_zeroed_night` and `mw_removed_total`
toward zero; the clamp going quiet is the measurement, and the clamp staying busy
after a retrain means the retrain did not work.

Read the first two columns before the counts. An exempt country's
`hours_zeroed_night = 0` means "nothing may be zeroed here", not "the fit is
clean" — the two are otherwise indistinguishable, which is why ABL-425 added
them rather than letting ES go quiet in the instrument. `night_mask_applied` is
also False for a country with no representative point, and
`night_generation_possible` is what tells those two states apart. Rows written
before ABL-425 carry `NULL` on all three: that run predates the exemption and
night-zeroed every country unconditionally.

Sun elevation comes from `src/solar_geometry.py` — one capacity-weighted
representative point per country, taken from `weather_location`. Import it; do
not write a second copy (a training-side solar-geometry feature must use the same
number the serving clamp uses). The -8 deg threshold was chosen by measurement,
not convention: at -6 the mask would zero hours that recorded up to 18.7 MW of
real DE generation, at -8 up to 3.6 MW, and below -10 it stops covering 02:00 UTC
in August, which is one of the hours the defect appears in. Re-measure before
changing it:

```bash
python scripts/abl335_solar_night_probe.py --check-actuals     # threshold vs actuals
python scripts/abl335_solar_night_probe.py --stored-forecasts  # negative/night rows
```

Caveat worth knowing before trusting that check: FR's `energy_renewable.solar_mw`
itself carries 137-440 MW at sun elevations down to -65 deg on 337 distinct days,
so FR's "the mask would zero a real actual" count is dominated by an actuals
defect rather than by the threshold.

The clamp sits in `save_forecasts()`, so it covers every serving writer that
goes through it, by construction rather than by each caller remembering to
clamp. Two writers import it: `scripts/forecast_daily.py` and
`src/tso_correction_forecaster.py:39`.

The second one could not run at all until ABL-354. `forecast_daily.py` launched
it as a subprocess **by file path**, and ABL-340 moved it to relative imports,
so it died at import with `attempted relative import with no known parent
package` — every BE solar / wind row from the `tso-correction` runner failed
that way, while the run summary still reported `[DONE]`. It now launches as
`-m src.tso_correction_forecaster` (`build_runner_command`,
`scripts/forecast_daily.py:189`), so it reaches `save_forecasts()` and inherits
the clamp by construction; nothing about the clamp had to change for it.

### Tests

```bash
# The whole suite — run it from the repo root, under .venv
.venv\Scripts\python.exe -m pytest -q
```

`pytest.ini` pins `testpaths = tests`, so the bare command above and
`python -m pytest tests/` are the same run. That pin exists because pytest
otherwise walks the entire tree and collects anything named `test_*.py` —
including untracked scratch files, which made the bare command fail collection
for months (ABL-336). Files under `scripts/` that probe or benchmark something
are named `probe_*.py`, not `test_*.py`, for the same reason: they execute
training at import time and must never be collected.

If you add tests outside `tests/`, add that directory to `testpaths` — the bare
command will not find them otherwise.
