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
│   ├── data_quality.py     # Training-data invariants (ABL-188: rejects
│   │                       # suspect constant-value runs from energy_renewable)
│   ├── features.py         # Feature engineering (incl. holiday features)
│   ├── solar_geometry.py   # Sun elevation per (country, timestamp) — one
│   │                       # capacity-weighted point per country (ABL-337)
│   ├── solar_clamp.py      # Serving-path night mask + non-negativity floor
│   │                       # for solar, with per-run telemetry (ABL-337)
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
│   ├── abl335_solar_night_probe.py # Solar forecasts/actuals vs sun geometry
│   ├── forecast_chronos2.py  # Chronos-2 forecast generation
│   ├── compare_experiments.py # Cross-experiment backtest comparison
│   └── scheduler_setup.sh    # Cron setup
├── experiments/        # Versioned experiment configs (V001-Vnnn)
│   ├── registry.json       # Master experiment index
│   └── V00N/config.json    # Per-experiment configuration
├── models/             # Saved model artifacts
└── logs/               # Execution logs
```

### Importing this repo

`src/` is a package and is always imported as one. There is exactly one shape,
and every entry point and test uses it:

```python
sys.path.insert(0, str(Path(__file__).parent.parent))   # repo root, NOT src/
import config                                            # top-level, at the root
from src.db import load_training_data                    # package-qualified
```

Inside `src/`, siblings are imported **relatively** — `from .db import ...`,
`from ..features import ...`. Never `import db`.

This is not style. Putting `src/` on `sys.path` and importing flat gives a module
no parent package, so any relative import inside it raises `ImportError:
attempted relative import with no known parent package` — and where it does not
raise, it silently loads a *second* copy of the module under a second name, with
its own module-level state. `scripts/train.py` was dead by the first mechanism
from ABL-188 (`574eb80`, which added `src/db.py`'s `from .data_quality import
...`) until ABL-340 fixed it: seven months of a documented CLI that could not
run. Nine of 34 scripts were affected; five of them were the `test_*.py` probes
in `scripts/` that also broke bare `pytest` collection (ABL-336).

`tests/test_script_imports.py` holds the line — it executes the module-level
import block of every entry point in `scripts/` and the repo root, rejects any
flat sibling import inside `src/`, and (ABL-354) launches every
`config.MODEL_RUNNERS` entry with `--help` to prove it starts. A new script that
copies an old `sys.path.insert(..., 'src')` preamble fails there rather than
seven months later.

Two consequences worth knowing:

- Anything inside `src/` with a `__main__` — a demo like `src/features.py`, or a
  real entry point like `src/tso_correction_forecaster.py` — needs a parent
  package for its relative imports, so it runs as `python -m src.features`,
  never `python src/features.py`.

  **`config.MODEL_RUNNERS` launches two entry points that live inside `src/`,
  as subprocesses.** `scripts/forecast_daily.py:189` (`build_runner_command`) is
  the one place that builds that argv: a `script` under `src/` becomes
  `-m src.<module>` with `cwd` at the repo root; anything else stays a path.
  Before ABL-354 it was always a path, so `src/tso_correction_forecaster.py` —
  moved to relative imports by ABL-340, as the rule above requires — died at its
  import line on every run, and every BE solar / wind_onshore / wind_offshore
  forecast from the `tso-correction` runner was lost. **The job still exited
  `[DONE]`**: `run_external_model` records a dead subprocess as one failed
  *result*, and the summary line (`Total: 10, Success: 8, Failed: 2`) is the
  only trace. A runner that cannot start reads like a run that went fine, which
  is why the guard now launches them instead of trusting the summary.
- `src/evaluation.py` is dead code. `src/evaluation/` is a package and shadows
  it — `src.evaluation` always resolves to the directory. `src/__init__.py:44`
  already has its re-export commented out.

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

> **That claim is conditional, and the condition is unset by default.**
> `src/db.py:48` resolves a write target as `FORECAST_OUTPUT_DB or
> DATABASE_PATH`, and `config.py:23` is a bare `os.getenv` — no default, no
> assertion. With the variable unset the `or` does not fail; it falls through and
> every write connection targets **the replica**. So "all writes go to the
> sidecar" is a property of the environment, not of the code, for any caller that
> does not check.
>
> Callers that refuse the unset case rather than falling through:
> `scripts/train.py:908-929` (ABL-346, exit `2` before `initialize_all_tables()`
> at `scripts/train.py:940`)
> and `scripts/forecast_challengers.py:322-325`. Both also take `--sidecar-db`,
> as do `evaluate_scorecard.py`, `evaluate_net_position.py`,
> `evaluate_solar_retrain.py`, `evaluate_wind_retrain.py` and
> `attest_net_position_serve_faithfulness.py`. **Everything else still
> fallthrough-writes to the replica when the variable is unset** — if you add an
> entry point that writes, port the guard.
>
> `train.py` is the one that threads `--sidecar-db` back into
> `config.FORECAST_OUTPUT_DB`, because its writes go through `src/db.py`'s
> module-level helpers, which read that attribute per connection rather than
> taking a path. A `--sidecar-db` that only lands in `args` is decorative.

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

**`energy_renewable` can silently zero-fill a missing production type**
(ABL-188). Its per-column mapper (`energy-data-gathering/src/entsoe_client.py`
`_map_renewable_columns`, `:1607-1655`) initialises every renewable column to
0.0 before checking the source frame, unlike `energy_generation`'s
NaN-preserving twin mapper — so a type ENTSO-E didn't return for a window
(confirmed for DE solar, 2025-09-08 22:00–2025-11-14 15:45 UTC, 6,408
quarter-hours, `data_quality='actual'`) reads as a measured zero with no
signal anything is wrong. `energy_generation`'s same-fetch value for the
identical rows is the tell: real, non-null, non-zero (see
`reports/abl_188_solar_zero_adjudication.md`). `energy_renewable` is frozen
and redundant with `energy_generation` — retiring or re-deriving it is its
own cross-module migration requiring separate CEO/board approval, not a fix
available to this issue — so `src/data_quality.py`'s `exclude_suspect_constant_runs`
guards the training-data boundary instead: any individual-renewable-type
target loaded via `load_renewable_type_data` (`src/db.py:482`) that holds a
bit-identical value for 24+ hours is nulled before it can enter training,
with a `logger.warning` naming the exact excluded window. No stored row is
fixed by this — that needs a supplemental ENTSO-E re-fetch for the affected
window, proposed but not executed in the ABL-188 report.

**Neither generation table is hourly, and most countries are both** (ABL-332).
`energy_generation` and `energy_renewable` store whatever resolution ENTSO-E
published, and for most countries that changed partway through the history —
an hourly backbone for the early years, quarter-hourly later. Measured on the
replica 2026-08-12 over `config.SUPPORTED_COUNTRIES`: **22 of 24** carry
sub-hourly rows in `energy_renewable` and **20 of 24** in `energy_generation`.
Only **BE, BG, CH, LV, PT** are hourly throughout both. Do not reason about a
country's resolution from its name or its row count alone; the per-country
table is in `reports/abl_332_renewable_resolution.md`, regenerable with
`scripts/audit_renewable_resolution.py`.

Everything downstream of the read is hourly, and this is the contract:
`load_renewable_type_data` calls `aggregate_renewable_to_hourly`
(`src/db.py:398`) so **exactly one resolution leaves the read — the hourly
mean**. It has to be the read and not the consumer, because both consumers
already assumed hourly and disagreed about it: `features.py:227`'s
`create_lag_features` shifts by `days * 24` **rows** (a day only on an hourly
frame) and `src/wind_features.py` floors every lookup to the hour. Before
ABL-332 the serving builder therefore read the `:00` sub-sample and discarded
`:15`/`:30`/`:45` while training used the hourly mean — the same column name
carrying two different numbers, with no error and no log line. Measured on DE
solar over 2026-01-01 → 2026-08-12 (5,339 hours), the `:00` sub-sample differs
from its hour's mean by a median of **373.6 MW** (p90 3,211 MW, max 5,500 MW)
at a mean bias of only +3 MW — near-unbiased in aggregate, wrong in almost
every individual hour.

If you hand `src/wind_features.py` a sub-hourly series it now raises
`SubHourlyResolutionError` (`src/wind_features.py:142`) rather than
subsampling. Do not "fix" that by flooring the index — aggregate it.

The frame a model is **fitted** on did not change when ABL-332 landed —
`load_training_data`'s `resample('h').mean()` simply became a no-op — but
**`scripts/train.py`'s availability screen did** (`scripts/train.py:354`). It
reads the same loader and thresholds on `(target_value > 0).sum() / len(df)`
without resampling, and an hourly mean is non-zero whenever any sub-sample in
the hour is, so that fraction only rises. Measured over the screen's own
30-day window on 2026-08-12, all supported pairs, both source tables: 53 pairs
move the percentage without changing verdict and **one changes verdict —
IT/wind_offshore, 0.4865 → 0.5764 across the 0.50 threshold**, so it is now
eligible to train where it was previously skipped. Expect it to appear the
next time a training sweep runs; it is not a new data problem, it is the
screen finally measuring the hourly frame the model is fitted on.

**Which table an individual renewable type is read from is a property of the
model artifact, not a global** (ABL-331). `model_data["training_source"]` is
written by `Forecaster.save`/`_get_model_data` and read back by
`Forecaster.load` (`forecaster.py:1005`), which threads it into
`RenewableFeatureBuilder` at serve time and into `load_training_data` at train
time — so a pair is always served features from the table it was fitted on.
`db.RENEWABLE_TYPE_SOURCE_TABLE` (`db.py:361`) is now **only** the default for
a training run that names no source; it is no longer read at inference, and
flipping it moves no existing forecast. An artifact with no `training_source`
key predates ABL-331 and resolves to `db.LEGACY_RENEWABLE_TRAINING_SOURCE`
(`db.py:371`) — deliberately the literal `'energy_renewable'` rather than an
alias of the training default, because those artifacts were fitted on it and
must not follow a later flip.

That default is silent, and `load` reads every key with `.get(..., default)` —
so an artifact written **without** the key does not fail, it serves from
`energy_renewable` whatever it was fitted on. **`Forecaster.save` is therefore
the only writer of a renewable artifact** (ABL-342). Do not add a second one.
The two pre-registered gate harnesses used to `joblib.dump` seven keys of their
own; they now go through `src/evaluation/gate_artifacts.py:41`
`save_gate_artifact`, which takes the `RenewableFeatureBuilder` that produced
the training rows rather than a source string, so the recorded table cannot
drift from the series that was fitted. `ModelRegistry.save_model` takes a
caller's dict verbatim and cannot derive the value, so it **refuses** a
`RENEWABLE_TYPES` payload with no `training_source` (`model_registry.py:165`)
rather than let one reach `candidate/` or `production/`. Routing through `save`
also picks up the ABL-183 intercept witness, which the bare dumps omitted —
that is what made the guard a no-op for exactly the artifacts a gate produces.
`CascadeForecaster.save` (`forecaster.py:1408`) is not an exception: it stores
only the aggregate `load`/`renewable`/`price` types, which carry no source by
the rule above, and is read back by `CascadeForecaster.load_model`.

ABL-342 made that provenance faithful but gave neither harness a way to read
anything else. The **solar** harness now has one (ABL-345):
`scripts/evaluate_solar_retrain.py --renewable-source energy_generation`. It
resolves the source once (`evaluate_solar_retrain.py:208`) and hands the same
string to both read sites — the `RenewableFeatureBuilder`, which supplies the
fitted series, every lag and rolling feature, the D-7/persistence baselines and
the gate actuals; and `_constant_runs`, whose result drives `verdict`, so
screening the wrong table moves the disposition and not just the prose. The
resolved table is recorded in `meta.training_source` and printed in the report:
two gate reads are not comparable unless both name the table they read.

The **wind** harness (`scripts/evaluate_wind_retrain.py`) still has no source
argument **on `main`** and therefore still fits on `energy_renewable`. The
equivalent change exists on the unmerged `ABL-322-pilot` branch (`8662989`),
which also widens `PAIRS["wind_offshore"]` to BE/DE/FR/NL — a pilot scoping
decision, which is why ABL-345 left that file alone rather than conflict with it.
If that branch is dropped or rebased, the wind harness has no source argument at
all; do not assume `main` carries it.

Neither harness takes a **country** argument, and neither should get one as a
flag alone: `COUNTRIES`/`PAIRS` are the registered scope and `performance_pass`
is `len(gate_cells) == 9` (solar) / `== 15` (wind) against it, so a run scoped to
a subset reports `n/9` and FAILs on the count no matter how it scored. Extending
either to a new country is a new pre-registration, not a filter.

Why the source matters for the 37 unmodelled solar / wind_onshore pairs, measured
on the replica 2026-08-12: **33 of the 37 have under 365 days in
`energy_renewable`** (median 276 d), while **37 of 37 have over a year in
`energy_generation`** (median 2,049 d). Only BG and CH reach 2021 in both. A
harness pinned to `energy_renewable` gates those pairs on a model that has never
seen a full seasonal cycle.

One wrinkle both harnesses share: `--replica-db` governs only the incumbent, TSO
and contamination reads. The builder goes through `db.get_connection()` and so
opens **`config.DATABASE_PATH`** (`ENERGY_DB_PATH`) — point them at different
files and one run reads two databases. Pass `ENERGY_DB_PATH` explicitly; without
it the builder raises `sqlite3.OperationalError: unable to open database file`
before any fit, whatever `--replica-db` says.

The **training window** obeys the same rule. Both `train` entry points close an
open-ended window (`end_date is None`) with `db.get_latest_data_timestamp`, which
takes a `source=` and is handed `_resolved_training_source()`
(`forecaster.py:187`, `forecaster.py:458`). Until that was threaded, a run naming
`energy_generation` closed its window on `energy_renewable`'s last instant —
truncated where that table lags, and falling through to `datetime.now()` for a
pair with no rows in it at all, which is the normal case for the 39 unmodelled
pairs. Anything new that resolves a window or reports freshness for an
individual renewable type must pass the source; the constant is not the answer.

Train a pair on the other table with `scripts/train.py --renewable-source
energy_generation`. That CLI works again as of ABL-340 — it had been import-dead
since ABL-188 (`574eb80`), see "Importing this repo" below.

This exists because ABL-321 measured that switching globally makes 3 of the 10
serving pairs materially worse (AT solar +4.3%, DE wind_onshore +3.6%, BE
wind_onshore +2.7% relative WAPE) while the other 39 pairs cannot use
`energy_renewable` at all. Do not collapse it back to one constant.

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

## The interpreter is part of the configuration (ABL-69)

**This box has two Pythons, and a model artifact is only valid under the one
that wrote it.** The bare `python` on `PATH` is *not* the one the pipeline uses.

| role | interpreter | Python | xgboost |
|---|---|---|---|
| **the rail** — trains, serves, evaluates | `C:\Code\able\energy-forecast\.venv\Scripts\python.exe` | 3.14.3 | **3.3.0** |
| whatever `python` resolves to | `C:\Users\guill\miniconda3\python.exe` | 3.11.4 | 2.1.4 |

`scripts/workstation/run-net-position.ps1` invokes `$Repo\.venv\Scripts\python.exe`
explicitly for every step, so the scheduled job is consistent. **An interactive
run is not**, and that is where this bites.

An xgboost-3.3.0 pickle loaded under 2.1.4 does not fail. It keeps its trees and
**silently resets the fitted intercept to the 0.5 default** — FR's is 6,585.93 MW
— then predicts a near-zero-mean series. Measured on FR W12, 2026-08-08:

| interpreter | FR W12 MAE | SMAPE |
|---|---:|---:|
| `.venv` (3.3.0) | **1,688 MW** | 28% |
| conda (2.1.4) | 5,824 MW | 189% |

Predictions came back at mean −6 MW / std 575 against actuals at mean 5,818,
while correlation held at 0.615 — a model with shape and no level, which reads
as a bad model rather than a bad load. The only signal is a `UserWarning` about
serialized models. Nothing crashes and no test fails; the backtest simply reports
that the challenger lost.

`src/challengers/v014.py` now refuses this rather than trusting it.
`save_model` writes the xgboost version and the fitted intercept into the
artifact; `load_model` reads the intercept back out of the booster's own config
and raises `ModelArtifactError` when it has moved, naming the interpreter to use.
It checks the **symptom**, not version equality — so it stays silent across
upgrades that are actually fine, and fires whenever predictions would be wrong.
An artifact written before the guard carries no witness and still loads: absent
evidence is "cannot check", not "corrupt".

Run anything that loads a model — `train_v014.py`, `backtest_v014.py`,
`forecast_challengers.py`, `evaluate_net_position.py` — under `.venv`, and note
that `.env` is gitignored, so a **git worktree has no `.env`** and
`config.DATABASE_PATH` degrades to a bare `\data\energy_dashboard.db`. Pass
`ENERGY_DB_PATH` explicitly from a worktree.

One configured exception, measured rather than assumed (ABL-354): the
`tso-correction` runner is pinned to the conda interpreter at `config.py:490`,
not the rail. Its artifacts are **LightGBM**, not xgboost, and LightGBM
round-trips a booster as text. The three BE models
(`models/tso_correction/BE/*/model.joblib`, trained 2026-04-01) load with no
warning and predict identically to 6 dp under lightgbm 4.6.0 (conda) and 4.7.0
(`.venv`), and a full `-m src.tso_correction_forecaster --country BE --horizon 2`
gives the same `tso_raw` (1191.096365 MW mean) and `tso_corrected` (1254.571424)
under both. The ABL-69 failure does not reach this runner. That is a fact about
the artifact format, not a general licence — anything holding an xgboost pickle
still belongs on `.venv`.

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

This is a guard, not a fix. ABL-335 measured what the models emit: 22,718 of
131,356 stored solar rows negative, DE holding a 155-268 MW floor straight
through local midnight. The fit defect underneath is ABL-338's. **So the clamp
reports itself**: every run appends one row per country and model to
`forecast_clamp_log`, in the same database the clamped rows went into —

```sql
SELECT clamped_at, country_code, model_name, hours_zeroed_night,
       hours_raised_floor, mw_removed_night, mw_removed_total
FROM forecast_clamp_log ORDER BY clamped_at DESC;
```

A retrain that fixes the fit drives `hours_zeroed_night` and `mw_removed_total`
toward zero; the clamp going quiet is the measurement, and the clamp staying busy
after a retrain means the retrain did not work.

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
`src/tso_correction_forecaster.py:37`.

The second one does not currently run at all. `forecast_daily.py:226` launches
it as a subprocess **by file path**, and ABL-340 moved it to relative imports,
so it dies at import with `attempted relative import with no known parent
package` — every BE solar / wind row from the `tso-correction` runner has failed
since, while the run summary still reports `[DONE]`. ABL-354. That path
inherits the clamp the moment it can start; nothing about the clamp needs to
change for it.

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
day-ahead target's context 16h shorter than the live run really had, and
understates the pipeline. For `net_position` the serve-faithful **observation**
bound is **D 22:00**, not D 06:00 — verified by reproducing the live 2026-08-06
vintage **bit-exactly** (max |diff| 0.0 MW over 480 points; `predict_quantiles`
is deterministic, so an exact match really does mean an identical input).

**A serve-faithful reconstruction needs two bounds, not one** (ABL-68). One
`as_of` was doing double duty: it bounds where observations stop *and*, via
`_load_weather_forecast_range`, which weather runs had been issued
(`forecast_run_time <= ?`). Those are the same instant only when the target is
published in real time. For `net_position` they are 16h apart, so neither value
is right on its own:

- `D 22:00` is the correct observation bound, but it also admits a weather run
  issued at 12:00Z on D — information the 06:00Z run never had. Measured against
  the as-served 2026-08-06 vintage, this put the worst country **1,881 MW** away
  from what production served.
- `D 06:00` is the correct publication bound, but it truncates the context 16h.
  This is what `scripts/compare_experiments.py:178` still does for *every*
  forecast type, so its net_position weeks understate the pipeline.

`build_for_country` therefore takes `publication_as_of` alongside `as_of`
(`src/chronos2/input_builder.py:541`), defaulting to `as_of` so live and
existing callers are unchanged. With the bounds split, 16 of 19 countries
reproduce the as-served vintage to under 0.3% of mean |forecast|; LT (38.8%),
RO (5.9%) and BG (1.4%) do not, because **suffix-1 covariates cannot be bounded
at all**. TSO load forecasts, DA prices and cross-border flows are bounded by
timestamp only — `publication_timestamp_utc` records when we fetched, not when
the value was published, and is NULL on these rows — so a vintage reconstructed
days later legitimately sees revisions the live run did not. Any model fitted on
a reconstruction should treat those three countries as unverified rather than
assume the fit transfers.

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

### Net-position evaluation and the promotion gate

`src/evaluation/net_position.py` (ABL-30) scores the as-served vintages against
`net_position` actuals; `scripts/evaluate_net_position.py` is the entry point and
writes `reports/net_position_eval/`. Both databases are opened **readonly**.

```bash
# single model (default: chronos-2-V010)
python scripts/evaluate_net_position.py --replica-db ...\energy_dashboard.db \
    --sidecar-db ...\forecasts_local.db --stdout
# several models over one identical vintage window — the C2c deliverable
python scripts/evaluate_net_position.py --model chronos-2-V010 chronos-2-V012 ...
```

Four things about the gate are load-bearing (ABL-72):

- **The gate scores a vintage window; the report does not.** The tables cover
  every stored vintage, but `promotion_gate` reads `results["gate_scope"]`,
  which defaults to vintages at or after `cohort_split` (`FIX_DEPLOYED_UTC`).
  Without that restriction the champion is measured on the zero-padded-context
  era: measured on the replica 2026-08-07, all 18 vintages give MAE 1,439 MW /
  slope 0.26 against the serving model's 553 MW / 0.90, so a challenger faced a
  bar **2.60x easier** than the real one. The difference is not cosmetic —
  `slope_in_range_per_country` reads 0/19 contaminated and **11/19** windowed.
  Override with `--gate-vintage-start` / `--gate-vintage-end`; the window and its
  vintage count are printed in the report header.
- **All eight pre-registered criteria are emitted, and PASS requires all eight.**
  `PRE_REGISTERED_CHECKS` is the list; the gate checks itself against it and the
  report iterates it, so an absent criterion prints as `NOT IMPLEMENTED` instead
  of being silently skipped. The verdict is `PASS` / `FAIL` / `INCOMPLETE` — a
  criterion that cannot be evaluated (no `--candidate-backtest`, no
  `--serve-faithful-verified` attestation) yields `INCOMPLETE`, never `PASS`.
  Two of the eight had never been implemented, and because the old verdict
  spanned "only evaluable checks", their absence could not fail.
- **LU and GR are excluded by name**, not by symptom — `GATE_EXCLUDED_COUNTRIES`
  carries a reason for each (LU duplicates DE in A25; GR's actuals are
  fabricated zeros, ABL-35/ABL-67). GR was previously excluded only as a
  side-effect of having no paired actuals, so a partial upstream resume would
  have silently re-entered it and failed the gate on thin data.
- **A comparison shares one window across every column.** It is the intersection
  of the models' stored vintage spans, floored at `cohort_split`, and
  `compare_models` raises if the columns end up scored over different windows.
  Per-model vintage counts are printed rather than smoothed, and a model with no
  stored vintages reads as "Not scored", never as an empty column.

**The script does not discover model versions.** It scores exactly the
`--model` names given. Anything claiming it picks up new versions automatically
is wrong — that was ABL-68 scope item 1 and plan Rev 3:29.

#### Per-country re-read: zero baseline, level vs shape (ABL-280)

`src/evaluation/country_reread.py` + `scripts/reread_net_position_country.py`
answer "is this one zone's forecast actually worse than a free baseline, and
*how*". It reuses the eval's loaders, serve-faithful baselines and
`point_metrics`, so it cannot disagree with the gate about a country's MAE.

```bash
.venv\Scripts\python.exe scripts/reread_net_position_country.py --country RO --fleet \
    --replica-db ...\energy_dashboard.db --sidecar-db ...\forecasts_local.db --stdout
```

Three things it adds, each because a real reading went wrong without them:

- **The zero forecast is a named baseline, and `skill_vs_zero < 0` is
  identically `WAPE > 100%`** — the same fact twice, pinned as an equality in
  the tests. Naming it is what stops "WAPE 102.6%" reading as an emergency on
  its own: zero is not a baseline anyone would serve for net position, and RO
  loses to it while beating persistence by 20.6%. The decision-relevant row is
  climatology, not zero.
- **Level vs shape.** Demeaning both series *within each vintage day* separates
  a wrong profile from a right profile at the wrong level. Measured 2026-08-12
  on the 7-scored-vintage cohort, RO reads pooled corr 0.50 / within-day 0.83
  and a per-vintage-day bias sd of 721.5 MW against mean |actual| 709.0 MW.
  That is what refutes a *static* per-country offset for RO — a constant cannot
  track a bias that swings +259 to −1095 MW across six days. NL (+0.37 gap) and
  LV (+0.28) carry the same signature; it is a cluster, not one zone.
- **Vintages that carry evidence, counted separately from vintages that
  exist.** `build_gate_scope` counts off the left-merged frame, so a vintage
  whose D+2 targets have no published actuals still counts toward
  `min_live_shadow_vintages`. That gap is permanent, not incidental — the rail
  generates at D for D+2, so the two newest vintages are always unscorable.
  Measured 2026-08-12: **9 counted, 7 scored**. So `min_live_shadow_vintages`
  reaches 14 on 2026-08-17 with ~12 vintages of evidence behind it; 14 *scored*
  vintages land 2026-08-19. This module counts scored vintages and labels its
  own output `INTERIM` / `CONFIRMATORY`; it deliberately does **not** change the
  gate, which is pre-registered.

`--fleet` sweeps every `GATE_COUNTRIES` zone beside the named one. That is not
decoration: on the interim cohort 4 of 19 lose to climatology (RO −23.3%, NL
−18.3%, then BE −2.4% and HR −0.2% inside noise), so a fallback proposed for RO
alone would leave NL served by a model it also loses with. Evidence pack:
`reports/abl_280_ro_climatology_reread.md`. Dated outputs land in
`reports/net_position_eval/country_reread/`, which is gitignored like the rest
of that directory.

### All-type forecast scorecard (ABL-129)

`scripts/evaluate_scorecard.py` is the recurring answer to "is the served
forecast better than a free baseline?" It scores the production registry
snapshot for all nine served types over one target window and writes a dated
Markdown/JSON pair plus `latest.*` under `reports/forecast_scorecard/`
(`scripts/evaluate_scorecard.py:17`, `scripts/evaluate_scorecard.py:58`). It
opens the replica and optional sidecar read-only; its only writes are reports.

The selection rule is **latest vintage per country + target + model + horizon
band**, not one latest row per target. The latter erases the stored 24-64h
evidence because the newest daily run is always the shortest lead. Timestamps
are parsed before the join so both the ML `T` separator and Chronos space
separator pair (`src/evaluation/scorecard.py:95`). The ABL-35 `load_mw > 0`
guard applies to load only; measured zero is retained for solar, wind, price,
and every other type. GR net position is excluded by name using the reason from
`GATE_EXCLUDED_COUNTRIES`, not by detecting zero-shaped data
(`src/evaluation/scorecard.py:178`).

D-7 and persistence predictions go through `src/baselines.py`, via the pure
issued-row adapter at `src/baselines.py:297`. Persistence derives its lookback
from target minus `generated_at` and rounds the lead **up**: stored
`horizon_hours` floors partial hours, so using it directly can select an actual
from after generation. Net position instead reuses its evaluator's day-ahead
publication cutoff and persistence implementation. Missing actual/baseline pairs remain unmeasured, and
skill is computed only on the exact intersection available to both model and
baseline. This scorecard references the separate net-position promotion gate;
it does not copy or weaken it (`src/evaluation/scorecard.py:326`).

### Experiment System

Experiments are versioned V001-Vnnn with configs in `experiments/`. Both XGBoost and Chronos-2 run in parallel — forecasts stored with distinct `model_name` values in the `forecasts` table.

```bash
experiments/
├── registry.json           # Master index of all experiments
├── V001/config.json        # XGBoost baseline
├── V002/config.json        # Chronos-2 zero-shot
├── V003/config.json        # Chronos-2 fine-tuned (5000 steps)
├── V012/config.json        # Baseline ensemble (shadow challenger)
└── V016/config.json        # V010 + affine + AR(1) (shadow challenger)
```

### Champion / challenger shadow serving (ABL-68)

The daily 08:00 net-position job runs the champion, then
`scripts/forecast_challengers.py` runs every registered challenger on the same
serve-time inputs. Challengers write their own `model_name` rows to the sidecar
and **are never pushed to production**.

`model_name` is the identity that matters. `model_version` is the vintage
timestamp, not a model identity — two models sharing a `generated_at` are told
apart by `model_name` alone. Challengers are listed in
`src/challengers/registry.py`, not discovered, so what runs tomorrow is a
reviewable list.

**Two things enforce the "never pushed" invariant, and both are load-bearing.**
`push_net_position_forecast.py` names the champion (`CHAMPION_MODEL_NAME`,
default `chronos-2-V010`) and filters every query on it. Before ABL-68 it took
the newest `generated_at` for `forecast_type='net_position'` with no model
filter — correct only while the sidecar held one model. Challengers run *after*
the champion in the same job, so the newest vintage in the sidecar is now a
challenger's: verified 2026-08-07, the newest row was `chronos-2-V016` and the
unfixed script would have shipped it to the dashboard as the production
forecast.

**The eval scores every stored vintage, but one `model_name` per invocation.**
`evaluate_net_position.py --model` defaults to the champion, so a challenger is
scored only if the runner names it. `run-net-position.ps1` calls it once per
model, each with its own `--out-dir`, because the script always writes
`latest.md` beside the week-tagged report and a shared directory would leave
`latest.md` holding whichever model ran last — which ABL-30 and ABL-34 both read
expecting the champion.

**V012 does not reimplement its own baseline.** It calls
`src/evaluation/net_position.py::baseline_predictions`, the same function the
gate scores against. Two implementations of one baseline is the shape of the
renewable-share defect.

**Never compare two per-model eval reports to each other.** Each report is
scored on whatever rows its own model covers, and the champion's set also picks
up prod-pushed vintages that live in the replica and were never in the
reconstruction a challenger is rebuilt from. On V016's held-out window the
champion's report covered 57 vintages to the challenger's 49. Read report against
report, V016 looked *better* almost everywhere (FR 2,464 → 1,916 MW, DE 3,344 →
3,014 MW); scored on the rows both models actually cover, it is **worse**. Use
`scripts/compare_challenger.py`, which inner-joins on
`(country, target hour, run)` and reports the one-sided remainders
(`src/evaluation/head_to_head.py`).

**A run is not a `generated_at`, and on the live rail the two never match**
(ABL-82). The head-to-head's first cut joined on exact `generated_at` equality.
That is right for a reconstruction — one process replays every vintage and
stamps them all — and wrong for the daily shadow rail, where
`forecast_chronos2.py` and `forecast_challengers.py` are separate processes in
`run-net-position.ps1` and each calls its own `datetime.now()`. Measured on the
live sidecar 2026-08-09: champion `2026-08-09 06:00:55.715745`, all three
challengers `2026-08-09 06:01:08` — 12.3 s apart, and only the champion carries
microseconds. The exact join paired **0** rows for V012, V014 *and* V016 while
912 co-run pairs sat there, and it did so **while printing a full report**:
`0.0 MW` MAE for both models and "challenger is 0.0% worse". An empty
head-to-head that renders as a tie is this repo's usual defect in a new place.

Two vintages are now the same run when they agree on the **actuals they could
see** (`net_position.as_of_for_vintage`, the same serve-faithful cutoff the
eval's baselines use) *and* their `generated_at` are within `MAX_RUN_SKEW` (4 h)
of each other. The cutoff carries the meaning; the skew bound is a guard, since
one cutoff bucket is 24 h wide. `--max-run-skew-hours` tunes the bound only —
**it cannot pair two vintages that saw different actuals, at any value**, and
that is deliberate: an information mismatch is not a tolerance problem.

Three properties are load-bearing:

- **Backfills are refused, not paired.** The 2026-08-07 V012/V016 backfill ran
  15 h 25 m after that day's champion and V014's first vintage 5 h 36 m after
  (2026-08-08 11:36). Both saw a further day of actuals, so scoring them
  against a 06:00 champion would credit a challenger for information the
  champion never had. They land in `n_only_a`/`n_only_b`, where a reader sees
  them.
- **A champion re-run duplicates nothing.** 2026-08-06 holds two champion
  vintages (06:00:44 and 10:52:22) under one cutoff. The pair closest in time
  wins and the other falls to `n_only_a`; a naive day-level join would have
  matched both to the single challenger vintage and counted the challenger's
  hours twice.
- **Nothing paired reports no number.** `pooled_mae_*` is `None`, the report
  renders a "Not measured" block instead of a table, and
  `compare_challenger.py` **exits 1**. A promotion gate must not be able to
  read an empty comparison as "no difference".

The reconstruction path is unchanged: re-run 2026-08-09 with the new rule, the
V016 held-out comparison still returns exactly **22,344 paired rows over 49
runs**, V010 **775.2 MW** vs V016 **786.1 MW**, 1/19 materially better, 3
identical — the numbers below, to the decimal.

One timing consequence worth knowing before reading a fresh shadow report: the
rail forecasts **D+2**, so a co-run pair is not scoreable until its target day
lands. On 2026-08-09 the live head-to-head correctly reports *not measured* for
all three challengers — the 912 pairs it now forms target 2026-08-10 and 08-11.
The first live-rail pairs score on 2026-08-10, and by the C2c gate read
(~2026-08-26) roughly 17 daily runs are available.

**V016 refuses more than it corrects, and does not beat the champion.** Measured
on a held-out window (fit 2026-01-19..06-15, tested 06-17..08-04, 22,344
exactly-paired rows over 49 vintages): V010 **775.2 MW** MAE, V016 **786.1 MW** —
1.4% worse. It is materially better (≥0.5%) in **1 of 19 countries** (FR, −2.1%),
identical in 3 (BG/LT/RO pass through uncorrected), and within noise or worse in
the remaining 15. Forcing unit slope instead (`--method variance`) costs 11.4%:
863.8 MW, better in 0 of 19. Archived in `reports/head_to_head/V016/`
(deliberately *not* under `reports/net_position_eval/`, which is gitignored
because the scheduled eval rewrites it every run), reproduced 2026-08-08 with
`experiments/V016/correction_holdout.json`. AR(1)-only and a rolling 60-day
refit were also tried and also lost, so drift is not the explanation.

Two reasons, both worth knowing before proposing another correction layer:

- **Affine recalibration cannot fix the residual shrinkage.** For any affine
  map, `slope(corrected on actual) = b * slope(f on a)`, which under OLS is
  exactly `rho**2`. It *lowers* the slope in 15 of 16 countries (FR 0.480 →
  0.398, BE 0.298 → 0.201). And `b < 1` for 15 of 16, so the error-minimising
  move is to shrink the champion *further*: V010 is already close to affinely
  optimal per country. Unit slope requires inflating variance, measured at an
  11% MAE cost. **The gate's `slope ∈ [0.8, 1.2]` is therefore unreachable by
  any affine layer on V010** — it needs `rho ≥ 0.894`, and measured per-country
  `rho` is 0.41-0.88. That is a better-model problem (V014/V015).
- **AR(1) is bounded by the horizon, not the coefficient.** Residual lag-1
  autocorrelation is genuinely 0.85-0.96, but a 06:00Z run observes actuals only
  to D 21:00 while correcting D+2 00:00-23:00. The carry is `phi**27..phi**51` —
  0.04 to 0.32 at the nearest corrected hour. Small by construction.

**Two fit files, and they are not interchangeable.**
`experiments/V016/correction.json` is fitted on everything (`train_end: null`)
and is what the daily shadow run serves — correct for serving forward, and
*in-sample* for any window before the fit date. `correction_holdout.json` is
fitted to 2026-06-15 only and is the one every quoted V016 number above comes
from. Evaluating V016 with the full-sample fit would score it on data it was
fitted on and flatter it. Both drop the W11/W12 backtest target days.

Note the pooled-vs-per-country trap here: the plan's 0.894 correlation is
pooled, which mixes country means and is inflated by between-country variance
(the eval's own docstring says so). Per-country `rho` is much lower, and the
per-country numbers are what a per-country correction can use.

### V014 — the trained per-country XGBoost challenger (ABL-69)

The challenger the Board asked for, and the answer to the paragraph above: V010
is close to affinely optimal per country, so no correction layer on it can reach
the gate's `slope ∈ [0.8, 1.2]`. That needs a different model.

- `src/challengers/v014_features.py` — the feature builder (89 features)
- `src/challengers/v014.py` — model, refusals, artifact integrity
- `scripts/train_v014.py`, `scripts/backtest_v014.py`
- `experiments/V014/config.json`, `training_report.json`, `backtest_W01_W12.json`
- Artifacts: `models/net_position/V014/{CC}.joblib`, 19 countries (all supported
  net-position countries except LU, which duplicates DE in the A25 document, and
  GR, whose actuals are fabricated zeros — ABL-35/ABL-67)

**`models/` is gitignored, so merging the branch does not ship the model.**
The scheduled job runs from `C:\Code\able\energy-forecast` (`$Repo` in
`run-net-position.ps1`), and `config.MODELS_DIR` resolves against *that*
checkout — not the worktree the training ran in. Artifacts have to be copied
across by hand, and if they are not, the rail logs "no trained model for
AT,BE,…" and writes nothing for V014 while every other model succeeds and the
job still exits 0. Retraining in a worktree therefore has two steps:

```bash
python scripts/train_v014.py --countries all          # under .venv
cp -r models/net_position/V014 C:/Code/able/energy-forecast/models/net_position/
```

**Serve-faithfulness holds by construction, and it has to.** A tabular model
evaluates every feature *at the target timestamp*, so unlike Chronos-2 — whose
context simply ends where the data ends — each column must justify its own
availability. It cannot be verified after the fact from ingest metadata:
`fetched_at` and `publication_timestamp_utc` are last-write over a rolling
re-fetch window, so every FR `net_position` row for targets 2026-08-01..07
carries the identical `fetched_at`. An as-of query over them passes anything you
hand it. The construction is one documented per-source cutoff derived from the
run instant, applied identically in training, backtest and serving:

| source | cutoff at a 06:00Z run on D |
|---|---|
| `net_position`, `energy_price`, `energy_load_forecast`, `energy_generation_forecast` | D 21:00 (day-ahead publication) |
| `crossborder_flows` | target − 72h (ABL-74), with an `xb_missing` indicator |
| `weather_data` | at the target hour, `data_quality='forecast'` and `forecast_run_time <= run_ts` |

**Same-hour lags start at 72h, not 48h.** The binding target hour is D+2 23:00,
exactly 50h past the D 21:00 cutoff, so a 48h lag reaches D 22:00 and D 23:00 —
two hours that do not exist at run time. `assert_lag_is_serve_safe` checks every
lag on every build; the tautological check (filter to `<= cutoff`, then assert
the max is `<= cutoff`) cannot fail and was deliberately not written.

**W01-W10 are weather-blind for every model.** The issued-forecast archive
begins 2026-01-11 (FR's earliest `forecast_run_time`), so those ten weeks carry
NaN weather. The builder does **not** fall back to the `data_quality='actual'`
reanalysis, which is a nowcast (lead 0.0h) and would be observed weather handed
to the model as a forecast. `weather_available` records the regime per row.
The champion's loader filters the same way, so the comparison is fair — but
neither model is in its serving configuration there.

**Three refusals**, all because a tree returns a number for any row you give it:
no model file for a country raises rather than substituting another country's
model; fewer than 2 of 3 anchor features (`np_at_cutoff`, `np_lag72h`,
`np_last7d_mean`) yields NaN for that hour; and a refused hour is **dropped, never
written as 0.0** — a 0 MW net position is a real balanced-border reading. Nothing
is imputed: a mean would turn "we do not know this border's flow" into "the flow
was average".

**A late run is not a better-informed run.** `run_v014` derives its serve window
from the *target date*, never from `generated_at`, so a job that fires late gets
the cutoffs the schedule promises rather than the extra hours the clock handed
it. Otherwise a delayed vintage would be scored against models built on less.

Promotion is not decided here — only by the pre-registered gate read in C2c
(ABL-72). V014 supplies two of its eight criteria: G5 wants
`experiments/V014/backtest_W01_W12.json` via `--candidate-backtest`, and G6 wants
a bit-reproduced live vintage via `--serve-faithful-verified`.

**Tuning to MAE moves BE out of the gate's slope band, so nothing was adopted.**
`scripts/tune_v014.py` runs a small readable grid per country, selecting on
validation MAE over the same chronological split the fit uses — never on the
backtest weeks, which would make the backtest a training set. Measured
2026-08-08 on the four countries where V014 trails V010, only **BE** cleared the
2% bar, and it did so by breaking the criterion the model exists to satisfy:

| country | default MAE / slope | best candidate | its MAE / slope |
|---|---|---|---|
| BE | 1,658.9 / 1.047 | `shallow_slow` **−10.2%** | 1,489.0 / **1.395** |
| NL | 1,826.4 / 0.996 | `deeper_slow` +0.9% | 1,810.1 / 0.976 |
| AT | 1,000.2 / 1.068 | `shallow_slow` +1.6% | 983.9 / 0.980 |
| FR | 2,120.0 / 0.993 | `shallow_slow` +1.8% | 2,081.6 / 1.132 |

The gate wants `slope ∈ [0.8, 1.2]`; BE's MAE-optimal fit sits at 1.395. So the
one adoption the search offers trades the criterion V014 was built for against
the one it is currently losing. The script therefore **writes no model without
`--adopt`**, and `--adopt` was not used. Deciding the tuning objective against
the gate rather than against MAE is a program-level call (ABL-24), not a
parameter choice. Note `shallow_slow` is the best candidate in three of four
countries and stays inside the band in NL/AT/FR — it is BE specifically where
MAE and slope pull apart.

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
