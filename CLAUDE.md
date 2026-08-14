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
│   │                       # capacity-weighted point per country (ABL-337);
│   │                       # also NIGHT_GENERATION_POSSIBLE, the per-country
│   │                       # night-generation fact, no default (ABL-425)
│   ├── solar_clamp.py      # Serving-path night mask + non-negativity floor
│   │                       # for solar, with per-run telemetry (ABL-337).
│   │                       # ES is exempt from the night mask (ABL-425)
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
  is why the guard now launches them instead of trusting the summary. What the
  summary itself can now say is in "What a runner reports" below (ABL-370).
- `src/evaluation.py` is dead code. `src/evaluation/` is a package and shadows
  it — `src.evaluation` always resolves to the directory. `src/__init__.py:44`
  already has its re-export commented out.

### Help text is ASCII; report bodies are not

`--help` output must be plain ASCII. `scripts/train.py:262` held a literal `→`,
and `python scripts/train.py --help > /dev/null` exited **1** with
`UnicodeEncodeError` — CPython takes stdout's encoding from what stdout *is*, so
a console writes through `WriteConsoleW` and survives, while a pipe or a file
falls back to the locale codepage (cp1252 here) and `argparse` raises inside the
`--help` action itself (ABL-364). Interactively it looked fine; every harness,
CI step and agent that captures stdout saw a traceback instead of usage.

**A module docstring is help text.** 18 of the 38 entry points pass
`description=__doc__`, so an em dash in a docstring is the same defect one
codepage over (cp1252 encodes `—`, cp850 does not). Nine scripts beyond
`train.py` were carrying one.

`tests/test_help_text_encoding.py` holds the line: it reads every entry point's
parser out of the AST — `help=`, `description=`, `epilog=`, and `__doc__` where
it is passed as one — and rejects any character above U+007F. Write `->` and
`--`. It is a static sweep because `--help` on an arbitrary script means
executing that script to module scope, and it runs `scripts/train.py --help`
under `PYTHONIOENCODING=ascii` as the one end-to-end case, so the assertion does
not depend on the codepage of the box.

"Entry point" there means the same set as `test_script_imports.py`: every
`scripts/*.py`, the repo-root runners, **and every `config.MODEL_RUNNERS`
script**. That last group is not decoration — `test_model_runner_launches`
starts each runner with `--help` through a pipe, and two of them
(`src/chronos_forecaster.py`, `src/tso_correction_forecaster.py`) live under
`src/`, outside the `scripts/` glob. Both are ASCII-clean today. Note that
neither passes `description=__doc__` — they use short literals
(`tso_correction_forecaster.py:375`) — so unlike the `scripts/` entry points
above, **their module docstrings are not help text** and an em dash in one is
harmless. What is swept, and what matters, is their `help=`/`description=`
literals: a non-ASCII character there stops the runner from starting, and per
the bullet above `forecast_daily` books a runner that cannot start as a failed
*result* and still prints `[DONE]`.

Report bodies are the deliberate exception and keep `Δ`, `→`, `·`. They are
printed from one known place, so they re-encode the stream there
(`evaluate_net_position.py:125-132`, `compare_challenger.py:127-133`) — which is
not available to `--help`, since argparse prints before any line of `main()`
runs. Do not "fix" this the other way by forcing UTF-8 on stdout at import time:
that is a runtime change in 38 scripts, forgettable in the 39th, and it
re-encodes the log files the `scripts/workstation/*.ps1` jobs capture.

## What a runner reports (ABL-370)

The exit code says whether a runner crashed. It does not say whether it
*produced* anything, and `forecast_daily` used to read it as if it did: exit 0
was logged `OK`, and a row count was recovered only if stdout happened to
contain `Forecast (N rows)` or `Saved N forecast records`. A run that generated
nothing prints neither, so `tso-correction` skipping all three renewable types
on a day the upstream Elia forecast has not landed printed

```
[tso-correction] OK: BE solar D+2
Total: 10, Success: 2, Skipped: 8, Failed: 0
```

— indistinguishable from a run that saved 96 rows, and its 0 vanished inside a
`Total forecasts:` sum the in-process models push into the thousands. That is
the same reporting shape that hid ABL-354.

**Every external `MODEL_RUNNERS` entry emits one line on stdout, once per run,
zero included:**

```
FORECAST_RECORDS=<n>
```

`src/runner_report.py` owns both ends — `emit_record_count()` writes it,
`parse_record_count()` reads it — so the contract cannot drift. It imports
nothing but `typing`, deliberately: `chronos-bolt-small` runs under its own venv
and importing this must never be what breaks it.

`forecast_daily` then distinguishes four outcomes, not two:

| outcome | means |
|---|---|
| `success` | exit 0, reported ≥ 1 row |
| `empty` | exit 0, reported exactly 0 rows — ran fine, produced nothing |
| `unreported` | exit 0, no count line — what it did is **unknown**, and unknown is not 0 |
| `failed` | non-zero exit, timeout, or exception |

`records` is `None` for `unreported`, and contributes nothing to
`Total forecasts:` — recording it as 0 would be a number nobody measured. The
summary gains a per-runner block and an explicit
`Runners that produced no forecasts:` callout, which is the line a silent runner
now has to appear on.

`empty` is not a failure and does not change the exit code: skipping when the
upstream forecast is genuinely absent is correct behaviour, and today D+1/D+2
for BE legitimately produce nothing (`energy_generation_forecast` for BE ends at
the reference date). The defect was never the zero — it was that the zero was
unsayable.

Adding a runner: call `emit_record_count(len(df))` on every path that exits 0,
*before* any `if not df.empty:` guard. `tests/test_runner_reporting.py` checks
that statically for each configured runner and would otherwise report your
runner as `unreported` forever.

### Skipped is a flag, not a phrase

`failed` is reported net of `skipped` — "there was no model to run" is not a
failure — but the two used to be told apart by looking for `not found` in the
error text. `chronos-bolt-small` points at a venv that does not exist on this
box, so it fails with `Executable not found: [WinError 2]`, and a runner that
could not run *at all* was counted as benign. `generate_forecast` now sets
`result['skipped'] = True` at the one place that knows (the `FileNotFoundError`
from `Forecaster.load`), and `is_skip` reads only that.

Consequence worth knowing: a default run on this box now ends
`Skipped: 1, Failed: 1` and exits 1 for BE/price, where it used to exit 0.
`chronos-bolt-small` is genuinely unrunnable here; fix the path in
`config.MODEL_RUNNERS` or set `enabled: False`, but do not read the exit 0 that
preceded it as the job having been fine.

That same handler used to log `python_exe`, a name local to
`build_runner_command` since ABL-354 — a `NameError` raised *inside* an
`except` clause, which the sibling `except Exception` does not catch. A missing
runner interpreter killed the whole daily job before it printed any summary.
`--countries BE --types price` reproduces it on the pre-fix file.

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

### TSO day-ahead forecasts are guarded on the way in (ABL-431)

`energy_generation_forecast` and `energy_load_forecast` carry ENTSO-E's
published day-ahead forecasts verbatim, and verbatim includes a **×1000 unit
error**: HU's `wind_onshore_mw` reads **140,996 MW** against a fleet whose
p99.5 over five years is 283 MW. Dividing those 96 quarter-hours by 1000
reproduces HU's own measured generation for the same day (35.8–141.0 MW
predicted against 36.8–133.0 MW observed, rising together through the day), so
the shape is right and only the scale is wrong — which is why it is invisible
to every correlation- or shape-based check.

**Extent, measured on the replica 2026-08-14 and regenerable with
`scripts/abl431_tso_plausibility_census.py`: 213 of 14,610,819
column-observations (0.0015%)**, in three incidents — HU 2026-02-04 (96 rows,
one full CET market day, hitting `wind_onshore_mw` and the `total_forecast_mw`
it dominates), MK 2022-04-10 (10 rows), SK 2022-09-25 (1 row). **Zero rows in
`energy_load_forecast`.** So it is not one row, and it is not widespread
either.

`src/tso_plausibility.py` nulls a read value above `PLAUSIBILITY_TOLERANCE`
(3.0) times a per-country, per-column reference scale, logs one warning naming
the country, column, threshold, magnitude and window, and **never touches the
stored row** — a value that looks impossible is sometimes just not published
yet. Wired into `v014_features`, `chronos2/input_builder`, `scorecard`'s TSO
comparator and both `tso_correction` read sites; the guard runs at the
published resolution and *before* any hourly resample, so a bad quarter cannot
be smeared across its hour first.

Four things about it are load-bearing:

- **The reference is derived, not registered.** There is no installed-capacity
  table on the replica, and a committed one would go stale in the direction
  that matters — NL solar grew from nothing to 7.9 GW inside this history and a
  frozen bound would start rejecting real growth. It is
  `max(p99.5(actuals), p99.5(day-ahead forecasts))` over the whole series,
  recomputed at read time and cached per process. **Both sides, because neither
  alone is sound**: NL's `energy_generation.solar_mw` tops out at 428.8 MW while
  NL's own published solar forecast reaches 7,871 MW, so an actuals-only anchor
  would reject 18× of legitimate NL solar; and the forecast table is the
  defect's own home, so a forecast-only anchor could be set by the rows it is
  meant to catch. It is a quantile rather than a maximum for that second
  reason — which bounds it: a contaminated cluster covering more than 1 − q
  (0.5%, ~10 days of a five-year quarter-hourly series) would raise its own bar.
  HU's is 0.0487%.
- **3.0 is a measurement, not a convention.** Across all 146 evaluable pairs,
  `max / reference` runs HU 497.7× · HU total 37.3× · SK 8.70× · MK 6.05× ·
  MK total 4.12× — then nothing until PT solar at **1.82×**, PT wind_offshore
  1.77×, NL load 1.60×, p90 1.41×. 3.0 sits inside a measured empty band 2.3×
  wide. The census prints that ladder every run, so a healthy pair climbing
  toward the tolerance is visible before a fit meets it.
- **It is one-sided, and it refuses to evaluate rather than rejecting
  everything.** A published 0.0 is never flagged at any tolerance, so ABL-71's
  published zeros and ABL-109's 56 legitimate DE overnight solar zeros are
  untouched by construction. **28 of the 174 pairs are all-zero series** —
  landlocked countries reporting `wind_offshore_mw = 0.0` forever — where the
  reference is 0.0 and `value > 3 × 0` would flag every non-zero value a new
  fleet ever published. `ReferenceScale.evaluable` is False there and the series
  passes through carrying the reason. Same mechanism means a **brand-new
  fleet's first output is unguarded**, which is the deliberate direction: an
  unguarded new fleet is a bounded cost, a guard that deletes a country's first
  real generation is not.
- **No default, and `as_of` is the caller's choice.** An unregistered
  `(table, column)` raises `UnknownTsoSourceError` rather than guarding against
  a guessed scale. `reference_scale(..., as_of=...)` bounds both sides for a
  backtest reconstructing a past vintage; the default is the whole history,
  which is serve-faithful for serving because at serve time the whole history
  *is* everything available.

`tests/test_tso_plausibility.py` pins all of it, including a static sweep that
fails if any `src/` module names one of the two tables without calling the
guard or appearing on an exempt list with a reason. **That sweep is what ABL-247
will trip when it adds its feature read** — which is the point: this issue is
that issue's precondition.

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
resolves the source once (`evaluate_solar_retrain.py:351`) and hands the same
string to both read sites — the `RenewableFeatureBuilder`, which supplies the
fitted series, every lag and rolling feature, the D-7/persistence baselines and
the gate actuals; and `_constant_runs`, whose result drives `verdict`, so
screening the wrong table moves the disposition and not just the prose. The
resolved table is recorded in `meta.training_source` and printed in the report:
two gate reads are not comparable unless both name the table they read.

The **wind** harness (`scripts/evaluate_wind_retrain.py`) takes the same
`--renewable-source` argument, resolves it to the same two read sites, and
records it in `meta.training_source`.

Neither harness takes a **country** argument, and neither should get one as a
flag alone. `COUNTRIES`/`PAIRS` are the registered scope and `performance_pass`
is `len(gate_cells) ==` that scope's size, so a filtered run FAILs on the count
no matter how it scored — and a country filter cannot say "offshore only", so it
also drags serving pairs of the *other* stream into the gate. Scoping a run is a
new pre-registration, not a filter.

The wind harness therefore takes `--scope`, not `--countries`. `SCOPES` maps a
registered name to an explicit `(stream, country)` pair list, and the bar is that
list's size × `PRIMARY_BANDS` — read from the table in the file, never from what
the run turned out to score, so a pair that silently yields no gate rows still
shortfalls the count and reads FAIL. `abl195` (the default, so an unflagged run
reproduces ABL-195 exactly) is 5 pairs → 15 cells; `abl322-pilot` is DE/NL
`wind_offshore` → 6 cells and refits no serving pair. Adding a scope is a
pre-registration and belongs in review. `tests/test_gate_scope_registration.py`
pins all of this, including that `--countries` is not reintroduced.

A scope also registers its **gate basis** (`GATE_BASIS`): the columns that must
be *simultaneously finite* for a row to enter a gate cell. This is not a detail.
`common_scores` intersects on every column it is handed, and the harness handed
it `challenger, incumbent, seasonal_naive, persistence` — so a pair with **no
incumbent** has an empty intersection, and every cell scores `n=0` with every
score `None`. ABL-322 hit exactly this: DE and NL `wind_offshore` have 0 rows in
`forecasts`, so the first pilot run returned 0/6 cells and the harness rendered
`FAIL` — a model-quality verdict on a comparison that never happened. **Every
new country in the ABL-316 tranches is in that position**, so this would have
mis-dispositioned all 37 remaining pairs. `abl322-pilot` therefore gates on
`(challenger, seasonal_naive)` — the two columns its registered bar actually
names — and reports the incumbent and persistence on their own intersection with
that basis, each carrying its own n, so an absent comparator reads *Not measured*
instead of emptying the cell.

`abl195` deliberately **keeps** the four-way basis it was published under: its
48-64h cells scored 480 rows against the 510 the same report records as selected,
so the incumbent conjunct did drop rows there, and re-basing it would silently
move numbers that have already been dispositioned. Re-reading ABL-195 under the
narrower basis is a separate decision for whoever owns that gate.

Relatedly, a run in which any cell scores zero rows now returns verdict
`UNREADABLE`, not `FAIL`. A cell that scored nothing did not lose a race; saying
`FAIL` invites exactly the wrong next move (feature work on a model that was
never measured).

ABL-378 ported all of the above to the **solar** harness, so it is no longer the
exception this section used to describe. It takes `--scope` over a `SCOPES` table
of its own (`evaluate_solar_retrain.py:60`), registers a `GATE_BASIS` per scope
(`:98`), and derives its bar rather than hardcoding `== 9`:
`registered_cells = len(registered_countries) * len(PRIMARY_BANDS)`
(`:361`), compared in `disposition` (`:181`). `abl253` is the default and the
only registered solar scope today, so an unflagged run still reproduces ABL-253;
ABL-381's tranche registers the second.

**Neither harness fits the list `get_feature_columns()` builds.** Each declares
its own `FEATURE_COLUMNS` and hands it to `RenewableFeatureBuilder` through
`to_vector`, so ABL-394's guard — which covers the `scripts/train.py` path — did
not reach them, and nothing reviewed the harness lists. Measured on the ABL-381
read: a solar gate fit ran at **25 features where an ABL-338-current fit is 27**.
`RenewableFeatureBuilder` had emitted `sun_elevation_deg` and `is_night` for
solar since ABL-338 (`wind_features._solar_geometry_features`); only the list
never asked for them, so every read from ABL-253 through ABL-381 built artifacts
two features short while declaring nothing was missing — and CH predicted
negative in **80.5%** of night hours, the defect ABL-335/ABL-338 exist for.
ABL-395 splats `solar_features.SOLAR_GEOMETRY_FEATURES` onto the end of the list
(`solar_retrain.py:53`), so the list and the builder cannot name different
columns.

Three things follow, and the third is the one that bites:

- **This is the half of ABL-338 that was adopted.** The non-negativity
  constraint was measured and *rejected* there (+15.8% Tweedie, +36.8% Poisson
  daylight MAE), and `nonneg_objective=None` on every gate artifact correctly
  records that. Do not read ABL-395 as bringing it back.
- **The two harness lists are frozen** in `tests/feature_list_manifest.json`
  under `gate_harness`, checked by `tests/test_gate_feature_list_contract.py`,
  which also asserts the builder *produces* every declared name and that every
  `config.SUPPORTED_COUNTRIES` entry has a `solar_geometry` representative point
  — without one, `to_vector` raises and a tranche dies at its first fit row.
  Note the two paths fail in opposite directions: `select_feature_columns`
  **drops** an unproducible declared name and warns, `to_vector` **raises**.
- **A scope already read does not follow the constant.** The list moving is a
  real change to the challenger — measured, not assumed — so `SCOPE_FEATURES`
  (search the constant in `scripts/evaluate_solar_retrain.py`; it has moved twice
  and a line number here goes stale within a tranche) is a registration of the
  same kind `FIT_RULES` is, for the reason stated over that table: two gate reads
  are not comparable unless both say what they trained on. `abl253`, `abl376` and
  `abl316-t1b` pin the 25 they were read on; a scope that registers nothing gets
  the 27, which is what unblocks the remaining tranches without touching the
  table. The report and the JSON now name the set (`feature_set`, `n_features`),
  because a 25-column and a 27-column artifact are otherwise indistinguishable
  after the fact. Whether `abl253` or `abl376` is re-read at 27 is ABL-401.

  **`abl316-t1b`'s pin is ABL-404 and it was missing for two months of merges.**
  `SCOPE_FEATURES` is not one of the tables `check_registration_tables` checks —
  and after ABL-429 it is one of only two that are not — so its absence resolved
  through `features_for` to the 27 instead of aborting at import, and that scope's
  `SCOPE_OUTPUTS` row writes ABL-381's published PASS 6/6 — a `--scope abl316-t1b`
  run refitted BG and CH at the wrong challenger, overwrote the evidence in place
  under ABL-381's own heading, and exited 0. Merge order caused it (PR #40
  registered the scope, PR #46 added the table off an older branch) and neither
  merge conflicted.

- **The guard derives its scopes; do not re-hardcode them.**
  `test_a_dispositioned_scope_still_resolves_to_the_list_it_was_read_on` used to
  be `parametrize("scope", ["abl253", "abl376"])`, which is how it covered two of
  the three scopes that needed it. It now takes every scope in `SCOPE_OUTPUTS`
  whose `json_out` or `report_out` is **tracked in git** — published, not merely
  present, so a local gate run cannot promote an open scope — and holds it to the
  list that run recorded: `meta.feature_columns` where the record states it, the
  legacy 25 where it does not (ABL-395 added that field in the same change that
  made the list 27, so its absence dates the read). The rule is *dispositioned vs
  open*, **not** *pinned vs unpinned*: `abl316-t2a` is deliberately absent from
  `SCOPE_FEATURES` and inherits the 27, and is still guarded, against the 27
  literal names in its own committed record. Requiring every registered scope to
  appear in `SCOPE_FEATURES` would be wrong and would fail the suite.

`--with-geometry` on `scripts/abl376_night_seed_spread.py` is now
`LEGACY_FEATURE_COLUMNS` vs `FEATURE_COLUMNS`, not `X` vs `X + geometry`: written
the old way it would hand CatBoost both columns **twice** and label the
registered arm `legacy25` while fitting 27.

**The 80.5% that motivated the fix is one draw, not a measurement**, and the
eight-seed A/B that says so is the reason this section does not claim the fix
closed it (`scripts/abl395_geometry_feature_probe.py`,
`reports/abl_395_geometry_features.md`; one vintage frame per country, both arms
from the same retained rows, ABL-376's eight registered seeds plus the gate's
42). CH's night-negative rate over eight *control* fits — same data, same
columns, one integer apart — is **77.05% ± 10.11 with a 27.34pp single-seed
null**. Both 80.47% (f25) and 64.06% (f27) at seed 42 sit inside it; the paired
change is −3.85pp at 4/8 seeds. **Do not quote a one-seed night-hour fraction as
a defect measurement**, here or in ABL-381 §4.

What *is* readable is small and on the accuracy axis: CH loses 0.23-0.24pp of
WAPE on the two longer bands, **8/8 seeds**, sign p = 0.0078, and identically on
a daylight-only re-score, so it is not night rows flattering a denominator. BG
moves the other way (+0.44pp, 6/8, p = 0.29 — not significant), and the
prediction that explains it is BG's own data: ABL-381 §5 measured 76-85% of BG's
night hours carrying 152-246 MW, so `is_night` tells the model the sun is down on
hours the target books at 225 MW, where CH's night actuals are exactly 0.00.
**Screen a country's night floor before reading its solar gate** — the geometry
pair is a physical prior and is worth what its actuals' respect for that physics
is worth.

**Every read reports four model-free references** (ABL-389). `constant_causal` and
`constant_oracle` are a flat line at the **fit-window mean** — the honest "no
model" floor, using only what was knowable before the gate window opened — and at
the **gate-window median**, the hindsight upper bound on what *any* constant could
achieve. `climatology_causal` and `climatology_oracle` are the same two forms taken
**per hour of day**. All four are in `REPORTED_COMPARATORS`
(`evaluate_wind_retrain.py:207`, `evaluate_solar_retrain.py:134`) and defined once
in `src/evaluation/model_free_reference.py`, so the two harnesses cannot compute
the same named reference differently.

They exist because **the registered D-7 bar certifies close to nothing on a
low-capacity-factor pair**. ABL-380 passed 6/6 and reported, against its own pass,
that CH `wind_onshore` cleared all three cells at 47.42% WAPE while a constant at
the gate-window median scored 40.29% — the fitted model was 7.1pp *worse than a
flat line* — and that BG's registered D-7 bar of 93.75% is cleared outright by a
causal constant at 82.77%, with no model at all. Both numbers reached the evidence
pack only because a human went looking. `lost_to_a_model_free_reference`
(`model_free_reference.py:289`) now names such cells in the report unprompted, per
oracle, because losing to the level and losing to the average day are different
statements about a model.

**The climatology is there because the constant alone was measured and found
insufficient.** On solar a flat line scores 63–95% WAPE on every cell — it cannot
represent a diurnal cycle, and on solar the diurnal cycle is the signal — so it is
a comparator the challenger cannot lose to, which is the ABL-380 defect one level
up. An hour-of-day predictor is the tighter reference on **both** technologies,
because a constant is a climatology with one bucket. Measured against the replica
over ABL-348's windows on 2026-08-13, whole gate window per pair:

| pair | const causal | const oracle | clim causal | clim oracle |
|---|---:|---:|---:|---:|
| BG solar | 75.30% | 73.49% | 41.98% | 19.15% |
| CH solar | 95.08% | 94.65% | 37.53% | 9.02% |
| BG `wind_onshore` | 82.77% | 63.78% | 81.03% | 62.50% |
| CH `wind_onshore` | 79.07% | 40.29% | 77.82% | 38.20% |

So CH wind's challenger loses to the oracle climatology by **9.2pp**, where the
constant put the gap at 7.1pp. Keep both: the constant asks whether a model
predicts the *level*, the climatology whether it predicts the level *and the daily
shape*, and the gap between them is how much of the series is forced diurnal
structure — ~1.5pp on CH wind, ~86pp on CH solar.

**These are reported references and never gate criteria.** They are in no
`GATE_BASIS` entry, and a pair that clears D-7 while losing to one still reads
`PASS` — beside the number that qualifies it. Moving a bar after seeing a result is
what the pre-registration apparatus exists to prevent, and a conservative direction
does not exempt it; `tests/test_gate_model_free_reference.py` pins both halves,
reading `GATE_BASIS` from the *source literal* via `ast` rather than through the
imported module.

Each reference is attached as a **column** (`attach_model_free_references`) and
scored by the same path `seasonal_naive` and `persistence` take, not special-cased
inside the scorer — which is what preserves the ABL-322/ABL-378 property above. A
window holding no finite observation yields no level, an all-NaN column and `n=0`,
and reads *Not measured*; it never becomes a flat line at zero. The `scored`
closure both harnesses duplicated is now `scored_with_comparators`
(`src/evaluation/wind_retrain.py:113`).

**A climatology is 24 levels, so it is the first comparator that can be *partially*
measured.** An hour of day absent from its source window leaves those rows NaN;
they drop from that column's own intersection and lower only its `n`. Nothing is
filled from a neighbouring hour — that would be interpolating to close a visual
gap. **Read a climatology's `comparator_n` before comparing its WAPE to the
challenger's**: scored on different rows, they are not the same measurement. The
markdown levels table prints an `h` count per pair for exactly this, and anything
below 24 means rows were dropped.

### A PASS is graded, not just recorded (ABL-418)

**The bar is not re-opened.** Seasonal-naive D-7 is still the registered gate for
every scope already dispositioned and every scope still to come; ABL-348's frozen
windows, bands, metric, minimum n and source are unchanged, and a cell that clears
D-7 still reads `PASS`. What ABL-418 registers is **what that PASS entitles a cell
to**, because ABL-406 measured that on these pairs it entitles it to less than it
looks like: across 8 `wind_onshore` pairs the gate outcome was *fully* predicted
by whether a causal constant clears the bar on its own — 5 weak bars gave 5
passes, 3 strong bars gave 3 failures or ties, no exceptions — and NO passed 3/3
while **anti-correlated with its own target** (slope −0.08, corr −0.14). A PASS is
necessary and not sufficient. Tightening the bar after seeing that would be
shopping the registration; grading the pass is not, which is why the ladder was
pre-registered before the remaining tranches were fitted.

`src/evaluation/gate_grading.py` is the one implementation, imported by both
harnesses exactly as `model_free_reference.py` is. Per cell, from columns the
gate table already prints — no new baseline, no new fit:

| | test | from |
|---|---|---|
| **G1** gate | beats `seasonal_naive` by more than the readability floor | `skill vs D-7` |
| **G2** level | beats `constant_causal` | already printed |
| **G3** shape | beats `climatology_causal` | already printed |
| **G4** direction | `slope > 0` **and** `corr > 0` | already printed |

**A** = all four in every band (promotion-eligible, subject to any named data
hold); **B** = G1 holds, one or more of G2/G3/G4 fails, named; **C** = a readable
loss to D-7; **U** = the G1 margin sits inside the floor, so the cell is
unreadable at one seed — **U(+)** where G2–G4 clear readably, meaning *re-read at
k>1 seeds*, not *reject*. A pair takes the worst of its bands, `C` > `B` > `U` >
`A`. **`U` outranks `C`**: both are "G1 does not hold", but an unreadable margin
and a measured loss are different statements, and reporting the first as the
second invites the feature work `UNREADABLE` exists to prevent.

Three things about it are load-bearing.

- **The floor is ABL-385's `delta_min` with `c_B = 0`, not the published two-arm
  number.** Every reference on the ladder is *deterministic* — D-7, a flat line
  and an hour-of-day climatology do not move when the challenger is refitted — so
  the two-arm margin is a factor of √2 too wide, and the floor is `1.96 · c` =
  **10.65% on solar, 7.51% on wind** at the fleet p90 CV and one fit per cell.
  Quoting 15.06% against a constant is not conservatism, it is the wrong test.
  The two per-stream CVs are checked against `reports/abl_385_decision_margin.json`
  itself rather than retyped, because retyping is how ABL-381 came to quote
  another stream's margins. `GRADE_STREAM` in each harness picks the stream and
  is read out of the AST by the test.
- **Which denominator was a real choice, and it is reported rather than
  assumed.** G1 is registered on the printed `skill vs D-7` column,
  `1 − challenger/reference`; ABL-406 quoted its margins on the challenger's
  *own* error, `reference/challenger − 1`, which is the denominator the CV is
  measured in. They always agree in sign, so they can disagree only near the
  floor. Measured over both tranches: **no cell of the 48 changes grade**, and
  none sits between the exact floor and the 2-dp value published in prose.
- **Causal references only.** The two oracle references stay reported and gate
  nothing — an oracle is not causally available, so losing to one bounds what a
  verdict means rather than voiding it — and the bar-weakness flag is kept for the
  same reason. A condition that could not be *measured* is not satisfied and is
  named like any other failure (the net-position gate's `INCOMPLETE` rule).

`reports/abl_418_retro_grade.md` retro-grades tranches 2a and 2b from their
stored `results_*.json` — arithmetic only, no refit, generated by
`scripts/abl418_retro_grade.py` rather than restated in prose. **2a solar:** A ×
7 (BG, CH, CZ, PL, RO, SI, SK), **U(+)** HU — whose 4.6/4.6/7.6% skill was
published as a clean PASS against a floor the same pack registers at 10.65%.
**2b wind:** A × 4 (FI, GR, PL, SE), **B** NO (fails G4), **C** ES and PT,
**U(+)** IT. IT is the one cell where the ladder differs from the reading in the
ABL-418 description (`U` there): its G1 margin is inside the floor either way,
but it clears G2, G3 and G4 readably, so it is *re-read*, not *do not decide* —
while losing readably to both oracles, which is the qualifier that travels with
it. Neither tranche's verdict, report or results file moves; the grades land
under a new path and the six dispositioned scopes are byte-unchanged by blob
hash.

**A scope also registers where it writes** (ABL-387). `--artifact-dir`,
`--json-out` and `--report-out` used to carry fixed ABL-195/ABL-253 defaults,
which `argparse` resolves *before* `--scope` is consulted — so a scoped run that
omitted three flags overwrote a dispositioned gate read in place, succeeded, and
emitted a full report. Each harness now has a `SCOPE_OUTPUTS` table beside
`SCOPES`/`GATE_BASIS` (`evaluate_wind_retrain.py:112`,
`evaluate_solar_retrain.py:86`); the three flags default to `None` and resolve
against it after parsing, so an explicit path still overrides. `abl195` and
`abl253` keep their historical paths byte-for-byte. Those three tables are one
registration in three views (five on solar since ABL-429 — see below) and are
cross-checked at **import** by
`check_registration_tables` (`src/evaluation/gate_registration.py:39`, called at
`evaluate_wind_retrain.py:285` and `evaluate_solar_retrain.py:536`), so a scope
added to one and not the others fails before any fit rather than mid-run — it
raises on `import`, so even `--help` exits non-zero. That is deliberately louder
than a failing test: the tables disagreeing is **not** a textual conflict, so
GitHub reports such a merge `MERGEABLE / CLEAN` and no merge-order check on the
platform will show it.

**Registering a new scope means editing every registration table, and since
ABL-429 five of the seven are import-checked on solar.** The call is
`check_registration_tables(SCOPES=..., GATE_BASIS=..., SCOPE_OUTPUTS=...,
FIT_RULES=..., SCOPE_TITLES=...)`. **The two harnesses' calls now differ, and
that is not one twin missing a fix** — the recurring failure mode this pair has
(ABL-322/ABL-379, ABL-345/ABL-347): wind carries only the first three tables at
all, so all three of its tables are checked. Solar carries seven and two stay
out, each for a stated structural reason:

- `SCOPE_FEATURES` **cannot** join. `abl316-t2a`'s absence from it is correct and
  published — inheriting the current `FEATURE_COLUMNS` is the intended path for a
  new tranche — so requiring it would raise at import for a scope that is right.
- `SCOPE_NOT_EVALUABLE` is the one to check hardest, because it is the only
  remaining table that defaults **toward scoring**: a scope that forgets it scores
  every cell it can build, which for a pair ABL-348 declares NOT-EVALUABLE is a
  wrong verdict rather than self-documenting degradation.

So a scope missing from either of those two still resolves through a module-level
default **silently, at run time** — exactly how ABL-404 happened, which is why the
rows that depend on it each carry a comment saying so.

**What the check enforces is presence, not content.** It compares the tables'
**keys**; it never looks at a value. A tranche that registers
`exclude_impossible_night: True`, or a wrong title, imports and runs and exits 0
like a compliant one. Enforcement buys you "somebody wrote a row here" and
nothing more — the record of *what was chosen and why* is still the comment beside
the row, and for `FIT_RULES` it is pinned by
`tests/test_abl403_fit_rule_registration.py`.

Adding a table to the check is not free: it raises on `import` for every branch
already in flight, which is why ABL-429 waited for both repo queues to reach zero.
Read the `check_registration_tables(...)` call in the harness you are editing
rather than this sentence; that call is the list, and it is still shorter than the
set of tables you must edit by hand.

> **Count the tables the same way you are told to count the call — and check the
> recipe, not just the number.** This paragraph said **five** when the file
> carried six, and ABL-421 re-counted against the source to reach seven. But the
> recipe it left behind, `grep -E "^[A-Z_]+ = \{"`, returned **9** at the very
> commit that called it "the count": it also matches `DEFAULT_FIT_RULES` (keyed by
> rule name, not by scope) and `NOT_EVALUABLE_CAUSES` (keyed by country). The
> number was right and the recipe was wrong, which is the worse half — the recipe
> is what the next editor actually runs. Run the grep, then subtract any table not
> keyed by **scope name**; today that is those two, leaving seven.

**`SCOPE_NOT_EVALUABLE` is the exception to watch, because it defaults toward
scoring (ABL-421).** ABL-348 `not_evaluable` declares `EE/solar` and `FI/solar`
unscorable on 24-36h and 36-48h, before any fit existed, with a rule the harness
had no way to obey: *"It is not a FAIL and must not be counted as one; a gate
read that scores it has misread this registration."* `gate_cell` builds a cell
for every country-band that yields rows and marks it `pass: False` when `n` falls
under the registered minimum — so those four cells arrive as ordinary *failed*
cells and are counted into the bar. Tranches 2a-2c dodged this by excluding both
pairs; tranche 2d is the one they belong to. A declared cell is now subtracted
from `registered_cells` and routed to a `not_evaluable_cells` list that `passed`,
`disposition` and `attach_grades` never read — still measured and printed, so the
declaration is auditable, but carrying no gate outcome and no grade. Three things
follow:

- **The table is a transcription, not a discretion.** A scope that could declare
  its own cells unscorable is a scope that can drop whatever scores badly.
  `tests/test_abl421_not_evaluable.py` derives the declaration from
  `experiments/ABL348/config.json` and compares, so it can only ever mirror the
  pre-registration.
- **Only the bands the registration names.** ABL-348's `note_48_64h` says the
  48-64h band scales proportionally rather than being hard-bounded by
  `n_d7_scorable`, and that a declared pair "may still clear 456 in that band and
  should be reported if it does" — so 48-64h stays on the bar for both pairs.
  Where such a cell falls short it is a **coverage shortfall**
  (`enough_pairs: False`), not a loss to D-7; the cell dict carries the two flags
  separately.
- **Only one of the two shortfalls is ours.** EE's is an ABL-188 bit-identical
  zero run present in *both* source tables, so reverting the source would not
  recover it. FI's is `energy_generation` holding 663 of 720 gate hours against
  `energy_renewable`'s 717 — `source_dependent: true`, a cost of ABL-348's source
  change and a finding for whoever owns that decision rather than a fact about
  FI's model. The `source_dependent` flag is asserted by the same test for
  exactly this reason.

**What the fit was allowed to see is part of the registration too (ABL-376).**
`FIT_RULES` (`evaluate_solar_retrain.py`) carries `exclude_impossible_night` per
scope: a night row — night by `solar_geometry.is_night_hour`, the serving clamp's
own predicate, reached through `solar_features.night_mask` — whose actual exceeds
`IMPOSSIBLE_NIGHT_THRESHOLD_MW` (1 MW, ABL-338's threshold) is dropped **from the
fit and never from the score**. `energy_renewable` carries solar for FR at sun
elevations down to -65 deg, so a model fitted through it learns a night floor
faithfully; the defect is in the training target, not the model.

That asymmetry is the rule, not an implementation detail. We refuse to train on
values the sun says are impossible and still score against whatever the source
reports, so the challenger cannot delete the rows it is held to account on. A run
that filtered its own gate frame would fit, score, render every number and pass
every other test, so the call site is pinned by AST in
`tests/test_solar_night_fit_exclusion.py` rather than by any output.

The rule is stated over countries, not for FR — the predicate is the sun's, so a
country whose data is clean loses nothing, and a `0` in the report's per-country
table means the rule ran and found nothing rather than that it was off. There is
one country it may not run for at all: **`exclude_impossible_night_rows` raises
`IncoherentNightExclusionError` for any country registered `True` in
`solar_geometry.NIGHT_GENERATION_POSSIBLE`** (ES, ABL-425). The rule's warrant is
"the sun says this row cannot exist", which is false by measurement for a fleet
that dispatches stored heat after sunset, and no evidence can make the
combination coherent — so it is refused at the one choke point that drops rows
rather than resolved to one side. That guard changed no registered rule value;
ABL-403's are as they were. It is conservative by construction: `is_night_hour`
requires the sun below threshold for the *whole* hour, so shoulder contamination
survives it. The threshold and
the per-country row count are printed in the scorecard so a later run can tell a
data fix from a rule change. `abl253` registers the rule **off** and keeps its
report heading character-for-character, so the dispositioned read still
reproduces; `abl376` is the same countries, basis and windows with the rule on —
a controlled A/B on the rule alone. Do not re-read a dispositioned scope under a
changed fit rule; register a new one.

**Leave that rule off, and know why (ABL-403).** The 2x2 the ABL-395 handover
asked for — geometry (25/27) x the rule (off/on), BG and CH, ABL-376's eight
seeds, 64 fits — measured what it costs on a country whose night rows carry real
MW. On **BG the rule alone doubles night MAE, 44.8 -> 105.9 MW at t = +9.6, 8/8
seeds**, drives night bias from -2.1 to +88.5 MW, costs **1.4-1.9pp of gate-band
WAPE** and eats **47% of the D-7 margin** ABL-405's PASS was carrying (+4.99pp ->
+2.63pp at 24-36h; still clears at 8/8, so this is cushion, not a flip). CH
measures nothing on any exclusion contrast. `reports/abl_403_night_rule_interaction.md`.

Three things follow, and the first is the general rule:

- **A fit-side exclusion is only defensible when the excluded rows are both
  genuinely contaminated *and* a small enough minority that the score is not
  dominated by them.** The asymmetry above keeps the rows in the score by design;
  on FR that meant refusing 113 targets, and on **BG it means refusing 76.4% of
  the night fit rows while 25.3% of the scored gate rows are night rows at a
  225 MW mean**. You cannot forbid a model to learn what you still grade it on
  once that is a quarter of the score. Contaminated actuals are an upstream
  repair (ABL-67/ABL-210's "repair beats delete"), not a fit filter under an
  unchanged score. This holds whether or not BG's floor is genuine — grant
  ABL-396 §9.3 that it is contaminated and the 1.4-1.9pp is still the cost.
  ES is the strictly stronger case: its overnight MW is real CSP dispatch, so
  the rule would delete generation rather than noise.
- **Never disposition a night-floor change on the negative-prediction rate.** It
  cannot see the level, and on BG it cannot be read at all. Over the same eight
  paired fits, night MAE rises **+61.05 MW** (rule at 25 features, 8/8 seeds,
  p = 0.0078) against a 6.96 MW control-vs-control null — readable — while **not
  one** negative-rate contrast clears its own null of **14.06pp**: the rule's
  apparent *improvement* is -7.12pp at 25 features (7/8, p = 0.070) and -11.78pp
  at 27 (8/8, p = 0.0078), and both sit inside the noise. So the metric moved the
  way that would have adopted the rule, on fits where the level metric says the
  rule roughly doubles the error, and it did so without being readable in the
  first place. That is the metric ABL-381 §4 and ABL-395 both reported. Report
  night MAE and night bias beside it, and read `outside_the_null` before quoting
  any of the three — an 8/8 sign test is not readability when the single-seed
  null is wider than the effect.

  Quote the two factors' contrasts, never the 25-off -> 27-on diagonal. Those
  endpoints (20.09% -> 9.85%) differ by *both* changes at once, so they are not a
  measurement of either; the machine record keeps them apart as
  `exclusion_at_f25` / `exclusion_at_f27` / `both_vs_neither` for that reason.
- **ABL-376's 27x mechanism is real in structure and useless in direction.** The
  interaction on night MAE is -14.2 MW (7/8 seeds, sign p = 0.070, clearing a
  conservative 4-fit null of 11.3): geometry makes the rule do *less damage*, not
  make it work. On the night-negative axis no interaction is readable at all. And
  ABL-395 §5c's proposed mechanism for BG's +0.44pp geometry regression is
  **tested and not supported** — removing the "lying" night rows roughly doubles
  that regression (+0.46 -> +0.91pp at 24-36h, 6/8 -> 8/8) instead of curing it.

**A one-seed solar A/B on this harness cannot resolve anything under ~5%
(ABL-376 §5).** Refitting the solar gate's CatBoost at eight seeds, changing
nothing else, moves daylight MAE by up to **4.4% (FR), 3.7% (DE) and 5.4% (BE)**
between two seeds — the same order ABL-375 measured on DE. So a gap quoted from
a single fit per arm is not a measurement, and both of this rule's headline
numbers dissolved when one was run: FR's night level moved −0.33 MW against a
within-arm spread of 19.6, and its daylight MAE moved the *wrong* way by 0.38%.
Pair the arms by seed — same seed, same frames, one integer apart, so the
across-seed variance cancels inside the difference — and quote the effect
against a null built from every control-vs-control seed pair, which is what a
single-seed gap looks like with nothing changed at all.
`scripts/abl376_night_seed_spread.py` is the worked example; it builds each
country's frames once and refits around them, which is what makes 16 fits per
country affordable (~4–5 min of building, ~4–5 s per fit).

A corollary worth keeping: **a fit-side rule can only move what the feature
vector can represent.** The same exclusion is 27× more effective on FR's night
level once `sun_elevation_deg` and `is_night` are in the vector (−8.81 MW, 7
seeds of 8) than on the gate's 25 legacy columns (−0.33 MW, 5 of 8), because
nothing in those 25 distinguishes "0 W/m² because the sun is down" from "0 W/m²
at a dark winter dawn". Before concluding a target-side fix does nothing, check
the model has a handle for the thing you removed.

**Which way the two `.gitignore` globs cut — they do not cut the same way.**
Entries stay exactly one directory deep under `experiments/`, and below that the
resemblance ends. `.gitignore:56` (`experiments/*/artifacts/`) matches on the
**directory name**, so any one-level path ending `artifacts` is ignored and no
`artifact_dir` is committable. `.gitignore:53` (`experiments/*/results.json`)
matches on the **exact filename**, so a one-level `json_out` named anything else
is **tracked**. Depth alone therefore does not decide tracking, and both
conventions are live:

| scope | `json_out` | tracked? |
|---|---|---|
| `abl195`, `abl253`, `abl322-pilot` | `experiments/<ID>/results.json` | no — ignored at `.gitignore:53` |
| `abl380-tranche1a` | `experiments/ABL348/results_abl380_tranche1a.json` | **yes** |

**Prefer the tracked form for any new scope whose read will be dispositioned.**
An ignored `results.json` is the one gate record `git checkout --` cannot recover
and a reviewer cannot diff, which is the same blind spot that made this issue's
failure mode unobservable: an overwritten gate read shows nothing in
`git status`, no conflict, no reviewer signal. `abl195`/`abl253` keep the ignored
form only because relocating them would break the path every already-published
report cites. Do not rename `abl380-tranche1a`'s `json_out` to `results.json` for
consistency — that silently untracks the machine record
`reports/abl_380_tranche1a_findings.md:9` cites for a PASS the Board was asked to
review, and `tests/test_gate_scope_outputs.py` pins against it.

Why the source matters for the 37 unmodelled solar / wind_onshore pairs, measured
on the replica 2026-08-12: **33 of the 37 have under 365 days in
`energy_renewable`** (median 276 d), while **37 of 37 have over a year in
`energy_generation`** (median 2,049 d). Only BG and CH reach 2021 in both. A
harness pinned to `energy_renewable` gates those pairs on a model that has never
seen a full seasonal cycle.

`--replica-db` governs the whole run in both harnesses — since ABL-355, and not
before it. It used to cover only the incumbent, TSO and contamination reads: the
builder went through `db.get_connection()` and so opened
**`config.DATABASE_PATH`** (`ENERGY_DB_PATH`), so one run could fit a challenger
on one file, score it against an incumbent from another, and print a single path
under `Replica:` as if it were the source of everything. `get_connection` now
takes a read-only `db_path` (`src/db.py:33`) threaded through
`load_renewable_type_data` (`src/db.py:527`) and `RenewableFeatureBuilder`
(`src/wind_features.py:516`), and both harnesses hand it the resolved
`--replica-db` (`scripts/evaluate_solar_retrain.py:374`,
`scripts/evaluate_wind_retrain.py:378`). A write connection **refuses** a
`db_path` rather than honour or ignore it, so the sidecar guard keeps its single
rule. `meta['databases']` records every file the run opened
(`src/evaluation/scorecard.py:193`) and the report names them, including an
`ENERGY_DB_PATH` that differs and was *not* read.

So the gate harnesses no longer need `ENERGY_DB_PATH` at all when `--replica-db`
is passed. Omit both from a worktree and the run refuses at argparse — the flag
defaults to `str(config.DATABASE_PATH)`, which is the degraded bare
`\data\energy_dashboard.db` that does not exist — rather than fitting against
whatever the environment happened to name. Serving passes no `db_path` and still
reads `config.DATABASE_PATH`; this is an override for callers that have already
resolved a file, not a new default.

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
- days_to_holiday - Days until next holiday (capped at 7, `src/features.py:177`)
- days_from_holiday - Days since last holiday (capped at 7, `src/features.py:185`)
- is_bridge_day - Workday between holiday and weekend

> **Declared, but in no serving artifact** (ABL-386/ABL-394, measured 2026-08-13;
> mechanism corrected by ABL-407). All 66 artifacts that carry a
> `feature_columns` list carry none of these four, and dropping exactly those
> four reproduces the served list length on all eight types
> (23/23/26/25/27/25/24/24) — one plumbing gap, not eight drifts. They are live
> for the **next** fit of any country and have never been evaluated on any target
> — ABL-386's read on solar is MIXED. The frozen lists and the recorded gap are
> in `tests/feature_list_manifest.json`; the narrowing now warns instead of
> dropping silently (`select_feature_columns`, `src/features.py:534`).
>
> **Why they are missing is provenance, not a regression.** Do not repeat the
> earlier story that ABL-338 (`5cf2296`) threaded `country_code` into
> `create_all_features` and so made them live; it does not reproduce.
> `git show 5cf2296 --stat -- scripts/train.py` is empty, and at `5cf2296^` the
> training site already read
> `create_all_features(df, forecast_type, country_code=country_code)`. Both the
> four names and that threading trace to `996c45a` *Initial commit*, 2026-03-05.
> The one pre-ABL-338 site that omitted `country_code` was
> `evaluate_against_baselines` — the **validation** frame, which writes no
> artifact's `feature_columns`; that is **ABL-397**, and it is a different defect.
> 60 of the 66 artifacts were saved 2025-12-26..2026-02-23, before this repo
> existed, so no current training path produced them. The remaining **6**
> (BE/DE/FR × load, price) were saved 2026-04-04, a month *after* the migration,
> and still carry none of the four — for those the cause is **not established**.
> Full measurement: `reports/abl_407_holiday_gap_provenance.md`.
>
> **This list is not what the two gate harnesses fit.** They declare their own
> `FEATURE_COLUMNS` and never call `get_feature_columns()`, which is how the solar
> harness came to sit two names short of an ABL-338-current fit until ABL-395 —
> see "Neither harness fits the list `get_feature_columns()` builds" above. Both
> harness lists are frozen in the same manifest, under `gate_harness`.
>
> **The solar null does not transfer to load** (ABL-393).
> `scripts/abl338_solar_holdout.py --type load|price` fits the same arms on the two
> aggregate targets, paired by seed over the standing eight seeds — the instrument
> ABL-386 named as its own weakest. Registration `experiments/ABL393/config.json`,
> verdict and numbers `reports/abl_393_load_price_holiday_verdict.md`. **Do not
> read ABL-386's MIXED as covering the other seven types**: it was registered on a
> target whose prior was "no effect" — solar output is set by irradiance — and it
> says so. On load the prior is the opposite, and `control_noholiday` there is
> *exactly* the serving list (26 names on load, 25 on price, all 48 artifacts, name
> for name and in order), so the contrast is what is served against what the next
> retrain builds, with nothing else moving.

Three things about that read are reusable and would otherwise be re-derived:

- **`create_lag_features` shifts by rows, so a source gap poisons the fortnight
  after it.** `days * 24` is a day only on a gapless hourly frame, and
  `energy_price` is not gapless: measured 2026-08-13, AT is missing **2,236 h** and
  DE **2,483 h**, almost all of it 2025-09 to 2025-12 (AT's largest single hole is
  1,651 h, DE's 1,309 h), while `energy_load` over the same span misses one 27–29 h
  outage on 2026-02-15 common to all four majors plus 26 h of FR over New Year
  2026. A holdout placed within 14 days of a hole scores rows whose D-1/D-7/D-14
  lags reach across it. This is what disqualified December for price in ABL-393 —
  AT and DE are 67.3% covered there — and `reports/abl_393_source_gaps.json` is the
  regenerable inventory. **Check it before choosing a window**, on either table.
- **December is not the densest holiday window of the year**, for three of the four
  majors. Measured on the `holidays` calendar: 2025-12-06..2026-01-18 holds AT 5,
  BE 2, DE 3, FR 1 holiday days against 2026-04-30..2026-06-12's AT 4, BE 4, DE 3,
  FR 4 — Labour Day, Ascension, Whit Monday, FR's 8 May and AT's Corpus Christi all
  fall in the second. What December has instead is a contiguous low-demand
  fortnight, which `days_to_holiday`/`days_from_holiday` mark and a count of red
  days does not.
- **A holiday is 2–5 days in a 44-day window, so an all-hours mean dilutes a
  holiday effect roughly twentyfold.** `--holiday-subsets` scores each arm over
  `holiday`, `holiday_affected` (holiday, bridge day, or within a day of one) and
  `ordinary`, from `src/features.holiday_subset_masks` — one predicate, shared with
  the pre-fit density probe, so a window cannot be registered under one definition
  and read under another. The two subsets partition the holdout and MAE × n is a
  sum of absolute errors, so their gains add to the total exactly: **which subset
  the gain lands in is the internal check on any headline here.**

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

**Scoring truth lives in one dict, `scorecard.ACTUAL_SPECS`, and since ABL-410
the renewable family reads `energy_generation` — the same table the dashboard
publishes against.** Before that it read the frozen `energy_renewable` while the
dashboard had moved (ABL-399), so one model, country and window had two
published WAPEs and neither was wrong. Three things to hold onto:

- This is **not** ABL-321's rejected switch. That is the *training* source,
  `db.RENEWABLE_TYPE_SOURCE_TABLE`, still `energy_renewable` and untouched.
  Scoring truth and training source are independent post-ABL-331.
- It touches **no promotion gate**. `ACTUAL_SPECS` is read only by
  `scorecard._load_actuals`; both gate harnesses take actuals from
  `RenewableFeatureBuilder` → `db.load_renewable_type_data`.
- `hydro_total` is `db.RENEWABLE_TYPE_COLUMNS['hydro_total']` **imported, not
  restated**. A strict `hydro_run_mw + hydro_reservoir_mw` is survivable on the
  frozen table only because `REAL DEFAULT 0` means nothing there is NULL; on
  `energy_generation` it erases the 9 countries that report one component.

Two caveats travel with every renewable-family figure: `energy_generation` has
an open FR ingest gap (2026-06-30 → 2026-07-22, ABL-318 §3) that shrinks FR
samples and therefore moves **pooled** rows on composition alone; and the models
are still fitted on `energy_renewable`, so where the tables disagree about the
target, part of the WAPE is target mismatch. `reports/abl_410_scoring_truth.md`
decomposes both, and records the finding that BE `hydro_total` is a
pumped-storage forecast under a hydro label.

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
