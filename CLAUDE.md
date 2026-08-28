# CLAUDE.md

Guidance for Claude Code working with the energy forecasting module.

## How to maintain this file

This file auto-loads into every agent context; its size is a per-turn tax on the
whole fleet (it once reached 2,290 lines — see the ABL-536 precedent in the
sibling frontend repo).

- **Hard budget: 700 lines / 40 KB.** If an edit would cross it, move material
  to `docs/claude/` first.
- **Durable rules only** — commands, invariants, gotchas, each stated once.
  Incident narratives, dated measurements and per-issue forensics go in the
  matching `docs/claude/` topic file.
- **Correct in place.** When a rule changes, rewrite it; never append a
  "this used to say…" paragraph. History lives in git and `docs/claude/`.
- **Baselines and counts rot** — keep the command or test that re-derives a
  number, not the number (this file repeatedly carried stale table counts; the
  authority is named below where it matters).
- `tests/test_abl403_reported_numbers.py` parses this file: the ABL-403
  night-floor doctrine passage below is load-bearing, and the test splits on
  that passage's opening words — never quote its first sentence elsewhere in
  this file. Run the suite after editing it.

## Module Overview

D+2 energy forecasting for European electricity markets: 24-hour forecasts for
the day after tomorrow, 24 countries (AT BE BG CH CZ DE EE ES FI FR GR HR HU IT
LT LV NL NO PL PT RO SE SI SK). Forecast types: load, price, renewable (total),
solar, wind_onshore, wind_offshore, hydro_total, biomass, and net_position
(Chronos-2 only).

`scripts/scheduler_setup.sh` installs `forecast_daily.py` at 18:00, but the
Chronos-2 net-position runs are generated ~**06:00 UTC**, scheduled elsewhere.
`RUN_HOUR` in `compare_experiments.py` tracks that measured time — check it
against real `generated_at` values before trusting a backtest.

## Architecture

```
energy_forecast/
├── config.py               # paths, countries, model params, MODEL_RUNNERS
├── src/
│   ├── db.py                   # DB operations; hourly aggregation contract
│   ├── data_quality.py         # ABL-188/200 training-data guards
│   ├── features.py             # feature engineering (holiday features incl.)
│   ├── solar_geometry.py       # sun elevation per country; NIGHT_GENERATION_POSSIBLE
│   ├── solar_clamp.py          # serving-path night mask + non-negativity floor
│   ├── tso_plausibility.py     # ABL-431 read guard for TSO forecast tables
│   ├── wind_features.py        # RenewableFeatureBuilder
│   ├── metrics.py, forecaster.py, hyperopt.py, feature_selection.py,
│   │   validation.py, baselines.py, model_registry.py, deployment.py
│   ├── runner_report.py        # FORECAST_RECORDS= contract (imports typing only)
│   ├── challengers/            # v014 (XGBoost per-country), registry.py
│   ├── evaluation/             # net_position eval, scorecard, gate_grading,
│   │                           #   model_free_reference, gate_registration, ...
│   └── chronos2/               # engine, input_builder, finetuner, covariate_mapper
├── scripts/                # train.py, forecast_daily.py, forecast_chronos2.py,
│                           #   evaluate_*.py gate harnesses, compare_*.py,
│                           #   workstation/*.ps1 scheduled jobs
├── experiments/            # V001-Vnnn + ABLnnn registrations (config.json)
├── models/                 # artifacts (gitignored!)
├── reports/                # evidence packs; net_position_eval/ is gitignored
└── tests/                  # the suite; testpaths pinned here
```

### Importing this repo

`src/` is a package, always imported as one. One shape only:

```python
sys.path.insert(0, str(Path(__file__).parent.parent))   # repo root, NOT src/
import config                                            # top-level, at the root
from src.db import load_training_data                    # package-qualified
```

Inside `src/`, siblings import **relatively** (`from .db import ...`); never
`import db`. Anything in `src/` with a `__main__` runs as
`python -m src.<module>`, never `python src/<module>.py` —
`build_runner_command` (`scripts/forecast_daily.py:189`) is the one place that
builds runner argv and turns `src/` scripts into `-m` form. `src/evaluation.py`
is dead code (shadowed by the `src/evaluation/` package).
`tests/test_script_imports.py` holds the line, including launching every
`config.MODEL_RUNNERS` entry with `--help`.

### Help text is ASCII; report bodies are not

`--help` output must be plain ASCII: through a pipe, stdout falls back to the
locale codepage and argparse dies with `UnicodeEncodeError` inside the help
action. **A module docstring is help text** where it is passed as
`description=__doc__`. Write `->` and `--`. Report bodies keep `Δ`/`→`/`·` by
re-encoding at their known print sites; do not force UTF-8 on stdout at import
time. `tests/test_help_text_encoding.py` holds the line (AST sweep over every
entry point, incl. `MODEL_RUNNERS` scripts under `src/`).

## What a runner reports

The exit code says whether a runner crashed, not whether it produced anything.
Every external `MODEL_RUNNERS` entry emits `FORECAST_RECORDS=<n>` once per run,
zero included — `src/runner_report.py` owns both ends (`emit_record_count` /
`parse_record_count`). `forecast_daily` distinguishes:

| outcome | means |
|---|---|
| `success` | exit 0, reported ≥ 1 row |
| `empty` | exit 0, reported 0 rows — ran fine, produced nothing (not a failure) |
| `unreported` | exit 0, no count line — unknown, and unknown is **not** 0 |
| `failed` | non-zero exit, timeout, or exception |

Adding a runner: call `emit_record_count(len(df))` on every exit-0 path,
*before* any `if not df.empty:` guard (`tests/test_runner_reporting.py` checks
statically). **Skipped is a flag, not a phrase**: `result['skipped']` is set at
the one place that knows (`Forecaster.load`'s `FileNotFoundError`), never by
grepping error text. `chronos-bolt-small` is genuinely unrunnable on this box
(venv missing), so a default BE/price run ends `Skipped: 1, Failed: 1` and
exits 1 — fix `config.MODEL_RUNNERS` or set `enabled: False`, and never read
the old exit 0 as the job having been fine.

## Database

Two files; pointing at the wrong one is the trap (ABL-73):

| role | path | env var |
|---|---|---|
| **replica** (read) | `C:\Code\able\data\energy_dashboard.db` | `ENERGY_DB_PATH` |
| **sidecar** (write) | `C:\Code\able\data\forecasts_local.db` | `FORECAST_OUTPUT_DB` |

Replica refreshed 07:00 by `able-db-sync`; forecast job at 08:00.
`scripts/workstation/run-net-position.ps1` sets both. **All writes go to the
sidecar; the replica is a read-only mirror of prod.**

- **That claim is conditional**: `src/db.py:48` resolves the write target as
  `FORECAST_OUTPUT_DB or DATABASE_PATH`, and with the variable unset every
  write falls through to **the replica**. `scripts/train.py` and
  `forecast_challengers.py` refuse the unset case (exit 2); the evaluate
  harnesses take `--sidecar-db`. **Any new entry point that writes must port
  the guard.** `train.py` threads `--sidecar-db` back into
  `config.FORECAST_OUTPUT_DB` because `src/db.py` helpers read that attribute —
  a `--sidecar-db` that only lands in `args` is decorative.
- `.env` is gitignored → **a git worktree has no `.env`** and
  `config.DATABASE_PATH` degrades to a bare `\data\energy_dashboard.db`. Pass
  `ENERGY_DB_PATH` (or the harness `--replica-db`) explicitly from a worktree.
- **There is a decoy**: `../energy-data-gathering/energy_dashboard.db` (3 GB)
  is a stale partial snapshot — majors missing or years stale, numbers that
  look fine. Never point anything at it; do not delete it either.
- `validate_config()` requires `net_position` rows within 48h for BE/NL/AT/FR/DE
  and fails with a per-country reason (`python config.py` prints the verdict;
  `ALLOW_STALE_DB=1` downgrades for deliberate partial-DB runs). It is called
  by the train entry points and `forecast_daily.py`, **not** by
  `forecast_chronos2.py`.

Forecasts land in the `forecasts` table: `(country_code, forecast_type,
renewable_type, target_timestamp_utc, generated_at, horizon_hours,
forecast_value, model_name, model_version)`. `model_name` is the identity;
`model_version` is only the vintage timestamp.

## Training-data and read guards

- **`energy_renewable` silently zero-fills missing production types**
  (ABL-188): its ingest mapper initialises columns to 0.0, unlike
  `energy_generation`'s NaN-preserving twin. `exclude_suspect_constant_runs`
  nulls any 24h+ bit-identical run at the training read
  (`load_renewable_type_data`).
- **A zero is adjudicated against the twin table, not a duration** (ABL-200):
  `exclude_zeros_disproved_by_sibling` disproves an exact 0.0 in
  `energy_renewable` when `energy_generation` reports real generation at the
  identical instant. One-sided (a negative sibling never disproves — A75
  netting is real), per-pair calibrated floor, abstains under 1,000
  calibration rows, aligns on **parsed instants** (the two tables store
  different separator forms), and runs strictly **after** the ABL-188 guard.
- **Neither generation table is hourly** (ABL-332); most countries switch to
  quarter-hourly mid-history. **Exactly one resolution leaves the read — the
  hourly mean** (`aggregate_renewable_to_hourly` in `load_renewable_type_data`).
  `src/wind_features.py` raises `SubHourlyResolutionError` on a sub-hourly
  series: aggregate it, never floor the index.
- **TSO forecast tables are guarded on the way in** (ABL-431):
  `src/tso_plausibility.py` nulls a read value above 3.0× a derived per-country
  reference scale (ENTSO-E ships ×1000 unit errors), never touches stored
  rows, runs **before** any hourly resample, and raises `UnknownTsoSourceError`
  for an unregistered (table, column). A static sweep fails any module naming
  `energy_generation_forecast`, `energy_load_forecast` or
  `forecast_vintage_archive` without calling the guard or being exempt with a
  reason. A new read calls `guard_tso_frame(..., frame_column=...)` first —
  the archive read is registered and this applies to ABL-247's future feature
  read.
- **The sweep walks the whole repo, not one directory** (ABL-462). It shipped
  walking `src/` alone; a byte-identical unguarded reader under `scripts/` —
  where the analysis actually lives — passed silently for two weeks, and
  `reports/abl_430_ro_diagnosis.json` published an HU row (`corr 0.02`,
  `WAPE 597%`) that was 96 rows of 140,996 MW, not a zone defect. `tests/` is
  the one excluded directory, by name, because fixtures create these tables.
  Every swept directory carries a positive control; **do not add an exemption
  for a directory**, exempt a file with a reason.
- **Which table a renewable artifact reads is a property of the artifact**
  (ABL-331): `training_source` is written by `Forecaster.save` and threaded to
  serve/train reads; `db.RENEWABLE_TYPE_SOURCE_TABLE` is only the default for
  a run that names none, and legacy artifacts resolve to `energy_renewable`.
  **`Forecaster.save` is the only writer of a renewable artifact** (ABL-342);
  `ModelRegistry.save_model` refuses a renewable payload with no
  `training_source`. Do not collapse the source back to one constant: ABL-321
  measured 3 of 10 serving pairs materially worse on a global switch, while 33
  of the 37 unmodelled pairs have under a year of `energy_renewable` history.
  Train on the other table with `--renewable-source energy_generation`; window
  closing follows the same source (`get_latest_data_timestamp(source=...)`).

## Gate harnesses and pre-registration

`scripts/evaluate_solar_retrain.py` / `evaluate_wind_retrain.py` read
pre-registered scopes. Full history and per-amendment rationale:
`docs/claude/04-database.md`. The standing rules:

- **A scope is a registration, never a filter.** Harnesses take `--scope` over
  a `SCOPES` table (no `--countries`); the bar is the registered pair-list size
  × bands, read from the file, so a pair yielding no rows still shortfalls.
  Scoping, re-reading under a changed rule, or correcting a read is a **new
  scope** — a dispositioned scope's outputs are never overwritten
  (`SCOPE_OUTPUTS` resolves output paths per scope after parsing; a selection
  that would rewrite another scope's record is refused). Prefer a **tracked**
  `json_out` for any scope whose read will be dispositioned (an ignored
  `results.json` cannot be diffed or recovered).
- **Per-scope registration tables** are cross-checked at **import** by
  `check_registration_tables` (raises even on `--help`), but only for tables
  where no absence is deliberate (`SCOPE_SOURCES` joined for that reason).
  Tables whose absence encodes a choice stay out: `SCOPE_FEATURES` (inheriting
  the current list is the intended path), `SCOPE_NOT_EVALUABLE` (watch this one
  hardest — it defaults **toward scoring**), and `CAUSAL_LEVELLING` /
  `G23_READABILITY` / `SEED_READABILITY` (default toward their amendments;
  published scopes are pinned **by value**). The check enforces presence, not
  content. **Never trust a table count written here** — derive it from
  `tests/test_gate_scope_registration.py::test_every_per_scope_table_is_checked_or_declares_why_not`.
- **A cell that scored zero rows reads `UNREADABLE`, not `FAIL`**, and an empty
  comparison must never render as a tie.
- **Four model-free references are reported beside every read**
  (`constant_causal`/`_oracle`, `climatology_causal`/`_oracle`, one
  implementation in `src/evaluation/model_free_reference.py`) — reported,
  **never gate criteria**. Read a climatology's `comparator_n` before comparing
  its WAPE. `lost_to_a_model_free_reference` names losing cells unprompted.
- **A PASS is graded** (`src/evaluation/gate_grading.py`): G0 coverage
  (registered minimum n), G1 gate vs seasonal-naive D-7 beyond the readability
  floor, G2 level vs `constant_causal`, G3 shape vs `climatology_causal`, G4
  direction (slope > 0 and corr > 0). Grades, best to worst:
  **A** (all clear, promotion-eligible subject to named `HOLDS`),
  **N** (G2/G3 margin inside the floor — abstention, not promotable),
  **U** (G1 margin inside the floor — re-read at k>1, not reject; `U(+)` when
  G2-G4 clear readably), **X** (coverage short — nothing below decidable),
  **B** (G1 holds, a named G2/G3/G4 failure), **C** (readable loss to D-7).
  A pair takes its worst band. Floors: 10.65% solar / 7.51% wind at k = 1
  (`c_B = 0` — deterministic references); at k > 1 the readability test is the
  Student-t interval on the cell's own seed draws (`SEED_READABILITY`).
  Causal references are levelled on a trailing 28d window by default (ABL-437;
  fit-window levelling inflates G2/G3 on seasonal series); published scopes are
  pinned `fit_window`. Screen a level change against `energy_renewable` before
  crediting seasonality — a revision-vintage seam looks identical.
- **The TSO forecast is never a gate criterion** (Board directive 2026-08-14,
  standing): reported beside every read, never scored as a condition under any
  levelling or readability form. A registration making a TSO comparison a
  conjunct of a pass, fail or grade is invalid on its face — refuse it.
- **`--replica-db` governs the whole run** (ABL-355): builder, incumbent, TSO
  and contamination reads all take the resolved path; a write connection
  refuses a `db_path`; `meta['databases']` records every file opened. In a
  worktree with neither flag nor env set, the run refuses at argparse.
- **What the fit was allowed to see is registered too** (`FIT_RULES`):
  `exclude_impossible_night` drops contaminated night rows **from the fit and
  never from the score**. ES raises `IncoherentNightExclusionError` (real CSP
  night generation). Leave the rule off unless registered on: ABL-403 measured
  it doubling BG's night MAE and eating half its D-7 margin — a fit-side
  exclusion is defensible only when the excluded rows are both genuinely
  contaminated *and* a small minority of what is still scored.

**Never disposition a night-floor change on the negative-prediction rate.** It
cannot see the level, and it can move adoptably while unreadable: over the
ABL-403 eight paired fits, night MAE rose readably (+61 MW against a 7 MW null)
while **no** negative-rate contrast cleared its own single-seed null of
**14.06pp** — the apparent improvement sat inside the noise. Report night MAE
and night bias beside it, and read `outside_the_null` before quoting any of the
three; an 8/8 sign test is not readability when the null is wider than the
effect. Quote the two factors' contrasts separately, never the 25-feature-off
to 27-feature-on diagonal — those endpoints differ by both changes at once, and
the machine record keeps `exclusion_at_f25` / `exclusion_at_f27` /
`both_vs_neither` apart for that reason.

**A one-seed A/B on these harnesses cannot resolve anything under ~5%**: seed
alone moves daylight MAE up to ~5% between two fits. Pair the arms by seed and
quote the effect against a null built from control-vs-control seed pairs
(`scripts/abl376_night_seed_spread.py` is the worked example).

## Solar is clamped to physical reality on the way out

`save_forecasts()` (`src/db.py`) is the choke point every serving write goes
through. `src/solar_clamp.py` zeroes hours whose sun stays below −8° (measured
threshold) and floors the rest at zero — `renewable_type='solar'` only, **new
rows only**, stored history never rewritten. The night zero is per-country via
`solar_geometry.NIGHT_GENERATION_POSSIBLE`: **there is no default** — an
undeclared country raises `UndeclaredNightGenerationError` and the save writes
nothing (adding a country to `SUPPORTED_COUNTRIES` means adding it to that
table in the same commit). ES is exempt (real CSP night output). The
non-negativity floor is fleet-wide by measured justification, not physics.
Sun elevation comes from `src/solar_geometry.py` — import it, never write a
second copy. The clamp logs itself to `forecast_clamp_log` in the output DB;
read `night_generation_possible`/`night_mask_applied` before the counts — an
exempt country's zero count does not mean a clean fit. A retrain is verified by
the clamp going quiet, not by the retrain having run.

## The interpreter is part of the configuration

Two Pythons on this box; **the rail is `.venv\Scripts\python.exe`**
(Python 3.14, xgboost 3.3.0); bare `python` is conda (3.11, xgboost 2.1.4).
An xgboost-3.3.0 pickle loaded under 2.1.4 does not fail — it **silently resets
the fitted intercept** and predicts a near-zero-mean series with intact shape,
which reads as a bad model rather than a bad load. `src/challengers/v014.py`
writes the intercept into the artifact and `load_model` raises
`ModelArtifactError` when it has moved. Run anything that loads a model under
`.venv`. One configured exception: the `tso-correction` runner is pinned to
conda deliberately — its artifacts are LightGBM, which round-trips as text.

## Model storage

`models/{country}/{type}/model.joblib` — all metadata embedded (model,
feature_columns, training_source, version, metrics); only the latest version
kept; discovered by filesystem traversal. **`models/` is gitignored**: the
scheduled job resolves `config.MODELS_DIR` against `C:\Code\able\energy-forecast`,
so artifacts trained in a worktree must be copied there by hand — otherwise the
rail logs "no trained model", writes nothing for that model, and still exits 0.

## Key commands

```bash
# Everything runs under .venv from the repo root.
# Training needs a sidecar target or train.py exits 2 without writing:
python scripts/train.py --countries DE --types load --sidecar-db C:\Code\able\data\forecasts_local.db
python scripts/train.py --countries all --types all          # all models
#   variants: --algorithm lightgbm|catboost, --algorithms a,b,c (compare),
#   --optuna --n-trials 50, --walk-forward --n-folds 6, --feature-selection,
#   --start YYYY-MM-DD, --exclude-backtest, --renewable-source energy_generation

python scripts/forecast_daily.py [--dry-run] [--countries DE,FR]

# Chronos-2 (separate venv, GPU for fine-tuning):
python scripts/forecast_chronos2.py --experiment V010 --countries DE --types net_position --target-date ...
python scripts/train_chronos2.py --experiment V003 --device cuda
python scripts/compare_experiments.py --experiments V001,V003 --weeks all ...

# Tests — the whole suite:
.venv\Scripts\python.exe -m pytest -q
```

`pytest.ini` pins `testpaths = tests` (the bare command otherwise collects
scratch files). Probe/benchmark scripts are named `probe_*.py`, never
`test_*.py` — they execute training at import time and must not be collected.

## Chronos-2 serve-faithfulness

- **The context ends where the data ends, not where the schedule says.**
  `build_for_country` measures the last real observation, forecasts across the
  gap plus the target day, and the caller publishes the last 24 points.
  Interior gaps beyond the ffill limit still become 0.0 — prefer a NaN hole
  over an invented zero. A degenerate (constant-zero) context still forecasts
  and publishes; GR is the standing example.
- **`net_position` is day-ahead published**, so a 06:00Z run legitimately holds
  actuals through D 21:00 — the serve-faithful observation bound is **D 22:00**,
  not the run instant. **A reconstruction needs two bounds**:
  `as_of` (observations) and `publication_as_of` (which weather runs were
  issued), 16h apart for net_position (`src/chronos2/input_builder.py:541`).
  Suffix-1 covariates (TSO forecasts, DA prices, flows) cannot be bounded at
  all — `publication_timestamp_utc` records fetch time — so treat
  LT/RO/BG reconstructions as unverified.
- `compare_experiments.py` reads the DB as it stands today: **without `as_of`
  its numbers measure leaked information.**

## Net-position evaluation and the promotion gate

`scripts/evaluate_net_position.py` scores exactly the `--model` names given —
it discovers nothing. Both DBs open readonly. Rules that bite (full history:
`docs/claude/09-model-details.md`):

- **The gate scores a vintage window** (`gate_scope`, vintages at/after
  `cohort_split`): unwindowed, the champion is measured on the
  zero-padded-context era and the bar is ~2.6× easier.
- **All 8 pre-registered criteria are emitted; PASS requires all 8.** A
  criterion that cannot be evaluated yields `INCOMPLETE`, never PASS.
- **LU and GR are excluded by name** with recorded reasons (LU duplicates DE in
  A25; GR actuals are fabricated zeros), not by symptom.
- A comparison shares one window across every column; a model with no vintages
  reads "Not scored", never an empty column.
- **Never compare two per-model eval reports** — each covers its own rows. Use
  `scripts/compare_challenger.py`: it pairs vintages by the actuals they could
  see (`as_of_for_vintage`) within a 4h skew bound, refuses backfills (an
  information mismatch is not a tolerance), de-duplicates re-runs, and **exits
  1 when nothing paired** — an empty head-to-head must not read as a tie.
- Champion push (`push_net_position_forecast.py`) filters every query on
  `CHAMPION_MODEL_NAME`; challengers (in `src/challengers/registry.py`, listed
  not discovered) write their own `model_name` to the sidecar and are **never
  pushed**. The runner calls the eval once per model with its own `--out-dir`
  (shared dirs leave `latest.md` holding whichever ran last).
- Per-country re-read (`scripts/reread_net_position_country.py`):
  `skill_vs_zero < 0` is identically WAPE > 100% — zero is a named baseline,
  not an emergency; climatology is the decision-relevant row; scored vintages
  are counted separately from stored ones (`INTERIM`/`CONFIRMATORY`).
- **V014** (per-country XGBoost challenger): serve-faithful by construction —
  per-source cutoffs from the run instant (day-ahead tables D 21:00, flows
  target − 72h with `xb_missing`, weather by `forecast_run_time <= run_ts`);
  same-hour lags start at 72h; three refusals (no model file raises; < 2 of 3
  anchors → NaN; refused hours dropped, never written 0.0); a late run derives
  its window from the target date, not the clock. W01-W10 are weather-blind for
  every model. MAE-tuning was measured and **not adopted** (BE's MAE-optimal
  fit leaves the gate's slope band); promotion is only decided by the
  pre-registered C2c gate read.
- V016 (affine + AR(1) on the champion) **loses** — affine recalibration
  provably cannot reach the gate's slope band on V010 (`slope → rho²`), and the
  AR carry is negligible at a D+2 horizon. Do not propose another correction
  layer without reading `docs/claude/09-model-details.md` first.

## All-type forecast scorecard

`scripts/evaluate_scorecard.py` scores the production registry for all served
types over one window → `reports/forecast_scorecard/`. Selection is latest
vintage per (country, target, model, **horizon band**) — one-latest-row erases
the 24-64h evidence. Scoring truth lives in `scorecard.ACTUAL_SPECS` only; the
renewable family is scored on `energy_generation` (ABL-410) — that is the
*scoring* truth, independent of the training source, and touches no gate.
Baselines go through `src/baselines.py` (persistence rounds the lead **up** —
stored `horizon_hours` floors and can select a post-generation actual).

## Features

Time (incl. cyclical encodings), same-hour lags D-1/D-7/D-14, 24h/168h rolling
stats, holiday features, per-type weather. Notes that bite:

- **`create_lag_features` shifts by rows**, so a source gap poisons the
  fortnight after it (`energy_price` has thousands of missing hours in
  2025-09..12 for AT/DE). Check `reports/abl_393_source_gaps.json` before
  choosing an evaluation window.
- **The four holiday features are declared but in no serving artifact**
  (ABL-386/394/407): live for the *next* fit of any country, never yet
  evaluated on any target. Frozen lists in `tests/feature_list_manifest.json`;
  `select_feature_columns` warns on narrowing instead of dropping silently.
  The gate harnesses declare their own `FEATURE_COLUMNS` (also frozen in the
  manifest under `gate_harness`) and never call `get_feature_columns()`.
- Algorithms: XGBoost (default), LightGBM, CatBoost, Chronos-2 (120M foundation
  model, separate venv, 672h context).

## Troubleshooting

- "Model not found" → train it first; check `models/<CC>/<type>/model.joblib`
  exists at the top level — variant subdirectories are not a trained model.
- "Database error" → `ENERGY_DB_PATH` unset or stale; `python config.py`
  prints the currency verdict.
- Low accuracy → check data-quality guards fired (warnings name excluded
  windows), interpreter (see above), and resolution assumptions.

## Archive

`docs/claude/` holds the full pre-2026-08-27 narrative this file was distilled
from — one file per former section: incident forensics, dated measurements,
grading-ladder amendment history (`04-database.md`), evaluation deep-dives
(`09-model-details.md`). Start there when a rule here needs its evidence.
