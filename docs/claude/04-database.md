> **Archived from `CLAUDE.md` on 2026-08-27** (companion to the ABL-536
> energy-dashboard-frontend trim). Historical narrative, incident forensics
> and dated measurements; `file:line` references are frozen as of the archive
> date. The durable rules distilled from this material live in the repo-root
> `CLAUDE.md`; where they conflict, the root file wins.
# Database

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

### A zero is adjudicated against the twin table, not against a duration (ABL-200)

The guard above can only reject a zero **for lasting 24 hours**, and that
question has no good answer for wind. BE `wind_offshore` carries **105**
flat-zero runs of 6 h or longer (re-measured 2026-08-14) and only **9** reach the
default, so the threshold either misses the rest or, lowered, starts deleting the
genuine calm and curtailment spells a duration test cannot tell them apart from.
ABL-42 settled the formally identical MD hydro case by **cross-referencing the
better table instead of sizing a threshold**, and that is what
`exclude_zeros_disproved_by_sibling` does here: an exact `0.0` in
`energy_renewable` is **disproved** when `energy_generation` — the NaN-preserving
twin of the same fetch — reports real generation at the identical instant, at any
run length. Wired at the one training read site (`load_renewable_type_data`),
only when reading `energy_renewable`, and only *after* the ABL-188 guard.
`reports/abl_200_cross_table_zero_disproof.md` is the evidence pack;
`scripts/abl200_cross_table_zero_census.py` regenerates every number in it.

**Read §1 of that report before quoting the issue that filed this.** The premise
overstates the harm by two orders of magnitude on the pair it is about: of the
1,432 rows inside BE `wind_offshore`'s 105 runs, `energy_generation` is
**negative on 1,378 and positive on 54**, and only **2 of the 105 runs** contain
a single positive sibling value. A negative sibling is A75 netting — an idle farm
drawing house load — and a gross `0.0` is the correct reading of it. The
headline 2,175 MW instant (2025-11-17 04:00) is **already excluded today** by the
24 h rule, and ABL-198's own adjudicated window (2026-03-08/10) reads −11 to −30
MW on all 40 rows and correctly does not fire. The rule's marginal contribution
for that pair is **10 rows**; its case is fleet-wide — **564 rows over 38 of 120
pairs** — not BE.

Four things about it are load-bearing.

- **The floor is calibrated per pair, because there is no band to put a global
  one in.** Sibling value ÷ fleet p99.5 runs continuously across four decades
  over all 18,900 raw candidates (q05 0.000065, q50 0.000478, q95 0.1395, max
  1.018) with no gap wider than 4× anywhere — the **opposite** of ABL-431, whose
  3.0 sits inside a measured empty band 2.3× wide. So the floor is
  `q0.99(|renewable − generation|)` over the instants where the renewable side is
  **strictly positive** — how far these two tables routinely sit apart on this
  very series, estimated where the zero-fill defect provably is not what is being
  looked at. Bit-identical pairs (32 of 100) get a floor of exactly 0.0 and any
  positive sibling disproves; vintage-divergent pairs (NL `wind_onshore`,
  `energy_generation` higher at 83.5% of instants, median +311.8 MW — the ABL-439
  seam) set their own high bar and the rule falls quiet on them. Nobody chooses a
  number for NL. It is **not** a knife edge: 896 / 739 / **564** / 416 rows at
  q = 0.90 / 0.95 / 0.99 / 1.00, no acceptance case changing verdict in that
  range. q = 1.00 was rejected because one contaminated calibration row would set
  an unreachable floor and silently disable the rule for that pair forever.
- **It is one-sided, and it refuses to evaluate rather than guessing.** A
  negative sibling never disproves anything (`energy_renewable` holds no negative
  value in any of the 120 pairs, so the sign carries information). And below
  `SIBLING_DISPROOF_MIN_CALIBRATION_ROWS` positive-value instants the rule
  abstains and carries the reason, ABL-431's `evaluable` pattern for ABL-431's
  reason: **20 of the 120 pairs have a calibration population of exactly 0** —
  the all-zero series, landlocked countries reporting `wind_offshore_mw = 0.0`
  forever — where a floor of 0.0 would let any sibling value delete a new fleet's
  first output. The constant is 1000 and **that is a measurement**: the smallest
  non-zero population is 2,559, so anything in (0, 2559) is the same rule.
- **Alignment is on parsed instants, never on the stored string, and the order
  against ABL-188 is fixed.** `energy_renewable` stores BE's 2025-11-09 →
  2025-11-25 rows in the `2025-11-14T16:00:00` form while `energy_generation`
  stores every row in the `2025-11-14 16:00:00` form, so a SQL join on
  `timestamp_utc` returns NULL for **all 540** of them — including every row of
  the worked example the rule exists for. And it runs strictly **after**
  `exclude_suspect_constant_runs`: that guard measures a run over the
  observations present, so nulling rows inside a long flat run first would split
  it at the new gap and drop both halves under `min_run_hours`, weakening the
  older guard instead of adding to it.
- **It moves 15 pairs under 7 registered scopes, and zero gate rows.** All 170
  fit-window exclusions land in ABL-348's *fit* window (`abl406-tranche2b` 4/8,
  `abl417-tranche2e` 4/8, `abl316-t2d` 3/6, `abl316-t2c` 2/5, `abl253`/`abl376`
  1/3, `abl322-pilot` 1/2); **0** land in the gate window, and the latest
  exclusion anywhere is 2026-07-03 00:45, which is before the earliest gate row's
  D-7 and 168 h lookbacks. So gate truth, gate rows and the seasonal-naive
  baseline are byte-unchanged — but the **causal references are not**, since
  `constant_causal`/`climatology_causal` level on the fit window (or, under
  ABL-437, on a trailing 28 d that reaches into it). Per ABL-401 a re-read of any
  of those scopes against the new training set is a **new pre-registration**, not
  a re-read of a published path. None has been run. **Expect raised hours, not
  missing ones:** the guard runs on sub-hourly rows and ABL-332 averages after
  it, so an hour disappears only when *every* row in it was disproved — measured
  post-merge, 6 hourly observations deleted against 25 whose mean rose (all 25
  upward, up to +5.4 GW on ES `wind_onshore`). The 170 is a sub-hourly count.
  See §5 of the report for the per-pair split.

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

`energy_generation_forecast`, `energy_load_forecast` and the `source='tso'`
half of `forecast_vintage_archive` carry ENTSO-E's published day-ahead
forecasts verbatim, and verbatim includes a **×1000 unit
error**: HU's `wind_onshore_mw` reads **140,996 MW** against a fleet whose
p99.5 over five years is 283 MW. Dividing those 96 quarter-hours by 1000
reproduces HU's own measured generation for the same day (35.8–141.0 MW
predicted against 36.8–133.0 MW observed, rising together through the day), so
the shape is right and only the scale is wrong — which is why it is invisible
to every correlation- or shape-based check.

**Extent, measured on the replica 2026-08-14 and regenerable with
`scripts/abl431_tso_plausibility_census.py`: 320 of 26,805,465
column-observations (0.00119%)**, in three incidents — HU 2026-02-04 (96 rows,
one full CET market day, hitting `wind_onshore_mw`, its archived twin, and the
`total_forecast_mw` it dominates), MK 2022-04-10 (10 rows), SK 2022-09-25
(1 row). **Zero load rows anywhere.** So it is not one row, and it is not
widespread either.

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
- **3.0 is a measurement, not a convention.** Across all 257 evaluable pairs,
  `max / reference` runs HU 497.7× · HU total 37.3× · SK 8.70× · MK 6.07× ·
  MK total 4.12× — then nothing until PT solar at **1.82×**, PT wind_offshore
  1.77×, NL load 1.60×, p90 1.41×. 3.0 sits inside a measured empty band 2.3×
  wide. The census prints that ladder every run, so a healthy pair climbing
  toward the tolerance is visible before a fit meets it. Adding the archive
  (ABL-458) moved neither edge of that band.
- **It is one-sided, and it refuses to evaluate rather than rejecting
  everything.** A published 0.0 is never flagged at any tolerance, so ABL-71's
  published zeros and ABL-109's 56 legitimate DE overnight solar zeros are
  untouched by construction. **56 of the 318 pairs are all-zero series** —
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
fails if any `src/` module names one of the three tables without calling the
guard or appearing on an exempt list with a reason. **That sweep is what ABL-247
will trip when it adds its feature read** — which is the point: this issue is
that issue's precondition.

**The archive is registered but not yet read (ABL-458).** ABL-431 wired the two
live tables; ABL-247 reads neither of them. It needs *issued* vintages — what a
TSO published at a given run time — and `forecast_vintage_archive` is the only
place those exist. The same 96 HU rows are in it, at the identical 140,996.245
over the identical window, so a two-table guard would have left the poison on
the one read path it was built for, with the sweep passing. The archive is
**tall** where the others are wide (one `forecast_value` column, discriminated
by `source` and `forecast_type`, horizon in `model_name`), so it needed no new
rule — same reference, same tolerance, same one-sidedness — only a read shape.
That shape lives in `forecast_read`, which both this module and the census call,
because two callers computing the reference differently is the defect one level
up. Two exclusions there, both of which can only *lower* the bar: `source='ml'`
rows are our own forecasts, not a published TSO series; and `tso-week_ahead`
reaches **4.76% of a pair's rows** (DK load), an order of magnitude past the
0.5% cluster the quantile tolerates, so it could lift the threshold above its
own unit error. No archive counterpart to `total_forecast_mw` — the archive
stores no aggregate row. **ABL-247's read site is still to be written; when it
is, it calls `guard_tso_frame(..., 'forecast_vintage_archive', <forecast_type>,
frame_column='forecast_value')` before any resample.**

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

**Which table a scope reads is registration, not a flag default (ABL-426).**
A flag alone was not enough, and the way it failed is worth carrying. ABL-345
made the source selectable but left it *opt-in*, so an omitted flag fell through
to the global `db.RENEWABLE_TYPE_SOURCE_TABLE` — a constant consulted without any
reference to the scope. ABL-405 ran `--scope abl316-t2a` without it. ABL-348
registers `energy_generation` for all 37 tranche pairs and names the source table
in `voids_this_registration`; the run fitted, scored and graded eight countries on
`energy_renewable`, emitted a 24-cell evidence pack and **exited 0**. Nothing in
its own output contradicted it: the machine record was truthful throughout
(`meta.training_source`), while the report H1 and the findings pack both said
`energy_generation`. ABL-348 had even written the failure down in advance, under
`harness_prerequisite`. The missing piece was never the flag — it was that
nothing tied the flag to the scope. It is the only such read in the programme:
every other ABL-348 tranche record, wind and solar, carries `energy_generation`.

Since ABL-426 the solar harness resolves `args.renewable_source or
source_for(scope)` — `SCOPE_SOURCES[scope]`, elected in the file, in review,
before the fit. An explicit `--renewable-source` still wins, so ABL-345's contract
holds and an exploratory read is one flag away; what it can no longer do is pass
unnoticed, because `meta.source_is_scope_registered` goes `false` and the report
prints **OFF-REGISTRATION** with both tables named.

Two rules about that table, both learned the hard way:

- **It records the table each scope *was read on*, not the one its registration
  wanted.** `SCOPE_SOURCES['abl316-t2a']` is `energy_renewable`. Pinning it to
  `energy_generation` would look like a fix and would be ABL-404 again: an
  unflagged re-run would then refit eight countries on a different table and
  overwrite a *dispositioned* pack in place, under a heading naming ABL-405.
  `test_every_published_scope_registers_the_table_its_record_says_it_read` holds
  the row to the committed record.
- **Correcting a read is a new scope, never a re-base of the old one.**
  `abl316-t2a-generation` is 2a's eight countries on the registered table, with
  every other registered value held identical so the pair is a controlled A/B on
  the source alone, and its own output paths so it writes nowhere 2a writes.

The wind harness has not been given `SCOPE_SOURCES`. That is a known asymmetry
rather than an oversight — no wind read is off-registration — but it is the twin
divergence this pair keeps paying for, and it is filed.

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

**The retro-grade takes a `--tranches` selector; it is not one script per
tranche** (ABL-438). Tranche 1b (BG/CH solar, ABL-381) was never graded — not
because its record lacked anything, but because ABL-418 ran over 2a and 2b only.
Its `meta.reported_comparators` already listed all eight, so grading it needed
**no refit and no re-read**, and it landed by adding a row to `TRANCHES` in
`scripts/abl418_retro_grade.py` rather than by writing a second grader. **1b
solar: A × 2 (BG, CH)** — six cells, all four conditions in every band, all six
clearing `enough_pairs`. Recorded in `reports/abl_438_retro_grade.json` and
qualified in `reports/abl_438_tranche1b_findings.md`.

Three rules came out of that, and they apply to the next tranche too.

- **A selection writes where its issue writes.** The defaults reproduce ABL-418's
  artifacts; any other selection is **refused** if it points at
  `abl_418_retro_grade.*`. This is the `SCOPE_OUTPUTS` failure below, one
  directory over — a run that keeps a default output path rewrites a
  dispositioned report under a heading that no longer describes it, and exits 0.
- **The floor applies to any margin a reader ranks on**, not only to the one G1
  gates on (ABL-417). Both 1b pairs beat the *oracle* hour-of-day climatology in
  every band and neither readably: BG by **+1.41%** at its worst band, CH by
  **+3.47%**, against a 10.65% floor. The renderer now prints `yes, inside the
  floor (+x%)` rather than `yes`. Three of ABL-418's own pairs are in that
  position (2a CH +8.15%, 2a RO +5.93%, 2b FI `wind_onshore` +7.48% against a
  7.51% floor); its published report predates the qualifier and is not rewritten
  here, so **regenerating `abl_418_retro_grade.md` no longer reproduces the
  committed file** — the delta is the `n ≥ min` column, the coverage note and
  those three cells, and no grade moves. Whether that report is regenerated is a
  CEO call, filed separately.
- **A hold is data, not prose.** Grade A reads *promotion-eligible, subject to
  any named data hold*, so the hold is registered in `HOLDS`, carried into the
  JSON and rendered under its pair's table. BG solar's is ABL-396's live
  night-contamination hold, whose displacement band is far wider than the +1.41%
  above. A grade of A must not be reported for BG solar without it.

**`enough_pairs` is reported beside every grade**, because the ladder cannot see
it: a grade reads a *margin*, so a coverage-short cell that beat D-7 would grade
`A` exactly as a full-coverage one does. It nests under `gate`, where a flat
lookup passes vacuously — assert the value, not its presence.

### Which causal reference G2 and G3 read is registered per scope (ABL-437)

**The two `*_causal` references are levelled on the fit window and scored on the
gate window, and those are different seasons.** ABL-348 fits
2026-01-14 → 2026-07-11 and gates 2026-07-11 → 2026-08-10, so on a seasonal
series the "causal constant" is a winter-and-spring average scored against high
summer — not an estimate of the gate window's level at all. G2 and G3 are
registered on exactly those two references, so the reference is a strawman and
the grade is inflated for free. Re-derived by import over all 137 committed
cells, worst band per pair: **15 of the 18 `wind_onshore` pairs sit at or above
17%**, topping out at NL's **205%** (225.54% against an oracle constant at
73.85% — a flat line three times worse there than forecasting zero), while all
19 solar pairs sit between −1.2% and 7.8%, because a flat line's WAPE on solar is
dominated by the diurnal cycle rather than by the level. The ten-row table in the
ABL-437 description reproduces to the digit and is not the whole set: HU, NO, RO,
SE and EE are also above 17%. This was the third instance of one pattern
(ABL-406 bar weakness, ABL-417 on RO, ABL-435 on BG/CH).

The amendment keeps both conditions on the ladder and **re-levels the reference
they read**, to `constant_causal_28d` / `climatology_causal_28d`: the same two
predictors over the 28 days ending at **the row's own `generated_at`**. The
alternative — flagging G2/G3 not-evaluable outside a registered level band — was
evaluated and rejected, because the band is keyed on the *country's seasonality*
rather than on the challenger and so cannot separate the two cases it would be
used to separate: BG (43%) and CH (96%) both trip any band, and BG beats the
corrected references where CH loses to them. Its diagnostic survives as a printed
`level inflation` column. `reports/abl_437_causal_levelling_registration.md` and
`experiments/ABL437/config.json` are the registration.

Five things follow, and the third is the one that bites:

- **The causality claim is not a new argument.** The window is anchored at
  `generated_at.floor("h")`, inclusive, spanning `28*24 - 1` hours back, over the
  same ABL-188-filtered series — character for character the bound
  `wind_features._rolling_features` applies to `target_value_roll_168h_mean`,
  which is one of the challenger's own 24 features. The reference uses no
  information the challenger did not have, and the test reads that bound out of
  `wind_features.py`'s source rather than restating it.
- **28 days, and both forms share the window.** A constant is a climatology with
  one bucket, so levelling them differently breaks the reading that the gap
  between them is forced diurnal structure. The shared window has to serve the
  climatology, which needs samples per hour-of-day bucket — 28 days gives 28,
  7 would give 7, and a noisy climatology is a weak reference, i.e. the same
  defect in a new place. The window is **in the column name** so two reads
  levelled differently cannot wear one name.
- **`CAUSAL_LEVELLING` defaults *toward* the amendment**, which is the opposite
  of `SCOPE_FEATURES` and the opposite of `SCOPE_NOT_EVALUABLE`. A scope absent
  from it grades on `trailing_28d`, because inheriting the old reference silently
  would hand a new tranche the inflated one on pairs nobody has looked at yet.
  The cost is that an absence can no longer reproduce an old read, so every
  published scope is **pinned to `fit_window`** and
  `test_every_published_scope_pins_its_levelling` derives that set from
  `SCOPE_OUTPUTS` **and git** rather than from a list. `scripts/abl418_retro_grade.py`
  is pinned for the same reason, and pinned at the *cell* rather than per tranche,
  so it covers ABL-438's `1b` row above and whatever `TRANCHES` gains next: none
  of the three committed records carries a trailing column at all — checked, not
  assumed — so the amended default would rewrite a published page of A's as B's.
  The table is deliberately **not** passed to `check_registration_tables`, and
  that is structural rather than cautious: that check requires every scope in the
  union to appear in *every* table it is given, so registering `CAUSAL_LEVELLING`
  there would force each scope to be pinned and delete the default this bullet
  exists to describe.
- **The ladder's rules did not move, only the reference.** Given two reference
  pairs carrying identical numbers, every case grades identically under either
  levelling; that is asserted directly rather than argued. G1 is still
  seasonal-naive D-7 under both, both oracles are still on neither ladder, and
  ABL-348's windows, bands, metric, baseline, minimum n and source are untouched,
  so `voids_this_registration` is not triggered.
- **A trailing window converges; it does not teleport.** On a step change at the
  gate boundary it still carries the old level on day 1 and only halves the
  reference's error over a 30-day window. On ABL-348's windows it *starts* as the
  last 28 days of the fit window — mid-June to mid-July, already the gate season
  — so the real residual is smaller, but it is **reported per cell** rather than
  assumed away. Do not quote the corrected reference as exact.

### What the corrected reference did to two real pairs, and what it costs to land (ABL-443)

**A letter moving is the small half of the finding; the margins are the big
half.** DE and NL `wind_offshore` were re-read at `trailing_28d` under a new scope
`abl443-offshore-trailing` (ABL-436's `abl322-pilot` read stands byte-unchanged,
per ABL-401). One letter moved — DE **A → B**, fails G3 at 48-64h by 0.47pp — and
that is not the number to quote. **All six of DE's G2/G3 margins collapse from
+10.18…+12.72pp to +0.33…+1.32pp and −0.47pp, every one inside the 7.51% floor.**
Its published double-digit margins *were the reference's mis-levelling*, so DE has
demonstrated neither level nor shape in **either direction**; its G1 (+24…+26pp vs
D-7) is untouched and ABL-436's PASS stands. NL holds **A** at +18.80…+21.78pp,
still readable at ~2.5–2.9× the floor, and beats both oracles readably. Read the
margin table before the letter table, on any re-read.

Three things that generalise:

- **Screen the level change against `energy_renewable` before crediting
  seasonality.** ABL-439 found NL `wind_onshore`'s 3× shift on `energy_generation`
  was a *revision vintage*, and a vintage seam is indistinguishable from
  seasonality in the inflation diagnostic — the challenger is fitted straight
  through it either way. Here the two tables agree within **0–5% in every month**
  for both pairs, so the 1.63×/1.96× shift is the real winter→summer cycle. That
  screen is one query and it is the difference between a finding and an artifact.
- **The correction is partial and its residual does not rank with the raw one.**
  Inflation fell 18–27% → **5.5–8.0%**, but NL's *residual* is larger than DE's
  even though NL's raw figure was worse — a trailing window lags a level that is
  still falling. Rank on the residual, not on the improvement.
- **`grade_cell` gaining two additive keys turns strict dict equality red on
  merge.** ABL-437 adds `causal_levelling` and `level_inflation_pct` and
  deliberately does not regenerate `reports/abl_418_retro_grade.json` (its §8);
  ABL-438 landed on main meanwhile with `fresh == stored` against those exact
  bytes. Green on each side, **2 failed on the merge**, no textual conflict, and
  the collision is in a file neither branch's issue is about — so neither PR's own
  test run can see it. The fix is `without_abl437_provenance` in
  `tests/test_abl438_retro_grade_1b.py`: strip the two provenance keys at any
  depth, compare, then assert the `fit_window` pin **separately** — which keeps a
  moved letter failing, where a bare subset check would not. **Do not "fix" this by
  regenerating the committed record**; that overwrites a published page of letters
  to make a test pass.
- **Two branches can find the same merge collision and fix it twice.** ABL-443 and
  ABL-437 both hit this within one pass and wrote different helpers for it; the
  base branch's won. Before writing a merge fix, re-fetch and check whether the
  branch you are stacked on has already moved — a PR head is not stable while you
  work against it.

### How wide a G2/G3 margin has to be is registered per scope too (ABL-444)

**G1 carries a readability floor; G2 and G3 were registered as bare sign tests,
`skill > 0`.** So a G2/G3 verdict could turn on a margin far inside the spread one
seed resolves — ABL-437's re-read moved PL solar from `A` to `B` on **0.36%** of
skill, 3.4% of the solar floor — and the ladder reported it with the letter it
uses for a decisive result.

**Read this as a coherence fix, not a new bar.** ABL-418 *already* applies
`readability_floor_pct` to G2 and G3, on the same `skill vs X` column and in the
same function, when it decides `U` against `U(+)`: the plus requires both to clear
*readably*. So the same +2% margin was "not readable" if G1 happened to be
unreadable and "G3 holds, grade A" if G1 happened to be clear. ABL-444 carries the
existing test to the other branch and adds no constant of its own.

A margin at or inside the floor now grades **`N` — not readable**: an abstention,
not a failure and not a pass, and **not promotion-eligible**, on the rule ABL-418
already states (*a condition that could not be measured is not satisfied*). It
cannot wear `U`, which is undecided on the *gate*, nor `B`, which claims a failure.
`G23_READABILITY` in each harness registers the form per scope and defaults to
`floored`; all twelve published scopes are pinned to `sign_test`, so **no committed
letter moves**. `reports/abl_444_g23_readability_floor_registration.md` and
`experiments/ABL444/config.json` are the registration; the re-read of every
committed cell under it is `reports/abl_444_g23_floor_reread.md`.

Four things follow, and the last two are the ones that catch a reader out:

- **Severity is `A < N < U < B < C`, and `N` is *better* than `B`.** A definite
  failure outranks an abstention — a cell that also fails G4 reads `B`, because
  there is something to report and `N` would bury it. `N` ranks above `U` because
  an `N` cell cleared the registered gate readably where a `U` cell could not.
  **Do not assert that the floored form raises severity monotonically**: a `B → N`
  move lowers it while leaving the pair exactly as non-promotable. Assert it on
  the `A` set, which is what the caveat actually claims.
- **The margin prints either way**, per the CEO's constraint: the floor decides
  gradeability, not the number. An `N` cell carries both denominators, the floor
  and a reason naming the margin, and the per-pair table gained a **not readable**
  column beside **failed conditions** so the two are never collapsed.
- **The two denominators no longer agree, and the disagreement is one cell.**
  ABL-418 measured `skill` against ABL-385's own-error form over its 48 cells and
  found no cell moved. Over the programme's **476** G2/G3 condition-observations
  **3** change readability status and **one cell letter moves** — 1b BG solar
  36-48h, `N` on the registered column (+10.56% against a 10.65% floor) and `A` on
  ABL-385's (+11.81%). No *pair* letter moves. Quote the 119-cell number, not
  ABL-418's "they never disagree".
- **A tightest-band margin is not a pair-level one.** The ABL-444 description names
  PL solar (0.36pp) and CH `wind_onshore` (0.52pp) as the two flips inside the
  floor. PL reproduces at pair level; **CH does not** — its 24-36h band fails G2
  and G3 *readably* (−7.93%, −12.89%) and a pair takes its worst band, so CH stays
  `B`. ABL-437's flip-margin column is labelled "tightest–widest" for this reason.
- **A published record that predates the floor is graded on the default, and the
  default is the amendment.** Five scripts read a committed record through
  `grade_cell`/`cell_grade` — the ABL-418 retro-grade, ABL-437's and ABL-443's
  re-reads, and ABL-419's and ABL-421's tranche reads — and every one now names
  `SIGN_TEST` explicitly, including the two where the record carries a recorded
  grade and the default is unreachable today. ABL-443's landed on main mid-issue
  and went red on the merge with no textual conflict, which is the same shape as
  the additive-key collision its own CLAUDE.md block describes. Its DE
  `wind_offshore` is the eleventh mover: all six of its G2/G3 margins are inside
  the floor, which its own record had already labelled *not readable at one seed*
  while recording `g2_g3_floor_is_a_ladder_condition: false`.

**At k > 1 the readability test is the Student-t interval on the seed draws, not
`delta_min`** (ABL-467, filed by ABL-427 §7.3).
`reports/abl_467_seed_interval_readability_registration.md` and
`experiments/ABL467/config.json` are the registration; `SEED_READABILITY` in both
harnesses is the per-scope table, values `delta_min` / `student_t`.

- **`delta_min` is not wrong, it is the k = 1 tool.** It imports `c_A` from a
  fleet p90 because one fit carries no internal estimate of its own spread. At
  k > 1 the cell has k honest draws of the graded quantity and the import answers
  the wrong question — *how much do fits of this stream vary* rather than *how
  much does this cell vary*. **`delta_min` is untouched at k = 1 and every
  published letter is a k = 1 letter.**
- **The rule does not change form.** A condition is readable iff
  `|margin| > half_width`; `CI excludes 0` and `|mean| > t*se` are the same
  statement. Only the estimator of the width moves, and **the point estimate does
  not move at all** — skill is affine in WAPE against a deterministic reference,
  so the mean of the draws *is* the printed `skill vs X` column (agreeing to under
  `1.3e-14` pp on all six ABL-427 cells).
- **Do not describe this as the more permissive test.** The t half-width exceeds
  the unamended fleet floor wherever the cell's own seed CV exceeds about
  `z/t_{k-1}` of the fleet p90 — ~93% of it at k = 12. **All three HR cells are
  graded against a *wider* half-width than `readability_floor_pct("solar", 12)`
  and still clear it.** The near-coincidence of the two floors at k = 12 is
  arithmetic on these cells, not a theorem.
- **One set of draws, three half-widths.** G1/G2/G3 have different denominators so
  each gets its own width, all derived from the per-seed *challenger* WAPEs
  because `c_B = 0` for every reference on this ladder. **A fitted reference voids
  the registration** and every width has to be recomputed.
- **The draws are passed, not a precomputed interval.** The ladder owns the one
  implementation of its test; a caller could hand in a one-sided interval, or one
  built with `z`, or with the wrong `df`, and `grade_cell` could not tell. Two
  guards make the draws provably the cell's own — a seed count disagreeing with
  `k` raises, and draws whose mean is not the cell's recorded challenger WAPE
  raise (the silent failure being a paste from another cell).
- **ABL-434's property survives.** `seed_wapes` defaults to `None`, so `grade_cell`
  with `scores` alone is byte-for-byte the function ABL-434 registers; the draws
  are read off the *cell* by `cell_grade`/`attach_grades`, the same two functions
  ABL-434 uses for coverage. The difference that earns a per-scope table: coverage
  is one-way, **this is not** — a sharper test can raise a letter.
- **The table's fall-through is the *less* conservative direction**, unlike
  ABL-444's, and is safe for a reason that is about `k` rather than about the
  value: a fall-through row can only bind a read at k > 1, and at k = 1 there are
  no degrees of freedom so `delta_min` decides whatever the table says. Published
  scopes are pinned by **value**, not by presence.
- **The blast radius is measured, not asserted**: no call site anywhere passes
  k > 1; of **631** committed graded cell-records, **613** carry `floor_pct` of
  exactly `10.6482` or `7.5054` and the other **18** are all in
  `reports/abl_427_tranche2c_seed_reread.json` — ABL-427's six cells under three
  candidate floors, and no other file has a non-k=1 floor at all; and **1,568**
  replays of the 196 committed `scores` blocks under the amended module are
  byte-identical to the pre-amendment one. The assertion is an **equality**, so a
  second k > 1 read landing anywhere goes red until it is named.
- **Normality is stated, not waved at, and is not load-bearing here.** Shapiro-Wilk
  on 12 draws is failure-to-reject, not evidence. What answers the objection is
  that **Wilcoxon and a percentile bootstrap agree with t on all six cells**. The
  **sign test is the lone dissenter and it dissents in both directions**, exactly
  inverting the pair verdicts — it discards magnitude and at n = 12 has about
  three attainable p-values. Wilcoxon is the registered fallback for visibly
  skewed draws; the bootstrap is not, because its lower bound moved −0.117 to
  −0.005 across 10 RNG seeds and a registered verdict must reproduce exactly.
- **`T_CRIT_95` is pinned in the module, not imported from `scipy`**, for the same
  reason `STREAM_FLEET_CV_P90` is pinned rather than read from its report: a
  registered verdict must not move on a dependency upgrade. Tests check every row
  against `scipy`. Above `df = 30` it falls back to `z`, anti-conservative by at
  most **3.9%** of the correct half-width — a figure that was wrong in this
  module's first draft and is now asserted, not merely commented.

### The ladder reads the cell's minimum n before its margin (ABL-434)

Everything above grades a **margin**. `grade_cell` is handed a cell's `scores`
and nothing else — never `gate.n`, `gate.minimum_n` or `gate.enough_pairs` — so a
cell that beat D-7 by more than the floor while falling **short of its registered
minimum n** graded `A`, which means promotion-eligible. Tranche 2d is where the
combination first arose (1a/1b/2a/2b/2c are fully covered): **EE and FI solar
48-64h clear D-7 by +29.0% and +36.8% and miss 456 rows, FI by three**, so
`grade: A` sat beside `pass: false` inside the same cell with nothing in the
record to reconcile them. It is the only one of ABL-316's four open corrections
that can put a *false A* into promotion evidence — ABL-426 and ABL-440 make a
pair unreadable, which is visible.

A cell that fails the new `G0` grades **`X` — not readable at the registered
coverage**: it does not have the rows, so nothing on the ladder below it is
decidable, and it is not promotion-eligible. Registration
`experiments/ABL434/config.json`, evidence
`reports/abl_434_coverage_gate_registration.md`.

Four things follow, and the second is the one that decides where you put a fix
like this.

- **It is not a new bar and there is no per-scope table.** `enough_pairs` already
  decides the gate `pass`; what moved is only whether the *grade* may disagree
  with it. Unlike `CAUSAL_LEVELLING` and `G23_READABILITY` there is nothing to
  register per scope, because the direction is one-way — a table could only ever
  be used to let a scope declare its own cells covered.
- **`grade_cell` stays a function of `scores` alone, and the gate lives one level
  up.** `cell_grade` and `attach_grades` are handed a whole cell, so they read its
  coverage; both harnesses record and render through those two, so no future
  tranche can write a coverage-blind `A`. Keeping `grade_cell` ungated is what
  leaves the four published margin-only re-reads (ABL-418's retro-grade, ABL-437,
  ABL-443, ABL-444) reproducing byte-for-byte instead of being silently re-graded.
  Those four are registered with reasons in
  `tests/test_abl434_coverage_gate.py::MARGIN_ONLY_READERS` and an AST sweep fails
  any unregistered `grade_cell` caller — the `tso_plausibility` pattern.
- **The gate applies to a grade rebuilt from a record, not just a computed one.**
  That is the whole point: a stored `A` beside the record's own
  `enough_pairs: false` is the defect, and every later reader now gets the hold
  without keeping its own books, which is what ABL-421 had to do by hand.
  Unrecorded coverage is **not** a pass — a cell with no `gate` block grades `X`
  naming that, which is why every committed record was checked to carry the column
  (143 of 143) before landing it.
- **Severity is `A < N < U < X < B < C`, and `X` is *better* than `B`.** Deeper
  than `U` (a `U` cell has the rows and cannot resolve the margin), shallower than
  a definite failure, on ABL-444's rule that grading it `X` at pair level would
  bury a band that had the rows and lost readably. A pair takes its worst band, so
  one short band takes the pair — stricter than ABL-421's hold, which fired only
  when *no* band was decidable; the two agree on every cell measured so far.

**Two letters move and nothing is regenerated**: EE and FI solar 48-64h, `A` →
`X`, cell and pair, both already reported as `—` with a named coverage hold in
ABL-421's pack. 2d's `FAIL` verdict does not move — `passed` and `disposition`
never read a grade. Re-running ABL-421's read would print `X` in its ladder column
where the committed file prints `A`; that regeneration is a disposition and is the
CEO's call, filed separately.

### The TSO forecast is never a gate criterion (Board directive 2026-08-14)

The TSO day-ahead and week-ahead series are **reported beside every gate read and never scored
as a condition**, under any levelling and any readability form.

A registration that made a TSO comparison a conjunct of a pass, a fail, or a grade would be
**invalid on its face** and must be refused rather than debated.  This is a Board directive of
2026-08-14 (ABL-316 comment, item 1), standing, and is not a per-scope choice.

Concretely:
- `LADDER_REFERENCES` in `src/evaluation/gate_grading.py` names only `constant_causal` /
  `climatology_causal` (fit_window) and their `_28d` forms (trailing_28d).  No TSO entry
  appears under either levelling; `conditions_for()` builds G1–G4 from `seasonal_naive` and
  those two references only.
- Both harnesses print beside every scoped report: "Baseline: literal seasonal-naive D-7.
  TSO is revision-contaminated context only and is not a gate criterion."
  (`scripts/evaluate_solar_retrain.py:1205`,
  `scripts/evaluate_wind_retrain.py:644`)

Beating the TSO is the long-term goal; it is a benchmark to *report*, never a conjunct in any
pass/fail or grade.

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

**Registering a new scope means editing every registration table, and only some
of them are import-checked.** *Do not trust a count written here.* This paragraph
has carried a stale one at nearly every tranche — it said "five of seven" while
the file held ten — and ABL-421 left a `grep -E "^[A-Z_]+ = \{"` recipe that was
already wrong at the commit that called it "the count". Derive it in the harness
you are actually editing, or read the authority:
`tests/test_gate_scope_registration.py::test_every_per_scope_table_is_checked_or_declares_why_not`.
That test decides "per-scope registration table" from the source (**keys are
scope names**) and requires each one to be either in `check_registration_tables`
or in `UNCHECKED_REGISTRATION_TABLES` with the reason it cannot join, so a table
added by a later tranche fails until its author chooses. As of ABL-426 solar
holds **11** such tables, **6** checked and **5** declared unchecked — a fact with
a shelf life, which is why the derivation matters more than the number.

**The two harnesses' calls differ, and that is not one twin missing a fix** — the
recurring failure mode this pair has (ABL-322/ABL-379, ABL-345/ABL-347): wind
carries only the first three tables at all, so all three of its are checked.

The exemptions are not a convenience; each is a structural argument, and the
*shape* of the argument matters more than the list:

- `SCOPE_FEATURES` **cannot** join. `abl316-t2a`'s absence from it is correct and
  published — inheriting the current `FEATURE_COLUMNS` is the intended path for a
  new tranche — so requiring it would raise at import for a scope that is right.
- `SCOPE_NOT_EVALUABLE` is the one to check hardest, because it defaults **toward
  scoring**: a scope that forgets it scores every cell it can build, which for a
  pair ABL-348 declares NOT-EVALUABLE is a wrong verdict rather than
  self-documenting degradation.
- `CAUSAL_LEVELLING`, `G23_READABILITY` and `SEED_READABILITY` default *toward*
  their registered amendments (ABL-437 / ABL-444 / ABL-467), so requiring them
  would force every scope to be pinned and delete the behaviour they exist to
  provide. Their published pins are asserted **by value** instead, which is
  strictly stronger than the presence check joining would give.

So a scope missing from any of those five still resolves through a module-level
default **silently, at run time** — exactly how ABL-404 happened, which is why the
rows that depend on it each carry a comment saying so.

**ABL-426 added `SCOPE_SOURCES`, and it joins the call rather than the exemption
list.** The lesson is the test for which side a new table belongs on: ask whether
any absence is *deliberate*. For the five above an absence encodes a real choice,
so requiring a row would abort on scopes that are correct. A scope is read on
exactly one table, always, with no third state — so no absence is deliberate here,
and the exemption list's contract ("an omitted row defaults silently") is the very
mechanism that produced ABL-426. Joining also widened nothing:
`check_registration_tables` requires every scope in the *union* of the given
tables' keys to appear in all of them, and this table's keys are `SCOPES`' keys.

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
