# ABL-321 — training source for individual renewable types: findings

Analysis companion to the generated A/B tables in
`reports/abl_321_source_switch.md` (window 1) and
`reports/abl_321_source_switch_w2.md` (window 2). The pre-registered protocol
and decision rule are in `experiments/ABL321/protocol.md`, both registered
before the numbers they govern were read.

**Protocol.** Replica `C:\Code\able\data\energy_dashboard.db` (9,432,453,120
bytes), read 2026-08-12, opened `file:...?mode=ro` with `uri=True`. No write of
any kind touched the replica or any sidecar. Every census count below is a
census, not a sample; where a number is a model result it says so, with window,
n and baseline.

**Contamination touching these windows.** ABL-67 is net-position-only;
ABL-109/111 are load-only; ABL-71's known wrong-write modes are load and net
position. None of them is a proof that solar/wind ingest is clean — this is a
provenance caveat, not a clean bill of health. The FR gap in §2 is not covered
by any of them and was first filed in ABL-318 §3.

---

## 1. What the decision actually is

The question is not "is `energy_generation` better at predicting?" The case for
the switch is the census evidence in §2: history, NULL-vs-0, duplicates and
coverage. The backtest's job is narrower and is the one the acceptance criteria
set — **catch a regression in the four countries that already serve**. It is a
non-inferiority check, and it is reported as one.

## 2. The census: four arguments, one exception

### History

Across the 49 trainable country/stream pairs (ABL-318):

| | median usable history | pairs reaching 365 d |
|---|---:|---:|
| `energy_generation` | **2,049 d** | **49 / 49** |
| `energy_renewable` | 277 d | **10 / 49** |

For the four countries that already serve, `energy_renewable` begins:

| country | `energy_renewable` starts | days to 2026-08-12 | `energy_generation` starts |
|---|---|---:|---|
| AT | 2025-11-07 23:00 | 278 | 2021-01-01 |
| BE | 2024-01-01 00:00 | 954 | 2021-01-01 |
| DE | 2025-09-08 22:00 | 338 | 2021-01-01 |
| FR | 2023-01-01 00:00 | 1,319 | 2021-01-01 |

**AT and DE cannot be retrained on a full seasonal cycle from
`energy_renewable`.** DE misses 365 d by 27 days; AT by 87. That is not a
hypothetical about the 39 unbuilt pairs — it constrains two of the four models
already serving.

### NULL vs 0

`energy_generation.wind_offshore_mw` is NULL in every row for 15 countries — the
honest encoding of "the TSO does not report this". The same 15 read **100.0 %
exactly 0.0** in `energy_renewable`, whose columns are `REAL DEFAULT 0` and
whose mapper initialises each to 0.0 before checking the response (ABL-188). A
trainer pointed there receives a complete, non-null, perfectly valid all-zero
series for a country with no offshore fleet, and nothing in the read path
distinguishes it from a measurement.

Same shape for `biomass`: **CH, GR, SE** are 100 % NULL in `energy_generation`
and zero-filled in `energy_renewable`. Under the switch those three correctly
become "not reported" (empty frame) rather than a fabricated flat-zero series.
That is the intended behaviour and it is also a coverage change, so it is
flagged rather than buried.

### Duplicate instants

`energy_renewable`'s UNIQUE index is on `(country_code, timestamp_utc)` as a
*string*, so one instant is storable under several spellings: **78,510
duplicate-instant rows across all 24 countries, 5,425 carrying disagreeing
values**. `energy_generation` has zero. Nothing downstream chooses between the
two answers.

### Coverage — and the one exception

Census over 2025-10-01 → 2026-08-11 (the era where both tables have broad
coverage), distinct hourly instants, all 34 countries in `energy_generation`:

| | hours |
|---|---:|
| in `energy_generation`, not in `energy_renewable` | **24,694** |
| in `energy_renewable`, not in `energy_generation` | **586** |

42:1 for `energy_generation`. Of the 586, **518 are FR** — the
2026-06-30 23:45 → 2026-07-22 14:15 gap filed as new in ABL-318 §3 — and
`energy_renewable` has real data across it (FR solar daily maxima 13.7–21.8 GW,
wind_onshore mean 3,732 MW, 0.0 % zeros). The remaining 68 are MD (50) and CY
(18), neither in the 24 supported countries.

So `energy_generation` does **not** strictly dominate, and the exception lands
inside the window this backtest scores. That is handled by common-row scoring
(§4), and the lost coverage is reported as its own consequence rather than
absorbed into a metric.

I deliberately did **not** backfill `energy_generation` from `energy_renewable`
for FR. Mixing sources inside one series would put a zero-fillable segment
inside an otherwise NULL-honest one with nothing recording which rows came from
where. Closing that gap is an ingest re-fetch, not a training-source trick.

## 3. Two things the issue's own description got wrong

I wrote the issue; both corrections are to my own text.

**"The `column_map` transfers unchanged" is false for `hydro_total`.** It is
`hydro_run_mw + hydro_reservoir_mw`, and SQL's `+` propagates NULL. For **9 of
the 24 supported countries exactly one component is 100 % NULL in
`energy_generation`** — BE, EE, FI, LT, LV, NL, SI report run-of-river and never
reservoir; GR and SE report reservoir and never run-of-river. A literal column
swap returns NULL for every row of all nine and erases them. `energy_renewable`
hides this by zero-filling the absent component. The diff uses NULL-aware
addition: sum the reported components, return NULL only when *both* are absent.
Pinned by `tests/test_renewable_training_source.py`.

**Dropping NULL rows is not optional.** Without it a not-reported stream still
arrives as a feature-complete zero series and the switch buys nothing. That is
acceptance criterion 3, and it is the property the tests pin hardest.

## 4. The two tables disagree in more ways than the issue claims

The issue frames `energy_renewable` as zero-filling. Measured per served pair on
shared hourly instants, that is only one of at least three distinct classes, and
it is not the largest:

| class | example | scale |
|---|---|---|
| zero-fill (ABL-188) | BE wind_offshore: `energy_renewable` reads exactly 0.0 where `energy_generation` reads materially non-zero | 274 h in the fit window, **91 h in the scoring window** |
| frozen low value | AT wind_onshore reads 4.0 MW (63 h) or 8.0 MW (13 h); `energy_generation` contradicts ~8 of those, up to 2,476 MW. Longest contiguous low run **18 h — under `exclude_suspect_constant_runs`' 24 h minimum**, so it is invisible to the guard | small but real |
| level/revision divergence on non-zero values | BE wind_onshore: `energy_renewable` sits systematically ~190 MW above `energy_generation` on 2,380 of 4,580 fit-window hours (means 809.6 vs 710.5), converging by May; AT solar differs on 956 h with near-symmetric sign | **the largest class**, and not previously named |

The third class is not zero-fill and is not covered by ABL-188. Which table is
right for it is not established here, and I am not claiming it is. It matters
because it — not zero-fill — dominates what the two arms actually train on for
AT and BE.

**BE wind_offshore is the only served pair whose *scoring* window differs
between the tables** (91 materially disagreeing hours, all of them
`energy_renewable`-zero against `energy_generation`-non-zero). Every other
served pair has identical truth under both definitions in the scoring window,
which is why the primary and secondary truth columns agree everywhere else — the
switch is not grading its own homework.

## 5. A merge precondition the issue did not name

`src/forecaster.py:771` constructs `RenewableFeatureBuilder` with no explicit
source, so it takes `db.py`'s default. Flipping that default therefore changes
the lag and rolling features fed to the **four already-frozen serving
artifacts** the moment it merges — without retraining them. That is train/serve
skew, not a training-data change.

Worse, it would be silent: the served artifacts
(`models/AT/solar/production/model.joblib`,
`models/BE/wind_offshore/production/model.joblib`, …) carry
`algorithm`, `feature_columns`, `model_version`, `training_metrics`, `saved_at`
— and **no training-source field at all**. Nothing in the artifact or the loader
could detect the mismatch.

This is the same failure shape as the xgboost-intercept trap in CLAUDE.md: no
crash, no failing test, just a model that quietly predicts from inputs it was
not fitted on. Two recommendations follow, neither of which is mine to make:

1. The switch must not reach serving unpaired with a retrain of the ten served
   country/stream pairs. Either land it together with the retrain, or have the
   serving path pin its source until the retrain lands.
2. `save_model` should record the training source and the loader should refuse a
   mismatch — the pattern `src/xgboost_artifact_guard.py` already establishes
   for the interpreter. That is Founding Engineer territory (serving path), so
   it is a proposal, not a patch in this branch.

## 6. Operational note

`energy-forecast/.env` sets `DATABASE_PATH=C:/Code/energy-data-gathering/energy_dashboard.db`
— a path that does not exist, one directory away from the 3.0 GB stale decoy
CLAUDE.md warns about. It fails loudly (`sqlite3.OperationalError`) rather than
silently reading the decoy, so nothing here is contaminated, but every run must
pass `ENERGY_DB_PATH` explicitly. `.env` is gitignored and machine-local, so
this is a note, not a patch.
