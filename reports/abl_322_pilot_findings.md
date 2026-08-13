# ABL-322 — offshore tranche + pilot: findings

Companion to the generated gate read in `reports/abl_322_pilot_gate.md` and
`experiments/ABL322/results.json`. The registration this is read against is
`experiments/ABL322/config.json`, frozen 2026-08-12T21:00Z, before any DE or NL
`wind_offshore` model existed.

**Disposition: PASS, 6/6 registered cells.** No promotion. Evidence only.

---

## 1. The gate read

| | window | n | metric | challenger | baseline (seasonal-naive D-7) | margin |
|---|---|---:|---|---:|---:|---:|
| DE 24-36h | 2026-07-11 → 2026-08-10 | 720 | WAPE | 66.11% | 88.86% | **+25.6%** skill |
| DE 36-48h | 2026-07-11 → 2026-08-10 | 720 | WAPE | 65.66% | 88.86% | **+26.1%** |
| DE 48-64h | 2026-07-11 → 2026-08-10 | 510 | WAPE | 66.15% | 87.09% | **+24.0%** |
| NL 24-36h | 2026-07-11 → 2026-08-10 | 720 | WAPE | 60.46% | 81.79% | **+26.1%** |
| NL 36-48h | 2026-07-11 → 2026-08-10 | 720 | WAPE | 61.26% | 81.79% | **+25.1%** |
| NL 48-64h | 2026-07-11 → 2026-08-10 | 510 | WAPE | 63.75% | 88.51% | **+28.0%** |

Out-of-sample by target timestamp; fit window 2026-01-14 → 2026-07-11 exclusive,
gate targets never fitted. Registered minimum n is 684/684/456; every cell clears
it (720/720/510). Source table `energy_generation` for the fitted series, all
lag/rolling features, both baselines, the gate actuals and the contamination
screen.

**Integrity check that matters more than the pass.** The D-7 bar was measured on
2026-08-12 by `scripts/abl322_preregistration_probe.py` *while no challenger
existed*: DE 88.82%, NL 81.78%. The harness, fitting and scoring independently a
day later, produced 88.86% and 81.79%. The bar was not moved.

**Contamination.** None of the four known issues touches this window's targets.
ABL-67 is net-position-only; ABL-109 and ABL-111 are load-only; ABL-71's known
wrong-write modes are load and net position — a provenance caveat, not proof wind
ingest is clean. ABL-188 is an `energy_renewable` zero-fill defect and these
pairs train from `energy_generation`; the constant-run screen was run against
`energy_generation` (the table actually fitted) and found **no** ≥24-hour
bit-identical run in either pair. 34,176 of 34,176 intended fit rows retained,
0 excluded, for both countries.

**Which database the fit actually read.** ABL-355 found that `--replica-db`
governed only the incumbent, TSO and contamination screen, while the fitted
series and the weather archive came from `config.DATABASE_PATH` (that is,
`ENERGY_DB_PATH`). This gate read predates that fix and the generated report
records only the one path, so these numbers are trustworthy only if the two
resolved to the same data. They did. The check, rather than the assurance:

- **The two values that would have made them differ are both dead paths.** With
  `ENERGY_DB_PATH` unset, `config.DATABASE_PATH` falls back to
  `/data/energy_dashboard.db` → `C:\data\energy_dashboard.db`, which does not
  exist. The checked-in `.env` — absent from every worktree, since it is
  gitignored — names `C:\Code\energy-data-gathering\energy_dashboard.db`, whose
  *directory* does not exist. `db.get_connection` opens read-only with
  `mode=ro`, which raises on a missing file rather than creating one, so a run
  under either value would have died on its first read instead of retaining
  34,176 of 34,176 rows.
- **The 3.08 GB partial snapshot is ruled out outright.** The decoy at
  `energy-data-gathering/energy_dashboard.db` — the nearest file to every wrong
  path this module has been pointed at — has **no `energy_generation` table at
  all**. It cannot be the source of a fit whose recorded `training_source` is
  `energy_generation`.
- **The two remaining candidates are identical over the window.** Exactly two
  files on this workstation carry `energy_generation`: the live replica
  `C:\Code\able\data\energy_dashboard.db` (9,432,453,120 bytes — the size the
  report records for `--replica-db`) and `backups_ops/ops_backup_2026-08-12.db`.
  Over the builder's full span, 2025-12-31 → 2026-08-10, they agree byte for
  byte: DE target series n=21,312 sha256[:16] `7c9ebf8cd0e576ea` in both, NL
  n=21,312 `776b3312038f5506` in both, and `weather_data` 1,925,928 rows with
  max `forecast_run_time` 2026-08-09 23:00:00 in both. Whichever the environment
  named, the fit saw the bytes the scoring saw.
- **Every count in the report reproduces from the live replica's coverage.**
  17,088 fifteen-minute DE offshore rows in the fit window ÷ 4 (ABL-332 hourly
  mean) = 4,272 unique fit targets, × 8 vintages = 34,176 fit rows; 2,880
  gate-window rows ÷ 4 = 720, the reported n for the 24-36h and 36-48h cells.
  NL is identical. Nothing in the report requires a second source to explain it.

So the ABL-355 defect is real and worth its fix, but it did not move this gate
read: the numbers stand as reported and no re-run is needed. What is genuinely
missing is the *record* — once ABL-355 lands, the harness prints both paths and
a future run of this scope states them instead of needing this reconstruction.

## 2. A PASS here is not a good model — read this before promoting anything

The registered bar is "beats seasonal-naive D-7". On offshore wind D-7 is close
to uninformative (87-89% WAPE for DE), so a pair can clear it with very little
dynamic skill. DE does exactly that:

| | challenger WAPE | correlation | slope | MAE | **TSO WAPE** |
|---|---:|---:|---:|---:|---:|
| DE | 65.95% | **0.143** | **0.066** | 1,637.8 MW | **21.13%** |
| NL | 61.61% | 0.44 | 0.22 | 699.7 MW | 68.97% |

Same n = 1,950 rows for every number in that table.

A slope of 0.066 and correlation of 0.14 describe a model that is close to
predicting a constant near the mean. It beats D-7 because D-7 is bad, not because
it tracks German offshore wind. **The DE TSO forecast achieves 21.1% WAPE on the
identical rows — roughly three times better than our challenger.** TSO is
revision-contaminated and cannot support promotion, but a gap that size is not a
revision artifact.

NL is the healthier pair: correlation ~0.44 and it beats its TSO (69.0%).

The actionable read: **DE offshore has large headroom and the TSO series is the
obvious next feature**, not more hyperparameter work. This is a model-quality
finding, and it is the reason I am not writing "promote these" anywhere.

## 3. Per-pair training cost — the number for tranche sizing

Wall-clock on the rail interpreter (`.venv`, Python 3.14.3, xgboost 3.3.0), one
pair at a time in a single process, on a workstation running other work:

| | fit rows | feature build | fit | gate build + predict | pair total |
|---|---:|---:|---:|---:|---:|
| DE | 34,176 | 55.9 s | 2.3 s | 7.3 s | **65.5 s** |
| NL | 34,176 | 46.1 s | 2.0 s | 8.6 s | **56.8 s** |

**≈ 60 s per country/stream pair. Use 90 s as the planning figure.**

The shape matters more than the total: **the fit is 2-3 seconds; the feature
build is 46-56 seconds**, ~85% of the cost. Tranche sizing is therefore governed
by feature-build throughput, not by model training, and buying a bigger model
(more estimators, deeper trees) is nearly free while adding features is not.

DE and NL are the *expensive* case — both are 15-minute countries (21,408
sub-hourly rows aggregated to 5,352 hourly means per ABL-332). Hourly countries
carry a quarter of the rows into the same aggregation. So ≈60 s is an upper
bound, not a median.

**Sizing the 37 remaining pairs: 37 × 90 s ≈ 55 minutes of compute.** That is one
sitting, and the constraint on the tranches is review and adjudication, not
compute. Caveat: this figure is xgboost/`wind_offshore` through the ABL-183 wind
builder. `wind_onshore` fits catboost, and solar runs a different harness and
builder — size the solar tranche off `evaluate_solar_retrain.py`'s own
measurement, not off this number.

## 4. What the pilot broke, which is why it was run on two countries

### 4a. The gate was unreadable for any pair with no incumbent

The first run of this pilot returned **0/6 cells, every score `None`, verdict
`FAIL`**. Nothing was wrong with the models or the data — 34,176/34,176 fit rows
retained, 0 excluded, 0 constant runs.

`common_scores` intersects on every column handed to it, and the harness handed
it `challenger, incumbent, seasonal_naive, persistence`. DE and NL
`wind_offshore` have **0 rows in `forecasts`**, so `incumbent` is NaN on every
row, the intersection is empty, and every cell scores nothing. The harness then
rendered that as `FAIL` — a model-quality verdict on a comparison that never
happened — and crashed in the report writer dividing `None` by `None`.

**Every new country in ABL-316's remaining 37 pairs has no incumbent.** Left
alone, this would have mis-dispositioned the entire programme, and each report
would have said "only 0/6 primary cells clear the registered bar" for models that
in fact clear all of them by 24-28 points.

Fixed by making the gate basis a registered property of the scope
(`GATE_BASIS`). `abl322-pilot` gates on `(challenger, seasonal_naive)` — the two
columns its registered bar actually names — and reports the incumbent and
persistence on their own intersection with that basis, each carrying its own n,
so an absent comparator reads *Not measured* instead of emptying the cell.

This is not bar-shopping: no challenger score existed when the change was made.
Every cell was `n=0` with `None` on both sides, so no performance information had
been revealed. The windows, bands, metric, baseline and minimum n are untouched.

**`abl195` deliberately keeps the four-way basis it was published under.** Its
48-64h cells scored 480 rows against the 510 the same report records as selected,
so the incumbent conjunct did drop rows there — re-basing it would silently move
numbers that have already been dispositioned. **Open question for whoever owns
that gate: ABL-195's published read should be re-taken under the narrower basis
before anyone compares its numbers to a tranche report.** I did not do it here.

### 4b. A zero-row cell now reads UNREADABLE, not FAIL

A cell that scored no rows did not lose a race. Reporting `FAIL` invites exactly
the wrong next move — feature work on a model that was never measured. Any run
with a zero-row cell now returns verdict `UNREADABLE` with a recommendation that
refuses to disposition.

### 4c. Two hardcoded ABL-195 specifics leaked into every scope's report

The contamination section rendered one hardcoded sentence about a **BE** offshore
zero run for any scope, including scopes that never fit BE; it now tabulates the
runs the screen actually found. The protocol-count sentence (210/570/720/720/510)
is a measured ABL-195 fact and is now rendered only for that scope.

## 5. The DE orphaned artifact

Confirmed as described in the issue. `models/DE/wind_offshore/` holds
`candidate/`, `centroid/`, `multipoint/` and `production/model.joblib` but **no
top-level `model.joblib`**, the only path `Forecaster.load` reads. `models/NL/`
does not exist at all. Measured today:

| | top-level `model.joblib` | `production/model.joblib` | rows in `forecasts` |
|---|---:|---:|---:|
| BE | 5,587,992 B | 5,587,992 B | 34,036 |
| FR | 5,676,526 B | 5,676,526 B | 32,664 |
| DE | **absent** | 662,350 B | **0** |
| NL | **absent** | — | **0** |

The instruction not to rescue the orphan was right, and the pilot gives a second
reason. BE and FR's `production/` artifact is byte-identical in size to their
top-level one; DE's 662,350 B is not that shape. The freshly trained DE artifact
here is **6,134,322 B** — same order as BE/FR, ~9× the orphan. The orphan is not
a smaller model of the same kind.

I deleted nothing. The variant directories (`candidate/`, `centroid/`,
`multipoint/`) are inert with respect to serving: `Forecaster.load` reads only
the top-level path, so they cannot be picked up by accident. They are safe to
leave in place; a separate cleanup issue could remove them, but nothing depends
on it.

## 6. Would a fresh top-level artifact be picked up? Yes — and that is why I did not write one

Traced, not assumed. `scripts/forecast_daily.py` iterates
`config.SUPPORTED_COUNTRIES × (FORECAST_TYPES + RENEWABLE_TYPES)`, calls
`Forecaster.load(country, type)` (`:158`), and on `FileNotFoundError` logs
`[SKIP] … Model not trained yet` (`:177`). Measured on the current config:

- `DE` and `NL` are both in `SUPPORTED_COUNTRIES`;
- `wind_offshore` is in `RENEWABLE_TYPES`;
- `SKIP_RENEWABLE_TYPES` has **no entry** for either DE or NL.

So there is no country gate anywhere in the path — ABL-319's finding reproduces.
**The presence of `models/<CC>/wind_offshore/model.joblib` is the only thing
standing between these trained models and live rows in `forecasts`.**

That is precisely why the artifacts stay in `experiments/ABL322/artifacts/`.
Writing to the top-level path is not a filesystem detail here — it *is* the
promotion act, and it would start serving an unpromoted model on the next
scheduler run. The registration anticipated this and fixed the location in
advance (`"artifact_location": "experiments/ABL322/artifacts … not a serving
registry"`, `"promotion": "none"`).

**Acceptance criterion 1 asks for the artifact at the top-level path; criterion 4
forbids a serving change. On these two pairs those are the same action.** I have
honoured criterion 4 and the frozen registration. The CEO has everything needed
to close criterion 1 as a deliberate promotion decision — the copy is one command
per country, and the artifacts carry truthful provenance
(`training_source: energy_generation`, `base_score: 3263.82`, `xgboost_version:
3.3.0`), so they will serve from the table they were fitted on.

## 7. Feature-builder observations for the tranches

- The ABL-183 shared builder handled both countries **with no modification** and
  no per-country branch — the thing the issue asked to be told if it failed. It
  did not fail.
- ABL-332's hourly aggregation fired for both: `21408 sub-hourly rows → 5352
  hourly means, 0 partial hours`. Without it both pairs would have been fitted on
  the `:00` sub-sample, a quarter of their data.
- **23,674 of 34,176 fit rows (69%) carry a degraded lag-1d feature** in both
  countries, identically. It is the same figure for DE and NL, so it is a
  property of the eight-vintage schedule rather than of either country's data.
  Worth understanding before the tranches, since it applies to all of them.
- No weather-station, timezone or resolution handling needed country-specific
  work.

## 8. Boundaries

No promotion, no serving-registry change, no serving-config change, no ingest
change, no dashboard change. The replica was opened read-only
(`file:…?mode=ro`, `uri=True`); nothing was written to it or to the sidecar. All
artifacts are under `experiments/ABL322/`.
