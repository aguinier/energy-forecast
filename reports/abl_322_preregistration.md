# ABL-322 — Offshore pilot: pre-registration and blocked-window measurements

**Disposition: PRE-REGISTERED. No model trained, no promotion, no serving change.**

Generated: 2026-08-12 UTC
Replica: `C:\Code\able\data\energy_dashboard.db` (9,432,453,120 bytes), opened with SQLite `mode=ro`, `uri=True`.
Sidecar: not opened. Nothing was written to either database.
Interpreter: `.venv\Scripts\python.exe` (Python 3.14.3, xgboost 3.3.0) — the rail.
Reproduce with `scripts/abl322_preregistration_probe.py`.

ABL-322 is blocked on **ABL-331** (per-artifact training source) and **ABL-332**
(the hourly feature builder), both in progress with the Founding Engineer. No
training was performed. This document freezes the gate *before* any DE/NL model
exists — which is the one part of the deliverable that is strictly better done
while blocked — and records three measurements that were available read-only.

## 1. The gate, frozen

Windows, run instants, metric, bands and minimum counts are **inherited verbatim
from `experiments/ABL195/config.json`**. They were fixed by a prior issue before
ABL-322 existed, so they are not a choice made here and cannot be shopped against
a challenger score.

| | |
|---|---|
| Fit targets | 2026-01-14 00:00 → 2026-07-11 00:00 (exclusive) |
| Gate targets | 2026-07-11 00:00 → 2026-08-10 00:00 (exclusive) |
| Pairs | DE `wind_offshore`, NL `wind_offshore` — xgboost, ABL-195's frozen params |
| Training source | `energy_generation`, per-artifact (ABL-331) |
| Metric | WAPE |
| Baseline | literal seasonal-naive D-7 |
| Bands | 24-36h, 36-48h, 48-64h |
| Minimum n | 684 / 684 / 456 |
| Bar | challenger WAPE < D-7 WAPE in **all 6 cells**, each meeting its minimum n |

**Neither country has an incumbent.** `forecasts` holds `wind_offshore` rows for
BE (33,024) and FR (32,664) only; DE and NL have **0 rows**. The incumbent column
will read "Not measured" by construction, not by omission.

## 2. The bar, measured before a challenger exists

Seasonal-naive D-7 on the frozen gate window, from `energy_generation`, on the
plain hourly target series. Taken while no DE/NL model existed:

| country | n | mean actual | **D-7 WAPE** | D-7 MAE |
|---|---:|---:|---:|---:|
| DE `wind_offshore` | 720 | 2,513.2 MW | **88.82%** | 2,232.2 MW |
| NL `wind_offshore` | 720 | 1,131.0 MW | **81.78%** | 925.8 MW |

Reference, same window and method, for the two offshore pairs ABL-195 already
gate-read: BE 104.87%, FR 73.90%. ABL-195's own pooled all-D+2 D-7 for BE was
105.6% against my 104.87% — a 0.7pp gap explained by its finite-intersection
scoring and its `energy_renewable` source. The method agrees; the harness's
per-band, finite-intersection D-7 remains the authoritative gate number.

For scale: ABL-195's offshore challengers beat D-7 by +26% to +30% relative.

## 3. Three findings from the blocked window

### 3.1 FR — an already-serving pair — is also a 15-minute country

This is the finding that matters most, because it bears on whether ABL-332 must
land before this pilot trains.

Rows with non-null `wind_offshore_mw`, 2026-01-14 → 2026-08-10, by minute offset:

| country | `energy_renewable` | `energy_generation` | cadence |
|---|---:|---:|---|
| BE | 4,964 (`:00` only) | 4,992 (`:00` only) | hourly |
| **FR** | **19,856 (`:00`/`:15`/`:30`/`:45`)** | 17,895 (4 offsets) | **15-minute** |
| DE | 19,858 (4 offsets) | 19,968 (4 offsets) | 15-minute |
| NL | 19,852 (4 offsets) | 19,968 (4 offsets) | 15-minute |

FR offshore is **15-minute in both tables**. It was trained through the same
hourly ABL-183 builder, gate-read by ABL-195 on this exact window, and it
**passed in all three bands** (+30.6% / +30.6% / +28.2% skill vs D-7). It serves
today with 32,664 rows.

So the hourly-subsampling hazard is not a DE/NL-specific precondition — it
already applies to a pair in production, and that pair cleared the bar with a
wide margin. ABL-332 is a real improvement opportunity; the evidence does not
support it being a *correctness* gate on ABL-322. See §4.

### 3.2 The aggregation convention cannot move the bar

The `:00` instant versus the mean of its own hour, over fully-observed hours only:

| country | hours | mean | instant−mean MAE | as WAPE | bias |
|---|---:|---:|---:|---:|---:|
| DE | 4,992 | 3,155.3 MW | 114.7 MW | 3.64% | −0.51 MW |
| NL | 4,992 | 1,621.9 MW | 79.3 MW | 4.89% | −0.69 MW |
| FR | 4,473 | 702.7 MW | 40.7 MW | 5.79% | −0.08 MW |

Sampling the instant is noisy (3.6–5.8% WAPE against the hourly mean) but
essentially **unbiased** (|bias| < 0.7 MW on means of 700–3,200 MW). Carried
through to the gate baseline, switching conventions moves D-7 by **0.04pp** for
DE (88.82 → 88.86) and **0.01pp** for NL (81.78 → 81.79).

Whatever ABL-332 decides — keep the `:00` read, or aggregate to the hourly mean —
**it cannot materially move the bar this gate is read against.** Its effect is on
training-set size and model quality, not on the target being cleared. I have
therefore pre-registered the `:00` instant, matching the served hourly product.

### 3.3 `energy_generation` is not uniformly more complete than `energy_renewable`

FR offshore has a coverage hole in `energy_generation` across the gate window:
**441 of 720 gate hours present, 273 scored** after D-7 pairing, against BE, DE
and NL at 720/720. `energy_generation` also carries *fewer* FR rows than
`energy_renewable` over the fit window (17,895 vs 19,856).

This does not touch ABL-322 — **DE and NL are both complete at 720/720**, and
19,968 rows each with no missing quarter-hours. It is recorded because the
"`energy_generation` is the clean source" framing is true for DE/NL and not
universally true, which matters to the 37 pairs behind this pilot.

## 4. Consequence for the blockers

> **Superseded in part by §7 (2026-08-12).** ABL-331 has since landed. The
> paragraph below is kept as written for provenance; read §7 for current state.

- **ABL-331 remains a genuine blocker.** I cannot train DE/NL from
  `energy_generation` while the source is a global constant read at inference
  time — that is precisely what ABL-321 refused. Nothing in these measurements
  changes that.
- **ABL-332 is, on this evidence, a quality improvement rather than a
  correctness gate on ABL-322.** FR is the existence proof: a 15-minute country,
  trained through the hourly builder, passing this same gate by +28–30%. And
  §3.2 shows the bar itself is invariant to the fix. If the CEO wants the pilot
  moving, ABL-322 can train once ABL-331 lands and be **re-read after ABL-332**,
  which also yields a clean before/after on the builder change at two-country
  cost. I am not asserting the pilot should proceed — that is a sequencing call —
  only that the blocking rationale should be made on this evidence.

## 5. The harness corrections this issue owns

Four sites in `scripts/evaluate_wind_retrain.py` stand between this
pre-registration and a runnable pilot. Enumerated against `25f94c8`; the first
was missed by the original revision of this section and is the most
consequential of the four.

| # | site | defect | owner |
|---|---|---|---|
| 1 | `:178` | `RenewableFeatureBuilder(...)` is constructed **without `actuals_source`** | ABL-322 |
| 2 | `:64` | `_constant_runs` hardcodes `FROM energy_renewable` | ABL-322 |
| 3 | `:34-36` | `PAIRS` is `wind_offshore: (BE, FR)` — no DE, no NL | ABL-322 |
| 4 | `:188` | bare `joblib.dump` of 7 keys, not `Forecaster.save` | **ABL-342** |

**1 — the training source never reaches the builder.** `actuals_source` is an
optional kwarg (`src/wind_features.py:__init__`) and `None` takes `db.py`'s
default, `RENEWABLE_TYPE_SOURCE_TABLE = 'energy_renewable'`. The string
`actuals_source` does not occur anywhere in the harness. §1 of this document
pre-registers the training source as `energy_generation`; the harness as written
would fit both pairs on `energy_renewable` instead — the table holding NL
offshore's 447 provably zero-filled rows and 668 duplicate instants, which is
the precise outcome the ABL-321 dependency was written to prevent. It fails
silently: the run completes, the gate reads, and only the artifact's resolved
source records which table was used.

**2 — the contamination screen reads the wrong table.** For a pair trained from
`energy_generation` the ABL-188 screen inspects a table the model never saw, so
it would report a clean window while the actual training source went unscreened.

**3 — neither pilot pair is in scope.** `PAIRS` also trains three `wind_onshore`
pairs that ABL-322 explicitly excludes.

**4 is not mine.** The CEO's ABL-342 (2026-08-12) scopes "route **both** gate
harnesses' artifact writes through `Forecaster.save`", which includes this file.
Both ABL-342's description and the CEO's ABL-322 comment record this site as
"already corrected on `ABL-322-preregistration` (`25f94c8`)". **That is not
so** — `25f94c8` added `scripts/abl322_artifact_shape_probe.py`, which
*demonstrates* the skew and does not repair it. The bare `joblib.dump` is intact
on this branch. Verified live on the rail (Python 3.14.3, xgboost 3.3.0): the
harness shape loads clean and resolves to `energy_renewable`, `Forecaster.save`
to `energy_generation`.

**Sequencing.** 1–3 are independent of both blockers but touch the same file as
4, so they are made in the diff that runs the pilot, rebased onto a tree
containing ABL-342 — not before it, to avoid contending with the Founding
Engineer for the file. The harness is additionally shared with the reproducible
ABL-195 and ABL-253 gate reads, so those must still reproduce after the edit.

## 6. Limits

- No model was trained; no challenger number exists in this document.
- All D-7 figures are pooled over the gate window on the plain hourly series.
  They are the pre-committed bar, not the harness's per-band finite-intersection
  D-7, which will differ slightly and is authoritative for the gate read.
- One 30-day summer holdout. Out-of-sample by target timestamp, not a year-round
  robustness claim.
- No production deploy, serving-registry change, model promotion, ingest change,
  dashboard change, replica write or sidecar write was performed.

## 7. Update 2026-08-12 — blocker state, and the artifact-shape trap

Appended after the Founding Engineer reported ABL-331 merged. §1–§3 are
unchanged: **no window, band, metric, baseline or minimum n is touched here**, no
challenger score exists, and this registration is not voided by anything below.

### 7.1 Blocker state

| | state | effect on ABL-322 |
|---|---|---|
| **ABL-331** per-artifact training source | **done** — `1d6f9ee` (PR #14) | cleared |
| **ABL-339** window closed on the artifact's own source | **merged** — `1a133d6`, merge `87edd50` (PR #17) | cleared; was already moot, see §7.3 |
| **ABL-332** hourly builder discards 15-min samples | **`in_review`** | **still blocking** |
| ABL-340 `scripts/train.py` import | open | **does not touch ABL-322**, see §7.4 |

PR #17 merged after the Founding Engineer's comment described it as awaiting
merge; `origin/main` is at `87edd50`.

### 7.2 The gate harness writes an artifact that silently serves from the wrong table

This is the finding of this update, and it is exactly the class of thing the
pilot exists to catch. It is in **my** harness, not the Founding Engineer's code.

`scripts/evaluate_wind_retrain.py:186-191` does not save through
`Forecaster.save`. It writes a bare `joblib.dump` of seven keys. Post-ABL-331,
`Forecaster.load` resolves an absent `training_source` to
`LEGACY_RENEWABLE_TRAINING_SOURCE = 'energy_renewable'` — correct and deliberate
for the 88 legacy artifacts, and **wrong for a pair fitted on
`energy_generation`**.

Measured, not inferred. Both shapes built and round-tripped on the rail
(`.venv`, Python 3.14.3, xgboost 3.3.0) against worktree `87edd50`; synthetic
2-feature fit, no replica read:

| | keys written | `training_source` | `base_score` / `xgboost_version` | `Forecaster.load` | **resolves to** |
|---|---|---|---|---|---|
| **A** — harness bare `joblib.dump` | 7 | absent | absent | **OK, no error** | **`energy_renewable`** |
| **B** — `Forecaster.save(training_source='energy_generation')` | 14 | present | present | OK | **`energy_generation`** |

Every key in `load` is read with `.get(..., default)`, so shape A does not fail —
it loads clean and serves the wrong table. A DE/NL artifact written this way to
`models/<CC>/wind_offshore/model.joblib` would be **fitted on
`energy_generation` and served from `energy_renewable`**: precisely the
train/serve skew ABL-321 was withheld to prevent, reintroduced through the
artifact writer rather than the global constant. NL offshore is the worst pair in
the audit for that table — 447 zero-filled rows, 668 disagreeing duplicates.

Second defect in the same shape: absent `base_score`/`xgboost_version` make
ABL-183's `assert_survived_load` intercept witness a **no-op**. That is the guard
against an xgboost-3.3.0 pickle loading under 2.1.4 with its intercept silently
reset to 0.5. The pilot artifact would ship with that protection disabled.

Neither defect is reachable by test today because no renewable artifact on disk
was written by this harness to a serving path.

### 7.3 ABL-339 — measured, and it did not bite these two pairs

The Founding Engineer asked whether `energy_renewable` carries DE/NL offshore
rows at all, since an absent series would have sent the pre-fix window end to
`datetime.now()`. It does, and the head is not behind:

| table | DE non-null rows | DE max ts | NL non-null rows | NL max ts |
|---|---:|---|---:|---|
| `energy_renewable` | 33,503 | 2026-08-12 13:00 | 27,485 | 2026-08-12 13:00 |
| `energy_generation` | 196,756 | 2026-08-12 12:45 | 196,757 | 2026-08-12 13:00 |

So the pre-fix end would have closed on `energy_renewable`'s last instant, which
is **level with or 15 minutes ahead of** `energy_generation`'s — truncation of
zero. The `datetime.now()` fall-through never applied here.

Independently, **ABL-339 could not have reached this gate read**: the harness
passes explicit bounds (`fit_start - 14d`, `gate_end`) to
`RenewableFeatureBuilder` at `:179-180`, so `end_date=None` never arises. The
registered windows are closed on both ends.

The depth difference is at the *start*, not the end — `energy_renewable` begins
2025-09-08 (DE) / 2025-11-09 (NL) against `energy_generation`'s 2021-01-01. The
registered fit window opens 2026-01-14, after both, so it is not affected.

### 7.4 ABL-340 does not touch this issue

`scripts/train.py` is not on ABL-322's path. The gate harness imports `src.*`
directly (`from src.wind_features import RenewableFeatureBuilder`) and fits an
`XGBRegressor` itself at `:184`. It has never routed through `scripts/train.py`,
including for the reproducible ABL-195 and ABL-253 reads. No workaround needed
and no waiting on ABL-340.

### 7.5 Harness corrections this issue now owns

Superseding §5, three corrections, all in `scripts/evaluate_wind_retrain.py`,
all made in the same diff that runs the gate:

1. **`:179-180`** — pass `actuals_source='energy_generation'` to
   `RenewableFeatureBuilder`. Not passed today, so it defaults through the global
   constant to `energy_renewable`. Without this the model is *fitted* on the
   wrong table.
2. **`:186-191`** — write the serving-path artifact through `Forecaster.save`
   with `training_source='energy_generation'`, not the bare `joblib.dump`. §7.2.
   Without this the model is *served* from the wrong table.
3. **`:64`** — `_constant_runs` hardcodes `FROM energy_renewable`; must screen
   the table actually trained from. Carried unchanged from §5.

Corrections 1 and 2 are independent failure modes and both are silent. Together
they are the reason this pilot was worth running on two countries.

## 8. Update 2026-08-12 — ABL-345 sized against the registered window

ABL-345 was made a first-class blocker on this issue after the CEO triage, and
the finding behind it is mine (`0e6b8ba`). This section does not contest the
blocker. It corrects the number attached to it, because that number is being
used to size the 37 tranches behind this pilot.

Reproduce with `scripts/abl322_source_sensitivity_probe.py` (read-only,
`mode=ro`, `uri=True`; nothing written to either database).

### 8.1 The "13-16%" figure does not describe this pilot

The triage states the pilot "fits 13-16% of the window §1 pre-registers",
from whole-table spans — 337 d (DE) / 275 d (NL) of `energy_renewable` against
2,049 d of `energy_generation`.

That ratio is a property of the **tables**, not of the **registered window**.
§1 does not register a 2,049-day fit. It inherits ABL-195's frozen 178-day fit
window (2026-01-14 → 2026-07-11); with the builder's 14-day lag warm-up the
earliest instant this pilot asks any source for is **2025-12-31**. DE
`energy_renewable` opens 2025-09-08 and NL 2025-11-09 — **both before it**.

Measured coverage inside the registered windows, `wind_offshore_mw` non-null:

| window | | `energy_generation` | `energy_renewable` | ratio |
|---|---|---:|---:|---:|
| builder (2025-12-31 → 2026-08-10) | DE | 21,312 | 21,202 | **99.5%** |
| | NL | 21,312 | 21,196 | **99.5%** |
| fit (2026-01-14 → 2026-07-11) | DE | 17,088 | 16,978 | **99.4%** |
| | NL | 17,088 | 16,972 | **99.3%** |
| gate (2026-07-11 → 2026-08-10) | DE | 2,880 | 2,880 | **100.0%** |
| | NL | 2,880 | 2,880 | **100.0%** |

Neither source truncates the registered window. The gate window — the one that
produces the reported number — is **row-for-row identical in count**. The
shortfall is 0.5-0.7%, not 84-87%.

### 8.2 What it is actually worth: the bar moves 0.00pp on DE, -0.97pp on NL

The harness computes its baseline from `builder._actuals`
(`evaluate_wind_retrain.py:200`, `attach_baselines(selected, builder._actuals)`),
so the defect moves the **D-7 bar** as well as the fit. §2 registered the bar
from `energy_generation`. Recomputed from `energy_renewable` by the identical
method, same window, n=720 in every cell:

| pair | registered bar (`energy_generation`) | counterfactual (`energy_renewable`) | delta |
|---|---:|---:|---:|
| DE `wind_offshore` | 88.82% | 88.82% | **+0.00pp** |
| NL `wind_offshore` | 81.78% | 80.81% | **−0.97pp** |

DE is unmoved to two decimal places. NL's bar tightens by about one point,
because `energy_renewable` carries 4 zero-filled rows inside the gate window
where `energy_generation` carries none, and a mean actual 10.1 MW higher.
Same under the `hourly_mean` aggregation (DE +0.00pp, NL −0.98pp), so this is
not an artefact of the `:00` convention.

For scale: ABL-195's offshore challengers beat D-7 by 26-30 **percentage points
relative**. A 0.97pp baseline shift does not decide a pass at that margin — but
it is not nothing on a pair with no incumbent and no prior.

### 8.3 ABL-345 still blocks this issue, on compliance not magnitude

Stating the correction is not an argument to unblock, and I am not asking for
one. Three reasons survive the correction, and the first is dispositive on its
own:

1. **The run cannot comply with its own pre-registration.** §1 registers
   `training_source = energy_generation`. Until a source argument reaches the
   builder there is no way to execute the registered protocol. Magnitude is
   irrelevant to this one.
2. **Contamination inside the fit window.** `energy_renewable` carries 12 (DE)
   and 35 (NL) fabricated zeros where `energy_generation` carries 0 — the
   ABL-67 / ABL-111 class. NL has 4 more inside the *scoring* window.
3. **Provenance.** The artifact would record `training_source='energy_renewable'`,
   contradicting §1 — which is precisely the skew ABL-342 built the writer to
   make visible.

### 8.4 The ABL-345 guard will be silent for this pilot — measured

The triage instructs me to pass the source flag explicitly, noting ABL-345 adds
a guard that refuses a fit window starting before the source's first row, "so a
mistake here should become loud".

**Measured, that guard cannot fire on either pilot pair.** Both tables begin
before the builder's 2025-12-31 start (`covers_builder_start: YES`, all four
table/country combinations in §8.1). A silent fall-through to `energy_renewable`
passes the guard cleanly and returns a plausible gate read.

This is a finding *for* the guard, not against it: it is correctly scoped to the
33 tranche pairs whose `energy_renewable` opens 2025-10 or later, and it is
those pairs it will protect. It simply does not protect this pilot. So §1 gains
one amendment:

> **Amendment (§8.4).** The source is passed explicitly on the command line, and
> after the run the artifact's recorded `training_source` is asserted to equal
> `energy_generation` for both pairs before any gate number is reported. The
> explicit flag is the only protection this pilot has; the recorded provenance
> is the only check that it took effect.

### 8.5 Harness-correction ownership, superseding §7.5

§7.5's three corrections have moved. Recorded so the pilot diff does not
collide with an in-flight one:

| §7.5 correction | site on `84371b3` | owner now |
|---|---|---|
| 1. builder `actuals_source` | `evaluate_wind_retrain.py:179` | **ABL-345** (Founding Engineer) |
| 2. artifact via `Forecaster.save` | `:186-191` | **landed** — ABL-342, PR #21 |
| 3. `_constant_runs` source | `:64` | **ABL-345** (Founding Engineer) |

Nothing in §7.5 remains mine. What remains ABL-322's own is the `PAIRS` entry —
`wind_offshore` is `("BE", "FR")` on `84371b3` and the pilot needs DE and NL —
plus the gate read itself.

An implementation of corrections 1 and 3, with the measurements behind them,
exists unpushed on branch `ABL-322-pilot` (`8662989`). It is offered to ABL-345
as reference and will not be landed from here. One divergence to settle there:
that branch names the flag `--renewable-source`; the CEO's instruction on this
issue says `--actuals-source`. **ABL-345 picks the name**; §1's registered
command line will be written against whatever ships.

### 8.6 The intercept witness does not cover the `wind_onshore` tranches

Confirmed by reading `src/forecaster.py:922-931`: the `base_score` witness and
`xgboost_version` are written only when `algorithm == "xgboost"`, since
`base_score` is a booster attribute. `PAIRS` on `84371b3` sets `wind_offshore`
to xgboost and `wind_onshore` to catboost.

So this pilot (`wind_offshore`, xgboost) **is** covered by the ABL-183 guard —
its artifacts carry the witness. The `wind_onshore` half of the ABL-316
tranches is not, and cannot be by this mechanism. Correct behaviour, not a short
write; recorded here so it is not rediscovered mid-tranche.

### 8.7 Preconditions, re-checked at this heartbeat

- `87edd501` and `84371b3` are both ancestors of `origin/main` (now `82529c0`).
- **ABL-346 landed** (`d0d19aa`): `scripts/train.py` now exits 2 rather than
  falling through to the replica when `FORECAST_OUTPUT_DB` is unset. It does not
  change this pilot — §7.4 established the gate harness never routes through
  `scripts/train.py` — and the harness already surfaces `--sidecar-db` at
  `:155`. The harness's only database access is `_ro_connect` (`:61`); it reads
  the replica read-only and writes no rows.
- Blockers verified live, not read off the card: **ABL-332** PR #16 `OPEN`,
  `MERGEABLE`, awaiting the Board click; **ABL-345** `in_progress` with the
  Founding Engineer. No gate read and no training was run this heartbeat.
