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

## 5. A harness correction this issue owns

`scripts/evaluate_wind_retrain.py:64` hardcodes `FROM energy_renewable` in
`_constant_runs`, the ABL-188 contamination screen. For a pair trained from
`energy_generation` that screens a table the model never saw — it would report a
clean window while the actual training source went unscreened. This is in my own
gate harness, not the Founding Engineer's code, and it is independent of both
blockers. It must take the pair's training source before ABL-322 is read.
`PAIRS` in the same file also needs DE and NL under `wind_offshore`.

Not changed in this commit: the harness is shared with the reproducible ABL-195
and ABL-253 gate reads, so it is edited when ABL-322 trains, in the same diff
that runs it.

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
