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
