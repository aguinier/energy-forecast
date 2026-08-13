# ABL-385 side-finding: why DE solar's fit is short

Measured 2026-08-13 against the live replica `C:\Code\able\data\energy_dashboard.db`
(9.43 GB — verified not the 3.0 GB `energy-data-gathering` partial snapshot),
opened read-only. This is a provenance note, not a defect report: the condition
below is already adjudicated (ABL-188), already sized correctly in the code, and
already guarded. Nothing here is new to file.

It is recorded because **ABL-385's own premise and ABL-375's registration both
name the wrong cause** for the fact they build on, and the right cause changes
what "DE has a short fit" means.

## What the two parent documents say

`experiments/ABL375/config.json`:

> "DE solar in energy_renewable begins 2025-09-08T22:00; after feature
> lag/rolling warmup the featured frame starts 2025-11-26."

ABL-385's own description:

> "`energy_renewable` DE solar starts 2025-09-08, so ABL-375's fit spanned 156
> days and ~3,700 rows."

Both treat 2025-09-08 as the start of usable DE solar history, and ABL-375
attributes the 79-day gap to 2025-11-26 to feature lag and rolling warmup.

## What is actually there

`energy_renewable.solar_mw` for DE is **non-null and exactly 0.0** for
**6,408 consecutive quarter-hour rows**, `2025-09-08T22:00` through
`2025-11-14T15:45`. Not one non-zero value in 67 days. The first positive
reading is 0.02 MW at `2025-11-14T16:00`.

Three checks that rule out the innocent readings:

| check | result |
|---|---|
| Is it null-as-zero at the read? | No — `COUNT(solar_mw)` returns the full row count. They are explicit zeros. |
| Was the feed down? | No — `wind_onshore_mw` on **the same rows** is live: 6,313 rows, **zero** of them 0.0, max 46,810 MW. |
| Could Germany have produced no solar? | No — every midday hour is affected: hours 10–14 each have 264 rows in the block, **all** 0.0. Germany carries ~90 GW of installed solar. |

So the ingest wrote `solar_mw = 0` where it meant "not reported", for one
production type, on one country, for 67 days, while writing the neighbouring
type correctly.

## This is ABL-188, and it is handled

ABL-188 ("Adjudicate 5,096 zero-filled DE solar actuals…", **done**) is this
defect. Two notes on the bookkeeping:

- Its **description** records "5,096 quarter-hour rows … through 2025-10-31
  23:45". The run measured today ends 2025-11-14T15:45 and is **6,408** rows —
  1,312 rows and 14 days longer than the issue text.
- Its **remediation is correctly sized**: `src/db.py:667` names the run as
  "6,408 quarter-hours", matching the replica exactly.

The gap is in the issue prose, not in the guard, and it is harmless because
`exclude_suspect_constant_runs` is **condition-based, not date-based** — it
infers the series cadence and rejects long bit-identical runs wherever they sit.
A date-bounded fix would have leaked the last 14 days; this one does not.

**Verified on today's read:** the loaded DE/solar training frame begins
`2025-11-14 16:00`, exactly one step past the end of the run. The whole block is
excluded. **No ABL-385 fit, and no ABL-375 fit, trained or scored through these
rows.**

## The correction that matters

DE solar's short fit is **not** a data-availability fact. It is a data-quality
consequence. The 79 days ABL-375 attributes to "feature lag/rolling warmup"
decompose as:

| cause | span |
|---|---:|
| ABL-188 zero-run, excluded by the guard | 2025-09-08 → 2025-11-14 (**67 days**) |
| feature lag / rolling warmup | 2025-11-14 → 2025-11-26 (**12 days**) |

Warmup is real but minor; the zero-run is the dominant term and was not named.

Why it matters for this issue:

1. **It is the strongest available support for ABL-385's "underdetermined fit"
   reading.** DE's usable solar history is 270 days against BE's 953 and FR's
   1,318 — and the shortfall is repairable, not intrinsic.
2. **It makes AT the natural control for DE.** AT solar spans 277 days from
   2025-11-07 with no such block (its first positive value is 7 hours after its
   first row — an ordinary overnight gap). AT and DE have near-identical fit
   lengths and only DE has the contamination history, which is precisely the
   contrast registered prediction **P3** reads.
3. **If ABL-188's rows are ever repaired upstream, DE's fit length roughly
   doubles** and the DE-specific spread this issue measures should be re-read.
   That is a Founding Engineer / ingest decision, not this issue's.

## Contamination statement for the sweep

- **ABL-188 (DE solar zero-run)** — touches the window. Excluded at read by
  `exclude_suspect_constant_runs`, identically for every arm, seed and window,
  by construction. Verified above.
- **ABL-337 (impossible night solar)** — touches the fit frame: DE 4 rows,
  FR 517, AT 0, BE 0. Dropped from the fit and never from the score, identically
  on every arm, via `--drop-impossible-night`.
- **ABL-71 / ABL-67 / ABL-109 / ABL-111** — load and net position. Do not touch
  this scope. ABL-71 remains a provenance caveat: known wrong-write modes are
  load and net position, which is not proof renewable ingest is clean — and the
  finding above is exactly why that caveat is kept.

Every spread this issue reports is a **within-cell** quantity, across seeds
only. Contamination identical across the seeds of a cell shifts the cell's level
and cancels out of its CV. The cross-window and fit-length comparisons do not
have that protection, which is why the above is stated per window rather than
once.
