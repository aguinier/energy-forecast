# ABL-639 — a partial target day was weighted like a whole one

Scope: the paired Student-t intervals in `scripts/abl607_d2_load_diagnosis.py`.
Machine record: `reports/abl_607_d2_load_diagnosis_completeness.json`, section
`section_k_day_completeness`. Nothing here is a refit, a promotion or a serving
change.

## The defect

`paired_daily` appends one daily WAPE difference per country-day and then keys
`T_CRIT` on `k = len(d)`. Neither half is wrong alone; together a country-day
scored on 2 surviving hours entered the interval with the weight of a 24-hour
day and counted as a full observation. `panel_a` is an inner merge, so the
truncation is per-country: the countries were not scored over the same window.

## Protocol

- Replica `C:\Code\able\data\energy_dashboard.db`, opened `mode=ro` via URI.
- Window `2026-08-13 08:00` → `2026-08-28 00:00` inclusive, 16 target days,
  24 countries, **n = 8451** scored pairs on `panel_a`. Out-of-sample.
- Baseline: the D-7 seasonal naive, as in ABL-246 §4.1 and ABL-607 §A.
- Contamination: ABL-111/ABL-109 zero-as-missing rows are dropped by
  `load_actuals` (`zero_rows_dropped = 0` over this window). ABL-71 (stale prod
  ingest) is untouched by this change and is a caveat on the level of the
  underlying diagnosis, not on the screen. ABL-67 does not reach `energy_load`.
- `--min-day-completeness` defaults to `0.0`, which screens nothing.

## The default is numerically identical, proved on the live replica

Three full runs, all against the same replica: `origin/main` at 20:13, the new
script at default at 20:20, `origin/main` again at 20:21.

| comparison | differing leaves |
|---|---:|
| control 20:13 vs control 20:21 | **0** — the replica held still, so the rest is a code comparison and not a vintage one |
| control vs new run at the default | **0** across every key the control emits |

The new run adds keys and moves nothing: `section_k_day_completeness`, three
`meta` entries, and `k_days_short` / `k_days_screened_out` on each paired frame.
Compared leaf-by-leaf at bit equality, not at a tolerance — the daily
differences are summed in iteration order, so a reordered day loop would move a
marginal `ci_lo` in the last bits and a tolerant comparison would not see it.

## What the screen finds

**7 of 384 country-days on `panel_a` are short** (12 of 360 on `panel_g`), and
they fall on two countries.

| country | k, all days | k, complete days only | mean diff | `ci_lo` | readable |
|---|---:|---:|---|---|---|
| LV | 16 | 10 | +4.20 → +2.39 | +0.64 → -2.64 | yes → no |
| EE | 16 | 15 | -1.09 → -1.16 | -2.86 → -3.06 | no → no |

**LV is the finding.** It is one of the five marginal cells ABL-622 exists to
resolve, and its readable loss does not survive the screen: the interval crosses
zero once the six short days are removed. The readable-loser set goes from
`AT CZ ES LV PL PT SE SI SK` (9) to `AT CZ ES PL PT SE SI SK` (8). GR remains
the one readable winner under both arms. **LV's readability was an artifact of
short days**, and no other cell's readability moves.

This is a *conditional* result, not a promotion of the 8. Screening trades bias
for variance — `T_CRIT` is keyed on `k`, so `k` 16 → 10 widens LV's interval
before any change in the data. Both `k` values are in the record per country so
the trade is visible rather than assumed. That is also why the screen is
reported beside the primary and is **not** the default.

## Two causes, and only one of them is inside the window

The issue predicted LV `k` 16 → 13 from three truth gaps. The measured fall is
to **10**, because the screen counts *scored* hours rather than truth hours, and
the D-7 comparator is an inner merge:

- LV truth is short on `08-16` (23 h), `08-18` (22) and `08-19` (20) — exactly
  the issue's table. Each one truncates its own day **and the day seven later**,
  `08-23`, `08-25`, `08-26`, at the identical hour counts. Three gaps, six days.
- EE's short day is `08-17`, and EE truth is complete on `08-17`. The cause is a
  17-hour truth day on **`2026-08-10`**, a date entirely outside the scored
  window, reached only through the D-7 arm.

A truth-side screen bounded by the scored window cannot see either effect. That
is the argument for measuring completeness on the panel: `hours_present` counts
what the interval was actually computed on, so a missing vintage, a missing
comparator and a missing hour of truth all reduce it identically.

## The denominator is the window, never 24

The window's end days are legitimately partial: `2026-08-13` can only carry 16
target hours and `2026-08-28` exactly **1**. A "require 24 hours" screen would
drop both ends for all 24 countries, silently — and that terminal hour is the
single largest difference ABL-607 recorded between its two reads. The screen is
a ratio against per-day expected hours, and `hours_expected` is computed from
one fleet-wide span: derived per country it would be defined by the very
truncation it is meant to detect, and every country would score complete by
construction. Both properties are pinned in
`tests/test_abl639_day_completeness.py`.

## Not screened, on purpose

The pooled WAPE tables (`per_country`, `fleet_medians`, sections E/G/H) weight
by hour, so a short day already contributes proportionally there. Only the daily
paired intervals give every day equal weight, so only they are screened.

## For ABL-622

The ~30-day re-read should run the sensitivity arm and read LV from it. The
ABL-635 outage days (`2026-08-29`, `2026-08-30`) sit inside that window and are
far worse than anything in ABL-607's: partial-retention countries there enter at
full weight on 2–5 hours. Whether the re-read *screens* by default is a protocol
choice for the CEO, not one this change makes — the default here stays `0.0` so
the published pack is unchanged.
