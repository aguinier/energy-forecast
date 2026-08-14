# ABL-443 — Pre-registration: the trailing-reference read of DE/NL `wind_offshore`

**Status: REGISTRATION. Frozen before the trailing references were computed for
either pair.** Machine record: `experiments/ABL443/config.json`. The read itself
is a **separate document and a separate commit** —
`reports/abl_443_offshore_trailing_reread.md` — and that ordering is checkable in
git.

Scope id: **`abl443-offshore-trailing`**. It is a new scope, not an edit to
`abl322-pilot`.

---

## 1. What this read is for

ABL-436 read DE and NL `wind_offshore` onto the registered harness and graded both
**A** (PR #68, merged). That read stands. But its G2 and G3 conditions are scored
against `constant_causal` and `climatology_causal`, which ABL-437 identifies as
levelled on the **fit** window (2026-01-14 → 2026-07-11) and scored on the **gate**
window (2026-07-11 → 2026-08-10) — different seasons. Re-derived from ABL-436's own
committed record:

| pair | band | `constant_causal` | `constant_oracle` | inflation |
|---|---|---:|---:|---:|
| DE `wind_offshore` | 24-36h | 74.40% | 62.86% | **18.36%** |
| DE `wind_offshore` | 36-48h | 74.40% | 62.86% | **18.36%** |
| DE `wind_offshore` | 48-64h | 75.79% | 62.32% | **21.62%** |
| NL `wind_offshore` | 24-36h | 89.97% | 71.74% | **25.40%** |
| NL `wind_offshore` | 36-48h | 89.97% | 71.74% | **25.40%** |
| NL `wind_offshore` | 48-64h | 92.99% | 73.27% | **26.91%** |

Both pairs sit inside the 18–27% the issue states, and it reproduces to the digit.
The levels behind it are DE 3263.82 MW causal against 2005.99 MW oracle, and NL
1704.09 MW against 870.25 MW — the causal constant is **1.63×** and **1.96×** the
correctly-levelled one.

So neither pair's ABL-436 letter can currently be read as a quality statement about
the *level*: G2 and G3 were asked against a reference that is not an estimate of the
gate window's level. **That is a statement about the reference, not about either
challenger.** Every challenger WAPE, every D-7 comparison and both PASS verdicts in
ABL-436 stand exactly as recorded, under either levelling.

## 2. What is inherited and not re-decided here

The levelling is **ABL-437's**, registered at `experiments/ABL437/config.json` and
frozen there before it graded any pair. This issue does not amend it, tune it, or
re-argue it. Inherited verbatim:

| | |
|---|---|
| **references** | `constant_causal_28d`, `climatology_causal_28d` — the same flat line and the same hour-of-day mean over the 28 days ending at the row's own `generated_at`, anchored at `generated_at.floor("h")`, inclusive, spanning `28*24 - 1` hours back. |
| **series** | The same ABL-188-filtered target series the gate actuals and the D-7 baseline are read from. Same bound `wind_features._rolling_features` already applies to `target_value_roll_168h_mean`, one of the challenger's own 24 features — so the reference uses no information the challenger did not have. |
| **ladder** | ABL-418's G1–G4, unchanged. G1 seasonal-naive D-7 against the ABL-385 readability floor (**7.51%** wind at k=1); G2 `constant_causal_28d`; G3 `climatology_causal_28d`; G4 slope > 0 **and** corr > 0. |
| **oracles** | Reported, on neither ladder, under either levelling. Used here for exactly one thing: as the denominator of the inflation diagnostic. |
| **bar** | Seasonal-naive D-7 is G1 under both levellings. Re-levelling the registered bar would be re-opening it, and is not done. |

## 3. Why a separate scope id, and what must not be touched

Per the **ABL-401 ruling** and the ABL-418 retro-grade precedent. `abl322-pilot` is
a published scope and is pinned to `fit_window` in `CAUSAL_LEVELLING`
(`test_every_published_scope_pins_its_levelling` derives that set from
`SCOPE_OUTPUTS` and git, and requires the pin). Re-reading it in place would make a
committed page of letters disagree with the run that regenerates it. A second
reference read is a **second scope**, not an edit.

This read writes to `reports/abl_443_offshore_trailing_reread.{json,md}` and
`reports/abl_443_findings.md`, and to nothing else. It must not write to:

- `experiments/ABL322/results_abl436_offshore_reread.json` (SHA-256 `ef0f6474…`,
  blob `a9d0e814cd8a`) or `experiments/ABL322/abl436_preregistration_recheck.json`
- `reports/abl_436_offshore_reference_grade.md`,
  `reports/abl_436_offshore_evidence_pack.md`, `reports/abl_322_pilot_gate.md`
- `reports/abl_437_causal_levelling_reread.{json,md}`

**No refit.** No model is trained, loaded or scored again. The challenger, D-7,
persistence, both fit-window references and both oracle references are read out of
ABL-436's record as committed; only the two trailing references are computed, and
they are computed on the same rows.

## 4. The row set is proved, not assumed

Inherited from ABL-437's re-read, and it is the part that can silently go wrong.
Each cell's scored rows are rebuilt from ABL-348's eight registered run instants —
the harness's own `schedule_vintages` and `horizon_band`, latest vintage per
(target, band) — and then **checked by recomputing that cell's published
`constant_causal` and `climatology_causal` WAPE *and* MAE from it**, to `1e-09`. A
constant and a 24-bucket climatology agreeing on two statistics each is the row set;
one agreeing alone would not be.

A cell that does not reproduce to that tolerance is reported **NOT
RECONSTRUCTIBLE** and is graded by nobody. It is not quietly dropped and it is not
guessed at.

Where the schedule alone does not reproduce a cell, rows were dropped on a NaN
feature and only the feature vector knows which — `finite_training_rows` runs
*before* `select_latest_challenger_per_band`, so a dropped vintage does not merely
shrink n, it can promote the next-latest vintage into the band and **move that
row's `generated_at`**. Under a reference levelled on `generated_at` that is not a
detail, so such a pair is rebuilt through `RenewableFeatureBuilder` rather than
estimated. ABL-436's record carries no `meta.feature_columns`, which under ABL-404's
rule dates the read; wind's 24-column list has not moved, so the wind fallback is
the list that record was fitted on.

## 5. The G2/G3 readability floor: it did not land, and this read does not land it

The issue asks that if ABL-437's G2/G3 readability floor lands first, it be applied
here. **It did not.** ABL-437's re-read (§2) declined to widen G2 and G3 from a sign
test (`skill > 0`) to a floor test, on the grounds that it would be a second
registration change stacked on the levelling one, and printed the margin instead.

This read makes the same choice, and registers the consequence in advance:

- Every G2 and G3 margin is printed **with its sign and its magnitude**, in every
  case — including where no letter moves, and including where the verdict is *not
  readable*.
- A margin whose magnitude sits inside the **7.51%** k=1 wind floor is labelled
  `not readable at one seed`. That label is a **diagnostic on the margin, not a
  ladder condition**, and it does not move a letter.
- Where a letter turns on a sub-floor margin, **both** statements are made: the
  letter as the ladder computes it, and the flag that it is not demonstrated at one
  seed. Neither is suppressed in favour of the other.

Adding a G2/G3 floor condition *after* seeing which letters it would move would be
the exact defect this apparatus exists to prevent, and is listed under §7 as voiding
this registration.

## 6. What is registered as unknown

Committed before the trailing column exists for either pair, so that the read can
contradict it:

- **DE already loses to `constant_oracle` in all three bands** (66.11 / 65.66 /
  66.15% against 62.86 / 62.86 / 62.32%) and to `climatology_oracle` in all three.
  The trailing reference sits **between** the fit-window one and the oracle by
  construction, so DE's G2 and G3 margins should tighten and may cross zero. Whether
  they do is what is being measured.
- **NL beats both oracle references in all three bands** by roughly twice the floor
  (ABL-436 §5). Its G2/G3 margins should tighten and are less likely to cross.
- **A trailing window converges; it does not teleport.** On ABL-348's windows it
  *starts* as the last 28 days of the fit window — already the gate season — so the
  residual should be small, but it is **reported per cell** as a `level inflation`
  column rather than asserted away. The corrected reference must not be quoted as
  exact.
- These are predictions. **What the read measures is what gets reported, including
  where it contradicts this section.**

## 7. What would void this registration

- Any change to the trailing window length, its anchor, the two column names, or the
  set of references the ladder reads, after this read has been computed.
- Any edit to `experiments/ABL322/results_abl436_offshore_reread.json`.
- Regrading on a row set that did not reproduce its published references to `1e-09`.
- Adding a G2/G3 floor condition to the ladder after seeing which letters it would
  move.

## 8. Honesty note on what was already visible

ABL-436's fit-window columns, the inflation figures in §1 and DE's losses to both
oracles were **public before this registration was written** — the ABL-443 issue
description quotes them, and §1 above re-derives them from the committed record. What
had **not** been computed for either pair when this file was committed is the
`*_28d` column itself, or any letter derived from it. This registration is frozen
against that column, and the two commits are separate.

## 9. Dependency, stated plainly

ABL-437's PR **#70 has not merged** at the time of this registration; this branch is
stacked on it and carries its commits. If #70's levelling definition changes in
review — the window length, the anchor, the column names, or which references the
ladder reads — this read is void under §7 and must be re-run, not patched. That is
the risk of reading against an open registration and it is named here rather than
discovered later.

## 10. Contamination

- **ABL-67** is net-position-only. **ABL-109 / ABL-111** are load-only. Neither
  intersects `wind_offshore`.
- **ABL-71**'s known wrong-write modes are load and net position, not wind. That is a
  provenance caveat, not proof that wind ingest is pristine.
- This read touches no ingest, no source table and no window, so it can neither
  introduce nor repair any of them.
- The `energy_generation` revision-vintage seam ABL-439 measured on NL `wind_onshore`
  is a live question for a `wind_offshore` pair read off the same table, and is
  restated per pair in the findings rather than assumed absent.

## 11. Promotion

This read recommends no promotion and touches no serving registry. It restates two
grades against a corrected reference. Promotion remains a pre-registered gate read
plus a Board decision.
