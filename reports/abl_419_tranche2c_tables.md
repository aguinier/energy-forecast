# ABL-419 — tranche 2c generated tables

Generated from `experiments/ABL348/results_abl419_tranche2c.json`, SHA-256 `fe25b86c983040591cff48e6a84fdca255e51d2f070c658915ea4aa9d73044bf`, and from ABL-396's committed night-floor screen. No refit, no replica read, no recomputed metric; the grades are read back through `src/evaluation/gate_grading.py`, not re-derived. Regenerate with `.venv\Scripts\python.exe scripts/abl419_tranche2c_read.py`.

Scope `abl316-t2c`, source **`energy_generation`**, **27** features, fit rules `{'exclude_impossible_night': False}`.

## 1. Night floor, all five countries — the zeros stated, not omitted

`f` is ABL-396 section 2's `wape_floor_pct_if_clamped`: the share of the window's total |energy| booked at night, which is the **full width in WAPE points** of the interval an all-hours read can occupy relative to the daylight-only read of the same challenger. Source `energy_generation`, the table this tranche fits and scores on.

| country | window | night hrs | hrs > 1 MW | night mean | **f** |
|---|---|---:|---:|---:|---:|
| ES | fit | 1,573 | 1,562 | 262.44 MW | **1.3380%** |
| ES | gate | 211 | 211 | 515.55 MW | **1.3520%** |
| GR | fit | 1,576 | 15 | 0.07 MW | 0.0020% |
| GR | gate | 210 | 0 | 0.00 MW | 0.0000% |
| HR | fit | 1,513 | 5 | 0.01 MW | 0.0030% |
| HR | gate | 184 | 0 | 0.00 MW | 0.0000% |
| IT | fit | 1,524 | 2 | 0.01 MW | 0.0000% |
| IT | gate | 206 | 2 | 0.05 MW | 0.0000% |
| PT | fit | 1,580 | 1,444 | 9.35 MW | **0.4490%** |
| PT | gate | 211 | 7 | 0.38 MW | 0.0090% |

## 2. ES's night-floor band, on the face of the table

ES gate-window `f` = **1.3520%**. Every arm of the gate is scored on the same all-hours rows, so the verdict below is not adjusted by this band — the band is what **bounds** it: what a daylight-only read of this same challenger would have been. Exact, free, and it closes the only cell ABL-403's 2×2 could have moved on this tranche, which is why ABL-419 discharges that soft hold rather than waiting on it.

| band | n | all-hours challenger WAPE (measured) | implied daylight-only WAPE | if clamped to 0 at night | D-7 bar (same rows) | registered verdict | clamped-variant verdict |
|---|---:|---:|---:|---:|---:|:---:|:---:|
| 24-36h | 720 | 11.39% | 10.18%–11.55% | 11.39%–12.74% | 11.69% | PASS | **indeterminate** |
| 36-48h | 720 | 11.41% | 10.20%–11.57% | 11.41%–12.77% | 11.69% | PASS | **indeterminate** |
| 48-64h | 510 | 11.03% | 9.81%–11.18% | 11.03%–12.38% | 11.15% | PASS | **indeterminate** |

**Read the last two columns as answering different questions.** The *registered* verdict is a direct measurement: challenger and D-7 are scored on the identical all-hours rows, so ES's night floor cannot have moved it in either direction, and the band does not qualify it. The *clamped-variant* column is serving-side and is reported because `f` makes it free: the ABL-337 clamp forces a zero on this same night predicate, so a served version of this challenger would score somewhere in `[A, A+f]`. On all three ES bands that interval **straddles the D-7 bar**, so the bound cannot say whether a clamped ES would clear it. That is a finding to hand to whoever owns serving, not a qualification of the read above — and settling it needs an actual daylight-only read, which this bound deliberately does not substitute for.

**ES is capped at grade B regardless of G1–G4**, with `ABL-411 hold` named as the failed condition. The cap is a hold, not a measurement: the ABL-337 clamp question is serving-side and this read changes no serving path, so nothing above depends on it — but ES may not reach a promotion recommendation before ABL-411 settles. The cap only ever moves a grade down.

## 3. Pair grade against the pre-committed bar

The bar column is ABL-348's, measured before any challenger for these pairs existed. It is here because ABL-406 established across eight wind pairs that the gate outcome was *fully* predicted by whether a causal constant clears the bar on its own — a pass against a weak bar and a pass against a strong one are not the same evidence.

| pair | pre-committed D-7 bar | band grades | ladder pair grade | **reported** | failed conditions | bar weaker than a flat line? |
|---|---:|---|:---:|:---:|---|:---:|
| ES | 11.78% | U(+) / U(+) / U(+) | U(+) | **B** | ABL-411 hold | no |
| GR | 10.37% | C / C / C | C | **C** | G1 | no |
| HR | 16.43% | U(+) / U(+) / U(+) | U(+) | **U(+)** | — | no |
| IT | 7.11% | U(+) / U(+) / U(+) | U(+) | **U(+)** | — | no |
| PT | 13.09% | C / C / C | C | **C** | G1 | no |

**Do not average this tranche's pass rate against 2a's.** 2a's bars ran 18.35–26.11% plus CH at 12.67%; these run 7.11–16.43%. ABL-348 registered that reading in advance under `reading_caveats_not_band_changes`: same band, materially harder task, and a lower pass rate here is not model quality.

