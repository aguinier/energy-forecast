# ABL-444 — the G2/G3 readability floor, applied to every graded ABL-316 cell

Generated: 2026-08-14T03:01:52Z. Registration: `experiments/ABL444/config.json`, committed before this read existed.

Readability: **`sign_test` → `floored`**. Floor: `readability_floor_pct` at k=1 — **10.65% solar, 7.51% wind**.

Arithmetic over records already on disk: each tranche's committed `results_*.json` for the challenger's own scores, and `reports/abl_437_causal_levelling_reread.json` for ABL-437's trailing references on the same cells. **No refit, no new model, and no database read** — the trailing columns were computed against the replica by ABL-437 and are copied rather than recomputed, so this document cannot disagree with that one about a number.

**No committed record is edited by this read.** It is a new document, on the ABL-418 / ABL-437 retro-grade precedent.

Both registered axes stay live, so every cell is graded four ways: `{fit_window, trailing_28d} × {sign_test, floored}`. The `sign_test` column of each levelling is the published letter and is reproduced, not restated.

## 1. What the floor moves

`fit_window / sign_test` is what is **published today**. `trailing_28d / sign_test` is ABL-437's amended read.

### 1.1 Levelling `fit_window` — **2 of 41 pair-records move**, 0 of them from `A`

| tranche | pair | before | after | abstains on | hold |
|---|---|:---:|:---:|---|---|
| 2d | NL solar | B | **N** | G3 | — |
| 2e | HU wind_onshore | B | **N** | G2, G3 | — |

### 1.2 Levelling `trailing_28d` — **11 of 41 pair-records move**, 5 of them from `A`

| tranche | pair | before | after | abstains on | hold |
|---|---|:---:|:---:|---|---|
| 1b | BG solar | A | **N** | G3 | yes — see §4 |
| 2a | BG solar | A | **N** | G3 | yes — see §4 |
| 2a | PL solar | B | **N** | G3 | — |
| 2a | SK solar | B | **N** | G3 | — |
| 2d | EE solar | A | **N** | G3 | — |
| 2d | FI solar | A | **N** | G3 | — |
| 2d | LT solar | B | **N** | G3 | — |
| 2d | NL solar | B | **N** | G3 | — |
| 2d | SE solar | B | **N** | G3 | — |
| 2e | HR wind_onshore | A | **N** | G2, G3 | — |
| offshore | DE wind_offshore | B | **N** | G2, G3 | — |

## 2. Every abstaining cell, with its margin in both denominators

The CEO's constraint: the floor decides gradeability, it does not replace the number. `skill` is the registered column, `own` is ABL-385's own-error denominator, reported as the sensitivity. `n ≥ min` is ABL-434's column — reported beside every grade because the ladder cannot see it, and deliberately **not** folded into it.

| levelling | tranche | pair | band | n | n ≥ min | condition | skill % | own % | floor % | letter |
|---|---|---|---|---:|:---:|:---:|---:|---:|---:|:---:|
| fit_window | 2b | ES wind_onshore | 24-36h | 720 | yes | G3 | +1.58 | +1.61 | 7.51 | C |
| fit_window | 2b | ES wind_onshore | 36-48h | 720 | yes | G3 | +1.74 | +1.77 | 7.51 | C |
| fit_window | 2b | ES wind_onshore | 48-64h | 510 | yes | G3 | +1.26 | +1.27 | 7.51 | C |
| fit_window | 2d | NL solar | 24-36h | 720 | yes | G3 | -6.27 | -5.90 | 10.65 | N |
| fit_window | 2d | NL solar | 36-48h | 720 | yes | G3 | -8.27 | -7.64 | 10.65 | N |
| fit_window | 2d | NL solar | 48-64h | 510 | yes | G3 | -5.94 | -5.61 | 10.65 | N |
| fit_window | 2e | HU wind_onshore | 24-36h | 720 | yes | G2 | -1.75 | -1.72 | 7.51 | N |
| fit_window | 2e | HU wind_onshore | 24-36h | 720 | yes | G3 | -2.13 | -2.09 | 7.51 | N |
| fit_window | 2e | HU wind_onshore | 36-48h | 720 | yes | G2 | -2.21 | -2.17 | 7.51 | N |
| fit_window | 2e | HU wind_onshore | 36-48h | 720 | yes | G3 | -2.60 | -2.54 | 7.51 | N |
| fit_window | 2e | HU wind_onshore | 48-64h | 510 | yes | G2 | -4.90 | -4.67 | 7.51 | N |
| fit_window | 2e | HU wind_onshore | 48-64h | 510 | yes | G3 | -4.17 | -4.01 | 7.51 | N |
| trailing_28d | 1b | BG solar | 24-36h | 720 | yes | G3 | +8.86 | +9.72 | 10.65 | N |
| trailing_28d | 1b | BG solar | 36-48h | 720 | yes | G3 | +10.56 | +11.81 | 10.65 | N |
| trailing_28d | 1b | BG solar | 48-64h | 510 | yes | G3 | +9.26 | +10.21 | 10.65 | N |
| trailing_28d | 2a | BG solar | 24-36h | 720 | yes | G3 | +5.29 | +5.58 | 10.65 | N |
| trailing_28d | 2a | BG solar | 36-48h | 720 | yes | G3 | +6.18 | +6.59 | 10.65 | N |
| trailing_28d | 2a | BG solar | 48-64h | 510 | yes | G3 | +5.69 | +6.03 | 10.65 | N |
| trailing_28d | 2a | HU solar | 24-36h | 720 | yes | G3 | -9.69 | -8.83 | 10.65 | U |
| trailing_28d | 2a | HU solar | 36-48h | 720 | yes | G3 | -9.42 | -8.61 | 10.65 | U |
| trailing_28d | 2a | HU solar | 48-64h | 510 | yes | G3 | -3.60 | -3.48 | 10.65 | U |
| trailing_28d | 2a | PL solar | 24-36h | 720 | yes | G3 | -1.13 | -1.12 | 10.65 | N |
| trailing_28d | 2a | PL solar | 36-48h | 720 | yes | G3 | -1.11 | -1.10 | 10.65 | N |
| trailing_28d | 2a | PL solar | 48-64h | 510 | yes | G3 | -0.36 | -0.36 | 10.65 | N |
| trailing_28d | 2a | SK solar | 24-36h | 715 | yes | G3 | -7.70 | -7.15 | 10.65 | N |
| trailing_28d | 2a | SK solar | 36-48h | 715 | yes | G3 | -8.01 | -7.42 | 10.65 | N |
| trailing_28d | 2a | SK solar | 48-64h | 507 | yes | G3 | -2.29 | -2.24 | 10.65 | N |
| trailing_28d | 2c | IT solar | 24-36h | 720 | yes | G3 | -9.02 | -8.27 | 10.65 | U |
| trailing_28d | 2c | IT solar | 36-48h | 720 | yes | G3 | -10.19 | -9.25 | 10.65 | U |
| trailing_28d | 2c | IT solar | 48-64h | 510 | yes | G3 | -6.01 | -5.67 | 10.65 | U |
| trailing_28d | 2c | PT solar | 24-36h | 720 | yes | G3 | +5.93 | +6.30 | 10.65 | C |
| trailing_28d | 2c | PT solar | 36-48h | 720 | yes | G3 | +3.62 | +3.75 | 10.65 | C |
| trailing_28d | 2c | PT solar | 48-64h | 510 | yes | G3 | +0.05 | +0.05 | 10.65 | C |
| trailing_28d | 2d | EE solar | 48-64h | 388 | **no** | G3 | +0.35 | +0.35 | 10.65 | N |
| trailing_28d | 2d | FI solar | 48-64h | 453 | **no** | G3 | +0.59 | +0.60 | 10.65 | N |
| trailing_28d | 2d | LT solar | 24-36h | 720 | yes | G3 | -6.87 | -6.43 | 10.65 | N |
| trailing_28d | 2d | LT solar | 36-48h | 720 | yes | G3 | -6.12 | -5.77 | 10.65 | N |
| trailing_28d | 2d | LT solar | 48-64h | 510 | yes | G3 | -3.65 | -3.52 | 10.65 | N |
| trailing_28d | 2d | NL solar | 24-36h | 720 | yes | G3 | -3.86 | -3.72 | 10.65 | N |
| trailing_28d | 2d | NL solar | 36-48h | 720 | yes | G3 | -5.64 | -5.34 | 10.65 | N |
| trailing_28d | 2d | NL solar | 48-64h | 510 | yes | G3 | -1.95 | -1.92 | 10.65 | N |
| trailing_28d | 2d | SE solar | 24-36h | 720 | yes | G3 | -8.65 | -7.96 | 10.65 | N |
| trailing_28d | 2d | SE solar | 36-48h | 720 | yes | G3 | -6.51 | -6.11 | 10.65 | N |
| trailing_28d | 2d | SE solar | 48-64h | 510 | yes | G3 | -2.97 | -2.89 | 10.65 | N |
| trailing_28d | 2e | HR wind_onshore | 24-36h | 720 | yes | G2 | +2.80 | +2.88 | 7.51 | N |
| trailing_28d | 2e | HR wind_onshore | 24-36h | 720 | yes | G3 | +2.02 | +2.06 | 7.51 | N |
| trailing_28d | 2e | RO wind_onshore | 24-36h | 720 | yes | G2 | -6.77 | -6.34 | 7.51 | B |
| trailing_28d | 2f | CH wind_onshore | 36-48h | 720 | yes | G2 | -2.39 | -2.33 | 7.51 | N |
| trailing_28d | 2f | CH wind_onshore | 36-48h | 720 | yes | G3 | -6.92 | -6.48 | 7.51 | N |
| trailing_28d | 2f | CH wind_onshore | 48-64h | 510 | yes | G2 | -0.52 | -0.52 | 7.51 | N |
| trailing_28d | 2f | CH wind_onshore | 48-64h | 510 | yes | G3 | -3.84 | -3.70 | 7.51 | N |
| trailing_28d | offshore | DE wind_offshore | 24-36h | 720 | yes | G2 | +0.33 | +0.33 | 7.51 | N |
| trailing_28d | offshore | DE wind_offshore | 24-36h | 720 | yes | G3 | +0.52 | +0.52 | 7.51 | N |
| trailing_28d | offshore | DE wind_offshore | 36-48h | 720 | yes | G2 | +1.32 | +1.34 | 7.51 | N |
| trailing_28d | offshore | DE wind_offshore | 36-48h | 720 | yes | G3 | +1.32 | +1.34 | 7.51 | N |
| trailing_28d | offshore | DE wind_offshore | 48-64h | 510 | yes | G2 | +1.03 | +1.04 | 7.51 | N |
| trailing_28d | offshore | DE wind_offshore | 48-64h | 510 | yes | G3 | -0.47 | -0.47 | 7.51 | N |

## 3. Every cell, all four arms

| tranche | pair | band | n | fit/sign | fit/floor | 28d/sign | 28d/floor |
|---|---|---|---:|:---:|:---:|:---:|:---:|
| 1b | BG solar | 24-36h | 720 | A | A | A | **N** |
| 1b | BG solar | 36-48h | 720 | A | A | A | **N** |
| 1b | BG solar | 48-64h | 510 | A | A | A | **N** |
| 1b | CH solar | 24-36h | 720 | A | A | A | **A** |
| 1b | CH solar | 36-48h | 720 | A | A | A | **A** |
| 1b | CH solar | 48-64h | 510 | A | A | A | **A** |
| 2a | BG solar | 24-36h | 720 | A | A | A | **N** |
| 2a | BG solar | 36-48h | 720 | A | A | A | **N** |
| 2a | BG solar | 48-64h | 510 | A | A | A | **N** |
| 2a | CH solar | 24-36h | 720 | A | A | A | **A** |
| 2a | CH solar | 36-48h | 720 | A | A | A | **A** |
| 2a | CH solar | 48-64h | 510 | A | A | A | **A** |
| 2a | CZ solar | 24-36h | 720 | A | A | A | **A** |
| 2a | CZ solar | 36-48h | 720 | A | A | A | **A** |
| 2a | CZ solar | 48-64h | 510 | A | A | A | **A** |
| 2a | HU solar | 24-36h | 720 | U(+) | U(+) | U | **U** |
| 2a | HU solar | 36-48h | 720 | U(+) | U(+) | U | **U** |
| 2a | HU solar | 48-64h | 510 | U(+) | U(+) | U | **U** |
| 2a | PL solar | 24-36h | 720 | A | A | B | **N** |
| 2a | PL solar | 36-48h | 720 | A | A | B | **N** |
| 2a | PL solar | 48-64h | 510 | A | A | B | **N** |
| 2a | RO solar | 24-36h | 720 | A | A | A | **A** |
| 2a | RO solar | 36-48h | 720 | A | A | A | **A** |
| 2a | RO solar | 48-64h | 510 | A | A | A | **A** |
| 2a | SI solar | 24-36h | 720 | A | A | B | **B** |
| 2a | SI solar | 36-48h | 720 | A | A | B | **B** |
| 2a | SI solar | 48-64h | 510 | A | A | B | **B** |
| 2a | SK solar | 24-36h | 715 | A | A | B | **N** |
| 2a | SK solar | 36-48h | 715 | A | A | B | **N** |
| 2a | SK solar | 48-64h | 507 | A | A | B | **N** |
| 2b | ES wind_onshore | 24-36h | 720 | C | C | C | **C** |
| 2b | ES wind_onshore | 36-48h | 720 | C | C | C | **C** |
| 2b | ES wind_onshore | 48-64h | 510 | C | C | C | **C** |
| 2b | FI wind_onshore | 24-36h | 711 | A | A | A | **A** |
| 2b | FI wind_onshore | 36-48h | 711 | A | A | A | **A** |
| 2b | FI wind_onshore | 48-64h | 504 | A | A | A | **A** |
| 2b | GR wind_onshore | 24-36h | 720 | A | A | A | **A** |
| 2b | GR wind_onshore | 36-48h | 720 | A | A | A | **A** |
| 2b | GR wind_onshore | 48-64h | 510 | A | A | A | **A** |
| 2b | IT wind_onshore | 24-36h | 716 | U(+) | U(+) | U | **U** |
| 2b | IT wind_onshore | 36-48h | 715 | U(+) | U(+) | U | **U** |
| 2b | IT wind_onshore | 48-64h | 505 | U(+) | U(+) | U | **U** |
| 2b | NO wind_onshore | 24-36h | 720 | B | B | B | **B** |
| 2b | NO wind_onshore | 36-48h | 720 | B | B | B | **B** |
| 2b | NO wind_onshore | 48-64h | 510 | B | B | B | **B** |
| 2b | PL wind_onshore | 24-36h | 720 | A | A | A | **A** |
| 2b | PL wind_onshore | 36-48h | 720 | A | A | A | **A** |
| 2b | PL wind_onshore | 48-64h | 510 | A | A | A | **A** |
| 2b | PT wind_onshore | 24-36h | 720 | C | C | C | **C** |
| 2b | PT wind_onshore | 36-48h | 720 | C | C | C | **C** |
| 2b | PT wind_onshore | 48-64h | 510 | C | C | C | **C** |
| 2b | SE wind_onshore | 24-36h | 720 | A | A | A | **A** |
| 2b | SE wind_onshore | 36-48h | 720 | A | A | A | **A** |
| 2b | SE wind_onshore | 48-64h | 510 | A | A | A | **A** |
| 2c | ES solar | 24-36h | 720 | U(+) | U(+) | U | **U** |
| 2c | ES solar | 36-48h | 720 | U(+) | U(+) | U | **U** |
| 2c | ES solar | 48-64h | 510 | U(+) | U(+) | U | **U** |
| 2c | GR solar | 24-36h | 720 | C | C | C | **C** |
| 2c | GR solar | 36-48h | 720 | C | C | C | **C** |
| 2c | GR solar | 48-64h | 510 | C | C | C | **C** |
| 2c | HR solar | 24-36h | 720 | U(+) | U(+) | U | **U** |
| 2c | HR solar | 36-48h | 720 | U(+) | U(+) | U | **U** |
| 2c | HR solar | 48-64h | 510 | U(+) | U(+) | U | **U** |
| 2c | IT solar | 24-36h | 720 | U(+) | U(+) | U | **U** |
| 2c | IT solar | 36-48h | 720 | U(+) | U(+) | U | **U** |
| 2c | IT solar | 48-64h | 510 | U(+) | U(+) | U | **U** |
| 2c | PT solar | 24-36h | 720 | C | C | C | **C** |
| 2c | PT solar | 36-48h | 720 | C | C | C | **C** |
| 2c | PT solar | 48-64h | 510 | C | C | C | **C** |
| 2d | EE solar | 48-64h | 388 | A | A | A | **N** |
| 2d | FI solar | 48-64h | 453 | A | A | A | **N** |
| 2d | LT solar | 24-36h | 720 | A | A | B | **N** |
| 2d | LT solar | 36-48h | 720 | A | A | B | **N** |
| 2d | LT solar | 48-64h | 510 | A | A | B | **N** |
| 2d | LV solar | 24-36h | 708 | A | A | A | **A** |
| 2d | LV solar | 36-48h | 708 | A | A | A | **A** |
| 2d | LV solar | 48-64h | 506 | A | A | A | **A** |
| 2d | NL solar | 24-36h | 720 | B | N | B | **N** |
| 2d | NL solar | 36-48h | 720 | B | N | B | **N** |
| 2d | NL solar | 48-64h | 510 | B | N | B | **N** |
| 2d | SE solar | 24-36h | 720 | A | A | B | **N** |
| 2d | SE solar | 36-48h | 720 | A | A | B | **N** |
| 2d | SE solar | 48-64h | 510 | A | A | B | **N** |
| 2e | CZ wind_onshore | 24-36h | 720 | A | A | A | **A** |
| 2e | CZ wind_onshore | 36-48h | 720 | A | A | A | **A** |
| 2e | CZ wind_onshore | 48-64h | 510 | A | A | A | **A** |
| 2e | EE wind_onshore | 24-36h | 685 | A | A | A | **A** |
| 2e | EE wind_onshore | 36-48h | 684 | A | A | A | **A** |
| 2e | EE wind_onshore | 48-64h | 475 | A | A | A | **A** |
| 2e | HR wind_onshore | 24-36h | 720 | A | A | A | **N** |
| 2e | HR wind_onshore | 36-48h | 720 | A | A | A | **A** |
| 2e | HR wind_onshore | 48-64h | 510 | A | A | A | **A** |
| 2e | HU wind_onshore | 24-36h | 720 | B | N | B | **B** |
| 2e | HU wind_onshore | 36-48h | 720 | B | N | B | **B** |
| 2e | HU wind_onshore | 48-64h | 510 | B | N | B | **B** |
| 2e | LT wind_onshore | 24-36h | 720 | A | A | A | **A** |
| 2e | LT wind_onshore | 36-48h | 720 | A | A | A | **A** |
| 2e | LT wind_onshore | 48-64h | 510 | A | A | A | **A** |
| 2e | LV wind_onshore | 24-36h | 708 | B | B | B | **B** |
| 2e | LV wind_onshore | 36-48h | 708 | U | U | U | **U** |
| 2e | LV wind_onshore | 48-64h | 506 | U | U | U | **U** |
| 2e | NL wind_onshore | 24-36h | 720 | A | A | A | **A** |
| 2e | NL wind_onshore | 36-48h | 720 | A | A | A | **A** |
| 2e | NL wind_onshore | 48-64h | 510 | A | A | A | **A** |
| 2e | RO wind_onshore | 24-36h | 720 | B | B | B | **B** |
| 2e | RO wind_onshore | 36-48h | 720 | B | B | B | **B** |
| 2e | RO wind_onshore | 48-64h | 510 | B | B | B | **B** |
| 2f | BG wind_onshore | 24-36h | 720 | A | A | A | **A** |
| 2f | BG wind_onshore | 36-48h | 720 | A | A | A | **A** |
| 2f | BG wind_onshore | 48-64h | 510 | A | A | A | **A** |
| 2f | CH wind_onshore | 24-36h | 720 | A | A | B | **B** |
| 2f | CH wind_onshore | 36-48h | 720 | A | A | B | **N** |
| 2f | CH wind_onshore | 48-64h | 510 | A | A | B | **N** |
| offshore | DE wind_offshore | 24-36h | 720 | A | A | A | **N** |
| offshore | DE wind_offshore | 36-48h | 720 | A | A | A | **N** |
| offshore | DE wind_offshore | 48-64h | 510 | A | A | B | **N** |
| offshore | NL wind_offshore | 24-36h | 720 | A | A | A | **A** |
| offshore | NL wind_offshore | 36-48h | 720 | A | A | A | **A** |
| offshore | NL wind_offshore | 48-64h | 510 | A | A | A | **A** |

## 4. Holds that travel with these letters

- **1b BG solar** — ABL-396 night contamination: BG books 152-246 MW in 76-85% of its night hours, and 25.3% of its scored gate rows are night rows. The displacement band is far wider than any margin in this document.
- **2a BG solar** — ABL-396 night contamination: BG books 152-246 MW in 76-85% of its night hours, and 25.3% of its scored gate rows are night rows. The displacement band is far wider than any margin in this document.

## 5. What this does not say

- **It changes gradeability, not skill.** No model is better or worse for this read; some verdicts become honest abstentions.
- **It cannot raise a grade.** `N` is only ever reached from what would have been `A` or `B`. Note that `N` ranks *better* than `B` on the ladder — an abstention is a weaker negative than a named failure — so a `B → N` move lowers the severity while leaving the pair exactly as non-promotable.
- **It touches no part of ABL-348's registration** — windows, bands, metric, baseline, minimum n, source, `not_evaluable` — so `voids_this_registration` is not triggered.
- **Tranche 1a is absent**, for ABL-437's reason and not a new one: it was fitted before ABL-389 existed and carries no causal reference columns, so G2 and G3 read *not measured* there under every arm.
- **This promotes nothing.** Promotion remains a pre-registered gate read plus a Board decision.

Source records, SHA-256: `reports/abl_437_causal_levelling_reread.json` `bc34431a04f74cee…`; per tranche —

- `experiments/ABL348/results_abl381_tranche1b.json` `6ff1629cc4525683…`
- `experiments/ABL348/results_abl405_tranche2a.json` `895e1259c0da3921…`
- `experiments/ABL348/results_abl406_tranche2b.json` `972eea5fe8880668…`
- `experiments/ABL348/results_abl419_tranche2c.json` `fe25b86c98304059…`
- `experiments/ABL348/results_abl421_tranche2d.json` `ebbc4c448dbd5614…`
- `experiments/ABL348/results_abl417_tranche2e.json` `1225905d091b4417…`
- `experiments/ABL348/results_abl435_tranche2f.json` `70c6669b17cf74a4…`
- `reports/abl_443_offshore_trailing_reread.json` `9df8df76562607c3…`
