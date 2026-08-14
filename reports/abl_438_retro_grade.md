# ABL-438 — graded gate disposition (G1–G4), and the retro-grade of tranche 1b

**Generated from the stored results files, not restated in prose.** Every grade below is produced by `src/evaluation/gate_grading.py` — the same code both gate harnesses now call — reading `experiments/ABL348/results_abl381_tranche1b.json`. No refit, no new fit, no replica read, no write to any dispositioned path. Regenerate with `.venv\Scripts\python.exe scripts/abl418_retro_grade.py --tranches 1b --issue ABL-438`.

## What is registered, and what is not

**The bar is not re-opened.** Seasonal-naive D-7 stays the registered gate for every scope already dispositioned and every scope still to come. ABL-348's frozen windows, bands, metric, minimum n and source are unchanged. A cell that clears D-7 still reads PASS, and no verdict in any published report moves because of this document.

**What changes is what a PASS entitles a cell to.** ABL-406 established across eight `wind_onshore` pairs that the gate outcome was *fully* predicted by whether a causal constant clears the registered bar on its own — five weak bars gave five passes, three strong bars gave three failures or ties, no exceptions — and that NO passed 3/3 while anti-correlated with its own target (slope −0.08, corr −0.14). A PASS is necessary and not sufficient for a promotion recommendation. Tightening the bar after the fact would be shopping the registration; grading the pass is not, which is why the ladder was pre-registered on ABL-418 before any remaining tranche is fitted.

| condition | test | source column |
|---|---|---|
| **G1** gate | beats seasonal_naive D-7 by more than the readability floor | `skill vs D-7`, against the floor below |
| **G2** level | beats constant_causal -- a flat line at the fit-window mean | `constant causal WAPE`, already printed |
| **G3** shape | beats climatology_causal -- an hour-of-day mean over the fit window | `climatology causal WAPE`, already printed |
| **G4** direction | slope > 0 and correlation > 0 | `slope` and `corr`, already printed |

**A** — G1–G4 hold in every band. Promotion-eligible, subject to any named data hold. **B** — G1 holds, one or more of G2/G3/G4 fails; the failures are named. Not promotion-eligible. **C** — G1 fails readably. **U** — the G1 margin sits inside the readability floor, so the cell is unreadable at one seed; **U(+)** where G2–G4 clear readably, in which case the disposition is *re-read at k>1 seeds* per ABL-385, not *reject*.

`U` takes precedence over `C`: both are "G1 does not hold", but a measured loss and an absence of measurement are different statements, and calling an unreadable cell a failure invites the wrong next move. A pair takes the worst grade of its bands (`C` > `B` > `U` > `A`), because grade A requires all four conditions in *every* band; `U(+)` survives to the pair only if every unreadable band in it is `U(+)`.

**Causal references only.** The two oracle references stay reported and never gate — an oracle is not causally available, so losing to one bounds what a verdict means rather than voiding it. Both are reported below beside every grade, as is the bar-weakness flag (does `constant_causal` clear the registered D-7 bar on its own?).

## The readability floor

ABL-385 registers `delta_min(k) = 1.96 * sqrt(c_A^2 + c_B^2) / sqrt(k)` as the minimum readable relative gap. Every reference on this ladder is **deterministic** — D-7, a flat line and an hour-of-day climatology do not move when the challenger is refitted — so `c_B = 0`, and the published two-arm margin is a factor of √2 too wide. This tranche fits once per cell, so k = 1.

| stream | fleet p90 per-fit CV (ABL-385 §1) | two-arm δ_min at k=1 | **floor used** = δ_min/√2 | published in prose |
|---|---:|---:|---:|---:|
| solar | 5.4328% | 15.0589% | **10.6482%** | 10.64% |

The prose values are 2-dp renderings and are not what the ladder uses; the exact `1.96 · c` value is. The gap between them is under 0.01pp and no cell of any tranche sits inside it — checked per cell in §2.

## 1. Tranche 1b — `abl316-t1b` (solar)

Source: `experiments/ABL348/results_abl381_tranche1b.json`, SHA-256 `6ff1629cc4525683de630c72ec04dac1658b045da6cf0847f6d9c6f8f3e6184a`. Evidence pack: `reports/abl_381_tranche1b_findings.md`. Published disposition, restated unchanged: **PASS**. Gate window 2026-07-11 00:00:00 → 2026-08-10 00:00:00 (exclusive), target series `energy_generation`. Floor 10.6482% at k=1.

| pair | band | n | n ≥ min | gate | skill vs D-7 | vs constant causal | vs climatology causal | slope>0 & corr>0 | grade |
|---|---|---:|:---:|:---:|---:|---:|---:|:---:|:---:|
| BG | 24-36h | 720 | yes | PASS | +22.60% | +74.92% | +55.01% | yes | **A** |
| BG | 36-48h | 720 | yes | PASS | +23.77% | +75.30% | +55.70% | yes | **A** |
| BG | 48-64h | 510 | yes | PASS | +19.86% | +70.62% | +51.54% | yes | **A** |
| CH | 24-36h | 720 | yes | PASS | +35.59% | +91.42% | +78.25% | yes | **A** |
| CH | 36-48h | 720 | yes | PASS | +36.81% | +91.58% | +78.66% | yes | **A** |
| CH | 48-64h | 510 | yes | PASS | +32.98% | +90.24% | +77.04% | yes | **A** |

All 6 cells clear ABL-348's minimum n; the tightest is BG 24-36h at n = 720 against a minimum of 684 (×1.05). The ladder does not read this column — it grades a margin — so it is reported beside the grades, not folded into them.

| pair | bands | grade | failed conditions | bar weaker than a flat line? | beats constant oracle? | beats climatology oracle? | ABL-438 description | agrees? |
|---|---|:---:|---|:---:|:---:|:---:|:---:|:---:|
| BG | A / A / A | **A** | — | no | yes | yes, inside the floor (+1.41%) | A | yes |
| CH | A / A / A | **A** | — | no | yes | yes, inside the floor (+3.47%) | A | yes |

Pair grades: **A** × 2.

**Beating an oracle inside the floor is not beating it readably.** The floor is 10.6482% at k=1 and it applies to any margin a reader ranks on, not only to the one G1 gates on. These wins are positive in every band and none of them is readable at one seed:

- **BG** vs `climatology_oracle`: +1.41% at its worst band, against a 10.65% floor.
- **CH** vs `climatology_oracle`: +3.47% at its worst band, against a 10.65% floor.

**Live data hold on BG — ABL-396 (night contamination, upstream).** 76-85% of geometric-night hours carry above 1 MW, up to 1,097 MW, ~5-6% of all energy booked after dark. Identical on both actuals tables, so it is upstream of this module. ABL-396's screen found BG an outlier by 3.7x and the only country whose displacement band is wide enough to threaten a verdict -- and that band is far wider than BG's margin over the oracle climatology. Grade A must not be reported for BG solar without this line.

Every pair reproduces the reading in the ABL-438 description.

## 2. Sensitivity: which denominator, and the 2-dp rounding

ABL-418 registers G1 on the printed `skill vs D-7` column, `100 · (1 − challenger/reference)`. ABL-406 quoted its margins on the challenger's **own** error, `100 · (reference − challenger)/challenger`, which is the denominator ABL-385's CV is measured in. The two always agree in sign and differ only in magnitude, so they can disagree about a grade only for a cell sitting near the floor. Both are computed for every cell and both are in the JSON.

- **Tranche 1b (solar): no cell of 6 changes grade** under either denominator, and none sits between the exact floor (10.6482%) and the 2-dp value published in prose (10.64%). The choice of denominator decided nothing here.

## 3. Boundary

No promotion, no serving-registry change, no ingest change, no refit, no replica write, no sidecar write. The grades land here, under a new path; `abl253`, `abl376`, `abl316-t1a`, `abl316-t1b`, `abl316-t2a` and `abl406-tranche2b` results files and reports are byte-unchanged, verified by blob hash against the merge base and recorded on ABL-438.

A grade is not a promotion recommendation and does not become one. Grade **A** means *promotion-eligible*, subject to any named data hold — holds registered against a pair below are printed under that pair's table and carried in the JSON, and none of them is derivable from the scores. A hold named only in a tranche's own published disposition stays there; this document does not touch it.
