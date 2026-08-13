# ABL-418 — graded gate disposition (G1–G4), and the retro-grade of tranches 2a and 2b

**Generated from the stored results files, not restated in prose.** Every grade below is produced by `src/evaluation/gate_grading.py` — the same code both gate harnesses now call — reading `experiments/ABL348/results_abl405_tranche2a.json` and `experiments/ABL348/results_abl406_tranche2b.json`. No refit, no new fit, no replica read, no write to any dispositioned path. Regenerate with `.venv\Scripts\python.exe scripts/abl418_retro_grade.py`.

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

ABL-385 registers `delta_min(k) = 1.96 * sqrt(c_A^2 + c_B^2) / sqrt(k)` as the minimum readable relative gap. Every reference on this ladder is **deterministic** — D-7, a flat line and an hour-of-day climatology do not move when the challenger is refitted — so `c_B = 0`, and the published two-arm margin is a factor of √2 too wide. Both tranches fit once per cell, so k = 1.

| stream | fleet p90 per-fit CV (ABL-385 §1) | two-arm δ_min at k=1 | **floor used** = δ_min/√2 | published in prose |
|---|---:|---:|---:|---:|
| solar | 5.4328% | 15.0589% | **10.6482%** | 10.64% |
| wind | 3.8293% | 10.6143% | **7.5054%** | 7.51% |

The prose values are 2-dp renderings and are not what the ladder uses; the exact `1.96 · c` value is. The gap between them is under 0.01pp and no cell of either tranche sits inside it — checked per cell in §3.

## 1. Tranche 2a — `abl316-t2a` (solar)

Source: `experiments/ABL348/results_abl405_tranche2a.json`, SHA-256 `895e1259c0da3921f4de18c72b912780c5c9ddccd056efdc43274c6cb7d00920`. Evidence pack: `reports/abl_405_tranche2a_findings.md`. Published disposition, restated unchanged: **PERFORMANCE PASS — HOLD FOR CONTAMINATION ADJUDICATION**. Gate window 2026-07-11 00:00:00 → 2026-08-10 00:00:00 (exclusive), target series `energy_renewable`. Floor 10.6482% at k=1.

| pair | band | n | gate | skill vs D-7 | vs constant causal | vs climatology causal | slope>0 & corr>0 | grade |
|---|---|---:|:---:|---:|---:|---:|:---:|:---:|
| BG | 24-36h | 720 | PASS | +19.56% | +73.93% | +53.11% | yes | **A** |
| BG | 36-48h | 720 | PASS | +20.04% | +74.09% | +53.38% | yes | **A** |
| BG | 48-64h | 510 | PASS | +16.70% | +69.45% | +49.48% | yes | **A** |
| CH | 24-36h | 720 | PASS | +39.55% | +91.93% | +80.70% | yes | **A** |
| CH | 36-48h | 720 | PASS | +40.70% | +92.09% | +81.06% | yes | **A** |
| CH | 48-64h | 510 | PASS | +36.24% | +90.75% | +79.57% | yes | **A** |
| CZ | 24-36h | 720 | PASS | +46.28% | +86.29% | +56.11% | yes | **A** |
| CZ | 36-48h | 720 | PASS | +46.34% | +86.31% | +56.16% | yes | **A** |
| CZ | 48-64h | 510 | PASS | +41.85% | +83.72% | +50.83% | yes | **A** |
| HU | 24-36h | 720 | PASS | +4.60% | +81.87% | +43.90% | yes | **U(+)** |
| HU | 36-48h | 720 | PASS | +4.61% | +81.87% | +43.90% | yes | **U(+)** |
| HU | 48-64h | 510 | PASS | +7.62% | +81.34% | +44.56% | yes | **U(+)** |
| PL | 24-36h | 720 | PASS | +33.31% | +81.28% | +38.30% | yes | **A** |
| PL | 36-48h | 720 | PASS | +33.19% | +81.24% | +38.19% | yes | **A** |
| PL | 48-64h | 510 | PASS | +33.49% | +81.03% | +39.81% | yes | **A** |
| RO | 24-36h | 720 | PASS | +22.76% | +80.51% | +56.81% | yes | **A** |
| RO | 36-48h | 720 | PASS | +23.04% | +80.59% | +56.97% | yes | **A** |
| RO | 48-64h | 510 | PASS | +23.33% | +79.23% | +54.79% | yes | **A** |
| SI | 24-36h | 720 | PASS | +17.25% | +81.14% | +48.89% | yes | **A** |
| SI | 36-48h | 720 | PASS | +16.36% | +80.94% | +48.35% | yes | **A** |
| SI | 48-64h | 510 | PASS | +12.10% | +78.46% | +45.65% | yes | **A** |
| SK | 24-36h | 715 | PASS | +13.27% | +83.19% | +49.95% | yes | **A** |
| SK | 36-48h | 715 | PASS | +13.02% | +83.14% | +49.81% | yes | **A** |
| SK | 48-64h | 507 | PASS | +17.86% | +83.23% | +52.35% | yes | **A** |

| pair | bands | grade | failed conditions | bar weaker than a flat line? | beats constant oracle? | beats climatology oracle? | ABL-418 description | agrees? |
|---|---|:---:|---|:---:|:---:|:---:|:---:|:---:|
| BG | A / A / A | **A** | — | no | yes | no | A | yes |
| CH | A / A / A | **A** | — | no | yes | yes | A | yes |
| CZ | A / A / A | **A** | — | no | yes | yes | A | yes |
| HU | U(+) / U(+) / U(+) | **U(+)** | — | no | yes | no | U(+) | yes |
| PL | A / A / A | **A** | — | no | yes | no | A | yes |
| RO | A / A / A | **A** | — | no | yes | yes | A | yes |
| SI | A / A / A | **A** | — | no | yes | no | A | yes |
| SK | A / A / A | **A** | — | no | yes | no | A | yes |

Pair grades: **A** × 7, **U(+)** × 1.

Every pair reproduces the reading in the ABL-418 description.

## 2. Tranche 2b — `abl406-tranche2b` (wind)

Source: `experiments/ABL348/results_abl406_tranche2b.json`, SHA-256 `972eea5fe8880668cfd59630005e054f9a8153cce46ee287dedeea3386868843`. Evidence pack: `reports/abl_406_evidence_pack.md`. Published disposition, restated unchanged: **FAIL**. Gate window 2026-07-11 00:00:00 → 2026-08-10 00:00:00 (exclusive), target series `energy_generation`. Floor 7.5054% at k=1.

| pair | band | n | gate | skill vs D-7 | vs constant causal | vs climatology causal | slope>0 & corr>0 | grade |
|---|---|---:|:---:|---:|---:|---:|:---:|:---:|
| ES wind_onshore | 24-36h | 720 | FAIL | -32.24% | +12.57% | +1.58% | yes | **C — fails G1** |
| ES wind_onshore | 36-48h | 720 | FAIL | -32.04% | +12.71% | +1.74% | yes | **C — fails G1** |
| ES wind_onshore | 48-64h | 510 | FAIL | -36.00% | +18.00% | +1.26% | yes | **C — fails G1** |
| FI wind_onshore | 24-36h | 711 | PASS | +31.00% | +24.72% | +23.00% | yes | **A** |
| FI wind_onshore | 36-48h | 711 | PASS | +27.33% | +20.72% | +18.91% | yes | **A** |
| FI wind_onshore | 48-64h | 504 | PASS | +15.83% | +11.12% | +8.96% | yes | **A** |
| GR wind_onshore | 24-36h | 720 | PASS | +53.61% | +44.35% | +44.58% | yes | **A** |
| GR wind_onshore | 36-48h | 720 | PASS | +53.66% | +44.41% | +44.64% | yes | **A** |
| GR wind_onshore | 48-64h | 510 | PASS | +48.65% | +40.61% | +40.45% | yes | **A** |
| IT wind_onshore | 24-36h | 716 | FAIL | -1.06% | +22.40% | +21.30% | yes | **U(+)** |
| IT wind_onshore | 36-48h | 715 | FAIL | -0.77% | +22.50% | +21.42% | yes | **U(+)** |
| IT wind_onshore | 48-64h | 505 | PASS | +0.57% | +19.32% | +19.72% | yes | **U(+)** |
| NO wind_onshore | 24-36h | 720 | PASS | +15.83% | +13.96% | +13.59% | no | **B — fails G4** |
| NO wind_onshore | 36-48h | 720 | PASS | +15.50% | +13.62% | +13.24% | no | **B — fails G4** |
| NO wind_onshore | 48-64h | 510 | PASS | +15.94% | +10.19% | +10.66% | no | **B — fails G4** |
| PL wind_onshore | 24-36h | 720 | PASS | +41.67% | +11.52% | +9.42% | yes | **A** |
| PL wind_onshore | 36-48h | 720 | PASS | +43.40% | +14.15% | +12.11% | yes | **A** |
| PL wind_onshore | 48-64h | 510 | PASS | +45.62% | +19.64% | +16.54% | yes | **A** |
| PT wind_onshore | 24-36h | 720 | FAIL | -37.47% | +32.73% | +32.54% | yes | **C — fails G1** |
| PT wind_onshore | 36-48h | 720 | FAIL | -38.47% | +32.24% | +32.06% | yes | **C — fails G1** |
| PT wind_onshore | 48-64h | 510 | FAIL | -30.75% | +34.50% | +30.17% | yes | **C — fails G1** |
| SE wind_onshore | 24-36h | 720 | PASS | +43.56% | +30.91% | +29.26% | yes | **A** |
| SE wind_onshore | 36-48h | 720 | PASS | +43.42% | +30.73% | +29.09% | yes | **A** |
| SE wind_onshore | 48-64h | 510 | PASS | +42.65% | +31.90% | +29.38% | yes | **A** |

| pair | bands | grade | failed conditions | bar weaker than a flat line? | beats constant oracle? | beats climatology oracle? | ABL-418 description | agrees? |
|---|---|:---:|---|:---:|:---:|:---:|:---:|:---:|
| ES wind_onshore | C / C / C | **C** | G1 | no | no | no | C | yes |
| FI wind_onshore | A / A / A | **A** | — | yes | yes | yes | A | yes |
| GR wind_onshore | A / A / A | **A** | — | yes | yes | yes | A | yes |
| IT wind_onshore | U(+) / U(+) / U(+) | **U(+)** | — | no | no | no | U | **no** |
| NO wind_onshore | B / B / B | **B** | G4 | yes | no | no | B | yes |
| PL wind_onshore | A / A / A | **A** | — | yes | mixed | no | A | yes |
| PT wind_onshore | C / C / C | **C** | G1 | no | no | no | C | yes |
| SE wind_onshore | A / A / A | **A** | — | yes | yes | yes | A | yes |

Pair grades: **A** × 4, **B** × 1, **C** × 2, **U(+)** × 1.

**Disagreements with the ABL-418 description — the arithmetic wins, and the cells are named:**

- **IT wind_onshore: the description reads `U`, the ladder gives `U(+)`.** Its G1 margin is inside the floor in every band, so it is `U` either way — the `(+)` is what differs, and it follows from the ladder's own text, *"if G2–G4 clear readably"*. Per band: 24-36h U(+) / 36-48h U(+) / 48-64h U(+). Skill vs D-7 -1.06% / -0.77% / +0.57% against a 7.51% floor; vs `constant_causal` +22.4% / +22.5% / +19.3%; vs `climatology_causal` +21.3% / +21.4% / +19.7%; slope and correlation positive in all bands. All three conditions clear, and G2/G3 clear by more than the floor, so the disposition is **re-read at k>1 seeds**, not **report and do not decide**. It is the weaker of the two `U` readings to act on, and the qualifier belongs with it: IT wind_onshore loses readably to **both** oracle references, which gate nothing but bound what a re-read could establish.

## 3. Sensitivity: which denominator, and the 2-dp rounding

ABL-418 registers G1 on the printed `skill vs D-7` column, `100 · (1 − challenger/reference)`. ABL-406 quoted its margins on the challenger's **own** error, `100 · (reference − challenger)/challenger`, which is the denominator ABL-385's CV is measured in. The two always agree in sign and differ only in magnitude, so they can disagree about a grade only for a cell sitting near the floor. Both are computed for every cell and both are in the JSON.

- **Tranche 2a (solar): no cell of 24 changes grade** under either denominator, and none sits between the exact floor (10.6482%) and the 2-dp value published in prose (10.64%). The choice of denominator decided nothing here.
- **Tranche 2b (wind): no cell of 24 changes grade** under either denominator, and none sits between the exact floor (7.5054%) and the 2-dp value published in prose (7.51%). The choice of denominator decided nothing here.

## 4. Boundary

No promotion, no serving-registry change, no ingest change, no refit, no replica write, no sidecar write. The grades land here, under a new path; `abl253`, `abl376`, `abl316-t1a`, `abl316-t1b`, `abl316-t2a` and `abl406-tranche2b` results files and reports are byte-unchanged, verified by blob hash against the merge base and recorded on ABL-418.

A grade is not a promotion recommendation and does not become one. Grade **A** means *promotion-eligible*, subject to any named data hold — for tranche 2a that hold is live and named in its own published disposition, which this document does not touch.
