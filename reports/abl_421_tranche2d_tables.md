# ABL-421 — tranche 2d generated tables

Generated from `experiments/ABL348/results_abl421_tranche2d.json`, SHA-256 `ebbc4c448dbd5614267ebabb68b1ed6bf6b1a5590eef02ef0349d6d0c1896624`, and from ABL-396's committed night-floor screen. No refit, no replica read, no recomputed metric; the grades are read back through `src/evaluation/gate_grading.py`, not re-derived. Regenerate with `.venv\Scripts\python.exe scripts/abl421_tranche2d_read.py`.

Scope `abl316-t2d`, source **`energy_generation`**, **27** features, fit rules `{'exclude_impossible_night': False}`. **14 evaluable cells of 18** in the 6 x 3 grid.

## 1. The four cells the registration declares NOT-EVALUABLE

This is the first tranche to contain ABL-348's declared pairs, and it is the reason the bar is 14 and not 18. ABL-348 `not_evaluable`: *"A pair listed here is reported NOT-EVALUABLE on the named bands. It is not a FAIL and must not be counted as one; a gate read that scores it has misread this registration."* Both were declared **before any fit existed**. Their measured numbers are printed because a declaration nobody can check is indistinguishable from a challenger quietly dropped for scoring badly — but they carry no gate outcome and no grade.

| pair | declared bands | n_d7_scorable | registered min n | cause | source-dependent? |
|---|---|---:|---:|---|:---:|
| EE/solar | 24-36h, 36-48h | 630 | 684 | ABL-188 excludes a 44.8h bit-identical zero run, 2026-07-21 00:00 -> 2026-07-22 20:45, present identically in BOTH tables. Not caused by the source change. | no |
| FI/solar | 24-36h, 36-48h | 650 | 684 | energy_generation holds 663 of 720 gate hours against energy_renewable's 717 - the ABL-322 section 3.3 phenomenon on a second pair. Broken by the source change itself. | **yes** |

**Only one of the two is ours.** EE's shortfall is an ABL-188 bit-identical zero run present identically in *both* source tables, so reverting ABL-348's source change would not recover it. FI's is `energy_generation` holding 663 of the 720 gate hours against `energy_renewable`'s 717 — that one **is** a cost of the source change, and it is a finding for whoever owns that decision rather than a fact about FI's model.

What those cells measured, for audit only — no verdict attaches:

| country | horizon | n | min n | challenger WAPE | D-7 WAPE | skill vs D-7 |
|---|---|---:|---:|---:|---:|---:|
| EE | 24-36h | 543 | 684 | 23.80% | 36.22% | +34.3% |
| EE | 36-48h | 540 | 684 | 24.01% | 36.26% | +33.8% |
| FI | 24-36h | 629 | 684 | 26.29% | 38.12% | +31.0% |
| FI | 36-48h | 628 | 684 | 25.64% | 38.12% | +32.7% |

**48-64h is read for both pairs**, on ABL-348's own instruction (`not_evaluable.note_48_64h`): that band selects a 480-510 row subset, so its n scales proportionally rather than being hard-bounded by `n_d7_scorable`, and "a pair declared here may still clear 456 in that band and should be reported if it does". Where such a cell falls short of 456 it is a **coverage shortfall** (`enough_pairs: false`), not a loss to D-7; the two flags are separate in the record.

## 2. Night floor, all six countries — the zeros stated, not omitted

`f` is ABL-396 section 2's `wape_floor_pct_if_clamped`: the share of the window's total |energy| booked at night, which is the **full width in WAPE points** of the interval an all-hours read can occupy relative to the daylight-only read of the same challenger. Source `energy_generation`, the table this tranche fits and scores on. The signed share is printed beside it because for NL the two differ in sign, which is the whole of section 4 below.

| country | window | night hrs | hrs > 1 MW | night mean | signed share | **f** |
|---|---|---:|---:|---:|---:|---:|
| EE | fit | 1,153 | 937 | 6.19 MW | +0.9700% | **0.9710%** |
| EE | gate | 86 | 68 | 12.64 MW | +0.7200% | **0.7180%** |
| FI | fit | 1,047 | 646 | 2.64 MW | +0.2900% | **0.2850%** |
| FI | gate | 6 | 2 | 0.81 MW | +0.0000% | 0.0020% |
| LT | fit | 1,317 | 8 | 0.13 MW | +0.0100% | 0.0130% |
| LT | gate | 127 | 5 | 0.57 MW | +0.0200% | 0.0180% |
| LV | fit | 1,256 | 700 | 3.25 MW | +0.4200% | **0.4180%** |
| LV | gate | 116 | 18 | 0.73 MW | +0.0400% | 0.0410% |
| NL | fit | 1,390 | 0 | -1.22 MW | -0.6000% | **0.6000%** |
| NL | gate | 154 | 0 | -0.13 MW | -0.0400% | 0.0400% |
| SE | fit | 1,197 | 336 | 0.81 MW | +0.0600% | 0.0570% |
| SE | gate | 96 | 89 | 1.63 MW | +0.0300% | 0.0330% |

EE is the only country here with a material floor and carries the **third-largest in the fleet**. The other five are at or under 0.042% of gate-window energy, where the bound below is narrower than the rounding on a reported WAPE.

## 3. EE's night-floor band, on the face of the table

EE gate-window `f` = **0.7180%**. Every arm of the gate is scored on the same all-hours rows, so no verdict below is adjusted by this band — the band is what **bounds** it. Two of EE's three bands are NOT-EVALUABLE, so for those the bound is the only quantitative statement this tranche makes about them, which is exactly what makes it worth printing.

| band | status | n | all-hours WAPE `A` (measured) | implied daylight-only `W` | `[W(1-f), W(1-f)+f]` | clamped variant `[A, A+f]` |
|---|:---:|---:|---:|---:|---:|---:|
| 24-36h | NOT-EVALUABLE | 543 | 23.80% | 23.25%–23.97% | 23.08%–24.52% | 23.80%–24.52% |
| 36-48h | NOT-EVALUABLE | 540 | 24.01% | 23.46%–24.19% | 23.30%–24.73% | 24.01%–24.73% |
| 48-64h | gated | 388 | 25.05% | 24.51%–25.23% | 24.33%–25.77% | 25.05%–25.77% |


**For a known `W` the band `[W(1-f), W(1-f)+f]` has width exactly `f` = 0.7180 WAPE points.** `W` is bounded here rather than measured — the harness scores all hours — so the printed envelope is that band taken across the whole implied `W` range, which is `[A-f, A+f]` and contains the measured `A` by construction. The last column answers the separate serving-side question: the ABL-337 clamp forces a zero on this same night predicate, so a served version of this challenger would score in `[A, A+f]`. It gates nothing here.

**The widest of these intervals is the `[A-f, A+f]` envelope at 1.4360pp, against the 10.65pp readability floor ABL-418 registers for solar — a factor of 7.** (The band at a known `W`, and the clamped column, are half that at 0.7180pp.) So on EE the night floor cannot move a grade in either direction, and that is now measured rather than assumed. ABL-425 (open, PR #59) independently registers `EE: False` in `NIGHT_GENERATION_POSSIBLE` — EE's floor is contamination, not real generation — which is consistent with bounding it here rather than adjusting for it.

## 4. NL: what a signed target does to the WAPE denominator

NL solar is negative at **every** night hour — 1,544 of 1,544 across both windows (ABL-396 section 6, ABL-412). That is our own netting rule, not upstream and not a sign error, and ABL-412 fixed it at the dashboard *read site*, not in the data. This gate reads the data, so the question is what it does to the score.

**It is arithmetically negligible, and here is the number.** `score_predictions` uses `denom = sum(|actual|)`, so a negative night hour contributes its *magnitude* to the denominator rather than cancelling against daylight. The two conventions therefore differ by exactly NL's absolute night share, `f` = **0.0400%** of the denominator (the signed share is -0.0400% — same magnitude, opposite sign, which is the tell). Zeroing the night instead would shrink the denominator by that 0.0400% and raise WAPE by the same relative amount: on NL's 46.53% D-7 bar that is **0.0186pp**. Against ABL-418's 10.65pp solar floor it is four orders of magnitude short. The numerator is bounded the same way: the night actuals average -0.13 MW, so a non-negative prediction pays at most that per night hour.

**So NL's margin, whatever it is, is not a netting artefact — but NL's *level* is the finding.** Its gate-window mean is 66.7 MW against a 251.3 MW window maximum, and that series is **bit-identical in both source tables**, so it is upstream rather than ours. For scale, over the same 720 hours `energy_generation` books BE at 8,140 MW max and even EE — a country of 1.3 million — at 771.6 MW. NL's published solar series is a small metered subset, stable in that shape across 18 months, not its fleet. The gate read below is a valid read *of that series*; it must not be quoted as "we can forecast NL solar", and any NL promotion recommendation has to carry this.

## 5. Pair grade, against the pre-committed bar and the level

The bar column is ABL-348's, measured before any challenger for these pairs existed. It is here because ABL-406 established across eight wind pairs that the gate outcome was *fully* predicted by whether a causal constant clears the bar on its own, and ABL-417 reproduced the anti-correlation on RO. **These are the loosest solar bars in the programme** — 23.92% to 47.85%, against 2c's 7.11-16.43% — which is precisely the combination (loose bar, low level) that produced 2b's spurious wind passes.

The level column carries `SK/solar, 114.8 MW, graded A in tranche 2a`, the lowest solar fleet already dispositioned, as a reference line. ABL-348 registers no decision-grade threshold for solar — the one it states explicitly is `CH_wind_onshore_is_not_decision_grade` at 12.9 MW — so this is a **comparison, not a registered bar**.

| pair | pre-committed D-7 bar | gate-window mean | vs SK line | bands gated | bands decidable | band grades | ladder grade | **reported** | failed conditions / hold | bar weaker than a flat line? |
|---|---:|---:|:---:|:---:|:---:|---|:---:|:---:|---|:---:|
| EE | 36.67% | 223.0 MW | above | 1/3 | 0/3 | A | A | **—** | no band meets the registered minimum n | no |
| FI | 37.88% | 448.0 MW | above | 1/3 | 0/3 | A | A | **—** | no band meets the registered minimum n | no |
| LT | 30.84% | 543.8 MW | above | 3/3 | 3/3 | A / A / A | A | **A** | — | no |
| LV | 47.85% | 292.5 MW | above | 3/3 | 3/3 | A / A / A | A | **A** | — | no |
| NL | 46.53% | 66.7 MW | **below** | 3/3 | 3/3 | B / B / B | B | **B** | G3 | no |
| SE | 23.92% | 651.5 MW | above | 3/3 | 3/3 | A / A / A | A | **A** | — | no |


**EE and FI grade `A` on the margin and are reported `—`, and the gap between those two things is the finding of this tranche.** ABL-418's ladder is handed a cell's `scores` and nothing else — it never sees `gate.enough_pairs` or `gate.n` — so it grades a *margin*. Both pairs clear D-7 readably on their single gated band (EE +29.0%, FI +36.8%, against a 10.65pp floor) while missing the registered minimum of 456 rows, **FI by three**. A margin the registration does not consider readable cannot carry a promotion, so the hold is named in the ladder's own vocabulary — `A` is defined as promotion-eligible *subject to any named data hold* — and the ladder grade is printed beside it rather than replaced. This is deliberately **not** a change to `gate_grading.py`: editing the ladder after seeing a result is the shopping the pre-registration exists to prevent, and that module is shared with the wind harness. The combination had not arisen before — every cell in 2a, 2b and 2c met its minimum — so this is a gap the ladder has never been exercised against, and it is a candidate for its own pre-registered issue rather than a patch here.

**On the level, exactly one pair sits below the SK reference line: NL.** ABL-421's description anticipated "several"; measured, EE is the next lowest at 223.0 MW, which is nearly twice SK's 114.8 MW. The distinction matters because it is the level, not the bar, that decides whether a WAPE can carry a promotion decision at all.

**Do not average this tranche's pass rate against 2a's or 2c's.** The bars are not comparable: 2c's ran 7.11-16.43% on Mediterranean July solar that is nearly D-7 periodic, and these run 23.92-47.85%. ABL-348 registered that reading in advance under `reading_caveats_not_band_changes`. A pass against a loose bar and a pass against a tight one are not the same evidence, which is what the grade ladder exists to say.

## 6. Which references each pair actually beats

ABL-417's lesson, re-run here: of its five A-graded pairs only two beat all four model-free references. The two oracles are hindsight and **gate nothing** — that is registered, and losing to one bounds what a verdict means rather than voiding it — but an A that loses to the average day in hindsight is a different object from one that does not.

**On solar the constant is a formality and the climatology is the real test.** A flat line scores 80.4-103.2% here (NL's causal constant is *above 100%*: worse than predicting zero), because a constant cannot represent a diurnal cycle and on solar the diurnal cycle is the signal. So `bar weaker than a flat line? no` in section 5 is uninformative on this stream, exactly as CLAUDE.md records — read the climatology columns instead.

| pair | worst-band challenger | clim causal | clim oracle | const causal | beats all four? |
|---|---:|---:|---:|---:|:---:|
| EE | 25.05% | 29.14% | **23.29%** | 80.42% | **no** |
| FI | 23.99% | 45.07% | **22.56%** | 82.52% | **no** |
| LT | 20.94% | 44.80% | **17.47%** | 90.88% | **no** |
| LV | 32.17% | 36.01% | 33.89% | 89.91% | yes |
| NL | 37.66% | **34.75%** | **32.57%** | 85.18% | **no** |
| SE | 21.16% | 40.66% | **17.89%** | 87.49% | **no** |

Bold is a reference the challenger's **worst** band does not beat. Compared on the toughest band per pair, which is the conservative direction and matches the ladder's worst-band rule.

**1 of 6 beat all four (LV); EE, FI, LT, NL, SE do not.** Every one of the shortfalls is against the **oracle** climatology, which is causally unavailable and gates nothing — so this qualifies the reads rather than overturning them. NL is the exception and the serious one: it is the only pair that loses to the *causal* climatology, on all three bands, which is what G3 caught and why it grades B.

