# ABL-595 — C2c promotion-gate read, net position, four models

**Issue:** ABL-595 (read) · **Parent:** ABL-70 (decision) · **Author:** Forecasting Scientist
**Date:** 2026-08-28 · **Machine record:** `reports/abl_595_gate_read.json`

**Bottom line: no candidate passes. All four models return FAIL against the
eight pre-registered criteria.** The champion `chronos-2-V010` fails three of
the eight and is still the best model in the field; every challenger is worse
than the champion on identically-paired rows. **Recommendation: do not promote
any candidate.**

Nothing here was refitted, retrained or re-registered. The gate was read exactly
as registered in `src/evaluation/net_position.py` at `origin/main` `a69e003`
(blob `191575ee44923eba09f920219adc033fabe66221`, byte-identical to the file the
daily rail runs). Both databases were opened read-only
(`file:...?mode=ro`, `uri=True`); nothing was written to either.

---

## 1. Protocol

| | |
|---|---|
| Harness | `scripts/evaluate_net_position.py`, one invocation per model, each with its own `--candidate-backtest` and its own `--out-dir` |
| Gate window | `--gate-vintage-start 2026-08-07 --gate-vintage-end 2026-08-27` (end exclusive) — vintages **generated** 2026-08-07 .. 2026-08-26 |
| Read from | `gate_scope.per_country` in each model's results JSON. **The all-vintage `per_country` table was not read** — it mixes pre-fix vintages and has previously inverted the decision it was read to support |
| Replica | `C:\Code\able\data\energy_dashboard.db` (10.7 GB, refreshed 2026-08-28 07:31) — actuals + prod-pushed vintages |
| Sidecar | `C:\Code\able\data\forecasts_local.db` (refreshed 2026-08-28 08:01) — as-served vintages |
| Interpreter | `C:\Code\able\energy-forecast\.venv\Scripts\python.exe` (3.14.3) |
| Head-to-head | `scripts/compare_challenger.py` — pairs on `(country, target hour, run)`, a run being the actuals cutoff the vintage could see, within the 4 h skew bound |

Sidecar-vs-prod-pushed overlap max |Δ| is **0.000 MW** for all four models: the
rows scored are the rows served.

## 2. The vintage clock — re-derived independently, and it agrees

Counted directly out of both databases before running anything, per the
instruction not to take the recorded clock on trust:

| model | vintages generated 08-07 .. 08-26 | run-days | pairs scored | zones |
|---|---:|---:|---:|---:|
| `chronos-2-V010` (champion) | **20** | 20 | 9,082 | 19 |
| `baseline-V012` | **20** | 20 | 9,082 | 19 |
| `chronos-2-V016` | **20** | 20 | 9,082 | 19 |
| `xgboost-V014` | **19** | 19 | 8,626 | 19 |

Pre-registered threshold is 14 live shadow vintages. Met by every candidate.
Extending to 2026-08-28 reproduces the counts recorded on ABL-595 exactly
(22 / 22 / 22 / 21). **No disagreement on the clock.**

One correction of precision, not of substance: the window is *not* fully paired.
The newest `net_position` actual is `2026-08-28 21:00`, and the 2026-08-26
vintage targets 2026-08-28 00:00–23:00, so its last **two hours are unpaired in
every zone**. Per country that is 478 of 480 possible pairs (454 of 456 for
V014) — 38 hours out of 9,120, 0.4%. It does not move any number in this report.

Two vintages in the window are off-schedule catch-up runs rather than the 06:00Z
rail: `baseline-V012` / `chronos-2-V016` at `2026-08-07 21:25`, and
`xgboost-V014` at `2026-08-08 11:36`. They are legitimately scored inside each
model's own gate window (their serve-faithful baselines use their own later
cutoff), and they are correctly **rejected** from the champion head-to-heads —
a challenger that ran 15 h later saw a further day of actuals.

## 3. Per-criterion verdict, per model

**PASS requires all eight.** Verdicts, with the number behind each:

| # | criterion | V010 | V012 | V014 | V016 |
|---|---|---|---|---|---|
| 1 | `min_live_shadow_vintages` (≥ 14) | ✅ 20 | ✅ 20 | ✅ 19 | ✅ 20 |
| 2 | `excluded_zones_LU_GR` | ✅ | ✅ | ✅ | ✅ |
| 3 | `beat_baseline_ensemble_80pct` (≥ 80% of 19) | ✅ **17/19 (89%)** | ❌ 4/19 (21%) | ❌ 10/19 (53%) | ❌ 14/19 (74%) |
| 4 | `bias_under_5pct_per_country` (19/19) | ❌ **3/19** | ❌ 8/19 | ❌ 6/19 | ❌ 2/19 |
| 5 | `slope_in_range_per_country` [0.8, 1.2] (19/19) | ❌ **3/19** | ❌ 3/19 | ❌ 0/19 | ❌ 2/19 |
| 6 | `coverage_10_90_in_band_per_country` [75, 85]% (19/19) | ❌ **8/19** | ❌ 0/19 | ❌ 0/19 | ❌ 6/19 |
| 7 | `no_regression_W01_W12` | ✅ (vacuous — §7b) | ❌ NL 1,988 → 2,010 | ✅ | ❌ BE 1,140 → 1,188 |
| 8 | `serve_faithful_inputs_verified` | ✅ | ✅ | ✅ | ✅ |
| | **verdict** | **FAIL** (3 of 8) | **FAIL** (5 of 8) | **FAIL** (4 of 8) | **FAIL** (5 of 8) |

No criterion was `INCOMPLETE` or missing for any model: the two prerequisites
that would have forced that outcome — ABL-192 (W01-W12 artefacts) and ABL-193
(serve-faithful attestation) — are both discharged, and both read cleanly here.
Every model's attestation reproduces its 2026-08-11 06:00Z vintage at
**max |Δ| = 0 MW over 456 rows / 19 countries**, using only inputs published
before `2026-08-11 22:00`.

Countries failing each per-country screen (at `gate_scope`):

- **bias ≥ 5%** — V010: BE BG CZ DE EE ES FI FR HR HU LT NL PL RO SI SK ·
  V012: BE BG CZ DE FI FR HU LT PL SI SK · V014: BE BG DE ES FI HR HU LT NL PL
  RO SI SK · V016: AT BE BG CZ DE ES FI FR HR HU LT LV NL PL RO SI SK
- **slope outside [0.8, 1.2]** — V010 and V012: all but CZ, HU, SK ·
  V014: all 19 · V016: all but HU, SK
- **10-90 coverage outside [75, 85]%** — V010: AT BE CZ DE EE ES FR PL PT RO SI ·
  V012 and V014: all 19 (no quantiles emitted at all — see §6) ·
  V016: AT BE DE EE ES FI FR LV NL PL PT RO SI

## 4. Per-country screens at `gate_scope`

n is identical across the four columns within a row for V010/V012/V016 (478) and
is 454 for V014, which shadowed one fewer day.

| country | mean abs NP | MAE V010 | MAE V012 | MAE V014 | MAE V016 | slope V010 | slope V012 | slope V014 | slope V016 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AT | 1,381 | 760 | 853 | 768 | 802 | 0.41 | 0.39 | 0.34 | 0.34 |
| BE | 3,971 | 1,007 | 1,015 | 1,979 | 1,003 | 0.23 | 0.27 | 0.00 | 0.16 |
| BG | 853 | 371 | 437 | 628 | 371 | 0.71 | 0.65 | 0.22 | 0.71 |
| CZ | 1,150 | 301 | 364 | 427 | 323 | **0.91** | **0.89** | 0.54 | 0.75 |
| DE | 5,191 | 3,992 | 4,347 | 5,285 | 4,011 | 0.48 | 0.46 | 0.28 | 0.47 |
| EE | 319 | 100 | 114 | 122 | 104 | 0.56 | 0.54 | 0.46 | 0.51 |
| ES | 2,483 | 935 | 1,019 | 1,128 | 1,082 | 0.74 | 0.78 | 0.58 | 0.60 |
| FI | 960 | 713 | 876 | 794 | 734 | 0.28 | 0.18 | 0.08 | 0.28 |
| FR | 5,296 | 2,466 | 2,339 | 2,651 | 2,454 | 0.36 | 0.40 | 0.22 | 0.29 |
| HR | 743 | 342 | 368 | 340 | 331 | 0.48 | 0.40 | 0.31 | 0.46 |
| HU | 1,832 | 540 | 499 | 610 | 525 | **0.83** | **0.84** | 0.63 | **0.82** |
| LT | 368 | 276 | 344 | 310 | 276 | 0.32 | 0.16 | 0.19 | 0.32 |
| LV | 337 | 127 | 139 | 152 | 129 | 0.47 | 0.42 | 0.24 | 0.42 |
| NL | 2,491 | 1,744 | 2,044 | 1,711 | 1,717 | 0.06 | −0.12 | 0.04 | 0.04 |
| PL | 1,574 | 1,002 | 1,104 | 989 | 1,003 | 0.43 | 0.44 | 0.40 | 0.38 |
| PT | 2,307 | 590 | 633 | 613 | 581 | 0.79 | 0.78 | 0.53 | 0.76 |
| RO | 770 | 574 | 595 | 563 | 574 | 0.48 | 0.50 | 0.37 | 0.48 |
| SI | 479 | 108 | 168 | 162 | 116 | 0.44 | 0.04 | 0.33 | 0.39 |
| SK | 416 | 196 | 202 | 183 | 197 | **0.90** | **0.87** | 0.66 | **0.91** |

**Slope is the deepest failure and it is not close.** At least sixteen of
nineteen zones sit below 0.8 for every model — V010 16, V012 16, V016 17,
V014 19 — and **no zone in any model is ever above 1.2**; the misses are
one-sided. Most are far below: NL is 0.06 for the champion, BE 0.23, FI 0.28.
This is the ABL-24 shrinkage signature — the models
predict the shape of net position and roughly a third to a half of its
amplitude. Only CZ, HU and SK are inside the band, for any model. No amount of
further vintage accrual moves a systematically shrunken forecast into
[0.8, 1.2]; this needs a modelling change, not more days.

**Bias**, as a percentage of each zone's mean |net position| (bar: < 5%):

| country | V010 | V012 | V014 | V016 |
|---|---:|---:|---:|---:|
| AT | 3.4% | 3.6% | 4.9% | 20.6% |
| BE | 11.6% | 7.7% | 47.0% | 9.3% |
| BG | 5.5% | 5.3% | 16.5% | 5.5% |
| CZ | 10.4% | 13.4% | 4.1% | 8.1% |
| DE | 23.5% | 17.5% | 68.8% | 21.3% |
| EE | 5.2% | 1.0% | 2.9% | 3.4% |
| ES | 6.9% | 2.0% | 9.0% | 11.2% |
| FI | 24.4% | 27.1% | 25.7% | 25.6% |
| FR | 8.1% | 9.6% | 0.0% | 11.5% |
| HR | 14.7% | 1.8% | 13.4% | 14.5% |
| HU | 15.0% | 10.5% | 6.0% | 12.3% |
| LT | 15.2% | 5.9% | 37.4% | 15.2% |
| LV | 0.2% | 3.5% | 0.4% | 6.6% |
| NL | 6.0% | 0.3% | 12.6% | 9.7% |
| PL | 21.6% | 20.2% | 24.2% | 19.3% |
| PT | 0.4% | 0.3% | 0.0% | 2.3% |
| RO | 10.4% | 3.2% | 7.5% | 10.4% |
| SI | 10.1% | 7.8% | 21.2% | 13.7% |
| SK | 14.2% | 6.0% | 8.4% | 15.0% |

DE (−1,221 MW, 23.5%), FI (+234 MW, 24.4%) and PL (−340 MW, 21.6%) are
persistent, one-signed and large in every column — a static offset the models
share, not model-specific noise.

**10-90 coverage** (bar: 75–85%): V010 is inside in 8 zones; of its 11 misses,
10 are *under*-coverage (bands too narrow: FR 67.8%, PT 67.2%, ES 68.2%) and one
is over (CZ 87.9%). V016's correction narrows the bands further — 13 zones under
75%, none over 85% (ES 51.0%, NL 54.0%, BE 54.6%). V012 and V014 store no
quantiles at all — §7(a).

## 5. Each candidate against its own baseline, like for like

`compare_challenger.py`, exactly-paired rows only. This is the comparison the
early read (2026-08-11) rested on, now over the full window.

| comparison | paired rows | runs | baseline MAE | candidate MAE | Δ | materially better |
|---|---:|---:|---:|---:|---:|---:|
| `chronos-2-V016` vs `baseline-V012` | 9,082 | 19 | 919.0 MW | **859.7 MW** | **−6.4%** | 16/19 |
| `chronos-2-V010` vs `baseline-V012` | 8,626 | 19 | 913.8 MW | **848.4 MW** | **−7.2%** | 16/19 |
| `xgboost-V014` vs `baseline-V012` | 8,170 | 18 | 926.1 MW | 1,032.6 MW | **+11.5%** | 10/19 |

And against the incumbent:

| comparison | paired rows | runs | champion MAE | challenger MAE | Δ | materially better | pass-through |
|---|---:|---:|---:|---:|---:|---:|---:|
| `chronos-2-V016` vs `chronos-2-V010` | 8,626 | 19 | 848.4 MW | 859.0 MW | **+1.2% worse** | 4/19 | 3 |
| `baseline-V012` vs `chronos-2-V010` | 8,626 | 19 | 848.4 MW | 913.8 MW | **+7.7% worse** | 2/19 | 0 |
| `xgboost-V014` vs `chronos-2-V010` | 8,170 | 18 | 861.8 MW | 1,032.6 MW | **+19.8% worse** | 3/19 | 0 |

**The early read's V016-over-V012 finding is confirmed in direction and
overturned in significance.** V016 does beat its own baseline — 859.7 vs
919.0 MW, −6.4%, better in 16 of 19 zones — but the margin is smaller than the
−11% measured on five vintages, and the claim that made it interesting no longer
holds: **it is not the only like-for-like win over V012.** The champion beats
V012 by more (−7.2%), and head to head V016 is 1.2% *worse* than the champion,
materially better in only 4 of 19 zones and a literal pass-through in 3 (BG, LT,
RO have no fitted correction). V016 is the champion plus a correction layer that,
measured over 19 shared runs, costs slightly more than it returns — consistent
with the standing `docs/claude/09-model-details.md` result that affine
recalibration cannot reach the gate's slope band on V010 (`slope → ρ²`), which
this read confirms: V016's slope is strictly *lower* than the champion's in 15
of 19 zones, identical in 3 (the pass-throughs) and higher in exactly one (SK,
0.91 vs 0.90). The correction moves slope away from the band, never toward it.

**V014 loses to its own baseline**, driven by BE (2,044 vs 1,043 MW, slope 0.00
— it predicts essentially no amplitude there) and DE (5,344 vs 4,483 MW). It is
better than V012 in 10 of 19 zones, so the pooled figure overstates a uniform
loss, but it is the weakest candidate on every pooled measure and the only model
with 0/19 on slope.

## 6. Four-way on common rows (reported, never a criterion)

The four gate columns above are scored on each model's own vintage set. To
remove that as an explanation, the same four models on the **8,170 rows all four
share** (18 runs, 19 zones, targets 2026-08-11 .. 2026-08-28):

| model | pooled MAE | pooled bias | WAPE | zones best-of-4 |
|---|---:|---:|---:|---:|
| V010 | **861.8 MW** | −34.8 MW | **49.4%** | **12/19** |
| V016 | 872.5 MW | −26.3 MW | 50.0% | 3/19 |
| V012 | 926.1 MW | −4.5 MW | 53.1% | 2/19 |
| V014 | 1,032.6 MW | −140.1 MW | 59.2% | 2/19 |

Screen counts on those common rows: bias 7 / 10 / 5 / 4 of 19
(V010 / V012 / V014 / V016), slope 3 / 3 / 0 / 2, coverage 10 / 0 / 0 / 5.
Different window, same conclusion — **no screen reaches its bar under any
pairing**, and the ordering V010 > V016 > V012 > V014 is stable.

## 7. Three things the verdict rests on that are worth naming

**(a) Two candidates cannot pass criterion 6 at all.** `baseline-V012` and
`xgboost-V014` write **no rows to `forecast_quantiles`** — zero, for every
vintage. `n_with_band` is 0 across the whole gate scope for both, so
`coverage_10_90_pct` is `None` in all 19 zones and the criterion fails
structurally. This is not a near miss and it is not fixable by accrual: a point
model with no probabilistic head can never satisfy a 10-90 coverage band.
Both were nevertheless failing three other criteria, so this changes no verdict
— but the Board should know that promoting either would require amending a
pre-registered criterion, which is a new pre-registration, not a re-read.

**(b) The champion's `no_regression_W01_W12` PASS is a self-comparison.** The
only V010 W01-W12 artefact on disk is `comparison_net_position_servefaithful.json`,
which is simultaneously the reference the check compares against. Scoring V010
against it compares the model to itself, so the PASS carries no information. It
does not change V010's verdict (already FAIL on three criteria), and it does not
touch the challengers, whose candidate artefacts are independent.

**(c) Criterion 7 covers 4 of 19 gated zones for every model.** The reference
backtest holds AT, BE, FR, NL only, so `no_regression_W01_W12` is decided on
those four and is silent about the other fifteen. The harness says so in its own
detail string; it is repeated here because two of the four verdicts (V012, V016)
turn on a single zone each — NL +22 MW and BE +48 MW respectively, both around
1–4% — and a criterion that fails on one zone out of four gated by four out of
nineteen deserves that label attached to it.

## 8. Contamination

| issue | touches this window? | handling |
|---|---|---|
| **ABL-67** — fabricated `net_position` zeros (GR, IE) | **No.** The rows were deleted; GR's `net_position` now ends `2025-09-30 21:00` and it has no rows at all in the scored window | GR is excluded **by name**, not by symptom, so it cannot re-enter on thin data if publication resumes |
| **ABL-71** — prod ingest stale, fixes undeployed | **Provenance risk only.** `done`; replica currency verified (refreshed 2026-08-28 07:31, actuals to 08-28 21:00) | Recorded, not certified: currency is not correctness |
| **ABL-111 / ABL-109** — zero-as-missing actual **load** | **No.** Net position is scored on the `net_position` table; actual-load rows are not consumed by any of the four models' scoring path | — |

Direct screen of the 9,082 scored actual hours (19 zones, targets 2026-08-09 ..
2026-08-28): **0 nulls, 0 missing hours, 0 duplicated hours, 2 exact zeros**
(PL, isolated single hours — genuine near-balanced crossings, not a run). The
longest bit-identical run anywhere is 4 hours (ES, PT). The actuals under this
read are clean.

LU has 579 rows ending `2026-08-11 21:00` and is excluded by name; no model
forecasts it, so the exclusion costs nothing here but remains correct — LU
duplicates DE in the A25 day-ahead net position.

## 9. Recommendation

**Do not promote any candidate.** Per candidate:

- **`chronos-2-V016` — do not promote.** Fails 5 of 8. Beats its own baseline
  (−6.4%) but loses to the incumbent it corrects (+1.2%), lowers slope in 12 of
  19 zones, and narrows already-too-narrow prediction bands. The mechanism is
  understood and is a ceiling, not a tuning problem.
- **`xgboost-V014` — do not promote.** Fails 4 of 8. Worst on every pooled
  measure, 0/19 on slope, loses to both its baseline (+11.5%) and the champion
  (+19.8%), and emits no quantiles.
- **`baseline-V012` — do not promote.** Fails 5 of 8. It is the reference, not a
  candidate for serving; it beats the internally-computed serve-faithful
  ensemble in 4 of 19 zones, which is the coin-flip you would expect of a model
  that *is* that ensemble, and confirms the criterion-3 denominator is behaving.
- **`chronos-2-V010` — remains the served champion, unpromoted and ungated.**
  It fails 3 of 8 on the same read. Nothing in this issue changes what is
  serving; it is worth stating plainly that the incumbent would not clear its own
  gate today, and that its lead over every challenger is nonetheless real and
  measured on identical rows.

**On what would change this.** Bias and coverage are correctable in principle;
**slope is the binding constraint** and no candidate is within reach of it —
16 of 19 zones below 0.8 for every model, several below 0.3. Any next lever
should be argued against amplitude shrinkage specifically. That argument, the
V015 pre-consideration, and the question of whether a gate the incumbent also
fails is the right instrument are all **the CEO's and the Board's calls, not
this read's** — recorded here and not acted on.

---

**Every number above:** window = vintages generated 2026-08-07 .. 2026-08-26
(20 vintage-days; 19 for V014), targets 2026-08-09 .. 2026-08-28 21:00;
n = 9,082 pairs per model (8,626 for V014), 478 per zone (454 for V014), 19
gated zones; baseline = each model's own serve-faithful persistence/climatology
ensemble for criterion 3 and `baseline-V012` for the like-for-like table;
**out-of-sample throughout** — these are live shadow vintages scored against
actuals published after they were generated, and nothing was fitted in this read.
