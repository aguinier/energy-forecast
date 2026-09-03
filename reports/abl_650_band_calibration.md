# ABL-650 — the net-position p10-p90 band: which defect, and the fix

**Issue:** ABL-650 · **Parent:** ABL-70 (Board decision 2026-09-03, `narrow_fixes` → `band`)
**Author:** Forecasting Scientist · **Date:** 2026-09-03
**Machine records:** `reports/abl_650_band_calibration.json`, `reports/abl_650_serving_verification.json`
**Registration:** `experiments/net_position_quantile_calibration.json`

---

## 0. Bottom line

**This is defect (a) — the quantiles are genuinely too narrow.** The levels
served are the model's own 0.10 and 0.90, the client draws exactly those rows
and captions them "p10–p90", and there is no hardcoded "80%" anywhere in the
dashboard. Nothing in this fix belongs to the Founding Engineer.

Three things follow, and the second is a correction to the issue's premise.

1. **V016's band is decisively too narrow.** Pooled 10-90 coverage over the gate
   window is **69.1%, 95% CI [65.5, 72.4]** — 10 of 19 zones individually
   outside their own null. A single pooled multiplier fixes it in all three
   held-out folds, at a **+22% width** cost.
2. **The champion's band is only mildly too narrow, and the 68% figure is three
   zones, not the fleet.** V010's pooled coverage over the same window is
   **75.8%, CI [72.1, 79.1]** — outside 80%, so a real defect, but ~4pp rather
   than ~12pp. Of the three zones the issue names, only **ES (67.9%) and PT
   (67.1%)** have intervals excluding 80%; **FR's 67.7% has CI [54.6, 80.0] and
   is inside its own null**.
3. **Per-zone calibration does not work and must not ship.** Fitted on one
   fortnight and scored on the next it *loses to doing nothing* on the champion
   — 9 → 4 zones inside [75, 85]%, mean absolute per-zone coverage error
   9.9 → 13.9pp. A zone's measured coverage does not persist window to window
   (fit→eval Pearson r **+0.27 / −0.20 / +0.06** across three folds, with
   per-zone shifts up to 27pp between adjacent fortnights).

**Shipped: a pooled two-sided conformal multiplier per model, applied at the
quantile write path.** The served point forecast is bit-identical — measured at
**max |Δ| = 0.0 MW over 6,138 held-out rows** for both models, not asserted.

---

## 1. Which defect: (a) or (b)

The wake comment asked this first. Both arms were checked.

**The code path, end to end.** `ChronosEngine.forecast` calls
`predict_quantiles(quantile_levels=config.CHRONOS2_QUANTILE_LEVELS)` with
`[0.1 … 0.9]`; `forecast_chronos2.py:217` maps output row `qi` to
`CHRONOS2_QUANTILE_LEVELS[qi]` in that same order; `save_quantile_forecasts`
stores `float(q_level)`; the dashboard server selects
`MAX(CASE WHEN q.quantile = 0.1 …)` and `= 0.9` explicitly
(`server/src/services/netPositionService.ts:267-268`); the client assigns them
to `min`/`max` and captions the chart **"shaded band = p10–p90"**
(`NetPositionTab.tsx:342`, `:683`). **The string "80%" does not appear in the
client or the server.** The 80% is the arithmetic implication of the p10–p90
label, plus the pre-registered gate range `GATE_COVERAGE_RANGE = (75.0, 85.0)`.
So the rendering is faithful and the label is arithmetically correct — what is
wrong is the number behind it.

**The measurement agrees.** On the gate window (vintages 2026-08-07..08-26,
9,120 pairs, 19 zones, 20 vintages):

| model | q10-q90 (nom. 80%) | q20-q80 (60%) | q30-q70 (40%) | q40-q60 (20%) |
|---|---:|---:|---:|---:|
| `chronos-2-V010` | 75.8% | 54.7% | 35.4% | 18.2% |
| `chronos-2-V016` | 69.1% | 48.4% | 30.5% | 15.1% |

All four nested intervals under-cover together and monotonically. A mislabelled
or misdrawn band moves *one* interval; a scale defect in the predictive
distribution moves all of them, which is what happened. Two further checks
point the same way: **zero quantile crossings** in 9,120 rows for either model,
and V010's marginal PIT lands its nine stored levels at 12.4 / 23.3 / 33.2 /
42.3 / 51.8 / 60.5 / 68.6 / 78.0 / 88.3% — a smooth interior miscalibration a
couple of points wide, not a level swap (a secretly-0.25 level would read ~25%).

One incidental finding worth recording: **the stored `q50` is bit-identical to
the served point value** on all 9,120 rows, max |Δ| = 0.0000 MW. V010 serves the
median, not the predictive mean. That is what makes a median-anchored
calibration able to satisfy the Board's constraint exactly.

**Verdict: (a). No dashboard change is required and none was made.**

---

## 2. Protocol

| | |
|---|---|
| Replica | `C:\Code\able\data\energy_dashboard.db` (10.7 GB, refreshed 2026-09-03 17:09) — actuals + prod-pushed vintages |
| Sidecar | `C:\Code\able\data\forecasts_local.db` — as-served vintages |
| Access | both opened read-only (`file:…?mode=ro`, `uri=True`); **nothing written to either** |
| Interpreter | `C:\Code\able\energy-forecast\.venv\Scripts\python.exe` (3.14.3) |
| Loader | `src/evaluation/net_position.py` `load_forecasts` / `load_actuals` — the same join the gate reads |
| Zones | the 19 `GATE_COUNTRIES`; LU and GR excluded by name as pre-registered |
| Models | `chronos-2-V010`, `chronos-2-V016` only. `baseline-V012` and `xgboost-V014` write **zero** rows to `forecast_quantiles` (ABL-595 §7a) and were not touched |
| Fit window | vintages **2026-08-05 .. 2026-08-18** (V010 15 vintages / 6,840 rows; V016 12 / 5,472) |
| Eval window | vintages **2026-08-19 .. 2026-09-01**, 14 vintages / 6,138 rows per model — **strictly later, never seen by the fit** |
| Folds | three expanding-origin folds on top of that single split (§4) |
| Uncertainty | 2,000-draw **vintage-block** bootstrap, seed 650 |

**Why the fit starts 2026-08-05.** `FIX_DEPLOYED_UTC` is 2026-08-04 14:29 — the
context-cutoff fix. Vintages before it ran the zero-padded context and are a
different model regime; fitting a calibration on them would calibrate a model
that is no longer served. V010 has stored quantiles back to 2026-07-24, and they
were deliberately not used. That caps the usable history at **29 vintages**, and
that cap is the binding constraint on everything in §4.

**Contamination.** Of the four standing issues, **ABL-67** touches this window:
GR's fabricated net-position zeros. GR is excluded from the gate scope by name,
so it enters nothing here. ABL-71 (stale prod ingest) is not load-bearing —
scoring reads `net_position` actuals from the replica and the sidecar-vs-prod
overlap is 0.000 MW. ABL-111/ABL-109 are actual-**load** rows and do not touch
net position. No window in this pack was trimmed for contamination.

**A sample-size correction that changes how every number below reads.** A
vintage forecasts 24 consecutive hours from one context; a run that misreads the
day misses most of it. Treating 480 hours as 480 independent trials makes the
interval roughly five times too tight. Every interval here **resamples whole
vintages**, so the sample size is 20 runs, not 9,120 rows.

---

## 3. The defect, measured — gate window, uncalibrated

`chronos-2-V010`, vintages 2026-08-07..08-26, n = 480/zone, 20 vintages.
Pooled: **75.83%, CI [72.08, 79.12]** — excludes 80%.

| zone | n | coverage | 95% CI | outside 80%? | below p10 | above p90 | width MW | MAE MW | width/MAE |
|---|---:|---:|---|:--:|---:|---:|---:|---:|---:|
| AT | 480 | 73.8% | [66.0, 81.0] | | 13.8% | 12.5% | 2,123 | 759 | 2.80 |
| BE | 480 | 72.1% | [61.7, 80.4] | | 22.1% | 5.8% | 2,684 | 1,003 | 2.67 |
| BG | 480 | 76.5% | [71.0, 81.5] | | 13.8% | 9.8% | 1,030 | 371 | 2.78 |
| CZ | 480 | 87.9% | [79.8, 94.4] | | 10.8% | 1.2% | 1,199 | 300 | 4.00 |
| DE | 480 | 71.7% | [61.7, 81.9] | | 7.1% | 21.2% | 10,518 | 3,992 | 2.64 |
| EE | 480 | 75.0% | [65.4, 84.6] | | 10.6% | 14.4% | 329 | 100 | 3.31 |
| **ES** | 480 | **67.9%** | **[60.2, 75.4]** | **yes** | 15.2% | 16.9% | 2,530 | 939 | 2.69 |
| FI | 480 | 78.1% | [67.3, 88.8] | | 13.5% | 8.3% | 2,286 | 712 | 3.21 |
| FR | 480 | 67.7% | [54.6, 80.0] | | 19.4% | 12.9% | 6,734 | 2,465 | 2.73 |
| HR | 480 | 78.3% | [67.9, 87.5] | | 17.3% | 4.4% | 1,054 | 343 | 3.07 |
| HU | 480 | 82.5% | [74.0, 90.0] | | 2.1% | 15.4% | 1,688 | 539 | 3.13 |
| LT | 480 | 79.4% | [68.7, 88.5] | | 9.4% | 11.2% | 827 | 275 | 3.00 |
| LV | 480 | 82.3% | [72.5, 90.6] | | 10.4% | 7.3% | 411 | 127 | 3.25 |
| NL | 480 | 79.0% | [67.9, 89.2] | | 12.3% | 8.8% | 5,329 | 1,741 | 3.06 |
| PL | 480 | 74.8% | [65.2, 84.0] | | 6.0% | 19.2% | 2,896 | 1,001 | 2.89 |
| **PT** | 480 | **67.1%** | **[59.6, 74.0]** | **yes** | 16.9% | 16.0% | 1,489 | 592 | 2.51 |
| RO | 480 | 72.7% | [61.3, 81.9] | | 11.9% | 15.4% | 1,596 | 574 | 2.78 |
| SI | 480 | 74.6% | [64.2, 84.8] | | 19.0% | 6.5% | 316 | 108 | 2.92 |
| SK | 480 | 79.6% | [71.7, 86.7] | | 4.8% | 15.6% | 592 | 196 | 3.02 |

**Reproduction of the issue's numbers.** FR 67.7 / PT 67.1 / ES 67.9 against the
issue's FR 67.8 / PT 67.2 / ES 68.2. The difference is 2 hours per zone: ABL-595
scored 478 pairs where the 2026-08-26 vintage's last two hours were unpaired,
and those hours now have actuals. Same reason 9 of 19 zones sit in [75, 85] here
where ABL-595 recorded 8 — EE moved from just under 75% to exactly 75.0%. **The
direction, magnitude and zone ordering all reproduce.**

**Where this corrects the issue.** The issue reads "covers roughly 68%". Pooled
across the 19 gated zones the champion's band covers **75.8%**. The 67–68%
figures are the three worst zones, and of those only ES and PT survive their own
vintage-block interval. Ten of nineteen zones miss the [75, 85] screen, nine of
them low — matching the issue's "10 of 11 misses are under-coverage" up to the
same two-hour shift. **The defect is real and worth fixing; it is about 4pp
pooled, not 12pp, and the champion is not confidently wrong in FR.**

`chronos-2-V016` is the severe case. Pooled **69.11%, CI [65.54, 72.36]**, and
ten zones individually outside their own null: AT, BE, DE, EE, ES, FR, NL, PL,
PT, SI. **ES covers 50.8%** in a band labelled p10–p90.

---

## 4. What survives out-of-sample

Four calibrations, each fitted only on vintages strictly earlier than the ones
it is scored on, over three expanding-origin folds. `none` is the status quo.

`chronos-2-V010` — mean absolute per-zone coverage error, pp (lower is better):

| variant | fold 1 | fold 2 | fold 3 | pooled coverage f1/f2/f3 |
|---|---:|---:|---:|---|
| none | 9.87 | 9.93 | 10.19 | 72.7 / 75.3 / 82.4% |
| **pooled** | **9.88** | **9.48** | **10.25** | 72.0 / 77.9 / 84.7% |
| per_zone | 13.87 | 9.74 | 12.91 | 67.8 / 77.6 / 83.3% |
| shrink 0.5 | 11.56 | 9.37 | 11.83 | 69.8 / 78.2 / 85.1% |

`chronos-2-V016`:

| variant | fold 1 | fold 2 | fold 3 | pooled coverage f1/f2/f3 |
|---|---:|---:|---:|---|
| none | 16.08 | 12.73 | 13.01 | 65.3 / 68.7 / 74.9% |
| **pooled** | **10.92** | **11.59** | **12.77** | **72.2 / 80.3 / 84.8%** |
| per_zone | 16.72 | 8.61 | 12.94 | 64.5 / 79.3 / 83.9% |
| shrink 0.5 | 12.08 | 9.88 | 12.18 | 69.1 / 80.3 / 84.8% |

**Per-zone loses, and it is not bad luck.** In every fold, a zone's coverage in
the fit window barely predicts its coverage in the next: Pearson r **+0.27,
−0.20, +0.06** for V010 (Spearman +0.26, −0.19, +0.03), with the largest
zone-level shift between adjacent windows at **21.7–27.3pp**. FR is the clean
illustration: 67.7% over the gate window, **85.1%** over the following
fortnight. There is nothing stable there to calibrate against. V016's
persistence is higher (r 0.37/0.39/0.45) only because a *fleet-wide* narrowing
sits under every zone, which is exactly the component the pooled fit captures.

**Pooled is a clear win for V016 and roughly neutral for V010.** V016 improves
in all three folds on both the pooled level and the per-zone error. V010's
pooled level moves the right way in two folds of three and its per-zone error is
flat — consistent with a ~4pp pooled defect that a 20-vintage fit can only
partly resolve. **Pinball loss is unmoved in every fold and every variant**
(V010 360.05 → 359.98 / 348.02 → 346.66 / 283.94 → 285.22 MW), so nothing here
buys coverage by wrecking the proper score.

**A note on why the gate's per-zone screen cannot be reached.** Recentring each
zone's bootstrap draws on exactly 80% and reading off how often the measurement
lands inside [75, 85]: a **perfectly calibrated** band on 20 vintages would put
an expected **13.7 of 19** zones in band, with P(all 19) = **0.0019**. FR's own
ceiling is 0.54. This is a property of window length, not of calibration.
**It is reported here and handed to ABL-649; it is not acted on, and this fix is
not a gate candidate.**

---

## 5. Before and after, on held-out vintages

Multipliers fitted on vintages 2026-08-05..08-18, scored on 2026-08-19..09-01
(14 vintages, 6,138 rows). `s_lo` / `s_hi` scale the lower and upper half-widths
about `q50`.

### `chronos-2-V010` — s_lo 0.9895, s_hi 1.0597

Pooled **76.80% [71.98, 81.30] → 77.73% [72.87, 82.28]**.
Mean width **2,567 → 2,631 MW (+2.5%)**. Pinball 325.56 → 325.50 MW.
Zones in [75, 85]: **9 → 11**.

| zone | cov before | cov after | Δpp | width before | width after | Δwidth | width/MAE after |
|---|---:|---:|---:|---:|---:|---:|---:|
| AT | 82.7 | 84.3 | +1.6 | 2,251 | 2,306 | +2.5% | 3.38 |
| BE | 70.2 | 70.5 | +0.3 | 2,861 | 2,927 | +2.3% | 2.56 |
| BG | 77.2 | 78.2 | +1.0 | 1,136 | 1,163 | +2.4% | 3.27 |
| CZ | 75.9 | 76.2 | +0.3 | 1,146 | 1,173 | +2.4% | 3.03 |
| DE | 72.4 | 75.6 | +3.2 | 12,530 | 12,842 | +2.5% | 3.03 |
| EE | 87.5 | 87.2 | −0.3 | 355 | 366 | +3.0% | 4.47 |
| ES | 55.8 | 57.4 | +1.6 | 2,583 | 2,648 | +2.5% | 2.21 |
| FI | 73.7 | 75.3 | +1.6 | 2,119 | 2,178 | +2.8% | 3.06 |
| FR | 85.1 | 86.0 | +0.9 | 7,256 | 7,426 | +2.4% | 4.19 |
| HR | 74.4 | 74.4 | 0.0 | 1,113 | 1,140 | +2.4% | 3.10 |
| HU | 91.0 | 92.0 | +1.0 | 1,787 | 1,830 | +2.4% | 4.08 |
| LT | 81.2 | 82.1 | +0.9 | 796 | 816 | +2.6% | 3.19 |
| LV | 77.9 | 78.5 | +0.6 | 383 | 393 | +2.6% | 3.06 |
| NL | 76.5 | 76.5 | 0.0 | 5,323 | 5,448 | +2.3% | 3.08 |
| PL | 75.3 | 76.0 | +0.7 | 2,968 | 3,046 | +2.6% | 3.23 |
| PT | 67.9 | 70.2 | +2.3 | 1,794 | 1,839 | +2.5% | 2.67 |
| RO | 70.7 | 71.3 | +0.6 | 1,739 | 1,790 | +2.9% | 3.16 |
| SI | 84.4 | 85.0 | +0.6 | 338 | 346 | +2.5% | 3.49 |
| SK | 77.5 | 78.7 | +1.2 | 604 | 619 | +2.6% | 2.83 |

### `chronos-2-V016` — s_lo 1.1906, s_hi 1.2483

Pooled **69.94% [65.22, 74.25] → 78.97% [75.27, 82.50]**.
Mean width **2,192 → 2,673 MW (+22.0%)**. Pinball 333.09 → 330.58 MW (improves).
Zones in [75, 85]: 6 → 6 — the pooled level is fixed, the per-zone spread is not,
and per §4 nothing fitted on this data can fix the spread.

| zone | cov before | cov after | Δpp | width before | width after | width/MAE after |
|---|---:|---:|---:|---:|---:|---:|
| AT | 75.3 | 83.0 | +7.7 | 1,860 | 2,268 | 3.40 |
| BE | 51.3 | 58.7 | +7.4 | 1,919 | 2,338 | 2.05 |
| BG | 77.2 | 87.5 | +10.3 | 1,136 | 1,385 | 3.89 |
| CZ | 64.6 | 73.2 | +8.6 | 944 | 1,150 | 2.88 |
| DE | 69.9 | 79.5 | +9.6 | 11,746 | 14,326 | 3.37 |
| EE | 81.0 | 89.9 | +8.9 | 326 | 399 | 4.63 |
| ES | 32.1 | 42.0 | +9.9 | 2,071 | 2,526 | 1.74 |
| FI | 68.9 | 80.4 | +11.5 | 2,045 | 2,499 | 3.32 |
| FR | 80.4 | 85.1 | +4.7 | 5,766 | 7,026 | 4.01 |
| HR | 73.5 | 81.0 | +7.5 | 1,033 | 1,260 | 3.58 |
| HU | 90.1 | 98.4 | +8.3 | 1,762 | 2,148 | 4.88 |
| LT | 81.2 | 88.1 | +6.9 | 796 | 971 | 3.80 |
| LV | 69.2 | 76.3 | +7.1 | 343 | 419 | 3.30 |
| NL | 53.6 | 61.6 | +8.0 | 3,245 | 3,953 | 2.35 |
| PL | 70.2 | 78.8 | +8.6 | 2,644 | 3,228 | 3.39 |
| PT | 66.7 | 74.7 | +8.0 | 1,715 | 2,092 | 2.99 |
| RO | 70.7 | 85.6 | +14.9 | 1,739 | 2,127 | 3.76 |
| SI | 74.3 | 85.3 | +11.0 | 297 | 363 | 3.31 |
| SK | 76.9 | 89.2 | +12.3 | 610 | 745 | 3.38 |

### Is the calibrated band still decision-useful?

Asked as the wake comment framed it: does this reach 80% by becoming useless?

**For V010, no — the band barely moves.** +2.5% width. `width/MAE` goes from a
median of 3.03 to 3.10; the widest zone (EE) goes 4.34 → 4.47. A reader who
found the band usable yesterday will not notice the change.

**For V016, the width cost is real but the band is still narrower than the
champion's** on the same rows: 2,673 MW against V010's 2,631 MW, and its pinball
loss *improves*. In relative terms the median `width/MAE` goes 2.77 → 3.38.

**The honest caveat on both.** A p10–p90 band about three times the model's own
MAE is wide in absolute terms — a DE band of ±6.4 GW around a mean |net
position| of 5.2 GW spans more than the quantity itself. That is a property of a
D+2 net-position forecast with WAPE near 50%, not something this calibration
introduced (it was already 2.6–4.0 × MAE before). **Making it honest makes it
slightly wider; it does not make it useless, and a narrower band that is wrong
is worse.** If the Board wants a band a trader can act on, the lever is forecast
accuracy, not the interval.

---

## 6. The change

`src/quantile_calibration.py`, applied at both quantile write paths
(`scripts/forecast_chronos2.py` for the champion,
`scripts/forecast_challengers.py` for V016), registered in
`experiments/net_position_quantile_calibration.json`:

```
q'_t = q50 - s_lo * (q50 - q_t)   for t < 0.5
q'_t = q50 + s_hi * (q_t - q50)   for t > 0.5
q'_50 = q50
```

Split-conformal: `s_lo` is the (n+1)-corrected empirical quantile of the
normalised lower deviation `(q50 - a) / (q50 - q10)` at level 1 − α, so a
fraction α of exchangeable rows fall below the calibrated p10. Two one-sided
fits rather than one symmetric one, because V010's PIT is *shifted* as well as
narrow (q50 at empirical 51.8%) and a symmetric widening would reach 80% total
with unbalanced tails.

Registered multipliers, fitted on all 29 post-fix vintages (2026-08-05..09-01):

| model | s_lo total | s_hi total | s_lo applied | s_hi applied | vintages |
|---|---:|---:|---:|---:|---:|
| `chronos-2-V010` | 1.0722 | 1.0091 | 1.0722 | 1.0091 | 29 |
| `chronos-2-V016` | 1.3138 | 1.1623 | **1.2253** | **1.1518** | 26 |

**V016's applied multiplier is an increment, not a total.** Its band is an
affine image of the champion's, so its half-widths are exactly proportional to
V010's and it inherits the champion's widening for free. `load_registry` refuses
a registration where `s_applied × upstream ≠ s_total`, so a future change to
V010's number cannot silently double-widen V016.

**The multipliers are noisy at this sample size and the registration says so.**
Fitted on the first fortnight V010 reads 0.9895 / 1.0597; on all 29 vintages,
1.0722 / 1.0091. Both are small; the registered pair implies about +4% width
rather than the +2.5% measured out-of-sample above. Re-deriving them is
`scripts/abl650_band_calibration.py --write-registration`, and they should be
re-derived once there are meaningfully more post-fix vintages.

### Invariants, and how each is held

| invariant | held by |
|---|---|
| the served point forecast is bit-identical | `q50` is a fixed point of the map, and `forecasts.forecast_value` is never touched. Measured on 6,138 held-out rows per model: **max \|Δ\| = 0.0 MW, every row byte-equal**, both for the point column and for `q50` |
| no new quantile crossings | the map is increasing either side of `q50` with non-negative multipliers. **0 crossings** after calibration on the same rows |
| stored history is never rewritten | the calibration runs at the write path on new rows only, mirroring the solar-clamp contract |
| only pooled ships | `load_registry` raises `CalibrationRegistrationError` on any other `mode` |
| a model with no band is left alone | `baseline-V012` and `xgboost-V014` are unregistered; an unregistered model serves what its head emitted |
| a replay is not calibrated | `reconstruct_v010_vintages.py` and `apply_v016_to_vintages.py` are exempt with reasons — a replay must reproduce the row it replays. The serve-faithful attestation reads point rows only, which this change cannot move |
| a new writer cannot forget | a static sweep fails any file that reaches `forecast_quantiles` — either by issuing `INSERT OR REPLACE INTO forecast_quantiles` itself **or by calling `save_quantile_forecasts`** — and neither imports `src.quantile_calibration` nor is exempt with a reason |

Tests: `tests/test_abl650_quantile_calibration.py` (18). Full suite
**1,717 passed, 1 skipped** under `.venv` (3.14.3).

---

## 7. What this does not do

- **It is not a promotion and not a gate candidate.** Criterion 6 is not
  reachable by any calibration at this window length (§4); ABL-649 owns that.
- **It does not touch `energy-dashboard-frontend`.** No client change is needed:
  the caption already names the levels it draws, and after this the levels mean
  what they say. If the Board would rather the caption also carried the measured
  coverage, that is a separate, scoped Founding Engineer issue — say so and it
  will be filed.
- **It does not change what we forecast.** Verified numerically, not assumed.
- **It does not deploy.** The change is code plus a registration; it takes effect
  on the next scheduled run of the two quantile-emitting models, and shipping it
  is the Founding Engineer's and Deployment Engineer's call.

---

## 8. Revert runbook

Added after the PR #107 merge review, which executed the revert rather than
reading it and found that it is **serving-exact and not test-clean**. Merged at
`3419031`.

### The trigger, as pre-registered on ABL-674

Over the **first 10 post-merge vintages**, revert if either holds:

- the champion's pooled 10–90 coverage measured on `gate_scope` lands **outside
  [70, 90]%**; or
- **any zone's mean band width exceeds 5× that zone's MAE**.

Read the coverage against a **vintage-block** interval, never a row count: 10
vintages is 10 blocks of 24 hours, not 240 trials (§3).

### The revert

```
rm experiments/net_position_quantile_calibration.json
```

Nothing else. `load_registry` reads a missing file as "no calibration", logs
`no quantile-calibration registration at <path>; every band is served as
emitted`, and every model then serves the band its head emitted. No code change
is needed and no stored row is touched — the calibration only ever ran on new
rows, so already-served vintages are unaffected either way.

### The five tests that go red, by design

They pin the *shipped* registration, so deleting it is supposed to fail them.
They are **not** made to skip on a missing registry: an accidental deletion has
to stay loud. Executed on a throwaway worktree at the branch head — **5 failed,
26 passed** across the two affected files:

| test | why it reds |
|---|---|
| `test_abl650_quantile_calibration.py::test_the_shipped_registration_loads_and_composes` | there is no shipped registration to load |
| `…::test_only_the_two_models_that_emit_quantiles_are_registered` | `load_registry()` returns `{}`, not the two models |
| `…::test_calibrate_quantile_dict_applies_the_registered_multipliers` | no multipliers to apply |
| `…::test_the_registry_path_resolves_inside_the_repo` | asserts `REGISTRY_PATH.exists()` |
| `test_challenger_rail.py::test_v016_applies_the_fit_and_keeps_quantiles_ordered` | `KeyError: 'chronos-2-V016'` reading the spec |

**Any other failure means the revert misfired.** The list is derived, not
remembered: `test_the_deregistration_redlist_covers_every_test_that_reads_the_registry`
walks every test that reads the default registry and fails if one is missing
from `DEREGISTRATION_REDLIST` / `DEREGISTRATION_STILL_GREEN`, so a test added
later cannot surprise whoever runs the revert.

### What must still be green after the revert — the serving contract

If any of these reds, the revert did *not* restore the previous behaviour:

- `test_a_missing_registry_means_no_calibration_not_an_error` — a missing file
  is "no calibration", not a crash
- `test_an_unregistered_model_serves_the_band_it_emitted` — pass-through is exact
- `test_v016_passthrough_country_reproduces_the_champion` — the challenger rail
  still reproduces the champion
- `test_the_serving_entry_points_call_the_calibration`,
  `test_every_forecast_quantiles_writer_calibrates_or_says_why_not`,
  `test_the_sweep_sees_the_champions_own_serving_path` — the write paths still
  route through the calibration, so re-registering later needs no code change

```
.venv\Scripts\python.exe -m pytest tests/test_abl650_quantile_calibration.py \
    tests/test_challenger_rail.py -v
```

### To land the revert on `main` rather than run it locally

Delete the registration **and** the five pins in the same commit, and say in the
message that the deregistration is deliberate. Do not weaken the pins to make
the suite green while the registration is still shipped.
