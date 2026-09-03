# ABL-651 — Intercept-only static bias correction, net position: measurement and result

**Issue:** ABL-651 · **Parent:** ABL-70 · **Author:** Forecasting Scientist
**Date:** 2026-09-03 · **Machine record:** `reports/abl_651_static_bias.json`
**Frozen coefficients:** `experiments/ABL651/static_bias_coefficients.json`

**Bottom line.** Re-measured on current data, **DE, FI and PL do not qualify for
a static intercept, and none of them is corrected.** One zone does: **SI**,
which was not on the authorising list. Its intercept, fitted on a window that
ends two target days before the scored window begins, cuts SI's MAE by **7.05%**
and its bias from **9.0% to 1.0%** of mean |net position| out of sample, with
per-zone slope, corr and sd_ratio unchanged to **1.1e-16**.

The fleet-level effect of that is **−0.040% pooled MAE** (829.01 → 828.68 MW).
It is a real, clean improvement in one small zone, not a material move in the
published number. **Nothing is served; the serve decision is the CEO's.**

---

## 1. Protocol

| | |
|---|---|
| Model corrected | `chronos-2-V010` (the served champion) |
| Correction | `corrected = forecast - intercept_mw`, per zone. One free parameter per zone, **no slope term** |
| Cohort | post-fix vintages only, `generated_at >= 2026-08-04 14:29` (`FIX_DEPLOYED_UTC`) — 28 vintages |
| Scored target days | **2026-08-07 .. 2026-09-02, 27 complete days**, 12,768 pairs, **672 per zone**, 19 gated zones |
| Day completeness | 2026-09-03 is **dropped**: only 210 of 456 zone-hours were published at the replica refresh. A partly-published trailing day changes *which* zones a day contributes and moves a per-zone bias for no modelling reason |
| Fit window | target days 2026-08-07 .. 2026-08-20 (14 days, 6,840 pairs), vintages 2026-08-05 .. 2026-08-18 |
| Buffer | target days 2026-08-21 .. 2026-08-22 (912 pairs), in **neither** window |
| Held-out window | vintages 2026-08-21 .. 2026-08-31 (11), target days 2026-08-23 .. 2026-09-02, 5,016 pairs, **264 per zone** |
| Interpreter | `C:\Code\able\energy-forecast\.venv\Scripts\python.exe` (3.14.3) |
| Databases | replica `energy_dashboard.db`, sidecar `forecasts_local.db`, **both opened read-only** (`file:...?mode=ro`, `uri=True`). Nothing was written to either |

**The buffer is the serve-faithfulness argument, not a cosmetic gap.** The last
actual the fit reads is `2026-08-20 23:00`; the first held-out vintage runs at
`2026-08-21 06:00` and its own publication cutoff is `2026-08-21 22:00`. So
every row the intercept was fitted on was published before the first vintage it
corrects was generated — asserted in the machine record as
`fit_is_observable_at_first_eval_vintage: true`, not left to the reader.

Conventions are the gate's own, taken from `src/evaluation/net_position.py` and
pinned by test: **bias = mean(forecast − actual)**; **slope = OLS of forecast on
actual**. This module cannot disagree with the gate about what a zone's bias is.

### The qualification test

Four tests, all measured on the fitting window. Each exists because a specific
failure mode would otherwise pass:

| test | bar | why |
|---|---|---|
| **material** | \|bias\| ≥ 5% of mean \|net position\| | the gate's own `GATE_BIAS_FRAC`, so "material" means here what it means in the criterion this correction exists to move |
| **sign agrees across halves** | sign(h1) = sign(h2) = sign(pooled) | a zone that changes sign has no static offset; fitting the pooled mean bakes in whichever half was larger |
| **magnitude agrees across halves** | each half ≥ 50% of the pooled bias **and** ≥ 5% on its own half | a large pooled bias built from one quiet half and one extreme half is a level excursion |
| **separated from zero** | \|t\| ≥ 2 on **target-day** mean biases | the correction changes expected out-of-sample MSE by `se²(1 − t²)`, so the **break-even is \|t\| = 1**. The bar is deliberately conservative *of a stated break-even*, and what that conservatism costs is reported in §4 |

Plus a coverage floor of **20 target days / 400 pairs**, and the ABL-31 refusal
for a degenerate (all-within-1 MW) forecast series.

**The independence unit is the target day, not the hour.** Hourly net-position
residuals are autocorrelated at 0.75–0.97 lag-1, so a t built on 672 hourly
errors overstates its evidence by roughly √24 — the `1/n_eff` mistake ABL-65 §2
named. Day-level bias autocorrelation is reported per zone beside every verdict
(`day_acf1` 0.01–0.60), and a Newey–West standard error (Bartlett, lag 3) is
reported beside the plain one because positive day-level autocorrelation makes
the plain t **overstate** its evidence — which matters only for a zone the test
lets through. For SI the two agree in direction and the robust value is
*larger* (t = +3.30, Newey–West +4.27).

---

## 2. Currency: the four zones re-measured, and the fifteen others

Every gated zone, post-fix window, 672 pairs each. Halves split by target day.

| zone | bias MW | % of mean \|NP\| | half 1 | half 2 | t | NW t | day sign frac | slope | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| AT | −34 | 2.7% | −4 (0.3%) | −65 (5.1%) | −0.35 | −0.39 | 0.52 | 0.44 | material, magnitude, \|t\| < 2 |
| BE | +309 | 8.1% | +91 (2.6%) | +527 (12.5%) | +2.03 | +1.70 | 0.56 | 0.25 | magnitude |
| BG | +42 | 4.8% | −17 (2.0%) | +101 (10.9%) | +0.95 | +0.74 | 0.59 | 0.74 | material, sign, magnitude, \|t\| < 2 |
| CZ | +49 | 4.1% | +17 (1.4%) | +80 (7.1%) | +1.04 | +1.08 | 0.59 | 0.86 | material, magnitude, \|t\| < 2 |
| **DE** | **−1,172** | **21.5%** | **−162 (3.0%)** | **−2,182 (39.9%)** | −1.69 | −1.30 | 0.67 | 0.55 | **magnitude, \|t\| < 2** |
| EE | −7 | 2.0% | −20 (6.5%) | +7 (2.1%) | −0.39 | −0.46 | 0.52 | 0.58 | material, sign, magnitude, \|t\| < 2 |
| ES | +35 | 1.3% | −62 (2.8%) | +132 (4.5%) | +0.35 | +0.28 | 0.44 | 0.73 | material, sign, magnitude, \|t\| < 2 |
| **FI** | **+175** | **18.3%** | **+172 (19.1%)** | **+178 (17.6%)** | **+1.34** | +1.24 | 0.74 | 0.32 | **\|t\| < 2** |
| FR | +513 | 9.2% | +841 (18.5%) | +186 (2.8%) | +1.17 | +0.78 | 0.63 | 0.40 | magnitude, \|t\| < 2 |
| HR | +120 | 14.3% | −26 (4.1%) | +267 (25.4%) | +2.29 | +1.65 | 0.67 | 0.57 | sign, magnitude |
| HU | −150 | 8.1% | −136 (6.4%) | −164 (10.6%) | −1.70 | −1.63 | 0.63 | 0.85 | \|t\| < 2 |
| LT | −24 | 6.8% | −31 (8.8%) | −18 (4.8%) | −0.55 | −0.57 | 0.48 | 0.37 | magnitude, \|t\| < 2 |
| LV | +11 | 3.2% | +20 (5.6%) | +3 (0.8%) | +0.49 | +0.44 | 0.52 | 0.44 | material, magnitude, \|t\| < 2 |
| NL | +417 | 18.2% | +12 (0.4%) | +823 (41.5%) | +1.38 | +1.18 | 0.63 | 0.07 | magnitude, \|t\| < 2 |
| **PL** | **−109** | **7.8%** | **−354 (24.5%)** | **+135 (10.0%)** | −0.68 | −0.73 | 0.52 | 0.44 | **sign, \|t\| < 2** |
| PT | −78 | 3.4% | −151 (6.9%) | −4 (0.2%) | −1.13 | −0.94 | 0.63 | 0.82 | material, magnitude, \|t\| < 2 |
| RO | −46 | 5.8% | −268 (36.1%) | +176 (21.3%) | −0.44 | −0.48 | 0.37 | 0.45 | sign, \|t\| < 2 |
| **SI** | **+49** | **10.3%** | **+49 (11.2%)** | **+48 (9.5%)** | **+3.30** | **+4.27** | **0.78** | 0.50 | **corrected** |
| SK | −82 | 19.1% | −17 (3.8%) | −147 (35.1%) | −2.77 | −2.31 | 0.67 | 0.90 | magnitude |

### DE — one-signed, large, and not an offset

DE's pooled bias is still −1,172 MW (21.5%), so the ABL-595 reading was not
wrong. What has changed is that a week more data makes the shape visible:
**−162 MW (3.0%) in the first half against −2,182 MW (39.9%) in the second.**
An intercept fitted on either half is close to the negative of what the other
half needs.

The mechanism is measurable, not a guess. For a forecast that regresses on the
actual as `f = a0 + slope·a`, the mean error over any window is
`a0 − (1 − slope)·mean(actual)` — so a shrunken forecast produces a
**window-dependent bias with no static offset present at all**. DE's own mean
net position moved **+3,799 MW** between the halves; at DE's measured slope of
**0.55** that alone predicts a bias change of **−1,708 MW** against an observed
**−2,019 MW**, i.e. **85% of the movement is shrinkage against a level that
moved.** This is the ABL-24 amplitude problem showing up in the bias column, and
an intercept is the wrong instrument for it — which is precisely why the
magnitude test exists.

### PL — no longer one-signed

ABL-595's −340 MW does not survive: **−354 MW (24.5%) in the first half,
+135 MW (10.0%) in the second.** The pooled −109 MW is the average of two
opposite regimes and describes neither. This reproduces ABL-65's structural
finding for the fleet, and PL is the zone where it now bites.

### FI — the honest near-miss

FI is the most reproducible bias in the fleet: **+172 MW (19.1%)** and
**+178 MW (17.6%)** across halves — the two agree to within 4%. It is material,
one-signed, and stable, and it is one-signed in **all four** models (§5). It
fails only `|t| ≥ 2`, at **t = +1.34** (Newey–West +1.24): FI's day-to-day bias
scatter is large enough that 27 target days cannot separate +175 MW from zero at
that bar. It is above the **break-even of |t| = 1**, so the expected
out-of-sample MSE effect of correcting it is positive.

**I have not corrected FI, and §4 shows what that costs.** Moving the bar after
seeing the held-out outcome would be selecting on the test set. FI is the first
zone to re-qualify as vintages accrue; at its current effect size it reaches
|t| = 2 at roughly 60 target days.

### Two observations I am not acting on here

- **DE and NL are deteriorating.** On the held-out window DE's bias is **46.8%**
  of mean |net position| (MAE 4,121 MW) and NL's **51.7%** (MAE 1,841 MW), at
  slopes 0.51 and 0.05. This is worse than the 23.5% / 6.0% ABL-595 measured on
  2026-08-28 and is a modelling problem (ABL-24 shrinkage), not a bias problem.
- **SK, BE and HR** all clear materiality and two of them clear |t| ≥ 2, and all
  three fail magnitude agreement on the same shape as DE — a bias concentrated
  in the second half.

---

## 3. The correction, and the invariance it was authorised on

Delivered set: **SI only**, intercept **+48.90 MW** fitted on all 27 target days
(`experiments/ABL651/static_bias_coefficients.json`). Every other zone is a
pass-through, with the measured reason recorded per zone.

**Out-of-sample validation.** Selection and coefficient both come from the fit
window alone (SI also qualifies there, at **+51.19 MW**); the numbers below are
the disjoint held-out window, 264 pairs over 11 target days:

| | before | after | change |
|---|---:|---:|---|
| MAE | 89.37 MW | **83.07 MW** | **−7.05%** |
| bias | +45.86 MW | −5.33 MW | 9.0% → **1.0%** of mean \|net position\| |
| RMSE | 125.91 MW | 117.38 MW | −6.8% |
| WAPE | 17.49% | 16.26% | −1.23 pp |
| **slope** | **0.483167** | **0.483167** | **unchanged** |

The coefficient is also stable across the two ways of fitting it: **+51.19 MW**
on the first 14 target days, **+48.90 MW** on all 27 — a 4.5% difference on a
zone whose bias is 10.3% of its mean flow.

**Slope invariance, measured across all 19 zones on the held-out window:**

| quantity | max \|Δ\| over 19 zones |
|---|---:|
| per-zone **slope** | **1.11e-16** |
| per-zone **corr** | 1.11e-16 |
| per-zone **sd_ratio** | 1.11e-16 |

That is floating-point noise, i.e. exact. This is the property the Board relied
on and it holds by construction — `cov(actual, forecast + c) = cov(actual,
forecast)` — but it is measured rather than asserted, and
`tests/test_static_bias.py` pins it at 1e-12 along with a regression guard that
fails if a scale term is ever added.

**One number that does move, stated so nobody finds it later.** The *pooled*
slope goes 0.7293607 → 0.7295817 (2.2e-4). Pooled slope regresses across zones
and so mixes country means; shifting one zone's mean moves it. The gate reads
**per-country** slope, which is invariant to 1.1e-16. `net_position.py` already
warns that the pooled slope is inflated by between-country variance; this is the
same caveat, not a new one.

---

## 4. What the qualification test is worth

Two counterfactuals on the same held-out rows, because a test that does not beat
the alternatives is bureaucracy.

| arm | zones corrected | improved | worsened | pooled MAE | vs champion |
|---|---:|---:|---:|---:|---:|
| champion, uncorrected | 0 | — | — | 829.01 MW | — |
| **delivered (SI)** | **1** | **1** | **0** | **828.68 MW** | **−0.040%** |
| test at a 14-day floor | 2 | 1 | 1 | 837.51 MW | +1.02% |
| **no test at all** | 19 | 6 | **13** | 838.26 MW | **+1.11%** |

**Correcting every zone with its own measured mean bias makes 13 of 19 zones
worse.** That is ABL-65's negative result reproduced on live post-fix data, and
it is the reason a per-zone test — rather than the authorising list — decides.

**The 14-day arm is where the coverage floor comes from.** Fitted on 14 target
days, the test admitted **RO** (−279 MW, 37.7% of mean |net position|,
one-signed across both 7-day halves, t = −2.03 — it passed all four tests). On
the held-out window that frozen intercept cost **+32.6% MAE** (515 → 683 MW) and
pushed RO's bias from 27.4% to **62.0%**. The same test over 27 days rejects RO
on sign disagreement (−268 / +176 MW). A half needs enough target days to be
able to disagree, so the floor is **ten target days per half** — a number
measured off this failure, not a convention.

Consequence, stated plainly: **at the coverage floor the evidence supports, the
post-fix cohort is not yet long enough to both fit and validate.** The 27 days
support the measurement; only 14 are left over for a fit if 13 are held back.
SI is delivered because it qualifies at *both* lengths and wins out of sample;
it is the only zone of which that is true.

**And the cost of the |t| ≥ 2 bar is visible in the same table.** In the no-test
arm, FI improves **−6.8%** and HU **−6.6%** on the held-out window — both are
above the break-even |t| = 1 and below the bar. That is evidence the bar is
leaving something on the table. It is **not** grounds to move the bar now, after
seeing the outcome; it is grounds to re-read FI and HU when the window supports
|t| ≥ 2 on its own.

---

## 5. The same measurement on the other three models

The issue's premise is that DE/FI/PL are one-signed "in every one of the four"
models. Re-measured over each model's own post-fix window (V010 27 target days,
V012/V016 25, V014 24), with half-signs shown as `h1h2`. Every zone and every
model: `reports/abl_651_cross_model_bias.json`.

| zone | V010 | V012 | V014 | V016 |
|---|---|---|---|---|
| DE | −1,172 (21.5%) `--` | −1,033 (19.0%) `+-` | −4,052 (74.6%) `--` | −1,254 (23.1%) `--` |
| FI | +175 (18.3%) `++` | +248 (25.0%) `++` | +191 (19.4%) `++` | +232 (23.4%) `++` |
| PL | −109 (7.8%) `-+` | −147 (10.0%) `-+` | −178 (12.3%) `--` | −131 (9.0%) `-+` |
| SI | +49 (10.3%) `++` | +35 (7.2%) `++` | +103 (20.7%) `++` | +65 (13.4%) `++` |

- **DE** stays one-signed in three of four and large in all four — so the sign
  claim holds; it is the *stability* claim that fails, in every model.
- **FI** is one-signed and material in **all four**, which is why it is a
  near-miss rather than a rejection.
- **PL** flips sign across halves in three of four.
- **SI** is one-signed and material in all four, and independently **qualifies**
  under the full test for V010, V014 and V016. A bias that survives four
  independently-built models is a property of the zone or its inputs, not a
  champion artifact — which is the strongest argument for correcting it and,
  equally, an argument that the underlying cause is worth finding.

---

## 6. Contamination and provenance

| issue | touches this window? | handling |
|---|---|---|
| **ABL-67** — fabricated `net_position` rows (GR, IE) | **No.** GR is excluded by name and has no rows in the window; IE is not a gated zone | exclusion is by name, not by symptom, so GR cannot re-enter on thin data |
| **ABL-71** — prod ingest stale, fixes undeployed | **Provenance risk only.** Replica refreshed 2026-09-03 17:09, actuals through 2026-09-04 21:00 | recorded, not certified: currency is not correctness |
| **ABL-111 / ABL-109** — zero-as-missing actual **load** | **No.** Net position is scored on the `net_position` table; no actual-load row is consumed | — |

Direct screen of the 12,312 scored actual hours (19 zones, 27 target days):
**0 nulls, 0 duplicated hours, 7 exact zeros** (all PL, isolated — a signed
quantity crossing zero, not a forward-fill run), **longest bit-identical run 4
hours** (ES), and **0 degenerate vintage-zone-days** under the ABL-31 1 MW
floor. Sidecar vs prod-pushed overlap max |Δ| = **0.000 MW**: the rows scored
are the rows served.

LU and GR are excluded by name throughout, with the gate's recorded reasons.

---

## 7. Recommendation

1. **Correct SI. Do not correct DE, FI or PL.** DE's bias is a shrinkage
   artifact against a moved level (85% explained), PL's has flipped sign, and
   FI's is real but inside its own noise at the stated bar.
2. **The serve decision is yours.** The measured effect on the published number
   is **−0.040% pooled MAE**, concentrated entirely in one small zone. Serving
   it is defensible; so is judging it too small to be worth a change to what we
   publish. Both readings are consistent with this evidence, and the choice is
   not mine.
3. **Nothing here is a gate candidate.** Per the issue, this is a fix to what we
   serve, independent of ABL-649. If SI is served, its bias criterion at the
   next gate read would move from failing to passing — that is a consequence to
   disclose at that read, not a promotion argument.
4. **Re-qualify before serving if more than 14 target days have accrued** since
   2026-09-02. Three of the four zones on the authorising list changed verdict
   in one week; a coefficient set is a property of its window.
5. **Re-read FI and HU** when the post-fix cohort reaches roughly 60 target days
   (~2026-10-05 at one vintage a day). Both sit above the break-even and below
   the bar today.
6. **DE and NL belong to ABL-24, not here.** At 46.8% and 51.7% bias with slopes
   0.51 and 0.05 on the held-out window, they are the amplitude problem, and no
   correction layer of this shape can reach them.

### On keeping the champion scoreable

The champion is untouched by construction: no V010 row is mutated anywhere, the
correction lives in a separate frozen coefficient file, and every table above
scores both series on identical rows. What I have **not** done is register a
`chronos-2-V017` model name and backfill corrected rows into the sidecar. That
is the parallel-model convention, and it is the right move the moment there is a
decision to serve — but standing it up now would create a shadow model that is a
pass-through in 18 of 19 zones and that nothing on the daily rail writes to,
which is a dead registry entry rather than evidence. It is a ten-minute
follow-up on a yes.

If the CEO decides to serve, exposing or defaulting the corrected series needs a
change to `server/src/config/forecastModels.ts`, which is the Founding
Engineer's file — I have not touched `energy-dashboard-frontend`, and that would
be a scoped follow-up issue, not part of this one.

---

## 8. Reproducing this

```bash
cd energy-forecast
.venv\Scripts\python.exe scripts\abl651_static_bias.py \
    --replica-db C:\Code\able\data\energy_dashboard.db \
    --sidecar-db C:\Code\able\data\forecasts_local.db \
    --json-out reports\abl_651_static_bias.json \
    --coefficients-out experiments\ABL651\static_bias_coefficients.json

# section 5: the same run per model, condensed into
# reports\abl_651_cross_model_bias.json
.venv\Scripts\python.exe scripts\abl651_static_bias.py --model baseline-V012 ... \
    --json-out <path>

.venv\Scripts\python.exe -m pytest tests\test_static_bias.py -q
```

Every figure quoted above was re-derived from the machine record and
string-compared at the decimal places printed here, cell by cell, rather than
read off by eye.

**Every number above:** cohort = post-fix vintages (`generated_at >=
2026-08-04 14:29`); scored target days 2026-08-07 .. 2026-09-02, 27 complete
days, 672 pairs per zone, 19 gated zones; held-out window 264 pairs per zone
over 11 target days; baseline = the uncorrected champion `chronos-2-V010` on
identical rows. **Out of sample throughout except the §2 currency table**, which
is a measurement of the window it describes and fits nothing, and the
`fit_window_in_sample` block in the machine record, which is labelled as such
and is not a result.
