# ABL-65 — Net-position residual/bias correction layer: recommendation

**Verdict: do not build one.** Measured serve-faithfully, no correction shape
beats the uncorrected champion, and the largest gain any shape could deliver
*even fitted with perfect hindsight* is a 0.01% MAE reduction on a per-country
offset and 0.86% on an hour-of-day table.

This is the measured negative ABL-65 explicitly asked for if that is where the
evidence landed. It is not a failure to find a correction; it is a finding about
what a D+2 product can read.

**Author:** Forecasting Scientist · **Date:** 2026-08-12 · Read-only against the
replica and the reconstruction sidecar; nothing written to any shared database,
no serving path touched.

---

## 1. The one number that decides it

The champion is a D+2 product. The run fires **06:00Z on day D**, net position is
day-ahead published so actuals reach **D 21:00**, and the target hours are
**D+2 00:00–23:00**. So the freshest residual any correction may read is
**27 to 50 hours older than the hour it is correcting**
(`as_of_for_vintage`, pinned in `test_serve_leads_match_the_real_vintage_geometry`).

This issue's headline candidate is residual AR: "lag-1 is 0.75–0.97 (median 0.88)
in every one of the 19 zones." That reproduces — but it is measured between
**adjacent** hours, which the run never observes together. Measured on the
198-day reconstruction, **n = 4,752 residual hours per country**:

| lag | 1 | 24 | **27** | 36 | **48** | **50** | 72 | 168 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| median ACF | **0.915** | 0.310 | **0.182** | 0.006 | **−0.010** | **−0.026** | −0.019 | 0.007 |

Lags 27–50 are the only ones a D+2 correction can read. The signal is gone by
lag 36. At lag 48 — which is exactly *same hour, two days ago*, the most natural
predictor available — the median correlation is **−0.010**, and it is negative in
12 of 19 zones.

**And `phi ** lead` is not a usable stand-in for the measurement.** V016's AR term
extrapolates the lag-1 coefficient; that is wrong in both directions, not merely
small. BG: `phi**27` = 0.009 against a measured 0.182 (20× under). FI: 0.369
against 0.108 (3.4× over). At the far end of the horizon the AR(1) prediction is
positive in all 19 zones (`phi**50` = 0.0002 … 0.158) where the measurement at
lag 48 is negative in 12 of 19.

The same question at the **day** level, which is the shape RO needs:

| | day-bias corr, lag 1 day | day-bias corr, **lag 2 days** |
|---|---:|---:|
| median over 19 zones | **+0.384** | **−0.059** |
| range | +0.241 … +0.504 | −0.231 … +0.118 |
| sign | positive in 19/19 | **negative in 13/19** |

Yesterday's bias predicts today's. The run cannot see yesterday — it sees the day
before yesterday, and that predicts **nothing**. The two-day gap is not a
detail of this product; it is the product.

## 2. The premise's per-country shares are a small-sample artifact

The corrected decomposition in the issue (median static bias 3.1%, exceeding 10%
of MSE in DE/FI/FR/NL/PT/RO) is fitted on 166 pairs per country over 6 days. The
Founding Engineer flagged the shares as in-sample upper bounds. Measured on
198 days instead of 6, at 4,752 pairs per country, they collapse:

| | FR | RO | PT | NL | DE | FI | median (19) |
|---|---:|---:|---:|---:|---:|---:|---:|
| static bias, 6-day cohort | 39.2% | 22.4% | 19.0% | 16.3% | 13.0% | 11.4% | 3.10% |
| static bias, **198 days** | **0.49%** | **1.07%** | **0.01%** | **0.10%** | **0.20%** | **0.74%** | **0.28%** |

Every one of the six zones the issue identified as offset-justified has
essentially no persistent bias. Diurnal moves the same way: median 8.20% → **1.41%**
(HR 25.1% → 0.61%). Residual is **96.9%** of MSE at 198 days.

**Two checks, because "the bias is recent, and 198 days dilutes it" is the fair
objection.**

*Check A — apparent bias grows as the window shrinks*, which is the signature of
a noise floor rather than a recent regime:

| recon window | 198d | 60d | 30d | 14d |
|---|---:|---:|---:|---:|
| median static-bias share | 0.28% | 0.75% | 1.02% | 3.58% |

*Check B — quantify the floor.* The error is strongly autocorrelated, so 166
hourly points do not hold 166 independent observations:
`n_eff = n(1−φ)/(1+φ)` gives **3.1 to 14.5** effective points per country. Under a
**true bias of exactly zero**, the expected static-bias share is ≈ `1/n_eff`:

| | median | range |
|---|---:|---:|
| noise floor, `1/n_eff` at n=166 | **13.5%** | 6.9% – 32.7% |
| observed 6-day static-bias share | **3.1%** | 0.0% – 39.2% |

**The observed median sits well below what pure noise produces.** Only FR, PT, RO
and SI exceed their own floor — and at 198 days those four measure 0.49%, 0.01%,
1.07% and 0.00%.

## 3. The ceiling, and a maximum-power holdout

To rule out "the rolling window was too short", fit on 99 days and score the next
99 — the most estimation power this data can give — and compare against an oracle
fitted on the scored data itself:

| MAE reduction | median | range | zones helped |
|---|---:|---:|---:|
| **oracle** per-country offset (in-sample, upper bound) | **0.01%** | −0.19% … +2.47% | — |
| **oracle** hour-of-day table (in-sample, upper bound) | **0.86%** | −0.14% … +2.76% | — |
| holdout offset (99d fit → 99d score) | **−0.13%** | −7.15% … +0.63% | 6/19, all < 0.65% |
| holdout hour-of-day table | **−0.80%** | −7.40% … +1.48% | 3/19 |

A *perfect* per-country constant, chosen with hindsight, is worth one hundredth
of one percent. There is no estimation problem to solve here; there is nothing
to estimate.

The hour-of-day profile also fails to reproduce: split-half correlation of the
profile is **negative in 7 of 19 zones** (EE −0.750, NL −0.716, AT −0.355,
LV −0.337, FR −0.284, BE −0.273, HU −0.002), median +0.345. Only SK (0.862),
DE (0.718), CZ (0.646), ES (0.611) and LT (0.591) carry a reproducible one.

## 4. Serve-faithful backtest, all shapes

Rolling, vintage-ordered, expanding history; every shape reads only residuals
from hours before `as_of` produced by vintages that had already run. No-lookahead
is structural and pinned by `test_no_lookahead` (appending a 50,000 MW future
changes no output, bit for bit).

**Reconstruction cohort — 198 vintages, 19 zones, 90,288 pairs per shape:**

| shape | median MAE vs uncorrected | zones improved | beats ensemble |
|---|---:|---:|---:|
| **uncorrected** | **0.00%** | — | **18/19** |
| level_ar_fitted | −0.20% | 1/19 | 18/19 |
| lead_ar_28d | −1.06% | 0/19 | 18/19 |
| lead_ar_7d | −2.26% | 0/19 | 17/19 |
| offset_28d | −2.71% | 0/19 | 17/19 |
| diurnal_28d_shrunk | −2.71% | 0/19 | 17/19 |
| offset_14d | −3.69% | 0/19 | 16/19 |
| offset_7d | −6.44% | 0/19 | 15/19 |
| diurnal_7d | −9.94% | 0/19 | 11/19 |
| offset_3d | −13.99% | 0/19 | 9/19 |

The ordering is the tell: the more a shape shrinks toward doing nothing (longer
window, heavier shrinkage, `level_ar` with a fitted φ that lands near zero), the
closer it gets to identity — and identity wins. That is what "the true correction
is zero" looks like in a table.

**Live post-fix cohort — 7 vintages / 6 run-days, 166 pairs per country, 3,154
total.** The harness reproduces the gate exactly: uncorrected **17/19, median
skill +11.83%** vs the persistence+climatology ensemble. Every shape is worse:

| shape | beats ensemble | median skill | median MAE vs uncorrected |
|---|---:|---:|---:|
| **uncorrected** | **17/19** | **+11.83%** | **0.00%** |
| level_ar_fitted | 13/19 | +7.21% | −0.88% |
| lead_ar_7d | 11/19 | +9.13% | −6.86% |
| offset_28d | 3/19 | −66.36% | −90.08% |
| offset_7d | 4/19 | −81.68% | −106.80% |
| diurnal_7d | 2/19 | −86.35% | −110.48% |

The catastrophic rows carry a **confound worth naming rather than hiding**: a
correction deployed at the fix date reads *pre-fix* residual history, and the
pre-fix vintages ran the zero-padded context. That is an honest cold start, not a
strawman — but it is a transient, so the structural verdict rests on the
reconstruction cohort in §3–§4, not on these numbers. The live cohort cannot be
run warm: with post-fix vintages starting 08-05 and `MIN_HISTORY_DAYS = 3`, the
first warm-startable vintage targets 08-14, whose actuals do not exist yet.

**Even cherry-picking the best shape per country in-sample** — not a legitimate
procedure, run only to bound the answer — reaches **17/19, median +12.8%**. It
does not beat 17/19 either. The acceptance bar ("materially more countries than
17/19") is not met by any shape, including one allowed to cheat.

**Live corroboration from a correction that already exists.** `chronos-2-V016` is
this layer, already built and shadow-serving since 2026-08-07: per-country affine
plus AR(1), fitted on five months with strict refusal guards. Over the common
4-vintage window (874 pairs, ~46/country — underpowered, corroborative only) its
median MAE change against V010 is **+0.00%**, improving 8 zones and hurting 8.

## 5. RO — the named sub-question

**Recommendation: explicit exclusion. Not a per-vintage level correction.**

RO's day-level bias does swing as reported (198-day sd **476.5 MW** against
mean |actual| 688.3 MW). But it swings *across the gap the run must bridge*:

| RO day-bias autocorrelation | lag 1d | **lag 2d** | lag 3d | lag 7d |
|---|---:|---:|---:|---:|
| | +0.292 | **−0.171** | −0.117 | +0.000 |

It is **negative** at exactly the lag a per-vintage correction would have to use.
A level carried from the freshest observable day pushes systematically the wrong
way — and this is not marginal:

| carry strength | RO day-bias sd | change |
|---|---:|---:|
| none | 478.3 MW | — |
| φ = 0.25 | 512.5 MW | **−7.2%** |
| φ = 0.50 | 570.3 MW | **−19.2%** |
| φ = 1.00 | 732.4 MW | **−53.1%** |

Fleet-wide the same test at φ = 0.25 improves **0 of 19 zones** (median −4.50%).
So this is not RO being special; RO is where the harm is easiest to see.

*Caveat, stated because it bears on this zone specifically:* RO is one of three
zones whose reconstruction is not serve-parity-verified (LT 38.8%, **RO 5.9%**,
BG 1.4% from the as-served 2026-08-06 vintage). A 5.9% amplitude deviation does
not plausibly flip a +0.29 lag-1 correlation to −0.17 at lag 2, and the live
cohort agrees directionally (RO's best shape scores −1.4% skill against
uncorrected +0.5%). ABL-280's confirmatory re-read at ~14 vintages is not
duplicated here.

## 6. What I recommend

1. **Build no correction layer for net position.** No offset, no diurnal table,
   no AR term, no rescaling. There is no issue for the Founding Engineer to
   receive from this one.
2. **Exclude RO from any future correction** if one is ever revisited, and treat
   its swing as a model/input problem — it is anti-correlated at the serve gap.
3. **Retire the AR term from V016**, or state its measured carry in the report
   beside it. `phi ** lead` is not the decay this residual follows.
   V016's own docstring already says the term is bounded by the horizon; the
   measurement here is that it is also *mis-signed* over the back half of the
   horizon in 12 of 19 zones. A note for whoever owns V016, not a change made here.
4. **Do not read a 6-day per-country decomposition as a design input again.**
   With `n_eff` of 3–15, the static-bias share it reports is mostly its own noise
   floor. Where a decomposition drives a build/no-build decision, report
   `1/n_eff` beside it.
5. **The residual is 96.9% of MSE and none of it is reachable from the past.**
   If net-position error is to come down, it comes from inputs and model — the
   group-B correlation problem (9 zones below corr 0.80, NL worst at 0.298) —
   not from a layer on top of the output. That is ABL-66/V014/V015 territory.

## 7. Protocol, caveats, contamination

- **Windows.** Reconstruction: targets 2026-01-21 … 2026-08-06, 198 daily
  vintages, 4,752 pairs/country, 19 gate zones (LU/GR excluded by name).
  Live: vintages ≥ `FIX_DEPLOYED_UTC` (2026-08-04 14:29), targets 2026-08-07 …
  2026-08-12 21:00, 166 pairs/country, 3,154 total.
- **Out-of-sample** everywhere except the two rows explicitly labelled *oracle*
  in §3, which are in-sample upper bounds and are not results.
- **Baseline** is the same serve-faithful persistence+climatology ensemble the
  gate reads, built by the eval module's own `baseline_predictions` rather than
  reimplemented, so this study and the report cannot disagree about it.
- **Per-country throughout.** No pooled statistic appears in any verdict; pooled
  slope 0.897 against a per-country median 0.528 is why.
- **Reconstruction is a reconstruction.** LT (38.8%), RO (5.9%) and BG (1.4%)
  deviate from the as-served 2026-08-06 vintage. Their structural numbers stand;
  their magnitudes carry that caveat.
- **Contamination.** ABL-67 (fabricated `net_position` rows): GR and LU are
  excluded from the gate set by name, IE is not in it, and the 216 fabricated
  rows were deleted 2026-08-11 — the 19 scored zones are untouched.
  ABL-109/ABL-111 (zero-as-missing load) are `energy_load` defects and do not
  reach `net_position`. ABL-71: actuals are current to 2026-08-12 21:00, so no
  ingest lag truncates the scored window.
  Independently re-checked rather than assumed: over the full 198-day window,
  19 zones, 92,986 actual rows — **zero country-days under the ABL-35 1 MW
  degenerate floor**, and 156 isolated exact-zero hours (PL 154 over 77 days,
  worst day 6h of 24; PT 2 on one day), which is a signed quantity crossing zero,
  not the 22–24h whole-day signature of a forward-fill fabrication. The post-fix
  scored window contains **0** exact-zero actuals.

## 8. Reproducing this

```bash
cd energy-forecast
.venv\Scripts\python.exe scripts\abl65_correction_study.py --cohort recon \
    --start 2026-01-01 --end 2026-08-13 --out reports\abl_65\recon.json
.venv\Scripts\python.exe scripts\abl65_correction_study.py --cohort live \
    --out reports\abl_65\live.json
.venv\Scripts\python.exe -m pytest tests\test_residual_correction.py -q
```

`src/evaluation/residual_correction.py` holds the shapes and the serve-faithful
driver; `reports/abl_65/*.json` hold every per-country number quoted above.
