# ABL-385 - seed variance across the served renewable pairs, and a registered decision margin

Generated 2026-08-13T14:32:51. Every number below comes from the sweep the frozen registration defines.

**Pre-registration provenance** (scope item 1, checked rather than asserted):

- Registration commit `f5c7136be3a9`, committed `2026-08-13T13:14:40+02:00`.
- Earliest fit in this sweep: `holdout_W1_catboost_cleaned.json` at `2026-08-13T11:18:46.422777+00:00`.
- **ORDERED - the registration was committed 4.1 min before the earliest fit in this sweep (holdout_W1_catboost_cleaned.json)**
- Working tree: clean - the config.json read here is byte-identical to the frozen commit.

Replica `C:\Code\able\data\energy_dashboard.db`, read-only. Interpreter: the rail (`.venv`, Python 3.14.3, xgboost 3.3.0).

## What was measured

- **280 cells**, each a (country, type, algorithm, arm, window) fitted at the **12 registered seeds** - 3360 fits in total.
- **14 served pairs** of the 14 on disk, over **6 of the 6 registered contiguous non-overlapping 30-day rolling-origin windows** (2026-02-13 .. 2026-08-11). No holdout row is scored twice.
- Solar is read on **daylight MAE**; every other type on **all-hours MAE**. Night is reported in MW and never as a percentage - its denominator is ~0.
- Every arm is a **refit** on the identically truncated window. The live artifacts are fitted through roughly today, so scoring them on a recent holdout would be in-sample and would flatter the incumbent.

**All numbers are out-of-sample** with respect to the fit frame, which ends strictly before each holdout starts.

**Determinism check: PASS.** The W6 primary invocation and the `abl-2023-01-01` ablation point carry identical arguments and were run twice; 16 arms compared, largest disagreement 0 MW. The CV estimate assumes the only thing moving is the seed, and this is what checks that.

## 1. The headline - the registered decision margin

For two arms A and B scored on the same holdout, each reported as the mean of k seeds, the delta method gives Var(g) ~ (c_A^2 + c_B^2) / k for the relative gap g. So a gap is readable at two-sided 95% only if

```
delta_min(k) = 1.96 * sqrt(c_A^2 + c_B^2) / sqrt(k)
```

with c the per-fit CV of each arm. **This is the number a future registration cites instead of a remembered noise floor.**

**The margin is itself an estimate, and it carries its own error bar.** A sd from n draws is chi-square distributed. At the 12 registered seeds a *single cell's* CV has a 95% interval of about -29%/+70% of its point estimate - so a one-window, 12-seed spread is not a number to hang a decision on either. Pooling the registered windows takes the interval to roughly -14%/+19%, and that is the quantitative reason this registration reads six rolling-origin windows rather than one. The per-pair intervals are in section 2. Quoting delta_min without them would repeat, one level up, exactly the mistake this issue was filed on.

The fleet percentile below carries a further uncertainty that is *not* in those intervals: it is a percentile over a modest number of units, and no parametric interval is claimed for it. Prefer a pair-specific CV where one exists.

### Solar - 16 (pair, algorithm, arm) units

Pooled per-fit CV: median 2.32%, p80 4.47%, **p90 5.43%**, max 6.08%.

| seeds k | delta_min at p90 | delta_min at p80 |
|---:|---:|---:|
| 1 | 15.06% | 12.40% |
| 3 | 8.69% | 7.16% |
| 5 | 6.73% | 5.54% |
| 10 | 4.76% | 3.92% |
| 12 | 4.35% | 3.58% |
| 20 | 3.37% | 2.77% |

To read a 5% gap on this stream takes **10 seeds**; a 10% gap takes **3**.

### Wind - 12 (pair, algorithm, arm) units

Pooled per-fit CV: median 2.15%, p80 2.59%, **p90 3.83%**, max 4.99%.

| seeds k | delta_min at p90 | delta_min at p80 |
|---:|---:|---:|
| 1 | 10.61% | 7.19% |
| 3 | 6.13% | 4.15% |
| 5 | 4.75% | 3.21% |
| 10 | 3.36% | 2.27% |
| 12 | 3.06% | 2.07% |
| 20 | 2.37% | 1.61% |

To read a 5% gap on this stream takes **5 seeds**; a 10% gap takes **2**.

### Biomass / hydro (served, never gated) - 8 (pair, algorithm, arm) units

Pooled per-fit CV: median 4.49%, p80 7.76%, **p90 9.56%**, max 12.19%.

| seeds k | delta_min at p90 | delta_min at p80 |
|---:|---:|---:|
| 1 | 26.49% | 21.52% |
| 3 | 15.30% | 12.43% |
| 5 | 11.85% | 9.63% |
| 10 | 8.38% | 6.81% |
| 12 | 7.65% | 6.21% |
| 20 | 5.92% | 4.81% |

To read a 5% gap on this stream takes **29 seeds**; a 10% gap takes **8**.

### Sensitivity: the registration admits two readings of this percentile

`estimator.pooling_across_windows` pools **per (pair, algorithm, arm)**; `estimator.fleet_value` then takes percentiles of **the per-(pair, algorithm)** pooled CV distribution. Solar is the only type with two arms, so under the per-arm reading every solar pair enters the percentile twice, from two correlated fits of the same pair. The frozen text supports both and this pack does not pick one silently:

| stream | p90 CV, per (pair, algorithm, arm) | p90 CV, arms collapsed (max) | delta_min(k=3), per-arm | delta_min(k=3), collapsed |
|---|---:|---:|---:|---:|
| solar (16 vs 8 units) | 5.43% | 5.29% | 8.69% | 8.47% |
| wind (12 vs 12 units) | 3.83% | 3.83% | 6.13% | 6.13% |
| other (8 vs 8 units) | 9.56% | 9.56% | 15.30% | 15.30% |

Wind and the never-gated pairs carry one arm each, so the two readings coincide there by construction and only solar moves. **Cite the larger of the two** where they differ: this issue exists because a margin was quoted too small, and the arms-collapsed figure rests on the smaller unit count.

## 2. Per-pair spread

The pair-specific CV is the one to cite when it exists; the fleet percentile above is for a pair this sweep did not measure.

| pair / algorithm / arm | CV (RMS over 6 windows) | 95% CI on that CV | CV (worst window) | delta_min at k=1 | at k=3 | at k=10 |
|---|---:|---:|---:|---:|---:|---:|
| BE/biomass/catboost/control | 12.19% | 10.42-14.69% | 24.57% | 33.8% | 19.5% | 10.7% |
| BE/biomass/xgboost/control | 8.43% | 7.21-10.16% | 13.07% | 23.4% | 13.5% | 7.4% |
| FR/biomass/catboost/control | 6.76% | 5.78-8.15% | 11.39% | 18.7% | 10.8% | 5.9% |
| FR/biomass/xgboost/control | 6.11% | 5.23-7.37% | 8.69% | 16.9% | 9.8% | 5.4% |
| DE/solar/catboost/control | 6.08% | 5.19-7.33% | 9.38% | 16.8% | 9.7% | 5.3% |
| DE/solar/catboost/geometry | 5.91% | 5.05-7.12% | 8.05% | 16.4% | 9.5% | 5.2% |
| BE/wind_offshore/catboost/control | 4.99% | 4.26-6.01% | 11.55% | 13.8% | 8.0% | 4.4% |
| AT/solar/catboost/geometry | 4.96% | 4.24-5.98% | 9.70% | 13.7% | 7.9% | 4.3% |
| AT/solar/catboost/control | 4.47% | 3.82-5.39% | 6.93% | 12.4% | 7.2% | 3.9% |
| BE/wind_onshore/catboost/control | 3.96% | 3.39-4.78% | 8.55% | 11.0% | 6.3% | 3.5% |
| FR/solar/xgboost/control | 3.19% | 2.72-3.84% | 6.55% | 8.8% | 5.1% | 2.8% |
| BE/hydro_total/xgboost/control | 2.86% | 2.44-3.45% | 4.07% | 7.9% | 4.6% | 2.5% |
| BE/wind_offshore/xgboost/control | 2.61% | 2.23-3.15% | 4.18% | 7.2% | 4.2% | 2.3% |
| FR/solar/xgboost/geometry | 2.56% | 2.19-3.09% | 3.60% | 7.1% | 4.1% | 2.2% |
| FR/wind_onshore/catboost/control | 2.50% | 2.14-3.02% | 4.70% | 6.9% | 4.0% | 2.2% |
| DE/solar/xgboost/control | 2.50% | 2.14-3.02% | 4.16% | 6.9% | 4.0% | 2.2% |
| BE/hydro_total/catboost/control | 2.45% | 2.10-2.96% | 4.88% | 6.8% | 3.9% | 2.2% |
| BE/wind_onshore/xgboost/control | 2.34% | 2.00-2.82% | 3.55% | 6.5% | 3.7% | 2.0% |
| BE/solar/catboost/control | 2.33% | 1.99-2.80% | 2.94% | 6.4% | 3.7% | 2.0% |
| FR/solar/catboost/control | 2.31% | 1.97-2.78% | 3.22% | 6.4% | 3.7% | 2.0% |
| AT/wind_onshore/xgboost/control | 2.25% | 1.92-2.71% | 3.43% | 6.2% | 3.6% | 2.0% |
| BE/solar/catboost/geometry | 2.22% | 1.90-2.68% | 2.83% | 6.2% | 3.6% | 1.9% |
| BE/solar/xgboost/control | 2.21% | 1.89-2.66% | 3.46% | 6.1% | 3.5% | 1.9% |
| FR/solar/catboost/geometry | 2.21% | 1.89-2.66% | 3.02% | 6.1% | 3.5% | 1.9% |
| DE/solar/xgboost/geometry | 2.12% | 1.81-2.56% | 3.17% | 5.9% | 3.4% | 1.9% |
| FR/wind_onshore/xgboost/control | 2.05% | 1.75-2.47% | 3.21% | 5.7% | 3.3% | 1.8% |
| DE/wind_onshore/catboost/control | 2.03% | 1.73-2.45% | 2.69% | 5.6% | 3.2% | 1.8% |
| AT/solar/xgboost/control | 1.88% | 1.61-2.27% | 2.65% | 5.2% | 3.0% | 1.6% |
| BE/solar/xgboost/geometry | 1.87% | 1.60-2.25% | 3.00% | 5.2% | 3.0% | 1.6% |
| AT/wind_onshore/catboost/control | 1.81% | 1.55-2.18% | 2.35% | 5.0% | 2.9% | 1.6% |
| DE/wind_onshore/xgboost/control | 1.80% | 1.54-2.17% | 2.10% | 5.0% | 2.9% | 1.6% |
| FR/wind_offshore/catboost/control | 1.52% | 1.30-1.84% | 2.08% | 4.2% | 2.4% | 1.3% |
| FR/wind_offshore/xgboost/control | 1.42% | 1.21-1.71% | 1.82% | 3.9% | 2.3% | 1.2% |
| FR/hydro_total/xgboost/control | 1.40% | 1.20-1.69% | 1.97% | 3.9% | 2.2% | 1.2% |
| AT/solar/xgboost/geometry | 1.38% | 1.18-1.67% | 1.63% | 3.8% | 2.2% | 1.2% |
| FR/hydro_total/catboost/control | 1.12% | 0.95-1.34% | 1.64% | 3.1% | 1.8% | 1.0% |

## 3. Window variance against seed variance

Scope item 3. The six windows sit at very different levels - solar MAE in February is not solar MAE in July - so the split is done on log MAE, where both components are relative and comparable. `sd_seed` is the spread from reseeding within one window; `sd_window` is the spread of the window means.

| pair / algorithm / arm | sd_seed (log) | sd_window (log) | seed share of variance |
|---|---:|---:|---:|
| BE/biomass/catboost/control | 0.1113 | 0.6441 | 2.9% |
| BE/biomass/xgboost/control | 0.0851 | 0.5837 | 2.1% |
| FR/biomass/catboost/control | 0.0685 | 0.4729 | 2.1% |
| FR/biomass/xgboost/control | 0.0600 | 0.5665 | 1.1% |
| DE/solar/catboost/control | 0.0590 | 0.5583 | 1.1% |
| DE/solar/catboost/geometry | 0.0579 | 0.5230 | 1.2% |
| BE/wind_offshore/catboost/control | 0.0485 | 0.2134 | 4.9% |
| AT/solar/catboost/geometry | 0.0473 | 0.5274 | 0.8% |
| AT/solar/catboost/control | 0.0442 | 0.5312 | 0.7% |
| BE/wind_onshore/catboost/control | 0.0394 | 0.2831 | 1.9% |
| FR/solar/xgboost/control | 0.0310 | 0.1820 | 2.8% |
| BE/hydro_total/xgboost/control | 0.0282 | 0.0987 | 7.6% |
| BE/wind_offshore/xgboost/control | 0.0262 | 0.2061 | 1.6% |
| FR/solar/xgboost/geometry | 0.0257 | 0.1780 | 2.0% |
| DE/solar/xgboost/control | 0.0249 | 0.4895 | 0.3% |
| FR/wind_onshore/catboost/control | 0.0244 | 0.3115 | 0.6% |
| BE/hydro_total/catboost/control | 0.0241 | 0.0975 | 5.8% |
| BE/wind_onshore/xgboost/control | 0.0233 | 0.3062 | 0.6% |
| BE/solar/catboost/control | 0.0232 | 0.1348 | 2.9% |
| FR/solar/catboost/control | 0.0230 | 0.1721 | 1.8% |
| AT/wind_onshore/xgboost/control | 0.0224 | 0.1040 | 4.4% |
| BE/solar/catboost/geometry | 0.0221 | 0.1555 | 2.0% |
| FR/solar/catboost/geometry | 0.0220 | 0.1557 | 2.0% |
| BE/solar/xgboost/control | 0.0219 | 0.1642 | 1.7% |
| DE/solar/xgboost/geometry | 0.0213 | 0.4232 | 0.3% |
| FR/wind_onshore/xgboost/control | 0.0203 | 0.2733 | 0.6% |
| DE/wind_onshore/catboost/control | 0.0203 | 0.2710 | 0.6% |
| AT/solar/xgboost/control | 0.0188 | 0.5425 | 0.1% |
| BE/solar/xgboost/geometry | 0.0187 | 0.1720 | 1.2% |
| AT/wind_onshore/catboost/control | 0.0182 | 0.1378 | 1.7% |
| DE/wind_onshore/xgboost/control | 0.0180 | 0.2484 | 0.5% |
| FR/wind_offshore/catboost/control | 0.0152 | 0.1928 | 0.6% |
| FR/wind_offshore/xgboost/control | 0.0142 | 0.1810 | 0.6% |
| FR/hydro_total/xgboost/control | 0.0140 | 0.4246 | 0.1% |
| AT/solar/xgboost/geometry | 0.0138 | 0.5289 | 0.1% |
| FR/hydro_total/catboost/control | 0.0112 | 0.4132 | 0.1% |

## 4. The independence assumption, measured

delta_min treats c_A and c_B as independent. That is exact for two different algorithms, whose RNG draws are unrelated by construction. For two arms of the *same* algorithm at matched seeds it had to be measured, and the solar control-vs-geometry cells are what measures it.

Fleet correlation across 48 matched cells (Fisher-z mean): **0.113**.

| pair / algorithm | correlation at matched seeds |
|---|---:|
| AT/catboost | 0.080 |
| AT/xgboost | 0.162 |
| BE/catboost | -0.129 |
| BE/xgboost | 0.172 |
| DE/catboost | -0.008 |
| DE/xgboost | 0.325 |
| FR/catboost | 0.081 |
| FR/xgboost | 0.204 |

Positive correlation means the independent margin is conservative: matched seeds move the two arms together and part of the noise cancels out of the gap. Near-zero means the independent margin is about right even within one algorithm.

## 5. The three pre-specified predictions

### P1 - HELD

*Registered statement:* DE CatBoost solar CV exceeds DE XGBoost solar CV on a majority of the six windows.

HELD - CatBoost CV is higher on 6 of 6 windows

| window | CatBoost CV | XGBoost CV | CatBoost higher |
|---|---:|---:|:--:|
| W1 | 3.21% | 2.15% | yes |
| W2 | 4.12% | 1.60% | yes |
| W3 | 5.58% | 3.17% | yes |
| W4 | 6.02% | 2.02% | yes |
| W5 | 7.06% | 1.67% | yes |
| W6 | 8.05% | 1.70% | yes |

### P2 - AMBIGUOUS

*Registered statement:* If DE CatBoost's instability is a short-fit artefact, CV falls as fit length rises in the W6 ablation. If it is a property of CatBoost on solar, CV is roughly flat in fit length.

AMBIGUOUS - the two countries disagree, or the movement falls between the registered thresholds

| country | n_train longest | CV longest | n_train shortest | CV shortest | ratio |
|---|---:|---:|---:|---:|---:|
| BE | 22,053 | 1.78% | 2,880 | 3.76% | 0.47 |
| FR | 30,359 | 2.47% | 2,808 | 4.53% | 0.55 |

### P3 - NOT HELD

*Registered statement:* If the instability is DE-specific rather than a fit-length effect, BE and FR CatBoost solar CV at a matched 134-day fit will be materially below DE's at its own 134-day fit.

NOT HELD - DE's CV is 1.25x the median of the other three at a matched fit, short of the registered factor of 2

At a matched fit start of 2026-03-01:

| country | n_train | CV | mean daylight MAE |
|---|---:|---:|---:|
| AT | 2,880 | 2.35% | 313.2 MW |
| BE | 2,880 | 3.76% | 584.6 MW |
| DE | 2,878 | 4.70% | 3,324.0 MW |
| FR | 2,808 | 4.53% | 1,743.5 MW |

## 6. Re-reading ABL-375's DE question under this margin

ABL-375 read holdout 2026-04-30 .. 2026-06-12 at k = 3 and observed a 4.5% relative gap favouring XGBoost.

- DE CatBoost geometry CV (pooled, 6 windows, 12 seeds): **5.91%**
- DE XGBoost geometry CV: **2.12%**
- delta_min(k=3) = **7.10%**

The observed gap does not reach that margin, so **AMBIGUOUS was correct**. Reading a 4.5% gap at 95% would have taken **8 seeds**, not 3.

Nothing here is a verdict on whether DE solar should move to XGBoost. This supplies the error bar; the call is ABL-375's registered question and the CEO's decision.

## 7. Contamination

- **ABL-337**: Physically impossible night solar actuals. Dropped from the fit and never from the score, identically on every arm, via --drop-impossible-night. Counted per (pair, window) in reports/abl_385_probe.json: AT 0, BE 0, DE 0-4, FR 430-517 rows in the fit frame. Solar only.
- **ABL-188**: The constant-run screen is applied by db.load_renewable_type_data to whatever table is read, so it is identical across arms and seeds by construction. It fires for BE/wind_offshore (319 rows) and BE/hydro_total (127 rows) in this scope.
- **ABL-332**: Sub-hourly rows are aggregated to hourly means at the read. FR pairs aggregate 74,688 sub-hourly rows to 31,596 hourly. Identical across arms and seeds.
- **ABL-67**: Net position only. Does not touch this scope.
- **ABL-109_ABL-111**: Load only. Does not touch this scope.
- **ABL-71**: Provenance caveat. Known wrong-write modes are load and net position; that is not proof renewable ingest is clean.
- **why_this_matters_less_here_than_usual**: Every quantity this issue reports is a spread *within* a fixed (pair, window, arm, algorithm) cell, across seeds only. Contamination that is identical across the seeds of a cell shifts the cell's level and cancels out of its CV. It is still recorded, because the fit-length and cross-window comparisons do not have that protection.

## 8. Boundaries

- Evidence only. No promotion, no serving change, no ingest change, no registry change.
- The replica is opened read-only. Nothing is written to it.
- No model artifact is written. models/ is untouched.
- This registration does not change any pre-registered gate. It supplies an error bar that future registrations may cite.
