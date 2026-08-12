# ABL-338 — solar non-negativity and solar geometry: evidence pack

**Verdict: the non-negativity constraint as specified degrades daylight accuracy,
so nothing was retrained and no artifact is proposed for promotion.** The
CEO's stop condition on ABL-338 was "if daylight accuracy degrades, say so and
stop"; the log-link constraint costs up to **+15.8%** daylight MAE (Tweedie) and
**+36.8%** (Poisson) against a like-for-like refit. The solar-geometry feature is
daylight-safe and lands here; the constraint lands as a reviewed, **unadopted**
capability.

Measured 2026-08-12 against the live replica `C:\Code\able\data\energy_dashboard.db`
(the run passes `ENERGY_DB_PATH` explicitly — the repo `.env` points at
`C:\Code\energy-data-gathering\energy_dashboard.db`, which is not the replica).
All fits under `.venv\Scripts\python.exe` (Python 3.14.3, xgboost 3.3.0).

---

## 1. The defect reproduces, and the mechanism is visible

Reconstructing the exact serve-time feature vectors for target **2026-08-14**
at `observation_as_of = 2026-08-12 06:00` (`RenewableFeatureBuilder` + the live
artifacts) gives night predictions of **AT -16.8 to -17.0 MW, BE -12.9 to -38.5,
DE +171.3 to +263.2, FR -3.4 to +79.7**. DE's night total is **1,524.2 MW**,
which reproduces ABL-337's independently measured `mw_removed_total` of 1,524 MW
to the decimal — the two reconstructions agree.

At every one of those hours the inputs are the same:

| | shortwave | direct | diffuse | lag_1d | lag_7d |
|---|---:|---:|---:|---:|---:|
| all night hours, all four countries | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

**Identical zero inputs, a non-zero country-specific output.** Nothing in the
25-name feature vector distinguishes "0 W/m² because the sun is down" from
"0 W/m² at a dark winter dawn", so the ensemble's value near the origin of the
radiation features is wherever its residual happened to settle — an incidental
constant of order 0.3–4% of fleet capacity, whose **sign is incidental too**.
That is why ABL-335's two failure shapes have one cause, and why a sign
constraint alone cannot fix DE: DE's floor is positive.

Caveat on AT/BE: ABL-337 reported AT 118 MW and BE −310 MW for `mw_removed_total`
where this reconstruction gives −150.7 and −245.9. DE and the per-country
`hours_zeroed_night` (AT 8, BE 7, DE 7, FR 8) match exactly. The two runs used
different generation instants, so the rolling anchor and weather vintage differ;
DE agrees because its night output is a flat floor and is insensitive to both.

## 2. Protocol

Four countries, two held-out windows, all fits refitted on the truncated window.

- **Windows** (60 days each): spring **2026-03-01 .. 2026-04-29**, summer
  **2026-06-13 .. 2026-08-11**. n_holdout 1,440 hours per country per window.
- **Control is a refit, not the live artifact.** The live artifacts were fitted
  through roughly today, so scoring them on either window would be in-sample and
  would set a bar nothing could clear.
- **Bands.** `night` is the serving clamp's own predicate (sun below −8° for the
  whole hour). `shoulder` is not-night with the sun below the horizon at the
  hour midpoint — the band ABL-337 flagged as the clamp's blind spot. `daylight`
  is the rest. Night is reported in **MW only**: its actuals are ~0, so a
  percentage there measures the denominator, not the model.
- **Baseline.** Seasonal-naive D-7 same hour, scored on the identical rows.
- **Algorithm.** Two sweeps. The **objective** comparison (§3a) forces every arm
  to **xgboost**, because CatBoost cannot fit a log-link arm at all (§5) and a
  comparison against a fit that never started is not a comparison. The
  **feature** comparison (§3b) keeps each country's incumbent algorithm (AT
  xgboost, BE/DE/FR catboost), so it changes exactly one variable.
  Forcing xgboost is **not** free: on the summer control it is 10.7% better for
  DE but 6.3% *worse* for BE and 4.1% worse for FR (§5). §3a's percentages are
  therefore against a baseline that is weaker than the incumbent for BE and FR —
  which makes the log-link degradation reported there an **under**-estimate.
- **Not serve-faithful.** Features come from the training-time pipeline, whose
  lags and rolling windows are target-anchored; at serving they are anchored at
  the generation instant (ABL-183). Every arm carries that identically, so the
  comparison is sound and the absolute MW are optimistic. The ABL-338 geometry
  features are the exception — identical in both paths by construction, since
  both call `solar_features.solar_geometry_frame`.
- **Noise floor.** Dropping 4 of DE's 1,957 night training rows moved its
  daylight MAE by 1.3%. Read anything under ~1.5% as fit noise.

**Contamination.** ABL-71 / ABL-111 / ABL-109 do not touch these solar windows.
What does is the FR defect ABL-337 filed: FR's `energy_renewable.solar_mw` reads
above 1 MW at **488 of 11,614** night training rows, up to **439.3 MW**. Those
rows are excluded from every fit reported here (`--drop-impossible-night`) and
never from the score. DE has 4 such rows (max 1.7 MW); AT and BE have none.
Excluding them fixed FR's night entirely — mean night prediction **22.46 → 0.05
MW** — *and* improved FR daylight MAE by 1.5%. FR's night floor was substantially
the model learning its target faithfully.

**History.** AT's `energy_renewable` solar starts 2025-11-07 and DE's 2025-09-08,
so neither has seen a full seasonal cycle. On the spring window that leaves AT
with 2,207 and DE with 2,314 training hours, and both controls there are at or
worse than seasonal-naive (AT daylight WAPE 78.2% vs naive 49.4%). **The spring
AT and DE columns are not a fair test of anything** and are excluded from the
verdict; BE (18,837 h) and FR (27,228 h) carry it.

## 3a. The objective — daylight MAE, % change vs the xgboost control refit

Negative is better. 8 country-windows; the AT/DE spring columns lack history
(see §2) and are excluded from the verdict.

| arm | spr/AT | spr/BE | spr/DE | spr/FR | sum/AT | sum/BE | sum/DE | sum/FR | worst | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `geometry` | −5.8 | +1.6 | −1.7 | +1.0 | +0.9 | −4.0 | −3.1 | +2.9 | **+2.9** | −1.0 |
| `geometry_tweedie` | +4.1 | +11.1 | +15.8 | +13.7 | +0.6 | −7.6 | −3.2 | −5.2 | **+15.8** | +3.7 |
| `geometry_poisson` | +5.9 | +16.8 | +36.8 | +8.3 | −1.0 | −6.1 | −2.8 | −4.0 | **+36.8** | +6.7 |
| `geometry_nightw100` | −2.3 | −0.3 | +0.3 | −2.1 | +11.2 | −2.9 | +6.4 | +3.0 | **+11.2** | +1.7 |
| `daylight_fit` | −5.1 | +3.0 | +0.2 | −1.5 | −0.5 | −7.5 | −0.4 | −3.8 | **+3.0** | −2.0 |
| `daylight_fit_tweedie` | +0.6 | +18.6 | +13.2 | +0.3 | −0.5 | −2.3 | −3.9 | −11.5 | **+18.6** | +1.8 |

Night behaviour of the same arms — mean prediction at night, and negative
predictions per 1,440 holdout hours:

| arm | night mean pred (spring / summer, worst country) | negative predictions |
|---|---|---|
| control | +14.7 / +94.4 MW | 0 – 486 |
| `geometry` | +52.7 / +12.2 MW | 0 – 474 |
| `geometry_tweedie` | +0.43 / +0.05 MW | **0 everywhere** |
| `geometry_poisson` | +4.92 / +1.48 MW | **0 everywhere** |
| `daylight_fit` | 0.00 / 0.00 MW (by construction) | 0 – 95, worst −47.2 MW |
| `daylight_fit_tweedie` | 0.00 / 0.00 MW | **0 everywhere** |

## 3b. The feature — same table, each country on its own incumbent algorithm

One variable changes here: the two geometry columns. This is the honest read of
what the feature costs, uncontaminated by the algorithm switch §3a needed.

| arm | spr/AT | spr/BE | spr/DE | spr/FR | sum/AT | sum/BE | sum/DE | sum/FR | worst | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `geometry` | −5.8 | +4.9 | −12.7 | +0.8 | +0.9 | −2.8 | −4.1 | −6.5 | **+4.9** | −3.2 |
| `geometry_tweedie` | +4.1 | +44.7 | +68.7 | +18.6 | +0.6 | +30.2 | +313.5 | +16.4 | **+313.5** | +62.1 |
| `daylight_fit` | −5.1 | +1.6 | −12.4 | −1.3 | −0.5 | −2.6 | −5.2 | −6.9 | **+1.6** | −4.0 |
| `daylight_fit_tweedie` | +0.6 | +41.3 | +91.8 | +118.4 | −0.5 | +34.7 | +214.0 | +33.1 | **+214.0** | +66.7 |

The three-figure Tweedie cells are the CatBoost countries and are the §5 defect,
not a property of the loss. The two squared-error arms are unaffected by it.

Night behaviour on the incumbent algorithm is worth reading beside this, because
it is what rules the feature out as a standalone fix — **on CatBoost the
geometry feature does not reliably reduce the night floor and on DE it makes it
worse**: DE summer control 220.1 MW → geometry **453.8 MW**, and DE summer
shoulder MAE 284.9 → 501.4 MW. Under XGBoost (AT) the same feature behaves. This
is the oblivious-tree structure CatBoost uses: one split per level for the whole
tree, so it cannot carve out the night region the way an asymmetric tree can.

### What this says

1. **Every arm that guarantees non-negativity in the fit degrades daylight.**
   Tweedie +15.8% worst, Poisson +36.8%, and the daylight-only Tweedie +18.6% —
   all on the two countries that have the history to judge it. This is the
   answer to the issue's request #2, and it is negative.

   The degradation is **not under-convergence**. `geometry_tweedie_deep`
   (1,500 trees instead of 500) returns numbers identical to the decimal —
   early stopping halts well before the extra capacity is reached. The log link
   is converged and still worse. The cost is structural: a log link optimises
   *relative* error, so it re-weights the high-MW daylight hours that carry the
   MAE against the low-MW ones. That it *gains* 5–8% in summer and loses 11–14%
   in spring is the same mechanism seen from two sides.

2. **The geometry feature does not cost daylight accuracy** — within-algorithm
   worst +4.9% (BE spring), mean −3.2%; forced to xgboost, worst +2.9%, mean
   −1.0%. Under xgboost it also improves the shoulder band the clamp does not
   cover (DE summer shoulder MAE **189.3 → 79.2 MW**, BE **22.5 → 15.2**).

   But on its own it does **not** fix the night, and on CatBoost it can make it
   worse (§3b: DE summer 220.1 → 453.8 MW). Even under xgboost, DE's spring
   night mean rose 14.7 → 52.7 MW and up to 474 negative predictions survive.
   Consistent with §1: the residue is incidental, so a feature that lets the
   model *find* night does not force it to land on zero.

3. **The only arm that is both daylight-safe and exactly zero at night is
   `daylight_fit`** — a hurdle: fit the regression on lit rows only, emit zero
   at night from the geometry. It is the best arm under both sweeps and both
   windows: within-algorithm worst **+1.6%**, mean −4.0%; forced to xgboost
   worst +3.0%, mean −2.0%. Night is exactly 0. It comes with a caveat that
   makes it a CEO call rather than a selection this harness can make (§4).

## 4. Why `daylight_fit` is not simply the recommendation

A model that is zero at night **by construction** drives `forecast_clamp_log`'s
`hours_zeroed_night` and `mw_removed_night` to zero by construction as well.
That log is the independent success metric the CEO required precisely so the fit
could not mark its own homework. A hurdle would make it uninformative.

It is also not a serving no-op: the zeroing has to live inside the model (or the
artifact), because a regression that never saw a night row will extrapolate
arbitrarily there, and the existing ABL-337 clamp would then be doing exactly
the work it is supposed to stop doing. That is a serving-path change, which is
Founding Engineer territory, not mine.

And it does not deliver non-negativity: `daylight_fit` still emits negative
values at lit hours (worst −47.2 MW, BE spring), which the clamp's floor absorbs.

One further note for whoever owns the counter: with any log-link arm,
`hours_zeroed_night` stays at 7–8 while `mw_removed_night` goes to ~0.05 MW,
because `exp(margin)` is never exactly zero and the counter tests `!= 0.0`.
**`mw_removed_night` is the instrument to read; `hours_zeroed_night` needs a
tolerance** or it will report a perfect fit as unchanged.

## 5. CatBoost's log-link fits are broken, not merely worse

Left as measured because it would otherwise look like evidence against the loss.
On the summer window with the incumbent algorithms, CatBoost `Poisson` returned
daylight WAPE 100% for both DE and FR, predicting a constant **1.0 MW**
everywhere. `tree_count_` is **1**: `use_best_model` selected iteration 0.
CatBoost does not boost from the average for Poisson/Tweedie, so the fit starts
at `exp(0) = 1 MW` and must climb ~10 in log space to reach a 20–50 GW fleet;
it never started. CatBoost `Tweedie` on DE kept 189 trees and predicted a mean of
7,233 MW against 15,288 MW of actual — under-fit by the same mechanism.

This is a fit that did not run, not a verdict on the loss, so §3a forces xgboost
for the objective comparison. The §3b Tweedie row shows the same defect from the
other side: +313.5% on DE, +214.0% on `daylight_fit_tweedie`.

**Forcing xgboost is not a free upgrade, and an earlier draft of this pack got
that wrong.** Like-for-like on the summer control arm — same features, same
loss, only the algorithm differs:

| country | incumbent | incumbent daylight MAE | xgboost | xgboost vs incumbent |
|---|---|---:|---:|---:|
| AT | xgboost | 291.9 | 291.9 | — |
| BE | catboost | 508.1 | 540.3 | **+6.3%** (worse) |
| DE | catboost | 3,614.6 | 3,227.6 | **−10.7%** (better) |
| FR | catboost | 1,395.9 | 1,452.7 | **+4.1%** (worse) |

So XGBoost is the better algorithm for **DE only**. Two consequences: §3a's
percentages sit on a baseline that is weaker than the incumbent for BE and FR,
so the log-link degradation there is understated; and any algorithm switch is a
per-country question, not a fleet one.

## 6. What landed, and what did not

**Landed** (all inert until something is retrained — no artifact was touched, no
serving behaviour changed):

- `src/solar_features.py` — the two geometry features, one function called by
  both the training pipeline and the serve-faithful builder, so train/serve skew
  is impossible by construction rather than by convention. Two features, not
  three: a `sin(elevation)` clear-sky factor was left out because a tree
  ensemble is invariant to a monotone transform of a single feature, so it
  cannot help these models.
- `src/features.py`, `src/wind_features.py` — wiring, solar only. Artifacts
  trained before this keep their own 25-name `feature_columns` and serve
  unchanged.
- `src/forecaster.py` — `nonneg_objective` (`tweedie` / `poisson`), recorded in
  and restored from the artifact, with a pre-fit refusal on a negative target.
  **Not adopted by any artifact.**
- `scripts/abl338_solar_holdout.py`, `reports/abl_338_solar/*` — the harness and
  the raw numbers.
- `tests/test_solar_geometry_features.py` — 15 tests, including the bit-identity
  of the training feature and the clamp's own predicate. Full suite 316 passed.

**Not done, deliberately:** the four artifacts were not retrained and nothing is
proposed for promotion. The serving clamp is untouched, and on this evidence it
should stay doing the work it is doing.

## 7. Decision requested

1. **Accept the negative result on the non-negativity constraint** as specified
   (log link), on the numbers in §3.
2. Choose the follow-up: **(a)** geometry feature only — daylight-safe, better
   shoulder, night unfixed; **(b)** the `daylight_fit` hurdle — daylight-safe and
   night exactly zero, at the cost of the clamp-log instrument and a
   serving-path change; **(c)** neither, keep the clamp and close.
3. Separately, and out of scope here: **DE solar is on CatBoost and loses 10.7%
   daylight MAE to a like-for-like XGBoost refit** (§5). BE and FR are better on
   CatBoost, so this is a DE-specific follow-up, not a fleet switch. Worth its
   own issue; it is a larger measured win than anything ABL-338 asked for.
