# ABL-607 — why our D+2 load model loses to a D-7 seasonal naive

**Author:** Forecasting Scientist · **Date:** 2026-08-28 · **Status:** diagnosis, no fix, no serving change

Machine records: `reports/abl_607_d2_load_diagnosis.json` (the 15:58 UTC run,
which every number here is quoted from) and
`reports/abl_607_d2_load_diagnosis_reread.json` (the ABL-619 re-read on the
rebuilt replica, which carries the §2.1 census and moves the headline count
from 10 to 9 — **read §3.1 before quoting a count from this document**).
Reproduce:
`.venv\Scripts\python.exe scripts/abl607_d2_load_diagnosis.py --replica-db C:\Code\able\data\energy_dashboard.db --json-out reports/abl_607_d2_load_diagnosis.json --models-dir models`

---

## 1. Headline

**The cause is named, it is a single mechanism, it is shared by all 24 countries, and
it is already known to this repo under a different forecast type.**

Load reaches inference through `Forecaster.predict_d2`'s **proxy-row** branch
(`src/forecaster.py:801-874`). For each target hour it takes
`same_hour_data.iloc[-1:]` — the most recent historical row carrying that
hour-of-day — and overrides only the calendar and weather columns. **Every
history feature keeps the value it had for the last observed day.** That is 11
of the artifact's 26 columns (three lags, eight rolling statistics) and
**55–91% of its decision weight** (fleet median 66.8%).

Measured, the anchor sits **exactly 2 days** before the target on every
country and every run in the window, and the anchor row's own lag block
describes the day before *that*. The consequence is not subtle:

- **Our D+2 forecast for day T is a better description of day T-3 than of day T.**
  Scored against `actual(T - k days)`, the WAPE-minimising k is **3 in 20 of 23
  evaluable countries** (2 in AT and SI, 0 in FI). Median WAPE **9.08% at k=0
  against 5.09% at k=3**. Control: the D-7 arm's own argmin is k=7 in all 24.
- **100% of served `target_value_lag_7d` values are on the wrong day of the
  week.** The served value is `actual(T - 9 days)`, not `actual(T - 7 days)`;
  the two differ by a median **10.5%** (range 4.5–15.6%).
- The residual bias is therefore weekday-shaped: **+13.9% on Sundays, -10.3% on
  Tuesdays**, against a D-7 lag that is flat (-0.9% to +1.9%) at every weekday.

**"Our horizon is longer" does not explain the loss, and neither does the
algorithm.** A D-7 lag is computable at a 64h horizon; the serving path simply
never computes it. It hands the model a 9-day-old, wrong-weekday value under
the name `lag_7d`.

**This is the ABL-179 defect, still live on load.** ABL-179 diagnosed exactly
this for wind ("inference copied the latest same-hour historical feature row and
overrode only calendar and wind-speed fields, so lags carried the wrong meaning
at D+1/D+2"), ABL-183/ABL-191 built the serve-faithful builder that fixed it,
and `wind_features.py:189-192` records why load was left behind: *"load/price
remain out of scope for a different reason: different artifact shape, not
diagnosed by ABL-179, left on the original code path."* This pack is that
missing diagnosis. I am not claiming a new mechanism; I am closing a known
exclusion with a measurement.

---

## 2. Protocol

Inherited from ABL-246 so the two packs are comparable.

| | |
|---|---|
| **Window** | target hours 2026-08-13 08:00 → 2026-08-28 00:00 UTC (16 target days) |
| **Sample** | **8,436** scored (country, hour) pairs on the reproduction panel |
| **Countries** | 24 served; **23 evaluable**, NL held out (§7) |
| **Source** | `forecast_vintage_archive` (ABL-184), `first_seen_at >= 2026-08-12` |
| **Basis** | **out-of-sample** throughout, except the one column labelled in-sample in §5.4 |
| **Truth** | `energy_load`, hourly means (ABL-332), 0.0 rows dropped |
| **Reads** | replica opened `file:...?mode=ro`; nothing written outside `reports/` |
| **Plausibility** | nothing filtered; the ABL-431 guard would have refused **0 of 67,008** archive rows, 24/24 countries evaluable — §2.1, §3.1 |
| **Interpreter** | the rail — Python 3.14.3, xgboost 3.3.0 |

**Arms.** `ml_band` is ABL-246's arm verbatim (latest leak-free vintage in the
scorecard's registered 24-64h band). `ml_g1` / `ml_g2` split the same archive by
**run-day offset** — the run that fired on the target's eve, and the one that
fired the day before it. `d{k}_naive` is `actual(T - k days)` for k in
{1, 2, 3, 7, 14}. `generated_at` is recovered as `target - horizon_hours`, which
is exact; `first_seen_at` is only the poller's sighting.

**Three panels, because each added arm narrows the intersection and ABL-246's
most marginal cell (LV, ci_lo = +0.03) is decided by a handful of rows.**
`panel_a` (n = **8,436**) carries the band arm, D-7 and truth — §3 only.
`panel` (n = 8,402) adds the full lag ladder. `panel_g` (n = 8,025, 15 target
days) additionally requires both run-day offsets for the same target hour. Every
section below says which it used.

### 2.1 The plausibility guard, and why this read is exempt from it

ABL-462 widened the TSO plausibility sweep (ABL-431/458) past `src/`, and it
named this script. The disposition, settled on ABL-611, is **`EXEMPT_READS`**,
and it is a measurement question rather than a hygiene one, so it is stated here
next to the numbers it protects.

**The read is not a TSO read.** `load_archive` selects
`forecast_type = 'load' AND source = 'ml'` — our own forecasts. No TSO row is
read anywhere in this script. The guard's archive reference is the exact
complement: `forecast_read` bounds it on
`source = 'tso' AND model_name = 'tso-day_ahead'`, because "`source = 'ml'` rows
are our own forecasts … not a published TSO series"
(`src/tso_plausibility.py`). The sweep is a substring match on three table
names; here it matched a table *name*, not a TSO read.

**And filtering it would be a scoring defect.** The guard is one-sided — a bare
`>` against `3 × reference`. On the arm *under test*, the only rows it can
remove are our own largest over-forecasts, which are exactly the errors this
pack measures. Every arm scores on one shared (country, target-hour)
intersection, so a dropped ML row takes the D-7 comparator's cell with it, in
precisely the hour our model was worst. The bias is one-directional: filtering
could only ever **shrink** the D+2-vs-D-7 gap, never widen it. ABL-431's case is
the mirror image — there the implausible value is an *input* about to be fitted
on; here it would be an *output* being graded. Same table, same rows, opposite
role.

**The measurement is kept anyway.** `plausibility_census` computes the same
reference over exactly the scored rows and reports what the guard *would* have
refused, using `implausible_mask` (the predicate) rather than `guard_series`
(the filter).

**Measured: the guard would have refused nothing. 0 of 67,008 archive rows**
(ABL-619, machine record `reports/abl_607_d2_load_diagnosis_reread.json`
section 0 — the census landed in the script after this report was first
generated, and the run that produced it is §3.1's re-read).

All **24 of 24** countries had an evaluable reference, so this is a zero that
was actually tested rather than a zero that was never evaluated. The headroom
is wide and it is wide everywhere: the largest value our model published, as a
fraction of that country's refusal threshold, runs from **0.1997 (SE)** to
**0.3234 (ES)**. Nowhere did we come within a factor of three of the bar.

| | reference `R` | threshold `3R` | our max | max / threshold |
|---|---:|---:|---:|---:|
| ES — tightest | 37,376 MW | 112,128 MW | 36,260 MW | **0.3234** |
| DE — largest fleet | 75,512 MW | 226,535 MW | 59,772 MW | 0.2639 |
| SE — loosest | 23,474 MW | 70,422 MW | 14,061 MW | **0.1997** |

`R = max(p99.5 TSO day-ahead vintages, p99.5 energy_load)` per country over the
whole history (`reference_scale`, `as_of=None`), with our own `source = 'ml'`
rows excluded from setting it by `forecast_read` — so an arm that overshot
could not lift the bar it is judged against.

Note that the census **cannot change any number in this report**: it filters
nothing by construction, so it decides how the ranking is *read*, never its
arithmetic. A non-zero count would have been a finding about our own model and
would have gone to its own issue. It is zero, so the ranking below rests on
inputs measured plausible rather than assumed so.

An exemption that could not have detected anything would be the vacuous kind, so
both halves are pinned by `tests/test_abl607_guarded_read.py`: that the panel
comes back row-for-row identical (`assert_frame_equal`, and a source check that
no filtering entry point is ever called here), and that the census fires at the
ABL-431 incident's own scale. A non-zero count would be a finding about **our
own model** — a published load forecast above three times a country's fleet peak
deserves its own issue — and it would be scored and reported, never deleted.

---

## 3. Control — the finding reproduces exactly

On `panel_a`, n = **8,436**, the same number ABL-246 scored:

| | ABL-246 §4.1 | this pack |
|---|---|---|
| readable losers | 10: SI AT CZ PL SK ES DE LV PT SE | **the same 10** |
| readable winners | 1: GR | **GR** |
| fleet median ML D+2 | 8.85% | **8.85%** |
| fleet median D-7 | 5.10% | **5.10%** |
| SI (worst) | +10.07 [+5.22, +14.93] | **+10.071 [+5.215, +14.926]** |
| GR (the win) | -6.60 [-10.16, -3.04] | **-6.600 [-10.162, -3.038]** |

The diagnosis below is built on a reproduced finding, not a re-measured one.

### 3.1 Re-read on a later replica vintage — the count is 9, not 10

Everything above and below is the run of **2026-08-28 15:58 UTC**. ABL-619 then
required a second run to publish §2.1's census, and the replica had been
rebuilt in between. The re-read uses the **identical pinned window** and the
identical script (`DEFAULT_MAX_TARGET = 2026-08-28 01:00`,
`GENUINE_VINTAGE_FLOOR = 2026-08-12`); machine record
`reports/abl_607_d2_load_diagnosis_reread.json`.

**It does not reproduce 10 of 23. It gives 9, and the difference is DE.**

| | 15:58 run | re-read | |
|---|---|---|---|
| `panel_a` scored pairs | 8,436 | **8,451** | +15 |
| target days per country | 16 for 12, 15 for 12 | **16 for all 24** | uniform basis |
| readable losers | **10** | **9** | DE drops out |
| readable winners | 1: GR | 1: GR | unchanged |
| DE, ML − D-7 | +3.59 [+0.88, +6.30] | **+2.70 [−0.46, +5.85]** | crosses zero |
| LV (was most marginal) | +3.39 [+0.03, +6.75] | +4.20 [+0.64, +7.76] | strengthens |
| SI (worst) | +10.07 [+5.22, +14.93] | +10.08 [+5.23, +14.94] | unchanged |
| GR (the win) | −6.60 [−10.16, −3.04] | −6.45 [−9.95, −2.95] | unchanged |
| fleet median ML D+2 | 8.85% | 8.85% | |
| fleet median D-7 | 5.10% | 5.07% | |

**This is a data-vintage effect, not a correction.** The two runs were compared
on inputs before any metric, precisely so this could be told apart. **12 of 24
countries gained a target day** — 11 of them by exactly one row, BG by four —
for +15 rows in total, `8,436 → 8,451`. A country gaining one row and one day
can only have gained the window's final hour, `2026-08-28 00:00`, whose actual
had not landed when the first run read the replica. Separately, **IT, SE and SI
moved their WAPE with both `n` and `days` unchanged** (10.33→10.21, 5.91→5.79,
15.66→15.40), so already-scored actuals were revised underneath the pinned
window as well. Neither run is wrong: the re-read has the completer data, on a
uniform 16 days for every country instead of a 12/12 split at 16 and 15.

**What it does and does not change.** DE's central estimate is still a **loss**
of +2.70 pp; what it stopped being is *readable* at 95% on 16 paired days. Nine
countries still lose readably, one still wins, and the mechanism in §4 — which
is about which row the model anchors on, not about interval width — is
untouched. This is precisely the caveat this pack and ABL-246 both carried:
**a fortnight is short, and a marginal cell can be moved by a single day.** It
is an argument for the ~30-day re-read before any intervention, not against the
finding. Treat **9 of 23** as the current count.

**9 is not a firmer number than 10 was.** DE is not *the* borderline cell; it is
the cell that happened to cross. Measuring the margins against the size of the
step that moved it — one vintage step shifted `ci_lo` by at most **1.34 pp**
(DE), median 0.06, p75 0.20 across the 24 countries — five of the nine
survivors sit inside that observed maximum, and three cells with a positive
central estimate sit just under the line on the other side:

| readable losers, by margin | `ci_lo` (pp) | | below the line | `ci_lo` (pp) |
|---|---:|---|---|---:|
| PT | +0.15 | | NO | −0.05 |
| PL | +0.56 | | FR | −0.27 |
| LV | +0.64 | | DE | −0.46 |
| SE | +0.89 | | | |
| ES | +1.03 | | | |
| SK, AT, CZ, SI | +1.82 … +5.23 | | | |

So the count is not a stable property of the fortnight: it is a threshold
crossing on cells whose margins are the same size as one day of data arriving.
Only **SK, AT, CZ and SI** — and the GR win, whose binding bound is `ci_hi` =
−2.95 — are outside the range a single vintage step has already been observed
to move. Read those five as the findings; read the count itself as provisional
until the ~30-day re-read.

Every number outside this subsection is the 15:58 vintage, quoted from
`reports/abl_607_d2_load_diagnosis.json`. They are not restated against the
re-read, because the shifts are in the third decimal and mixing two vintages
inside one table is how a report starts lying quietly; the re-read record is
committed whole for anyone who needs it.

---

## 4. The mechanism, measured

### 4.1 What the proxy row actually carried

Reconstructing the served feature vector for every (country, run day, target
hour) in the g=2 arm — same lookback window, same positional shift semantics,
same `iloc[-1:]` selection:

| | fleet |
|---|---|
| anchor gap (target day - proxy day) | **exactly 2.0 days**, every country, every run |
| served `lag_7d` on the target's weekday | **0 of 7,751 evaluable rows (0%)**; 0 of 8,088 including NL |
| \|served `lag_7d` - target-aligned `lag_7d`\| | median **10.5%**, range 4.5% (NO) – 15.6% (LV) |

One row, spelled out. CZ, target **Wednesday** 2026-08-26 09:00, run day
2026-08-24:

| feature | served value | what the name promises | source |
|---|---:|---:|---|
| `target_value_lag_1d` | **6,861 MW** | 8,026 MW | **Sunday** 08-23, not Tuesday 08-25 |
| `target_value_lag_7d` | **8,189 MW** | 8,137 MW | **Monday** 08-17, not Wednesday 08-19 |
| *actual* | — | *8,237 MW* | |

The model's most heavily weighted recency feature is a **Sunday** value fed into
a **Wednesday** forecast, 14.5% low. Nothing errors; the artifact is fine; the
frame is built correctly for a day that is not the target.

### 4.2 The forecast describes T-3

Scoring `ml_g2` against `actual(T - k days)` and taking the argmin
(`panel_g`, 23 evaluable countries):

| argmin k | countries |
|---|---|
| **3** | **20** — BE BG CH CZ DE EE ES FR GR HR HU IT LT LV NO PL PT RO SE SK |
| 2 | 2 — AT, SI |
| 0 | 1 — FI (the one country whose model tracks its own target best) |

Median WAPE **9.08% at k = 0** against **5.09% at the argmin**. A forecast is
allowed to be inaccurate; a forecast that fits a three-day-old day roughly twice
as well as its own is describing the wrong day. **Control:** the same statistic
on the D-7 arm returns k = 7 for all 24 countries, which is what a correctly
anchored lag must return.

### 4.3 The weekday signature the mechanism predicts

An anchor 3 days early carries a *different weekday's* level. Relative bias by
target weekday, evaluable countries pooled (`panel_g`):

| target | anchor weekday | predicted sign | `ml_g2` relbias | `d7_naive` relbias |
|---|---|---|---:|---:|
| Mon | Fri | ~neutral | **-0.52%** | +0.17% |
| Tue | Sat | under | **-10.31%** | +0.48% |
| Wed | Sun | under (worst) | **-7.88%** | -0.79% |
| Thu | Mon | ~neutral | **-1.93%** | -0.92% |
| Fri | Tue | ~neutral | **+1.86%** | +1.89% |
| Sat | Wed | over | **+9.16%** | +1.30% |
| Sun | Thu | over (worst) | **+13.92%** | +1.14% |

Seven of seven signs as predicted, a **24 pp** swing across the week, against a
weekday-aligned lag that stays inside ±1.9 pp. This is the falsifiable test the
mechanism had to pass and it is the strongest single piece of evidence here.

### 4.4 Two axes explain every readable cell, including the one win

A lag baseline pays *level drift*; a misaligned anchor pays *weekday amplitude*.
Both are measured on the truth series alone — no forecast enters either, so
neither can be fitted to the answer.

- `weekly_amplitude_pct` = (weekday mean - weekend mean) / mean load
- `week_drift_pct` = mean \|actual(t) - actual(t-168h)\| / mean load

Sorted by their ratio, against the paired daily `ml_g2 - D-7` difference
(`panel_g`; `*` = interval excludes zero; positive means our model is worse):

| country | amp % | drift % | ratio | ml_g2 | D-7 | ml_g2 - D-7 [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| CZ | 17.0 | 2.1 | 7.92 | 8.54 | 2.18 | **+6.63 [+3.09, +10.16]** * |
| AT | 22.1 | 3.1 | 7.07 | 10.58 | 3.07 | **+7.97 [+3.47, +12.48]** * |
| SK | 13.8 | 3.2 | 4.38 | 8.31 | 3.06 | **+5.38 [+2.71, +8.06]** * |
| SI | 22.8 | 5.3 | 4.28 | 16.25 | 5.48 | **+10.70 [+5.21, +16.19]** * |
| PL | 20.2 | 4.7 | 4.27 | 10.50 | 4.44 | **+6.41 [+1.98, +10.85]** * |
| DE | 16.6 | 4.2 | 3.95 | 8.20 | 4.21 | **+4.15 [+1.38, +6.92]** * |
| NO | 6.0 | 1.7 | 3.54 | 2.97 | 1.62 | **+1.25 [+0.16, +2.34]** * |
| SE | 11.2 | 3.6 | 3.09 | 6.20 | 3.73 | **+2.68 [+1.14, +4.22]** * |
| ES | 15.3 | 5.1 | 3.00 | 10.14 | 5.24 | **+4.83 [+1.23, +8.42]** * |
| BE | 13.8 | 5.0 | 2.77 | 7.33 | 4.56 | +2.60 [-0.26, +5.46] |
| FR | 12.5 | 4.5 | 2.76 | 7.45 | 4.44 | **+3.09 [+0.08, +6.09]** * |
| PT | 13.0 | 4.9 | 2.68 | 8.04 | 5.06 | **+3.07 [+0.26, +5.88]** * |
| RO | 18.5 | 8.9 | 2.09 | 11.98 | 8.42 | +3.38 [-1.54, +8.29] |
| LV | 13.5 | 6.8 | 1.98 | 10.72 | 6.49 | **+4.44 [+1.54, +7.35]** * |
| IT | 17.9 | 11.0 | 1.63 | 10.63 | 10.45 | +0.08 [-3.36, +3.52] |
| HR | 13.6 | 9.1 | 1.49 | 10.74 | 9.05 | +2.08 [-0.90, +5.06] |
| LT | 12.7 | 8.6 | 1.47 | 9.08 | 8.49 | +0.70 [-2.18, +3.58] |
| HU | 13.2 | 10.4 | 1.27 | 9.43 | 9.65 | -0.62 [-4.51, +3.27] |
| FI | 3.5 | 3.0 | 1.16 | 2.42 | 2.87 | -0.45 [-1.24, +0.33] |
| **GR** | 16.6 | **17.1** | 0.97 | 11.61 | 17.64 | **-6.22 [-10.09, -2.35]** * |
| BG | 6.5 | 7.9 | 0.83 | 7.21 | 7.95 | -0.71 [-2.77, +1.34] |
| CH | 3.6 | 7.5 | 0.47 | 7.11 | 7.55 | -0.45 [-1.51, +0.62] |
| EE | -1.0 | 11.1 | -0.09 | 10.52 | 10.73 | -0.24 [-1.74, +1.27] |

**Spearman(ratio, gap) = +0.876** over 23 countries (amplitude alone +0.639,
drift alone -0.515). Every readable loser sits at ratio >= 1.98; the only
readable win sits at 0.97. **GR is not a country where our model is good — it is
the country where the D-7 baseline is broken**, with a week-on-week drift of
17.1% against a 5.1% fleet median. We win there by being anchored 3 days back
instead of 7, which is the same defect paying off once.

The one cross-country statistic that does *not* order the losses is the artifact
importance share (Spearman +0.230) — expected, since every country is above
0.55 and importances are not comparable across catboost and xgboost.

---

## 5. What the cause is not

### 5.1 Not the algorithm

AT, BE and FR serve xgboost; the other 21 catboost. Of the 12 readable losers on
the clean g=2 arm, 10 are catboost and 2 are xgboost — the same 21:3 ratio as
the fleet. The mechanism lives in `predict_d2`, which does not branch on
algorithm.

**Caveat that belongs to whoever reuses §6:** the served `model_name` in the
archive says **DE serves catboost**, while `models/DE/load/model.joblib` on this
box is **xgboost** (saved 2026-04-04). The artifact directory is not
production's serving set for DE. The other 23 agree.

### 5.2 Not the horizon per se

One extra day of anchor staleness (`ml_g2` against `ml_g1`, paired daily) costs a
median **2.16 pp**, readable in 11 of 23 countries. Real, but it is not the
whole loss: **even the D+1 arm readably loses to D-7 in 7 countries** (AT CZ DE
LV PL SI SK). Freshness moves the anchor from T-3 to T-2; it does not align it
to the target's weekday, so the defect survives at every horizon this module
serves.

### 5.3 Not the diurnal shape

The error is a **daily-level** error. Splitting each arm's MAE into a
daily-mean-error part and a within-day-shape part, the level share for `ml_g2`
has a median of **0.929** (0.53–0.97). The model gets the shape of the day
approximately right and the level of the day wrong, which is what carrying
another day's level does.

### 5.4 And it is **not** fixable by calibration — the useful negative result

A leak-free trailing bias correction (each day corrected by the mean error over
prior days only) makes it **worse in 23 of 23 evaluable countries** — GR
10.65 → 12.89, AT 9.81 → 11.09, SI 15.66 → 17.27. Even the *in-sample* bias
removal, an upper bound that is not forecastable, is worse in 6 of 23 and never
better than 0.4 pp.

The reason is in §4.3: the level error is not a constant offset, it is a
**weekday-alternating** one that averages to near zero (fleet relative bias
-1.8% to +3.4%). A scalar or affine correction cannot remove a signal that
changes sign twice a week. **Any proposal to fix this with bias or affine
calibration is mathematically incapable of helping, and this is the cheapest
candidate fix, so it is worth killing early.**

### 5.5 Artifact age is real but is not this

Every load artifact was saved **2026-02-01/02** except BE, DE, FR
(2026-04-04) — 5 to 7 months old at serve time, and none carries the four
ABL-393 holiday features. That is a hygiene finding worth its own issue, but it
is not the driver here: a stale training window produces a level bias, and
§5.4 shows the level error is not a bias a level correction can touch.

---

## 6. A correction the scorecard owner should carry

**The registered 24-64h "D+2" band is 68.2% D+1 forecasts.** Every run emits 48
target hours, so for target hours 08:00-23:00 the band's latest leak-free
vintage comes from the **T-1** run (target T 23:00, generated T-1 20:00, leads
27h — inside the band). Only hours 00:00-07:00 are genuinely g=2. Measured
across the fleet: **5,726 rows at g=1 against 2,676 at g=2.**

Consequence: every number published on that band, ABL-246 §4.1 included,
**understates the true D+2 loss**. On the clean g=2 arm the fleet median moves
8.96 → 9.08 and the readable losers go from 10 to **12** (adding FR and NO), with
GR still the only win.

I am not re-grading anything on that basis — the band is a registered choice and
ABL-246's recommendation is with the Board. It makes the D+2 case *worse*, so it
does not change any standing recommendation. Recorded here, and commented on
ABL-246, for whoever next reads a D+2 number off that band.

---

## 7. Caveats

- **15-16 target days is short.** Every claim is carried by a paired daily
  interval over k = 14-16 days rather than by a WAPE point estimate. A re-read
  at ~30 days is cheap and should precede any large intervention. The mechanism
  claims in §4.1-4.2 do not depend on the window at all — they are properties of
  the serving path, not of a sample.
- **The 2-day anchor gap is a lower bound.** It assumes every actual up to the
  serving path's own date bound had been ingested when the run fired. **ABL-71**
  (prod ingest stale, fixes undeployed) can only push the last observed day
  further back, never forward.
- **NL is excluded and must stay excluded.** Its realized series is net of
  behind-the-meter solar while our forecast is gross (ABL-277 / ABL-505 /
  ABL-506). It is scored in the JSON and held out of every conclusion. It is
  also, with FI, one of two countries whose argmin is k=0 — for NL that reflects
  the basis mismatch dominating everything else, not a healthy anchor.
- **A second, smaller defect surfaced in the same reconstruction.**
  `create_lag_features` and `create_rolling_features` shift by **rows**, not by
  hours, and `load_training_data` drops NaN hours first — so a gap in
  `energy_load` makes a "24-hour" lag something other than 24 hours. Measured on
  the served frames: **0% of rows affected in 21 countries, 39.8% in EE and
  68.0% in LV**. It compounds the anchor defect for exactly those two and is
  worth carrying into the fix, not a separate cause.
- **The archive read is deliberately unfiltered** (§2.1), so no plausibility
  screen stands between the archive and these numbers. That is the correct
  choice for an arm being *graded* rather than fitted on, but it is a choice,
  and the protection it gives up is measured rather than assumed: the ABL-431
  guard would have refused **0 of 67,008** rows, across 24/24 countries with an
  evaluable reference, with our largest published value never above **32.3%**
  of any country's threshold. So the claim is "nothing implausible is in here,
  and that was tested" — not "nothing implausible could be".
- **The count is 9, not 10, on the later replica vintage (§3.1).** DE's loss
  stays a loss in point estimate (+2.70 pp) but stops being readable at 95%
  once the window's final target day lands. A fortnight is short enough that
  one day moves a marginal cell; that is an argument for the 30-day re-read,
  and it is the same caveat ABL-246 carried.
- **Contamination.** ABL-111/ABL-109 (zero-as-missing actual load): **1 row** in
  the whole window, dropped. ABL-67 (fabricated net_position) does not touch
  load. **ABL-71** applies equally to every arm, so it cannot manufacture a gap
  between two arms scored on the same hours, but it is not quantified here.
- **§5.1's importance shares are read off the local `models/` tree**, which
  matches production's algorithm for 23 of 24 countries (DE excepted). They are
  a proxy for the served artifacts, not the served artifacts themselves.
- Replica read-only throughout. No model was trained or promoted; no registry,
  serving path or ingest was touched.

---

## 8. What I recommend, and what I am not doing

**The fix is not mine to write and is not in this issue** — the issue scoped
one landable change as the diagnosis, and the change it implies reaches the
production serving path.

**Recommendation, for the CEO to route:**

1. **Extend the ABL-183/ABL-191 serve-faithful feature builder to load**, and
   delete the proxy-row branch for it. This is the fix that already worked for
   wind and solar, and `wind_features.py:189-192` says load was excluded only
   for want of this diagnosis. Filed as a child issue of ABL-607.
2. **It requires a refit, not just a serving change.** At a 24-64h horizon
   `target_value_lag_1d` and both 24h rolling windows are *genuinely*
   unavailable — they describe D+1, which is still in the future. A
   serve-faithful builder cannot conjure them; it can only refuse them. So the
   load feature set has to be re-specified to what a D+2 forecast can actually
   hold — lags at 2/3/7/14 days and rolling windows ending at T-48h — and the
   models refit on it. **A serving-only change would make the numbers worse, not
   better**, because it would replace a wrong value with no value.
3. **Do not spend anything on bias or affine calibration for load D+2** (§5.4).
4. **Re-read at ~30 target days before any large intervention**, per ABL-246's
   own caveat. The archive reaches that around 2026-09-12.
5. The **D-7 lag is the baseline any refit must beat**, per country, on a paired
   daily interval — not the current model. On this window it is 5.10% against
   our 8.85%, and it is free.

**Follow-ups.** I searched before filing. The mechanism is ABL-179's and is not
re-filed; the serving-path exclusion is `wind_features.py`'s own and is now
answered. Two observations here are new and neither belongs in this issue:

| observation | disposition |
|---|---|
| serve-faithful builder must cover load, with a D+2-buildable feature set (§8.1-8.2) | **child issue of ABL-607**, needs a CEO routing decision — it touches serving |
| registered 24-64h band is 68% D+1 rows (§6) | **commented on ABL-246**; a scorecard-registration matter, changes no standing recommendation |
| positional row-shift lags under gappy `energy_load`, EE 39.8% / LV 68.0% (§7) | carried into the child issue; not a separate cause |
| load artifacts 5-7 months old, no holiday features (§5.5) | noted, not filed — ABL-393 owns the holiday half; age is downstream of the refit above |

**What I am not doing:** no retrain, no promotion, no registry edit, no serving
change, no ingest change. This pack names a cause. Acting on it is a CEO/Board
decision and a Founding Engineer deliverable.

---

*Every number here is out-of-sample except the one column labelled in-sample in
§5.4. Window, n, baseline and contamination are stated in §2 and §7.*
