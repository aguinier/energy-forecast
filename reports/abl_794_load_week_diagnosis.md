# ABL-794: the 09-07..09-13 ML load WAPE doubling

**Author:** Forecasting Scientist · **Date:** 2026-09-14 · **Status:** diagnosis only. No serving, model or artifact change.

Machine record: `reports/abl_794_load_week_diagnosis.json`. Reproduce:
`.venv/Scripts/python.exe scripts/abl794_load_week_diagnosis.py --replica-db C:/Code/able/data/energy_dashboard.db --models-dir C:/Code/able/energy-forecast/models --json-out reports/abl_794_load_week_diagnosis.json`

---

## 0. Answer

1. **Yes, the ABL-179 / ABL-607 anchor defect reaches D+1. This is shown from the served rows, not the code.** I replayed every stored D+1 value through the real serving functions, using the artifact whose `model_version` matches. In 20 countries, **7,126 of 9,600 served D+1 rows (74%)** are reproduced to within 0.5 MW by exactly one proxy row. A further 10% are identified less tightly (§1). None of those rows is target-aligned. The history block comes from **1 day** before the target for hours the 19:00 UTC run had already observed, and **2 days** before for the rest. On the 08-31..09-02 and 09-09 targets it is 3–5 days old, because ingest was stale.
2. **Yes, the doubling is that defect meeting a bigger displacement.** I replayed the same artifact on target-aligned history, with the same weather and calendar. That comparison forecast moved only a little (flagged median **3.39% → 4.45%**), while the served forecast went **7.07% → 10.58%**. The served-minus-aligned gap carries **75–81%** of the rise in IT, HU, HR and SK. The per-unit cost of the defect did not change: a slope fitted on week 1 predicts week 2's daily cost from week 2's displacement (fleet median **6.94 predicted vs 6.91 observed**). What grew is how far the served history block sat from the target. There are two reasons. The week was sharper: IT's weekday/weekend contrast went from 21% to 35%, above every July week, with a late-week temperature drop. And the baseline week's Tue/Wed forecasts came from runs made stale by the 08-29..09-02 ingest collapse. Their anchors landed on weekday rows, so the defect looked cheaper than usual.
3. **No other serving input degraded.** Every well-reproduced country used the latest 12:00 UTC weather run (7.0 h old). Frames are complete, with no zero or duplicate hours. The artifact has no calendar features that could break. The stored values reproduce exactly on both sides of the 09-10 container rebuild. The 09-07/08 outage compounds **one** target day, 09-09, whose anchor went from 2 days to 3.
4. **ABL-608 has the right mechanism and fix, but the wrong scope.** It is written for D+2 only: the feature spec ("rolling windows ending at T-48h"), the refit and the bar. The dashboard's headline D+1 is served by the same branch, and 64.5% of the dashboard's "D+2" hours come from the day-before run. **Recommended next change: ABL-608, re-scoped to both horizons as set out in §4.**

---

## 1. Protocol

| | |
|---|---|
| **Window** | target days 2026-08-24..09-13 in three Mon–Sun weeks: **w0** 08-24..30, **w1** 08-31..09-06 (baseline), **w2** 09-07..13 |
| **Sample** | 24 countries × 504 target hours per arm. w1 has 144 D+1 hours per country, because all four 09-02 runs are absent, so 09-03 has no D+1. w2 has 165–168. |
| **Arms** | `d1_19` is the T-1 19:00 UTC run's D+1 output. It is identical to the dashboard's D+1 on 99.4% of hours; the rest are missing runs. `d2_19` is the T-2 19:00 run's D+2. `oracle_d1` is the same artifact on the target-aligned feature row, with the matched weather. `t7_d1` is the same artifact on the row at T-7. `d7` is actual(T-7). |
| **Replay** | `load_training_data` + `create_all_features` over `predict_d2`'s 21-day lookback, with its calendar and weather overrides. Candidates: anchor day = run day − 0..6, times the three latest weather runs at or before the run instant, plus none. Match = the candidate closest to the stored `forecast_value`. **Exact**: ≤ 0.5 MW. **Near**: ≤ 0.5% of the value, with every other anchor day ≥ 3× further away. |
| **Replayable** | 21 of 24 countries have a local artifact version equal to the served one. **BE, DE and FR do not** (served `20251229_061652`, `20260105_210648`, `20260201_221331`; local `2026-04-04`), so they get served-output evidence only. |
| **Truth** | `energy_load` hourly means (ABL-332), 0.0 rows dropped |
| **Reads** | replica through `get_connection(readonly=True)`, i.e. `mode=ro`. Tables: `forecasts`, `energy_load`, `weather_data`. Nothing written outside `reports/`. |
| **Basis** | out-of-sample. The only fitted quantity is the §2.2 slope, fitted on w1 and read on w2. |
| **Interpreter** | the rail, Python 3.14 (`.venv`) |
| **Control** | my dashboard-D+1 daily WAPE matches the CEO's ABL-793 triage table on 65 country-days (IT HU FR FI DE): median \|diff\| **0.19 pp**, p90 0.58. The residual comes from the API's end-inclusive 25-hour windows. |

**Replay quality.** 15 countries reproduce exactly on ≥ 50% of D+1 hours in every week, and AT CZ EE FI HU LT NO PL SK reach 98–100% in w2. The misses cluster on particular run days. For example, all 24 IT hours on 09-04 miss; they were served from the 09-03 run as ingest recovered. That is consistent with anchor-day actuals revised after the run, but not verified. **SI, PT, GR and BG reproduce poorly** (exact 4–39%). Their served row is not identified, so they are excluded from the §1 claim and their oracle numbers are indicative only.

---

## 2. Q1: the served D+1 row is anchored 1–2 days early

Fleet, 20 replayable evaluable countries, exactly matched rows:

| anchor gap (days) | 1 | 2 | 3 | 4 | 5+ | **0** |
|---|---:|---:|---:|---:|---:|---:|
| D+1 (`d1_19`) | 3,725 | 2,452 | 518 | 347 | 84 | **0** |
| D+2 (`d2_19`) | — | 3,700 | 2,426 | 523 | 435 | **0** |

**The gap follows the target hour, which says what the run could see.** Target hours 00–13 mostly anchor at gap 1. Hours 14–17 are mixed, hour 18 is 57 vs 202, and hours 19–23 anchor at gap 2. So at the 19:00 UTC run the last observed actual is about 13:00–17:00 UTC of the run day, and for later hours the "most recent same-hour row" is the day before.

For D+1 the served `target_value_lag_1d` is therefore actual(T-2) or actual(T-3). `lag_7d` is actual(T-8) or actual(T-9). `roll_24h` ends one or two days early. For D+2 the anchor is 2–3 days early. That refines ABL-607's "exactly 2", which was its stated lower bound, and agrees with its argmin k = 3.

**The weekday signature is the history block and nothing else.** Relative bias by target weekday, median over the replayable fleet (+ = under-forecast):

| | Mon | Tue | Wed | Thu | Fri | Sat | Sun |
|---|---:|---:|---:|---:|---:|---:|---:|
| served D+1, w1 | +0.4 | +4.6 | +4.4 | — | +0.6 | −6.0 | −10.8 |
| served D+1, w2 | +4.5 | **+11.2** | **+10.5** | +0.6 | −0.4 | **−9.9** | −10.6 |
| oracle D+1, w2 (same artifact, aligned rows) | +0.4 | +0.5 | −0.6 | −0.1 | 0.0 | −1.2 | −0.4 |
| D-7 naive, w2 | −0.3 | +0.9 | +1.1 | +0.4 | −1.5 | −2.3 | 0.0 |

This is the pattern the CEO read off the API: weekday positive, weekend negative, larger in w2. It disappears when the same model sees aligned rows. FR, DE and BE are not replayable but show the same served signature: FR w2 is Mon +10.5 / Tue +9.7 / Sat −12.2 / Sun −10.6. FR's served D+1 fits actual(T-2) better than actual(T) in w0 and w2.

**Item 4 of the brief (D+1 ≈ D+2) is the dashboard's band, not the defect.** The dashboard's D+2 takes the latest vintage with `horizon_hours` in [24, 54]. Only **35.5%** of those hours are genuine D+2 (the T-2 19:00 run). **64.5%** come from T-1 runs, and 26–37% are byte-identical to D+1. That share is the same in every country because the schedule is. ABL-607 §6 found the same for the scorecard's 24–64h band.

---

## 3. Q2: same defect, bigger displacement

### 3.1 Decomposition

Paired D+1 WAPE (%) on identical hours per week. The oracle uses the served model, weather and calendar, so **served minus oracle is what the mis-anchored history block cost**. The oracle is *not servable*: its `lag_1d` and `roll_24h` include hours a 19:00 run cannot see. It is an upper bound on aligned history, not a candidate forecast.

| cc | served w1→w2 | oracle w1→w2 | served−oracle w1→w2 | share of rise | D-7 w1→w2 |
|---|---|---|---|---:|---|
| **IT** | 7.95 → **14.52** | 3.90 → 5.15 | 4.05 → **9.37** | 81% | 6.48 → 8.23 |
| **HU** | 5.46 → **11.13** | 3.39 → 4.45 | 2.07 → **6.68** | 81% | 5.91 → 5.98 |
| **HR** | 5.96 → **10.58** | 3.01 → 4.16 | 2.94 → **6.43** | 75% | 5.98 → 10.05 |
| SI ‡ | 11.77 → 17.69 | 4.47 → 6.30 | 7.29 → 11.39 | 69% | 2.99 → 5.09 |
| SK | 6.85 → 9.26 | 1.84 → 2.34 | 5.01 → 6.92 | 79% | 2.92 → 2.54 |
| PL | 7.07 → 9.29 | 1.63 → 1.58 | 5.44 → 7.71 | ~100% | 2.68 → 2.06 |
| LV | 7.33 → 8.82 | 4.32 → 4.92 | 3.01 → 3.90 | 60% | 5.01 → 6.52 |
| FI | 1.92 → 3.45 | 1.77 → 2.58 | 0.14 → 0.87 | 48% | 2.84 → 3.81 |
| EE | 8.68 → 13.97 | 7.84 → 11.33 | 0.84 → 2.63 | 34% | 10.81 → 14.97 |
| FR † | 6.25 → 8.43 | — | — | — | 2.89 → 4.27 |
| *AT (fell)* | 11.04 → 7.86 | 1.88 → 1.39 | 9.16 → 6.47 | | 2.33 → 2.49 |
| *CZ (fell)* | 9.14 → 7.24 | 1.55 → 1.47 | 7.59 → 5.78 | | 1.91 → 2.09 |
| **fleet median (20)** | 7.80 → 8.97 | 2.87 → 3.86 | 4.20 → 5.47 | | 5.00 → 5.80 |
| **flagged median (9)** | 7.07 → 10.58 | 3.39 → 4.45 | 3.01 → 6.68 | | 5.01 → 5.98 |

‡ poorly reproduced (§1), indicative only. † not replayable. n = 144 (w1) / 165–168 (w2) hours per country.

EE is the exception: most of its rise is a genuinely harder week (its oracle and D-7 rose too). IT, HU, HR, SK and PL are the defect.

### 3.2 The CEO's swing test, run on the served rows

For each target day I compared the defect's daily level cost (served relbias − oracle relbias) with how far the served history block sat from the aligned one (`roll_24h_mean` displacement, % of load).

- **The two track each other almost perfectly.** Across w1 and w2 days the correlation is |r| ≥ 0.84 in all 20 countries (median −0.96). Negative means the served block sits high, so the forecast sits high.
- **The per-unit cost did not change between weeks.** A through-origin slope fitted on w1 predicts w2's mean daily |cost| from w2's displacement: fleet median **6.94 predicted vs 6.91 observed**. Flagged countries: HU 8.71 vs 8.43, HR 7.84 vs 6.87, LV 7.89 vs 7.28, PL 7.72 vs 9.42, SK 6.92 vs 8.29, IT 9.20 vs 12.30. FI is not fittable (w1 displacement ≈ 0).
- **The displacement grew.** Flagged median |roll_24h displacement| went **5.5% → 8.8%**, and |cost| **5.7 → 8.3**.

IT, per target day. `gap` is the anchor gap of exactly matched D+1 rows, and temperature is the matched forecast's daily mean:

| day | srv WAPE | oracle | served relbias | roll24 displacement | gap | °C |
|---|---:|---:|---:|---:|---|---:|
| Tue 09-01 | 6.10 | 3.79 | +5.9 | −3.1 | 3–4 | 27.9 |
| Wed 09-02 | 7.25 | 3.92 | +7.3 | −4.4 | 4–5 | 27.3 |
| Sun 09-06 | 16.90 | 5.11 | −16.9 | +23.9 | 1–2 | 27.0 |
| Tue 09-08 | 14.00 | 3.65 | +14.0 | **−22.0** | 1–2 | 27.3 |
| Wed 09-09 | 15.47 | 3.67 | +15.5 | **−23.6** | **2–3** | 26.9 |
| Sat 09-12 | 19.99 | 6.67 | −20.0 | +20.2 | 1–2 | 23.8 |
| Sun 09-13 | 30.03 | 13.43 | −30.0 | +29.8 | 1–2 | 23.9 |

HU, HR and PL show the same days: Tue/Wed displacement −14 to −20%, Sat/Sun +11 to +20%. Note PL. Its truth-only swing did *not* grow (contrast 16.8% → 16.4%), yet its cost grew, because its 09-01/02 anchors were stale weekday rows (gap 2–4) and its 09-08/09 anchors were Sunday/Monday rows.

### 3.3 What made the displacement bigger

- **A sharper week, not the end of August holidays.** Truth-only weekday/weekend contrast, w1 → w2: IT 21.3 → **34.6** (July mean 19.7), HR 11.8 → **24.6** (8.1), HU 15.4 → 17.6, EE 7.8 → 14.2, FR 10.7 → 14.8. Mean |two-day swing|: IT 13.2 → 20.0, HR 6.7 → 11.7, HU 9.1 → 12.1, EE 6.1 → 12.0. IT's w2 contrast is above every July week, so this is not a return to normal. The matched temperature forecasts fall through the week (Tue → Sat: IT 27.3 → 23.8 °C, HU 24.1 → 13.7, HR 19.1 → 9.1, PL 19.4 → 12.9), and IT daily mean load fell from ~40.4 GW on Tue to ~24.5 GW on Sun. Across countries, the change in cost ranks with the change in two-day swing at Spearman **+0.45**, and with the change in contrast at **+0.50** (n = 20). That is moderate. The served-row displacement is the better predictor because it also carries the anchor-gap mix.
- **A flattering baseline.** The w1 Tue/Wed forecasts (09-01/02) came from runs made stale by the 08-29..09-02 ingest collapse (fleet anchor gap 3–5). Those stale anchors landed on weekday rows, and IT's Tue/Wed displacement was −3/−4% instead of the −22/−24% a normal run carries into that transition. The CEO's "compare like weekdays" caveat understates this: the baseline's Mon–Wed were served on abnormal anchors.

---

## 4. Q3: other serving inputs, and Q4: ABL-608

**Serving inputs.**
- **Weather.** In the 15 well-reproduced countries, 95–100% of exact matches (LV lowest at 95.5%, HR 96.8%, the rest ≥ 98.7%) use the latest run at or before the 19:00 instant (the 12:00 UTC run, median age **7.0 h**), in all three weeks. There is no weather degradation, and the oracle, which uses the same weather, stays flat.
- **Calendar and holidays.** The served load artifacts carry no holiday features (26 columns), and the calendar override is deterministic, so there is nothing to degrade.
- **Lag rows.** Every serving frame holds 520–528 of its 528 lookback hours (minimums: EE 520, LV 521, BG 525, SE 526), with **0** zero-valued and **0** duplicated hours.
- **Outage and ingest staleness.** This shows up directly as anchor gap (fleet D+1, exact rows):
  - Targets 08-31, 09-01 and 09-02 anchor at gaps 2–3, 3–4 and 4–5 (the ingest collapse, an ABL-71-class event).
  - Target **09-09** has 165 of 342 rows at gap 3 (the 09-07/08 ENTSO-E outage).
  - Every other day is at 1–2.
  - The outage compounds 09-09 only.
- **Deploys.** Stored values reproduce exactly from the same artifact and code path on targets after the 09-10 10:13Z rebuild (HU, PL, SK, CZ, EE, FI, LT, NO: 100% exact in w2). The rebuild did not change what serving computes.
- **Actuals.** IT 09-13 carries two partial hours, 21:00 and 23:00: 3 of 4 quarter-hours each, with raw values of 8,505–10,457 MW. Excluding them, IT w2 served goes 14.52 → 14.08 and oracle 5.15 → 4.80, so they inflate Sunday but not the finding. Elsewhere in the window there are ≤ 4 partial hours per country and no zero rows.

**Does ABL-608 as written fix it?** The mechanism is right, and the structural half, "extend the serve-faithful builder to load and delete the proxy-row branch", fixes D+1 too, because `predict_d2` serves both horizons through that branch. But as written it would not deliver a D+1 fix, for three reasons:

1. **Scope and title are D+2 only**, and the bar is a D+2-band read. The user-facing D+1 is the larger and more visible cost.
2. **The feature spec is not buildable as stated.** "Lags at 2/3/7/14 days, rolling windows ending at T-48h": at the 19:00 run the last observed hour is about 13:00–17:00 UTC, so T-48h is unobservable for D+2 target hours ≳ 14. Applied to D+1, the spec would throw away a day of information that D+1 can hold.
3. **The bar reads a band**: 24–64h, or the dashboard's 24–54h, which is 64.5% T-1-run rows. It does not read run offsets.

**Re-scope, exactly:**
- **Title/scope:** "Fix the load anchor at D+1 and D+2". Both horizons, one change, because it is one branch.
- **Feature availability rule instead of fixed offsets.** A same-hour lag of k days is in the spec only if it is observed for *every* target hour at the run instant: **k ≥ 2 for D+1 and k ≥ 3 for D+2** at the 19:00 UTC run. Rolling windows end at the last observed hour. Either fit one artifact per horizon, or one lead-conditioned artifact. The serve-faithful builder refuses anything later.
- **Bar:** the D-7 seasonal naive, per country, on a paired daily interval, **for both D+1 and D+2**, read on genuine run offsets (T-1 and T-2 runs), not on horizon bands.
- **Carry over unchanged:** the EE/LV row-shift defect, the NL holdout, and the rule of no bias/affine calibration. The same holds here: the oracle's weekday bias is flat at ±1.2 while the served bias alternates ±11, so there is no constant to remove.

---

## 5. Recommendation: one change

**Promote ABL-608, re-scoped as in §4 to cover D+1 and D+2.** This diagnosis does not depend on ABL-622's re-read: the mechanism is proven on served rows. ABL-608's own sequencing let the serving-path half proceed ahead of that re-read, and nothing here argues for waiting.

Not recommended, with the measurement behind each:
- **A serving-only T-7 proxy row**, i.e. using the same-weekday row a week back. Fleet median paired D+1 WAPE was 5.36 / 5.00 / 5.32 against served 7.39 / 7.80 / 8.97, but it is essentially the free D-7 naive (6.11 / 5.00 / 5.80). It also fails where D-7 fails: in w0, IT got 9.09 against served 6.79, HU 9.40 against 7.40, GR 11.71 against 8.77. It would need its own gate read to serve, and the D-7 baseline costs nothing.
- **Rolling back any deploy.** Nothing in the window changed what serving computes (§4).
- **Bias or affine correction** (§4, and ABL-607 §5.4).

---

## 6. Caveats

- **Three weeks, one week pair.** §3 is a descriptive decomposition, not an interval-backed effect estimate. The §2 mechanism claims (anchor gap, weekday signature) do not depend on the window.
- **The oracle is leaky by construction.** It bounds what aligned history is worth; it is not a forecast. A servable refit will land between the oracle and the served numbers.
- **BE, DE and FR are not replayable.** Production serves older versions than `models/` (ABL-607 noted DE; BE and FR are new). The local artifact tree is not production's serving set for those three. If FR needs a served-row proof, the next measurement is to pull the three prod artifacts read-only and replay.
- **SI, PT, GR and BG replay poorly** (§1). Their served row is not identified.
- **Contamination.**
  - ABL-111/109 (zero actuals): 0 rows in the window.
  - ABL-67 does not touch load.
  - ABL-71-class ingest staleness is measured directly as anchor gap (§4), not assumed.
  - IT 09-13 partial hours: sensitivity given in §4.
  - NL is scored in the JSON and held out (ABL-277/505/506).
- **Actuals are still inside ENTSO-E's ~7-day revision window for w2.** Expect second-decimal movement, as in ABL-607 §3.1.

---

## 7. Queued, not run

A note on **ABL-622** (the date-gated D+2 re-read, act date 2026-09-19) records what this changes for that read. No re-score now, per the anti-churn rule.
