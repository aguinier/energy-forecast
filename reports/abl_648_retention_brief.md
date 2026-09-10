# ABL-648 — `weather_observation` retention: decision brief for the Board

**Author:** Forecasting Scientist · **Date:** 2026-09-10 · **Scope:** read-only analysis. No deletion is proposed, authorised or performed here.
**Evidence:** `reports/abl_648_retention_model.json`, reproduced by `.venv\Scripts\python.exe scripts/abl648_weather_retention_model.py --replica-db C:\Code\able\data\energy_dashboard.db`.
**Protocol:** replica opened `file:...?mode=ro`. **No prod query was issued and `weather_observation` was never `COUNT(*)`-ed**, per the constraint on this issue.

---

## 1. Recommendation

**Retain on vintage depth (`fetched_at`), not on calendar age. Adopt 30-day full vintages plus a latest-vintage tail — the Board's existing Option B from ABL-212 — and decide it inside the next ~6 weeks.**

Two things the Board should know before reading further, because both change the question as it was posed to me:

1. **All four calendar horizons you asked me to price — 6, 12, 18 and 24 months — free approximately nothing.** They are not a weaker version of the right answer; they are a different axis from the one the volume is on. Numbers in §3.
2. **The forecast programme reads nothing from `weather_observation`, and no retention horizon on it forecloses any model work I can name — including ABL-338's.** That constraint is in a different table. Evidence in §2 and §4.

I am the wrong owner for the *execution* of any of this and am not asking to own it. What this brief settles is the product question: **how much weather history the forecast programme needs.** The answer is "none of what is in this table", and that should make the Board's call substantially easier than it looked.

---

## 2. What the forecast programme actually reads from `weather_observation` today

**Nothing. Not one model, not one evaluator, not one backtest.**

Every weather read in `energy-forecast` goes to a different table, **`weather_data`**. Verified by exhaustive search of the repo: the only files in `energy-forecast` that contain the string `weather_observation` are the two I wrote for this brief. This independently re-confirms what ABL-273 reported on 2026-08-12 and it has not drifted since.

`weather_data` and `weather_observation` are **not** one derived from the other. They are separately populated by two independent cron entries in `energy-data-gathering/docker/crontab`:

| table | writer | cadence |
|---|---|---|
| `weather_data` (legacy) | `scripts/update_weather.py --forecast` | daily 15:00 UTC |
| `weather_observation` (versioned) | `update_weather_observation_hourly.py` + `update_weather_observation.py` | hourly + 3×/day |

`energy-data-gathering/WEATHER_DB.md` names the dual-write explicitly and lists "migrating `weather_data` consumers (dashboards, `energy_forecast` XGBoost) onto `weather_observation`" as **out-of-scope future work**. So deleting rows from `weather_observation` cannot reach the forecast programme's inputs through any path that exists today.

### What we read, per model, and how far back

All reads below are `weather_data`, columns `temperature_2m_k`, `relative_humidity_2m_frac`, `wind_speed_10m_ms`, `wind_speed_100m_ms`, `shortwave_radiation_wm2`, `direct_radiation_wm2`, `diffuse_radiation_wm2`.

| model / component | code | population | reach back |
|---|---|---|---|
| XGB/LGBM/CatBoost per-country (load, price, renewables) — training | `db.load_weather_data` | `data_quality='actual'` | **2023-01-01** (whole archive) |
| the same, serving | `db.load_weather_forecast` | `'forecast'`, freshest run | target day only |
| `RenewableFeatureBuilder` (solar/wind gates) | `wind_features._load_weather_archive` | `'forecast'` | **2026-01-11** |
| Chronos-2 (champion, net_position) | `chronos2/input_builder` | `'forecast'`, `MAX(forecast_run_time)` | 672 h context (28 d) |
| V014 challenger | `challengers/v014_features` | `'forecast'`, `run_time <= run_ts` | **2026-01-11**; W01-W10 weather-blind |
| TSO-correction runner | `tso_correction_forecaster` | `'forecast'`, freshest run | target day only |

Measured on the replica, all **24 supported countries**, uniformly: `actual` spans 2023-01-01 → 2026-09-09 (903,192 rows); `forecast` spans 2026-01-11 → 2026-09-22 (2,129,568 rows). No country starts later than another. `weather_data` totals ~3.0M rows against `weather_observation`'s ~1.2 billion — **roughly 0.25%** of the row count, and it is the 0.25% we use.

> **Caveat, stated rather than buried.** The migration named in `WEATHER_DB.md` is real future work. If the forecast programme ever moves onto `weather_observation` — which I would want, for per-NWP-model and per-zone weather we cannot get today — then whatever depth survives becomes our fit-window bound at that moment. §5 handles this, and it is the only reason I am not recommending the most aggressive option.

---

## 3. The five retention horizons, priced

`weather_observation`'s **first `fetched_at` is 2026-04-22**. The table is **141 days old.** Every figure below is **modelled**, from ABL-163's four rowid→`fetched_at` anchors and ABL-206's 372.0 GiB sizing, at 405.4 bytes/row — not measured, because measuring it is the thing this issue forbids. Modelled size today: **458.9 GiB, ~1.215 B rows.**

### On `fetched_at` (vintage age) — the natural reading

| horizon | cutoff | freed today | headroom bought | first date it frees anything |
|---|---|---:|---|---|
| 6 months | 2026-03-11 | **0.0 GiB** | none | 2026-10-22 |
| 12 months | 2025-09-10 | **0.0 GiB** | none | 2027-04-22 |
| 18 months | 2025-03-11 | **0.0 GiB** | none | 2027-10-22 |
| 24 months | 2024-09-10 | **0.0 GiB** | none | 2028-04-21 |
| keep all | — | 0.0 GiB | none | — |

**A 6-month policy adopted today is indistinguishable from "keep all" until 2026-10-22, and does not free a material amount until well after the wall.** The 12/18/24-month options do not begin to act until 2027, 2027 and 2028 — all of them after the deadline they were meant to solve. At the 90% wall a 6-month rule would free 65.5 GiB against 628.7 GiB retained: **it arrives late and it arrives small.**

### On `valid_at` (target time) — the other reading, and it is worse

The only population with `valid_at` older than the table itself is the ERA5 backfill (one vintage per location-hour, from 2024-01-01). The vintage-bearing populations are forward-looking. So a calendar cut on target time deletes ERA5 and essentially nothing else:

| horizon | cutoff | freed | % of table | % of ERA5 deep history destroyed |
|---|---|---:|---:|---:|
| 6 months | 2026-03-11 | 2.6 GiB | 0.57% | **81.4%** |
| 12 months | 2025-09-10 | 2.0 GiB | 0.44% | **62.9%** |
| 18 months | 2025-03-11 | 1.4 GiB | 0.31% | 44.3% |
| 24 months | 2024-09-10 | 0.8 GiB | 0.18% | 25.7% |

**This is the trap in the calendar axis: it deletes ~100% of the only deep weather history in the table to recover ~0.5% of the volume.** The entire ERA5 archive is 3.2 GiB — a rounding error against 458.9 GiB. Whatever else the Board decides, **ERA5 should be exempt by name**, not by a date filter that happens to spare it.

### The axis the volume is actually on

The table is large because of **vintage multiplicity**, not history. `PRIMARY KEY (source_id, location_id, valid_at, fetched_at)` means every re-fetch of the same triple is a new immutable row. Modelled: **~44.5 M distinct `(source, location, valid_at)` triples carrying ~27.3 vintages each.**

| option | retained | freed | freed % | steady-state growth |
|---|---:|---:|---:|---|
| latest vintage only | 16.8 GiB | 442.0 | 96.3% | 0.095 GiB/day (**÷30**) |
| **30 d full vintages + latest tail** | **99.7 GiB** | **359.2** | **78.3%** | **0.095 GiB/day (÷30)** |
| 60 d full vintages | 182.5 GiB | 276.3 | 60.2% | 0.095 GiB/day (÷30) |
| 90 d full vintages | 265.4 GiB | 193.5 | 42.2% | 0.095 GiB/day (÷30) |

**This is the decisive contrast.** Every vintage option converts unbounded ~1 TB/year growth into a **bounded steady state ~30× smaller**. Every calendar option leaves the growth rate exactly as it is. The choice among 30/60/90 days trades one-off reclaim against replay depth; the choice of axis is what actually solves the problem.

---

## 4. Which model work each horizon forecloses

**None of them foreclose any model work in the forecast programme, on any horizon, including latest-vintage-only.** We do not read the table (§2).

On ABL-338 specifically — the CEO's brief cites it as showing history length is the binding constraint on renewable fits, with AT and DE under one seasonal cycle. **I reproduced that constraint and it is real, but it is not in a weather table.** Measured read-only, `solar_mw` history for the four ABL-338 countries plus ES:

| country | `energy_renewable` (the training source the 4 solar artifacts carry) | `energy_generation` |
|---|---|---|
| **AT** | 2025-11-07 → **10.1 months** | 2021-01-01 → 4.7 y |
| **DE** | 2025-09-08 → **12.0 months** | 2021-01-01 → 4.7 y |
| BE | 2024-01-01 → 2.7 y | 2021-01-01 → 4.7 y |
| ES | 2025-11-08 → 10.1 months | 2021-01-01 → 4.7 y |
| FR | 2023-01-01 → 3.7 y | 2021-01-01 → 4.7 y |

AT and DE are indeed under/at one seasonal cycle — **in `energy_renewable`, a production table, not in weather.** The remedy already exists and is documented: train on `energy_generation` (`--renewable-source energy_generation`), which reaches 2021-01-01 for all five. Weather is not the binding constraint for these fits: `weather_data` actuals go back to 2023-01-01, i.e. 2.7 seasonal cycles, deeper than AT's or DE's targets.

So the two constraints are on **different tables in different repos, and a retention policy on `weather_observation` can neither relieve nor worsen ABL-338.** I want to be unambiguous about this because it is the premise the decision was being held on: **there is no forecast-programme cost to price here.**

The one genuine forward cost is the migration in §2's caveat, and it is a cost in *vintage depth*, not calendar depth — which is exactly what §5 protects.

---

## 5. Is downsampling a real middle option?

**Two different things are being called "downsampling", and only one of them is a real option.**

### (a) `valid_at` downsampling, hourly → 3-hourly — **recommend against**

- **Volume:** retains 153.0 GiB, frees 305.9 GiB. But growth only drops by a factor of 3, to ~0.96 GiB/day. **It does not produce a bounded steady state — it postpones the same wall.** The vintage multiplicity that causes the growth is untouched.
- **Model cost, and this one is disqualifying:** every model we run is hourly, and the feature that suffers most is **shortwave / direct / diffuse radiation**. A 3-hourly sample interpolated back to hourly destroys the diurnal shape precisely at sunrise and sunset — the shoulder hours. ABL-337/338 established the shoulder as the one remaining user-visible defect in solar (DE 2026-08-14 19:00 at 204.7 MW with the sun 3.9° down), and the geometry feature that halved it works *because* it separates hours the radiation channel cannot. Coarsening that channel re-creates the defect the programme just spent two issues fixing.
- **It also cannot be undone.** Unlike vintage thinning, which discards redundant copies, this discards distinct information.

Worse volume properties *and* a real model cost. There is no reading on which this is the right instrument.

### (b) `fetched_at` vintage thinning — **this is the real middle option, and it is the recommendation**

Keep every vintage inside a recent window (30 d), keep exactly one vintage per triple outside it. This is (i) bounded, (ii) lossless for every consumer that reads one vintage per issue time, and (iii) the only option that separates the two things the schema conflates: *scoring recent forecasts honestly* (needs full vintages, weeks) versus *reproducing an old run* (needs one vintage, not 27).

**Cost to build.** Modest, and mostly procedure rather than code:

| item | estimate |
|---|---|
| filtered copy-out `INSERT ... SELECT` with a `MAX(fetched_at)` group-wise pick | ~1 day |
| ERA5 exemption by `source_id` + a positive test that ERA5 survives | ~0.5 day |
| staged write under ABL-181/215/256 (fresh re-enumeration, pre-image, single txn, independent post-commit verification) | ~1 day, owner: Founding/Deployment Engineer |
| ongoing prune job + monitoring | ~0.5 day |

**The real cost is not engineering days — it is that the window to do it at all is closing.** From `free_now = 322.1 GiB`, at the measured 3.906 GiB/day, with a 50 GiB operating margin:

| reclaim path | copy size | fits today? | window closes |
|---|---:|---|---|
| full `VACUUM INTO` (no policy) | 466.3 GiB | **no — already closed** | — |
| filtered copy-out, 3-hourly | 153.0 GiB | yes | **2026-10-10** |
| filtered copy-out, 30 d vintages | 99.7 GiB | yes | **2026-10-24** |
| filtered copy-out, latest-only | 16.8 GiB | yes | 2026-11-14 |

Deleting rows in SQLite reclaims no disk without a copy-out, and **the unfiltered copy-out is already impossible.** A *filtered* copy-out is the reclaim path — it is both the deletion and the vacuum in one pass — and the 30-day variant needs the decision made and executed **before roughly 2026-10-24.** That is ~6 weeks, and it is why I am recommending 30 days rather than 60 or 90: the 60/90-day copies are larger and their windows close sooner.

---

## 6. One correction to the deadline in this issue's framing

This issue and the CEO's brief both work from ~3 GB/day and a **~January 2027** wall (150 days of headroom measured 2026-08-11). **I get an earlier date, and the Board should have it.**

From `data/ops-status-snapshots.jsonl`, **1,326 local snapshots over 14.0 days**, 2026-08-27 → 2026-09-10 — no prod query issued. Prod is **585.04 GiB used of 907.13 GiB (64.5%), 322.1 GiB free.** Following ABL-212's own lesson, I decompose rather than fit a line, and quote daily medians because they are phase-matched to the daily backup staircase that produced ABL-212's false alarm:

| rate | GiB/day | GiB/week |
|---|---:|---:|
| daily medians (14 d) | **3.906** | 27.3 |
| baseline, step events excluded | 2.881 | 20.2 |
| ABL-212's committed rate | 2.214 | 15.5 |

Step events over the window: 58 events, +150.02 GiB written, −132.34 GiB deleted, net +17.68 GiB retained. So the staircase is *not* fully self-cancelling here, and the median rate is the honest one to plan against.

| wall | at 3.906 GiB/day | at 2.881 | at ABL-212's 2.214 |
|---|---|---|---|
| **90% of volume** | **2026-11-08** (59 d) | 2026-11-29 | 2026-12-23 |
| 100% | 2026-12-01 (82 d) | 2026-12-30 | 2027-02-02 |

**The measured rate is 1.76× the rate ABL-212's Option C was sized against, and the 90% wall is 2026-11-08 — about two months earlier than this issue assumes.** I would not over-read a 14-day window, which is why all three rates are shown; but even on ABL-212's own committed rate the 90% threshold falls on 2026-12-23, still ahead of January. **On every rate I can defend, the wall precedes the date the decision is currently being planned around.**

This is the finding I would most want acted on. It does not change which option is right — it compresses the time available to take it, and it lands inside the reclaim windows in §5.

---

## 7. What I recommend the Board decide

1. **Adopt the vintage-depth axis and reject the calendar axis.** 6/12/18/24 months free 0.0 GiB on `fetched_at` and 0.18–0.57% on `valid_at`. Neither is a partial solution.
2. **Adopt 30-day full vintages + latest-vintage tail** (ABL-212's Option B). Retains 99.7 GiB, frees 359.2 GiB, converts ~1 TB/year into ~35 GiB/year. 60/90-day are defensible if replay depth is valued more than reclaim; their copy-out windows close sooner.
3. **Exempt ERA5 by `source_id`, explicitly.** It is 3.2 GiB — 0.7% of the table — and it is the only deep weather history we have. Do not let any date filter reach it.
4. **Do not downsample `valid_at`.** Worse volume behaviour than vintage thinning *and* a real cost to solar shoulder accuracy.
5. **Decide by ~2026-10-10 so execution completes by ~2026-10-24.** The unfiltered `VACUUM INTO` path is already gone; the 30-day filtered copy-out is the reclaim mechanism and its window closes then.
6. **Re-check §6's growth rate before committing.** My window is 14 days from a local snapshot log. If ops can confirm ≥30 days, the wall date firms up; if it disagrees with 3.906 GiB/day, item 5's dates move and nothing else in this brief does.

**No deletion is authorised by this brief.** Execution is a Board-approved staged write under the ABL-181/215/256 procedure, owned by the Founding or Deployment Engineer, with a fresh re-enumeration immediately before writing — the ABL-67 lesson, where 48% of an approved delete list had self-repaired between approval and execution.

---

## 8. Caveats

- **Every `weather_observation` byte figure here is modelled, not measured** — from ABL-163's rowid anchors and ABL-206's 372.0 GiB, at 405.4 B/row. The table is absent from the replica and must not be counted on prod. The *shape* of the conclusion (calendar frees ~0, vintages are 27× the volume) follows from `PRIMARY KEY (…, fetched_at)` and the first-`fetched_at` date, and does not depend on the model being precise.
- **Measured figures** are the replica probe (§2, §4) and the volume growth series (§6). Everything else is labelled modelled in the JSON.
- **Contamination:** none of ABL-71 / ABL-67 / ABL-111 / ABL-109 touches this analysis. No forecast accuracy is scored here; the production reads in §4 are history-extent only (`MIN`/`MAX`/`COUNT` of non-null `solar_mw`), which zero-as-missing rows do not distort.
- `forecast_run_time` is **NULL throughout `weather_observation`** — Open-Meteo does not expose it. `fetched_at` is the only vintage key, so a policy cannot dedupe by NWP run even if that were preferable.
- I could not locate the AT/DE seasonal-cycle figure verbatim on ABL-338; §4 is my own measurement of the same claim, which reproduces it and locates it in `energy_renewable`.
- I do not own execution and am not asking to. This brief answers the product question only.
