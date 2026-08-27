> **Archived from `CLAUDE.md` on 2026-08-27** (companion to the ABL-536
> energy-dashboard-frontend trim). Historical narrative, incident forensics
> and dated measurements; `file:line` references are frozen as of the archive
> date. The durable rules distilled from this material live in the repo-root
> `CLAUDE.md`; where they conflict, the root file wins.
# Model Details

## Model Details

### Features

**Time Features:**
- hour, day_of_week, month, is_weekend
- Cyclical encoding: hour_sin/cos, day_sin/cos, month_sin/cos

**Lag Features (same hour):**
- D-1 (24h ago)
- D-7 (1 week ago)
- D-14 (2 weeks ago)

**Rolling Statistics:**
- 24h and 168h (1 week) rolling mean, std, min, max

**Holiday Features:**
- is_holiday - Binary flag for public holidays
- days_to_holiday - Days until next holiday (capped at 7, `src/features.py:177`)
- days_from_holiday - Days since last holiday (capped at 7, `src/features.py:185`)
- is_bridge_day - Workday between holiday and weekend

> **Declared, but in no serving artifact** (ABL-386/ABL-394, measured 2026-08-13;
> mechanism corrected by ABL-407). All 66 artifacts that carry a
> `feature_columns` list carry none of these four, and dropping exactly those
> four reproduces the served list length on all eight types
> (23/23/26/25/27/25/24/24) — one plumbing gap, not eight drifts. They are live
> for the **next** fit of any country and have never been evaluated on any target
> — ABL-386's read on solar is MIXED. The frozen lists and the recorded gap are
> in `tests/feature_list_manifest.json`; the narrowing now warns instead of
> dropping silently (`select_feature_columns`, `src/features.py:534`).
>
> **Why they are missing is provenance, not a regression.** Do not repeat the
> earlier story that ABL-338 (`5cf2296`) threaded `country_code` into
> `create_all_features` and so made them live; it does not reproduce.
> `git show 5cf2296 --stat -- scripts/train.py` is empty, and at `5cf2296^` the
> training site already read
> `create_all_features(df, forecast_type, country_code=country_code)`. Both the
> four names and that threading trace to `996c45a` *Initial commit*, 2026-03-05.
> The one pre-ABL-338 site that omitted `country_code` was
> `evaluate_against_baselines` — the **validation** frame, which writes no
> artifact's `feature_columns`; that is **ABL-397**, and it is a different defect.
> 60 of the 66 artifacts were saved 2025-12-26..2026-02-23, before this repo
> existed, so no current training path produced them. The remaining **6**
> (BE/DE/FR × load, price) were saved 2026-04-04, a month *after* the migration,
> and still carry none of the four — for those the cause is **not established**.
> Full measurement: `reports/abl_407_holiday_gap_provenance.md`.
>
> **This list is not what the two gate harnesses fit.** They declare their own
> `FEATURE_COLUMNS` and never call `get_feature_columns()`, which is how the solar
> harness came to sit two names short of an ABL-338-current fit until ABL-395 —
> see "Neither harness fits the list `get_feature_columns()` builds" above. Both
> harness lists are frozen in the same manifest, under `gate_harness`.
>
> **The solar null does not transfer to load** (ABL-393).
> `scripts/abl338_solar_holdout.py --type load|price` fits the same arms on the two
> aggregate targets, paired by seed over the standing eight seeds — the instrument
> ABL-386 named as its own weakest. Registration `experiments/ABL393/config.json`,
> verdict and numbers `reports/abl_393_load_price_holiday_verdict.md`. **Do not
> read ABL-386's MIXED as covering the other seven types**: it was registered on a
> target whose prior was "no effect" — solar output is set by irradiance — and it
> says so. On load the prior is the opposite, and `control_noholiday` there is
> *exactly* the serving list (26 names on load, 25 on price, all 48 artifacts, name
> for name and in order), so the contrast is what is served against what the next
> retrain builds, with nothing else moving.

Three things about that read are reusable and would otherwise be re-derived:

- **`create_lag_features` shifts by rows, so a source gap poisons the fortnight
  after it.** `days * 24` is a day only on a gapless hourly frame, and
  `energy_price` is not gapless: measured 2026-08-13, AT is missing **2,236 h** and
  DE **2,483 h**, almost all of it 2025-09 to 2025-12 (AT's largest single hole is
  1,651 h, DE's 1,309 h), while `energy_load` over the same span misses one 27–29 h
  outage on 2026-02-15 common to all four majors plus 26 h of FR over New Year
  2026. A holdout placed within 14 days of a hole scores rows whose D-1/D-7/D-14
  lags reach across it. This is what disqualified December for price in ABL-393 —
  AT and DE are 67.3% covered there — and `reports/abl_393_source_gaps.json` is the
  regenerable inventory. **Check it before choosing a window**, on either table.
- **December is not the densest holiday window of the year**, for three of the four
  majors. Measured on the `holidays` calendar: 2025-12-06..2026-01-18 holds AT 5,
  BE 2, DE 3, FR 1 holiday days against 2026-04-30..2026-06-12's AT 4, BE 4, DE 3,
  FR 4 — Labour Day, Ascension, Whit Monday, FR's 8 May and AT's Corpus Christi all
  fall in the second. What December has instead is a contiguous low-demand
  fortnight, which `days_to_holiday`/`days_from_holiday` mark and a count of red
  days does not.
- **A holiday is 2–5 days in a 44-day window, so an all-hours mean dilutes a
  holiday effect roughly twentyfold.** `--holiday-subsets` scores each arm over
  `holiday`, `holiday_affected` (holiday, bridge day, or within a day of one) and
  `ordinary`, from `src/features.holiday_subset_masks` — one predicate, shared with
  the pre-fit density probe, so a window cannot be registered under one definition
  and read under another. The two subsets partition the holdout and MAE × n is a
  sum of absolute errors, so their gains add to the total exactly: **which subset
  the gain lands in is the internal check on any headline here.**

**Weather Features:**
- Load: temperature, heating/cooling degree days
- Price: temperature, wind speed, solar radiation
- Renewable (total): solar radiation, wind speeds
- Solar: shortwave/direct/diffuse radiation
- Wind (onshore/offshore): wind speed at 10m and 100m
- Hydro: temperature, precipitation
- Biomass: temperature

### Supported Algorithms

| Algorithm | Description |
|-----------|-------------|
| XGBoost | Default. Gradient boosting with regularization |
| LightGBM | Fast gradient boosting with histogram-based learning |
| CatBoost | Gradient boosting with built-in categorical handling |
| Chronos-2 | Foundation model (120M params). Requires GPU + separate venv |

### XGBoost Configuration

```python
{
    'n_estimators': 500,
    'max_depth': 8,
    'learning_rate': 0.05,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
```

### Chronos-2 Configuration (ported from netpredict2)

```python
{
    'model': 'amazon/chronos-2',       # 120M param foundation model
    'context_length': 672,             # 4 weeks of hourly data
    'prediction_length': 24,           # 1 day ahead
    'fine_tune_steps': 5000,           # Cosine LR schedule
    'learning_rate': 1e-5,
    'quantiles': [0.1, 0.2, ..., 0.9] # 9 quantile levels
}
```

`prediction_length: 24` is the *published* horizon, not the horizon the model
is asked for. See below.

**Covariates (suffix convention from netpredict2):**
- **Suffix-0** (future-known, through D+2): Weather (Open-Meteo), time features, holidays
- **Suffix-1** (past-only, through D+1): TSO load/generation forecasts, DA prices, neighbor features

### The context ends where the data ends, not where the schedule says

A D+2 run fires around 06:00 UTC on day D for the whole of day D+2. The
schedule's nominal context cutoff is D+1 23:00 — roughly **42 hours after the
run actually happens**. No observation exists for that span.

`build_for_country` therefore measures the last real observation
(`_last_available_timestamp`) and ends the context there, then forecasts across
the gap *and* the target day, and the caller publishes the **last 24** points.
`future_index` names their timestamps; `forecast_chronos2.py` asserts that tail
is exactly the target day rather than trusting the arithmetic. When observations
do reach the nominal cutoff there is no gap and the horizon collapses to a
plain 24.

**How long the horizon actually comes back depends on how the target is
published, and `net_position` is the exception.** Where actuals stop near real
time (`load`, `price`), a 06:00Z run is ~42h short of the nominal cutoff and
`prediction_length` is ~66. But `net_position` is **day-ahead** published —
day D's values appear around 12:45 CET on D−1 — so a 06:00Z run on D
legitimately holds actuals through **D 21:00**, the gap is 26h, and
`prediction_length` comes back as **50**. Measured 2026-08-06 across all 16
stored vintages: 26h staleness and a 50h horizon for all 19 live countries,
without exception (ABL-28).

This is also a trap for `as_of`. `as_of` bounds on *target* timestamp, not on
ingest time, so setting it to the run instant (`RUN_HOUR` on D) cuts a
day-ahead target's context 16h shorter than the live run really had, and
understates the pipeline. For `net_position` the serve-faithful **observation**
bound is **D 22:00**, not D 06:00 — verified by reproducing the live 2026-08-06
vintage **bit-exactly** (max |diff| 0.0 MW over 480 points; `predict_quantiles`
is deterministic, so an exact match really does mean an identical input).

**A serve-faithful reconstruction needs two bounds, not one** (ABL-68). One
`as_of` was doing double duty: it bounds where observations stop *and*, via
`_load_weather_forecast_range`, which weather runs had been issued
(`forecast_run_time <= ?`). Those are the same instant only when the target is
published in real time. For `net_position` they are 16h apart, so neither value
is right on its own:

- `D 22:00` is the correct observation bound, but it also admits a weather run
  issued at 12:00Z on D — information the 06:00Z run never had. Measured against
  the as-served 2026-08-06 vintage, this put the worst country **1,881 MW** away
  from what production served.
- `D 06:00` is the correct publication bound, but it truncates the context 16h.
  This is what `scripts/compare_experiments.py:178` still does for *every*
  forecast type, so its net_position weeks understate the pipeline.

`build_for_country` therefore takes `publication_as_of` alongside `as_of`
(`src/chronos2/input_builder.py:541`), defaulting to `as_of` so live and
existing callers are unchanged. With the bounds split, 16 of 19 countries
reproduce the as-served vintage to under 0.3% of mean |forecast|; LT (38.8%),
RO (5.9%) and BG (1.4%) do not, because **suffix-1 covariates cannot be bounded
at all**. TSO load forecasts, DA prices and cross-border flows are bounded by
timestamp only — `publication_timestamp_utc` records when we fetched, not when
the value was published, and is NULL on these rows — so a vintage reconstructed
days later legitimately sees revisions the live run did not. Any model fitted on
a reconstruction should treat those three countries as unverified rather than
assume the fit transfers.

Before this, the context was built out to the nominal cutoff regardless, where
`_align_to_index` forward-filled 6h and wrote **0.0** into the remaining ~36.
The model's most recent context was a block of zeros. Net position is signed and
centred near zero, so nothing downstream looked wrong — but measured forecasts
came out at 6% of actual for FR and sign-flipped for DE, and the dashboard
showed an 8 GW discontinuity at each day boundary where one run's recovered tail
met the next run's near-zero start.

**This is why offline experiment scores did not catch it.** `compare_experiments.py`
read the database as it stands *today*, so its context ran right up to D+1 23:00
with real data — the harness was scoring a model that never existed in
production. Both it and any new backtest must pass `as_of` (the moment the run
would have fired: D+2 minus two days at `RUN_HOUR`), which bounds every query
including the weather-forecast run time. Without it, offline numbers are
measuring leaked information.

Interior gaps shorter than the ffill limit are still filled, and anything longer
still becomes `0.0` via `_align_to_index`. It is the same failure mode, so
prefer leaving a genuine hole NaN over inventing a zero. Two things this still
bites, both measured 2026-08-06 (ABL-28):

- **The target, for a country that has stopped publishing.** For the 19 live
  net-position countries the 672h context is 672/672 real observations with
  zero fill, so "coverage is near-complete in practice" holds — but GR is
  **24 real hours and 648 zero-filled**, and its 24 real hours are themselves
  exactly `0.0` upstream. A constant-zero context yields a constant-zero
  forecast (1e-10..4.6e-7 MW), which the pipeline still publishes and pushes;
  the dashboard withholds it at render time (ABL-25). GR's horizon reaches
  **362h**. Filed as its own issue: refuse rather than forecast when the
  context is degenerate or stale.
- **Covariates, which the context-cutoff fix does not cover.** Each is aligned
  to the *target's* index, so a covariate whose source stops earlier is ffilled
  6h and then zeroed. `weather_data` `data_quality='actual'` is retained on a
  rolling 672h window, so on a current vintage `weather__temperature_2m_k`
  reads **297 K for 656 hours and then 0 K for the last 16**. Measured cost:
  under 1% of MAE, so it is filed rather than fixed in flight.

**Key dependencies:** `torch>=2.1`, `transformers>=4.40`, `chronos-forecasting>=2.0` (separate venv)

### Net-position evaluation and the promotion gate

`src/evaluation/net_position.py` (ABL-30) scores the as-served vintages against
`net_position` actuals; `scripts/evaluate_net_position.py` is the entry point and
writes `reports/net_position_eval/`. Both databases are opened **readonly**.

```bash
# single model (default: chronos-2-V010)
python scripts/evaluate_net_position.py --replica-db ...\energy_dashboard.db \
    --sidecar-db ...\forecasts_local.db --stdout
# several models over one identical vintage window — the C2c deliverable
python scripts/evaluate_net_position.py --model chronos-2-V010 chronos-2-V012 ...
```

Four things about the gate are load-bearing (ABL-72):

- **The gate scores a vintage window; the report does not.** The tables cover
  every stored vintage, but `promotion_gate` reads `results["gate_scope"]`,
  which defaults to vintages at or after `cohort_split` (`FIX_DEPLOYED_UTC`).
  Without that restriction the champion is measured on the zero-padded-context
  era: measured on the replica 2026-08-07, all 18 vintages give MAE 1,439 MW /
  slope 0.26 against the serving model's 553 MW / 0.90, so a challenger faced a
  bar **2.60x easier** than the real one. The difference is not cosmetic —
  `slope_in_range_per_country` reads 0/19 contaminated and **11/19** windowed.
  Override with `--gate-vintage-start` / `--gate-vintage-end`; the window and its
  vintage count are printed in the report header.
- **All eight pre-registered criteria are emitted, and PASS requires all eight.**
  `PRE_REGISTERED_CHECKS` is the list; the gate checks itself against it and the
  report iterates it, so an absent criterion prints as `NOT IMPLEMENTED` instead
  of being silently skipped. The verdict is `PASS` / `FAIL` / `INCOMPLETE` — a
  criterion that cannot be evaluated (no `--candidate-backtest`, no
  `--serve-faithful-verified` attestation) yields `INCOMPLETE`, never `PASS`.
  Two of the eight had never been implemented, and because the old verdict
  spanned "only evaluable checks", their absence could not fail.
- **LU and GR are excluded by name**, not by symptom — `GATE_EXCLUDED_COUNTRIES`
  carries a reason for each (LU duplicates DE in A25; GR's actuals are
  fabricated zeros, ABL-35/ABL-67). GR was previously excluded only as a
  side-effect of having no paired actuals, so a partial upstream resume would
  have silently re-entered it and failed the gate on thin data.
- **A comparison shares one window across every column.** It is the intersection
  of the models' stored vintage spans, floored at `cohort_split`, and
  `compare_models` raises if the columns end up scored over different windows.
  Per-model vintage counts are printed rather than smoothed, and a model with no
  stored vintages reads as "Not scored", never as an empty column.

**The script does not discover model versions.** It scores exactly the
`--model` names given. Anything claiming it picks up new versions automatically
is wrong — that was ABL-68 scope item 1 and plan Rev 3:29.

#### Per-country re-read: zero baseline, level vs shape (ABL-280)

`src/evaluation/country_reread.py` + `scripts/reread_net_position_country.py`
answer "is this one zone's forecast actually worse than a free baseline, and
*how*". It reuses the eval's loaders, serve-faithful baselines and
`point_metrics`, so it cannot disagree with the gate about a country's MAE.

```bash
.venv\Scripts\python.exe scripts/reread_net_position_country.py --country RO --fleet \
    --replica-db ...\energy_dashboard.db --sidecar-db ...\forecasts_local.db --stdout
```

Three things it adds, each because a real reading went wrong without them:

- **The zero forecast is a named baseline, and `skill_vs_zero < 0` is
  identically `WAPE > 100%`** — the same fact twice, pinned as an equality in
  the tests. Naming it is what stops "WAPE 102.6%" reading as an emergency on
  its own: zero is not a baseline anyone would serve for net position, and RO
  loses to it while beating persistence by 20.6%. The decision-relevant row is
  climatology, not zero.
- **Level vs shape.** Demeaning both series *within each vintage day* separates
  a wrong profile from a right profile at the wrong level. Measured 2026-08-12
  on the 7-scored-vintage cohort, RO reads pooled corr 0.50 / within-day 0.83
  and a per-vintage-day bias sd of 721.5 MW against mean |actual| 709.0 MW.
  That is what refutes a *static* per-country offset for RO — a constant cannot
  track a bias that swings +259 to −1095 MW across six days. NL (+0.37 gap) and
  LV (+0.28) carry the same signature; it is a cluster, not one zone.
- **Vintages that carry evidence, counted separately from vintages that
  exist.** `build_gate_scope` counts off the left-merged frame, so a vintage
  whose D+2 targets have no published actuals still counts toward
  `min_live_shadow_vintages`. That gap is permanent, not incidental — the rail
  generates at D for D+2, so the two newest vintages are always unscorable.
  Measured 2026-08-12: **9 counted, 7 scored**. So `min_live_shadow_vintages`
  reaches 14 on 2026-08-17 with ~12 vintages of evidence behind it; 14 *scored*
  vintages land 2026-08-19. This module counts scored vintages and labels its
  own output `INTERIM` / `CONFIRMATORY`; it deliberately does **not** change the
  gate, which is pre-registered.

`--fleet` sweeps every `GATE_COUNTRIES` zone beside the named one. That is not
decoration: on the interim cohort 4 of 19 lose to climatology (RO −23.3%, NL
−18.3%, then BE −2.4% and HR −0.2% inside noise), so a fallback proposed for RO
alone would leave NL served by a model it also loses with. Evidence pack:
`reports/abl_280_ro_climatology_reread.md`. Dated outputs land in
`reports/net_position_eval/country_reread/`, which is gitignored like the rest
of that directory.

### All-type forecast scorecard (ABL-129)

`scripts/evaluate_scorecard.py` is the recurring answer to "is the served
forecast better than a free baseline?" It scores the production registry
snapshot for all nine served types over one target window and writes a dated
Markdown/JSON pair plus `latest.*` under `reports/forecast_scorecard/`
(`scripts/evaluate_scorecard.py:17`, `scripts/evaluate_scorecard.py:58`). It
opens the replica and optional sidecar read-only; its only writes are reports.

The selection rule is **latest vintage per country + target + model + horizon
band**, not one latest row per target. The latter erases the stored 24-64h
evidence because the newest daily run is always the shortest lead. Timestamps
are parsed before the join so both the ML `T` separator and Chronos space
separator pair (`src/evaluation/scorecard.py:95`). The ABL-35 `load_mw > 0`
guard applies to load only; measured zero is retained for solar, wind, price,
and every other type. GR net position is excluded by name using the reason from
`GATE_EXCLUDED_COUNTRIES`, not by detecting zero-shaped data
(`src/evaluation/scorecard.py:178`).

**Scoring truth lives in one dict, `scorecard.ACTUAL_SPECS`, and since ABL-410
the renewable family reads `energy_generation` — the same table the dashboard
publishes against.** Before that it read the frozen `energy_renewable` while the
dashboard had moved (ABL-399), so one model, country and window had two
published WAPEs and neither was wrong. Three things to hold onto:

- This is **not** ABL-321's rejected switch. That is the *training* source,
  `db.RENEWABLE_TYPE_SOURCE_TABLE`, still `energy_renewable` and untouched.
  Scoring truth and training source are independent post-ABL-331.
- It touches **no promotion gate**. `ACTUAL_SPECS` is read only by
  `scorecard._load_actuals`; both gate harnesses take actuals from
  `RenewableFeatureBuilder` → `db.load_renewable_type_data`.
- `hydro_total` is `db.RENEWABLE_TYPE_COLUMNS['hydro_total']` **imported, not
  restated**. A strict `hydro_run_mw + hydro_reservoir_mw` is survivable on the
  frozen table only because `REAL DEFAULT 0` means nothing there is NULL; on
  `energy_generation` it erases the 9 countries that report one component.

Two caveats travel with every renewable-family figure: `energy_generation` has
an open FR ingest gap (2026-06-30 → 2026-07-22, ABL-318 §3) that shrinks FR
samples and therefore moves **pooled** rows on composition alone; and the models
are still fitted on `energy_renewable`, so where the tables disagree about the
target, part of the WAPE is target mismatch. `reports/abl_410_scoring_truth.md`
decomposes both, and records the finding that BE `hydro_total` is a
pumped-storage forecast under a hydro label.

D-7 and persistence predictions go through `src/baselines.py`, via the pure
issued-row adapter at `src/baselines.py:297`. Persistence derives its lookback
from target minus `generated_at` and rounds the lead **up**: stored
`horizon_hours` floors partial hours, so using it directly can select an actual
from after generation. Net position instead reuses its evaluator's day-ahead
publication cutoff and persistence implementation. Missing actual/baseline pairs remain unmeasured, and
skill is computed only on the exact intersection available to both model and
baseline. This scorecard references the separate net-position promotion gate;
it does not copy or weaken it (`src/evaluation/scorecard.py:326`).

### Experiment System

Experiments are versioned V001-Vnnn with configs in `experiments/`. Both XGBoost and Chronos-2 run in parallel — forecasts stored with distinct `model_name` values in the `forecasts` table.

```bash
experiments/
├── registry.json           # Master index of all experiments
├── V001/config.json        # XGBoost baseline
├── V002/config.json        # Chronos-2 zero-shot
├── V003/config.json        # Chronos-2 fine-tuned (5000 steps)
├── V012/config.json        # Baseline ensemble (shadow challenger)
└── V016/config.json        # V010 + affine + AR(1) (shadow challenger)
```

### Champion / challenger shadow serving (ABL-68)

The daily 08:00 net-position job runs the champion, then
`scripts/forecast_challengers.py` runs every registered challenger on the same
serve-time inputs. Challengers write their own `model_name` rows to the sidecar
and **are never pushed to production**.

`model_name` is the identity that matters. `model_version` is the vintage
timestamp, not a model identity — two models sharing a `generated_at` are told
apart by `model_name` alone. Challengers are listed in
`src/challengers/registry.py`, not discovered, so what runs tomorrow is a
reviewable list.

**Two things enforce the "never pushed" invariant, and both are load-bearing.**
`push_net_position_forecast.py` names the champion (`CHAMPION_MODEL_NAME`,
default `chronos-2-V010`) and filters every query on it. Before ABL-68 it took
the newest `generated_at` for `forecast_type='net_position'` with no model
filter — correct only while the sidecar held one model. Challengers run *after*
the champion in the same job, so the newest vintage in the sidecar is now a
challenger's: verified 2026-08-07, the newest row was `chronos-2-V016` and the
unfixed script would have shipped it to the dashboard as the production
forecast.

**The eval scores every stored vintage, but one `model_name` per invocation.**
`evaluate_net_position.py --model` defaults to the champion, so a challenger is
scored only if the runner names it. `run-net-position.ps1` calls it once per
model, each with its own `--out-dir`, because the script always writes
`latest.md` beside the week-tagged report and a shared directory would leave
`latest.md` holding whichever model ran last — which ABL-30 and ABL-34 both read
expecting the champion.

**V012 does not reimplement its own baseline.** It calls
`src/evaluation/net_position.py::baseline_predictions`, the same function the
gate scores against. Two implementations of one baseline is the shape of the
renewable-share defect.

**Never compare two per-model eval reports to each other.** Each report is
scored on whatever rows its own model covers, and the champion's set also picks
up prod-pushed vintages that live in the replica and were never in the
reconstruction a challenger is rebuilt from. On V016's held-out window the
champion's report covered 57 vintages to the challenger's 49. Read report against
report, V016 looked *better* almost everywhere (FR 2,464 → 1,916 MW, DE 3,344 →
3,014 MW); scored on the rows both models actually cover, it is **worse**. Use
`scripts/compare_challenger.py`, which inner-joins on
`(country, target hour, run)` and reports the one-sided remainders
(`src/evaluation/head_to_head.py`).

**A run is not a `generated_at`, and on the live rail the two never match**
(ABL-82). The head-to-head's first cut joined on exact `generated_at` equality.
That is right for a reconstruction — one process replays every vintage and
stamps them all — and wrong for the daily shadow rail, where
`forecast_chronos2.py` and `forecast_challengers.py` are separate processes in
`run-net-position.ps1` and each calls its own `datetime.now()`. Measured on the
live sidecar 2026-08-09: champion `2026-08-09 06:00:55.715745`, all three
challengers `2026-08-09 06:01:08` — 12.3 s apart, and only the champion carries
microseconds. The exact join paired **0** rows for V012, V014 *and* V016 while
912 co-run pairs sat there, and it did so **while printing a full report**:
`0.0 MW` MAE for both models and "challenger is 0.0% worse". An empty
head-to-head that renders as a tie is this repo's usual defect in a new place.

Two vintages are now the same run when they agree on the **actuals they could
see** (`net_position.as_of_for_vintage`, the same serve-faithful cutoff the
eval's baselines use) *and* their `generated_at` are within `MAX_RUN_SKEW` (4 h)
of each other. The cutoff carries the meaning; the skew bound is a guard, since
one cutoff bucket is 24 h wide. `--max-run-skew-hours` tunes the bound only —
**it cannot pair two vintages that saw different actuals, at any value**, and
that is deliberate: an information mismatch is not a tolerance problem.

Three properties are load-bearing:

- **Backfills are refused, not paired.** The 2026-08-07 V012/V016 backfill ran
  15 h 25 m after that day's champion and V014's first vintage 5 h 36 m after
  (2026-08-08 11:36). Both saw a further day of actuals, so scoring them
  against a 06:00 champion would credit a challenger for information the
  champion never had. They land in `n_only_a`/`n_only_b`, where a reader sees
  them.
- **A champion re-run duplicates nothing.** 2026-08-06 holds two champion
  vintages (06:00:44 and 10:52:22) under one cutoff. The pair closest in time
  wins and the other falls to `n_only_a`; a naive day-level join would have
  matched both to the single challenger vintage and counted the challenger's
  hours twice.
- **Nothing paired reports no number.** `pooled_mae_*` is `None`, the report
  renders a "Not measured" block instead of a table, and
  `compare_challenger.py` **exits 1**. A promotion gate must not be able to
  read an empty comparison as "no difference".

The reconstruction path is unchanged: re-run 2026-08-09 with the new rule, the
V016 held-out comparison still returns exactly **22,344 paired rows over 49
runs**, V010 **775.2 MW** vs V016 **786.1 MW**, 1/19 materially better, 3
identical — the numbers below, to the decimal.

One timing consequence worth knowing before reading a fresh shadow report: the
rail forecasts **D+2**, so a co-run pair is not scoreable until its target day
lands. On 2026-08-09 the live head-to-head correctly reports *not measured* for
all three challengers — the 912 pairs it now forms target 2026-08-10 and 08-11.
The first live-rail pairs score on 2026-08-10, and by the C2c gate read
(~2026-08-26) roughly 17 daily runs are available.

**V016 refuses more than it corrects, and does not beat the champion.** Measured
on a held-out window (fit 2026-01-19..06-15, tested 06-17..08-04, 22,344
exactly-paired rows over 49 vintages): V010 **775.2 MW** MAE, V016 **786.1 MW** —
1.4% worse. It is materially better (≥0.5%) in **1 of 19 countries** (FR, −2.1%),
identical in 3 (BG/LT/RO pass through uncorrected), and within noise or worse in
the remaining 15. Forcing unit slope instead (`--method variance`) costs 11.4%:
863.8 MW, better in 0 of 19. Archived in `reports/head_to_head/V016/`
(deliberately *not* under `reports/net_position_eval/`, which is gitignored
because the scheduled eval rewrites it every run), reproduced 2026-08-08 with
`experiments/V016/correction_holdout.json`. AR(1)-only and a rolling 60-day
refit were also tried and also lost, so drift is not the explanation.

Two reasons, both worth knowing before proposing another correction layer:

- **Affine recalibration cannot fix the residual shrinkage.** For any affine
  map, `slope(corrected on actual) = b * slope(f on a)`, which under OLS is
  exactly `rho**2`. It *lowers* the slope in 15 of 16 countries (FR 0.480 →
  0.398, BE 0.298 → 0.201). And `b < 1` for 15 of 16, so the error-minimising
  move is to shrink the champion *further*: V010 is already close to affinely
  optimal per country. Unit slope requires inflating variance, measured at an
  11% MAE cost. **The gate's `slope ∈ [0.8, 1.2]` is therefore unreachable by
  any affine layer on V010** — it needs `rho ≥ 0.894`, and measured per-country
  `rho` is 0.41-0.88. That is a better-model problem (V014/V015).
- **AR(1) is bounded by the horizon, not the coefficient.** Residual lag-1
  autocorrelation is genuinely 0.85-0.96, but a 06:00Z run observes actuals only
  to D 21:00 while correcting D+2 00:00-23:00. The carry is `phi**27..phi**51` —
  0.04 to 0.32 at the nearest corrected hour. Small by construction.

**Two fit files, and they are not interchangeable.**
`experiments/V016/correction.json` is fitted on everything (`train_end: null`)
and is what the daily shadow run serves — correct for serving forward, and
*in-sample* for any window before the fit date. `correction_holdout.json` is
fitted to 2026-06-15 only and is the one every quoted V016 number above comes
from. Evaluating V016 with the full-sample fit would score it on data it was
fitted on and flatter it. Both drop the W11/W12 backtest target days.

Note the pooled-vs-per-country trap here: the plan's 0.894 correlation is
pooled, which mixes country means and is inflated by between-country variance
(the eval's own docstring says so). Per-country `rho` is much lower, and the
per-country numbers are what a per-country correction can use.

### V014 — the trained per-country XGBoost challenger (ABL-69)

The challenger the Board asked for, and the answer to the paragraph above: V010
is close to affinely optimal per country, so no correction layer on it can reach
the gate's `slope ∈ [0.8, 1.2]`. That needs a different model.

- `src/challengers/v014_features.py` — the feature builder (89 features)
- `src/challengers/v014.py` — model, refusals, artifact integrity
- `scripts/train_v014.py`, `scripts/backtest_v014.py`
- `experiments/V014/config.json`, `training_report.json`, `backtest_W01_W12.json`
- Artifacts: `models/net_position/V014/{CC}.joblib`, 19 countries (all supported
  net-position countries except LU, which duplicates DE in the A25 document, and
  GR, whose actuals are fabricated zeros — ABL-35/ABL-67)

**`models/` is gitignored, so merging the branch does not ship the model.**
The scheduled job runs from `C:\Code\able\energy-forecast` (`$Repo` in
`run-net-position.ps1`), and `config.MODELS_DIR` resolves against *that*
checkout — not the worktree the training ran in. Artifacts have to be copied
across by hand, and if they are not, the rail logs "no trained model for
AT,BE,…" and writes nothing for V014 while every other model succeeds and the
job still exits 0. Retraining in a worktree therefore has two steps:

```bash
python scripts/train_v014.py --countries all          # under .venv
cp -r models/net_position/V014 C:/Code/able/energy-forecast/models/net_position/
```

**Serve-faithfulness holds by construction, and it has to.** A tabular model
evaluates every feature *at the target timestamp*, so unlike Chronos-2 — whose
context simply ends where the data ends — each column must justify its own
availability. It cannot be verified after the fact from ingest metadata:
`fetched_at` and `publication_timestamp_utc` are last-write over a rolling
re-fetch window, so every FR `net_position` row for targets 2026-08-01..07
carries the identical `fetched_at`. An as-of query over them passes anything you
hand it. The construction is one documented per-source cutoff derived from the
run instant, applied identically in training, backtest and serving:

| source | cutoff at a 06:00Z run on D |
|---|---|
| `net_position`, `energy_price`, `energy_load_forecast`, `energy_generation_forecast` | D 21:00 (day-ahead publication) |
| `crossborder_flows` | target − 72h (ABL-74), with an `xb_missing` indicator |
| `weather_data` | at the target hour, `data_quality='forecast'` and `forecast_run_time <= run_ts` |

**Same-hour lags start at 72h, not 48h.** The binding target hour is D+2 23:00,
exactly 50h past the D 21:00 cutoff, so a 48h lag reaches D 22:00 and D 23:00 —
two hours that do not exist at run time. `assert_lag_is_serve_safe` checks every
lag on every build; the tautological check (filter to `<= cutoff`, then assert
the max is `<= cutoff`) cannot fail and was deliberately not written.

**W01-W10 are weather-blind for every model.** The issued-forecast archive
begins 2026-01-11 (FR's earliest `forecast_run_time`), so those ten weeks carry
NaN weather. The builder does **not** fall back to the `data_quality='actual'`
reanalysis, which is a nowcast (lead 0.0h) and would be observed weather handed
to the model as a forecast. `weather_available` records the regime per row.
The champion's loader filters the same way, so the comparison is fair — but
neither model is in its serving configuration there.

**Three refusals**, all because a tree returns a number for any row you give it:
no model file for a country raises rather than substituting another country's
model; fewer than 2 of 3 anchor features (`np_at_cutoff`, `np_lag72h`,
`np_last7d_mean`) yields NaN for that hour; and a refused hour is **dropped, never
written as 0.0** — a 0 MW net position is a real balanced-border reading. Nothing
is imputed: a mean would turn "we do not know this border's flow" into "the flow
was average".

**A late run is not a better-informed run.** `run_v014` derives its serve window
from the *target date*, never from `generated_at`, so a job that fires late gets
the cutoffs the schedule promises rather than the extra hours the clock handed
it. Otherwise a delayed vintage would be scored against models built on less.

Promotion is not decided here — only by the pre-registered gate read in C2c
(ABL-72). V014 supplies two of its eight criteria: G5 wants
`experiments/V014/backtest_W01_W12.json` via `--candidate-backtest`, and G6 wants
a bit-reproduced live vintage via `--serve-faithful-verified`.

**Tuning to MAE moves BE out of the gate's slope band, so nothing was adopted.**
`scripts/tune_v014.py` runs a small readable grid per country, selecting on
validation MAE over the same chronological split the fit uses — never on the
backtest weeks, which would make the backtest a training set. Measured
2026-08-08 on the four countries where V014 trails V010, only **BE** cleared the
2% bar, and it did so by breaking the criterion the model exists to satisfy:

| country | default MAE / slope | best candidate | its MAE / slope |
|---|---|---|---|
| BE | 1,658.9 / 1.047 | `shallow_slow` **−10.2%** | 1,489.0 / **1.395** |
| NL | 1,826.4 / 0.996 | `deeper_slow` +0.9% | 1,810.1 / 0.976 |
| AT | 1,000.2 / 1.068 | `shallow_slow` +1.6% | 983.9 / 0.980 |
| FR | 2,120.0 / 0.993 | `shallow_slow` +1.8% | 2,081.6 / 1.132 |

The gate wants `slope ∈ [0.8, 1.2]`; BE's MAE-optimal fit sits at 1.395. So the
one adoption the search offers trades the criterion V014 was built for against
the one it is currently losing. The script therefore **writes no model without
`--adopt`**, and `--adopt` was not used. Deciding the tuning objective against
the gate rather than against MAE is a program-level call (ABL-24), not a
parameter choice. Note `shallow_slow` is the best candidate in three of four
countries and stays inside the band in NL/AT/FR — it is BE specifically where
MAE and slope pull apart.

**Backtest evaluation:** 12 held-out weeks (W01-W12) spanning 2024-2026, NaN-masked during training of ALL models. Use `--exclude-backtest` flag for XGBoost, automatic for Chronos-2.

### Expected Performance

| Type | Typical MAPE |
|------|-------------|
| Load | 2-5% |
| Price | 10-20% |
| Renewable | 15-30% |
