# Serve-faithful wind retrain gate — registered scope `abl406-tranche2b`

**Disposition: FAIL**

Generated: 2026-08-13 20:18 UTC
Fit targets: 2026-01-14 00:00:00 → 2026-07-11 00:00:00 (exclusive).
Out-of-sample gate targets: 2026-07-11 00:00:00 → 2026-08-10 00:00:00 (exclusive).
Baseline: literal seasonal-naive D-7. TSO is revision-contaminated context only and is not a gate criterion.
Replica: `C:\Code\able\data\energy_dashboard.db` (9,432,453,120 bytes), opened with SQLite `mode=ro`, `uri=True`.
That one file is the source of the TSO series, the contamination screen, and — since ABL-355 — the fitted target series, its lag/rolling features, the D-7 and persistence baselines, the gate actuals and the weather archive. The incumbent forecasts are the only read it does not hold alone; see the sidecar below.
Sidecar: `C:\Code\able\data\forecasts_local.db`, also opened `mode=ro`, and read for locally generated incumbent forecasts only. Where a sidecar row and a replica row carry the same vintage, the sidecar's is the one scored.

## Gate read

Registered scope `abl406-tranche2b`: ES wind_onshore, FI wind_onshore, GR wind_onshore, IT wind_onshore, NO wind_onshore, PL wind_onshore, PT wind_onshore, SE wind_onshore.
Target series, features, baselines and contamination screen: `energy_generation`.
Gate basis — the columns that must be simultaneously finite for a row to be scored: `challenger`, `seasonal_naive`. Comparators outside the basis are scored on their own intersection with it and carry their own n, so a comparator that does not exist for a pair reads Not measured instead of emptying the cell.
Strict full PASS requires challenger WAPE < D-7 in all 24 country × primary D+2-band cells and ≥95% of intended pairs. Result: **16/24 cells pass**.

Model-free reference (ABL-389) — four predictors with no model in them, reported beside every cell. `constant_causal` is a flat line at the **fit-window mean**, the honest "no model" floor, using only what was knowable before the gate window opened; `constant_oracle` is a flat line at the **gate-window median**, a hindsight upper bound on what any constant could have achieved. `climatology_causal` and `climatology_oracle` are the same two forms taken **per hour of day** — the fit-window hourly mean and the gate-window hourly median — which is the tighter reference on every pair measured so far, because a constant is a climatology with one bucket. Read the pair together: the constant says whether the model predicts the *level*, the climatology says whether it predicts the level *and the daily shape*, and the gap between them is how much of this series is forced diurnal structure.

All four are **reported references and not gate criteria**: none is in the gate basis, none can move a cell's verdict, and a pair that clears its D-7 bar while losing to one still reads PASS. They are the number that qualifies the PASS — a challenger that does not beat `climatology_oracle` has not demonstrated skill beyond the average day, and a D-7 bar that `constant_causal` clears on its own was not a demanding bar. **Check each reference's own n before comparing it to the challenger.** A climatology is 24 levels, so an hour of day absent from its source window leaves those rows unscored for that column alone; scored on different rows, two WAPEs are not the same measurement. Nothing is interpolated to close that gap.

| type | country | horizon | n | challenger WAPE | D-7 WAPE | skill vs D-7 | constant causal WAPE | constant oracle WAPE | climatology causal WAPE | climatology oracle WAPE | incumbent WAPE | MAE | bias | slope | corr | gate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| wind_onshore | ES | 24-36h | 720 | 54.3% | 41.0% | -32.2% | 62.1% | 41.5% | 55.1% | 27.5% | Not measured | 2275.6 MW | 50.0% | 0.6 | 0.6 | FAIL |
| wind_onshore | ES | 36-48h | 720 | 54.2% | 41.0% | -32.0% | 62.1% | 41.5% | 55.1% | 27.5% | Not measured | 2272.0 MW | 49.8% | 0.6 | 0.6 | FAIL |
| wind_onshore | ES | 48-64h | 510 | 52.4% | 38.5% | -36.0% | 63.9% | 44.2% | 53.1% | 26.1% | Not measured | 2197.3 MW | 47.3% | 0.6 | 0.7 | FAIL |
| wind_onshore | FI | 24-36h | 711 | 41.1% | 59.6% | +31.0% | 54.6% | 53.5% | 53.4% | 51.4% | Not measured | 824.8 MW | -27.0% | 0.4 | 0.6 | PASS |
| wind_onshore | FI | 36-48h | 711 | 43.3% | 59.6% | +27.3% | 54.6% | 53.5% | 53.4% | 51.4% | Not measured | 868.6 MW | -29.2% | 0.4 | 0.6 | PASS |
| wind_onshore | FI | 48-64h | 504 | 46.2% | 54.9% | +15.8% | 52.0% | 51.5% | 50.7% | 49.9% | Not measured | 967.8 MW | -33.1% | 0.3 | 0.6 | PASS |
| wind_onshore | GR | 24-36h | 720 | 29.6% | 63.8% | +53.6% | 53.2% | 51.7% | 53.4% | 51.4% | Not measured | 399.4 MW | -5.1% | 0.7 | 0.8 | PASS |
| wind_onshore | GR | 36-48h | 720 | 29.6% | 63.8% | +53.7% | 53.2% | 51.7% | 53.4% | 51.4% | Not measured | 399.0 MW | -6.4% | 0.7 | 0.8 | PASS |
| wind_onshore | GR | 48-64h | 510 | 30.2% | 58.9% | +48.7% | 50.9% | 49.0% | 50.8% | 48.7% | Not measured | 401.6 MW | -9.9% | 0.7 | 0.8 | PASS |
| wind_onshore | IT | 24-36h | 716 | 71.4% | 70.6% | -1.1% | 92.0% | 52.3% | 90.7% | 45.1% | Not measured | 1132.8 MW | 43.3% | 0.3 | 0.3 | FAIL |
| wind_onshore | IT | 36-48h | 715 | 71.2% | 70.6% | -0.8% | 91.9% | 52.3% | 90.6% | 45.1% | Not measured | 1130.6 MW | 42.1% | 0.2 | 0.3 | FAIL |
| wind_onshore | IT | 48-64h | 505 | 66.8% | 67.2% | +0.6% | 82.8% | 50.8% | 83.3% | 42.4% | Not measured | 1134.8 MW | 40.7% | 0.3 | 0.4 | PASS |
| wind_onshore | NO | 24-36h | 720 | 51.4% | 61.0% | +15.8% | 59.7% | 42.4% | 59.4% | 41.9% | Not measured | 588.0 MW | 15.9% | -0.1 | -0.1 | PASS |
| wind_onshore | NO | 36-48h | 720 | 51.6% | 61.0% | +15.5% | 59.7% | 42.4% | 59.4% | 41.9% | Not measured | 590.3 MW | 17.3% | -0.1 | -0.1 | PASS |
| wind_onshore | NO | 48-64h | 510 | 51.8% | 61.6% | +15.9% | 57.7% | 43.1% | 58.0% | 42.4% | Not measured | 603.2 MW | 16.4% | -0.1 | -0.2 | PASS |
| wind_onshore | PL | 24-36h | 720 | 54.1% | 92.8% | +41.7% | 61.1% | 51.2% | 59.7% | 47.4% | Not measured | 968.0 MW | 37.0% | 0.4 | 0.6 | PASS |
| wind_onshore | PL | 36-48h | 720 | 52.5% | 92.8% | +43.4% | 61.1% | 51.2% | 59.7% | 47.4% | Not measured | 939.3 MW | 33.7% | 0.4 | 0.7 | PASS |
| wind_onshore | PL | 48-64h | 510 | 51.4% | 94.4% | +45.6% | 63.9% | 52.3% | 61.5% | 48.2% | Not measured | 897.9 MW | 31.1% | 0.4 | 0.7 | PASS |
| wind_onshore | PT | 24-36h | 720 | 68.2% | 49.6% | -37.5% | 101.4% | 50.2% | 101.1% | 39.4% | Not measured | 543.7 MW | 46.2% | 0.2 | 0.3 | FAIL |
| wind_onshore | PT | 36-48h | 720 | 68.7% | 49.6% | -38.5% | 101.4% | 50.2% | 101.1% | 39.4% | Not measured | 547.7 MW | 45.3% | 0.2 | 0.2 | FAIL |
| wind_onshore | PT | 48-64h | 510 | 61.0% | 46.6% | -30.7% | 93.1% | 49.3% | 87.3% | 35.9% | Not measured | 511.2 MW | 36.6% | 0.2 | 0.3 | FAIL |
| wind_onshore | SE | 24-36h | 720 | 30.2% | 53.5% | +43.6% | 43.7% | 36.5% | 42.7% | 35.5% | Not measured | 1119.1 MW | -8.0% | 0.4 | 0.6 | PASS |
| wind_onshore | SE | 36-48h | 720 | 30.2% | 53.5% | +43.4% | 43.7% | 36.5% | 42.7% | 35.5% | Not measured | 1121.9 MW | -9.1% | 0.4 | 0.6 | PASS |
| wind_onshore | SE | 48-64h | 510 | 30.3% | 52.8% | +42.6% | 44.4% | 36.2% | 42.9% | 35.3% | Not measured | 1108.5 MW | -7.7% | 0.4 | 0.6 | PASS |

Reference levels used, from the same ABL-188-filtered target series the gate actuals and the D-7/persistence baselines come from — no refit, no second read, no additional upstream fetch. The hourly levels behind the climatology columns are in `results.json` in full; `h` is how many of the 24 hours of the day that level set covers, and anything below 24 means those rows were dropped from that column's n:

| type | country | constant causal | constant oracle | climatology causal | climatology oracle |
|---|---|---:|---:|---:|---:|
| wind_onshore | ES | 6318.95 MW | 3861.50 MW | 4359.65–8382.94 MW (24h) | 1512.00–6651.50 MW (24h) |
| wind_onshore | FI | 2195.70 MW | 1856.97 MW | 1855.54–2602.94 MW (24h) | 1057.64–2590.57 MW (24h) |
| wind_onshore | GR | 1367.77 MW | 1151.62 MW | 1102.78–1535.96 MW (24h) | 1000.38–1318.25 MW (24h) |
| wind_onshore | IT | 2790.04 MW | 1300.75 MW | 2487.76–3243.05 MW (24h) | 859.50–2469.25 MW (24h) |
| wind_onshore | NO | 1684.36 MW | 1049.35 MW | 1631.95–1722.85 MW (24h) | 891.27–1268.39 MW (24h) |
| wind_onshore | PL | 2180.49 MW | 1458.08 MW | 1687.39–2640.73 MW (24h) | 811.44–1938.13 MW (24h) |
| wind_onshore | PT | 1558.67 MW | 692.60 MW | 1333.66–1691.49 MW (24h) | 272.40–1149.70 MW (24h) |
| wind_onshore | SE | 4539.57 MW | 3449.10 MW | 3842.49–5157.61 MW (24h) | 2636.89–4022.98 MW (24h) |

## Per-country all-D+2 summary

Gate-basis values (actual, challenger, seasonal_naive) share one finite intersection; each comparator outside the basis is scored on its own intersection with it, and its n is given in `comparator_n` in the JSON. A comparator showing `Not measured` had no finite rows at all.

| type | country | n | challenger WAPE | D-7 WAPE | persistence WAPE | constant causal WAPE | constant oracle WAPE | climatology causal WAPE | climatology oracle WAPE | incumbent WAPE | TSO WAPE (revision-contaminated; n) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wind_onshore | ES | 1,950 | 53.8% | 40.4% | 53.4% | 62.6% | 42.2% | 54.6% | 27.2% | Not measured | 17.8% (n=1,950) |
| wind_onshore | FI | 1,926 | 43.3% | 58.3% | 67.6% | 53.9% | 52.9% | 52.7% | 51.0% | Not measured | 14.3% (n=1,926) |
| wind_onshore | GR | 1,950 | 29.8% | 62.5% | 54.5% | 52.6% | 51.0% | 52.7% | 50.7% | Not measured | 17.7% (n=1,950) |
| wind_onshore | IT | 1,936 | 70.1% | 69.7% | 65.4% | 89.4% | 51.9% | 88.6% | 44.4% | Not measured | 26.2% (n=1,936) |
| wind_onshore | NO | 1,950 | 51.5% | 61.2% | 48.2% | 59.2% | 42.6% | 59.0% | 42.0% | Not measured | 9.6% (n=1,950) |
| wind_onshore | PL | 1,950 | 52.8% | 93.2% | 69.3% | 61.9% | 51.5% | 60.2% | 47.6% | Not measured | 17.1% (n=1,950) |
| wind_onshore | PT | 1,950 | 66.4% | 48.8% | 58.4% | 99.1% | 49.9% | 97.4% | 38.5% | Not measured | 15.6% (n=1,950) |
| wind_onshore | SE | 1,950 | 30.2% | 53.3% | 49.9% | 43.9% | 36.4% | 42.7% | 35.4% | Not measured | 4.7% (n=1,950) |
## Training cost

Wall-clock on the rail interpreter, one pair at a time in a single process. Feature build and fit are separated because they scale on different things. Measured under whatever else this workstation was running; treat as an upper bound for sizing, not a benchmark.

| type | country | fit rows | feature build | fit | gate build + predict | pair total |
|---|---|---:|---:|---:|---:|---:|
| wind_onshore | ES | 34,176 | 51.5 s | 3.7 s | 7.8 s | **63.0 s** |
| wind_onshore | FI | 34,176 | 48.3 s | 3.8 s | 8.4 s | **60.5 s** |
| wind_onshore | GR | 34,176 | 44.6 s | 3.6 s | 7.8 s | **55.9 s** |
| wind_onshore | IT | 34,176 | 47.8 s | 3.6 s | 7.4 s | **58.8 s** |
| wind_onshore | NO | 34,176 | 43.7 s | 3.6 s | 7.5 s | **54.8 s** |
| wind_onshore | PL | 34,176 | 43.4 s | 3.6 s | 7.1 s | **54.1 s** |
| wind_onshore | PT | 34,176 | 51.2 s | 4.4 s | 8.5 s | **64.2 s** |
| wind_onshore | SE | 34,176 | 43.5 s | 3.3 s | 6.8 s | **53.6 s** |

Scope total across 8 pair(s): **464.9 s** (58.1 s mean per pair).

## Fit and missingness audit

Each training row was constructed by `RenewableFeatureBuilder.row(target, generated_at, generated_at)` on the measured eight-vintage schedule. Gate targets were never fitted.

| type | country | algorithm | retained / intended fit rows | unique fit targets | excluded missing | degraded lag-1d rows | artifact SHA-256 |
|---|---|---|---:|---:|---:|---:|---|
| wind_onshore | ES | catboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `51d7fa69c68d83fc080469f30952f52c1718ed948220b840ab29db887497e73d` |
| wind_onshore | FI | catboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `12155fd16c9f14df9f2e7a88c63f402c5efdc86cec4dc22feb70e002179ddee4` |
| wind_onshore | GR | catboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `6c8eba4950b1dfd01ceec9a712568daa1fe89deed5330dbae4be070e381c985e` |
| wind_onshore | IT | catboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `970238448ff79cfcd67740cdb8bbdc12740a8943fb4264db4005d727c1f0579b` |
| wind_onshore | NO | catboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `9991abed6929bd6efb6793695329e9b40110da89bdebde3a31d01a424a651669` |
| wind_onshore | PL | catboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `e7e5fac532847841929419b159769e5990f55d8bdd8a96489ef1edc0f8bcbf15` |
| wind_onshore | PT | catboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `109232e33d08df72a1f40c51011bbad41120abb33c6584e2cede804ec2fb2401` |
| wind_onshore | SE | catboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `0a28072ff483eeeab0816161356378a9469a1209bb51dde06fe0955dd58f18e2` |

## Data quality and limits

- ABL-188 constant-run screening found no ≥24-hour bit-identical wind run in any fitted/scored pair; no wind row was excluded by that invariant.
- ABL-67 is net-position-only; ABL-109/111 are load-only. They do not intersect these wind targets. ABL-71's known wrong-write modes are load and net position, not wind; this is a provenance caveat, not proof that wind ingest is pristine.
- Weather rows were admitted only where `forecast_run_time <= generated_at`; missing vintages were excluded and counted, never backfilled from the future.
- TSO values come from a replacement table without first-seen vintages. They may include revisions and cannot support promotion.
- **ES wind_onshore: the TSO forecast is better than the challenger** (17.8% vs 53.8% WAPE over the same n=1,950). The gate is against D-7 and this pair clears it, but clearing D-7 is not evidence the model is good — it bounds how uninformative D-7 is. Treat the TSO series as a feature to ingest, not merely as context.
- **FI wind_onshore: the TSO forecast is better than the challenger** (14.3% vs 43.3% WAPE over the same n=1,926). The gate is against D-7 and this pair clears it, but clearing D-7 is not evidence the model is good — it bounds how uninformative D-7 is. Treat the TSO series as a feature to ingest, not merely as context.
- **GR wind_onshore: the TSO forecast is better than the challenger** (17.7% vs 29.8% WAPE over the same n=1,950). The gate is against D-7 and this pair clears it, but clearing D-7 is not evidence the model is good — it bounds how uninformative D-7 is. Treat the TSO series as a feature to ingest, not merely as context.
- **IT wind_onshore: the TSO forecast is better than the challenger** (26.2% vs 70.1% WAPE over the same n=1,936). The gate is against D-7 and this pair clears it, but clearing D-7 is not evidence the model is good — it bounds how uninformative D-7 is. Treat the TSO series as a feature to ingest, not merely as context.
- **NO wind_onshore: the TSO forecast is better than the challenger** (9.6% vs 51.5% WAPE over the same n=1,950). The gate is against D-7 and this pair clears it, but clearing D-7 is not evidence the model is good — it bounds how uninformative D-7 is. Treat the TSO series as a feature to ingest, not merely as context.
- **PL wind_onshore: the TSO forecast is better than the challenger** (17.1% vs 52.8% WAPE over the same n=1,950). The gate is against D-7 and this pair clears it, but clearing D-7 is not evidence the model is good — it bounds how uninformative D-7 is. Treat the TSO series as a feature to ingest, not merely as context.
- **PT wind_onshore: the TSO forecast is better than the challenger** (15.6% vs 66.4% WAPE over the same n=1,950). The gate is against D-7 and this pair clears it, but clearing D-7 is not evidence the model is good — it bounds how uninformative D-7 is. Treat the TSO series as a feature to ingest, not merely as context.
- **SE wind_onshore: the TSO forecast is better than the challenger** (4.7% vs 30.2% WAPE over the same n=1,950). The gate is against D-7 and this pair clears it, but clearing D-7 is not evidence the model is good — it bounds how uninformative D-7 is. Treat the TSO series as a feature to ingest, not merely as context.
- **The challenger loses to a constant chosen with hindsight in 14 cell(s).** A flat line at the gate-window median scores better than the fitted model there, so whatever those cells earn over the D-7 bar they earn by predicting close to the level and varying little around it — and not even at the best level. This does not change any verdict above; it bounds what the verdict means:
  - ES wind_onshore 24-36h: challenger 54.27% vs oracle constant 41.49% (+12.78pp)
  - ES wind_onshore 36-48h: challenger 54.18% vs oracle constant 41.49% (+12.69pp)
  - ES wind_onshore 48-64h: challenger 52.43% vs oracle constant 44.23% (+8.20pp)
  - IT wind_onshore 24-36h: challenger 71.39% vs oracle constant 52.32% (+19.07pp)
  - IT wind_onshore 36-48h: challenger 71.19% vs oracle constant 52.29% (+18.90pp)
  - IT wind_onshore 48-64h: challenger 66.84% vs oracle constant 50.76% (+16.08pp)
  - NO wind_onshore 24-36h: challenger 51.36% vs oracle constant 42.42% (+8.93pp)
  - NO wind_onshore 36-48h: challenger 51.56% vs oracle constant 42.42% (+9.14pp)
  - NO wind_onshore 48-64h: challenger 51.80% vs oracle constant 43.07% (+8.72pp)
  - PL wind_onshore 24-36h: challenger 54.10% vs oracle constant 51.19% (+2.92pp)
  - PL wind_onshore 36-48h: challenger 52.50% vs oracle constant 51.19% (+1.31pp)
  - PT wind_onshore 24-36h: challenger 68.20% vs oracle constant 50.17% (+18.04pp)
  - PT wind_onshore 36-48h: challenger 68.70% vs oracle constant 50.17% (+18.53pp)
  - PT wind_onshore 48-64h: challenger 60.97% vs oracle constant 49.32% (+11.65pp)
- **The challenger loses to a climatology chosen with hindsight in 15 cell(s).** An hour-of-day median — the average day, with no model and no weather in it — scores better than the fitted model there. This is the weaker claim to lose and the stronger one to win: a challenger that beats it is doing something no table of hourly averages can do. This does not change any verdict above; it bounds what the verdict means:
  - ES wind_onshore 24-36h: challenger 54.27% vs oracle climatology 27.54% (+26.73pp)
  - ES wind_onshore 36-48h: challenger 54.18% vs oracle climatology 27.54% (+26.64pp)
  - ES wind_onshore 48-64h: challenger 52.43% vs oracle climatology 26.11% (+26.31pp)
  - IT wind_onshore 24-36h: challenger 71.39% vs oracle climatology 45.09% (+26.31pp)
  - IT wind_onshore 36-48h: challenger 71.19% vs oracle climatology 45.08% (+26.11pp)
  - IT wind_onshore 48-64h: challenger 66.84% vs oracle climatology 42.42% (+24.42pp)
  - NO wind_onshore 24-36h: challenger 51.36% vs oracle climatology 41.89% (+9.47pp)
  - NO wind_onshore 36-48h: challenger 51.56% vs oracle climatology 41.89% (+9.68pp)
  - NO wind_onshore 48-64h: challenger 51.80% vs oracle climatology 42.38% (+9.41pp)
  - PL wind_onshore 24-36h: challenger 54.10% vs oracle climatology 47.40% (+6.70pp)
  - PL wind_onshore 36-48h: challenger 52.50% vs oracle climatology 47.40% (+5.09pp)
  - PL wind_onshore 48-64h: challenger 51.35% vs oracle climatology 48.23% (+3.12pp)
  - PT wind_onshore 24-36h: challenger 68.20% vs oracle climatology 39.43% (+28.77pp)
  - PT wind_onshore 36-48h: challenger 68.70% vs oracle climatology 39.43% (+29.26pp)
  - PT wind_onshore 48-64h: challenger 60.97% vs oracle climatology 35.93% (+25.04pp)
- This is one 30-day summer holdout. It is out-of-sample by target timestamp, but not a year-round robustness claim.

## Recommendation to the CEO

Do not promote these artifacts: only 16/24 primary cells clear the registered bar. Treat the losing country/bands as a model-quality finding and move next to stronger wind features/model selection on a fresh pre-registered split.

No production deploy, serving-registry change, model promotion, ingest change, dashboard change, replica write, or sidecar write was performed.
