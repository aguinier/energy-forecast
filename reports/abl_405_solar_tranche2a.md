# ABL-405 — Serve-faithful solar retrain gate, ABL-316 tranche 2a: 8 continental countries on energy_renewable at 27 features

**Disposition: PERFORMANCE PASS — HOLD FOR CONTAMINATION ADJUDICATION**

Generated: 2026-08-13 20:18 UTC
Fit targets: 2026-01-14 00:00:00 → 2026-07-11 00:00:00 (exclusive).
Out-of-sample gate targets: 2026-07-11 00:00:00 → 2026-08-10 00:00:00 (exclusive).
Baseline: literal seasonal-naive D-7. TSO is revision-contaminated context only and is not a gate criterion.
Replica: `C:\Code\able\data\energy_dashboard.db` (9,432,453,120 bytes), opened with SQLite `mode=ro`, `uri=True`.
That one file is the source of the TSO series, the contamination screen, and — since ABL-355 — the fitted target series, its lag/rolling features, the D-7 and persistence baselines, the gate actuals and the weather archive. The incumbent forecasts are the only read it does not hold alone; see the sidecar below.
Sidecar: `C:\Code\able\data\forecasts_local.db`, also opened `mode=ro`, and read for locally generated incumbent forecasts only. Where a sidecar row and a replica row carry the same vintage, the sidecar's is the one scored.
Target series, features, baselines and contamination screen: `energy_renewable`.
Feature set: **legacy25+geometry** (27 columns), the module default -- this scope registers no feature set of its own.

## Gate read

Registered scope `abl316-t2a`: BG, CH, CZ, HU, PL, RO, SI, SK.
Gate basis — the columns that must be simultaneously finite for a row to be scored: `challenger`, `seasonal_naive`. Comparators outside the basis are scored on their own intersection with it and carry their own n, so a comparator that does not exist for a country reads Not measured instead of emptying the cell.
Strict full PASS requires challenger WAPE < D-7 in all 24 country × primary D+2-band cells and ≥95% of intended pairs. Result: **24/24 cells pass**.

Model-free reference (ABL-389) — four predictors with no model in them, reported beside every cell. `constant_causal` is a flat line at the **fit-window mean**, the honest "no model" floor, using only what was knowable before the gate window opened; `constant_oracle` is a flat line at the **gate-window median**, a hindsight upper bound on what any constant could have achieved. `climatology_causal` and `climatology_oracle` are the same two forms taken **per hour of day** — the fit-window hourly mean and the gate-window hourly median — which is the tighter reference on every pair measured so far, because a constant is a climatology with one bucket. Read the pair together: the constant says whether the model predicts the *level*, the climatology says whether it predicts the level *and the daily shape*, and the gap between them is how much of this series is forced diurnal structure.

All four are **reported references and not gate criteria**: none is in the gate basis, none can move a cell's verdict, and a pair that clears its D-7 bar while losing to one still reads PASS. They are the number that qualifies the PASS — a challenger that does not beat `climatology_oracle` has not demonstrated skill beyond the average day, and a D-7 bar that `constant_causal` clears on its own was not a demanding bar. **Check each reference's own n before comparing it to the challenger.** A climatology is 24 levels, so an hour of day absent from its source window leaves those rows unscored for that column alone; scored on different rows, two WAPEs are not the same measurement. Nothing is interpolated to close that gap.

| country | horizon | n | challenger WAPE | D-7 WAPE | skill vs D-7 | constant causal WAPE | constant oracle WAPE | climatology causal WAPE | climatology oracle WAPE | incumbent WAPE | MAE | bias | slope | corr | gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| BG | 24-36h | 720 | 19.6% | 24.4% | +19.6% | 75.3% | 73.5% | 41.9% | 19.2% | Not measured | 282.5 MW | -13.9% | 0.8 | 0.9 | PASS |
| BG | 36-48h | 720 | 19.5% | 24.4% | +20.0% | 75.3% | 73.5% | 41.9% | 19.2% | Not measured | 280.8 MW | -14.0% | 0.8 | 0.9 | PASS |
| BG | 48-64h | 510 | 20.8% | 25.0% | +16.7% | 68.1% | 63.8% | 41.2% | 20.4% | Not measured | 353.7 MW | -13.7% | 0.8 | 0.9 | PASS |
| CH | 24-36h | 720 | 7.7% | 12.7% | +39.5% | 94.9% | 94.6% | 39.7% | 9.0% | Not measured | 102.0 MW | -1.1% | 1.0 | 1.0 | PASS |
| CH | 36-48h | 720 | 7.5% | 12.7% | +40.7% | 94.9% | 94.6% | 39.7% | 9.0% | Not measured | 100.0 MW | -1.1% | 1.0 | 1.0 | PASS |
| CH | 48-64h | 510 | 8.0% | 12.5% | +36.2% | 86.3% | 87.9% | 39.1% | 8.7% | Not measured | 139.9 MW | -2.0% | 1.0 | 1.0 | PASS |
| CZ | 24-36h | 720 | 12.9% | 24.0% | +46.3% | 93.9% | 92.8% | 29.3% | 15.9% | Not measured | 123.3 MW | -1.9% | 0.9 | 1.0 | PASS |
| CZ | 36-48h | 720 | 12.9% | 24.0% | +46.3% | 93.9% | 92.8% | 29.3% | 15.9% | Not measured | 123.2 MW | -2.4% | 0.9 | 1.0 | PASS |
| CZ | 48-64h | 510 | 14.0% | 24.0% | +41.9% | 85.8% | 89.9% | 28.4% | 16.1% | Not measured | 169.8 MW | -2.9% | 0.9 | 1.0 | PASS |
| HU | 24-36h | 720 | 17.3% | 18.2% | +4.6% | 95.7% | 95.0% | 30.9% | 14.2% | Not measured | 230.1 MW | -12.5% | 0.8 | 1.0 | PASS |
| HU | 36-48h | 720 | 17.3% | 18.2% | +4.6% | 95.7% | 95.0% | 30.9% | 14.2% | Not measured | 230.1 MW | -12.5% | 0.8 | 1.0 | PASS |
| HU | 48-64h | 510 | 16.5% | 17.9% | +7.6% | 88.6% | 91.4% | 29.8% | 14.3% | Not measured | 269.8 MW | -9.4% | 0.9 | 1.0 | PASS |
| PL | 24-36h | 720 | 17.3% | 26.0% | +33.3% | 92.6% | 92.2% | 28.1% | 15.4% | Not measured | 707.4 MW | -11.4% | 0.8 | 1.0 | PASS |
| PL | 36-48h | 720 | 17.4% | 26.0% | +33.2% | 92.6% | 92.2% | 28.1% | 15.4% | Not measured | 708.7 MW | -11.5% | 0.8 | 1.0 | PASS |
| PL | 48-64h | 510 | 16.3% | 24.5% | +33.5% | 85.9% | 88.1% | 27.1% | 14.6% | Not measured | 817.7 MW | -8.8% | 0.9 | 1.0 | PASS |
| RO | 24-36h | 720 | 18.8% | 24.3% | +22.8% | 96.3% | 95.8% | 43.4% | 19.9% | Not measured | 161.6 MW | -9.5% | 0.8 | 1.0 | PASS |
| RO | 36-48h | 720 | 18.7% | 24.3% | +23.0% | 96.3% | 95.8% | 43.4% | 19.9% | Not measured | 161.0 MW | -9.1% | 0.8 | 1.0 | PASS |
| RO | 48-64h | 510 | 19.2% | 25.0% | +23.3% | 92.3% | 93.0% | 42.4% | 20.4% | Not measured | 197.1 MW | -10.1% | 0.8 | 1.0 | PASS |
| SI | 24-36h | 720 | 17.9% | 21.6% | +17.3% | 95.0% | 93.8% | 35.1% | 13.0% | Not measured | 59.5 MW | -10.0% | 0.8 | 1.0 | PASS |
| SI | 36-48h | 720 | 18.1% | 21.6% | +16.4% | 95.0% | 93.8% | 35.1% | 13.0% | Not measured | 60.1 MW | -10.5% | 0.8 | 1.0 | PASS |
| SI | 48-64h | 510 | 18.7% | 21.2% | +12.1% | 86.6% | 90.1% | 34.3% | 12.8% | Not measured | 79.8 MW | -12.1% | 0.8 | 1.0 | PASS |
| SK | 24-36h | 715 | 16.3% | 18.8% | +13.3% | 97.1% | 95.3% | 32.6% | 13.1% | Not measured | 18.7 MW | -9.9% | 0.9 | 1.0 | PASS |
| SK | 36-48h | 715 | 16.4% | 18.8% | +13.0% | 97.1% | 95.3% | 32.6% | 13.1% | Not measured | 18.8 MW | -9.7% | 0.9 | 1.0 | PASS |
| SK | 48-64h | 507 | 15.1% | 18.3% | +17.9% | 89.8% | 93.1% | 31.6% | 12.5% | Not measured | 21.8 MW | -6.9% | 0.9 | 1.0 | PASS |

Reference levels used, from the same ABL-188-filtered target series the gate actuals and the D-7/persistence baselines come from — no refit, no second read, no additional upstream fetch. The hourly levels behind the climatology columns are in `results.json` in full; `h` is how many of the 24 hours of the day that level set covers, and anything below 24 means those rows were dropped from that column's n:

| country | constant causal | constant oracle | climatology causal | climatology oracle |
|---|---:|---:|---:|---:|
| BG | 856.38 MW | 1087.86 MW | 2.17–1983.56 MW (24h) | 1.21–3282.79 MW (24h) |
| CH | 803.50 MW | 677.22 MW | 0.00–2413.51 MW (24h) | 0.00–3695.79 MW (24h) |
| CZ | 697.34 MW | 378.00 MW | 0.00–2058.80 MW (24h) | 0.00–2683.30 MW (24h) |
| HU | 940.48 MW | 614.62 MW | 0.51–2636.20 MW (24h) | 0.23–3362.25 MW (24h) |
| PL | 3035.49 MW | 2336.98 MW | 0.00–8492.75 MW (24h) | 0.00–10590.47 MW (24h) |
| RO | 511.94 MW | 408.12 MW | 0.00–1454.91 MW (24h) | 0.00–2424.62 MW (24h) |
| SI | 222.21 MW | 122.17 MW | 0.55–670.36 MW (24h) | 1.15–977.35 MW (24h) |
| SK | 79.05 MW | 37.70 MW | 0.22–246.50 MW (24h) | 0.84–352.99 MW (24h) |

## Per-country all-D+2 summary

Gate-basis values (actual, challenger, seasonal_naive) share one finite intersection; each comparator outside the basis is scored on its own intersection with it, and its n is given in `comparator_n` in the JSON. A comparator showing `Not measured` had no finite rows at all.

| country | n | challenger WAPE | D-7 WAPE | persistence WAPE | constant causal WAPE | constant oracle WAPE | climatology causal WAPE | climatology oracle WAPE | incumbent WAPE | TSO WAPE (revision-contaminated; n) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BG | 1,950 | 19.9% | 24.6% | 73.2% | 73.2% | 70.6% | 41.7% | 19.5% | Not measured | 33.2% (n=1,950) |
| CH | 1,950 | 7.7% | 12.6% | 87.5% | 92.2% | 92.5% | 39.5% | 8.9% | Not measured | 7.1% (n=1,950) |
| CZ | 1,950 | 13.2% | 24.0% | 86.1% | 91.4% | 91.9% | 29.0% | 16.0% | Not measured | 11.6% (n=1,950) |
| HU | 1,950 | 17.1% | 18.1% | 86.4% | 93.5% | 93.9% | 30.6% | 14.2% | Not measured | 14.7% (n=1,950) |
| PL | 1,950 | 17.0% | 25.5% | 86.2% | 90.6% | 90.9% | 27.8% | 15.2% | Not measured | 16.0% (n=1,950) |
| RO | 1,950 | 18.9% | 24.5% | 90.5% | 95.1% | 95.0% | 43.1% | 20.1% | Not measured | 30.2% (n=1,950) |
| SI | 1,950 | 18.2% | 21.5% | 86.6% | 92.4% | 92.7% | 34.8% | 12.9% | Not measured | 18.7% (n=1,950) |
| SK | 1,937 | 16.0% | 18.7% | 86.9% | 94.9% | 94.6% | 32.3% | 12.9% | Not measured | 14.4% (n=1,872) |

## Fit and missingness audit

Every training row was built with `RenewableFeatureBuilder.row(target, generated_at, observation_as_of=generated_at)`. Gate targets were never fitted.

| country | algorithm | retained / intended fit rows | unique fit targets | excluded missing | degraded lag-1d rows | artifact SHA-256 |
|---|---|---:|---:|---:|---:|---|
| BG | catboost | 33,140 / 34,176 | 4,173 | 1,036 | 22,940 | `399e00b0bac2f00c72a542175d7c3fc20f65a1d9eb3f284299bd7e62f1cb4aef` |
| CH | catboost | 33,287 / 34,176 | 4,188 | 889 | 23,042 | `360442b429425e95db32bf0325afdcbccebb74fc4dc16c5a9745ea13507de32e` |
| CZ | catboost | 30,954 / 34,176 | 3,881 | 3,222 | 21,399 | `4bce4a6987b98762b2f77981378e7faab18f8878fe01f5961e9ac4af51ea136e` |
| HU | catboost | 33,316 / 34,176 | 4,191 | 860 | 23,062 | `b0eeeac5a4f08778cb47e38340cd39d41d623b303362d516b6d31479585996f9` |
| PL | catboost | 33,316 / 34,176 | 4,191 | 860 | 23,062 | `69494bed4702a4d53b9ed2495e06f9a23a45420f47c5d1e1aa6eb7b63906cc3a` |
| RO | catboost | 33,316 / 34,176 | 4,191 | 860 | 23,062 | `92fae90fefb0271b449a0b6891cfc06f427dfca9407b6f5efdd4d587d51d5ced` |
| SI | catboost | 33,287 / 34,176 | 4,188 | 889 | 23,042 | `1ee2b2fa9605f54220687989347afe76beada8bd02e4b2f76967e3d07458cee6` |
| SK | catboost | 33,131 / 34,176 | 4,173 | 1,045 | 22,937 | `b35fc12554c64dc6428a838a666422b0fa45119b464372cdf0b75b3dd45b6a0a` |

### Physically impossible night rows (ABL-376)

Not registered for scope `abl316-t2a`. The fit saw every night row, including any whose actual the sun says is impossible.

## Data quality and limits

- ABL-188 screening found suspect solar runs for CZ in `energy_renewable`: `[{'start': '2026-02-11 17:00:00', 'end': '2026-02-15 13:45:00', 'value': 0.0, 'n_rows': 372, 'duration_hours': 92.75}]`. The builder nulls these before fit; see the training audit and recommendation.
- ABL-67 is net-position-only; ABL-109/111 are load-only. ABL-71's known wrong-write modes are load and net position, not solar; this is a provenance caveat, not proof that solar ingest is pristine.
- Weather rows were admitted only where `forecast_run_time <= generated_at`; missing vintages were excluded, never filled with future reanalysis.
- TSO values come from an `INSERT OR REPLACE` table without first-seen vintages. They may include revisions and cannot support promotion.
- This is one 30-day summer holdout. It is out-of-sample by target timestamp, but not year-round evidence.
- **The challenger loses to a climatology chosen with hindsight in 15 cell(s).** An hour-of-day median — the average day, with no model and no weather in it — scores better than the fitted model there. This is the weaker claim to lose and the stronger one to win: a challenger that beats it is doing something no table of hourly averages can do. This does not change any verdict above; it bounds what the verdict means:
  - BG solar 24-36h: challenger 19.63% vs oracle climatology 19.15% (+0.47pp)
  - BG solar 36-48h: challenger 19.51% vs oracle climatology 19.15% (+0.35pp)
  - BG solar 48-64h: challenger 20.82% vs oracle climatology 20.38% (+0.43pp)
  - HU solar 24-36h: challenger 17.35% vs oracle climatology 14.19% (+3.15pp)
  - HU solar 36-48h: challenger 17.35% vs oracle climatology 14.19% (+3.15pp)
  - HU solar 48-64h: challenger 16.53% vs oracle climatology 14.29% (+2.24pp)
  - PL solar 24-36h: challenger 17.34% vs oracle climatology 15.40% (+1.93pp)
  - PL solar 36-48h: challenger 17.37% vs oracle climatology 15.40% (+1.97pp)
  - PL solar 48-64h: challenger 16.30% vs oracle climatology 14.62% (+1.68pp)
  - SI solar 24-36h: challenger 17.91% vs oracle climatology 13.01% (+4.91pp)
  - SI solar 36-48h: challenger 18.11% vs oracle climatology 13.01% (+5.10pp)
  - SI solar 48-64h: challenger 18.65% vs oracle climatology 12.76% (+5.89pp)
  - SK solar 24-36h: challenger 16.32% vs oracle climatology 13.11% (+3.21pp)
  - SK solar 36-48h: challenger 16.37% vs oracle climatology 13.11% (+3.26pp)
  - SK solar 48-64h: challenger 15.07% vs oracle climatology 12.52% (+2.55pp)

## Recommendation to the CEO

The challenger clears the performance bar, but a suspect constant run touches the registered data window. Do not promote; send the run to the CEO/ingest owner for adjudication first.

No production deploy, serving-registry change, model promotion, ingest change, dashboard change, replica write, or sidecar write was performed.
