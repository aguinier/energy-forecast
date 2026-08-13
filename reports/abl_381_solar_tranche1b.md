# ABL-381 — Serve-faithful solar retrain gate, ABL-316 tranche 1b: BG and CH on energy_generation

**Disposition: PASS**

Generated: 2026-08-13 13:00 UTC
Fit targets: 2026-01-14 00:00:00 → 2026-07-11 00:00:00 (exclusive).
Out-of-sample gate targets: 2026-07-11 00:00:00 → 2026-08-10 00:00:00 (exclusive).
Baseline: literal seasonal-naive D-7. TSO is revision-contaminated context only and is not a gate criterion.
Replica: `C:\Code\able\data\energy_dashboard.db` (9,432,453,120 bytes), opened with SQLite `mode=ro`, `uri=True`.
That one file is the source of the TSO series, the contamination screen, and — since ABL-355 — the fitted target series, its lag/rolling features, the D-7 and persistence baselines, the gate actuals and the weather archive. The incumbent forecasts are the only read it does not hold alone; see the sidecar below.
Sidecar: `C:\Code\able\data\forecasts_local.db`, also opened `mode=ro`, and read for locally generated incumbent forecasts only. Where a sidecar row and a replica row carry the same vintage, the sidecar's is the one scored.
Target series, features, baselines and contamination screen: `energy_generation`.

## Gate read

Registered scope `abl316-t1b`: BG, CH.
Gate basis — the columns that must be simultaneously finite for a row to be scored: `challenger`, `seasonal_naive`. Comparators outside the basis are scored on their own intersection with it and carry their own n, so a comparator that does not exist for a country reads Not measured instead of emptying the cell.
Strict full PASS requires challenger WAPE < D-7 in all 6 country × primary D+2-band cells and ≥95% of intended pairs. Result: **6/6 cells pass**.

Model-free reference (ABL-389) — four predictors with no model in them, reported beside every cell. `constant_causal` is a flat line at the **fit-window mean**, the honest "no model" floor, using only what was knowable before the gate window opened; `constant_oracle` is a flat line at the **gate-window median**, a hindsight upper bound on what any constant could have achieved. `climatology_causal` and `climatology_oracle` are the same two forms taken **per hour of day** — the fit-window hourly mean and the gate-window hourly median — which is the tighter reference on every pair measured so far, because a constant is a climatology with one bucket. Read the pair together: the constant says whether the model predicts the *level*, the climatology says whether it predicts the level *and the daily shape*, and the gap between them is how much of this series is forced diurnal structure.

All four are **reported references and not gate criteria**: none is in the gate basis, none can move a cell's verdict, and a pair that clears its D-7 bar while losing to one still reads PASS. They are the number that qualifies the PASS — a challenger that does not beat `climatology_oracle` has not demonstrated skill beyond the average day, and a D-7 bar that `constant_causal` clears on its own was not a demanding bar. **Check each reference's own n before comparing it to the challenger.** A climatology is 24 levels, so an hour of day absent from its source window leaves those rows unscored for that column alone; scored on different rows, two WAPEs are not the same measurement. Nothing is interpolated to close that gap.

| country | horizon | n | challenger WAPE | D-7 WAPE | skill vs D-7 | constant causal WAPE | constant oracle WAPE | climatology causal WAPE | climatology oracle WAPE | incumbent WAPE | MAE | bias | slope | corr | gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| BG | 24-36h | 720 | 18.9% | 24.4% | +22.6% | 75.3% | 73.5% | 42.0% | 19.2% | Not measured | 271.8 MW | -12.3% | 0.8 | 0.9 | PASS |
| BG | 36-48h | 720 | 18.6% | 24.4% | +23.8% | 75.3% | 73.5% | 42.0% | 19.2% | Not measured | 267.7 MW | -12.5% | 0.8 | 1.0 | PASS |
| BG | 48-64h | 510 | 20.0% | 25.0% | +19.9% | 68.2% | 63.8% | 41.3% | 20.4% | Not measured | 340.3 MW | -12.2% | 0.8 | 0.9 | PASS |
| CH | 24-36h | 720 | 8.2% | 12.7% | +35.6% | 95.1% | 94.6% | 37.5% | 9.0% | Not measured | 108.6 MW | 1.4% | 1.0 | 1.0 | PASS |
| CH | 36-48h | 720 | 8.0% | 12.7% | +36.8% | 95.1% | 94.6% | 37.5% | 9.0% | Not measured | 106.6 MW | 1.5% | 1.0 | 1.0 | PASS |
| CH | 48-64h | 510 | 8.4% | 12.5% | +33.0% | 86.0% | 87.9% | 36.6% | 8.7% | Not measured | 147.0 MW | 0.3% | 1.0 | 1.0 | PASS |

Reference levels used, from the same ABL-188-filtered target series the gate actuals and the D-7/persistence baselines come from — no refit, no second read, no additional upstream fetch. The hourly levels behind the climatology columns are in `results.json` in full; `h` is how many of the 24 hours of the day that level set covers, and anything below 24 means those rows were dropped from that column's n:

| country | constant causal | constant oracle | climatology causal | climatology oracle |
|---|---:|---:|---:|---:|
| BG | 855.24 MW | 1087.86 MW | 2.17–1979.49 MW (24h) | 1.21–3282.79 MW (24h) |
| CH | 833.37 MW | 677.22 MW | 0.73–2586.18 MW (24h) | 0.00–3695.79 MW (24h) |

## Per-country all-D+2 summary

Gate-basis values (actual, challenger, seasonal_naive) share one finite intersection; each comparator outside the basis is scored on its own intersection with it, and its n is given in `comparator_n` in the JSON. A comparator showing `Not measured` had no finite rows at all.

| country | n | challenger WAPE | D-7 WAPE | persistence WAPE | constant causal WAPE | constant oracle WAPE | climatology causal WAPE | climatology oracle WAPE | incumbent WAPE | TSO WAPE (revision-contaminated; n) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BG | 1,950 | 19.1% | 24.6% | 73.2% | 73.2% | 70.6% | 41.8% | 19.5% | Not measured | 33.2% (n=1,950) |
| CH | 1,950 | 8.2% | 12.6% | 87.5% | 92.2% | 92.5% | 37.2% | 8.9% | Not measured | 7.1% (n=1,950) |

## Fit and missingness audit

Every training row was built with `RenewableFeatureBuilder.row(target, generated_at, observation_as_of=generated_at)`. Gate targets were never fitted.

| country | algorithm | retained / intended fit rows | unique fit targets | excluded missing | degraded lag-1d rows | artifact SHA-256 |
|---|---|---:|---:|---:|---:|---|
| BG | catboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `c6c62b6021bdf354eea7ae6f6070e8e434cffe277df9679d12587bca106af201` |
| CH | catboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `26ea961436e2710471d318da490347416d6491f8aa92d9ca3573ed3c4a500b95` |

### Physically impossible night rows (ABL-376)

Not registered for scope `abl316-t1b`. The fit saw every night row, including any whose actual the sun says is impossible.

## Data quality and limits

- ABL-188 constant-run screening found no ≥24-hour bit-identical solar run in `energy_generation` over the registered fit/scoring interval plus 14-day feature lookback (2025-12-31 → 2026-08-10 UTC). The builder still routes solar through `exclude_suspect_constant_runs`; the invariant was verified on the actual window, not assumed from ABL-191.
- ABL-67 is net-position-only; ABL-109/111 are load-only. ABL-71's known wrong-write modes are load and net position, not solar; this is a provenance caveat, not proof that solar ingest is pristine.
- Weather rows were admitted only where `forecast_run_time <= generated_at`; missing vintages were excluded, never filled with future reanalysis.
- TSO values come from an `INSERT OR REPLACE` table without first-seen vintages. They may include revisions and cannot support promotion.
- This is one 30-day summer holdout. It is out-of-sample by target timestamp, but not year-round evidence.

## Recommendation to the CEO

The challenger clears the pre-registered D-7 bar in every served solar D+2 country-band cell. Preserve these experiment artifacts and ask the CEO to initiate Board review; do not promote from this issue.

No production deploy, serving-registry change, model promotion, ingest change, dashboard change, replica write, or sidecar write was performed.
