# ABL-253 — Serve-faithful solar retrain gate

**Disposition: PASS**

Generated: 2026-08-13 08:59 UTC
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

| country | horizon | n | challenger WAPE | D-7 WAPE | skill vs D-7 | incumbent WAPE | MAE | bias | slope | corr | gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| BG | 24-36h | 720 | 18.9% | 24.4% | +22.6% | Not measured | 271.8 MW | -12.3% | 0.8 | 0.9 | PASS |
| BG | 36-48h | 720 | 18.6% | 24.4% | +23.8% | Not measured | 267.7 MW | -12.5% | 0.8 | 1.0 | PASS |
| BG | 48-64h | 510 | 20.0% | 25.0% | +19.9% | Not measured | 340.3 MW | -12.2% | 0.8 | 0.9 | PASS |
| CH | 24-36h | 720 | 8.2% | 12.7% | +35.6% | Not measured | 108.6 MW | 1.4% | 1.0 | 1.0 | PASS |
| CH | 36-48h | 720 | 8.0% | 12.7% | +36.8% | Not measured | 106.6 MW | 1.5% | 1.0 | 1.0 | PASS |
| CH | 48-64h | 510 | 8.4% | 12.5% | +33.0% | Not measured | 147.0 MW | 0.3% | 1.0 | 1.0 | PASS |

## Per-country all-D+2 summary

Gate-basis values (actual, challenger, seasonal_naive) share one finite intersection; each comparator outside the basis is scored on its own intersection with it, and its n is given in `comparator_n` in the JSON. A comparator showing `Not measured` had no finite rows at all.

| country | n | challenger WAPE | D-7 WAPE | persistence WAPE | incumbent WAPE | TSO WAPE (revision-contaminated; n) |
|---|---:|---:|---:|---:|---:|---:|
| BG | 1,950 | 19.1% | 24.6% | 73.2% | Not measured | 33.2% (n=1,950) |
| CH | 1,950 | 8.2% | 12.6% | 87.5% | Not measured | 7.1% (n=1,950) |

## Fit and missingness audit

Every training row was built with `RenewableFeatureBuilder.row(target, generated_at, observation_as_of=generated_at)`. Gate targets were never fitted.

| country | algorithm | retained / intended fit rows | unique fit targets | excluded missing | degraded lag-1d rows | artifact SHA-256 |
|---|---|---:|---:|---:|---:|---|
| BG | catboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `9bbe1e74b555a5a9d42ba929bcc83255d75affce86d12635b52c2bde485aa5ae` |
| CH | catboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `9ff1a53de87e5f9a306187be4894287bc7bfeb13fdadd099654db40ea580dd9f` |

## Data quality and limits

- ABL-188 constant-run screening found no ≥24-hour bit-identical solar run in `energy_generation` over the registered fit/scoring interval plus 14-day feature lookback (2025-12-31 → 2026-08-10 UTC). The builder still routes solar through `exclude_suspect_constant_runs`; the invariant was verified on the actual window, not assumed from ABL-191.
- ABL-67 is net-position-only; ABL-109/111 are load-only. ABL-71's known wrong-write modes are load and net position, not solar; this is a provenance caveat, not proof that solar ingest is pristine.
- Weather rows were admitted only where `forecast_run_time <= generated_at`; missing vintages were excluded, never filled with future reanalysis.
- TSO values come from an `INSERT OR REPLACE` table without first-seen vintages. They may include revisions and cannot support promotion.
- This is one 30-day summer holdout. It is out-of-sample by target timestamp, but not year-round evidence.

## Recommendation to the CEO

The challenger clears the pre-registered D-7 bar in every served solar D+2 country-band cell. Preserve these experiment artifacts and ask the CEO to initiate Board review; do not promote from this issue.

No production deploy, serving-registry change, model promotion, ingest change, dashboard change, replica write, or sidecar write was performed.
