# ABL-253 — Serve-faithful solar retrain gate

**Disposition: PASS**

Generated: 2026-08-13 11:39 UTC
Fit targets: 2026-01-14 00:00:00 → 2026-07-11 00:00:00 (exclusive).
Out-of-sample gate targets: 2026-07-11 00:00:00 → 2026-08-10 00:00:00 (exclusive).
Baseline: literal seasonal-naive D-7. TSO is revision-contaminated context only and is not a gate criterion.
Replica: `C:\Code\able\data\energy_dashboard.db` (9,432,453,120 bytes), opened with SQLite `mode=ro`, `uri=True`.
That one file is the source of the TSO series, the contamination screen, and — since ABL-355 — the fitted target series, its lag/rolling features, the D-7 and persistence baselines, the gate actuals and the weather archive. The incumbent forecasts are the only read it does not hold alone; see the sidecar below.
Sidecar: `C:\Code\able\data\forecasts_local.db`, also opened `mode=ro`, and read for locally generated incumbent forecasts only. Where a sidecar row and a replica row carry the same vintage, the sidecar's is the one scored.
`ENERGY_DB_PATH` resolved to `\data\energy_dashboard.db` and was **not** read by this run. Before ABL-355 that path, not the replica, is where the fitted series would have come from.
Target series, features, baselines and contamination screen: `energy_renewable`.

## Gate read

Registered scope `abl253`: BE, DE, FR.
Gate basis — the columns that must be simultaneously finite for a row to be scored: `challenger`, `incumbent`, `seasonal_naive`, `persistence`. Comparators outside the basis are scored on their own intersection with it and carry their own n, so a comparator that does not exist for a country reads Not measured instead of emptying the cell.
Strict full PASS requires challenger WAPE < D-7 in all 9 country × primary D+2-band cells and ≥95% of intended pairs. Result: **9/9 cells pass**.

| country | horizon | n | challenger WAPE | D-7 WAPE | skill vs D-7 | incumbent WAPE | MAE | bias | slope | corr | gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| BE | 24-36h | 720 | 15.6% | 32.9% | +52.7% | 23.5% | 336.9 MW | 1.5% | 0.9 | 1.0 | PASS |
| BE | 36-48h | 720 | 16.1% | 32.9% | +51.1% | 23.7% | 348.3 MW | 1.5% | 0.9 | 1.0 | PASS |
| BE | 48-64h | 480 | 18.9% | 33.2% | +43.1% | 23.7% | 544.3 MW | 3.1% | 0.9 | 0.9 | PASS |
| DE | 24-36h | 720 | 13.4% | 24.2% | +44.7% | 62.3% | 2257.5 MW | -1.9% | 1.0 | 1.0 | PASS |
| DE | 36-48h | 720 | 13.7% | 24.2% | +43.4% | 62.4% | 2312.5 MW | -2.1% | 1.0 | 1.0 | PASS |
| DE | 48-64h | 480 | 12.8% | 23.3% | +45.3% | 62.8% | 2743.6 MW | -3.9% | 0.9 | 1.0 | PASS |
| FR | 24-36h | 720 | 14.2% | 22.5% | +37.0% | 16.3% | 910.5 MW | -2.5% | 0.9 | 1.0 | PASS |
| FR | 36-48h | 720 | 14.7% | 22.5% | +34.4% | 16.3% | 948.5 MW | -2.9% | 0.9 | 1.0 | PASS |
| FR | 48-64h | 480 | 14.7% | 22.2% | +33.7% | 16.0% | 1281.2 MW | -4.6% | 0.9 | 1.0 | PASS |

The exact eight registered run instants imply 210/570/720/720/510 selected rows by band. As in ABL-195, the frozen registered minimum for 48–64h remains 456 (95% of 480), while the schedule offers 510 rows.

## Per-country all-D+2 summary

Gate-basis values (actual, challenger, incumbent, seasonal_naive, persistence) share one finite intersection; each comparator outside the basis is scored on its own intersection with it, and its n is given in `comparator_n` in the JSON. A comparator showing `Not measured` had no finite rows at all.

| country | n | challenger WAPE | D-7 WAPE | persistence WAPE | incumbent WAPE | TSO WAPE (revision-contaminated; n) |
|---|---:|---:|---:|---:|---:|---:|
| BE | 1,920 | 16.8% | 33.0% | 96.7% | 23.6% | 9.1% (n=1,920) |
| DE | 1,920 | 13.3% | 23.9% | 90.2% | 62.5% | 10.2% (n=1,920) |
| FR | 1,920 | 14.5% | 22.4% | 94.4% | 16.2% | 14.5% (n=1,920) |

## Fit and missingness audit

Every training row was built with `RenewableFeatureBuilder.row(target, generated_at, observation_as_of=generated_at)`. Gate targets were never fitted.

| country | algorithm | retained / intended fit rows | unique fit targets | excluded missing | degraded lag-1d rows | artifact SHA-256 |
|---|---|---:|---:|---:|---:|---|
| BE | catboost | 33,287 / 34,176 | 4,188 | 889 | 23,042 | `1319d8b7199eeca6f8dde988f9ced4c44eec91993b5662cd407848ed9948acd7` |
| DE | catboost | 33,316 / 34,176 | 4,191 | 860 | 23,062 | `e6dc829e5c2296c540bca9f8d6a440dec4e605c45c1adccf6d628c676014621b` |
| FR | catboost | 33,087 / 34,176 | 4,163 | 1,089 | 22,905 | `e73642b895b440bfa6affc9e8fb479e480b8cbffb0cb97b063e9ed1adcc5312e` |

### Physically impossible night rows (ABL-376)

Not registered for scope `abl253`. The fit saw every night row, including any whose actual the sun says is impossible.

## Data quality and limits

- ABL-188 constant-run screening found no ≥24-hour bit-identical solar run in `energy_renewable` over the registered fit/scoring interval plus 14-day feature lookback (2025-12-31 → 2026-08-10 UTC). The builder still routes solar through `exclude_suspect_constant_runs`; the invariant was verified on the actual window, not assumed from ABL-191.
- The known DE zero-fill run (2025-09-08 22:00 → 2025-11-14 15:45 UTC; 6,408 quarter-hours) is outside this fit/lookback window.
- The audit initially appeared to flag FR zero from 2025-12-31 17:00 to 2026-01-02 07:15 UTC, but the replica has no intervening New Year's Day rows and `energy_generation` independently agrees on zero for the available nighttime observations. `find_suspect_constant_runs` was incorrectly joining equal values across missing-time gaps despite its contiguous-run contract. The invariant now splits on cadence gaps; the original continuous DE defect remains covered by regression tests.
- ABL-67 is net-position-only; ABL-109/111 are load-only. ABL-71's known wrong-write modes are load and net position, not solar; this is a provenance caveat, not proof that solar ingest is pristine.
- Weather rows were admitted only where `forecast_run_time <= generated_at`; missing vintages were excluded, never filled with future reanalysis.
- TSO values come from an `INSERT OR REPLACE` table without first-seen vintages. They may include revisions and cannot support promotion.
- This is one 30-day summer holdout. It is out-of-sample by target timestamp, but not year-round evidence.

## Recommendation to the CEO

The challenger clears the pre-registered D-7 bar in every served solar D+2 country-band cell. Preserve these experiment artifacts and ask the CEO to initiate Board review; do not promote from this issue.

No production deploy, serving-registry change, model promotion, ingest change, dashboard change, replica write, or sidecar write was performed.
