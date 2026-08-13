# ABL-376 — Serve-faithful solar retrain gate, impossible night rows excluded from the fit

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

Registered scope `abl376`: BE, DE, FR.
Gate basis — the columns that must be simultaneously finite for a row to be scored: `challenger`, `incumbent`, `seasonal_naive`, `persistence`. Comparators outside the basis are scored on their own intersection with it and carry their own n, so a comparator that does not exist for a country reads Not measured instead of emptying the cell.
Strict full PASS requires challenger WAPE < D-7 in all 9 country × primary D+2-band cells and ≥95% of intended pairs. Result: **9/9 cells pass**.

| country | horizon | n | challenger WAPE | D-7 WAPE | skill vs D-7 | incumbent WAPE | MAE | bias | slope | corr | gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| BE | 24-36h | 720 | 15.6% | 32.9% | +52.7% | 23.5% | 336.9 MW | 1.5% | 0.9 | 1.0 | PASS |
| BE | 36-48h | 720 | 16.1% | 32.9% | +51.1% | 23.7% | 348.3 MW | 1.5% | 0.9 | 1.0 | PASS |
| BE | 48-64h | 480 | 18.9% | 33.2% | +43.1% | 23.7% | 544.3 MW | 3.1% | 0.9 | 0.9 | PASS |
| DE | 24-36h | 720 | 13.3% | 24.2% | +45.1% | 62.3% | 2241.1 MW | -2.1% | 0.9 | 1.0 | PASS |
| DE | 36-48h | 720 | 13.5% | 24.2% | +44.3% | 62.4% | 2275.0 MW | -2.2% | 0.9 | 1.0 | PASS |
| DE | 48-64h | 480 | 12.7% | 23.3% | +45.6% | 62.8% | 2726.9 MW | -4.1% | 0.9 | 1.0 | PASS |
| FR | 24-36h | 720 | 14.3% | 22.5% | +36.4% | 16.3% | 919.4 MW | -2.2% | 0.9 | 1.0 | PASS |
| FR | 36-48h | 720 | 14.9% | 22.5% | +33.7% | 16.3% | 958.1 MW | -2.5% | 0.9 | 1.0 | PASS |
| FR | 48-64h | 480 | 15.0% | 22.2% | +32.6% | 16.0% | 1303.2 MW | -4.1% | 0.9 | 1.0 | PASS |

## Per-country all-D+2 summary

Gate-basis values (actual, challenger, incumbent, seasonal_naive, persistence) share one finite intersection; each comparator outside the basis is scored on its own intersection with it, and its n is given in `comparator_n` in the JSON. A comparator showing `Not measured` had no finite rows at all.

| country | n | challenger WAPE | D-7 WAPE | persistence WAPE | incumbent WAPE | TSO WAPE (revision-contaminated; n) |
|---|---:|---:|---:|---:|---:|---:|
| BE | 1,920 | 16.8% | 33.0% | 96.7% | 23.6% | 9.1% (n=1,920) |
| DE | 1,920 | 13.2% | 23.9% | 90.2% | 62.5% | 10.2% (n=1,920) |
| FR | 1,920 | 14.7% | 22.4% | 94.4% | 16.2% | 14.5% (n=1,920) |

## Fit and missingness audit

Every training row was built with `RenewableFeatureBuilder.row(target, generated_at, observation_as_of=generated_at)`. Gate targets were never fitted.

| country | algorithm | retained / intended fit rows | unique fit targets | excluded missing | degraded lag-1d rows | artifact SHA-256 |
|---|---|---:|---:|---:|---:|---|
| BE | catboost | 33,287 / 34,176 | 4,188 | 889 | 23,042 | `2a5ed17dd6f1b10fa6c03e151691f826e6cbfa8b8d0136126812cc4bf813244b` |
| DE | catboost | 33,316 / 34,176 | 4,191 | 860 | 23,062 | `6a0ed3c0e7689ac07d54ca74ca2d4b15566a2ad0983e30a5e2bbbda77763b7c6` |
| FR | catboost | 33,087 / 34,176 | 4,163 | 1,089 | 22,905 | `4aa429940eba18d57fa2b8dc1c9827c5c4acb3c3108b79d89ca68af755b420ba` |

### Physically impossible night rows (ABL-376)

Night is `solar_geometry.is_night_hour` — the serving clamp's own predicate, sun below -8 deg geometric for the whole hour. A night row whose actual exceeds **1 MW** is physically impossible and was dropped **from the fit only**. The gate frame below was not filtered: a contaminated actual still scores against the challenger, which is why the daylight numbers above are not marking their own homework.

Rows are per (target, vintage); `hours` is the distinct contaminated target hours behind them — the row count is what the fit lost, the hour count is what the source got wrong.

| country | night fit rows | excluded rows | excluded hours | max excluded actual | mean night actual (before) |
|---|---:|---:|---:|---:|---:|
| BE | 10,856 | 0 | 0 | n/a | 0.0 MW |
| DE | 10,952 | 32 | 4 | 1.7 MW | 0.0 MW |
| FR | 11,648 | 904 | 113 | 285.9 MW | 17.2 MW |

A country reading 0 excluded is the rule finding clean data, not the rule being off — the predicate is the sun's, so it is stated over countries rather than for the one that prompted it.

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
