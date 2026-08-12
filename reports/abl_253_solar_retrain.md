# ABL-253 — Serve-faithful solar retrain gate

**Disposition: PASS**

Generated: 2026-08-12 06:51 UTC
Fit targets: 2026-01-14 00:00:00 → 2026-07-11 00:00:00 (exclusive).
Out-of-sample gate targets: 2026-07-11 00:00:00 → 2026-08-10 00:00:00 (exclusive).
Baseline: literal seasonal-naive D-7. TSO is revision-contaminated context only and is not a gate criterion.
Replica: `C:\Code\able\data\energy_dashboard.db` (9,432,453,120 bytes), opened with SQLite `mode=ro`, `uri=True`.

## Gate read

Strict full PASS requires challenger WAPE < D-7 in all 9 served-country × primary D+2-band cells and ≥95% of intended pairs. Result: **9/9 cells pass**.
The exact eight registered run instants imply 210/570/720/720/510 selected rows by band. As in ABL-195, the frozen registered minimum for 48–64h remains 456 (95% of 480), while the schedule offers 510 rows.

| country | horizon | n | challenger WAPE | D-7 WAPE | skill vs D-7 | incumbent WAPE | MAE | bias | slope | corr | gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| BE | 24-36h | 720 | 15.6% | 32.9% | +52.7% | 23.5% | 336.9 MW | 1.5% | 0.9 | 1.0 | PASS |
| BE | 36-48h | 720 | 16.1% | 32.9% | +51.1% | 23.7% | 348.3 MW | 1.5% | 0.9 | 1.0 | PASS |
| BE | 48-64h | 480 | 18.9% | 33.2% | +43.1% | 23.7% | 544.3 MW | 3.1% | 0.9 | 0.9 | PASS |
| DE | 24-36h | 720 | 13.3% | 24.2% | +45.2% | 62.4% | 2243.2 MW | -2.3% | 0.9 | 1.0 | PASS |
| DE | 36-48h | 720 | 13.7% | 24.2% | +43.4% | 62.4% | 2318.2 MW | -2.3% | 0.9 | 1.0 | PASS |
| DE | 48-64h | 480 | 13.0% | 23.5% | +44.7% | 63.9% | 2902.0 MW | -4.4% | 0.9 | 1.0 | PASS |
| FR | 24-36h | 720 | 14.3% | 22.8% | +37.4% | 20.2% | 916.4 MW | -2.5% | 0.9 | 1.0 | PASS |
| FR | 36-48h | 720 | 14.7% | 22.8% | +35.2% | 20.2% | 947.5 MW | -2.7% | 0.9 | 1.0 | PASS |
| FR | 48-64h | 480 | 14.6% | 22.6% | +35.6% | 17.8% | 1303.2 MW | -4.3% | 0.9 | 1.0 | PASS |

## Per-country all-D+2 summary

All model and baseline values use the identical finite challenger/incumbent/D-7/persistence/actual intersection.

| country | n | challenger WAPE | D-7 WAPE | persistence WAPE | incumbent WAPE | TSO WAPE (revision-contaminated; n) |
|---|---:|---:|---:|---:|---:|---:|
| BE | 1,920 | 16.8% | 33.0% | 96.7% | 23.6% | 9.1% (n=1,920) |
| DE | 1,920 | 13.4% | 24.0% | 92.6% | 62.9% | 4.9% (n=1,920) |
| FR | 1,920 | 14.5% | 22.7% | 95.4% | 19.5% | 7.5% (n=1,920) |

## Fit and missingness audit

Every training row was built with `RenewableFeatureBuilder.row(target, generated_at, observation_as_of=generated_at)`. Gate targets were never fitted.

| country | algorithm | retained / intended fit rows | unique fit targets | excluded missing | degraded lag-1d rows | artifact SHA-256 |
|---|---|---:|---:|---:|---:|---|
| BE | catboost | 33,287 / 34,176 | 4,188 | 889 | 23,042 | `16f559f6373d8f505ca115dae4fbbdb753dfb5cce436b56127752260fbd224c2` |
| DE | catboost | 33,316 / 34,176 | 4,191 | 860 | 23,062 | `19a18f9aef0b7ac66e160b0c70572ebbf69c509910ad38f21ade8b6b39daed64` |
| FR | catboost | 33,079 / 34,176 | 4,162 | 1,097 | 22,901 | `94a62af55ac8b014e54b36451131741d9fdb4dad2bc74c300b4306cf0e3d423d` |

## Data quality and limits

- ABL-188 constant-run screening found no ≥24-hour bit-identical solar run in the registered fit/scoring interval plus 14-day feature lookback (2025-12-31 → 2026-08-10 UTC). The known DE zero-fill run (2025-09-08 22:00 → 2025-11-14 15:45 UTC; 6,408 quarter-hours) is outside this fit/lookback window. The builder still routes solar through `exclude_suspect_constant_runs`; the invariant was verified on the actual window, not assumed from ABL-191.
- The audit initially appeared to flag FR zero from 2025-12-31 17:00 to 2026-01-02 07:15 UTC, but the replica has no intervening New Year's Day rows and `energy_generation` independently agrees on zero for the available nighttime observations. `find_suspect_constant_runs` was incorrectly joining equal values across missing-time gaps despite its contiguous-run contract. The invariant now splits on cadence gaps; the original continuous DE defect remains covered by regression tests.
- ABL-67 is net-position-only; ABL-109/111 are load-only. ABL-71's known wrong-write modes are load and net position, not solar; this is a provenance caveat, not proof that solar ingest is pristine.
- Weather rows were admitted only where `forecast_run_time <= generated_at`; missing vintages were excluded, never filled with future reanalysis.
- TSO values come from an `INSERT OR REPLACE` table without first-seen vintages. They may include revisions and cannot support promotion.
- This is one 30-day summer holdout. It is out-of-sample by target timestamp, but not year-round evidence.

## Recommendation to the CEO

The challenger clears the pre-registered D-7 bar in every served solar D+2 country-band cell. Preserve these experiment artifacts and ask the CEO to initiate Board review; do not promote from this issue.

No production deploy, serving-registry change, model promotion, ingest change, dashboard change, replica write, or sidecar write was performed.
