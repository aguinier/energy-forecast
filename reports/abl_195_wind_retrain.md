# ABL-195 — Serve-faithful wind retrain gate

**Disposition: PERFORMANCE PASS ? HOLD FOR CONTAMINATION ADJUDICATION**

Generated: 2026-08-11 13:00 UTC
Fit targets: 2026-01-14 00:00:00 → 2026-07-11 00:00:00 (exclusive).
Out-of-sample gate targets: 2026-07-11 00:00:00 → 2026-08-10 00:00:00 (exclusive).
Baseline: literal seasonal-naive D-7. TSO is revision-contaminated context only and is not a gate criterion.
Replica: `C:\Code\able\data\energy_dashboard.db` (5,905,461,248 bytes), opened with SQLite `mode=ro`, `uri=True`.

## Gate read

Strict full PASS requires challenger WAPE < D-7 in all 15 country × primary D+2-band cells and ≥95% of intended pairs. Result: **15/15 cells pass**.
Protocol count check (before fitting): the exact eight registered run instants produce 210/570/720/720/510 selected rows by band, not the registered 240/600/720/720/480. The primary 24–36h and 36–48h counts reproduce; 48–64h has 510 rows and is still judged against the frozen registered minimum of 456.

| type | country | horizon | n | challenger WAPE | D-7 WAPE | skill vs D-7 | incumbent WAPE | MAE | bias | slope | corr | gate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| wind_offshore | BE | 24-36h | 720 | 76.3% | 104.1% | +26.7% | 157.4% | 361.9 MW | -8.2% | 0.2 | 0.4 | PASS |
| wind_offshore | BE | 36-48h | 720 | 76.8% | 104.1% | +26.2% | 158.9% | 364.3 MW | -8.6% | 0.2 | 0.4 | PASS |
| wind_offshore | BE | 48-64h | 480 | 77.5% | 109.6% | +29.3% | 160.7% | 396.3 MW | -15.3% | 0.2 | 0.4 | PASS |
| wind_offshore | FR | 24-36h | 720 | 53.5% | 77.1% | +30.6% | 79.3% | 310.1 MW | -23.7% | 0.2 | 0.4 | PASS |
| wind_offshore | FR | 36-48h | 720 | 53.5% | 77.1% | +30.6% | 78.5% | 310.1 MW | -22.1% | 0.2 | 0.4 | PASS |
| wind_offshore | FR | 48-64h | 480 | 57.4% | 80.0% | +28.2% | 83.7% | 318.1 MW | -20.2% | 0.2 | 0.4 | PASS |
| wind_onshore | BE | 24-36h | 720 | 45.5% | 83.6% | +45.6% | 189.7% | 206.9 MW | -4.2% | 0.5 | 0.7 | PASS |
| wind_onshore | BE | 36-48h | 720 | 48.3% | 83.6% | +42.3% | 191.1% | 219.5 MW | -6.2% | 0.4 | 0.7 | PASS |
| wind_onshore | BE | 48-64h | 480 | 45.4% | 79.4% | +42.9% | 168.5% | 223.9 MW | -14.8% | 0.5 | 0.7 | PASS |
| wind_onshore | DE | 24-36h | 720 | 50.9% | 76.1% | +33.1% | 62.3% | 4308.0 MW | 32.4% | 0.4 | 0.6 | PASS |
| wind_onshore | DE | 36-48h | 720 | 51.6% | 76.1% | +32.1% | 61.6% | 4374.2 MW | 32.5% | 0.4 | 0.6 | PASS |
| wind_onshore | DE | 48-64h | 480 | 51.8% | 81.6% | +36.5% | 61.6% | 4424.2 MW | 35.9% | 0.4 | 0.6 | PASS |
| wind_onshore | FR | 24-36h | 720 | 41.7% | 55.3% | +24.5% | 99.1% | 1408.7 MW | -5.5% | 0.2 | 0.4 | PASS |
| wind_onshore | FR | 36-48h | 720 | 41.7% | 55.3% | +24.6% | 98.1% | 1406.8 MW | -5.5% | 0.3 | 0.4 | PASS |
| wind_onshore | FR | 48-64h | 480 | 41.7% | 56.8% | +26.5% | 109.0% | 1378.8 MW | -6.2% | 0.3 | 0.5 | PASS |

## Per-country all-D+2 summary

All model and baseline values use the identical finite challenger/incumbent/D-7/persistence/actual intersection.

| type | country | n | challenger WAPE | D-7 WAPE | persistence WAPE | incumbent WAPE | TSO WAPE (revision-contaminated; n) |
|---|---|---:|---:|---:|---:|---:|---:|
| wind_offshore | BE | 1,920 | 76.8% | 105.6% | 112.9% | 158.8% | 26.3% (n=1,920) |
| wind_offshore | FR | 1,920 | 54.5% | 77.8% | 75.1% | 80.1% | 25.5% (n=1,920) |
| wind_onshore | BE | 1,920 | 46.5% | 82.5% | 87.4% | 184.6% | 21.5% (n=1,920) |
| wind_onshore | DE | 1,920 | 51.4% | 77.5% | 72.4% | 61.8% | 12.1% (n=1,920) |
| wind_onshore | FR | 1,920 | 41.7% | 55.7% | 55.6% | 101.2% | 15.4% (n=1,920) |

## Fit and missingness audit

Each training row was constructed by `RenewableFeatureBuilder.row(target, generated_at, generated_at)` on the measured eight-vintage schedule. Gate targets were never fitted.

| type | country | algorithm | retained / intended fit rows | unique fit targets | excluded missing | degraded lag-1d rows | artifact SHA-256 |
|---|---|---|---:|---:|---:|---:|---|
| wind_offshore | BE | xgboost | 32,033 / 34,176 | 4,063 | 2,143 | 22,108 | `304460cc5345f1d51397066112fca9fa49c114f712458a13dd2e8a5262adb01e` |
| wind_offshore | FR | xgboost | 33,079 / 34,176 | 4,162 | 1,097 | 22,901 | `ccb5cb51d761cca429592b9be0fecc76d92dee8c17a42b49ab113da32a42d95b` |
| wind_onshore | BE | catboost | 33,287 / 34,176 | 4,188 | 889 | 23,042 | `0dab6910d9f35329178e96fe76b1621f7ea4c8e19874e1d67c89ac3ea0fd542d` |
| wind_onshore | DE | catboost | 33,316 / 34,176 | 4,191 | 860 | 23,062 | `195da7db5feff9da7ead952558cd28ff6d0122fc785f76c0978ba55a7d47b7ac` |
| wind_onshore | FR | catboost | 33,079 / 34,176 | 4,162 | 1,097 | 22,901 | `8b475ac2522a541b7f2764e04c15b699b581514f3c9b6834d74fc613b0b2f778` |

## Data quality and limits

- ABL-188 found one fit-window suspect run: BE offshore was exactly 0 MW from 2026-03-08 09:00 through 2026-03-10 00:00 UTC (40 hourly rows; 39 hours). Those labels and any dependent feature rows were treated as missing before fit. It does not intersect the July/August gate actuals (all 5,760 scheduled gate rows per pair were feature/label-complete), so the performance gate is evaluable; promotion remains on hold pending CEO/ingest adjudication.
- ABL-67 is net-position-only; ABL-109/111 are load-only. They do not intersect these wind targets. ABL-71's known wrong-write modes are load and net position, not wind; this is a provenance caveat, not proof that wind ingest is pristine.
- Weather rows were admitted only where `forecast_run_time <= generated_at`; missing vintages were excluded and counted, never backfilled from the future.
- TSO values come from a replacement table without first-seen vintages. They may include revisions and cannot support promotion.
- This is one 30-day summer holdout. It is out-of-sample by target timestamp, but not a year-round robustness claim.

## Recommendation to the CEO

The challenger clears the pre-registered D-7 performance bar in every served D+2 country-band cell. Do not promote yet: hand the newly detected BE offshore zero run to the CEO/ingest owner for adjudication, then return these experiment artifacts and this evidence to the CEO for Board review. This issue does not promote them.

No production deploy, serving-registry change, model promotion, ingest change, dashboard change, replica write, or sidecar write was performed.
