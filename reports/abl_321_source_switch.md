# ABL-321 — before/after backtest: `energy_renewable` vs `energy_generation`

Generated: 2026-08-12 19:17 UTC
Replica: `C:\Code\able\data\energy_dashboard.db` (9,432,453,120 bytes), opened `mode=ro`, `uri=True`. No write of any kind.
Fit targets: 2026-01-14 00:00:00 → 2026-07-11 00:00:00 (exclusive).
Out-of-sample scoring targets: 2026-07-11 00:00:00 → 2026-08-10 00:00:00 (exclusive). **Out-of-sample by target timestamp.**
Algorithm: catboost, identical in both arms. 8 pre-registered vintages per target hour.
Baseline: literal seasonal-naive D-7, rebuilt from each truth series.
Scoring: common rows only -- both arms finite and truth finite.

**Arms.** before = `energy_renewable`, after = `energy_generation`. The source table sets the training target *and* every lag and rolling feature; nothing else differs.

## Per country/stream, D+2 primary bands (24-36h, 36-48h, 48-64h)

### Scored against truth = `energy_generation (primary)`

| country | stream | n | before WAPE | after WAPE | Δ WAPE | relative | D-7 WAPE | before skill | after skill | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| AT | solar | 1,950 | 12.89% | 13.44% | +0.55 pp | +4.3% | 25.48% | +49.4% | +47.2% | **after WORSE** |
| BE | solar | 1,950 | 16.76% | 16.99% | +0.23 pp | +1.4% | 32.97% | +49.2% | +48.5% | no material change |
| DE | solar | 1,950 | 13.52% | 13.58% | +0.07 pp | +0.5% | 24.16% | +44.0% | +43.8% | no material change |
| FR | solar | 287 | 15.05% | 15.00% | -0.05 pp | -0.3% | 23.35% | +35.5% | +35.8% | no material change |
| BE | wind_offshore | 1,950 | 77.54% | 75.14% | -2.39 pp | -3.1% | 106.20% | +27.0% | +29.2% | **after better** |
| FR | wind_offshore | 287 | 44.04% | 39.72% | -4.33 pp | -9.8% | 53.69% | +18.0% | +26.0% | **after better** |
| AT | wind_onshore | 1,950 | 72.32% | 70.43% | -1.88 pp | -2.6% | 105.63% | +31.5% | +33.3% | **after better** |
| BE | wind_onshore | 1,950 | 46.56% | 47.81% | +1.26 pp | +2.7% | 82.57% | +43.6% | +42.1% | **after WORSE** |
| DE | wind_onshore | 1,950 | 51.63% | 53.50% | +1.87 pp | +3.6% | 77.91% | +33.7% | +31.3% | **after WORSE** |
| FR | wind_onshore | 287 | 39.27% | 32.02% | -7.25 pp | -18.5% | 55.85% | +29.7% | +42.7% | **after better** |

### Scored against truth = `energy_renewable (what the live scorecard uses)`

| country | stream | n | before WAPE | after WAPE | Δ WAPE | relative | D-7 WAPE | before skill | after skill | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| AT | solar | 1,950 | 12.89% | 13.44% | +0.55 pp | +4.3% | 25.48% | +49.4% | +47.2% | **after WORSE** |
| BE | solar | 1,950 | 16.76% | 16.99% | +0.23 pp | +1.4% | 32.97% | +49.2% | +48.5% | no material change |
| DE | solar | 1,950 | 13.52% | 13.58% | +0.07 pp | +0.5% | 24.16% | +44.0% | +43.8% | no material change |
| FR | solar | 287 | 15.05% | 15.00% | -0.05 pp | -0.3% | 23.35% | +35.5% | +35.8% | no material change |
| BE | wind_offshore | 1,950 | 77.38% | 74.94% | -2.44 pp | -3.1% | 105.48% | +26.6% | +28.9% | **after better** |
| FR | wind_offshore | 287 | 44.04% | 39.72% | -4.33 pp | -9.8% | 53.69% | +18.0% | +26.0% | **after better** |
| AT | wind_onshore | 1,950 | 72.32% | 70.43% | -1.88 pp | -2.6% | 105.63% | +31.5% | +33.3% | **after better** |
| BE | wind_onshore | 1,950 | 46.56% | 47.81% | +1.26 pp | +2.7% | 82.57% | +43.6% | +42.1% | **after WORSE** |
| DE | wind_onshore | 1,950 | 51.63% | 53.50% | +1.87 pp | +3.6% | 77.91% | +33.7% | +31.3% | **after WORSE** |
| FR | wind_onshore | 287 | 39.27% | 32.02% | -7.25 pp | -18.5% | 55.85% | +29.7% | +42.7% | **after better** |

## Per horizon band

| country | stream | truth | band | n | intended n | before WAPE | after WAPE | relative |
|---|---|---|---|---:|---:|---:|---:|---:|
| AT | solar | `gen` | 24-36h | 720 | 720 | 13.20% | 13.66% | +3.5% |
| AT | solar | `gen` | 36-48h | 720 | 720 | 12.94% | 13.35% | +3.2% |
| AT | solar | `gen` | 48-64h | 510 | 480 | 12.51% | 13.32% | +6.4% |
| AT | solar | `ren` | 24-36h | 720 | 720 | 13.20% | 13.66% | +3.5% |
| AT | solar | `ren` | 36-48h | 720 | 720 | 12.94% | 13.35% | +3.2% |
| AT | solar | `ren` | 48-64h | 510 | 480 | 12.51% | 13.32% | +6.4% |
| BE | solar | `gen` | 24-36h | 720 | 720 | 15.56% | 15.76% | +1.3% |
| BE | solar | `gen` | 36-48h | 720 | 720 | 16.09% | 16.49% | +2.5% |
| BE | solar | `gen` | 48-64h | 510 | 480 | 18.72% | 18.81% | +0.5% |
| BE | solar | `ren` | 24-36h | 720 | 720 | 15.56% | 15.76% | +1.3% |
| BE | solar | `ren` | 36-48h | 720 | 720 | 16.09% | 16.49% | +2.5% |
| BE | solar | `ren` | 48-64h | 510 | 480 | 18.72% | 18.81% | +0.5% |
| DE | solar | `gen` | 24-36h | 720 | 720 | 13.29% | 13.57% | +2.1% |
| DE | solar | `gen` | 36-48h | 720 | 720 | 13.73% | 13.82% | +0.7% |
| DE | solar | `gen` | 48-64h | 510 | 480 | 13.54% | 13.35% | -1.4% |
| DE | solar | `ren` | 24-36h | 720 | 720 | 13.29% | 13.57% | +2.1% |
| DE | solar | `ren` | 36-48h | 720 | 720 | 13.73% | 13.82% | +0.7% |
| DE | solar | `ren` | 48-64h | 510 | 480 | 13.54% | 13.35% | -1.4% |
| FR | solar | `gen` | 24-36h | 105 | 720 | 14.00% | 13.69% | -2.3% |
| FR | solar | `gen` | 36-48h | 105 | 720 | 15.28% | 15.84% | +3.7% |
| FR | solar | `gen` | 48-64h | 77 | 480 | 15.90% | 15.50% | -2.5% |
| FR | solar | `ren` | 24-36h | 105 | 720 | 14.00% | 13.69% | -2.3% |
| FR | solar | `ren` | 36-48h | 105 | 720 | 15.28% | 15.84% | +3.7% |
| FR | solar | `ren` | 48-64h | 77 | 480 | 15.90% | 15.50% | -2.5% |
| BE | wind_offshore | `gen` | 24-36h | 720 | 720 | 76.34% | 74.40% | -2.5% |
| BE | wind_offshore | `gen` | 36-48h | 720 | 720 | 77.97% | 75.94% | -2.6% |
| BE | wind_offshore | `gen` | 48-64h | 510 | 480 | 78.56% | 75.06% | -4.5% |
| BE | wind_offshore | `ren` | 24-36h | 720 | 720 | 76.15% | 74.17% | -2.6% |
| BE | wind_offshore | `ren` | 36-48h | 720 | 720 | 77.84% | 75.75% | -2.7% |
| BE | wind_offshore | `ren` | 48-64h | 510 | 480 | 78.41% | 74.87% | -4.5% |
| FR | wind_offshore | `gen` | 24-36h | 105 | 720 | 41.69% | 40.01% | -4.0% |
| FR | wind_offshore | `gen` | 36-48h | 105 | 720 | 42.43% | 40.29% | -5.0% |
| FR | wind_offshore | `gen` | 48-64h | 77 | 480 | 49.36% | 38.55% | -21.9% |
| FR | wind_offshore | `ren` | 24-36h | 105 | 720 | 41.69% | 40.01% | -4.0% |
| FR | wind_offshore | `ren` | 36-48h | 105 | 720 | 42.43% | 40.29% | -5.0% |
| FR | wind_offshore | `ren` | 48-64h | 77 | 480 | 49.36% | 38.55% | -21.9% |
| AT | wind_onshore | `gen` | 24-36h | 720 | 720 | 73.31% | 71.21% | -2.9% |
| AT | wind_onshore | `gen` | 36-48h | 720 | 720 | 69.52% | 68.62% | -1.3% |
| AT | wind_onshore | `gen` | 48-64h | 510 | 480 | 74.76% | 71.83% | -3.9% |
| AT | wind_onshore | `ren` | 24-36h | 720 | 720 | 73.31% | 71.21% | -2.9% |
| AT | wind_onshore | `ren` | 36-48h | 720 | 720 | 69.52% | 68.62% | -1.3% |
| AT | wind_onshore | `ren` | 48-64h | 510 | 480 | 74.76% | 71.83% | -3.9% |
| BE | wind_onshore | `gen` | 24-36h | 720 | 720 | 45.49% | 46.14% | +1.4% |
| BE | wind_onshore | `gen` | 36-48h | 720 | 720 | 48.28% | 49.94% | +3.5% |
| BE | wind_onshore | `gen` | 48-64h | 510 | 480 | 45.68% | 47.19% | +3.3% |
| BE | wind_onshore | `ren` | 24-36h | 720 | 720 | 45.49% | 46.14% | +1.4% |
| BE | wind_onshore | `ren` | 36-48h | 720 | 720 | 48.28% | 49.94% | +3.5% |
| BE | wind_onshore | `ren` | 48-64h | 510 | 480 | 45.68% | 47.19% | +3.3% |
| DE | wind_onshore | `gen` | 24-36h | 720 | 720 | 50.87% | 52.54% | +3.3% |
| DE | wind_onshore | `gen` | 36-48h | 720 | 720 | 51.65% | 53.06% | +2.7% |
| DE | wind_onshore | `gen` | 48-64h | 510 | 480 | 52.70% | 55.48% | +5.3% |
| DE | wind_onshore | `ren` | 24-36h | 720 | 720 | 50.87% | 52.54% | +3.3% |
| DE | wind_onshore | `ren` | 36-48h | 720 | 720 | 51.65% | 53.06% | +2.7% |
| DE | wind_onshore | `ren` | 48-64h | 510 | 480 | 52.70% | 55.48% | +5.3% |
| FR | wind_onshore | `gen` | 24-36h | 105 | 720 | 37.01% | 32.40% | -12.5% |
| FR | wind_onshore | `gen` | 36-48h | 105 | 720 | 37.18% | 29.18% | -21.5% |
| FR | wind_onshore | `gen` | 48-64h | 77 | 480 | 45.62% | 35.60% | -22.0% |
| FR | wind_onshore | `ren` | 24-36h | 105 | 720 | 37.01% | 32.40% | -12.5% |
| FR | wind_onshore | `ren` | 36-48h | 105 | 720 | 37.18% | 29.18% | -21.5% |
| FR | wind_onshore | `ren` | 48-64h | 77 | 480 | 45.62% | 35.60% | -22.0% |

## Coverage and data audit

| country | stream | rows both | before only | after only | truth hours (gen/ren) | truth disagreements | dup instants (ren/gen) | fit rows before/after |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| AT | solar | 2,730 | 0 | 0 | 2,730 / 2,730 | 0 | 0 / 0 | 33,287 / 34,176 |
| BE | solar | 2,730 | 0 | 0 | 2,730 / 2,730 | 0 | 0 / 0 | 33,287 / 34,176 |
| DE | solar | 2,730 | 0 | 0 | 2,730 / 2,730 | 0 | 0 / 0 | 33,316 / 34,176 |
| FR | solar | 395 | 2,335 | 0 | 1,669 / 2,730 | 0 | 0 / 0 | 33,079 / 32,048 |
| AT | wind_onshore | 2,730 | 0 | 0 | 2,730 / 2,730 | 0 | 0 / 0 | 33,287 / 34,176 |
| BE | wind_onshore | 2,730 | 0 | 0 | 2,730 / 2,730 | 0 | 0 / 0 | 33,287 / 34,176 |
| DE | wind_onshore | 2,730 | 0 | 0 | 2,730 / 2,730 | 0 | 0 / 0 | 33,316 / 34,176 |
| FR | wind_onshore | 395 | 2,335 | 0 | 1,669 / 2,730 | 0 | 0 / 0 | 33,079 / 32,048 |
| BE | wind_offshore | 2,730 | 0 | 0 | 2,730 / 2,730 | 561 | 0 / 0 | 32,033 / 34,176 |
| FR | wind_offshore | 395 | 2,335 | 0 | 1,669 / 2,730 | 12 | 0 / 0 | 33,079 / 32,048 |

## Caveats

- One 30-day summer holdout. Out-of-sample by target timestamp, not year-round evidence.
- FR `energy_generation` is missing 2026-06-30 23:45 → 2026-07-22 14:15 (518.5 h, ABL-318 §3, not covered by ABL-71/67/111/109). That eats the fit window's tail and the first 11.6 days of the scoring window for FR, so FR's `after` arm trains on less and scores on fewer rows. Common-row scoring keeps the comparison fair; the lost coverage is the separate finding.
- ABL-67 is net-position-only; ABL-109/111 are load-only; ABL-71's known wrong-write modes are load and net position. None is a proof that solar/wind ingest is pristine.
- TSO forecasts are not used here. They are revision-contaminated and cannot support promotion.
- No production deploy, serving-registry change, model promotion, ingest change, dashboard change, replica write or sidecar write was performed.
