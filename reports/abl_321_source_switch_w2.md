# ABL-321 — before/after backtest: `energy_renewable` vs `energy_generation`

Generated: 2026-08-12 19:12 UTC
Replica: `C:\Code\able\data\energy_dashboard.db` (9,432,453,120 bytes), opened `mode=ro`, `uri=True`. No write of any kind.
Fit targets: 2025-11-21 00:00:00 → 2026-02-15 00:00:00 (exclusive).
Out-of-sample scoring targets: 2026-02-15 00:00:00 → 2026-03-17 00:00:00 (exclusive). **Out-of-sample by target timestamp.**
Algorithm: catboost, identical in both arms. 8 pre-registered vintages per target hour.
Baseline: literal seasonal-naive D-7, rebuilt from each truth series.
Scoring: common rows only -- both arms finite and truth finite.

**Arms.** before = `energy_renewable`, after = `energy_generation`. The source table sets the training target *and* every lag and rolling feature; nothing else differs.

## Per country/stream, D+2 primary bands (24-36h, 36-48h, 48-64h)

### Scored against truth = `energy_generation (primary)`

| country | stream | n | before WAPE | after WAPE | Δ WAPE | relative | D-7 WAPE | before skill | after skill | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| AT | solar | 1,719 | 68.33% | 71.83% | +3.50 pp | +5.1% | 47.38% | -44.2% | -51.6% | **after WORSE** |
| BE | solar | 1,719 | 63.02% | 62.97% | -0.05 pp | -0.1% | 54.37% | -15.9% | -15.8% | no material change |
| DE | solar | 1,728 | 61.39% | 53.40% | -7.99 pp | -13.0% | 43.46% | -41.3% | -22.9% | **after better** |
| FR | solar | 1,719 | 37.35% | 37.08% | -0.27 pp | -0.7% | 28.25% | -32.2% | -31.3% | no material change |
| BE | wind_offshore | 1,495 | 69.29% | 68.98% | -0.31 pp | -0.4% | 110.03% | +37.0% | +37.3% | no material change |
| FR | wind_offshore | 1,719 | 68.73% | 68.53% | -0.21 pp | -0.3% | 84.18% | +18.4% | +18.6% | no material change |
| AT | wind_onshore | 1,719 | 76.00% | 73.09% | -2.90 pp | -3.8% | 98.24% | +22.6% | +25.6% | **after better** |
| BE | wind_onshore | 1,719 | 59.77% | 59.29% | -0.49 pp | -0.8% | 88.29% | +32.3% | +32.9% | no material change |
| DE | wind_onshore | 1,728 | 57.11% | 57.93% | +0.82 pp | +1.4% | 74.01% | +22.8% | +21.7% | no material change |
| FR | wind_onshore | 1,719 | 49.00% | 48.68% | -0.32 pp | -0.7% | 71.80% | +31.8% | +32.2% | no material change |

### Scored against truth = `energy_renewable (what the live scorecard uses)`

| country | stream | n | before WAPE | after WAPE | Δ WAPE | relative | D-7 WAPE | before skill | after skill | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| AT | solar | 1,643 | 70.37% | 73.82% | +3.45 pp | +4.9% | 47.43% | -48.4% | -55.6% | **after WORSE** |
| BE | solar | 1,643 | 63.26% | 63.21% | -0.05 pp | -0.1% | 54.05% | -17.0% | -16.9% | no material change |
| DE | solar | 1,654 | 62.60% | 54.53% | -8.08 pp | -12.9% | 44.54% | -40.6% | -22.4% | **after better** |
| FR | solar | 1,643 | 37.82% | 37.54% | -0.28 pp | -0.8% | 28.18% | -34.2% | -33.2% | no material change |
| BE | wind_offshore | 1,419 | 71.14% | 70.89% | -0.26 pp | -0.4% | 112.47% | +36.7% | +37.0% | no material change |
| FR | wind_offshore | 1,643 | 71.71% | 71.43% | -0.29 pp | -0.4% | 86.72% | +17.3% | +17.6% | no material change |
| AT | wind_onshore | 1,643 | 76.91% | 73.86% | -3.05 pp | -4.0% | 96.73% | +20.5% | +23.6% | **after better** |
| BE | wind_onshore | 1,643 | 58.53% | 60.29% | +1.76 pp | +3.0% | 88.02% | +33.5% | +31.5% | **after WORSE** |
| DE | wind_onshore | 1,654 | 57.86% | 58.62% | +0.76 pp | +1.3% | 74.47% | +22.3% | +21.3% | no material change |
| FR | wind_onshore | 1,643 | 50.50% | 50.49% | -0.02 pp | -0.0% | 71.72% | +29.6% | +29.6% | no material change |

## Per horizon band

| country | stream | truth | band | n | intended n | before WAPE | after WAPE | relative |
|---|---|---|---|---:|---:|---:|---:|---:|
| AT | solar | `gen` | 24-36h | 636 | 720 | 68.74% | 72.42% | +5.4% |
| AT | solar | `gen` | 36-48h | 636 | 720 | 68.87% | 72.53% | +5.3% |
| AT | solar | `gen` | 48-64h | 447 | 480 | 67.40% | 70.56% | +4.7% |
| AT | solar | `ren` | 24-36h | 609 | 720 | 70.65% | 74.30% | +5.2% |
| AT | solar | `ren` | 36-48h | 608 | 720 | 70.87% | 74.49% | +5.1% |
| AT | solar | `ren` | 48-64h | 426 | 480 | 69.60% | 72.69% | +4.4% |
| BE | solar | `gen` | 24-36h | 636 | 720 | 62.64% | 62.61% | -0.1% |
| BE | solar | `gen` | 36-48h | 636 | 720 | 63.19% | 63.16% | -0.1% |
| BE | solar | `gen` | 48-64h | 447 | 480 | 63.22% | 63.15% | -0.1% |
| BE | solar | `ren` | 24-36h | 609 | 720 | 62.79% | 62.76% | -0.1% |
| BE | solar | `ren` | 36-48h | 608 | 720 | 63.53% | 63.50% | -0.1% |
| BE | solar | `ren` | 48-64h | 426 | 480 | 63.45% | 63.37% | -0.1% |
| DE | solar | `gen` | 24-36h | 639 | 720 | 61.52% | 53.38% | -13.2% |
| DE | solar | `gen` | 36-48h | 639 | 720 | 61.81% | 54.04% | -12.6% |
| DE | solar | `gen` | 48-64h | 450 | 480 | 60.83% | 52.76% | -13.3% |
| DE | solar | `ren` | 24-36h | 612 | 720 | 62.75% | 54.49% | -13.2% |
| DE | solar | `ren` | 36-48h | 612 | 720 | 63.07% | 55.26% | -12.4% |
| DE | solar | `ren` | 48-64h | 430 | 480 | 61.98% | 53.83% | -13.2% |
| FR | solar | `gen` | 24-36h | 636 | 720 | 37.36% | 37.10% | -0.7% |
| FR | solar | `gen` | 36-48h | 636 | 720 | 37.27% | 36.99% | -0.8% |
| FR | solar | `gen` | 48-64h | 447 | 480 | 37.42% | 37.15% | -0.7% |
| FR | solar | `ren` | 24-36h | 609 | 720 | 37.75% | 37.48% | -0.7% |
| FR | solar | `ren` | 36-48h | 608 | 720 | 37.77% | 37.48% | -0.8% |
| FR | solar | `ren` | 48-64h | 426 | 480 | 37.95% | 37.67% | -0.7% |
| BE | wind_offshore | `gen` | 24-36h | 556 | 720 | 68.91% | 68.66% | -0.4% |
| BE | wind_offshore | `gen` | 36-48h | 557 | 720 | 68.26% | 67.72% | -0.8% |
| BE | wind_offshore | `gen` | 48-64h | 382 | 480 | 71.45% | 71.42% | -0.1% |
| BE | wind_offshore | `ren` | 24-36h | 529 | 720 | 70.45% | 70.28% | -0.2% |
| BE | wind_offshore | `ren` | 36-48h | 529 | 720 | 69.95% | 69.44% | -0.7% |
| BE | wind_offshore | `ren` | 48-64h | 361 | 480 | 74.09% | 74.09% | -0.0% |
| FR | wind_offshore | `gen` | 24-36h | 636 | 720 | 67.93% | 67.83% | -0.1% |
| FR | wind_offshore | `gen` | 36-48h | 636 | 720 | 67.79% | 67.60% | -0.3% |
| FR | wind_offshore | `gen` | 48-64h | 447 | 480 | 71.29% | 70.91% | -0.5% |
| FR | wind_offshore | `ren` | 24-36h | 609 | 720 | 70.76% | 70.58% | -0.2% |
| FR | wind_offshore | `ren` | 36-48h | 608 | 720 | 70.65% | 70.39% | -0.4% |
| FR | wind_offshore | `ren` | 48-64h | 426 | 480 | 74.71% | 74.21% | -0.7% |
| AT | wind_onshore | `gen` | 24-36h | 636 | 720 | 77.47% | 74.58% | -3.7% |
| AT | wind_onshore | `gen` | 36-48h | 636 | 720 | 75.58% | 72.69% | -3.8% |
| AT | wind_onshore | `gen` | 48-64h | 447 | 480 | 74.57% | 71.63% | -3.9% |
| AT | wind_onshore | `ren` | 24-36h | 609 | 720 | 78.49% | 75.48% | -3.8% |
| AT | wind_onshore | `ren` | 36-48h | 608 | 720 | 76.53% | 73.30% | -4.2% |
| AT | wind_onshore | `ren` | 48-64h | 426 | 480 | 75.27% | 72.42% | -3.8% |
| BE | wind_onshore | `gen` | 24-36h | 636 | 720 | 59.41% | 58.63% | -1.3% |
| BE | wind_onshore | `gen` | 36-48h | 636 | 720 | 58.59% | 58.49% | -0.2% |
| BE | wind_onshore | `gen` | 48-64h | 447 | 480 | 62.04% | 61.41% | -1.0% |
| BE | wind_onshore | `ren` | 24-36h | 609 | 720 | 58.01% | 59.48% | +2.5% |
| BE | wind_onshore | `ren` | 36-48h | 608 | 720 | 57.70% | 59.84% | +3.7% |
| BE | wind_onshore | `ren` | 48-64h | 426 | 480 | 60.51% | 62.15% | +2.7% |
| DE | wind_onshore | `gen` | 24-36h | 639 | 720 | 55.62% | 56.52% | +1.6% |
| DE | wind_onshore | `gen` | 36-48h | 639 | 720 | 56.04% | 56.85% | +1.4% |
| DE | wind_onshore | `gen` | 48-64h | 450 | 480 | 60.93% | 61.64% | +1.2% |
| DE | wind_onshore | `ren` | 24-36h | 612 | 720 | 56.39% | 57.24% | +1.5% |
| DE | wind_onshore | `ren` | 36-48h | 612 | 720 | 56.59% | 57.36% | +1.4% |
| DE | wind_onshore | `ren` | 48-64h | 430 | 480 | 61.96% | 62.57% | +1.0% |
| FR | wind_onshore | `gen` | 24-36h | 636 | 720 | 47.18% | 47.02% | -0.3% |
| FR | wind_onshore | `gen` | 36-48h | 636 | 720 | 47.46% | 47.35% | -0.2% |
| FR | wind_onshore | `gen` | 48-64h | 447 | 480 | 54.00% | 53.13% | -1.6% |
| FR | wind_onshore | `ren` | 24-36h | 609 | 720 | 48.52% | 48.61% | +0.2% |
| FR | wind_onshore | `ren` | 36-48h | 608 | 720 | 48.87% | 49.14% | +0.6% |
| FR | wind_onshore | `ren` | 48-64h | 426 | 480 | 55.99% | 55.37% | -1.1% |

## Coverage and data audit

| country | stream | rows both | before only | after only | truth hours (gen/ren) | truth disagreements | dup instants (ren/gen) | fit rows before/after |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| AT | solar | 2,409 | 4 | 321 | 2,734 / 2,626 | 1,084 | 0 / 0 | 6,288 / 6,288 |
| BE | solar | 2,409 | 4 | 321 | 2,734 / 2,626 | 0 | 0 / 0 | 6,288 / 6,288 |
| DE | solar | 2,421 | 4 | 309 | 2,734 / 2,631 | 2,631 | 0 / 0 | 6,288 / 6,288 |
| FR | solar | 2,409 | 4 | 321 | 2,734 / 2,626 | 0 | 0 / 0 | 6,080 / 6,080 |
| AT | wind_onshore | 2,409 | 4 | 321 | 2,734 / 2,626 | 94 | 0 / 0 | 6,288 / 6,288 |
| BE | wind_onshore | 2,409 | 4 | 321 | 2,734 / 2,626 | 2,626 | 0 / 0 | 6,288 / 6,288 |
| DE | wind_onshore | 2,421 | 4 | 309 | 2,734 / 2,631 | 2,631 | 0 / 0 | 6,288 / 6,288 |
| FR | wind_onshore | 2,409 | 4 | 321 | 2,734 / 2,626 | 0 | 0 / 0 | 6,080 / 6,080 |
| BE | wind_offshore | 2,111 | 8 | 619 | 2,738 / 2,630 | 368 | 0 / 0 | 6,288 / 6,288 |
| FR | wind_offshore | 2,409 | 4 | 321 | 2,734 / 2,626 | 31 | 0 / 0 | 6,080 / 6,080 |

## Caveats

- One 30-day summer holdout. Out-of-sample by target timestamp, not year-round evidence.
- FR `energy_generation` is missing 2026-06-30 23:45 → 2026-07-22 14:15 (518.5 h, ABL-318 §3, not covered by ABL-71/67/111/109). That eats the fit window's tail and the first 11.6 days of the scoring window for FR, so FR's `after` arm trains on less and scores on fewer rows. Common-row scoring keeps the comparison fair; the lost coverage is the separate finding.
- ABL-67 is net-position-only; ABL-109/111 are load-only; ABL-71's known wrong-write modes are load and net position. None is a proof that solar/wind ingest is pristine.
- TSO forecasts are not used here. They are revision-contaminated and cannot support promotion.
- No production deploy, serving-registry change, model promotion, ingest change, dashboard change, replica write or sidecar write was performed.
