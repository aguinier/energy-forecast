# ABL-187 — Price bias/affine correction holdout

Generated: 2026-08-11 08:38 UTC
Fit window: 2026-02-03 00:00:00 → 2026-07-11 00:00:00 (exclusive)
Holdout: 2026-07-11 00:00:00 → 2026-08-10 00:00:00 (exclusive)
Protocol: latest vintage per country + target + model + horizon band; fit and holdout are disjoint by target timestamp.
Sample: 52,440 out-of-sample rows across 19 countries; 243,983 training rows.
Baseline: literal seasonal-naive D−7, scored on the exact finite holdout intersection used by all variants.

## Verdict

**Do not ship this correction.** The best corrected variant (bias_only) is 3.2 WAPE points worse than free D−7 on the pooled holdout.

Bias-only helped raw CatBoost in **14/19** measured countries and beat D−7 in **3/19**. Slope+intercept helped raw in **14/19** and beat D−7 in **2/19**.

## Holdout results by country

All WAPE values below are out-of-sample. A negative Δ is an improvement over raw.

| country | n fit | n holdout | raw CatBoost | bias-only | Δ vs raw | beats D−7? | affine | Δ vs raw | beats D−7? | seasonal-naive D−7 |
|---|---:|---:|---:|---:|---:|:---:|---:|---:|:---:|---:|
| AT | 12,944 | 2,760 | 30.2% | 27.0% | -3.2 pp | no | 27.1% | -3.1 pp | no | 20.4% |
| BG | 12,944 | 2,760 | 32.3% | 28.5% | -3.9 pp | no | 28.7% | -3.6 pp | no | 26.3% |
| CH | 12,944 | 2,760 | 22.3% | 21.9% | -0.4 pp | no | 20.9% | -1.4 pp | no | 18.0% |
| CZ | 12,944 | 2,760 | 32.1% | 29.3% | -2.8 pp | no | 28.0% | -4.1 pp | no | 23.0% |
| EE | 12,944 | 2,760 | 88.2% | 93.1% | +4.9 pp | yes | 103.6% | +15.4 pp | yes | 116.5% |
| FI | 12,944 | 2,760 | 76.5% | 87.7% | +11.2 pp | no | 113.5% | +37.0 pp | no | 77.1% |
| GR | 12,643 | 2,760 | 30.8% | 28.3% | -2.4 pp | no | 28.7% | -2.0 pp | no | 26.2% |
| HR | 12,643 | 2,760 | 28.9% | 28.3% | -0.6 pp | no | 28.3% | -0.6 pp | no | 24.7% |
| HU | 12,643 | 2,760 | 29.3% | 27.1% | -2.2 pp | no | 27.3% | -2.0 pp | no | 23.3% |
| IT | 12,896 | 2,760 | 12.3% | 11.1% | -1.2 pp | yes | 10.8% | -1.5 pp | yes | 12.1% |
| LT | 12,944 | 2,760 | 65.7% | 67.0% | +1.3 pp | yes | 70.3% | +4.7 pp | no | 69.8% |
| LV | 12,944 | 2,760 | 66.6% | 69.7% | +3.0 pp | no | 72.3% | +5.7 pp | no | 69.4% |
| NL | 12,643 | 2,760 | 48.4% | 31.9% | -16.5 pp | no | 31.4% | -17.0 pp | no | 21.1% |
| NO | 12,944 | 2,760 | 29.7% | 23.5% | -6.2 pp | no | 22.8% | -6.9 pp | no | 16.2% |
| PL | 12,944 | 2,760 | 45.9% | 30.8% | -15.1 pp | no | 30.4% | -15.5 pp | no | 26.0% |
| RO | 12,643 | 2,760 | 29.7% | 27.0% | -2.7 pp | no | 26.9% | -2.8 pp | no | 25.2% |
| SE | 12,943 | 2,760 | 62.5% | 64.0% | +1.4 pp | no | 66.1% | +3.5 pp | no | 54.2% |
| SI | 12,849 | 2,760 | 26.4% | 26.4% | -0.0 pp | no | 26.4% | -0.0 pp | no | 23.8% |
| SK | 12,640 | 2,760 | 31.5% | 27.9% | -3.7 pp | no | 27.6% | -4.0 pp | no | 23.7% |
| **pooled** | **243,983** | **52,440** | **34.3%** | **31.0%** | **-3.3 pp** | **no** | **31.4%** | **-2.9 pp** | **no** | **27.8%** |

## Country-count discrepancy

The issue asks for 21 countries, but the cited CatBoost score contains only 19: AT, BG, CH, CZ, EE, FI, GR, HR, HU, IT, LT, LV, NL, NO, PL, RO, SE, SI, SK. The replica stores price forecasts for five additional countries as `xgboost`, not `catboost`; adding them would change the model under test. The denominator is therefore 19, and the 21-country premise does not reproduce.

## Fit parameters

These parameters were estimated only on the fit window; no holdout outcome selected or changed them.

| country | bias-only intercept | affine slope | affine intercept |
|---|---:|---:|---:|
| AT | 11.596 | 0.9743 | 13.902 |
| BG | 14.277 | 0.9433 | 19.035 |
| CH | 0.943 | 1.0942 | -8.720 |
| CZ | 9.414 | 1.1450 | -3.615 |
| EE | 15.326 | 0.7512 | 27.874 |
| FI | 15.797 | 0.8393 | 21.399 |
| GR | 11.682 | 0.9574 | 15.033 |
| HR | 5.077 | 1.0030 | 4.775 |
| HU | 13.026 | 1.1229 | 0.983 |
| IT | 4.029 | 1.0587 | -3.302 |
| LT | 15.926 | 0.8404 | 26.709 |
| LV | 19.845 | 0.8652 | 28.141 |
| NL | 39.101 | 1.0694 | 35.115 |
| NO | 10.118 | 0.8270 | 21.920 |
| PL | 32.144 | 1.0759 | 26.902 |
| RO | 14.172 | 1.0575 | 8.767 |
| SE | 21.307 | 0.9117 | 24.378 |
| SI | 0.382 | 0.9608 | 4.452 |
| SK | 12.630 | 1.0418 | 8.848 |

## Data integrity and limits

- **ABL-71 touches the period operationally:** the then-undeployed price-window fix delayed fetching tomorrow's day-ahead price. It does not identify fabricated price values, and no price rows were excluded here. Actuals are latest-replica values, while the scored CatBoost forecasts are stored issued rows; first-seen price-source vintages are not archived, so source revision uplift cannot be measured.
- **ABL-67 does not touch this result:** its 216 fabricated rows are confined to `net_position`; this experiment reads `energy_price` actuals and `price` forecasts.
- **ABL-111 does not touch this result:** its zero-as-missing contamination is confined to `energy_load`; no load actual is used here.
- The issued-weather archive starts 2026-01-11. This fit starts after that date and the holdout is in July/August, so this is not one of the W01–W10 weather-blind backtests. The underlying served model can still receive zero-filled covariates after the 6-hour forward-fill limit; this correction experiment neither repairs nor reconstructs those inputs.
- This is one 30-day summer holdout, not a year-round backtest. Stored forecasts begin only on 2026-02-03 for CatBoost price.
- Forecast rows are selected exactly as ABL-129: latest vintage per country + target + model + horizon band. Thus one target can contribute once per horizon band, matching the cited 34.3% comparison.

## Recommendation to the CEO

Do not ship or promote either correction. Use seasonal-naive D−7 as the minimum model-development bar and move to a better price model/features; an affine layer has not cleared that bar out-of-sample.

No model artifact, serving registry, dashboard code, ingest code, production container, replica row, or sidecar row was changed.
