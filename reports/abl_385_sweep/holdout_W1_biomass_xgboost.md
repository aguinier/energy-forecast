# Held-out A/B — biomass (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T13:22:57 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-02-13 .. 2026-03-14**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `biomass` has no band structure, so one all-hours row is the result.

## BE / biomass — xgboost, source `energy_renewable`

n_train 18,295 · n_holdout 692 · incumbent version 20251226_155417

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 20.9 | 28.2% | 29.3 | 4.0 | 78.1 | 0 |
| control@42 | 7.5 | 10.1% | 11.8 | 3.7 | 77.8 | 0 |
| control@1337 | 6.9 | 9.3% | 11.3 | 2.4 | 76.5 | 0 |
| control@2718 | 6.5 | 8.8% | 10.7 | 2.3 | 76.3 | 0 |
| control@7 | 6.3 | 8.5% | 10.3 | 2.3 | 76.4 | 0 |
| control@13 | 6.5 | 8.8% | 10.9 | 1.6 | 75.7 | 0 |
| control@101 | 7.3 | 9.8% | 11.2 | 3.1 | 77.2 | 0 |
| control@271 | 6.0 | 8.1% | 9.8 | 1.9 | 75.9 | 0 |
| control@314 | 6.2 | 8.4% | 10.2 | 1.5 | 75.6 | 0 |
| control@577 | 6.7 | 9.0% | 10.4 | 2.6 | 76.7 | 0 |
| control@863 | 6.6 | 9.0% | 11.2 | 1.6 | 75.7 | 0 |
| control@1024 | 6.8 | 9.2% | 11.4 | 2.0 | 76.0 | 0 |
| control@1729 | 6.5 | 8.8% | 10.9 | 1.7 | 75.8 | 0 |

ABL-337 night screen: not applicable to biomass.

## FR / biomass — xgboost, source `energy_renewable`

n_train 26,980 · n_holdout 692 · incumbent version 20251226_134331

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 52.9 | 11.8% | 75.8 | -30.1 | 419.4 | 0 |
| control@42 | 21.0 | 4.7% | 32.2 | -16.8 | 432.7 | 0 |
| control@1337 | 21.5 | 4.8% | 32.9 | -17.3 | 432.2 | 0 |
| control@2718 | 21.3 | 4.7% | 32.1 | -16.9 | 432.7 | 0 |
| control@7 | 22.6 | 5.0% | 33.4 | -18.7 | 430.8 | 0 |
| control@13 | 21.4 | 4.8% | 32.8 | -17.2 | 432.3 | 0 |
| control@101 | 24.2 | 5.4% | 34.5 | -20.8 | 428.7 | 0 |
| control@271 | 21.5 | 4.8% | 32.1 | -17.0 | 432.5 | 0 |
| control@314 | 20.7 | 4.6% | 32.2 | -15.5 | 434.0 | 0 |
| control@577 | 20.3 | 4.5% | 31.8 | -14.4 | 435.1 | 0 |
| control@863 | 21.9 | 4.9% | 31.6 | -17.7 | 431.9 | 0 |
| control@1024 | 21.1 | 4.7% | 33.6 | -16.5 | 433.0 | 0 |
| control@1729 | 23.9 | 5.3% | 34.3 | -20.6 | 429.0 | 0 |

ABL-337 night screen: not applicable to biomass.
