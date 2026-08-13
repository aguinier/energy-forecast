# Held-out A/B — biomass (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T13:21:58 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-02-13 .. 2026-03-14**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `biomass` has no band structure, so one all-hours row is the result.

## BE / biomass — catboost, source `energy_renewable`

n_train 18,295 · n_holdout 692 · incumbent version 20251226_155417

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 20.9 | 28.2% | 29.3 | 4.0 | 78.1 | 0 |
| control@42 | 5.8 | 7.8% | 9.2 | 2.5 | 76.5 | 0 |
| control@1337 | 5.4 | 7.3% | 8.5 | 2.1 | 76.2 | 0 |
| control@2718 | 6.0 | 8.1% | 9.3 | 3.0 | 77.1 | 0 |
| control@7 | 5.9 | 8.0% | 9.4 | 2.0 | 76.1 | 0 |
| control@13 | 5.8 | 7.8% | 9.3 | 2.1 | 76.2 | 0 |
| control@101 | 5.8 | 7.9% | 9.3 | 2.1 | 76.2 | 0 |
| control@271 | 5.8 | 7.8% | 9.3 | 2.7 | 76.8 | 0 |
| control@314 | 6.5 | 8.8% | 10.4 | 1.8 | 75.8 | 0 |
| control@577 | 5.9 | 7.9% | 9.3 | 2.2 | 76.3 | 0 |
| control@863 | 5.7 | 7.7% | 9.1 | 2.5 | 76.6 | 0 |
| control@1024 | 6.0 | 8.1% | 9.4 | 2.8 | 76.9 | 0 |
| control@1729 | 5.5 | 7.5% | 9.3 | 1.9 | 75.9 | 0 |

ABL-337 night screen: not applicable to biomass.

## FR / biomass — catboost, source `energy_renewable`

n_train 26,980 · n_holdout 692 · incumbent version 20251226_134331

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 52.9 | 11.8% | 75.8 | -30.1 | 419.4 | 0 |
| control@42 | 17.5 | 3.9% | 29.9 | -11.6 | 438.0 | 0 |
| control@1337 | 17.0 | 3.8% | 28.9 | -10.9 | 438.6 | 0 |
| control@2718 | 17.1 | 3.8% | 29.1 | -9.2 | 440.3 | 0 |
| control@7 | 17.8 | 4.0% | 29.4 | -12.5 | 437.0 | 0 |
| control@13 | 15.9 | 3.5% | 27.3 | -8.0 | 441.6 | 0 |
| control@101 | 17.1 | 3.8% | 28.4 | -9.6 | 439.9 | 0 |
| control@271 | 17.4 | 3.9% | 28.9 | -11.5 | 438.1 | 0 |
| control@314 | 16.6 | 3.7% | 28.7 | -9.7 | 439.8 | 0 |
| control@577 | 17.3 | 3.8% | 27.7 | -10.2 | 439.4 | 0 |
| control@863 | 16.3 | 3.6% | 25.5 | -8.7 | 440.8 | 0 |
| control@1024 | 17.2 | 3.8% | 27.5 | -10.2 | 439.4 | 0 |
| control@1729 | 16.6 | 3.7% | 27.7 | -10.2 | 439.4 | 0 |

ABL-337 night screen: not applicable to biomass.
