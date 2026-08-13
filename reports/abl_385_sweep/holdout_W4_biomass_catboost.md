# Held-out A/B — biomass (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T13:53:13 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-05-14 .. 2026-06-12**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `biomass` has no band structure, so one all-hours row is the result.

## BE / biomass — catboost, source `energy_renewable`

n_train 20,427 · n_holdout 720 · incumbent version 20251226_155417

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 31.6 | 16.3% | 49.6 | 18.8 | 212.7 | 0 |
| control@42 | 23.5 | 12.1% | 29.0 | -19.7 | 174.2 | 0 |
| control@1337 | 18.7 | 9.6% | 22.5 | -14.6 | 179.3 | 0 |
| control@2718 | 16.8 | 8.7% | 20.5 | -11.4 | 182.5 | 0 |
| control@7 | 23.2 | 12.0% | 27.0 | -20.3 | 173.6 | 0 |
| control@13 | 18.3 | 9.5% | 21.3 | -14.3 | 179.6 | 0 |
| control@101 | 18.3 | 9.4% | 21.1 | -14.0 | 179.9 | 0 |
| control@271 | 18.2 | 9.4% | 22.3 | -14.9 | 179.0 | 0 |
| control@314 | 17.1 | 8.8% | 20.3 | -11.8 | 182.1 | 0 |
| control@577 | 24.3 | 12.5% | 27.7 | -20.6 | 173.3 | 0 |
| control@863 | 20.9 | 10.8% | 25.0 | -17.6 | 176.3 | 0 |
| control@1024 | 21.7 | 11.2% | 26.9 | -16.2 | 177.7 | 0 |
| control@1729 | 36.0 | 18.6% | 39.2 | -33.0 | 160.8 | 0 |

ABL-337 night screen: not applicable to biomass.

## FR / biomass — catboost, source `energy_renewable`

n_train 29,112 · n_holdout 720 · incumbent version 20251226_134331

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 70.1 | 19.6% | 89.7 | 1.7 | 358.4 | 0 |
| control@42 | 16.1 | 4.5% | 26.0 | -2.4 | 354.3 | 0 |
| control@1337 | 14.7 | 4.1% | 25.2 | -2.8 | 353.9 | 0 |
| control@2718 | 14.7 | 4.1% | 23.7 | -0.8 | 355.9 | 0 |
| control@7 | 14.5 | 4.1% | 24.4 | -0.0 | 356.6 | 0 |
| control@13 | 13.5 | 3.8% | 24.2 | -1.8 | 354.9 | 0 |
| control@101 | 14.1 | 3.9% | 24.7 | -0.2 | 356.5 | 0 |
| control@271 | 14.3 | 4.0% | 24.0 | -1.7 | 355.0 | 0 |
| control@314 | 13.3 | 3.7% | 23.5 | -0.4 | 356.3 | 0 |
| control@577 | 13.9 | 3.9% | 25.4 | -3.0 | 353.7 | 0 |
| control@863 | 14.5 | 4.1% | 24.5 | -1.7 | 355.0 | 0 |
| control@1024 | 13.9 | 3.9% | 23.0 | -0.7 | 355.9 | 0 |
| control@1729 | 14.4 | 4.0% | 24.4 | -1.4 | 355.3 | 0 |

ABL-337 night screen: not applicable to biomass.
