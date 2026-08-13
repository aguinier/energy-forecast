# Held-out A/B — biomass (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T13:53:49 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-05-14 .. 2026-06-12**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `biomass` has no band structure, so one all-hours row is the result.

## BE / biomass — xgboost, source `energy_renewable`

n_train 20,427 · n_holdout 720 · incumbent version 20251226_155417

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 31.6 | 16.3% | 49.6 | 18.8 | 212.7 | 0 |
| control@42 | 14.7 | 7.6% | 18.4 | -8.2 | 185.7 | 0 |
| control@1337 | 16.4 | 8.5% | 19.8 | -11.0 | 182.9 | 0 |
| control@2718 | 18.7 | 9.7% | 22.6 | -13.9 | 180.0 | 0 |
| control@7 | 18.4 | 9.5% | 22.0 | -13.7 | 180.2 | 0 |
| control@13 | 15.2 | 7.8% | 18.6 | -9.8 | 184.1 | 0 |
| control@101 | 20.4 | 10.5% | 23.6 | -16.5 | 177.4 | 0 |
| control@271 | 16.4 | 8.5% | 20.2 | -11.0 | 182.9 | 0 |
| control@314 | 19.3 | 10.0% | 22.7 | -15.5 | 178.4 | 0 |
| control@577 | 19.8 | 10.2% | 23.0 | -15.9 | 178.0 | 0 |
| control@863 | 21.1 | 10.9% | 24.5 | -17.0 | 176.9 | 0 |
| control@1024 | 21.5 | 11.1% | 25.4 | -17.0 | 176.9 | 0 |
| control@1729 | 15.7 | 8.1% | 19.5 | -9.2 | 184.7 | 0 |

ABL-337 night screen: not applicable to biomass.

## FR / biomass — xgboost, source `energy_renewable`

n_train 29,112 · n_holdout 720 · incumbent version 20251226_134331

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 70.1 | 19.6% | 89.7 | 1.7 | 358.4 | 0 |
| control@42 | 15.9 | 4.5% | 26.0 | -6.4 | 350.3 | 0 |
| control@1337 | 16.2 | 4.5% | 26.8 | -7.2 | 349.5 | 0 |
| control@2718 | 16.3 | 4.6% | 27.2 | -5.5 | 351.2 | 0 |
| control@7 | 16.2 | 4.5% | 26.9 | -7.2 | 349.5 | 0 |
| control@13 | 16.0 | 4.5% | 26.9 | -7.5 | 349.2 | 0 |
| control@101 | 15.5 | 4.4% | 27.0 | -7.2 | 349.5 | 0 |
| control@271 | 15.9 | 4.5% | 26.4 | -6.7 | 350.0 | 0 |
| control@314 | 15.8 | 4.4% | 26.4 | -7.0 | 349.6 | 0 |
| control@577 | 16.7 | 4.7% | 26.7 | -7.4 | 349.3 | 0 |
| control@863 | 15.8 | 4.4% | 26.6 | -6.6 | 350.1 | 0 |
| control@1024 | 16.7 | 4.7% | 27.8 | -7.6 | 349.1 | 0 |
| control@1729 | 15.7 | 4.4% | 25.4 | -6.1 | 350.6 | 0 |

ABL-337 night screen: not applicable to biomass.
