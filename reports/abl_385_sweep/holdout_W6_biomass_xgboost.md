# Held-out A/B — biomass (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T14:14:35 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-07-13 .. 2026-08-11**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `biomass` has no band structure, so one all-hours row is the result.

## BE / biomass — xgboost, source `energy_renewable`

n_train 21,867 · n_holdout 720 · incumbent version 20251226_155417

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 28.1 | 14.3% | 46.8 | 7.8 | 204.7 | 0 |
| control@42 | 11.3 | 5.8% | 15.4 | -2.1 | 194.8 | 0 |
| control@1337 | 10.9 | 5.5% | 14.6 | -2.1 | 194.8 | 0 |
| control@2718 | 11.7 | 5.9% | 15.8 | -2.5 | 194.4 | 0 |
| control@7 | 11.4 | 5.8% | 15.6 | -3.1 | 193.8 | 0 |
| control@13 | 11.1 | 5.6% | 15.0 | -3.4 | 193.5 | 0 |
| control@101 | 11.5 | 5.8% | 15.4 | -3.0 | 193.9 | 0 |
| control@271 | 11.0 | 5.6% | 15.0 | -3.0 | 194.0 | 0 |
| control@314 | 11.1 | 5.7% | 15.0 | -1.4 | 195.5 | 0 |
| control@577 | 11.4 | 5.8% | 15.2 | -2.8 | 194.1 | 0 |
| control@863 | 11.2 | 5.7% | 15.3 | -3.8 | 193.1 | 0 |
| control@1024 | 10.9 | 5.5% | 15.4 | -1.5 | 195.5 | 0 |
| control@1729 | 10.8 | 5.5% | 14.5 | -1.5 | 195.5 | 0 |

ABL-337 night screen: not applicable to biomass.

## FR / biomass — xgboost, source `energy_renewable`

n_train 30,552 · n_holdout 720 · incumbent version 20251226_134331

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 9.8 | 3.4% | 12.6 | 3.0 | 287.3 | 0 |
| control@42 | 7.4 | 2.6% | 9.1 | -1.8 | 282.5 | 0 |
| control@1337 | 8.0 | 2.8% | 10.0 | -2.8 | 281.4 | 0 |
| control@2718 | 7.5 | 2.6% | 9.1 | -1.3 | 282.9 | 0 |
| control@7 | 9.2 | 3.2% | 11.8 | -5.1 | 279.2 | 0 |
| control@13 | 8.6 | 3.0% | 10.7 | -4.1 | 280.2 | 0 |
| control@101 | 7.9 | 2.8% | 9.9 | -3.0 | 281.2 | 0 |
| control@271 | 8.4 | 2.9% | 10.5 | -1.9 | 282.4 | 0 |
| control@314 | 10.0 | 3.5% | 12.9 | -6.3 | 278.0 | 0 |
| control@577 | 8.3 | 2.9% | 10.6 | -3.4 | 280.9 | 0 |
| control@863 | 8.6 | 3.0% | 10.6 | -3.0 | 281.3 | 0 |
| control@1024 | 7.9 | 2.8% | 10.0 | -1.3 | 282.9 | 0 |
| control@1729 | 8.2 | 2.9% | 10.0 | -1.2 | 283.0 | 0 |

ABL-337 night screen: not applicable to biomass.
