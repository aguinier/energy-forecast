# Held-out A/B — biomass (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T14:13:32 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-07-13 .. 2026-08-11**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `biomass` has no band structure, so one all-hours row is the result.

## BE / biomass — catboost, source `energy_renewable`

n_train 21,867 · n_holdout 720 · incumbent version 20251226_155417

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 28.1 | 14.3% | 46.8 | 7.8 | 204.7 | 0 |
| control@42 | 10.6 | 5.4% | 13.8 | 1.6 | 198.5 | 0 |
| control@1337 | 9.7 | 4.9% | 13.5 | 1.2 | 198.1 | 0 |
| control@2718 | 9.8 | 5.0% | 13.1 | -1.4 | 195.5 | 0 |
| control@7 | 9.6 | 4.9% | 13.2 | -1.0 | 196.0 | 0 |
| control@13 | 9.7 | 4.9% | 13.1 | -1.6 | 195.3 | 0 |
| control@101 | 11.2 | 5.7% | 14.9 | -3.4 | 193.5 | 0 |
| control@271 | 10.6 | 5.4% | 14.0 | 3.1 | 200.0 | 0 |
| control@314 | 10.7 | 5.4% | 14.4 | -2.5 | 194.4 | 0 |
| control@577 | 10.1 | 5.1% | 13.4 | 0.8 | 197.8 | 0 |
| control@863 | 10.8 | 5.5% | 14.2 | -2.9 | 194.0 | 0 |
| control@1024 | 9.5 | 4.8% | 13.0 | -1.4 | 195.5 | 0 |
| control@1729 | 11.1 | 5.7% | 14.7 | 3.4 | 200.3 | 0 |

ABL-337 night screen: not applicable to biomass.

## FR / biomass — catboost, source `energy_renewable`

n_train 30,552 · n_holdout 720 · incumbent version 20251226_134331

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 9.8 | 3.4% | 12.6 | 3.0 | 287.3 | 0 |
| control@42 | 7.4 | 2.6% | 10.2 | 6.5 | 290.8 | 0 |
| control@1337 | 6.7 | 2.4% | 8.6 | 3.4 | 287.7 | 0 |
| control@2718 | 6.2 | 2.2% | 8.1 | 3.7 | 288.0 | 0 |
| control@7 | 6.3 | 2.2% | 7.2 | -0.6 | 283.7 | 0 |
| control@13 | 6.0 | 2.1% | 7.1 | 0.4 | 284.7 | 0 |
| control@101 | 7.2 | 2.5% | 7.8 | -1.2 | 283.1 | 0 |
| control@271 | 6.6 | 2.3% | 7.3 | -1.6 | 282.6 | 0 |
| control@314 | 6.2 | 2.2% | 7.9 | 3.4 | 287.6 | 0 |
| control@577 | 6.6 | 2.3% | 7.6 | -0.4 | 283.8 | 0 |
| control@863 | 7.7 | 2.7% | 9.1 | -3.6 | 280.6 | 0 |
| control@1024 | 6.7 | 2.3% | 7.7 | 0.5 | 284.8 | 0 |
| control@1729 | 6.3 | 2.2% | 7.2 | -1.7 | 282.5 | 0 |

ABL-337 night screen: not applicable to biomass.
