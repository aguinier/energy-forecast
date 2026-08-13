# Held-out A/B — biomass (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T13:32:30 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-03-15 .. 2026-04-13**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `biomass` has no band structure, so one all-hours row is the result.

## BE / biomass — catboost, source `energy_renewable`

n_train 18,987 · n_holdout 720 · incumbent version 20251226_155417

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 11.1 | 17.5% | 14.4 | 0.2 | 63.7 | 0 |
| control@42 | 4.1 | 6.5% | 5.3 | 2.0 | 65.5 | 0 |
| control@1337 | 4.2 | 6.6% | 5.4 | 2.2 | 65.7 | 0 |
| control@2718 | 4.0 | 6.2% | 5.2 | 1.9 | 65.3 | 0 |
| control@7 | 4.0 | 6.3% | 5.1 | 1.8 | 65.2 | 0 |
| control@13 | 4.1 | 6.4% | 5.3 | 2.1 | 65.6 | 0 |
| control@101 | 4.3 | 6.8% | 5.7 | 2.7 | 66.1 | 0 |
| control@271 | 4.5 | 7.1% | 5.9 | 2.9 | 66.3 | 0 |
| control@314 | 4.1 | 6.4% | 5.4 | 2.0 | 65.5 | 0 |
| control@577 | 4.3 | 6.7% | 5.6 | 2.9 | 66.3 | 0 |
| control@863 | 4.3 | 6.7% | 5.5 | 2.1 | 65.6 | 0 |
| control@1024 | 4.9 | 7.7% | 6.4 | 3.4 | 66.9 | 0 |
| control@1729 | 4.2 | 6.7% | 5.5 | 2.4 | 65.9 | 0 |

ABL-337 night screen: not applicable to biomass.

## FR / biomass — catboost, source `energy_renewable`

n_train 27,672 · n_holdout 720 · incumbent version 20251226_134331

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 14.5 | 3.1% | 28.7 | -5.3 | 455.0 | 0 |
| control@42 | 15.0 | 3.3% | 23.8 | -11.9 | 448.5 | 0 |
| control@1337 | 16.7 | 3.6% | 24.3 | -14.1 | 446.3 | 0 |
| control@2718 | 17.6 | 3.8% | 25.8 | -15.0 | 445.4 | 0 |
| control@7 | 17.6 | 3.8% | 25.5 | -15.2 | 445.2 | 0 |
| control@13 | 13.9 | 3.0% | 23.0 | -10.4 | 450.0 | 0 |
| control@101 | 15.0 | 3.3% | 23.2 | -11.8 | 448.6 | 0 |
| control@271 | 14.6 | 3.2% | 23.0 | -10.9 | 449.5 | 0 |
| control@314 | 15.0 | 3.3% | 23.3 | -11.9 | 448.4 | 0 |
| control@577 | 18.3 | 4.0% | 24.7 | -15.8 | 444.6 | 0 |
| control@863 | 16.5 | 3.6% | 24.5 | -13.0 | 447.3 | 0 |
| control@1024 | 12.1 | 2.6% | 22.4 | -7.2 | 453.1 | 0 |
| control@1729 | 16.6 | 3.6% | 23.3 | -13.6 | 446.8 | 0 |

ABL-337 night screen: not applicable to biomass.
