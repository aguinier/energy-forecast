# Held-out A/B — biomass (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T13:33:13 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-03-15 .. 2026-04-13**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `biomass` has no band structure, so one all-hours row is the result.

## BE / biomass — xgboost, source `energy_renewable`

n_train 18,987 · n_holdout 720 · incumbent version 20251226_155417

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 11.1 | 17.5% | 14.4 | 0.2 | 63.7 | 0 |
| control@42 | 4.4 | 6.9% | 5.7 | 2.7 | 66.1 | 0 |
| control@1337 | 4.6 | 7.2% | 6.1 | 3.2 | 66.6 | 0 |
| control@2718 | 4.4 | 6.9% | 5.8 | 2.5 | 65.9 | 0 |
| control@7 | 4.5 | 7.1% | 5.9 | 2.9 | 66.3 | 0 |
| control@13 | 4.5 | 7.1% | 5.9 | 3.0 | 66.4 | 0 |
| control@101 | 4.2 | 6.6% | 5.6 | 2.3 | 65.7 | 0 |
| control@271 | 4.0 | 6.3% | 5.3 | 1.9 | 65.4 | 0 |
| control@314 | 4.0 | 6.4% | 5.3 | 2.2 | 65.6 | 0 |
| control@577 | 4.9 | 7.8% | 6.4 | 3.9 | 67.3 | 0 |
| control@863 | 4.4 | 6.9% | 5.7 | 2.8 | 66.3 | 0 |
| control@1024 | 4.2 | 6.7% | 5.6 | 2.6 | 66.0 | 0 |
| control@1729 | 4.7 | 7.5% | 6.4 | 3.1 | 66.5 | 0 |

ABL-337 night screen: not applicable to biomass.

## FR / biomass — xgboost, source `energy_renewable`

n_train 27,672 · n_holdout 720 · incumbent version 20251226_134331

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 14.5 | 3.1% | 28.7 | -5.3 | 455.0 | 0 |
| control@42 | 23.0 | 5.0% | 33.1 | -21.5 | 438.9 | 0 |
| control@1337 | 23.9 | 5.2% | 33.1 | -22.5 | 437.9 | 0 |
| control@2718 | 25.7 | 5.6% | 32.8 | -24.3 | 436.1 | 0 |
| control@7 | 22.8 | 4.9% | 32.8 | -21.3 | 439.1 | 0 |
| control@13 | 22.7 | 4.9% | 33.0 | -21.1 | 439.3 | 0 |
| control@101 | 21.1 | 4.6% | 31.6 | -19.1 | 441.3 | 0 |
| control@271 | 23.2 | 5.0% | 32.6 | -21.7 | 438.7 | 0 |
| control@314 | 23.3 | 5.1% | 32.1 | -21.7 | 438.7 | 0 |
| control@577 | 22.4 | 4.9% | 31.4 | -20.6 | 439.8 | 0 |
| control@863 | 21.0 | 4.6% | 31.1 | -19.2 | 441.1 | 0 |
| control@1024 | 22.3 | 4.8% | 31.5 | -20.4 | 440.0 | 0 |
| control@1729 | 22.1 | 4.8% | 31.6 | -20.6 | 439.8 | 0 |

ABL-337 night screen: not applicable to biomass.
