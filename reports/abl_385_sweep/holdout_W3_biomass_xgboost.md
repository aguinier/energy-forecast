# Held-out A/B — biomass (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T13:44:20 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-04-14 .. 2026-05-13**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `biomass` has no band structure, so one all-hours row is the result.

## BE / biomass — xgboost, source `energy_renewable`

n_train 19,707 · n_holdout 720 · incumbent version 20251226_155417

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 53.0 | 32.3% | 85.7 | -34.0 | 130.2 | 0 |
| control@42 | 19.7 | 12.0% | 28.3 | -12.6 | 151.6 | 0 |
| control@1337 | 16.9 | 10.3% | 26.6 | -8.8 | 155.4 | 0 |
| control@2718 | 18.3 | 11.1% | 27.4 | -10.2 | 154.0 | 0 |
| control@7 | 18.6 | 11.3% | 27.3 | -10.6 | 153.6 | 0 |
| control@13 | 14.9 | 9.1% | 25.0 | -5.0 | 159.2 | 0 |
| control@101 | 17.9 | 10.9% | 27.1 | -9.0 | 155.2 | 0 |
| control@271 | 19.5 | 11.9% | 28.1 | -12.2 | 152.0 | 0 |
| control@314 | 20.6 | 12.5% | 28.3 | -13.6 | 150.5 | 0 |
| control@577 | 21.7 | 13.2% | 29.4 | -14.0 | 150.2 | 0 |
| control@863 | 19.8 | 12.1% | 28.1 | -12.2 | 151.9 | 0 |
| control@1024 | 23.7 | 14.4% | 31.5 | -16.7 | 147.5 | 0 |
| control@1729 | 22.0 | 13.4% | 30.8 | -14.5 | 149.6 | 0 |

ABL-337 night screen: not applicable to biomass.

## FR / biomass — xgboost, source `energy_renewable`

n_train 28,392 · n_holdout 720 · incumbent version 20251226_134331

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 41.2 | 10.1% | 71.6 | 37.9 | 445.1 | 0 |
| control@42 | 15.6 | 3.8% | 25.1 | -9.8 | 397.5 | 0 |
| control@1337 | 16.5 | 4.1% | 26.8 | -11.2 | 396.0 | 0 |
| control@2718 | 15.9 | 3.9% | 27.2 | -10.0 | 397.2 | 0 |
| control@7 | 15.7 | 3.9% | 26.7 | -10.4 | 396.8 | 0 |
| control@13 | 16.5 | 4.1% | 27.0 | -10.9 | 396.3 | 0 |
| control@101 | 14.4 | 3.5% | 24.1 | -8.4 | 398.8 | 0 |
| control@271 | 16.1 | 3.9% | 26.2 | -10.8 | 396.4 | 0 |
| control@314 | 14.7 | 3.6% | 24.0 | -9.2 | 398.0 | 0 |
| control@577 | 16.4 | 4.0% | 26.6 | -11.8 | 395.5 | 0 |
| control@863 | 15.1 | 3.7% | 25.4 | -9.9 | 397.3 | 0 |
| control@1024 | 17.3 | 4.2% | 27.7 | -12.0 | 395.3 | 0 |
| control@1729 | 15.7 | 3.8% | 27.8 | -9.5 | 397.8 | 0 |

ABL-337 night screen: not applicable to biomass.
