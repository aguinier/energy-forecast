# Held-out A/B — biomass (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T13:43:39 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-04-14 .. 2026-05-13**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `biomass` has no band structure, so one all-hours row is the result.

## BE / biomass — catboost, source `energy_renewable`

n_train 19,707 · n_holdout 720 · incumbent version 20251226_155417

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 53.0 | 32.3% | 85.7 | -34.0 | 130.2 | 0 |
| control@42 | 15.3 | 9.3% | 23.1 | -8.1 | 156.1 | 0 |
| control@1337 | 20.3 | 12.4% | 28.0 | -14.6 | 149.6 | 0 |
| control@2718 | 19.0 | 11.6% | 26.8 | -11.9 | 152.2 | 0 |
| control@7 | 23.3 | 14.2% | 31.1 | -17.1 | 147.0 | 0 |
| control@13 | 21.5 | 13.1% | 28.8 | -15.9 | 148.3 | 0 |
| control@101 | 20.5 | 12.5% | 27.7 | -13.9 | 150.2 | 0 |
| control@271 | 20.8 | 12.7% | 28.0 | -15.2 | 149.0 | 0 |
| control@314 | 16.1 | 9.8% | 24.2 | -8.7 | 155.5 | 0 |
| control@577 | 19.9 | 12.1% | 27.5 | -13.4 | 150.8 | 0 |
| control@863 | 25.1 | 15.3% | 32.6 | -19.5 | 144.7 | 0 |
| control@1024 | 20.1 | 12.2% | 27.7 | -13.7 | 150.5 | 0 |
| control@1729 | 18.9 | 11.5% | 26.7 | -12.5 | 151.7 | 0 |

ABL-337 night screen: not applicable to biomass.

## FR / biomass — catboost, source `energy_renewable`

n_train 28,392 · n_holdout 720 · incumbent version 20251226_134331

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 41.2 | 10.1% | 71.6 | 37.9 | 445.1 | 0 |
| control@42 | 11.6 | 2.8% | 22.0 | -2.5 | 404.8 | 0 |
| control@1337 | 11.0 | 2.7% | 22.8 | 1.0 | 408.2 | 0 |
| control@2718 | 9.8 | 2.4% | 20.6 | 0.1 | 407.3 | 0 |
| control@7 | 11.1 | 2.7% | 21.5 | -2.6 | 404.6 | 0 |
| control@13 | 10.1 | 2.5% | 21.4 | 1.2 | 408.5 | 0 |
| control@101 | 10.4 | 2.6% | 21.9 | -1.8 | 405.5 | 0 |
| control@271 | 10.6 | 2.6% | 22.1 | -0.1 | 407.2 | 0 |
| control@314 | 10.4 | 2.6% | 21.6 | -0.3 | 406.9 | 0 |
| control@577 | 10.8 | 2.7% | 20.6 | -1.8 | 405.4 | 0 |
| control@863 | 10.4 | 2.6% | 22.3 | 0.1 | 407.3 | 0 |
| control@1024 | 10.7 | 2.6% | 22.1 | -0.6 | 406.6 | 0 |
| control@1729 | 10.9 | 2.7% | 21.3 | -2.4 | 404.8 | 0 |

ABL-337 night screen: not applicable to biomass.
