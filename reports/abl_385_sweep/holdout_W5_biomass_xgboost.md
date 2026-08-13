# Held-out A/B — biomass (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T14:03:43 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-06-13 .. 2026-07-12**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `biomass` has no band structure, so one all-hours row is the result.

## BE / biomass — xgboost, source `energy_renewable`

n_train 21,147 · n_holdout 720 · incumbent version 20251226_155417

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 58.5 | 32.9% | 81.8 | -15.3 | 162.4 | 0 |
| control@42 | 13.4 | 7.6% | 20.7 | -1.3 | 176.4 | 0 |
| control@1337 | 13.0 | 7.3% | 21.3 | -1.3 | 176.4 | 0 |
| control@2718 | 13.4 | 7.5% | 21.8 | -1.5 | 176.2 | 0 |
| control@7 | 13.9 | 7.8% | 22.0 | -3.3 | 174.5 | 0 |
| control@13 | 13.3 | 7.5% | 20.7 | 0.0 | 177.8 | 0 |
| control@101 | 13.9 | 7.8% | 21.9 | 0.2 | 177.9 | 0 |
| control@271 | 15.3 | 8.6% | 22.8 | -6.0 | 171.7 | 0 |
| control@314 | 14.2 | 8.0% | 22.0 | -3.1 | 174.6 | 0 |
| control@577 | 13.2 | 7.4% | 21.6 | -1.4 | 176.3 | 0 |
| control@863 | 13.3 | 7.5% | 21.6 | -1.6 | 176.1 | 0 |
| control@1024 | 13.9 | 7.8% | 21.6 | -2.0 | 175.7 | 0 |
| control@1729 | 14.4 | 8.1% | 21.6 | -2.3 | 175.4 | 0 |

ABL-337 night screen: not applicable to biomass.

## FR / biomass — xgboost, source `energy_renewable`

n_train 29,832 · n_holdout 720 · incumbent version 20251226_134331

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 12.5 | 4.1% | 16.9 | 2.6 | 310.2 | 0 |
| control@42 | 5.1 | 1.6% | 6.9 | -0.3 | 307.3 | 0 |
| control@1337 | 6.0 | 2.0% | 8.4 | 4.7 | 312.3 | 0 |
| control@2718 | 5.2 | 1.7% | 7.0 | 1.0 | 308.5 | 0 |
| control@7 | 5.4 | 1.8% | 7.6 | -0.5 | 307.0 | 0 |
| control@13 | 5.4 | 1.8% | 7.7 | 0.2 | 307.7 | 0 |
| control@101 | 5.0 | 1.6% | 7.1 | 0.7 | 308.3 | 0 |
| control@271 | 5.6 | 1.8% | 7.7 | -1.6 | 305.9 | 0 |
| control@314 | 5.8 | 1.9% | 7.8 | -0.5 | 307.0 | 0 |
| control@577 | 5.0 | 1.6% | 7.1 | 1.4 | 309.0 | 0 |
| control@863 | 5.4 | 1.8% | 7.4 | 0.8 | 308.4 | 0 |
| control@1024 | 5.6 | 1.8% | 8.1 | 0.4 | 308.0 | 0 |
| control@1729 | 6.3 | 2.0% | 8.8 | -1.5 | 306.0 | 0 |

ABL-337 night screen: not applicable to biomass.
