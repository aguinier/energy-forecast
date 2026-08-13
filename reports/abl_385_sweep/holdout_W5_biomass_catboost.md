# Held-out A/B — biomass (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T14:03:10 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-06-13 .. 2026-07-12**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `biomass` has no band structure, so one all-hours row is the result.

## BE / biomass — catboost, source `energy_renewable`

n_train 21,147 · n_holdout 720 · incumbent version 20251226_155417

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 58.5 | 32.9% | 81.8 | -15.3 | 162.4 | 0 |
| control@42 | 11.7 | 6.6% | 18.6 | 0.1 | 177.8 | 0 |
| control@1337 | 12.2 | 6.8% | 19.5 | 0.8 | 178.6 | 0 |
| control@2718 | 12.1 | 6.8% | 19.1 | -1.8 | 175.9 | 0 |
| control@7 | 11.5 | 6.5% | 18.4 | 0.6 | 178.3 | 0 |
| control@13 | 11.7 | 6.6% | 18.3 | -0.6 | 177.1 | 0 |
| control@101 | 12.0 | 6.8% | 19.1 | 1.0 | 178.7 | 0 |
| control@271 | 13.0 | 7.3% | 19.7 | 3.6 | 181.4 | 0 |
| control@314 | 12.2 | 6.9% | 19.4 | 1.8 | 179.5 | 0 |
| control@577 | 12.5 | 7.0% | 19.4 | 0.6 | 178.3 | 0 |
| control@863 | 12.8 | 7.2% | 19.4 | 2.8 | 180.5 | 0 |
| control@1024 | 12.3 | 6.9% | 18.9 | 1.1 | 178.8 | 0 |
| control@1729 | 11.5 | 6.4% | 18.2 | -2.6 | 175.1 | 0 |

ABL-337 night screen: not applicable to biomass.

## FR / biomass — catboost, source `energy_renewable`

n_train 29,832 · n_holdout 720 · incumbent version 20251226_134331

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 12.5 | 4.1% | 16.9 | 2.6 | 310.2 | 0 |
| control@42 | 5.4 | 1.8% | 7.1 | 0.4 | 308.0 | 0 |
| control@1337 | 5.8 | 1.9% | 7.7 | 1.0 | 308.5 | 0 |
| control@2718 | 5.6 | 1.8% | 7.5 | 1.1 | 308.7 | 0 |
| control@7 | 5.5 | 1.8% | 7.3 | 0.9 | 308.4 | 0 |
| control@13 | 5.8 | 1.9% | 8.1 | 2.7 | 310.3 | 0 |
| control@101 | 4.9 | 1.6% | 6.5 | 0.4 | 308.0 | 0 |
| control@271 | 5.1 | 1.6% | 6.9 | 2.0 | 309.6 | 0 |
| control@314 | 5.7 | 1.9% | 7.9 | 2.8 | 310.4 | 0 |
| control@577 | 5.3 | 1.7% | 7.1 | 2.0 | 309.6 | 0 |
| control@863 | 5.7 | 1.8% | 7.8 | 2.2 | 309.8 | 0 |
| control@1024 | 5.3 | 1.7% | 7.2 | 2.2 | 309.8 | 0 |
| control@1729 | 5.8 | 1.9% | 7.5 | -0.0 | 307.6 | 0 |

ABL-337 night screen: not applicable to biomass.
