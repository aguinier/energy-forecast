# Held-out A/B — wind_offshore (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T13:52:03 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-05-14 .. 2026-06-12**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `wind_offshore` has no band structure, so one all-hours row is the result.

## BE / wind_offshore — catboost, source `energy_renewable`

n_train 20,090 · n_holdout 720 · incumbent version 20251226_155415

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 732.2 | 92.0% | 942.8 | -134.4 | 661.9 | 0 |
| control@42 | 371.0 | 46.6% | 470.9 | 142.6 | 938.9 | 0 |
| control@1337 | 374.2 | 47.0% | 472.5 | 139.3 | 935.6 | 0 |
| control@2718 | 389.3 | 48.9% | 493.6 | 181.8 | 978.1 | 0 |
| control@7 | 373.1 | 46.9% | 471.8 | 118.1 | 914.4 | 0 |
| control@13 | 390.0 | 49.0% | 493.7 | 157.0 | 953.2 | 0 |
| control@101 | 370.6 | 46.5% | 469.9 | 125.3 | 921.6 | 0 |
| control@271 | 366.7 | 46.0% | 471.8 | 128.7 | 924.9 | 0 |
| control@314 | 367.0 | 46.1% | 466.4 | 99.3 | 895.5 | 0 |
| control@577 | 358.2 | 45.0% | 465.8 | 120.7 | 917.0 | 0 |
| control@863 | 367.8 | 46.2% | 470.0 | 124.1 | 920.4 | 0 |
| control@1024 | 363.7 | 45.7% | 468.4 | 113.1 | 909.4 | 0 |
| control@1729 | 368.2 | 46.2% | 468.9 | 119.6 | 915.9 | 0 |

ABL-337 night screen: not applicable to wind_offshore.

## FR / wind_offshore — catboost, source `energy_renewable`

n_train 29,436 · n_holdout 720 · incumbent version 20251226_134328

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 626.5 | 95.3% | 803.5 | -22.0 | 635.6 | 0 |
| control@42 | 252.9 | 38.5% | 320.8 | 54.7 | 712.3 | 0 |
| control@1337 | 250.1 | 38.0% | 319.4 | 62.2 | 719.8 | 0 |
| control@2718 | 247.4 | 37.6% | 318.0 | 52.2 | 709.8 | 0 |
| control@7 | 247.5 | 37.6% | 318.6 | 47.3 | 704.9 | 0 |
| control@13 | 250.5 | 38.1% | 320.8 | 41.3 | 698.9 | 0 |
| control@101 | 249.8 | 38.0% | 321.3 | 50.3 | 707.9 | 0 |
| control@271 | 250.0 | 38.0% | 321.5 | 59.5 | 717.1 | 0 |
| control@314 | 261.8 | 39.8% | 330.4 | 58.5 | 716.0 | 0 |
| control@577 | 254.1 | 38.6% | 325.4 | 55.2 | 712.8 | 0 |
| control@863 | 252.2 | 38.4% | 323.4 | 68.6 | 726.2 | 0 |
| control@1024 | 252.9 | 38.5% | 319.8 | 60.6 | 718.2 | 0 |
| control@1729 | 250.6 | 38.1% | 317.8 | 41.8 | 699.3 | 0 |

ABL-337 night screen: not applicable to wind_offshore.
