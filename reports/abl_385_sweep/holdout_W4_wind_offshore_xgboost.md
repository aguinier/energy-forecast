# Held-out A/B — wind_offshore (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T13:52:49 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-05-14 .. 2026-06-12**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `wind_offshore` has no band structure, so one all-hours row is the result.

## BE / wind_offshore — xgboost, source `energy_renewable`

n_train 20,090 · n_holdout 720 · incumbent version 20251226_155415

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 732.2 | 92.0% | 942.8 | -134.4 | 661.9 | 0 |
| control@42 | 378.2 | 47.5% | 484.0 | 98.7 | 895.0 | 0 |
| control@1337 | 373.8 | 46.9% | 479.6 | 100.4 | 896.6 | 0 |
| control@2718 | 378.3 | 47.5% | 481.8 | 106.4 | 902.7 | 0 |
| control@7 | 381.4 | 47.9% | 486.5 | 122.3 | 918.6 | 0 |
| control@13 | 377.0 | 47.3% | 481.4 | 109.5 | 905.7 | 0 |
| control@101 | 379.3 | 47.6% | 486.0 | 107.2 | 903.5 | 0 |
| control@271 | 371.3 | 46.6% | 473.8 | 110.8 | 907.1 | 0 |
| control@314 | 381.0 | 47.9% | 490.8 | 124.1 | 920.3 | 0 |
| control@577 | 373.6 | 46.9% | 474.6 | 115.1 | 911.4 | 0 |
| control@863 | 385.8 | 48.4% | 486.3 | 100.4 | 896.7 | 0 |
| control@1024 | 380.7 | 47.8% | 486.6 | 109.0 | 905.3 | 0 |
| control@1729 | 389.8 | 48.9% | 489.9 | 129.9 | 926.2 | 0 |

ABL-337 night screen: not applicable to wind_offshore.

## FR / wind_offshore — xgboost, source `energy_renewable`

n_train 29,436 · n_holdout 720 · incumbent version 20251226_134328

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 626.5 | 95.3% | 803.5 | -22.0 | 635.6 | 0 |
| control@42 | 256.4 | 39.0% | 331.7 | 38.1 | 695.7 | 0 |
| control@1337 | 260.6 | 39.6% | 340.5 | 48.3 | 705.9 | 0 |
| control@2718 | 258.1 | 39.3% | 329.9 | 34.3 | 691.9 | 0 |
| control@7 | 262.7 | 39.9% | 334.5 | 52.0 | 709.6 | 0 |
| control@13 | 251.1 | 38.2% | 326.9 | 40.6 | 698.2 | 0 |
| control@101 | 255.7 | 38.9% | 334.5 | 27.4 | 685.0 | 0 |
| control@271 | 260.6 | 39.6% | 337.8 | 65.3 | 722.8 | 0 |
| control@314 | 255.2 | 38.8% | 336.5 | 41.8 | 699.3 | 0 |
| control@577 | 254.7 | 38.7% | 331.4 | 34.3 | 691.8 | 0 |
| control@863 | 255.0 | 38.8% | 331.7 | 63.7 | 721.3 | 0 |
| control@1024 | 256.6 | 39.0% | 330.5 | 29.6 | 687.2 | 0 |
| control@1729 | 258.6 | 39.3% | 333.1 | 50.8 | 708.4 | 0 |

ABL-337 night screen: not applicable to wind_offshore.
