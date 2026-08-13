# Held-out A/B — wind_offshore (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T13:32:14 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-03-15 .. 2026-04-13**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `wind_offshore` has no band structure, so one all-hours row is the result.

## BE / wind_offshore — xgboost, source `energy_renewable`

n_train 18,650 · n_holdout 720 · incumbent version 20251226_155415

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,021.9 | 127.4% | 1,252.9 | 45.3 | 847.2 | 0 |
| control@42 | 398.8 | 49.7% | 488.8 | 70.9 | 872.8 | 0 |
| control@1337 | 388.0 | 48.4% | 487.4 | 43.1 | 845.0 | 0 |
| control@2718 | 381.6 | 47.6% | 486.4 | 31.2 | 833.1 | 0 |
| control@7 | 392.7 | 49.0% | 490.4 | 34.6 | 836.5 | 0 |
| control@13 | 400.6 | 50.0% | 494.8 | 53.6 | 855.5 | 0 |
| control@101 | 392.0 | 48.9% | 481.2 | 60.9 | 862.8 | 0 |
| control@271 | 385.2 | 48.0% | 485.5 | 60.9 | 862.8 | 0 |
| control@314 | 391.7 | 48.8% | 488.4 | 50.5 | 852.4 | 0 |
| control@577 | 398.3 | 49.7% | 486.4 | 49.9 | 851.9 | 0 |
| control@863 | 394.7 | 49.2% | 492.9 | 54.4 | 856.3 | 0 |
| control@1024 | 384.0 | 47.9% | 484.7 | 35.9 | 837.8 | 0 |
| control@1729 | 378.8 | 47.2% | 482.4 | 49.0 | 851.0 | 0 |

ABL-337 night screen: not applicable to wind_offshore.

## FR / wind_offshore — xgboost, source `energy_renewable`

n_train 27,996 · n_holdout 720 · incumbent version 20251226_134328

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 596.1 | 84.7% | 719.7 | 54.0 | 757.6 | 0 |
| control@42 | 334.1 | 47.5% | 414.1 | 63.6 | 767.2 | 0 |
| control@1337 | 331.4 | 47.1% | 414.6 | 80.1 | 783.6 | 0 |
| control@2718 | 339.2 | 48.2% | 417.2 | 46.0 | 749.6 | 0 |
| control@7 | 341.5 | 48.5% | 420.4 | 68.6 | 772.2 | 0 |
| control@13 | 327.6 | 46.6% | 419.8 | 102.8 | 806.4 | 0 |
| control@101 | 331.0 | 47.0% | 415.5 | 73.1 | 776.7 | 0 |
| control@271 | 338.9 | 48.2% | 423.9 | 73.8 | 777.4 | 0 |
| control@314 | 325.8 | 46.3% | 414.5 | 93.6 | 797.2 | 0 |
| control@577 | 336.5 | 47.8% | 414.2 | 58.2 | 761.7 | 0 |
| control@863 | 330.2 | 46.9% | 417.5 | 85.8 | 789.4 | 0 |
| control@1024 | 324.6 | 46.1% | 418.9 | 93.6 | 797.2 | 0 |
| control@1729 | 336.3 | 47.8% | 424.7 | 85.7 | 789.3 | 0 |

ABL-337 night screen: not applicable to wind_offshore.
