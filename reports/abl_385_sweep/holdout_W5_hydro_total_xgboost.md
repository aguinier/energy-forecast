# Held-out A/B — hydro_total (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T14:05:23 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-06-13 .. 2026-07-12**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `hydro_total` has no band structure, so one all-hours row is the result.

## BE / hydro_total — xgboost, source `energy_renewable`

n_train 21,002 · n_holdout 720 · incumbent version 20251226_155416

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 71.0 | 51.3% | 140.9 | -4.6 | 133.8 | 0 |
| control@42 | 75.9 | 54.9% | 125.4 | 2.9 | 141.3 | 0 |
| control@1337 | 78.3 | 56.6% | 132.7 | -7.2 | 131.2 | 2 |
| control@2718 | 86.6 | 62.5% | 136.5 | 6.2 | 144.6 | 0 |
| control@7 | 78.8 | 57.0% | 135.0 | -10.5 | 127.9 | 17 |
| control@13 | 79.5 | 57.4% | 131.6 | -0.1 | 138.3 | 5 |
| control@101 | 79.2 | 57.2% | 129.9 | -0.9 | 137.5 | 2 |
| control@271 | 75.7 | 54.7% | 128.6 | -3.4 | 135.1 | 0 |
| control@314 | 77.4 | 55.9% | 129.6 | -1.4 | 137.0 | 0 |
| control@577 | 83.5 | 60.3% | 134.9 | 0.1 | 138.5 | 3 |
| control@863 | 78.7 | 56.8% | 134.8 | -4.0 | 134.4 | 0 |
| control@1024 | 77.9 | 56.3% | 131.8 | -6.4 | 132.0 | 0 |
| control@1729 | 77.6 | 56.0% | 129.1 | -3.4 | 135.1 | 5 |

ABL-337 night screen: not applicable to hydro_total.

## FR / hydro_total — xgboost, source `energy_renewable`

n_train 29,742 · n_holdout 720 · incumbent version 20251226_134329

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 825.3 | 15.1% | 1,073.2 | 136.7 | 5,600.2 | 0 |
| control@42 | 502.6 | 9.2% | 667.4 | -54.6 | 5,408.9 | 0 |
| control@1337 | 500.5 | 9.2% | 658.0 | -57.2 | 5,406.3 | 0 |
| control@2718 | 485.9 | 8.9% | 635.2 | -34.9 | 5,428.6 | 0 |
| control@7 | 492.2 | 9.0% | 650.5 | -42.6 | 5,420.9 | 0 |
| control@13 | 471.5 | 8.6% | 621.1 | -38.3 | 5,425.2 | 0 |
| control@101 | 480.3 | 8.8% | 622.9 | -12.0 | 5,451.5 | 0 |
| control@271 | 491.0 | 9.0% | 650.4 | -14.7 | 5,448.9 | 0 |
| control@314 | 476.6 | 8.7% | 621.1 | -45.7 | 5,417.8 | 0 |
| control@577 | 481.8 | 8.8% | 637.6 | -54.8 | 5,408.7 | 0 |
| control@863 | 494.9 | 9.1% | 647.2 | -30.0 | 5,433.5 | 0 |
| control@1024 | 479.6 | 8.8% | 626.5 | -34.8 | 5,428.7 | 0 |
| control@1729 | 489.5 | 9.0% | 640.2 | -55.6 | 5,408.0 | 0 |

ABL-337 night screen: not applicable to hydro_total.
