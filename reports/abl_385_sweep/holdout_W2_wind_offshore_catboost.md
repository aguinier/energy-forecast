# Held-out A/B — wind_offshore (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T13:31:46 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-03-15 .. 2026-04-13**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `wind_offshore` has no band structure, so one all-hours row is the result.

## BE / wind_offshore — catboost, source `energy_renewable`

n_train 18,650 · n_holdout 720 · incumbent version 20251226_155415

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,021.9 | 127.4% | 1,252.9 | 45.3 | 847.2 | 0 |
| control@42 | 376.6 | 47.0% | 477.1 | 84.3 | 886.2 | 0 |
| control@1337 | 386.8 | 48.2% | 482.8 | 71.3 | 873.2 | 0 |
| control@2718 | 391.5 | 48.8% | 493.5 | 72.2 | 874.1 | 0 |
| control@7 | 386.8 | 48.2% | 483.1 | 83.0 | 884.9 | 0 |
| control@13 | 384.1 | 47.9% | 482.8 | 95.5 | 897.4 | 0 |
| control@101 | 398.6 | 49.7% | 490.9 | 124.7 | 926.6 | 0 |
| control@271 | 396.2 | 49.4% | 492.3 | 93.6 | 895.5 | 0 |
| control@314 | 391.8 | 48.9% | 482.4 | 101.6 | 903.5 | 0 |
| control@577 | 383.7 | 47.8% | 481.5 | 75.1 | 877.0 | 0 |
| control@863 | 396.0 | 49.4% | 489.6 | 94.1 | 896.0 | 0 |
| control@1024 | 389.6 | 48.6% | 487.4 | 80.6 | 882.5 | 0 |
| control@1729 | 386.7 | 48.2% | 482.6 | 81.3 | 883.2 | 0 |

ABL-337 night screen: not applicable to wind_offshore.

## FR / wind_offshore — catboost, source `energy_renewable`

n_train 27,996 · n_holdout 720 · incumbent version 20251226_134328

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 596.1 | 84.7% | 719.7 | 54.0 | 757.6 | 0 |
| control@42 | 323.1 | 45.9% | 399.8 | 76.8 | 780.4 | 0 |
| control@1337 | 326.1 | 46.3% | 405.1 | 81.4 | 785.0 | 0 |
| control@2718 | 327.8 | 46.6% | 402.3 | 78.3 | 781.9 | 0 |
| control@7 | 323.8 | 46.0% | 405.9 | 88.4 | 792.0 | 0 |
| control@13 | 328.2 | 46.6% | 405.8 | 76.4 | 779.9 | 0 |
| control@101 | 324.9 | 46.2% | 403.4 | 82.2 | 785.8 | 0 |
| control@271 | 324.7 | 46.2% | 402.9 | 78.2 | 781.8 | 0 |
| control@314 | 326.0 | 46.3% | 402.4 | 82.7 | 786.2 | 0 |
| control@577 | 322.0 | 45.8% | 402.2 | 87.0 | 790.6 | 0 |
| control@863 | 327.0 | 46.5% | 404.5 | 80.0 | 783.6 | 0 |
| control@1024 | 318.5 | 45.3% | 396.8 | 71.7 | 775.3 | 0 |
| control@1729 | 325.1 | 46.2% | 402.8 | 83.5 | 787.1 | 0 |

ABL-337 night screen: not applicable to wind_offshore.
