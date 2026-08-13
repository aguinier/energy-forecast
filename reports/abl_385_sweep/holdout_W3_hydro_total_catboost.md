# Held-out A/B — hydro_total (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T13:44:37 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-04-14 .. 2026-05-13**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `hydro_total` has no band structure, so one all-hours row is the result.

## BE / hydro_total — catboost, source `energy_renewable`

n_train 19,562 · n_holdout 720 · incumbent version 20251226_155416

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 93.9 | 75.6% | 188.8 | -1.7 | 122.4 | 0 |
| control@42 | 71.2 | 57.4% | 126.0 | 3.6 | 127.8 | 0 |
| control@1337 | 70.7 | 57.0% | 124.8 | 3.2 | 127.3 | 19 |
| control@2718 | 71.1 | 57.3% | 125.1 | 3.4 | 127.5 | 1 |
| control@7 | 69.7 | 56.2% | 125.8 | 2.1 | 126.3 | 4 |
| control@13 | 71.2 | 57.4% | 126.9 | 5.7 | 129.9 | 9 |
| control@101 | 71.8 | 57.8% | 126.3 | 3.3 | 127.5 | 0 |
| control@271 | 71.7 | 57.8% | 127.6 | 0.3 | 124.4 | 0 |
| control@314 | 71.7 | 57.7% | 127.0 | 2.0 | 126.1 | 0 |
| control@577 | 71.5 | 57.6% | 127.1 | 2.8 | 126.9 | 2 |
| control@863 | 70.3 | 56.6% | 126.1 | -1.1 | 123.1 | 0 |
| control@1024 | 72.5 | 58.4% | 128.0 | 1.7 | 125.8 | 0 |
| control@1729 | 70.1 | 56.5% | 125.8 | 4.6 | 128.7 | 27 |

ABL-337 night screen: not applicable to hydro_total.

## FR / hydro_total — catboost, source `energy_renewable`

n_train 28,302 · n_holdout 720 · incumbent version 20251226_134329

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,265.6 | 17.5% | 1,609.8 | -236.7 | 6,987.0 | 0 |
| control@42 | 679.3 | 9.4% | 919.5 | -74.7 | 7,149.1 | 0 |
| control@1337 | 690.3 | 9.6% | 940.1 | -103.9 | 7,119.9 | 0 |
| control@2718 | 687.4 | 9.5% | 930.8 | -76.3 | 7,147.5 | 0 |
| control@7 | 691.6 | 9.6% | 933.9 | -69.9 | 7,153.9 | 0 |
| control@13 | 686.4 | 9.5% | 927.3 | -91.4 | 7,132.4 | 0 |
| control@101 | 681.0 | 9.4% | 929.8 | -91.3 | 7,132.5 | 0 |
| control@271 | 681.9 | 9.4% | 933.0 | -97.2 | 7,126.6 | 0 |
| control@314 | 685.1 | 9.5% | 935.7 | -55.8 | 7,167.9 | 0 |
| control@577 | 679.9 | 9.4% | 923.2 | -45.3 | 7,178.5 | 0 |
| control@863 | 673.7 | 9.3% | 913.3 | -47.0 | 7,176.7 | 0 |
| control@1024 | 685.3 | 9.5% | 938.3 | -106.4 | 7,117.4 | 0 |
| control@1729 | 688.5 | 9.5% | 936.2 | -97.9 | 7,125.8 | 0 |

ABL-337 night screen: not applicable to hydro_total.
