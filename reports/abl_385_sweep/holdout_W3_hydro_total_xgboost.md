# Held-out A/B — hydro_total (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T13:45:56 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-04-14 .. 2026-05-13**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `hydro_total` has no band structure, so one all-hours row is the result.

## BE / hydro_total — xgboost, source `energy_renewable`

n_train 19,562 · n_holdout 720 · incumbent version 20251226_155416

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 93.9 | 75.6% | 188.8 | -1.7 | 122.4 | 0 |
| control@42 | 72.7 | 58.5% | 128.4 | 3.8 | 127.9 | 0 |
| control@1337 | 72.4 | 58.3% | 128.6 | 3.2 | 127.3 | 4 |
| control@2718 | 73.3 | 59.0% | 128.6 | 3.1 | 127.2 | 1 |
| control@7 | 73.1 | 58.9% | 128.5 | 3.1 | 127.3 | 0 |
| control@13 | 72.6 | 58.5% | 127.0 | 6.2 | 130.3 | 5 |
| control@101 | 73.3 | 59.1% | 128.5 | 5.4 | 129.6 | 0 |
| control@271 | 73.3 | 59.1% | 128.4 | 4.0 | 128.1 | 1 |
| control@314 | 73.2 | 59.0% | 128.6 | 3.1 | 127.2 | 0 |
| control@577 | 72.1 | 58.1% | 127.1 | 5.6 | 129.7 | 0 |
| control@863 | 73.6 | 59.3% | 128.7 | 3.3 | 127.5 | 0 |
| control@1024 | 73.3 | 59.0% | 127.9 | 3.6 | 127.7 | 0 |
| control@1729 | 72.5 | 58.4% | 128.0 | 3.1 | 127.3 | 0 |

ABL-337 night screen: not applicable to hydro_total.

## FR / hydro_total — xgboost, source `energy_renewable`

n_train 28,302 · n_holdout 720 · incumbent version 20251226_134329

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,265.6 | 17.5% | 1,609.8 | -236.7 | 6,987.0 | 0 |
| control@42 | 723.2 | 10.0% | 978.5 | -85.0 | 7,138.8 | 0 |
| control@1337 | 712.5 | 9.9% | 966.3 | -66.8 | 7,156.9 | 0 |
| control@2718 | 705.8 | 9.8% | 962.4 | -95.4 | 7,128.3 | 0 |
| control@7 | 726.4 | 10.1% | 969.7 | -91.9 | 7,131.8 | 0 |
| control@13 | 712.2 | 9.9% | 966.4 | -93.0 | 7,130.8 | 0 |
| control@101 | 710.8 | 9.8% | 973.0 | -101.4 | 7,122.4 | 0 |
| control@271 | 700.5 | 9.7% | 957.9 | -99.5 | 7,124.3 | 0 |
| control@314 | 718.9 | 10.0% | 961.8 | -82.0 | 7,141.8 | 0 |
| control@577 | 726.4 | 10.1% | 976.8 | -52.5 | 7,171.2 | 0 |
| control@863 | 722.9 | 10.0% | 996.5 | -94.2 | 7,129.6 | 0 |
| control@1024 | 728.3 | 10.1% | 985.4 | -106.5 | 7,117.3 | 0 |
| control@1729 | 713.6 | 9.9% | 981.8 | -117.3 | 7,106.5 | 0 |

ABL-337 night screen: not applicable to hydro_total.
