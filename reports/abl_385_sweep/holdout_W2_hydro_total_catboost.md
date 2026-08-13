# Held-out A/B — hydro_total (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T13:33:34 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-03-15 .. 2026-04-13**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `hydro_total` has no band structure, so one all-hours row is the result.

## BE / hydro_total — catboost, source `energy_renewable`

n_train 18,842 · n_holdout 720 · incumbent version 20251226_155416

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 91.1 | 59.4% | 182.4 | 1.4 | 154.7 | 0 |
| control@42 | 86.4 | 56.4% | 145.5 | -14.3 | 139.0 | 4 |
| control@1337 | 90.1 | 58.8% | 148.6 | -11.9 | 141.4 | 0 |
| control@2718 | 85.8 | 55.9% | 145.0 | -11.8 | 141.5 | 0 |
| control@7 | 89.4 | 58.3% | 146.3 | -9.7 | 143.7 | 0 |
| control@13 | 86.9 | 56.7% | 144.0 | -8.1 | 145.2 | 0 |
| control@101 | 88.5 | 57.7% | 144.5 | -6.2 | 147.1 | 0 |
| control@271 | 87.9 | 57.4% | 142.0 | -3.2 | 150.1 | 0 |
| control@314 | 86.4 | 56.3% | 141.7 | -7.4 | 145.9 | 1 |
| control@577 | 88.5 | 57.7% | 144.4 | -6.4 | 146.9 | 0 |
| control@863 | 88.5 | 57.8% | 143.0 | -3.4 | 149.9 | 0 |
| control@1024 | 87.4 | 57.0% | 143.4 | -4.6 | 148.7 | 0 |
| control@1729 | 89.3 | 58.2% | 145.7 | -9.0 | 144.3 | 0 |

ABL-337 night screen: not applicable to hydro_total.

## FR / hydro_total — catboost, source `energy_renewable`

n_train 27,582 · n_holdout 720 · incumbent version 20251226_134329

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,848.9 | 25.3% | 2,416.8 | 592.1 | 7,910.0 | 0 |
| control@42 | 1,069.1 | 14.6% | 1,418.9 | 112.4 | 7,430.3 | 0 |
| control@1337 | 1,028.8 | 14.1% | 1,386.5 | 104.0 | 7,421.9 | 0 |
| control@2718 | 1,067.5 | 14.6% | 1,423.0 | 88.0 | 7,405.9 | 0 |
| control@7 | 1,061.3 | 14.5% | 1,426.8 | 107.8 | 7,425.7 | 0 |
| control@13 | 1,042.7 | 14.2% | 1,395.0 | 85.6 | 7,403.4 | 0 |
| control@101 | 1,059.1 | 14.5% | 1,413.9 | 80.7 | 7,398.6 | 0 |
| control@271 | 1,065.1 | 14.6% | 1,422.4 | 97.6 | 7,415.5 | 0 |
| control@314 | 1,048.0 | 14.3% | 1,404.7 | 111.2 | 7,429.1 | 0 |
| control@577 | 1,051.2 | 14.4% | 1,408.1 | 89.9 | 7,407.8 | 0 |
| control@863 | 1,026.8 | 14.0% | 1,375.7 | 67.0 | 7,384.9 | 0 |
| control@1024 | 1,080.7 | 14.8% | 1,432.2 | 86.3 | 7,404.2 | 0 |
| control@1729 | 1,033.1 | 14.1% | 1,383.0 | 85.3 | 7,403.2 | 0 |

ABL-337 night screen: not applicable to hydro_total.
