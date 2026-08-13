# Held-out A/B — hydro_total (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T13:34:44 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-03-15 .. 2026-04-13**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `hydro_total` has no band structure, so one all-hours row is the result.

## BE / hydro_total — xgboost, source `energy_renewable`

n_train 18,842 · n_holdout 720 · incumbent version 20251226_155416

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 91.1 | 59.4% | 182.4 | 1.4 | 154.7 | 0 |
| control@42 | 94.1 | 61.4% | 150.8 | -10.7 | 142.6 | 0 |
| control@1337 | 94.1 | 61.4% | 146.0 | 0.1 | 153.4 | 1 |
| control@2718 | 90.4 | 58.9% | 143.1 | 0.0 | 153.3 | 0 |
| control@7 | 89.7 | 58.5% | 142.3 | -1.8 | 151.5 | 1 |
| control@13 | 92.6 | 60.4% | 149.1 | -6.8 | 146.5 | 0 |
| control@101 | 90.0 | 58.7% | 147.7 | -9.9 | 143.4 | 0 |
| control@271 | 88.9 | 58.0% | 145.7 | -11.1 | 142.2 | 1 |
| control@314 | 99.8 | 65.1% | 153.6 | -1.7 | 151.6 | 1 |
| control@577 | 89.9 | 58.7% | 146.5 | -6.0 | 147.3 | 0 |
| control@863 | 91.7 | 59.8% | 149.9 | -8.1 | 145.2 | 0 |
| control@1024 | 94.7 | 61.8% | 150.7 | -5.4 | 147.9 | 1 |
| control@1729 | 90.9 | 59.3% | 146.6 | -6.8 | 146.5 | 0 |

ABL-337 night screen: not applicable to hydro_total.

## FR / hydro_total — xgboost, source `energy_renewable`

n_train 27,582 · n_holdout 720 · incumbent version 20251226_134329

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,848.9 | 25.3% | 2,416.8 | 592.1 | 7,910.0 | 0 |
| control@42 | 1,068.6 | 14.6% | 1,437.0 | 51.9 | 7,369.8 | 0 |
| control@1337 | 1,056.8 | 14.4% | 1,421.4 | 55.9 | 7,373.8 | 0 |
| control@2718 | 1,058.4 | 14.5% | 1,408.9 | 102.4 | 7,420.3 | 0 |
| control@7 | 1,065.2 | 14.6% | 1,427.4 | 62.5 | 7,380.4 | 0 |
| control@13 | 1,036.1 | 14.2% | 1,394.6 | 88.2 | 7,406.1 | 0 |
| control@101 | 1,048.2 | 14.3% | 1,415.7 | 42.9 | 7,360.8 | 0 |
| control@271 | 1,064.4 | 14.5% | 1,422.1 | 13.2 | 7,331.1 | 0 |
| control@314 | 1,050.2 | 14.4% | 1,410.6 | 41.9 | 7,359.8 | 0 |
| control@577 | 1,028.5 | 14.1% | 1,390.7 | 48.9 | 7,366.8 | 0 |
| control@863 | 1,052.5 | 14.4% | 1,416.5 | 96.4 | 7,414.3 | 0 |
| control@1024 | 1,048.6 | 14.3% | 1,392.0 | 40.5 | 7,358.4 | 0 |
| control@1729 | 1,067.8 | 14.6% | 1,429.0 | 76.5 | 7,394.4 | 0 |

ABL-337 night screen: not applicable to hydro_total.
