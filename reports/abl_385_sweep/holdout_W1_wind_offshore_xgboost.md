# Held-out A/B — wind_offshore (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T13:21:41 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-02-13 .. 2026-03-14**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `wind_offshore` has no band structure, so one all-hours row is the result.

## BE / wind_offshore — xgboost, source `energy_renewable`

n_train 17,998 · n_holdout 652 · incumbent version 20251226_155415

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,028.4 | 96.2% | 1,213.2 | -45.8 | 1,022.7 | 0 |
| control@42 | 471.6 | 44.1% | 543.7 | -17.6 | 1,050.9 | 0 |
| control@1337 | 475.6 | 44.5% | 549.2 | -56.8 | 1,011.7 | 0 |
| control@2718 | 438.4 | 41.0% | 522.1 | -15.0 | 1,053.5 | 0 |
| control@7 | 442.8 | 41.4% | 517.8 | -29.6 | 1,038.9 | 0 |
| control@13 | 486.8 | 45.6% | 560.0 | -78.6 | 989.9 | 0 |
| control@101 | 486.7 | 45.5% | 562.0 | -65.6 | 1,003.0 | 0 |
| control@271 | 481.9 | 45.1% | 559.2 | -70.2 | 998.4 | 0 |
| control@314 | 445.6 | 41.7% | 523.0 | -33.1 | 1,035.4 | 0 |
| control@577 | 439.5 | 41.1% | 517.0 | -15.9 | 1,052.6 | 0 |
| control@863 | 466.0 | 43.6% | 540.7 | -43.2 | 1,025.3 | 0 |
| control@1024 | 460.1 | 43.1% | 537.3 | -33.5 | 1,035.1 | 0 |
| control@1729 | 440.0 | 41.2% | 522.8 | -10.5 | 1,058.0 | 0 |

ABL-337 night screen: not applicable to wind_offshore.

## FR / wind_offshore — xgboost, source `energy_renewable`

n_train 27,304 · n_holdout 692 · incumbent version 20251226_134328

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 642.9 | 75.6% | 815.2 | 100.3 | 950.6 | 0 |
| control@42 | 324.9 | 38.2% | 439.3 | 5.3 | 855.6 | 0 |
| control@1337 | 328.5 | 38.6% | 445.9 | 9.4 | 859.7 | 0 |
| control@2718 | 331.0 | 38.9% | 436.4 | 28.2 | 878.6 | 0 |
| control@7 | 328.9 | 38.7% | 436.7 | 33.7 | 884.0 | 0 |
| control@13 | 339.5 | 39.9% | 451.5 | 19.3 | 869.6 | 0 |
| control@101 | 328.0 | 38.6% | 439.8 | 19.4 | 869.7 | 0 |
| control@271 | 333.8 | 39.3% | 445.9 | -2.2 | 848.1 | 0 |
| control@314 | 329.9 | 38.8% | 442.9 | 1.9 | 852.2 | 0 |
| control@577 | 327.7 | 38.5% | 432.4 | 25.4 | 875.7 | 0 |
| control@863 | 328.3 | 38.6% | 440.1 | 13.9 | 864.3 | 0 |
| control@1024 | 329.7 | 38.8% | 436.0 | 11.0 | 861.3 | 0 |
| control@1729 | 331.5 | 39.0% | 438.7 | 26.3 | 876.6 | 0 |

ABL-337 night screen: not applicable to wind_offshore.
