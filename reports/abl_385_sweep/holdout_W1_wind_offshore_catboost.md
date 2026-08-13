# Held-out A/B — wind_offshore (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T13:21:11 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-02-13 .. 2026-03-14**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `wind_offshore` has no band structure, so one all-hours row is the result.

## BE / wind_offshore — catboost, source `energy_renewable`

n_train 17,998 · n_holdout 652 · incumbent version 20251226_155415

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,028.4 | 96.2% | 1,213.2 | -45.8 | 1,022.7 | 0 |
| control@42 | 460.9 | 43.1% | 541.7 | 35.4 | 1,103.9 | 0 |
| control@1337 | 405.3 | 37.9% | 499.5 | 89.1 | 1,157.6 | 0 |
| control@2718 | 415.6 | 38.9% | 506.2 | 69.1 | 1,137.6 | 0 |
| control@7 | 541.2 | 50.7% | 607.4 | -50.1 | 1,018.4 | 0 |
| control@13 | 404.5 | 37.9% | 495.7 | 63.0 | 1,131.6 | 0 |
| control@101 | 409.5 | 38.3% | 503.3 | 90.4 | 1,158.9 | 0 |
| control@271 | 481.1 | 45.0% | 553.0 | 32.2 | 1,100.7 | 0 |
| control@314 | 400.3 | 37.5% | 494.5 | 79.3 | 1,147.8 | 0 |
| control@577 | 520.2 | 48.7% | 587.8 | -31.1 | 1,037.4 | 0 |
| control@863 | 488.1 | 45.7% | 561.0 | 3.0 | 1,071.5 | 0 |
| control@1024 | 422.2 | 39.5% | 513.3 | 70.0 | 1,138.5 | 0 |
| control@1729 | 391.2 | 36.6% | 483.7 | 85.8 | 1,154.3 | 0 |

ABL-337 night screen: not applicable to wind_offshore.

## FR / wind_offshore — catboost, source `energy_renewable`

n_train 27,304 · n_holdout 692 · incumbent version 20251226_134328

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 642.9 | 75.6% | 815.2 | 100.3 | 950.6 | 0 |
| control@42 | 321.8 | 37.8% | 420.9 | 16.5 | 866.8 | 0 |
| control@1337 | 323.8 | 38.1% | 419.3 | 16.8 | 867.1 | 0 |
| control@2718 | 327.6 | 38.5% | 424.0 | 24.7 | 875.0 | 0 |
| control@7 | 319.3 | 37.6% | 409.9 | 16.9 | 867.2 | 0 |
| control@13 | 321.8 | 37.8% | 417.3 | 27.5 | 877.8 | 0 |
| control@101 | 319.5 | 37.6% | 413.0 | 21.1 | 871.4 | 0 |
| control@271 | 328.0 | 38.6% | 421.2 | 0.5 | 850.8 | 0 |
| control@314 | 322.4 | 37.9% | 414.1 | 21.8 | 872.1 | 0 |
| control@577 | 318.2 | 37.4% | 415.7 | 9.7 | 860.0 | 0 |
| control@863 | 325.1 | 38.2% | 421.6 | 14.6 | 864.9 | 0 |
| control@1024 | 329.4 | 38.7% | 430.1 | 18.0 | 868.4 | 0 |
| control@1729 | 319.8 | 37.6% | 412.7 | 7.6 | 857.9 | 0 |

ABL-337 night screen: not applicable to wind_offshore.
