# Held-out A/B — wind_offshore (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T13:43:22 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-04-14 .. 2026-05-13**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `wind_offshore` has no band structure, so one all-hours row is the result.

## BE / wind_offshore — xgboost, source `energy_renewable`

n_train 19,370 · n_holdout 720 · incumbent version 20251226_155415

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 600.1 | 100.2% | 800.9 | -9.6 | 589.3 | 0 |
| control@42 | 267.9 | 44.7% | 348.8 | 75.5 | 674.4 | 0 |
| control@1337 | 265.1 | 44.3% | 334.4 | 74.8 | 673.7 | 0 |
| control@2718 | 267.7 | 44.7% | 346.5 | 70.8 | 669.6 | 0 |
| control@7 | 276.3 | 46.1% | 353.1 | 91.7 | 690.5 | 0 |
| control@13 | 253.1 | 42.3% | 328.3 | 65.6 | 664.5 | 0 |
| control@101 | 263.2 | 43.9% | 338.5 | 73.8 | 672.7 | 0 |
| control@271 | 263.6 | 44.0% | 336.8 | 68.1 | 667.0 | 0 |
| control@314 | 261.9 | 43.7% | 339.4 | 81.3 | 680.2 | 0 |
| control@577 | 265.8 | 44.4% | 344.9 | 86.7 | 685.6 | 0 |
| control@863 | 267.8 | 44.7% | 343.7 | 76.3 | 675.2 | 0 |
| control@1024 | 254.2 | 42.4% | 325.8 | 72.3 | 671.2 | 0 |
| control@1729 | 267.1 | 44.6% | 342.2 | 82.0 | 680.9 | 0 |

ABL-337 night screen: not applicable to wind_offshore.

## FR / wind_offshore — xgboost, source `energy_renewable`

n_train 28,716 · n_holdout 720 · incumbent version 20251226_134328

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 565.5 | 89.0% | 720.7 | -44.3 | 591.2 | 0 |
| control@42 | 314.3 | 49.5% | 390.5 | 3.8 | 639.3 | 0 |
| control@1337 | 320.6 | 50.5% | 402.4 | -13.6 | 621.9 | 0 |
| control@2718 | 308.2 | 48.5% | 386.5 | 23.0 | 658.5 | 0 |
| control@7 | 316.3 | 49.8% | 396.2 | 13.3 | 648.8 | 0 |
| control@13 | 315.1 | 49.6% | 397.7 | -1.3 | 634.2 | 0 |
| control@101 | 319.0 | 50.2% | 402.5 | -16.2 | 619.3 | 0 |
| control@271 | 309.7 | 48.7% | 384.8 | 7.1 | 642.6 | 0 |
| control@314 | 321.4 | 50.6% | 405.3 | -14.8 | 620.7 | 0 |
| control@577 | 318.1 | 50.1% | 394.6 | -1.3 | 634.2 | 0 |
| control@863 | 325.6 | 51.2% | 411.8 | -16.6 | 618.9 | 0 |
| control@1024 | 317.4 | 49.9% | 401.1 | -10.1 | 625.4 | 0 |
| control@1729 | 314.9 | 49.6% | 399.3 | 9.9 | 645.5 | 0 |

ABL-337 night screen: not applicable to wind_offshore.
