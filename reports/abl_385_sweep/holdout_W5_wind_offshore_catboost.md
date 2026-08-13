# Held-out A/B — wind_offshore (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T14:02:06 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-06-13 .. 2026-07-12**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `wind_offshore` has no band structure, so one all-hours row is the result.

## BE / wind_offshore — catboost, source `energy_renewable`

n_train 20,810 · n_holdout 720 · incumbent version 20251226_155415

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 657.7 | 153.3% | 890.6 | 160.9 | 589.9 | 0 |
| control@42 | 328.7 | 76.6% | 456.1 | 151.6 | 580.7 | 0 |
| control@1337 | 328.1 | 76.5% | 450.6 | 148.3 | 577.4 | 0 |
| control@2718 | 330.1 | 76.9% | 451.4 | 140.4 | 569.4 | 0 |
| control@7 | 327.9 | 76.4% | 460.2 | 167.2 | 596.3 | 0 |
| control@13 | 327.8 | 76.4% | 457.8 | 154.0 | 583.1 | 0 |
| control@101 | 331.7 | 77.3% | 461.4 | 156.4 | 585.4 | 0 |
| control@271 | 328.2 | 76.5% | 453.8 | 151.2 | 580.2 | 0 |
| control@314 | 326.2 | 76.0% | 453.8 | 152.4 | 581.4 | 0 |
| control@577 | 332.9 | 77.6% | 460.7 | 163.2 | 592.3 | 0 |
| control@863 | 327.5 | 76.3% | 456.7 | 153.6 | 582.6 | 0 |
| control@1024 | 331.2 | 77.2% | 455.4 | 151.4 | 580.5 | 0 |
| control@1729 | 333.8 | 77.8% | 458.7 | 157.3 | 586.3 | 0 |

ABL-337 night screen: not applicable to wind_offshore.

## FR / wind_offshore — catboost, source `energy_renewable`

n_train 30,156 · n_holdout 720 · incumbent version 20251226_134328

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 404.5 | 106.6% | 549.7 | 48.9 | 428.4 | 0 |
| control@42 | 206.2 | 54.3% | 269.1 | 89.2 | 468.6 | 0 |
| control@1337 | 211.3 | 55.7% | 273.6 | 95.2 | 474.7 | 0 |
| control@2718 | 211.6 | 55.8% | 273.2 | 101.3 | 480.8 | 0 |
| control@7 | 203.9 | 53.7% | 263.3 | 90.1 | 469.6 | 0 |
| control@13 | 205.2 | 54.1% | 264.3 | 87.0 | 466.5 | 0 |
| control@101 | 204.7 | 54.0% | 263.8 | 83.0 | 462.4 | 0 |
| control@271 | 206.8 | 54.5% | 266.6 | 90.0 | 469.4 | 0 |
| control@314 | 208.8 | 55.0% | 268.2 | 92.0 | 471.4 | 0 |
| control@577 | 204.3 | 53.8% | 267.3 | 93.5 | 473.0 | 0 |
| control@863 | 208.5 | 54.9% | 269.2 | 88.5 | 467.9 | 0 |
| control@1024 | 208.3 | 54.9% | 268.2 | 92.5 | 472.0 | 0 |
| control@1729 | 207.5 | 54.7% | 269.2 | 86.9 | 466.4 | 0 |

ABL-337 night screen: not applicable to wind_offshore.
