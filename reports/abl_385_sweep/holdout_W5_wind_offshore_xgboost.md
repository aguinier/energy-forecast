# Held-out A/B — wind_offshore (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T14:02:50 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-06-13 .. 2026-07-12**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `wind_offshore` has no band structure, so one all-hours row is the result.

## BE / wind_offshore — xgboost, source `energy_renewable`

n_train 20,810 · n_holdout 720 · incumbent version 20251226_155415

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 657.7 | 153.3% | 890.6 | 160.9 | 589.9 | 0 |
| control@42 | 338.0 | 78.8% | 485.2 | 180.2 | 609.2 | 0 |
| control@1337 | 334.6 | 78.0% | 476.4 | 170.5 | 599.5 | 0 |
| control@2718 | 326.1 | 76.0% | 468.4 | 165.6 | 594.6 | 1 |
| control@7 | 332.2 | 77.4% | 475.0 | 178.3 | 607.3 | 0 |
| control@13 | 332.7 | 77.5% | 473.2 | 180.4 | 609.4 | 0 |
| control@101 | 338.4 | 78.9% | 475.4 | 177.8 | 606.9 | 0 |
| control@271 | 335.4 | 78.2% | 481.3 | 178.8 | 607.9 | 1 |
| control@314 | 331.9 | 77.4% | 477.0 | 172.6 | 601.6 | 0 |
| control@577 | 335.0 | 78.1% | 478.3 | 181.0 | 610.1 | 0 |
| control@863 | 328.6 | 76.6% | 473.3 | 168.0 | 597.1 | 0 |
| control@1024 | 330.1 | 76.9% | 475.1 | 161.5 | 590.5 | 0 |
| control@1729 | 337.8 | 78.7% | 482.8 | 178.7 | 607.7 | 0 |

ABL-337 night screen: not applicable to wind_offshore.

## FR / wind_offshore — xgboost, source `energy_renewable`

n_train 30,156 · n_holdout 720 · incumbent version 20251226_134328

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 404.5 | 106.6% | 549.7 | 48.9 | 428.4 | 0 |
| control@42 | 217.9 | 57.4% | 282.6 | 111.7 | 491.1 | 0 |
| control@1337 | 211.6 | 55.8% | 279.9 | 106.2 | 485.6 | 0 |
| control@2718 | 216.1 | 56.9% | 282.8 | 115.2 | 494.7 | 0 |
| control@7 | 212.0 | 55.9% | 274.8 | 109.1 | 488.6 | 0 |
| control@13 | 215.9 | 56.9% | 281.3 | 107.7 | 487.2 | 0 |
| control@101 | 216.7 | 57.1% | 282.6 | 121.5 | 501.0 | 0 |
| control@271 | 216.1 | 56.9% | 281.4 | 108.8 | 488.2 | 0 |
| control@314 | 212.3 | 56.0% | 276.3 | 107.0 | 486.5 | 0 |
| control@577 | 215.5 | 56.8% | 282.4 | 112.8 | 492.3 | 0 |
| control@863 | 213.6 | 56.3% | 279.9 | 111.6 | 491.1 | 0 |
| control@1024 | 213.2 | 56.2% | 276.4 | 117.4 | 496.9 | 0 |
| control@1729 | 214.8 | 56.6% | 282.8 | 111.9 | 491.3 | 0 |

ABL-337 night screen: not applicable to wind_offshore.
