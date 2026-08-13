# Held-out A/B — wind_offshore (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T14:13:15 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-07-13 .. 2026-08-11**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `wind_offshore` has no band structure, so one all-hours row is the result.

## BE / wind_offshore — xgboost, source `energy_renewable`

n_train 21,530 · n_holdout 720 · incumbent version 20251226_155415

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 478.1 | 108.1% | 675.6 | 25.4 | 467.8 | 0 |
| control@42 | 290.9 | 65.7% | 352.3 | 175.3 | 617.7 | 0 |
| control@1337 | 276.1 | 62.4% | 346.9 | 171.0 | 613.5 | 0 |
| control@2718 | 301.9 | 68.2% | 366.7 | 183.9 | 626.4 | 0 |
| control@7 | 269.9 | 61.0% | 347.1 | 166.0 | 608.4 | 0 |
| control@13 | 289.3 | 65.4% | 360.5 | 188.6 | 631.1 | 0 |
| control@101 | 276.8 | 62.6% | 350.5 | 161.8 | 604.2 | 0 |
| control@271 | 292.7 | 66.2% | 362.3 | 185.2 | 627.7 | 0 |
| control@314 | 295.2 | 66.7% | 360.5 | 183.7 | 626.1 | 0 |
| control@577 | 292.1 | 66.0% | 357.4 | 179.3 | 621.7 | 0 |
| control@863 | 282.2 | 63.8% | 348.3 | 168.5 | 611.0 | 0 |
| control@1024 | 294.8 | 66.6% | 362.0 | 181.0 | 623.4 | 0 |
| control@1729 | 297.2 | 67.2% | 361.3 | 183.5 | 625.9 | 0 |

ABL-337 night screen: not applicable to wind_offshore.

## FR / wind_offshore — xgboost, source `energy_renewable`

n_train 30,876 · n_holdout 720 · incumbent version 20251226_134328

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 406.4 | 75.6% | 523.9 | 25.9 | 563.1 | 0 |
| control@42 | 247.7 | 46.1% | 322.3 | 52.6 | 589.8 | 0 |
| control@1337 | 243.0 | 45.2% | 318.7 | 43.4 | 580.6 | 0 |
| control@2718 | 248.9 | 46.3% | 322.5 | 46.8 | 584.0 | 0 |
| control@7 | 242.4 | 45.1% | 312.4 | 47.1 | 584.3 | 0 |
| control@13 | 255.5 | 47.6% | 328.1 | 42.1 | 579.3 | 0 |
| control@101 | 252.1 | 46.9% | 327.6 | 45.9 | 583.1 | 0 |
| control@271 | 254.2 | 47.3% | 328.1 | 40.5 | 577.7 | 0 |
| control@314 | 245.2 | 45.6% | 319.4 | 41.7 | 578.9 | 0 |
| control@577 | 252.9 | 47.1% | 325.1 | 48.6 | 585.8 | 0 |
| control@863 | 251.1 | 46.7% | 325.2 | 42.7 | 579.9 | 0 |
| control@1024 | 245.4 | 45.7% | 318.4 | 44.6 | 581.8 | 0 |
| control@1729 | 253.5 | 47.2% | 323.7 | 37.2 | 574.4 | 0 |

ABL-337 night screen: not applicable to wind_offshore.
