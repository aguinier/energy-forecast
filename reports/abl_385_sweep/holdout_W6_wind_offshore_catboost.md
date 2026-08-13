# Held-out A/B — wind_offshore (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T14:12:36 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-07-13 .. 2026-08-11**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `wind_offshore` has no band structure, so one all-hours row is the result.

## BE / wind_offshore — catboost, source `energy_renewable`

n_train 21,530 · n_holdout 720 · incumbent version 20251226_155415

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 478.1 | 108.1% | 675.6 | 25.4 | 467.8 | 0 |
| control@42 | 248.4 | 56.2% | 324.1 | 111.3 | 553.8 | 0 |
| control@1337 | 258.4 | 58.4% | 334.6 | 132.6 | 575.0 | 0 |
| control@2718 | 247.2 | 55.9% | 329.8 | 116.0 | 558.4 | 0 |
| control@7 | 251.6 | 56.9% | 331.6 | 114.1 | 556.5 | 0 |
| control@13 | 258.9 | 58.5% | 337.5 | 129.1 | 571.6 | 0 |
| control@101 | 249.8 | 56.5% | 329.4 | 113.1 | 555.5 | 0 |
| control@271 | 258.5 | 58.4% | 340.3 | 132.1 | 574.6 | 0 |
| control@314 | 249.8 | 56.5% | 325.6 | 117.5 | 559.9 | 0 |
| control@577 | 254.4 | 57.5% | 334.1 | 123.7 | 566.1 | 0 |
| control@863 | 258.2 | 58.4% | 334.4 | 129.8 | 572.2 | 0 |
| control@1024 | 246.1 | 55.6% | 325.4 | 109.3 | 551.8 | 0 |
| control@1729 | 252.5 | 57.1% | 330.6 | 118.0 | 560.5 | 0 |

ABL-337 night screen: not applicable to wind_offshore.

## FR / wind_offshore — catboost, source `energy_renewable`

n_train 30,876 · n_holdout 720 · incumbent version 20251226_134328

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 406.4 | 75.6% | 523.9 | 25.9 | 563.1 | 0 |
| control@42 | 234.1 | 43.6% | 304.6 | 39.6 | 576.8 | 0 |
| control@1337 | 229.5 | 42.7% | 300.0 | 33.0 | 570.1 | 0 |
| control@2718 | 230.6 | 42.9% | 307.0 | 43.2 | 580.4 | 0 |
| control@7 | 237.0 | 44.1% | 306.6 | 48.2 | 585.3 | 0 |
| control@13 | 225.4 | 42.0% | 295.7 | 39.9 | 577.1 | 0 |
| control@101 | 233.2 | 43.4% | 303.7 | 46.0 | 583.1 | 0 |
| control@271 | 230.2 | 42.9% | 302.0 | 33.2 | 570.4 | 0 |
| control@314 | 233.2 | 43.4% | 307.8 | 40.6 | 577.8 | 0 |
| control@577 | 240.1 | 44.7% | 311.7 | 48.3 | 585.5 | 0 |
| control@863 | 241.6 | 45.0% | 313.0 | 46.9 | 584.1 | 0 |
| control@1024 | 235.4 | 43.8% | 311.2 | 41.8 | 579.0 | 0 |
| control@1729 | 234.8 | 43.7% | 308.1 | 42.8 | 580.0 | 0 |

ABL-337 night screen: not applicable to wind_offshore.
