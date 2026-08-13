# Held-out A/B — wind_offshore (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T13:42:44 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-04-14 .. 2026-05-13**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `wind_offshore` has no band structure, so one all-hours row is the result.

## BE / wind_offshore — catboost, source `energy_renewable`

n_train 19,370 · n_holdout 720 · incumbent version 20251226_155415

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 600.1 | 100.2% | 800.9 | -9.6 | 589.3 | 0 |
| control@42 | 277.2 | 46.3% | 350.2 | 102.7 | 701.6 | 0 |
| control@1337 | 269.2 | 45.0% | 346.9 | 94.1 | 693.0 | 0 |
| control@2718 | 278.6 | 46.5% | 351.5 | 116.4 | 715.3 | 0 |
| control@7 | 274.6 | 45.8% | 347.3 | 118.3 | 717.2 | 0 |
| control@13 | 278.8 | 46.6% | 348.6 | 95.5 | 694.4 | 0 |
| control@101 | 269.0 | 44.9% | 343.9 | 79.6 | 678.5 | 0 |
| control@271 | 276.1 | 46.1% | 349.9 | 94.7 | 693.6 | 0 |
| control@314 | 280.5 | 46.8% | 351.7 | 97.4 | 696.3 | 0 |
| control@577 | 282.6 | 47.2% | 353.8 | 128.4 | 727.3 | 0 |
| control@863 | 272.7 | 45.5% | 347.6 | 103.3 | 702.1 | 0 |
| control@1024 | 278.5 | 46.5% | 345.1 | 104.0 | 702.9 | 0 |
| control@1729 | 269.7 | 45.0% | 340.7 | 96.2 | 695.1 | 0 |

ABL-337 night screen: not applicable to wind_offshore.

## FR / wind_offshore — catboost, source `energy_renewable`

n_train 28,716 · n_holdout 720 · incumbent version 20251226_134328

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 565.5 | 89.0% | 720.7 | -44.3 | 591.2 | 0 |
| control@42 | 311.9 | 49.1% | 396.7 | 20.1 | 655.6 | 0 |
| control@1337 | 321.6 | 50.6% | 399.2 | -5.8 | 629.7 | 0 |
| control@2718 | 323.4 | 50.9% | 399.7 | -10.0 | 625.5 | 0 |
| control@7 | 319.7 | 50.3% | 400.7 | -0.1 | 635.4 | 0 |
| control@13 | 312.7 | 49.2% | 392.8 | 8.7 | 644.2 | 0 |
| control@101 | 317.1 | 49.9% | 395.9 | -4.1 | 631.4 | 0 |
| control@271 | 307.2 | 48.3% | 391.2 | 18.7 | 654.2 | 0 |
| control@314 | 316.6 | 49.8% | 396.1 | -9.8 | 625.7 | 0 |
| control@577 | 311.7 | 49.0% | 397.2 | 8.3 | 643.8 | 0 |
| control@863 | 322.0 | 50.7% | 398.8 | -3.2 | 632.4 | 0 |
| control@1024 | 319.0 | 50.2% | 397.3 | -1.8 | 633.7 | 0 |
| control@1729 | 301.7 | 47.5% | 388.8 | 24.6 | 660.1 | 0 |

ABL-337 night screen: not applicable to wind_offshore.
