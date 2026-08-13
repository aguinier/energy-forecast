# Held-out A/B — hydro_total (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T14:14:59 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-07-13 .. 2026-08-11**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `hydro_total` has no band structure, so one all-hours row is the result.

## BE / hydro_total — catboost, source `energy_renewable`

n_train 21,722 · n_holdout 720 · incumbent version 20251226_155416

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 62.5 | 40.9% | 142.5 | -0.2 | 152.7 | 0 |
| control@42 | 72.6 | 47.5% | 123.9 | -4.5 | 148.3 | 30 |
| control@1337 | 74.8 | 49.0% | 125.4 | 5.9 | 158.7 | 30 |
| control@2718 | 72.8 | 47.6% | 124.7 | -7.4 | 145.4 | 2 |
| control@7 | 77.3 | 50.6% | 128.1 | -4.8 | 148.1 | 5 |
| control@13 | 75.0 | 49.1% | 127.2 | -1.0 | 151.8 | 30 |
| control@101 | 75.6 | 49.5% | 125.9 | 1.0 | 153.8 | 20 |
| control@271 | 74.5 | 48.8% | 125.5 | -1.8 | 151.0 | 38 |
| control@314 | 77.1 | 50.4% | 128.3 | -1.1 | 151.7 | 7 |
| control@577 | 76.8 | 50.2% | 125.8 | 1.9 | 154.7 | 0 |
| control@863 | 85.3 | 55.8% | 139.7 | -10.8 | 142.1 | 0 |
| control@1024 | 82.7 | 54.1% | 134.5 | 3.0 | 155.8 | 11 |
| control@1729 | 76.9 | 50.3% | 127.7 | -3.0 | 149.9 | 0 |

ABL-337 night screen: not applicable to hydro_total.

## FR / hydro_total — catboost, source `energy_renewable`

n_train 30,462 · n_holdout 720 · incumbent version 20251226_134329

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 741.4 | 16.8% | 970.1 | 161.2 | 4,575.9 | 0 |
| control@42 | 479.3 | 10.9% | 601.3 | 116.8 | 4,531.5 | 0 |
| control@1337 | 465.6 | 10.5% | 583.3 | 114.8 | 4,529.5 | 0 |
| control@2718 | 465.4 | 10.5% | 585.5 | 95.8 | 4,510.6 | 0 |
| control@7 | 465.1 | 10.5% | 586.5 | 116.9 | 4,531.6 | 0 |
| control@13 | 468.0 | 10.6% | 585.6 | 110.4 | 4,525.1 | 0 |
| control@101 | 468.9 | 10.6% | 588.5 | 110.9 | 4,525.6 | 0 |
| control@271 | 461.8 | 10.5% | 578.4 | 101.9 | 4,516.6 | 0 |
| control@314 | 477.2 | 10.8% | 592.4 | 157.4 | 4,572.1 | 0 |
| control@577 | 474.2 | 10.7% | 592.6 | 101.9 | 4,516.6 | 0 |
| control@863 | 466.7 | 10.6% | 589.1 | 78.0 | 4,492.7 | 0 |
| control@1024 | 467.9 | 10.6% | 586.9 | 115.3 | 4,530.0 | 0 |
| control@1729 | 474.3 | 10.7% | 596.3 | 81.1 | 4,495.8 | 0 |

ABL-337 night screen: not applicable to hydro_total.
