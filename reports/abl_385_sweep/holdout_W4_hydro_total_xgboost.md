# Held-out A/B — hydro_total (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T13:55:12 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-05-14 .. 2026-06-12**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `hydro_total` has no band structure, so one all-hours row is the result.

## BE / hydro_total — xgboost, source `energy_renewable`

n_train 20,282 · n_holdout 720 · incumbent version 20251226_155416

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 70.5 | 52.8% | 149.1 | 2.3 | 135.8 | 0 |
| control@42 | 70.2 | 52.6% | 119.9 | 6.3 | 139.7 | 0 |
| control@1337 | 69.5 | 52.1% | 120.5 | 5.3 | 138.8 | 0 |
| control@2718 | 70.1 | 52.5% | 120.0 | 7.7 | 141.2 | 0 |
| control@7 | 70.6 | 52.9% | 122.3 | 4.5 | 138.0 | 0 |
| control@13 | 69.9 | 52.4% | 120.1 | 6.6 | 140.0 | 0 |
| control@101 | 70.3 | 52.7% | 121.0 | 3.8 | 137.2 | 0 |
| control@271 | 71.1 | 53.3% | 122.3 | 5.2 | 138.7 | 0 |
| control@314 | 70.2 | 52.6% | 121.9 | 5.7 | 139.1 | 0 |
| control@577 | 70.5 | 52.8% | 122.7 | 7.0 | 140.5 | 0 |
| control@863 | 69.0 | 51.7% | 119.7 | 5.8 | 139.2 | 1 |
| control@1024 | 69.4 | 52.0% | 120.7 | 6.7 | 140.1 | 0 |
| control@1729 | 69.5 | 52.1% | 120.2 | 8.8 | 142.2 | 0 |

ABL-337 night screen: not applicable to hydro_total.

## FR / hydro_total — xgboost, source `energy_renewable`

n_train 29,022 · n_holdout 720 · incumbent version 20251226_134329

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,457.0 | 21.7% | 1,842.6 | 604.0 | 7,327.4 | 0 |
| control@42 | 824.2 | 12.3% | 1,141.7 | -76.9 | 6,646.5 | 0 |
| control@1337 | 855.6 | 12.7% | 1,159.9 | -95.8 | 6,627.6 | 0 |
| control@2718 | 832.3 | 12.4% | 1,123.2 | -61.7 | 6,661.7 | 0 |
| control@7 | 827.9 | 12.3% | 1,121.0 | -63.0 | 6,660.4 | 0 |
| control@13 | 842.3 | 12.5% | 1,140.0 | -65.6 | 6,657.8 | 0 |
| control@101 | 832.5 | 12.4% | 1,140.1 | -83.6 | 6,639.8 | 0 |
| control@271 | 844.4 | 12.6% | 1,135.4 | -51.2 | 6,672.1 | 0 |
| control@314 | 835.2 | 12.4% | 1,137.6 | -77.9 | 6,645.5 | 0 |
| control@577 | 825.5 | 12.3% | 1,114.9 | -53.0 | 6,670.4 | 0 |
| control@863 | 821.1 | 12.2% | 1,135.1 | -76.3 | 6,647.1 | 0 |
| control@1024 | 846.6 | 12.6% | 1,138.8 | -92.0 | 6,631.4 | 0 |
| control@1729 | 825.5 | 12.3% | 1,124.4 | -51.3 | 6,672.1 | 0 |

ABL-337 night screen: not applicable to hydro_total.
