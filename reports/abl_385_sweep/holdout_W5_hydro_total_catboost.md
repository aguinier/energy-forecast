# Held-out A/B — hydro_total (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T14:04:09 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-06-13 .. 2026-07-12**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `hydro_total` has no band structure, so one all-hours row is the result.

## BE / hydro_total — catboost, source `energy_renewable`

n_train 21,002 · n_holdout 720 · incumbent version 20251226_155416

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 71.0 | 51.3% | 140.9 | -4.6 | 133.8 | 0 |
| control@42 | 73.1 | 52.8% | 120.9 | -2.3 | 136.1 | 0 |
| control@1337 | 70.2 | 50.7% | 121.5 | -5.6 | 132.8 | 4 |
| control@2718 | 72.3 | 52.2% | 123.0 | -3.8 | 134.6 | 3 |
| control@7 | 72.5 | 52.4% | 124.5 | -8.1 | 130.3 | 0 |
| control@13 | 69.7 | 50.3% | 120.7 | -4.8 | 133.6 | 12 |
| control@101 | 71.6 | 51.7% | 124.1 | -9.4 | 129.1 | 4 |
| control@271 | 72.0 | 52.0% | 123.5 | -7.8 | 130.6 | 0 |
| control@314 | 71.1 | 51.4% | 122.3 | -5.5 | 132.9 | 3 |
| control@577 | 72.3 | 52.2% | 123.9 | -5.8 | 132.6 | 23 |
| control@863 | 71.2 | 51.5% | 123.5 | -6.6 | 131.8 | 13 |
| control@1024 | 70.7 | 51.1% | 120.1 | -4.4 | 134.0 | 1 |
| control@1729 | 70.3 | 50.8% | 123.0 | -6.5 | 131.9 | 0 |

ABL-337 night screen: not applicable to hydro_total.

## FR / hydro_total — catboost, source `energy_renewable`

n_train 29,742 · n_holdout 720 · incumbent version 20251226_134329

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 825.3 | 15.1% | 1,073.2 | 136.7 | 5,600.2 | 0 |
| control@42 | 452.5 | 8.3% | 594.3 | -51.0 | 5,412.5 | 0 |
| control@1337 | 449.8 | 8.2% | 589.0 | -7.9 | 5,455.7 | 0 |
| control@2718 | 459.8 | 8.4% | 594.6 | -25.4 | 5,438.1 | 0 |
| control@7 | 453.0 | 8.3% | 599.1 | -60.4 | 5,403.2 | 0 |
| control@13 | 453.1 | 8.3% | 591.3 | -37.3 | 5,426.3 | 0 |
| control@101 | 453.3 | 8.3% | 595.0 | -34.9 | 5,428.6 | 0 |
| control@271 | 447.4 | 8.2% | 585.9 | -44.5 | 5,419.0 | 0 |
| control@314 | 449.0 | 8.2% | 587.7 | -31.6 | 5,431.9 | 0 |
| control@577 | 456.1 | 8.3% | 598.2 | -48.9 | 5,414.6 | 0 |
| control@863 | 451.2 | 8.3% | 594.3 | -49.5 | 5,414.0 | 0 |
| control@1024 | 455.5 | 8.3% | 594.5 | -35.9 | 5,427.6 | 0 |
| control@1729 | 458.4 | 8.4% | 595.6 | -29.2 | 5,434.4 | 0 |

ABL-337 night screen: not applicable to hydro_total.
