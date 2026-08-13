# Held-out A/B — hydro_total (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T13:54:06 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-05-14 .. 2026-06-12**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `hydro_total` has no band structure, so one all-hours row is the result.

## BE / hydro_total — catboost, source `energy_renewable`

n_train 20,282 · n_holdout 720 · incumbent version 20251226_155416

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 70.5 | 52.8% | 149.1 | 2.3 | 135.8 | 0 |
| control@42 | 67.3 | 50.4% | 118.8 | 0.9 | 134.3 | 0 |
| control@1337 | 66.2 | 49.6% | 117.9 | 2.3 | 135.7 | 0 |
| control@2718 | 68.6 | 51.4% | 118.4 | 6.4 | 139.8 | 3 |
| control@7 | 66.4 | 49.7% | 117.4 | 0.8 | 134.2 | 0 |
| control@13 | 66.4 | 49.8% | 118.2 | 1.5 | 134.9 | 0 |
| control@101 | 67.5 | 50.6% | 116.8 | 3.2 | 136.6 | 0 |
| control@271 | 67.5 | 50.6% | 117.7 | 4.0 | 137.5 | 0 |
| control@314 | 67.0 | 50.2% | 117.3 | 3.1 | 136.5 | 0 |
| control@577 | 66.6 | 49.9% | 117.3 | 3.6 | 137.0 | 0 |
| control@863 | 67.7 | 50.8% | 118.8 | 0.0 | 133.5 | 0 |
| control@1024 | 67.0 | 50.2% | 117.6 | 1.7 | 135.2 | 0 |
| control@1729 | 64.3 | 48.2% | 115.9 | 0.8 | 134.3 | 0 |

ABL-337 night screen: not applicable to hydro_total.

## FR / hydro_total — catboost, source `energy_renewable`

n_train 29,022 · n_holdout 720 · incumbent version 20251226_134329

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,457.0 | 21.7% | 1,842.6 | 604.0 | 7,327.4 | 0 |
| control@42 | 779.4 | 11.6% | 1,061.4 | -100.7 | 6,622.7 | 0 |
| control@1337 | 785.3 | 11.7% | 1,080.1 | -89.4 | 6,634.0 | 0 |
| control@2718 | 795.1 | 11.8% | 1,094.3 | -122.9 | 6,600.5 | 0 |
| control@7 | 778.5 | 11.6% | 1,077.0 | -80.7 | 6,642.6 | 0 |
| control@13 | 795.1 | 11.8% | 1,089.8 | -122.8 | 6,600.6 | 0 |
| control@101 | 798.7 | 11.9% | 1,094.5 | -99.1 | 6,624.3 | 0 |
| control@271 | 788.8 | 11.7% | 1,081.9 | -123.0 | 6,600.4 | 0 |
| control@314 | 780.7 | 11.6% | 1,066.3 | -115.6 | 6,607.8 | 0 |
| control@577 | 783.1 | 11.6% | 1,069.3 | -86.0 | 6,637.4 | 0 |
| control@863 | 778.4 | 11.6% | 1,064.4 | -97.4 | 6,626.0 | 0 |
| control@1024 | 780.8 | 11.6% | 1,074.0 | -57.7 | 6,665.7 | 0 |
| control@1729 | 775.8 | 11.5% | 1,073.5 | -99.2 | 6,624.1 | 0 |

ABL-337 night screen: not applicable to hydro_total.
