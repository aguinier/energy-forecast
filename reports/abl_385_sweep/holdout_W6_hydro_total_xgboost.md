# Held-out A/B — hydro_total (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T14:16:13 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-07-13 .. 2026-08-11**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `hydro_total` has no band structure, so one all-hours row is the result.

## BE / hydro_total — xgboost, source `energy_renewable`

n_train 21,722 · n_holdout 720 · incumbent version 20251226_155416

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 62.5 | 40.9% | 142.5 | -0.2 | 152.7 | 0 |
| control@42 | 90.3 | 59.1% | 137.1 | 23.3 | 176.2 | 9 |
| control@1337 | 83.1 | 54.4% | 138.2 | -1.5 | 151.3 | 0 |
| control@2718 | 86.3 | 56.5% | 132.7 | 14.8 | 167.6 | 12 |
| control@7 | 85.6 | 56.0% | 135.9 | 8.3 | 161.2 | 0 |
| control@13 | 81.1 | 53.1% | 134.3 | 6.3 | 159.1 | 60 |
| control@101 | 85.8 | 56.1% | 134.1 | 15.3 | 168.1 | 4 |
| control@271 | 79.7 | 52.2% | 128.7 | 6.6 | 159.4 | 0 |
| control@314 | 88.6 | 58.0% | 139.4 | 16.5 | 169.3 | 1 |
| control@577 | 84.6 | 55.3% | 137.2 | 5.1 | 158.0 | 3 |
| control@863 | 81.1 | 53.0% | 129.8 | 9.0 | 161.9 | 0 |
| control@1024 | 82.2 | 53.8% | 132.5 | 5.1 | 157.9 | 2 |
| control@1729 | 79.8 | 52.2% | 130.2 | 7.9 | 160.7 | 5 |

ABL-337 night screen: not applicable to hydro_total.

## FR / hydro_total — xgboost, source `energy_renewable`

n_train 30,462 · n_holdout 720 · incumbent version 20251226_134329

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 741.4 | 16.8% | 970.1 | 161.2 | 4,575.9 | 0 |
| control@42 | 455.3 | 10.3% | 576.8 | 68.6 | 4,483.3 | 0 |
| control@1337 | 442.7 | 10.0% | 566.5 | 87.5 | 4,502.2 | 0 |
| control@2718 | 454.5 | 10.3% | 589.5 | 131.6 | 4,546.3 | 0 |
| control@7 | 446.4 | 10.1% | 576.9 | 97.3 | 4,512.0 | 0 |
| control@13 | 446.8 | 10.1% | 572.3 | 89.2 | 4,503.9 | 0 |
| control@101 | 453.7 | 10.3% | 575.6 | 78.5 | 4,493.2 | 0 |
| control@271 | 455.7 | 10.3% | 581.9 | 92.1 | 4,506.8 | 0 |
| control@314 | 458.2 | 10.4% | 583.2 | 109.2 | 4,523.9 | 0 |
| control@577 | 463.5 | 10.5% | 592.2 | 89.6 | 4,504.3 | 0 |
| control@863 | 446.7 | 10.1% | 571.0 | 92.8 | 4,507.5 | 0 |
| control@1024 | 461.3 | 10.4% | 585.0 | 92.6 | 4,507.3 | 0 |
| control@1729 | 449.7 | 10.2% | 575.7 | 84.3 | 4,499.0 | 0 |

ABL-337 night screen: not applicable to hydro_total.
