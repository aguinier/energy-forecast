# Held-out A/B — hydro_total (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T13:23:15 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-02-13 .. 2026-03-14**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `hydro_total` has no band structure, so one all-hours row is the result.

## BE / hydro_total — catboost, source `energy_renewable`

n_train 18,150 · n_holdout 692 · incumbent version 20251226_155416

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 95.7 | 73.9% | 189.1 | -18.7 | 110.7 | 0 |
| control@42 | 80.0 | 61.8% | 124.4 | -4.8 | 124.6 | 0 |
| control@1337 | 79.6 | 61.5% | 126.2 | -6.9 | 122.5 | 0 |
| control@2718 | 79.7 | 61.6% | 125.4 | -6.5 | 122.9 | 0 |
| control@7 | 77.0 | 59.5% | 123.6 | -6.2 | 123.2 | 0 |
| control@13 | 78.9 | 61.0% | 123.9 | -4.0 | 125.4 | 0 |
| control@101 | 79.8 | 61.6% | 125.7 | -6.6 | 122.8 | 0 |
| control@271 | 80.0 | 61.8% | 125.9 | -5.0 | 124.4 | 0 |
| control@314 | 78.2 | 60.5% | 124.5 | -7.6 | 121.8 | 0 |
| control@577 | 79.2 | 61.2% | 123.7 | -2.7 | 126.7 | 0 |
| control@863 | 83.7 | 64.7% | 126.2 | -3.6 | 125.8 | 0 |
| control@1024 | 80.3 | 62.1% | 127.4 | -7.8 | 121.6 | 0 |
| control@1729 | 78.4 | 60.6% | 125.3 | -7.8 | 121.6 | 4 |

ABL-337 night screen: not applicable to hydro_total.

## FR / hydro_total — catboost, source `energy_renewable`

n_train 26,890 · n_holdout 692 · incumbent version 20251226_134329

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,900.9 | 19.7% | 2,582.4 | -213.0 | 9,421.8 | 0 |
| control@42 | 1,247.4 | 12.9% | 1,645.5 | -293.3 | 9,341.5 | 0 |
| control@1337 | 1,236.5 | 12.8% | 1,626.7 | -272.1 | 9,362.7 | 0 |
| control@2718 | 1,264.1 | 13.1% | 1,660.2 | -333.0 | 9,301.8 | 0 |
| control@7 | 1,230.4 | 12.8% | 1,624.8 | -314.1 | 9,320.7 | 0 |
| control@13 | 1,246.9 | 12.9% | 1,635.0 | -274.5 | 9,360.3 | 0 |
| control@101 | 1,268.3 | 13.2% | 1,667.3 | -319.4 | 9,315.4 | 0 |
| control@271 | 1,249.0 | 13.0% | 1,645.1 | -307.8 | 9,327.0 | 0 |
| control@314 | 1,263.7 | 13.1% | 1,661.4 | -349.5 | 9,285.3 | 0 |
| control@577 | 1,226.2 | 12.7% | 1,623.9 | -301.8 | 9,333.1 | 0 |
| control@863 | 1,261.3 | 13.1% | 1,659.1 | -357.2 | 9,277.6 | 0 |
| control@1024 | 1,251.6 | 13.0% | 1,653.5 | -317.7 | 9,317.1 | 0 |
| control@1729 | 1,238.8 | 12.9% | 1,622.7 | -267.8 | 9,367.0 | 0 |

ABL-337 night screen: not applicable to hydro_total.
