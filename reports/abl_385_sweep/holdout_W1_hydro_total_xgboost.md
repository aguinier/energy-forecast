# Held-out A/B — hydro_total (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T13:24:02 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-02-13 .. 2026-03-14**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `hydro_total` has no band structure, so one all-hours row is the result.

## BE / hydro_total — xgboost, source `energy_renewable`

n_train 18,150 · n_holdout 692 · incumbent version 20251226_155416

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 95.7 | 73.9% | 189.1 | -18.7 | 110.7 | 0 |
| control@42 | 82.2 | 63.5% | 127.8 | -4.5 | 124.9 | 0 |
| control@1337 | 80.3 | 62.1% | 128.1 | -7.9 | 121.5 | 0 |
| control@2718 | 82.1 | 63.5% | 130.8 | -8.0 | 121.4 | 0 |
| control@7 | 82.5 | 63.7% | 127.3 | -4.1 | 125.3 | 0 |
| control@13 | 79.6 | 61.5% | 125.8 | -6.7 | 122.7 | 0 |
| control@101 | 81.6 | 63.1% | 127.5 | -5.9 | 123.5 | 0 |
| control@271 | 80.9 | 62.5% | 127.0 | -6.5 | 122.9 | 0 |
| control@314 | 82.3 | 63.6% | 128.7 | -5.3 | 124.1 | 0 |
| control@577 | 76.6 | 59.2% | 124.3 | -6.0 | 123.4 | 0 |
| control@863 | 80.9 | 62.5% | 128.8 | -9.1 | 120.3 | 0 |
| control@1024 | 78.2 | 60.5% | 125.8 | -6.6 | 122.8 | 3 |
| control@1729 | 81.9 | 63.3% | 130.0 | -8.3 | 121.1 | 0 |

ABL-337 night screen: not applicable to hydro_total.

## FR / hydro_total — xgboost, source `energy_renewable`

n_train 26,890 · n_holdout 692 · incumbent version 20251226_134329

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,900.9 | 19.7% | 2,582.4 | -213.0 | 9,421.8 | 0 |
| control@42 | 1,323.7 | 13.7% | 1,730.8 | -242.4 | 9,392.4 | 0 |
| control@1337 | 1,336.5 | 13.9% | 1,744.5 | -257.6 | 9,377.2 | 0 |
| control@2718 | 1,337.1 | 13.9% | 1,747.2 | -244.9 | 9,389.9 | 0 |
| control@7 | 1,321.3 | 13.7% | 1,727.7 | -279.7 | 9,355.1 | 0 |
| control@13 | 1,325.3 | 13.8% | 1,720.4 | -264.0 | 9,370.8 | 0 |
| control@101 | 1,334.9 | 13.9% | 1,737.2 | -290.9 | 9,343.9 | 0 |
| control@271 | 1,335.6 | 13.9% | 1,728.5 | -297.0 | 9,337.8 | 0 |
| control@314 | 1,364.7 | 14.2% | 1,773.5 | -264.9 | 9,369.9 | 0 |
| control@577 | 1,304.7 | 13.5% | 1,706.4 | -308.5 | 9,326.3 | 0 |
| control@863 | 1,336.9 | 13.9% | 1,737.3 | -267.2 | 9,367.6 | 0 |
| control@1024 | 1,343.2 | 13.9% | 1,743.9 | -241.4 | 9,393.5 | 0 |
| control@1729 | 1,349.6 | 14.0% | 1,767.4 | -270.7 | 9,364.1 | 0 |

ABL-337 night screen: not applicable to hydro_total.
