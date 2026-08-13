# ABL-338 — solar non-negativity and solar geometry: held-out A/B

Generated 2026-08-12T22:55:15 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-06-13 .. 2026-08-11**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. Night hours are
reported in MW, never as a percentage: their denominator is ~0.

## AT — xgboost, source `energy_renewable`

n_train 4,703 · n_holdout 1,440 (daylight 928 / shoulder 152 / night 360) · incumbent version 20260112_165237

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 536.5 | 24.0% | 0.7 | 2.9 | 0.00 | 0.0 | 0 |
| control | 291.9 | 13.1% | 6.0 | 2.1 | -3.42 | 3.2 | 370 |
| geometry | 294.6 | 13.2% | 10.1 | 12.6 | 10.74 | 10.8 | 0 |
| geometry_tweedie | 293.6 | 13.2% | 1.6 | 0.9 | 0.04 | 0.0 | 0 |
| geometry_poisson | 289.0 | 13.0% | 1.7 | 0.9 | 0.09 | 0.1 | 0 |
| geometry_tweedie_deep | 293.6 | 13.2% | 1.6 | 0.9 | 0.04 | 0.0 | 0 |
| daylight_fit | 290.6 | 13.0% | 10.0 | 12.4 | 0.00 | 0.0 | 0 |
| daylight_fit_tweedie | 290.5 | 13.0% | 1.4 | 1.3 | 0.00 | 0.0 | 0 |
| geometry_nightw100 | 324.6 | 14.5% | 4.5 | 6.8 | 0.05 | 0.0 | 10 |

Training-target contamination: 0 of 1,914 night rows read above 1 MW (max 0.0 MW); dropped from fit: True.

## BE — catboost, source `energy_renewable`

n_train 21,333 · n_holdout 1,440 (daylight 945 / shoulder 204 / night 291) · incumbent version 20260201_222022

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,032.2 | 29.7% | 2.4 | 3.5 | 0.00 | 0.0 | 0 |
| control | 508.1 | 14.6% | 19.2 | 7.8 | -4.21 | 28.3 | 247 |
| geometry | 493.8 | 14.2% | 13.6 | 8.4 | 1.13 | 28.2 | 214 |
| geometry_tweedie | 661.5 | 19.0% | 1.8 | 2.1 | 0.19 | 0.6 | 0 |
| geometry_poisson | 572.3 | 16.5% | 7.1 | 8.9 | 0.32 | 2.7 | 0 |
| geometry_tweedie_deep | 661.5 | 19.0% | 1.8 | 2.1 | 0.19 | 0.6 | 0 |
| daylight_fit | 495.0 | 14.2% | 31.2 | -21.6 | 0.00 | 0.0 | 165 |
| daylight_fit_tweedie | 684.4 | 19.7% | 2.9 | 4.3 | 0.00 | 0.0 | 0 |
| geometry_nightw100 | 524.7 | 15.1% | 31.1 | 31.2 | -2.23 | 1.4 | 259 |

Training-target contamination: 0 of 7,902 night rows read above 1 MW (max 0.1 MW); dropped from fit: True.

## DE — catboost, source `energy_renewable`

n_train 4,807 · n_holdout 1,440 (daylight 937 / shoulder 209 / night 294) · incumbent version 20260223_193822

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 5,933.4 | 23.2% | 26.4 | 61.7 | 0.00 | 0.0 | 0 |
| control | 3,852.2 | 15.1% | 284.9 | 304.3 | 220.12 | 669.6 | 52 |
| geometry | 3,694.8 | 14.5% | 501.4 | 549.3 | 453.76 | 815.3 | 7 |
| geometry_tweedie | 15,928.6 | 62.3% | 23.6 | 29.0 | 0.86 | 3.0 | 0 |
| geometry_poisson | 25,550.8 | 100.0% | 48.1 | 1.0 | 1.00 | 1.0 | 0 |
| geometry_tweedie_deep | 15,928.6 | 62.3% | 23.6 | 29.0 | 0.86 | 3.0 | 0 |
| daylight_fit | 3,653.8 | 14.3% | 912.9 | 957.0 | 0.00 | 0.0 | 3 |
| daylight_fit_tweedie | 12,095.0 | 47.3% | 24.6 | 38.0 | 0.00 | 0.0 | 0 |
| geometry_nightw100 | 4,331.4 | 17.0% | 174.8 | 220.0 | 1.04 | 100.6 | 134 |

Training-target contamination: 4 of 1,957 night rows read above 1 MW (max 1.7 MW); dropped from fit: True.

## FR — catboost, source `energy_renewable`

n_train 29,668 · n_holdout 1,440 (daylight 915 / shoulder 159 / night 366) · incumbent version 20260201_222014

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,989.0 | 19.0% | 47.0 | 217.1 | 34.85 | 321.5 | 0 |
| control | 1,374.0 | 13.2% | 90.7 | 156.5 | 15.40 | 169.2 | 163 |
| geometry | 1,285.1 | 12.3% | 91.0 | 159.5 | -2.63 | 118.0 | 187 |
| geometry_tweedie | 1,598.6 | 15.3% | 48.2 | 193.6 | 1.02 | 18.7 | 0 |
| geometry_poisson | 10,446.6 | 100.0% | 209.8 | 1.0 | 1.00 | 1.0 | 0 |
| geometry_tweedie_deep | 1,641.9 | 15.7% | 46.8 | 196.6 | 0.66 | 14.3 | 0 |
| daylight_fit | 1,279.6 | 12.2% | 81.8 | 197.0 | 0.00 | 0.0 | 1 |
| daylight_fit_tweedie | 1,829.4 | 17.5% | 36.0 | 203.8 | 0.00 | 0.0 | 0 |
| geometry_nightw100 | 1,438.7 | 13.8% | 86.1 | 183.7 | -7.78 | 21.5 | 265 |

Training-target contamination: 488 of 11,614 night rows read above 1 MW (max 439.3 MW); dropped from fit: True.
