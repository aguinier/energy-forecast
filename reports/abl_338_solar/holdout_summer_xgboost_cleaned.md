# ABL-338 — solar non-negativity and solar geometry: held-out A/B

Generated 2026-08-12T22:49:44 against replica `C:\Code\able\data\energy_dashboard.db`.

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

## BE — xgboost, source `energy_renewable`

n_train 21,333 · n_holdout 1,440 (daylight 945 / shoulder 204 / night 291) · incumbent version 20260201_222022

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,032.2 | 29.7% | 2.4 | 3.5 | 0.00 | 0.0 | 0 |
| control | 540.3 | 15.5% | 22.5 | 24.4 | 21.22 | 21.2 | 0 |
| geometry | 518.6 | 14.9% | 15.2 | 17.0 | 14.08 | 14.1 | 0 |
| geometry_tweedie | 499.4 | 14.4% | 1.1 | 2.3 | 0.02 | 0.0 | 0 |
| geometry_poisson | 507.5 | 14.6% | 2.2 | 3.7 | 1.15 | 1.2 | 0 |
| geometry_tweedie_deep | 499.4 | 14.4% | 1.1 | 2.3 | 0.02 | 0.0 | 0 |
| daylight_fit | 499.8 | 14.4% | 12.6 | 14.3 | 0.00 | 0.0 | 0 |
| daylight_fit_tweedie | 527.8 | 15.2% | 1.2 | 2.4 | 0.00 | 0.0 | 0 |
| geometry_nightw100 | 524.5 | 15.1% | 3.6 | 4.1 | 0.22 | 0.2 | 19 |

Training-target contamination: 0 of 7,902 night rows read above 1 MW (max 0.1 MW); dropped from fit: True.

## DE — xgboost, source `energy_renewable`

n_train 4,807 · n_holdout 1,440 (daylight 937 / shoulder 209 / night 294) · incumbent version 20260223_193822

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 5,933.4 | 23.2% | 26.4 | 61.7 | 0.00 | 0.0 | 0 |
| control | 3,173.3 | 12.4% | 189.3 | 237.0 | 94.37 | 99.4 | 0 |
| geometry | 3,076.3 | 12.0% | 79.2 | 122.0 | 12.15 | 42.6 | 40 |
| geometry_tweedie | 3,073.2 | 12.0% | 12.2 | 44.8 | 0.00 | 0.0 | 0 |
| geometry_poisson | 3,084.5 | 12.1% | 12.8 | 45.5 | 0.03 | 0.1 | 0 |
| geometry_tweedie_deep | 3,073.2 | 12.0% | 12.2 | 44.8 | 0.00 | 0.0 | 0 |
| daylight_fit | 3,160.4 | 12.4% | 216.4 | 264.3 | 0.00 | 0.0 | 0 |
| daylight_fit_tweedie | 3,048.6 | 11.9% | 11.6 | 50.0 | 0.00 | 0.0 | 0 |
| geometry_nightw100 | 3,376.2 | 13.2% | 48.1 | 92.4 | 0.80 | 0.8 | 34 |

Training-target contamination: 4 of 1,957 night rows read above 1 MW (max 1.7 MW); dropped from fit: True.

## FR — xgboost, source `energy_renewable`

n_train 29,668 · n_holdout 1,440 (daylight 915 / shoulder 159 / night 366) · incumbent version 20260201_222014

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,989.0 | 19.0% | 47.0 | 217.1 | 34.85 | 321.5 | 0 |
| control | 1,426.7 | 13.7% | 25.0 | 210.6 | 33.74 | 274.7 | 0 |
| geometry | 1,468.6 | 14.1% | 27.2 | 213.5 | 7.81 | 205.7 | 35 |
| geometry_tweedie | 1,353.2 | 13.0% | 27.7 | 215.2 | 0.05 | 0.3 | 0 |
| geometry_poisson | 1,368.9 | 13.1% | 27.2 | 220.0 | 1.48 | 4.0 | 0 |
| geometry_tweedie_deep | 1,353.2 | 13.0% | 27.7 | 215.2 | 0.05 | 0.3 | 0 |
| daylight_fit | 1,372.9 | 13.1% | 54.9 | 260.5 | 0.00 | 0.0 | 0 |
| daylight_fit_tweedie | 1,263.3 | 12.1% | 32.1 | 216.8 | 0.00 | 0.0 | 0 |
| geometry_nightw100 | 1,469.0 | 14.1% | 29.3 | 219.0 | 0.03 | 12.7 | 23 |

Training-target contamination: 488 of 11,614 night rows read above 1 MW (max 439.3 MW); dropped from fit: True.
