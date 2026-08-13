# ABL-338 — solar non-negativity and solar geometry: held-out A/B

Generated 2026-08-12T22:49:18 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-03-01 .. 2026-04-29**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. Night hours are
reported in MW, never as a percentage: their denominator is ~0.

## AT — xgboost, source `energy_renewable`

n_train 2,207 · n_holdout 1,440 (daylight 759 / shoulder 152 / night 529) · incumbent version 20260112_165237

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 789.6 | 49.4% | 0.2 | 0.0 | 0.00 | 0.0 | 0 |
| control | 1,250.6 | 78.2% | 3.1 | 0.8 | -1.32 | 22.0 | 486 |
| geometry | 1,178.5 | 73.7% | 1.9 | 0.7 | -0.07 | 4.0 | 474 |
| geometry_tweedie | 1,302.1 | 81.4% | 0.5 | 0.6 | 0.43 | 0.5 | 0 |
| geometry_poisson | 1,323.9 | 82.8% | 5.0 | 5.2 | 4.92 | 5.1 | 0 |
| geometry_tweedie_deep | 1,302.1 | 81.4% | 0.5 | 0.6 | 0.43 | 0.5 | 0 |
| daylight_fit | 1,186.2 | 74.2% | 3.8 | 0.7 | 0.00 | 0.0 | 95 |
| daylight_fit_tweedie | 1,257.5 | 78.6% | 2.6 | 2.8 | 0.00 | 0.0 | 0 |
| geometry_nightw100 | 1,221.3 | 76.4% | 4.7 | 4.8 | 2.46 | 8.3 | 59 |

Training-target contamination: 0 of 1,118 night rows read above 1 MW (max 0.0 MW); dropped from fit: True.

## BE — xgboost, source `energy_renewable`

n_train 18,837 · n_holdout 1,440 (daylight 760 / shoulder 165 / night 515) · incumbent version 20260201_222022

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,385.5 | 42.8% | 5.7 | 0.7 | 0.00 | 0.0 | 0 |
| control | 665.0 | 20.6% | 19.3 | 17.9 | -2.32 | 19.4 | 386 |
| geometry | 675.9 | 20.9% | 8.3 | 9.3 | -0.16 | 20.1 | 272 |
| geometry_tweedie | 738.5 | 22.8% | 3.6 | 4.7 | 0.00 | 0.0 | 0 |
| geometry_poisson | 776.6 | 24.0% | 3.4 | 5.0 | 0.01 | 0.0 | 0 |
| geometry_tweedie_deep | 738.5 | 22.8% | 3.6 | 4.7 | 0.00 | 0.0 | 0 |
| daylight_fit | 684.7 | 21.2% | 18.3 | 4.3 | 0.00 | 0.0 | 82 |
| daylight_fit_tweedie | 788.5 | 24.4% | 3.4 | 5.0 | 0.00 | 0.0 | 0 |
| geometry_nightw100 | 663.2 | 20.5% | 10.0 | 5.9 | -0.03 | 0.0 | 82 |

Training-target contamination: 0 of 7,155 night rows read above 1 MW (max 0.1 MW); dropped from fit: True.

## DE — xgboost, source `energy_renewable`

n_train 2,314 · n_holdout 1,440 (daylight 755 / shoulder 168 / night 517) · incumbent version 20260223_193822

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 8,420.3 | 38.3% | 40.6 | 16.4 | 0.04 | 1.7 | 0 |
| control | 7,766.9 | 35.3% | 140.6 | 188.5 | 14.73 | 88.1 | 31 |
| geometry | 7,634.9 | 34.7% | 99.0 | 151.9 | 52.66 | 111.3 | 18 |
| geometry_tweedie | 8,994.3 | 40.9% | 29.9 | 27.6 | 0.01 | 0.0 | 0 |
| geometry_poisson | 10,627.8 | 48.4% | 30.6 | 30.5 | 3.38 | 3.4 | 0 |
| geometry_tweedie_deep | 8,994.3 | 40.9% | 29.9 | 27.6 | 0.01 | 0.0 | 0 |
| daylight_fit | 7,784.1 | 35.4% | 110.3 | 163.4 | 0.00 | 0.0 | 2 |
| daylight_fit_tweedie | 8,794.8 | 40.0% | 33.0 | 31.6 | 0.00 | 0.0 | 0 |
| geometry_nightw100 | 7,793.8 | 35.5% | 225.9 | 281.3 | 114.91 | 121.6 | 2 |

Training-target contamination: 1 of 1,212 night rows read above 1 MW (max 1.7 MW); dropped from fit: True.

## FR — xgboost, source `energy_renewable`

n_train 27,228 · n_holdout 1,440 (daylight 760 / shoulder 145 / night 535) · incumbent version 20260201_222014

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 2,208.1 | 26.4% | 53.3 | 173.0 | 9.86 | 271.8 | 0 |
| control | 1,044.0 | 12.5% | 33.5 | 206.2 | 13.34 | 241.6 | 124 |
| geometry | 1,054.7 | 12.6% | 32.3 | 217.9 | 13.88 | 140.8 | 0 |
| geometry_tweedie | 1,187.2 | 14.2% | 53.6 | 166.2 | 0.00 | 0.0 | 0 |
| geometry_poisson | 1,130.9 | 13.5% | 38.7 | 193.1 | 0.29 | 0.9 | 0 |
| geometry_tweedie_deep | 1,187.2 | 14.2% | 53.6 | 166.2 | 0.00 | 0.0 | 0 |
| daylight_fit | 1,028.1 | 12.3% | 50.3 | 249.7 | 0.00 | 0.0 | 0 |
| daylight_fit_tweedie | 1,047.4 | 12.5% | 45.6 | 210.5 | 0.00 | 0.0 | 0 |
| geometry_nightw100 | 1,022.4 | 12.2% | 36.4 | 207.7 | 0.17 | 15.8 | 206 |

Training-target contamination: 432 of 10,802 night rows read above 1 MW (max 439.3 MW); dropped from fit: True.
