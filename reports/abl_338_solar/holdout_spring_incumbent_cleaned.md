# ABL-338 — solar non-negativity and solar geometry: held-out A/B

Generated 2026-08-12T22:54:09 against replica `C:\Code\able\data\energy_dashboard.db`.

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

## BE — catboost, source `energy_renewable`

n_train 18,837 · n_holdout 1,440 (daylight 760 / shoulder 165 / night 515) · incumbent version 20260201_222022

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,385.5 | 42.8% | 5.7 | 0.7 | 0.00 | 0.0 | 0 |
| control | 652.7 | 20.2% | 36.1 | 18.3 | 2.15 | 114.4 | 305 |
| geometry | 684.7 | 21.2% | 37.1 | 34.9 | 9.16 | 96.7 | 249 |
| geometry_tweedie | 944.5 | 29.2% | 4.5 | 4.9 | 0.12 | 0.7 | 0 |
| geometry_poisson | 805.9 | 24.9% | 6.5 | 9.0 | 0.11 | 1.5 | 0 |
| geometry_tweedie_deep | 944.5 | 29.2% | 4.5 | 4.9 | 0.12 | 0.7 | 0 |
| daylight_fit | 663.0 | 20.5% | 42.5 | 12.6 | 0.00 | 0.0 | 85 |
| daylight_fit_tweedie | 922.4 | 28.5% | 5.6 | 6.6 | 0.00 | 0.0 | 0 |
| geometry_nightw100 | 751.2 | 23.2% | 40.8 | 46.8 | 5.16 | 49.7 | 170 |

Training-target contamination: 0 of 7,155 night rows read above 1 MW (max 0.1 MW); dropped from fit: True.

## DE — catboost, source `energy_renewable`

n_train 2,314 · n_holdout 1,440 (daylight 755 / shoulder 168 / night 517) · incumbent version 20260223_193822

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 8,420.3 | 38.3% | 40.6 | 16.4 | 0.04 | 1.7 | 0 |
| control | 10,160.2 | 46.2% | 390.8 | 445.3 | 247.68 | 664.3 | 1 |
| geometry | 8,866.6 | 40.4% | 376.1 | 432.3 | 190.18 | 633.4 | 12 |
| geometry_tweedie | 17,140.5 | 78.0% | 35.2 | 21.9 | 0.31 | 1.3 | 0 |
| geometry_poisson | 21,971.1 | 100.0% | 55.8 | 1.0 | 1.00 | 1.0 | 0 |
| geometry_tweedie_deep | 17,140.5 | 78.0% | 35.2 | 21.9 | 0.31 | 1.3 | 0 |
| daylight_fit | 8,902.8 | 40.5% | 694.2 | 750.5 | 0.00 | 0.0 | 0 |
| daylight_fit_tweedie | 19,485.6 | 88.7% | 34.7 | 27.3 | 0.00 | 0.0 | 0 |
| geometry_nightw100 | 10,785.3 | 49.1% | 100.6 | 144.8 | 1.03 | 9.1 | 168 |

Training-target contamination: 1 of 1,212 night rows read above 1 MW (max 1.7 MW); dropped from fit: True.

## FR — catboost, source `energy_renewable`

n_train 27,228 · n_holdout 1,440 (daylight 760 / shoulder 145 / night 535) · incumbent version 20260201_222014

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 2,208.1 | 26.4% | 53.3 | 173.0 | 9.86 | 271.8 | 0 |
| control | 1,042.5 | 12.5% | 86.0 | 243.4 | 21.91 | 230.3 | 218 |
| geometry | 1,051.1 | 12.6% | 96.8 | 242.9 | 5.63 | 311.4 | 270 |
| geometry_tweedie | 1,236.2 | 14.8% | 48.6 | 194.9 | 0.66 | 10.7 | 0 |
| geometry_poisson | 8,361.1 | 100.0% | 206.1 | 1.0 | 1.00 | 1.0 | 0 |
| geometry_tweedie_deep | 1,262.3 | 15.1% | 44.4 | 195.2 | 0.31 | 6.1 | 0 |
| daylight_fit | 1,029.4 | 12.3% | 110.7 | 274.2 | 0.00 | 0.0 | 4 |
| daylight_fit_tweedie | 2,276.4 | 27.2% | 40.7 | 206.6 | 0.00 | 0.0 | 0 |
| geometry_nightw100 | 1,150.1 | 13.8% | 82.9 | 225.1 | 1.21 | 13.1 | 179 |

Training-target contamination: 432 of 10,802 night rows read above 1 MW (max 439.3 MW); dropped from fit: True.
