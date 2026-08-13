# ABL-338 — solar non-negativity and solar geometry: held-out A/B

Generated 2026-08-13T12:53:09 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-04-30 .. 2026-06-12**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. Night hours are
reported in MW, never as a percentage: their denominator is ~0.

## AT — xgboost, source `energy_renewable`

n_train 3,647 · n_holdout 1,056 (daylight 670 / shoulder 119 / night 267) · incumbent version 20260112_165237

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 781.9 | 37.2% | 1.0 | 0.3 | 0.00 | 0.0 | 0 |
| control_noholiday@42 | 527.9 | 25.1% | 2.2 | 2.1 | 0.66 | 10.3 | 132 |
| control_noholiday@1337 | 530.2 | 25.3% | 2.1 | 2.0 | -0.35 | 4.0 | 200 |
| control_noholiday@2718 | 523.0 | 24.9% | 4.7 | 5.1 | 2.73 | 9.4 | 85 |
| geometry_noholiday@42 | 510.1 | 24.3% | 2.9 | 3.1 | 2.25 | 9.5 | 48 |
| geometry_noholiday@1337 | 502.5 | 23.9% | 5.5 | 6.0 | 4.57 | 16.1 | 67 |
| geometry_noholiday@2718 | 502.5 | 23.9% | 4.5 | 5.0 | 3.89 | 11.8 | 51 |
| control@42 | 524.7 | 25.0% | 2.3 | 0.2 | -1.34 | 1.9 | 288 |
| control@1337 | 514.5 | 24.5% | 3.1 | 3.0 | -0.18 | 5.0 | 227 |
| control@2718 | 514.2 | 24.5% | 4.2 | 4.5 | 1.93 | 11.7 | 104 |
| geometry@42 | 494.0 | 23.5% | 3.9 | 4.2 | 3.42 | 15.1 | 36 |
| geometry@1337 | 499.9 | 23.8% | 3.2 | 3.2 | 1.68 | 8.7 | 141 |
| geometry@2718 | 494.7 | 23.6% | 7.1 | 8.0 | 7.06 | 23.6 | 34 |

Training-target contamination: 0 of 1,647 night rows read above 1 MW (max 0.0 MW); dropped from fit: True.

## BE — xgboost, source `energy_renewable`

n_train 20,277 · n_holdout 1,056 (daylight 681 / shoulder 143 / night 232) · incumbent version 20260201_222022

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,425.5 | 49.2% | 1.5 | 0.7 | 0.00 | 0.0 | 0 |
| control_noholiday@42 | 604.3 | 20.9% | 5.9 | 2.1 | -1.04 | 15.5 | 130 |
| control_noholiday@1337 | 624.2 | 21.5% | 5.4 | 5.7 | 2.86 | 19.2 | 35 |
| control_noholiday@2718 | 612.1 | 21.1% | 4.8 | 5.8 | 2.32 | 3.1 | 1 |
| geometry_noholiday@42 | 605.8 | 20.9% | 3.0 | 4.0 | 1.71 | 17.6 | 10 |
| geometry_noholiday@1337 | 619.2 | 21.4% | 3.7 | 3.4 | 0.06 | 9.3 | 90 |
| geometry_noholiday@2718 | 587.8 | 20.3% | 3.3 | 2.5 | -0.60 | 2.9 | 116 |
| control@42 | 635.5 | 21.9% | 8.4 | -0.6 | -5.74 | 6.3 | 175 |
| control@1337 | 643.1 | 22.2% | 8.2 | 2.3 | 0.02 | 38.2 | 190 |
| control@2718 | 627.2 | 21.6% | 8.0 | 9.6 | 7.17 | 7.2 | 0 |
| geometry@42 | 598.8 | 20.7% | 2.3 | 3.1 | 0.56 | 1.9 | 72 |
| geometry@1337 | 597.5 | 20.6% | 4.1 | 3.0 | 0.05 | 8.5 | 49 |
| geometry@2718 | 622.1 | 21.5% | 5.8 | 0.7 | -1.81 | 12.6 | 198 |

Training-target contamination: 0 of 7,670 night rows read above 1 MW (max 0.1 MW); dropped from fit: True.

## DE — xgboost, source `energy_renewable`

n_train 3,751 · n_holdout 1,056 (daylight 686 / shoulder 142 / night 228) · incumbent version 20260223_193822

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 6,509.3 | 29.5% | 14.8 | 13.9 | 0.00 | 0.4 | 0 |
| control_noholiday@42 | 3,909.1 | 17.7% | 51.9 | -12.4 | -37.73 | 26.9 | 240 |
| control_noholiday@1337 | 3,967.6 | 18.0% | 98.9 | -62.2 | -59.47 | 82.2 | 228 |
| control_noholiday@2718 | 4,032.2 | 18.3% | 62.6 | -23.2 | -35.45 | 54.1 | 195 |
| geometry_noholiday@42 | 4,381.6 | 19.9% | 43.2 | 3.6 | -33.08 | 3.4 | 161 |
| geometry_noholiday@1337 | 4,209.5 | 19.1% | 64.6 | -22.0 | -51.88 | 34.6 | 206 |
| geometry_noholiday@2718 | 4,332.4 | 19.6% | 42.4 | 0.3 | -34.38 | 2.4 | 198 |
| control@42 | 3,856.8 | 17.5% | 49.8 | -8.7 | -25.73 | 51.3 | 155 |
| control@1337 | 3,958.1 | 17.9% | 67.5 | -34.4 | -53.34 | 1.2 | 332 |
| control@2718 | 3,928.7 | 17.8% | 44.8 | -5.3 | -35.64 | 15.3 | 199 |
| geometry@42 | 4,362.6 | 19.8% | 81.5 | -42.9 | -63.19 | 40.1 | 189 |
| geometry@1337 | 4,161.8 | 18.9% | 40.1 | -0.3 | -24.57 | 24.5 | 267 |
| geometry@2718 | 4,224.2 | 19.1% | 56.7 | -12.8 | -45.87 | 14.8 | 153 |

Training-target contamination: 4 of 1,729 night rows read above 1 MW (max 1.7 MW); dropped from fit: True.

## FR — xgboost, source `energy_renewable`

n_train 28,636 · n_holdout 1,056 (daylight 656 / shoulder 123 / night 277) · incumbent version 20260201_222014

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 2,128.1 | 25.6% | 29.9 | 146.2 | 7.28 | 251.0 | 0 |
| control_noholiday@42 | 978.2 | 11.7% | 36.2 | 189.5 | 40.85 | 225.4 | 0 |
| control_noholiday@1337 | 975.9 | 11.7% | 31.2 | 179.6 | 32.75 | 226.8 | 0 |
| control_noholiday@2718 | 1,010.7 | 12.1% | 40.7 | 193.2 | 46.07 | 218.0 | 0 |
| geometry_noholiday@42 | 987.0 | 11.9% | 35.5 | 185.7 | 19.50 | 68.8 | 0 |
| geometry_noholiday@1337 | 959.8 | 11.5% | 48.0 | 202.6 | 36.80 | 78.9 | 0 |
| geometry_noholiday@2718 | 976.2 | 11.7% | 51.1 | 205.8 | 41.27 | 99.1 | 0 |
| control@42 | 979.8 | 11.8% | 33.0 | 184.5 | 36.81 | 192.2 | 0 |
| control@1337 | 982.4 | 11.8% | 47.6 | 202.3 | 45.62 | 223.1 | 0 |
| control@2718 | 980.0 | 11.8% | 33.9 | 182.2 | 29.82 | 208.0 | 0 |
| geometry@42 | 961.1 | 11.5% | 36.2 | 188.8 | 22.44 | 129.8 | 0 |
| geometry@1337 | 965.4 | 11.6% | 53.8 | 208.5 | 42.56 | 112.2 | 0 |
| geometry@2718 | 997.9 | 12.0% | 57.1 | 214.0 | 46.67 | 108.7 | 0 |

Training-target contamination: 464 of 11,337 night rows read above 1 MW (max 439.3 MW); dropped from fit: True.
