# ABL-338 — solar non-negativity and solar geometry: held-out A/B

Generated 2026-08-13T13:48:56 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-05-14 .. 2026-06-12**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. Night hours are reported in MW, never as a percentage: their denominator is ~0.

## AT — xgboost, source `energy_renewable`

n_train 3,983 · n_holdout 720 (daylight 472 / shoulder 68 / night 180) · incumbent version 20260112_165237

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 858.8 | 43.3% | 1.2 | 0.3 | 0.00 | 0.0 | 0 |
| control@42 | 418.8 | 21.1% | 2.8 | 1.5 | 0.80 | 5.7 | 88 |
| control@1337 | 419.8 | 21.1% | 4.2 | 2.7 | 0.49 | 10.0 | 103 |
| control@2718 | 413.9 | 20.8% | 3.3 | 0.7 | 1.62 | 14.2 | 111 |
| control@7 | 419.2 | 21.1% | 2.4 | 0.9 | 0.31 | 5.5 | 110 |
| control@13 | 441.6 | 22.2% | 3.2 | 1.6 | 0.95 | 7.1 | 110 |
| control@101 | 447.0 | 22.5% | 3.1 | -0.8 | -2.46 | 2.0 | 211 |
| control@271 | 423.3 | 21.3% | 4.7 | 3.0 | 1.99 | 10.5 | 55 |
| control@314 | 417.9 | 21.0% | 3.2 | 0.9 | -0.90 | 7.1 | 163 |
| control@577 | 415.0 | 20.9% | 2.6 | 1.8 | 1.07 | 6.4 | 70 |
| control@863 | 408.0 | 20.5% | 4.7 | -0.7 | -0.60 | 5.4 | 162 |
| control@1024 | 425.6 | 21.4% | 5.7 | -4.0 | -2.85 | 2.7 | 220 |
| control@1729 | 420.6 | 21.2% | 3.5 | 1.6 | 0.26 | 11.5 | 114 |
| geometry@42 | 460.4 | 23.2% | 1.8 | 1.8 | 0.78 | 6.9 | 62 |
| geometry@1337 | 449.4 | 22.6% | 2.2 | 0.7 | 1.03 | 7.0 | 88 |
| geometry@2718 | 441.7 | 22.2% | 2.5 | 1.8 | 0.90 | 6.8 | 56 |
| geometry@7 | 456.6 | 23.0% | 1.9 | 0.8 | 0.47 | 4.1 | 96 |
| geometry@13 | 447.6 | 22.5% | 3.0 | 0.2 | 0.17 | 6.7 | 116 |
| geometry@101 | 448.0 | 22.6% | 2.4 | -0.8 | -0.71 | 3.5 | 162 |
| geometry@271 | 445.0 | 22.4% | 2.4 | 0.8 | 0.70 | 6.3 | 92 |
| geometry@314 | 442.7 | 22.3% | 3.8 | 3.1 | 2.85 | 14.1 | 64 |
| geometry@577 | 439.2 | 22.1% | 2.2 | 1.4 | 1.18 | 7.9 | 78 |
| geometry@863 | 434.4 | 21.9% | 3.8 | 3.3 | 2.91 | 16.7 | 61 |
| geometry@1024 | 446.4 | 22.5% | 1.7 | 0.2 | -0.77 | 3.1 | 211 |
| geometry@1729 | 452.8 | 22.8% | 2.9 | -1.1 | -1.21 | 4.4 | 183 |

Training-target contamination: 0 of 1,734 night rows read above 1 MW (max 0.0 MW); dropped from fit: True.

## BE — xgboost, source `energy_renewable`

n_train 20,613 · n_holdout 720 (daylight 471 / shoulder 103 / night 146) · incumbent version 20260201_222022

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,359.4 | 45.8% | 1.9 | 1.0 | 0.00 | 0.0 | 0 |
| control@42 | 591.9 | 19.9% | 3.9 | 1.2 | -0.77 | 1.6 | 89 |
| control@1337 | 574.4 | 19.3% | 5.7 | 7.4 | 3.91 | 5.0 | 0 |
| control@2718 | 580.5 | 19.5% | 4.9 | 6.1 | 4.05 | 4.7 | 0 |
| control@7 | 575.1 | 19.4% | 7.6 | 9.7 | 6.99 | 7.2 | 0 |
| control@13 | 601.7 | 20.3% | 3.3 | 4.3 | 1.56 | 3.2 | 6 |
| control@101 | 590.1 | 19.9% | 7.2 | 8.9 | 6.39 | 31.9 | 11 |
| control@271 | 575.5 | 19.4% | 4.8 | 5.2 | 1.51 | 7.9 | 26 |
| control@314 | 583.8 | 19.7% | 4.1 | 5.5 | 2.89 | 3.8 | 5 |
| control@577 | 580.7 | 19.5% | 3.3 | 4.3 | 1.48 | 5.4 | 10 |
| control@863 | 596.7 | 20.1% | 3.5 | 4.3 | 1.78 | 5.8 | 13 |
| control@1024 | 583.9 | 19.7% | 4.4 | 5.5 | 3.00 | 4.2 | 6 |
| control@1729 | 605.0 | 20.4% | 5.9 | 7.0 | 2.67 | 5.1 | 15 |
| geometry@42 | 593.8 | 20.0% | 4.5 | 6.0 | 1.66 | 17.8 | 33 |
| geometry@1337 | 584.6 | 19.7% | 3.4 | 4.0 | 0.81 | 2.5 | 34 |
| geometry@2718 | 605.3 | 20.4% | 5.9 | -1.3 | -2.32 | 20.8 | 160 |
| geometry@7 | 573.6 | 19.3% | 4.4 | 6.3 | 3.67 | 4.9 | 0 |
| geometry@13 | 596.5 | 20.1% | 3.4 | 5.0 | 1.44 | 2.6 | 16 |
| geometry@101 | 590.2 | 19.9% | 4.2 | 4.5 | 3.79 | 20.8 | 19 |
| geometry@271 | 570.5 | 19.2% | 3.8 | 1.4 | -1.67 | 2.2 | 106 |
| geometry@314 | 566.5 | 19.1% | 4.8 | 6.6 | 3.66 | 4.2 | 0 |
| geometry@577 | 586.3 | 19.7% | 3.2 | 4.0 | 1.22 | 3.5 | 25 |
| geometry@863 | 581.6 | 19.6% | 3.4 | 4.9 | 2.21 | 3.1 | 1 |
| geometry@1024 | 590.5 | 19.9% | 5.3 | 0.0 | -2.46 | 7.8 | 134 |
| geometry@1729 | 594.6 | 20.0% | 3.8 | 2.2 | -0.98 | 1.4 | 88 |

Training-target contamination: 0 of 7,756 night rows read above 1 MW (max 0.1 MW); dropped from fit: True.

## DE — xgboost, source `energy_renewable`

n_train 4,087 · n_holdout 720 (daylight 477 / shoulder 99 / night 144) · incumbent version 20260223_193822

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 6,618.6 | 29.4% | 13.8 | 14.8 | 0.00 | 0.0 | 0 |
| control@42 | 2,988.5 | 13.3% | 112.5 | -80.6 | -77.15 | 44.1 | 197 |
| control@1337 | 2,900.8 | 12.9% | 73.9 | -41.0 | -63.68 | 7.7 | 189 |
| control@2718 | 3,005.4 | 13.3% | 69.5 | -27.7 | -47.49 | 10.3 | 166 |
| control@7 | 2,920.7 | 13.0% | 177.8 | -134.2 | -144.92 | 26.2 | 159 |
| control@13 | 2,969.1 | 13.2% | 76.6 | -43.8 | -66.58 | 13.6 | 201 |
| control@101 | 2,898.2 | 12.9% | 73.3 | -39.2 | -51.49 | 9.7 | 197 |
| control@271 | 3,004.4 | 13.3% | 93.0 | -54.0 | -69.17 | 38.5 | 157 |
| control@314 | 2,914.2 | 12.9% | 124.8 | -90.1 | -99.20 | 17.0 | 190 |
| control@577 | 2,933.0 | 13.0% | 124.2 | -95.5 | -92.76 | 22.8 | 210 |
| control@863 | 2,873.9 | 12.7% | 84.5 | -47.0 | -63.20 | 5.2 | 206 |
| control@1024 | 2,952.2 | 13.1% | 125.4 | -85.7 | -99.66 | 2.9 | 188 |
| control@1729 | 2,992.9 | 13.3% | 103.5 | -66.4 | -80.93 | 4.9 | 192 |
| geometry@42 | 3,331.7 | 14.8% | 115.2 | -73.4 | -74.18 | 75.5 | 142 |
| geometry@1337 | 3,307.7 | 14.7% | 94.8 | -64.6 | -86.01 | 3.2 | 161 |
| geometry@2718 | 3,392.5 | 15.0% | 153.3 | -109.1 | -124.66 | 47.7 | 143 |
| geometry@7 | 3,261.9 | 14.5% | 111.7 | -73.6 | -87.79 | 40.2 | 145 |
| geometry@13 | 3,291.6 | 14.6% | 150.7 | -112.9 | -125.20 | 95.6 | 171 |
| geometry@101 | 3,337.8 | 14.8% | 109.9 | -74.9 | -87.15 | 27.3 | 157 |
| geometry@271 | 3,498.0 | 15.5% | 89.4 | -44.8 | -69.95 | 56.2 | 126 |
| geometry@314 | 3,381.1 | 15.0% | 115.3 | -82.3 | -97.20 | 28.7 | 152 |
| geometry@577 | 3,265.3 | 14.5% | 140.4 | -103.9 | -112.53 | 29.1 | 157 |
| geometry@863 | 3,351.7 | 14.9% | 151.5 | -101.9 | -109.06 | 73.3 | 138 |
| geometry@1024 | 3,287.8 | 14.6% | 199.4 | -163.1 | -156.16 | 57.7 | 162 |
| geometry@1729 | 3,283.5 | 14.6% | 95.2 | -44.9 | -74.87 | 28.6 | 133 |

Training-target contamination: 4 of 1,813 night rows read above 1 MW (max 1.7 MW); dropped from fit: True.

## FR — xgboost, source `energy_renewable`

n_train 28,960 · n_holdout 720 (daylight 457 / shoulder 83 / night 180) · incumbent version 20260201_222014

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 2,152.6 | 25.1% | 29.2 | 146.7 | 5.38 | 251.0 | 0 |
| control@42 | 997.8 | 11.6% | 24.1 | 165.0 | 16.85 | 183.4 | 0 |
| control@1337 | 986.5 | 11.5% | 42.5 | 194.0 | 42.65 | 228.6 | 0 |
| control@2718 | 995.6 | 11.6% | 41.1 | 192.1 | 39.13 | 211.8 | 0 |
| control@7 | 989.8 | 11.5% | 54.5 | 208.8 | 52.48 | 217.1 | 0 |
| control@13 | 1,003.6 | 11.7% | 41.9 | 195.7 | 43.02 | 213.6 | 0 |
| control@101 | 991.2 | 11.6% | 59.4 | 215.1 | 65.73 | 252.6 | 0 |
| control@271 | 985.7 | 11.5% | 28.1 | 171.3 | 23.65 | 201.9 | 0 |
| control@314 | 1,013.9 | 11.8% | 47.7 | 201.0 | 53.96 | 242.4 | 0 |
| control@577 | 996.4 | 11.6% | 30.2 | 174.1 | 30.00 | 212.8 | 0 |
| control@863 | 1,008.9 | 11.8% | 20.6 | 157.8 | 11.28 | 200.9 | 20 |
| control@1024 | 984.9 | 11.5% | 31.7 | 179.7 | 24.47 | 191.6 | 0 |
| control@1729 | 987.3 | 11.5% | 60.3 | 216.3 | 68.50 | 238.6 | 0 |
| geometry@42 | 966.2 | 11.3% | 56.0 | 211.4 | 45.48 | 78.6 | 0 |
| geometry@1337 | 994.5 | 11.6% | 78.7 | 236.8 | 70.70 | 115.1 | 0 |
| geometry@2718 | 1,015.6 | 11.8% | 74.5 | 231.6 | 64.55 | 111.9 | 0 |
| geometry@7 | 984.7 | 11.5% | 38.1 | 188.2 | 19.39 | 60.9 | 0 |
| geometry@13 | 968.4 | 11.3% | 43.3 | 196.0 | 27.23 | 70.9 | 0 |
| geometry@101 | 976.8 | 11.4% | 58.9 | 213.9 | 50.85 | 93.0 | 0 |
| geometry@271 | 1,006.5 | 11.7% | 53.9 | 209.6 | 39.57 | 73.1 | 0 |
| geometry@314 | 1,000.2 | 11.7% | 51.9 | 203.1 | 35.68 | 73.7 | 0 |
| geometry@577 | 995.7 | 11.6% | 42.5 | 192.5 | 23.14 | 54.5 | 0 |
| geometry@863 | 955.9 | 11.1% | 40.3 | 191.4 | 25.13 | 55.7 | 0 |
| geometry@1024 | 989.9 | 11.5% | 34.5 | 185.0 | 15.05 | 52.5 | 0 |
| geometry@1729 | 939.5 | 11.0% | 63.1 | 219.9 | 53.00 | 96.2 | 0 |

Training-target contamination: 476 of 11,434 night rows read above 1 MW (max 439.3 MW); dropped from fit: True.
