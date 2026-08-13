# ABL-338 — solar non-negativity and solar geometry: held-out A/B

Generated 2026-08-13T14:24:30 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-07-13 .. 2026-08-11**, training from 2025-11-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. Night hours are reported in MW, never as a percentage: their denominator is ~0.

## AT — xgboost, source `energy_renewable`

n_train 5,423 · n_holdout 720 (daylight 448 / shoulder 92 / night 180) · incumbent version 20260112_165237

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 573.1 | 25.6% | 1.1 | 4.5 | 0.00 | 0.0 | 0 |
| control@42 | 299.4 | 13.4% | 4.9 | 0.7 | -4.01 | -1.9 | 243 |
| control@1337 | 284.6 | 12.7% | 3.4 | 6.5 | 0.18 | 1.0 | 32 |
| control@2718 | 294.4 | 13.1% | 9.5 | 13.1 | 3.38 | 12.5 | 14 |
| control@7 | 286.7 | 12.8% | 3.3 | 6.4 | 1.12 | 5.3 | 7 |
| control@13 | 288.9 | 12.9% | 4.7 | 7.5 | 0.08 | 5.3 | 125 |
| control@101 | 297.1 | 13.3% | 4.6 | 7.9 | 2.32 | 2.8 | 0 |
| control@271 | 295.0 | 13.2% | 9.0 | 8.4 | -1.39 | 5.6 | 202 |
| control@314 | 288.0 | 12.9% | 3.6 | 4.7 | -0.90 | 5.3 | 167 |
| control@577 | 292.1 | 13.0% | 6.2 | 9.9 | 3.50 | 5.9 | 0 |
| control@863 | 294.1 | 13.1% | 4.5 | 3.0 | -2.81 | 1.3 | 219 |
| control@1024 | 292.5 | 13.1% | 3.5 | 6.3 | 1.46 | 7.5 | 51 |
| control@1729 | 288.4 | 12.9% | 22.1 | 25.8 | 13.28 | 39.9 | 0 |
| geometry@42 | 290.0 | 12.9% | 2.4 | 3.3 | 1.21 | 1.4 | 0 |
| geometry@1337 | 295.3 | 13.2% | 2.3 | 4.9 | 1.82 | 1.9 | 0 |
| geometry@2718 | 296.8 | 13.2% | 3.8 | 5.4 | 0.80 | 2.1 | 0 |
| geometry@7 | 290.3 | 13.0% | 2.1 | 2.0 | -0.09 | 2.0 | 118 |
| geometry@13 | 289.4 | 12.9% | 3.5 | 6.0 | 4.21 | 4.2 | 0 |
| geometry@101 | 283.4 | 12.6% | 2.6 | 1.5 | -0.49 | 0.7 | 177 |
| geometry@271 | 292.6 | 13.1% | 2.7 | 3.2 | 1.46 | 1.5 | 0 |
| geometry@314 | 288.7 | 12.9% | 2.4 | 2.8 | 0.46 | 1.5 | 35 |
| geometry@577 | 287.2 | 12.8% | 3.7 | 0.2 | -1.97 | -1.9 | 246 |
| geometry@863 | 284.2 | 12.7% | 3.0 | 2.1 | 0.78 | 4.7 | 46 |
| geometry@1024 | 292.6 | 13.1% | 5.3 | 7.9 | 3.52 | 10.6 | 137 |
| geometry@1729 | 293.8 | 13.1% | 2.3 | 5.2 | 1.95 | 2.1 | 0 |

Training-target contamination: 0 of 2,094 night rows read above 1 MW (max 0.0 MW); dropped from fit: True.

## BE — xgboost, source `energy_renewable`

n_train 5,733 · n_holdout 720 (daylight 465 / shoulder 88 / night 167) · incumbent version 20260201_222022

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,090.1 | 32.9% | 3.3 | 5.1 | 0.00 | 0.0 | 0 |
| control@42 | 590.2 | 17.8% | 5.6 | 7.1 | 1.91 | 4.0 | 21 |
| control@1337 | 584.5 | 17.6% | 8.0 | 9.8 | 2.93 | 5.9 | 2 |
| control@2718 | 573.3 | 17.3% | 14.9 | 16.7 | 10.96 | 11.2 | 0 |
| control@7 | 577.7 | 17.4% | 9.0 | 10.8 | 6.26 | 6.5 | 0 |
| control@13 | 589.5 | 17.8% | 11.5 | 13.2 | 4.69 | 5.8 | 0 |
| control@101 | 588.8 | 17.8% | 18.0 | 19.1 | 14.85 | 126.2 | 99 |
| control@271 | 598.6 | 18.0% | 11.4 | 13.1 | 5.44 | 6.7 | 0 |
| control@314 | 588.6 | 17.7% | 4.8 | 5.7 | 0.31 | 5.0 | 58 |
| control@577 | 587.3 | 17.7% | 6.7 | 8.5 | 4.06 | 6.4 | 0 |
| control@863 | 597.7 | 18.0% | 6.6 | 8.3 | 3.11 | 4.0 | 0 |
| control@1024 | 583.5 | 17.6% | 10.5 | 12.3 | 7.11 | 7.7 | 0 |
| control@1729 | 605.4 | 18.3% | 6.5 | 7.8 | 1.02 | 3.5 | 69 |
| geometry@42 | 564.7 | 17.0% | 7.0 | 8.5 | 5.66 | 6.3 | 0 |
| geometry@1337 | 572.1 | 17.2% | 13.3 | 15.1 | 10.58 | 10.7 | 0 |
| geometry@2718 | 560.4 | 16.9% | 8.5 | 10.3 | 4.62 | 5.2 | 0 |
| geometry@7 | 552.9 | 16.7% | 8.0 | 9.7 | 6.23 | 6.2 | 0 |
| geometry@13 | 576.6 | 17.4% | 7.2 | 8.8 | 4.72 | 5.0 | 0 |
| geometry@101 | 557.2 | 16.8% | 3.4 | 4.6 | 2.03 | 4.0 | 5 |
| geometry@271 | 574.7 | 17.3% | 16.7 | 18.5 | 13.71 | 13.7 | 0 |
| geometry@314 | 580.1 | 17.5% | 8.1 | 9.6 | 6.69 | 6.7 | 0 |
| geometry@577 | 581.1 | 17.5% | 7.7 | 9.5 | 6.37 | 6.8 | 0 |
| geometry@863 | 581.4 | 17.5% | 7.7 | 9.1 | 7.40 | 7.4 | 0 |
| geometry@1024 | 575.3 | 17.3% | 3.2 | 3.8 | 1.41 | 2.3 | 16 |
| geometry@1729 | 590.9 | 17.8% | 4.2 | 4.5 | 3.43 | 5.8 | 10 |

Training-target contamination: 0 of 2,181 night rows read above 1 MW (max 0.0 MW); dropped from fit: True.

## DE — xgboost, source `energy_renewable`

n_train 5,527 · n_holdout 720 (daylight 457 / shoulder 99 / night 164) · incumbent version 20260223_193822

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 6,066.6 | 23.0% | 33.5 | 66.2 | 0.00 | 0.0 | 0 |
| control@42 | 3,065.3 | 11.6% | 146.4 | 158.9 | -15.53 | 94.3 | 168 |
| control@1337 | 3,062.7 | 11.6% | 231.9 | 252.3 | 2.08 | 69.0 | 105 |
| control@2718 | 3,165.8 | 12.0% | 213.6 | 242.6 | 2.25 | 65.7 | 84 |
| control@7 | 3,067.6 | 11.6% | 80.1 | 92.3 | -22.11 | 22.6 | 177 |
| control@13 | 3,004.7 | 11.4% | 170.8 | 185.1 | -19.51 | 25.1 | 155 |
| control@101 | 3,104.9 | 11.8% | 70.8 | 70.0 | -32.82 | 59.3 | 188 |
| control@271 | 3,091.2 | 11.7% | 270.0 | 301.5 | 15.35 | 63.2 | 30 |
| control@314 | 3,070.3 | 11.6% | 117.4 | 118.6 | -37.34 | 51.7 | 183 |
| control@577 | 3,065.6 | 11.6% | 62.6 | 78.9 | -10.62 | 93.7 | 107 |
| control@863 | 3,034.6 | 11.5% | 127.5 | 131.2 | -18.76 | 65.4 | 156 |
| control@1024 | 3,108.3 | 11.8% | 106.2 | 126.4 | -5.40 | 36.7 | 103 |
| control@1729 | 3,076.6 | 11.7% | 259.2 | 289.9 | 12.49 | 72.5 | 47 |
| geometry@42 | 3,098.8 | 11.8% | 95.7 | 123.9 | 10.98 | 66.8 | 43 |
| geometry@1337 | 3,075.0 | 11.7% | 26.6 | 43.1 | 2.75 | 92.2 | 73 |
| geometry@2718 | 3,166.7 | 12.0% | 89.2 | 105.2 | -23.12 | 63.2 | 121 |
| geometry@7 | 3,062.8 | 11.6% | 34.8 | 53.5 | -2.65 | 61.0 | 103 |
| geometry@13 | 3,058.3 | 11.6% | 62.1 | 53.2 | -26.70 | 17.0 | 174 |
| geometry@101 | 3,064.8 | 11.6% | 33.0 | 46.3 | 6.01 | 50.6 | 72 |
| geometry@271 | 3,146.7 | 11.9% | 51.8 | 52.4 | -19.95 | 37.7 | 122 |
| geometry@314 | 3,100.2 | 11.8% | 58.5 | 74.4 | -11.17 | 73.9 | 125 |
| geometry@577 | 3,061.7 | 11.6% | 42.9 | 54.3 | -3.09 | 52.4 | 100 |
| geometry@863 | 2,961.9 | 11.2% | 107.1 | 120.3 | -12.01 | 64.6 | 84 |
| geometry@1024 | 3,096.3 | 11.7% | 50.2 | 72.1 | 6.59 | 58.0 | 66 |
| geometry@1729 | 3,128.9 | 11.9% | 113.1 | 144.9 | 23.87 | 89.5 | 43 |

Training-target contamination: 4 of 2,087 night rows read above 1 MW (max 1.7 MW); dropped from fit: True.

## FR — xgboost, source `energy_renewable`

n_train 5,545 · n_holdout 720 (daylight 435 / shoulder 99 / night 186) · incumbent version 20260201_222014

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 2,368.6 | 22.3% | 50.8 | 270.4 | 34.88 | 321.5 | 0 |
| control@42 | 1,752.5 | 16.5% | 36.5 | 256.7 | 26.78 | 296.8 | 26 |
| control@1337 | 1,597.9 | 15.0% | 34.6 | 251.7 | 27.37 | 290.2 | 10 |
| control@2718 | 1,663.1 | 15.6% | 34.2 | 255.2 | 26.24 | 282.6 | 53 |
| control@7 | 1,603.2 | 15.1% | 32.1 | 253.0 | 25.76 | 283.3 | 27 |
| control@13 | 1,653.0 | 15.5% | 35.2 | 259.7 | 30.86 | 268.9 | 0 |
| control@101 | 1,623.3 | 15.3% | 47.3 | 270.4 | 59.82 | 405.2 | 94 |
| control@271 | 1,664.4 | 15.6% | 37.2 | 262.3 | 31.00 | 329.9 | 42 |
| control@314 | 1,670.0 | 15.7% | 46.8 | 265.0 | 34.51 | 320.3 | 38 |
| control@577 | 1,678.9 | 15.8% | 37.3 | 262.3 | 30.57 | 275.9 | 2 |
| control@863 | 1,750.9 | 16.5% | 33.3 | 262.1 | 30.17 | 307.3 | 7 |
| control@1024 | 1,658.0 | 15.6% | 25.3 | 252.4 | 28.89 | 289.6 | 18 |
| control@1729 | 1,650.5 | 15.5% | 37.7 | 260.5 | 28.27 | 261.6 | 6 |
| geometry@42 | 1,612.3 | 15.2% | 31.9 | 242.7 | 17.12 | 200.6 | 10 |
| geometry@1337 | 1,635.6 | 15.4% | 33.8 | 244.3 | 15.34 | 176.0 | 4 |
| geometry@2718 | 1,614.5 | 15.2% | 54.2 | 272.2 | 15.47 | 175.3 | 112 |
| geometry@7 | 1,490.1 | 14.0% | 29.2 | 235.1 | 19.70 | 204.6 | 7 |
| geometry@13 | 1,606.6 | 15.1% | 33.5 | 238.9 | 12.89 | 157.4 | 78 |
| geometry@101 | 1,503.8 | 14.1% | 30.4 | 238.8 | 20.42 | 199.2 | 33 |
| geometry@271 | 1,651.8 | 15.5% | 36.7 | 242.9 | 13.71 | 193.5 | 25 |
| geometry@314 | 1,569.5 | 14.8% | 33.6 | 241.6 | 16.04 | 196.2 | 0 |
| geometry@577 | 1,606.5 | 15.1% | 32.9 | 236.0 | 14.08 | 205.6 | 106 |
| geometry@863 | 1,540.2 | 14.5% | 27.1 | 236.9 | 13.58 | 200.8 | 127 |
| geometry@1024 | 1,574.6 | 14.8% | 32.5 | 245.1 | 17.55 | 171.1 | 57 |
| geometry@1729 | 1,540.4 | 14.5% | 33.2 | 239.7 | 16.77 | 190.1 | 37 |

Training-target contamination: 183 of 2,270 night rows read above 1 MW (max 285.9 MW); dropped from fit: True.
