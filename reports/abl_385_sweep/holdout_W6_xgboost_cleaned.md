# ABL-338 — solar non-negativity and solar geometry: held-out A/B

Generated 2026-08-13T14:08:50 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-07-13 .. 2026-08-11**, training from 2023-01-01 up to the holdout start.

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

n_train 22,053 · n_holdout 720 (daylight 465 / shoulder 88 / night 167) · incumbent version 20260201_222022

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,090.1 | 32.9% | 3.3 | 5.1 | 0.00 | 0.0 | 0 |
| control@42 | 560.4 | 16.9% | 3.9 | 5.2 | 2.56 | 3.0 | 10 |
| control@1337 | 569.4 | 17.2% | 10.8 | 12.5 | 8.02 | 8.1 | 0 |
| control@2718 | 546.7 | 16.5% | 5.5 | 6.8 | 3.68 | 4.7 | 1 |
| control@7 | 558.7 | 16.8% | 8.7 | 10.1 | 5.95 | 6.5 | 1 |
| control@13 | 558.4 | 16.8% | 7.0 | 8.4 | 3.85 | 4.4 | 0 |
| control@101 | 572.5 | 17.3% | 6.1 | 7.6 | 3.75 | 6.8 | 0 |
| control@271 | 573.0 | 17.3% | 5.1 | 2.9 | -0.78 | 9.0 | 112 |
| control@314 | 570.9 | 17.2% | 11.1 | 12.8 | 8.92 | 9.7 | 0 |
| control@577 | 579.2 | 17.5% | 5.2 | 6.7 | 1.94 | 4.2 | 4 |
| control@863 | 555.7 | 16.8% | 4.9 | 5.7 | 0.82 | 3.2 | 40 |
| control@1024 | 570.1 | 17.2% | 4.7 | 4.9 | 1.57 | 17.4 | 49 |
| control@1729 | 572.8 | 17.3% | 4.4 | 4.5 | 1.01 | 5.3 | 26 |
| geometry@42 | 547.3 | 16.5% | 3.2 | 1.3 | 0.34 | 4.6 | 97 |
| geometry@1337 | 549.2 | 16.6% | 2.6 | 3.8 | 1.72 | 3.1 | 14 |
| geometry@2718 | 543.6 | 16.4% | 3.8 | 5.1 | 2.33 | 5.9 | 15 |
| geometry@7 | 549.6 | 16.6% | 5.9 | 7.3 | 4.69 | 4.9 | 0 |
| geometry@13 | 542.0 | 16.3% | 6.5 | 8.0 | 5.10 | 5.1 | 0 |
| geometry@101 | 530.0 | 16.0% | 8.9 | 10.3 | 7.57 | 7.9 | 0 |
| geometry@271 | 556.9 | 16.8% | 3.1 | 2.3 | 0.42 | 4.3 | 62 |
| geometry@314 | 544.0 | 16.4% | 7.0 | 8.3 | 5.93 | 6.0 | 0 |
| geometry@577 | 545.0 | 16.4% | 5.7 | 7.2 | 4.74 | 4.8 | 0 |
| geometry@863 | 550.3 | 16.6% | 4.6 | 6.2 | 3.06 | 3.2 | 0 |
| geometry@1024 | 554.2 | 16.7% | 3.6 | 4.6 | 2.59 | 16.9 | 11 |
| geometry@1729 | 554.4 | 16.7% | 3.3 | 2.7 | -0.34 | 1.3 | 71 |

Training-target contamination: 0 of 8,026 night rows read above 1 MW (max 0.1 MW); dropped from fit: True.

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

n_train 30,359 · n_holdout 720 (daylight 435 / shoulder 99 / night 186) · incumbent version 20260201_222014

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 2,368.6 | 22.3% | 50.8 | 270.4 | 34.88 | 321.5 | 0 |
| control@42 | 1,481.6 | 13.9% | 25.8 | 249.3 | 35.28 | 290.8 | 0 |
| control@1337 | 1,433.1 | 13.5% | 30.5 | 242.2 | 26.07 | 260.5 | 0 |
| control@2718 | 1,526.4 | 14.3% | 33.3 | 237.9 | 24.39 | 275.6 | 41 |
| control@7 | 1,455.9 | 13.7% | 30.3 | 242.0 | 27.51 | 282.2 | 1 |
| control@13 | 1,443.9 | 13.6% | 29.0 | 244.3 | 29.96 | 292.6 | 0 |
| control@101 | 1,412.3 | 13.3% | 33.4 | 252.9 | 36.60 | 302.9 | 18 |
| control@271 | 1,514.3 | 14.2% | 29.6 | 242.1 | 22.68 | 288.3 | 62 |
| control@314 | 1,479.9 | 13.9% | 28.2 | 241.4 | 25.59 | 279.8 | 46 |
| control@577 | 1,458.5 | 13.7% | 26.8 | 242.8 | 31.49 | 248.3 | 0 |
| control@863 | 1,462.8 | 13.7% | 26.8 | 241.7 | 28.74 | 280.8 | 0 |
| control@1024 | 1,435.6 | 13.5% | 32.6 | 241.2 | 23.20 | 280.8 | 0 |
| control@1729 | 1,502.5 | 14.1% | 30.3 | 234.2 | 23.81 | 255.4 | 12 |
| geometry@42 | 1,323.4 | 12.4% | 28.0 | 247.3 | 12.71 | 187.9 | 0 |
| geometry@1337 | 1,375.2 | 12.9% | 32.0 | 243.3 | 6.73 | 150.7 | 40 |
| geometry@2718 | 1,442.8 | 13.6% | 29.2 | 249.5 | 9.35 | 127.1 | 6 |
| geometry@7 | 1,365.6 | 12.8% | 31.7 | 252.9 | 13.47 | 143.7 | 5 |
| geometry@13 | 1,422.1 | 13.4% | 28.3 | 253.1 | 16.18 | 176.1 | 0 |
| geometry@101 | 1,359.8 | 12.8% | 31.2 | 252.3 | 16.73 | 124.0 | 0 |
| geometry@271 | 1,412.9 | 13.3% | 28.7 | 246.5 | 8.67 | 179.9 | 23 |
| geometry@314 | 1,412.0 | 13.3% | 28.1 | 250.8 | 12.16 | 187.1 | 0 |
| geometry@577 | 1,440.6 | 13.5% | 28.7 | 252.3 | 14.69 | 161.3 | 0 |
| geometry@863 | 1,408.0 | 13.2% | 29.8 | 250.7 | 13.76 | 171.2 | 0 |
| geometry@1024 | 1,374.2 | 12.9% | 31.4 | 259.2 | 20.23 | 138.8 | 0 |
| geometry@1729 | 1,444.4 | 13.6% | 28.1 | 249.8 | 12.25 | 164.4 | 3 |

Training-target contamination: 517 of 11,794 night rows read above 1 MW (max 439.3 MW); dropped from fit: True.
