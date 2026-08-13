# ABL-338 — solar non-negativity and solar geometry: held-out A/B

Generated 2026-08-13T13:46:28 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-05-14 .. 2026-06-12**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. Night hours are reported in MW, never as a percentage: their denominator is ~0.

## AT — catboost, source `energy_renewable`

n_train 3,983 · n_holdout 720 (daylight 472 / shoulder 68 / night 180) · incumbent version 20260112_165237

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 858.8 | 43.3% | 1.2 | 0.3 | 0.00 | 0.0 | 0 |
| control@42 | 483.7 | 24.4% | 13.0 | 8.5 | 6.60 | 49.1 | 96 |
| control@1337 | 495.7 | 25.0% | 14.0 | 14.3 | 8.32 | 43.9 | 54 |
| control@2718 | 438.5 | 22.1% | 14.5 | 10.0 | 7.12 | 52.1 | 111 |
| control@7 | 504.2 | 25.4% | 30.2 | 31.6 | 23.70 | 67.5 | 0 |
| control@13 | 473.7 | 23.9% | 21.4 | 19.8 | 9.89 | 72.6 | 66 |
| control@101 | 532.1 | 26.8% | 45.1 | 46.6 | 35.50 | 72.3 | 0 |
| control@271 | 481.1 | 24.2% | 18.1 | 18.0 | 15.07 | 62.1 | 47 |
| control@314 | 489.1 | 24.6% | 18.3 | 18.8 | 12.29 | 49.9 | 28 |
| control@577 | 490.4 | 24.7% | 18.7 | 16.8 | 13.52 | 45.1 | 28 |
| control@863 | 482.7 | 24.3% | 21.3 | 21.0 | 13.26 | 74.3 | 64 |
| control@1024 | 476.6 | 24.0% | 19.7 | 15.8 | 11.42 | 60.1 | 85 |
| control@1729 | 472.9 | 23.8% | 20.6 | 21.6 | 15.92 | 61.2 | 24 |
| geometry@42 | 457.1 | 23.0% | 26.7 | 27.3 | 22.58 | 66.0 | 21 |
| geometry@1337 | 463.5 | 23.3% | 20.5 | 19.8 | 9.36 | 57.1 | 64 |
| geometry@2718 | 472.1 | 23.8% | 16.4 | 16.0 | 8.57 | 58.0 | 66 |
| geometry@7 | 476.8 | 24.0% | 24.9 | 25.0 | 16.66 | 68.9 | 27 |
| geometry@13 | 475.3 | 23.9% | 25.5 | 24.9 | 21.54 | 69.9 | 21 |
| geometry@101 | 463.1 | 23.3% | 27.1 | 28.0 | 21.55 | 67.4 | 15 |
| geometry@271 | 458.0 | 23.1% | 17.0 | 16.3 | 9.81 | 36.0 | 41 |
| geometry@314 | 496.7 | 25.0% | 29.5 | 30.9 | 20.31 | 49.2 | 2 |
| geometry@577 | 488.0 | 24.6% | 32.1 | 33.5 | 26.78 | 70.4 | 4 |
| geometry@863 | 515.2 | 25.9% | 35.9 | 37.3 | 31.16 | 67.7 | 0 |
| geometry@1024 | 512.9 | 25.8% | 23.6 | 24.9 | 19.15 | 60.0 | 13 |
| geometry@1729 | 474.1 | 23.9% | 23.5 | 24.7 | 18.46 | 63.5 | 19 |

Training-target contamination: 0 of 1,734 night rows read above 1 MW (max 0.0 MW); dropped from fit: True.

## BE — catboost, source `energy_renewable`

n_train 20,613 · n_holdout 720 (daylight 471 / shoulder 103 / night 146) · incumbent version 20260201_222022

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,359.4 | 45.8% | 1.9 | 1.0 | 0.00 | 0.0 | 0 |
| control@42 | 529.5 | 17.8% | 23.2 | 20.4 | -0.53 | 23.8 | 105 |
| control@1337 | 548.3 | 18.5% | 19.2 | 14.5 | -1.25 | 36.3 | 120 |
| control@2718 | 555.6 | 18.7% | 24.7 | -6.5 | -12.32 | 50.6 | 159 |
| control@7 | 546.7 | 18.4% | 16.4 | 6.5 | -4.69 | 33.4 | 117 |
| control@13 | 531.9 | 17.9% | 21.8 | 12.0 | -4.31 | 56.4 | 114 |
| control@101 | 529.9 | 17.8% | 17.7 | 4.6 | -4.60 | 22.6 | 127 |
| control@271 | 546.4 | 18.4% | 17.4 | 13.9 | -3.04 | 30.4 | 119 |
| control@314 | 557.4 | 18.8% | 20.7 | 3.5 | -6.86 | 32.6 | 135 |
| control@577 | 552.0 | 18.6% | 21.0 | 10.2 | -1.04 | 40.7 | 103 |
| control@863 | 534.1 | 18.0% | 15.3 | 9.1 | -1.82 | 32.2 | 110 |
| control@1024 | 566.7 | 19.1% | 20.4 | -0.4 | -5.52 | 51.1 | 148 |
| control@1729 | 540.7 | 18.2% | 12.6 | 8.9 | 0.48 | 30.7 | 100 |
| geometry@42 | 554.2 | 18.7% | 17.3 | -7.2 | -11.61 | 36.4 | 169 |
| geometry@1337 | 529.4 | 17.8% | 16.8 | 9.5 | -1.93 | 39.5 | 130 |
| geometry@2718 | 554.2 | 18.7% | 15.4 | 13.4 | -2.22 | 19.9 | 104 |
| geometry@7 | 535.8 | 18.0% | 13.2 | 9.1 | -3.68 | 32.7 | 125 |
| geometry@13 | 545.6 | 18.4% | 17.2 | 1.7 | -11.16 | 35.3 | 152 |
| geometry@101 | 544.6 | 18.3% | 16.4 | 10.3 | -7.56 | 19.6 | 149 |
| geometry@271 | 537.5 | 18.1% | 11.7 | 1.3 | -7.08 | 37.2 | 163 |
| geometry@314 | 546.2 | 18.4% | 16.4 | 8.1 | -6.15 | 21.6 | 141 |
| geometry@577 | 520.8 | 17.5% | 12.0 | 7.2 | -3.28 | 26.2 | 118 |
| geometry@863 | 541.4 | 18.2% | 15.5 | 3.7 | -9.33 | 23.2 | 155 |
| geometry@1024 | 533.4 | 18.0% | 18.5 | 17.0 | -6.90 | 33.8 | 129 |
| geometry@1729 | 554.7 | 18.7% | 17.2 | 13.1 | -2.33 | 39.3 | 112 |

Training-target contamination: 0 of 7,756 night rows read above 1 MW (max 0.1 MW); dropped from fit: True.

## DE — catboost, source `energy_renewable`

n_train 4,087 · n_holdout 720 (daylight 477 / shoulder 99 / night 144) · incumbent version 20260223_193822

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 6,618.6 | 29.4% | 13.8 | 14.8 | 0.00 | 0.0 | 0 |
| control@42 | 3,034.0 | 13.5% | 225.4 | 226.4 | 170.39 | 515.3 | 34 |
| control@1337 | 3,208.7 | 14.2% | 199.9 | 96.2 | 12.85 | 447.8 | 103 |
| control@2718 | 3,064.9 | 13.6% | 197.1 | 191.3 | 127.75 | 386.2 | 53 |
| control@7 | 3,101.0 | 13.8% | 254.4 | 279.3 | 231.48 | 410.5 | 1 |
| control@13 | 2,975.0 | 13.2% | 187.8 | 101.5 | 26.86 | 392.3 | 90 |
| control@101 | 2,938.5 | 13.0% | 131.0 | 88.7 | 39.60 | 438.4 | 85 |
| control@271 | 3,000.3 | 13.3% | 162.1 | 157.0 | 91.13 | 309.6 | 49 |
| control@314 | 3,020.8 | 13.4% | 160.3 | 32.4 | -60.63 | 211.1 | 137 |
| control@577 | 2,970.1 | 13.2% | 140.5 | 96.6 | 70.17 | 419.6 | 65 |
| control@863 | 3,153.3 | 14.0% | 176.9 | 133.5 | 52.50 | 341.2 | 81 |
| control@1024 | 3,094.9 | 13.7% | 174.9 | 150.8 | 80.97 | 386.0 | 61 |
| control@1729 | 3,238.8 | 14.4% | 231.5 | 238.2 | 202.69 | 479.4 | 17 |
| geometry@42 | 3,250.6 | 14.4% | 156.0 | 152.9 | 34.57 | 371.1 | 65 |
| geometry@1337 | 3,189.0 | 14.1% | 143.8 | 125.4 | 16.98 | 406.1 | 92 |
| geometry@2718 | 3,451.1 | 15.3% | 120.3 | 119.3 | 27.01 | 285.1 | 92 |
| geometry@7 | 3,266.6 | 14.5% | 127.4 | 63.1 | 1.21 | 212.3 | 98 |
| geometry@13 | 3,166.1 | 14.0% | 251.0 | 216.0 | 102.23 | 533.9 | 88 |
| geometry@101 | 3,262.2 | 14.5% | 182.2 | 117.2 | 35.76 | 467.4 | 84 |
| geometry@271 | 3,186.1 | 14.1% | 149.0 | -4.5 | -62.81 | 492.4 | 142 |
| geometry@314 | 3,718.2 | 16.5% | 339.4 | 348.9 | 329.82 | 749.7 | 15 |
| geometry@577 | 3,157.5 | 14.0% | 182.4 | 189.0 | 111.60 | 380.6 | 32 |
| geometry@863 | 3,647.9 | 16.2% | 186.6 | 133.7 | 55.57 | 422.8 | 78 |
| geometry@1024 | 3,581.6 | 15.9% | 427.4 | 453.9 | 489.33 | 790.0 | 0 |
| geometry@1729 | 3,248.1 | 14.4% | 217.3 | 235.9 | 126.74 | 448.0 | 41 |

Training-target contamination: 4 of 1,813 night rows read above 1 MW (max 1.7 MW); dropped from fit: True.

## FR — catboost, source `energy_renewable`

n_train 28,960 · n_holdout 720 (daylight 457 / shoulder 83 / night 180) · incumbent version 20260201_222014

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 2,152.6 | 25.1% | 29.2 | 146.7 | 5.38 | 251.0 | 0 |
| control@42 | 1,013.7 | 11.8% | 98.6 | 203.5 | 46.02 | 141.2 | 20 |
| control@1337 | 992.7 | 11.6% | 80.3 | 180.0 | 49.42 | 150.1 | 21 |
| control@2718 | 1,026.1 | 12.0% | 76.0 | 182.7 | 41.25 | 136.2 | 25 |
| control@7 | 997.0 | 11.6% | 83.8 | 193.8 | 57.80 | 176.0 | 18 |
| control@13 | 998.0 | 11.6% | 80.9 | 174.3 | 49.61 | 152.9 | 17 |
| control@101 | 1,000.7 | 11.7% | 72.9 | 167.7 | 47.85 | 161.1 | 28 |
| control@271 | 1,013.6 | 11.8% | 89.6 | 199.2 | 47.74 | 170.0 | 22 |
| control@314 | 1,045.3 | 12.2% | 93.6 | 215.3 | 50.46 | 172.8 | 10 |
| control@577 | 990.7 | 11.6% | 102.7 | 201.5 | 57.48 | 218.4 | 15 |
| control@863 | 1,017.8 | 11.9% | 94.9 | 196.2 | 54.82 | 163.7 | 13 |
| control@1024 | 1,024.3 | 11.9% | 74.7 | 197.1 | 42.04 | 160.8 | 29 |
| control@1729 | 1,025.4 | 12.0% | 82.9 | 183.3 | 52.66 | 200.2 | 37 |
| geometry@42 | 950.7 | 11.1% | 109.6 | 235.9 | 26.17 | 153.8 | 33 |
| geometry@1337 | 993.1 | 11.6% | 100.4 | 222.3 | 32.83 | 144.6 | 32 |
| geometry@2718 | 985.8 | 11.5% | 96.4 | 194.1 | 24.50 | 121.6 | 42 |
| geometry@7 | 993.2 | 11.6% | 103.6 | 204.8 | 25.69 | 138.0 | 42 |
| geometry@13 | 965.2 | 11.3% | 106.1 | 227.4 | 28.28 | 105.4 | 41 |
| geometry@101 | 981.1 | 11.4% | 92.6 | 231.5 | 24.92 | 154.8 | 45 |
| geometry@271 | 997.5 | 11.6% | 112.1 | 211.1 | 30.30 | 123.4 | 38 |
| geometry@314 | 988.9 | 11.5% | 108.1 | 229.7 | 29.87 | 117.9 | 44 |
| geometry@577 | 1,014.8 | 11.8% | 102.2 | 214.0 | 22.82 | 130.2 | 55 |
| geometry@863 | 1,024.3 | 11.9% | 117.1 | 235.2 | 44.42 | 116.3 | 8 |
| geometry@1024 | 982.0 | 11.5% | 96.1 | 185.6 | 34.82 | 165.2 | 37 |
| geometry@1729 | 985.3 | 11.5% | 96.5 | 219.3 | 27.85 | 111.6 | 29 |

Training-target contamination: 476 of 11,434 night rows read above 1 MW (max 439.3 MW); dropped from fit: True.
