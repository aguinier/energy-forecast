# ABL-338 — solar non-negativity and solar geometry: held-out A/B

Generated 2026-08-13T13:28:21 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-03-15 .. 2026-04-13**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. Night hours are reported in MW, never as a percentage: their denominator is ~0.

## AT — xgboost, source `energy_renewable`

n_train 2,543 · n_holdout 720 (daylight 374 / shoulder 82 / night 264) · incumbent version 20260112_165237

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 873.5 | 62.7% | 0.3 | 0.0 | 0.00 | 0.0 | 0 |
| control@42 | 986.1 | 70.8% | 6.0 | -5.6 | -6.33 | -2.5 | 355 |
| control@1337 | 968.6 | 69.6% | 3.3 | -2.9 | -3.59 | 0.5 | 341 |
| control@2718 | 971.6 | 69.8% | 5.3 | 2.7 | 2.21 | 8.2 | 180 |
| control@7 | 992.8 | 71.3% | 2.7 | -0.7 | -0.87 | 18.0 | 301 |
| control@13 | 981.9 | 70.5% | 4.0 | 2.7 | 2.56 | 9.2 | 169 |
| control@101 | 992.4 | 71.3% | 4.1 | 0.4 | -3.11 | 4.5 | 316 |
| control@271 | 991.8 | 71.2% | 4.6 | 2.2 | -2.28 | 6.8 | 299 |
| control@314 | 974.9 | 70.0% | 2.5 | -0.1 | -1.47 | 7.4 | 288 |
| control@577 | 1,006.5 | 72.3% | 3.0 | -0.5 | -1.11 | 6.6 | 257 |
| control@863 | 984.6 | 70.7% | 3.2 | -1.8 | -3.19 | 5.6 | 325 |
| control@1024 | 981.9 | 70.5% | 2.8 | 0.1 | 0.61 | 7.5 | 189 |
| control@1729 | 996.3 | 71.6% | 3.2 | 0.2 | 0.92 | 6.1 | 193 |
| geometry@42 | 962.9 | 69.2% | 1.5 | -0.6 | -1.19 | 1.3 | 324 |
| geometry@1337 | 937.0 | 67.3% | 1.2 | -0.4 | -0.97 | 1.5 | 327 |
| geometry@2718 | 963.1 | 69.2% | 3.6 | 0.0 | 0.09 | 8.1 | 175 |
| geometry@7 | 970.4 | 69.7% | 2.3 | -0.9 | -1.65 | 5.6 | 317 |
| geometry@13 | 980.4 | 70.4% | 2.1 | 0.1 | -0.44 | 3.0 | 170 |
| geometry@101 | 954.7 | 68.6% | 1.1 | -0.2 | -0.82 | 2.0 | 308 |
| geometry@271 | 953.0 | 68.5% | 1.5 | -1.0 | -1.49 | -0.8 | 343 |
| geometry@314 | 945.4 | 67.9% | 2.2 | 0.9 | -0.15 | 10.6 | 289 |
| geometry@577 | 961.9 | 69.1% | 1.3 | 1.2 | 0.68 | 6.8 | 137 |
| geometry@863 | 949.7 | 68.2% | 2.8 | -0.8 | -1.64 | 10.0 | 308 |
| geometry@1024 | 943.9 | 67.8% | 1.8 | 0.2 | -0.76 | 3.4 | 299 |
| geometry@1729 | 963.0 | 69.2% | 3.6 | -3.0 | -3.62 | 1.7 | 338 |

Training-target contamination: 0 of 1,258 night rows read above 1 MW (max 0.0 MW); dropped from fit: True.

## BE — xgboost, source `energy_renewable`

n_train 19,173 · n_holdout 720 (daylight 375 / shoulder 83 / night 262) · incumbent version 20260201_222022

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,528.4 | 48.7% | 4.4 | 0.8 | 0.00 | 0.0 | 0 |
| control@42 | 655.8 | 20.9% | 10.6 | 10.4 | -1.70 | 5.4 | 188 |
| control@1337 | 648.6 | 20.7% | 11.9 | 5.1 | -2.08 | 7.3 | 194 |
| control@2718 | 647.3 | 20.6% | 14.0 | 2.1 | -3.56 | 8.6 | 201 |
| control@7 | 656.0 | 20.9% | 12.6 | 4.8 | -0.57 | 29.4 | 182 |
| control@13 | 655.4 | 20.9% | 12.2 | 6.5 | -1.64 | 12.6 | 216 |
| control@101 | 667.9 | 21.3% | 14.2 | 9.0 | 2.79 | 21.7 | 98 |
| control@271 | 654.9 | 20.9% | 11.4 | 10.7 | -1.00 | 9.8 | 134 |
| control@314 | 643.5 | 20.5% | 15.2 | 2.5 | -2.31 | 8.7 | 212 |
| control@577 | 668.3 | 21.3% | 15.6 | 9.3 | -1.10 | 38.3 | 151 |
| control@863 | 649.4 | 20.7% | 13.9 | 8.1 | -0.71 | 37.9 | 159 |
| control@1024 | 669.3 | 21.3% | 18.4 | 7.4 | -1.85 | 15.0 | 204 |
| control@1729 | 646.4 | 20.6% | 10.4 | 6.1 | -0.92 | 7.3 | 179 |
| geometry@42 | 661.0 | 21.1% | 7.2 | 4.1 | 0.09 | 5.4 | 142 |
| geometry@1337 | 670.6 | 21.4% | 5.6 | 8.7 | 2.94 | 4.1 | 0 |
| geometry@2718 | 675.8 | 21.5% | 11.0 | 14.5 | 3.46 | 4.5 | 0 |
| geometry@7 | 669.6 | 21.3% | 8.7 | 3.6 | -2.33 | 3.1 | 195 |
| geometry@13 | 663.2 | 21.1% | 6.3 | 5.7 | 0.13 | 3.5 | 141 |
| geometry@101 | 664.4 | 21.2% | 13.4 | -1.8 | 0.32 | 12.8 | 149 |
| geometry@271 | 675.4 | 21.5% | 8.1 | 11.3 | 3.78 | 4.7 | 0 |
| geometry@314 | 665.8 | 21.2% | 7.5 | 2.4 | 0.53 | 16.8 | 148 |
| geometry@577 | 647.5 | 20.6% | 6.4 | 5.1 | 0.86 | 3.9 | 60 |
| geometry@863 | 666.5 | 21.2% | 8.8 | 8.4 | 1.26 | 10.2 | 75 |
| geometry@1024 | 654.2 | 20.9% | 7.7 | 5.1 | -1.07 | 7.8 | 173 |
| geometry@1729 | 641.6 | 20.5% | 4.4 | 5.7 | 0.43 | 12.6 | 131 |

Training-target contamination: 0 of 7,295 night rows read above 1 MW (max 0.1 MW); dropped from fit: True.

## DE — xgboost, source `energy_renewable`

n_train 2,649 · n_holdout 720 (daylight 377 / shoulder 83 / night 260) · incumbent version 20260223_193822

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 8,629.4 | 42.5% | 46.1 | 17.0 | 0.06 | 1.1 | 0 |
| control@42 | 6,992.7 | 34.5% | 91.0 | 131.3 | 2.79 | 21.6 | 122 |
| control@1337 | 7,051.6 | 34.7% | 227.3 | 283.5 | 125.63 | 156.5 | 12 |
| control@2718 | 7,200.1 | 35.5% | 219.5 | 276.3 | 147.52 | 185.5 | 9 |
| control@7 | 6,819.0 | 33.6% | 187.4 | 244.5 | 106.51 | 143.2 | 13 |
| control@13 | 7,177.0 | 35.4% | 219.0 | 274.4 | 150.93 | 172.9 | 0 |
| control@101 | 6,994.3 | 34.5% | 126.5 | 182.3 | 35.73 | 75.4 | 14 |
| control@271 | 7,102.4 | 35.0% | 116.4 | 164.1 | 33.95 | 54.3 | 2 |
| control@314 | 7,164.7 | 35.3% | 164.2 | 215.9 | 92.10 | 121.8 | 20 |
| control@577 | 7,312.8 | 36.0% | 245.1 | 303.8 | 147.13 | 182.5 | 7 |
| control@863 | 7,249.5 | 35.7% | 106.8 | 153.4 | 53.23 | 73.9 | 14 |
| control@1024 | 6,938.3 | 34.2% | 225.0 | 284.8 | 132.98 | 170.9 | 10 |
| control@1729 | 7,090.0 | 34.9% | 187.1 | 248.3 | 103.89 | 136.4 | 5 |
| geometry@42 | 6,647.6 | 32.8% | 99.0 | 152.7 | 73.10 | 93.0 | 6 |
| geometry@1337 | 6,920.7 | 34.1% | 153.8 | 210.2 | 120.55 | 140.5 | 20 |
| geometry@2718 | 6,755.5 | 33.3% | 162.4 | 217.7 | 121.93 | 149.6 | 20 |
| geometry@7 | 7,003.6 | 34.5% | 152.3 | 207.5 | 108.52 | 132.7 | 8 |
| geometry@13 | 6,654.0 | 32.8% | 66.2 | 120.0 | 24.59 | 42.1 | 9 |
| geometry@101 | 6,850.1 | 33.8% | 169.5 | 227.5 | 139.28 | 173.4 | 7 |
| geometry@271 | 6,879.6 | 33.9% | 86.6 | 140.4 | 41.88 | 61.6 | 0 |
| geometry@314 | 6,717.2 | 33.1% | 113.1 | 168.7 | 81.64 | 101.1 | 3 |
| geometry@577 | 6,706.7 | 33.0% | 90.9 | 143.7 | 47.69 | 71.0 | 15 |
| geometry@863 | 6,813.6 | 33.6% | 102.5 | 155.2 | 61.78 | 68.9 | 6 |
| geometry@1024 | 6,839.1 | 33.7% | 225.9 | 283.0 | 190.03 | 226.1 | 18 |
| geometry@1729 | 6,758.1 | 33.3% | 221.0 | 278.7 | 175.81 | 206.8 | 3 |

Training-target contamination: 2 of 1,356 night rows read above 1 MW (max 1.7 MW); dropped from fit: True.

## FR — xgboost, source `energy_renewable`

n_train 27,551 · n_holdout 720 (daylight 370 / shoulder 80 / night 270) · incumbent version 20260201_222014

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 2,690.1 | 30.8% | 73.6 | 184.4 | 14.19 | 259.8 | 0 |
| control@42 | 1,184.9 | 13.5% | 30.0 | 238.2 | 21.94 | 262.1 | 0 |
| control@1337 | 1,156.8 | 13.2% | 36.2 | 224.8 | 12.34 | 226.3 | 124 |
| control@2718 | 1,135.8 | 13.0% | 41.9 | 239.9 | 14.58 | 232.9 | 47 |
| control@7 | 1,175.5 | 13.4% | 33.1 | 231.9 | 19.43 | 265.7 | 0 |
| control@13 | 1,184.1 | 13.5% | 37.8 | 242.8 | 17.51 | 257.1 | 20 |
| control@101 | 1,144.7 | 13.1% | 38.7 | 238.9 | 14.14 | 235.7 | 122 |
| control@271 | 1,162.6 | 13.3% | 35.7 | 231.6 | 15.94 | 253.8 | 35 |
| control@314 | 1,178.6 | 13.5% | 38.8 | 235.0 | 22.08 | 241.9 | 0 |
| control@577 | 1,180.7 | 13.5% | 46.0 | 250.8 | 19.88 | 237.1 | 0 |
| control@863 | 1,113.0 | 12.7% | 35.1 | 236.4 | 16.46 | 250.3 | 22 |
| control@1024 | 1,185.6 | 13.6% | 33.1 | 240.2 | 28.38 | 264.7 | 0 |
| control@1729 | 1,198.1 | 13.7% | 38.8 | 237.3 | 18.51 | 219.9 | 0 |
| geometry@42 | 1,187.1 | 13.6% | 35.5 | 253.7 | 25.75 | 160.9 | 0 |
| geometry@1337 | 1,165.8 | 13.3% | 35.2 | 242.7 | 8.40 | 107.3 | 6 |
| geometry@2718 | 1,185.1 | 13.5% | 38.4 | 245.9 | 8.54 | 104.2 | 66 |
| geometry@7 | 1,168.9 | 13.4% | 33.8 | 238.7 | 7.35 | 85.3 | 0 |
| geometry@13 | 1,146.9 | 13.1% | 41.8 | 258.5 | 18.53 | 113.5 | 0 |
| geometry@101 | 1,104.8 | 12.6% | 38.8 | 241.0 | 8.61 | 81.3 | 33 |
| geometry@271 | 1,230.3 | 14.1% | 36.0 | 254.8 | 24.51 | 151.9 | 0 |
| geometry@314 | 1,156.0 | 13.2% | 37.3 | 234.0 | 7.97 | 137.3 | 6 |
| geometry@577 | 1,150.4 | 13.2% | 35.8 | 239.3 | 12.27 | 118.2 | 0 |
| geometry@863 | 1,110.9 | 12.7% | 35.7 | 232.2 | 5.71 | 101.9 | 41 |
| geometry@1024 | 1,143.2 | 13.1% | 34.3 | 241.8 | 5.74 | 89.0 | 71 |
| geometry@1729 | 1,184.2 | 13.5% | 35.7 | 251.1 | 19.01 | 129.2 | 0 |

Training-target contamination: 445 of 10,942 night rows read above 1 MW (max 439.3 MW); dropped from fit: True.
