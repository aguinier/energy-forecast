# ABL-338 — solar non-negativity and solar geometry: held-out A/B

Generated 2026-08-13T13:18:48 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-02-13 .. 2026-03-14**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. Night hours are reported in MW, never as a percentage: their denominator is ~0.

## AT — xgboost, source `energy_renewable`

n_train 1,851 · n_holdout 692 (daylight 313 / shoulder 76 / night 303) · incumbent version 20260112_165237

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 598.9 | 48.2% | 159.4 | 159.3 | 10.50 | 1,011.0 | 0 |
| control@42 | 876.2 | 70.5% | 47.2 | 47.6 | 13.83 | 399.5 | 0 |
| control@1337 | 874.4 | 70.3% | 57.8 | 58.1 | 9.24 | 346.0 | 6 |
| control@2718 | 881.0 | 70.8% | 57.8 | 58.1 | 9.94 | 399.3 | 0 |
| control@7 | 856.5 | 68.9% | 58.7 | 59.0 | 9.05 | 411.1 | 37 |
| control@13 | 872.6 | 70.2% | 44.8 | 44.8 | 5.46 | 389.2 | 89 |
| control@101 | 862.6 | 69.4% | 53.4 | 52.2 | 5.13 | 415.2 | 279 |
| control@271 | 863.2 | 69.4% | 51.6 | 51.5 | 6.49 | 384.6 | 85 |
| control@314 | 874.1 | 70.3% | 50.1 | 50.5 | 10.92 | 391.1 | 0 |
| control@577 | 864.5 | 69.5% | 50.4 | 50.7 | 13.10 | 421.0 | 0 |
| control@863 | 872.4 | 70.1% | 48.5 | 48.8 | 11.36 | 386.9 | 0 |
| control@1024 | 885.7 | 71.2% | 47.2 | 47.6 | 14.91 | 379.2 | 0 |
| control@1729 | 889.6 | 71.5% | 58.9 | 59.3 | 14.22 | 402.6 | 0 |
| geometry@42 | 854.3 | 68.7% | 30.1 | 30.5 | 21.63 | 182.6 | 0 |
| geometry@1337 | 839.1 | 67.5% | 30.5 | 30.8 | 20.22 | 135.0 | 0 |
| geometry@2718 | 842.8 | 67.8% | 35.0 | 35.3 | 23.14 | 157.7 | 0 |
| geometry@7 | 850.4 | 68.4% | 34.0 | 34.3 | 25.09 | 183.0 | 0 |
| geometry@13 | 821.8 | 66.1% | 28.6 | 29.0 | 16.93 | 203.9 | 0 |
| geometry@101 | 837.1 | 67.3% | 29.5 | 29.9 | 21.59 | 189.4 | 0 |
| geometry@271 | 833.1 | 67.0% | 34.6 | 34.9 | 23.32 | 174.1 | 0 |
| geometry@314 | 846.9 | 68.1% | 29.4 | 29.7 | 18.10 | 184.0 | 0 |
| geometry@577 | 831.2 | 66.8% | 28.1 | 28.4 | 19.61 | 202.5 | 0 |
| geometry@863 | 835.2 | 67.2% | 30.8 | 31.1 | 17.82 | 186.9 | 0 |
| geometry@1024 | 839.6 | 67.5% | 27.0 | 27.4 | 22.16 | 153.9 | 0 |
| geometry@1729 | 824.6 | 66.3% | 31.7 | 32.0 | 19.60 | 197.9 | 0 |

Training-target contamination: 0 of 955 night rows read above 1 MW (max 0.0 MW); dropped from fit: True.

## BE — xgboost, source `energy_renewable`

n_train 18,481 · n_holdout 692 (daylight 306 / shoulder 80 / night 306) · incumbent version 20260201_222022

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,327.6 | 61.6% | 130.6 | 124.7 | 52.55 | 2,967.3 | 0 |
| control@42 | 551.3 | 25.6% | 35.6 | 37.9 | 5.21 | 160.4 | 59 |
| control@1337 | 550.1 | 25.5% | 32.6 | 33.0 | 4.18 | 189.8 | 128 |
| control@2718 | 561.1 | 26.0% | 42.5 | 46.0 | 9.26 | 160.2 | 2 |
| control@7 | 547.4 | 25.4% | 37.3 | 39.8 | 4.79 | 158.0 | 12 |
| control@13 | 537.7 | 24.9% | 30.8 | 32.1 | 5.67 | 168.2 | 38 |
| control@101 | 531.8 | 24.7% | 39.7 | 43.2 | 8.47 | 280.1 | 58 |
| control@271 | 541.1 | 25.1% | 44.0 | 46.7 | 5.71 | 192.4 | 102 |
| control@314 | 564.1 | 26.2% | 43.6 | 46.9 | 12.68 | 188.0 | 2 |
| control@577 | 572.3 | 26.5% | 39.3 | 42.6 | 12.07 | 209.2 | 0 |
| control@863 | 552.1 | 25.6% | 39.4 | 42.8 | 8.69 | 163.4 | 2 |
| control@1024 | 553.2 | 25.7% | 36.6 | 39.4 | 7.11 | 114.1 | 1 |
| control@1729 | 559.9 | 26.0% | 47.7 | 51.2 | 14.39 | 174.4 | 0 |
| geometry@42 | 533.1 | 24.7% | 15.4 | 18.8 | 9.02 | 32.3 | 5 |
| geometry@1337 | 533.4 | 24.7% | 11.4 | 13.9 | 2.40 | 39.5 | 8 |
| geometry@2718 | 543.7 | 25.2% | 19.5 | 21.6 | 2.73 | 53.0 | 6 |
| geometry@7 | 537.5 | 24.9% | 12.0 | 14.0 | 3.55 | 51.5 | 7 |
| geometry@13 | 541.2 | 25.1% | 10.3 | 11.4 | 1.60 | 20.0 | 13 |
| geometry@101 | 548.5 | 25.4% | 20.2 | 23.8 | 12.58 | 93.0 | 0 |
| geometry@271 | 531.7 | 24.7% | 10.5 | 12.4 | 1.19 | 15.0 | 20 |
| geometry@314 | 555.6 | 25.8% | 12.5 | 14.3 | 7.42 | 43.4 | 8 |
| geometry@577 | 539.1 | 25.0% | 12.7 | 15.0 | 2.16 | 45.3 | 19 |
| geometry@863 | 538.7 | 25.0% | 18.9 | 22.1 | 4.48 | 52.5 | 5 |
| geometry@1024 | 536.4 | 24.9% | 11.9 | 13.9 | 2.02 | 63.6 | 12 |
| geometry@1729 | 544.4 | 25.2% | 15.2 | 15.1 | 1.06 | 78.7 | 36 |

Training-target contamination: 0 of 6,989 night rows read above 1 MW (max 0.1 MW); dropped from fit: True.

## DE — xgboost, source `energy_renewable`

n_train 1,958 · n_holdout 693 (daylight 311 / shoulder 73 / night 309) · incumbent version 20260223_193822

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 7,765.0 | 47.6% | 708.1 | 701.5 | 43.39 | 2,552.9 | 0 |
| control@42 | 8,878.1 | 54.4% | 198.3 | 202.5 | -7.78 | 75.3 | 298 |
| control@1337 | 8,881.0 | 54.4% | 176.1 | 188.2 | 18.03 | 135.2 | 84 |
| control@2718 | 9,226.7 | 56.6% | 232.7 | 252.9 | 27.62 | 104.3 | 49 |
| control@7 | 9,194.0 | 56.4% | 185.1 | 200.3 | 11.98 | 89.0 | 51 |
| control@13 | 8,913.8 | 54.6% | 280.8 | 300.7 | 13.13 | 156.7 | 41 |
| control@101 | 9,151.8 | 56.1% | 204.0 | 216.9 | 5.71 | 216.9 | 180 |
| control@271 | 8,942.0 | 54.8% | 202.0 | 221.4 | 12.72 | 111.2 | 74 |
| control@314 | 8,883.4 | 54.5% | 196.0 | 208.5 | 8.79 | 82.9 | 75 |
| control@577 | 9,186.6 | 56.3% | 192.5 | 210.5 | 24.69 | 156.8 | 76 |
| control@863 | 8,882.6 | 54.4% | 175.4 | 189.9 | 7.37 | 106.8 | 72 |
| control@1024 | 8,877.6 | 54.4% | 159.5 | 173.9 | -8.74 | 132.4 | 250 |
| control@1729 | 9,026.4 | 55.3% | 235.9 | 254.1 | 23.19 | 144.5 | 112 |
| geometry@42 | 7,669.7 | 47.0% | 48.8 | 74.1 | 13.54 | 47.4 | 95 |
| geometry@1337 | 7,839.9 | 48.1% | 49.7 | 74.9 | 2.69 | 80.0 | 221 |
| geometry@2718 | 7,991.5 | 49.0% | 68.4 | 90.3 | 9.95 | 35.0 | 38 |
| geometry@7 | 8,072.2 | 49.5% | 65.0 | 91.6 | 18.41 | 63.2 | 65 |
| geometry@13 | 7,856.9 | 48.2% | 50.8 | 75.0 | 5.45 | 38.8 | 82 |
| geometry@101 | 8,256.0 | 50.6% | 59.2 | 86.5 | 18.25 | 45.0 | 98 |
| geometry@271 | 7,963.6 | 48.8% | 36.7 | 60.3 | 5.91 | 44.2 | 85 |
| geometry@314 | 7,885.2 | 48.3% | 58.8 | 85.1 | 5.49 | 41.7 | 43 |
| geometry@577 | 8,136.3 | 49.9% | 44.8 | 62.3 | -3.95 | 11.5 | 299 |
| geometry@863 | 7,711.4 | 47.3% | 65.2 | 78.5 | -1.50 | 33.9 | 269 |
| geometry@1024 | 7,837.4 | 48.0% | 45.6 | 68.9 | -3.04 | 61.5 | 271 |
| geometry@1729 | 8,008.6 | 49.1% | 61.4 | 64.6 | -22.78 | 43.8 | 297 |

Training-target contamination: 0 of 1,047 night rows read above 1 MW (max 0.1 MW); dropped from fit: True.

## FR — xgboost, source `energy_renewable`

n_train 26,874 · n_holdout 692 (daylight 314 / shoulder 78 / night 300) · incumbent version 20260201_222014

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 2,551.8 | 40.1% | 852.9 | 959.0 | 97.38 | 3,275.8 | 0 |
| control@42 | 966.3 | 15.2% | 129.7 | 306.4 | 33.63 | 279.1 | 1 |
| control@1337 | 951.4 | 15.0% | 138.2 | 302.1 | 19.71 | 386.9 | 112 |
| control@2718 | 938.9 | 14.8% | 118.1 | 280.4 | 15.54 | 243.7 | 121 |
| control@7 | 912.0 | 14.3% | 148.9 | 321.0 | 21.93 | 225.4 | 0 |
| control@13 | 925.1 | 14.6% | 112.8 | 276.4 | 16.92 | 225.3 | 67 |
| control@101 | 919.2 | 14.5% | 110.5 | 267.8 | 21.42 | 472.3 | 86 |
| control@271 | 943.7 | 14.8% | 109.2 | 259.9 | 23.13 | 266.8 | 102 |
| control@314 | 946.5 | 14.9% | 133.5 | 313.1 | 45.76 | 405.4 | 0 |
| control@577 | 972.8 | 15.3% | 141.1 | 322.4 | 54.56 | 368.8 | 1 |
| control@863 | 933.2 | 14.7% | 125.9 | 290.5 | 20.16 | 216.8 | 71 |
| control@1024 | 938.7 | 14.8% | 110.9 | 270.6 | 18.29 | 430.5 | 109 |
| control@1729 | 953.4 | 15.0% | 126.2 | 287.9 | 22.46 | 237.7 | 1 |
| geometry@42 | 941.9 | 14.8% | 70.1 | 247.1 | 7.07 | 147.0 | 91 |
| geometry@1337 | 918.1 | 14.4% | 64.1 | 232.6 | 5.38 | 184.1 | 89 |
| geometry@2718 | 925.5 | 14.6% | 76.3 | 265.3 | 46.65 | 252.4 | 0 |
| geometry@7 | 922.0 | 14.5% | 73.7 | 246.7 | 7.45 | 142.6 | 58 |
| geometry@13 | 911.5 | 14.3% | 62.4 | 238.2 | 6.80 | 148.3 | 90 |
| geometry@101 | 926.1 | 14.6% | 63.5 | 231.1 | 1.03 | 135.9 | 118 |
| geometry@271 | 931.7 | 14.7% | 75.1 | 249.6 | 7.15 | 187.8 | 125 |
| geometry@314 | 921.6 | 14.5% | 61.5 | 238.2 | 3.89 | 145.5 | 93 |
| geometry@577 | 917.6 | 14.4% | 63.9 | 246.4 | 26.82 | 199.7 | 1 |
| geometry@863 | 908.2 | 14.3% | 71.2 | 238.6 | 9.75 | 126.8 | 1 |
| geometry@1024 | 956.3 | 15.0% | 77.7 | 268.5 | 54.27 | 198.2 | 0 |
| geometry@1729 | 926.3 | 14.6% | 65.6 | 234.8 | 4.94 | 134.1 | 104 |

Training-target contamination: 430 of 10,642 night rows read above 1 MW (max 439.3 MW); dropped from fit: True.
