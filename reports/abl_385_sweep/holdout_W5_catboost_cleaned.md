# ABL-338 — solar non-negativity and solar geometry: held-out A/B

Generated 2026-08-13T13:55:39 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-06-13 .. 2026-07-12**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. Night hours are reported in MW, never as a percentage: their denominator is ~0.

## AT — catboost, source `energy_renewable`

n_train 4,703 · n_holdout 720 (daylight 480 / shoulder 60 / night 180) · incumbent version 20260112_165237

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 502.3 | 22.6% | 0.1 | 0.4 | 0.00 | 0.0 | 0 |
| control@42 | 250.7 | 11.3% | 16.9 | -1.4 | -1.69 | 61.4 | 159 |
| control@1337 | 259.7 | 11.7% | 13.4 | 8.9 | 2.49 | 46.2 | 113 |
| control@2718 | 279.1 | 12.6% | 25.6 | 25.7 | 25.94 | 52.4 | 13 |
| control@7 | 255.1 | 11.5% | 17.6 | -0.7 | -0.46 | 61.7 | 153 |
| control@13 | 304.4 | 13.7% | 54.7 | 55.1 | 50.57 | 102.0 | 0 |
| control@101 | 270.1 | 12.1% | 24.1 | 21.7 | 18.85 | 77.0 | 39 |
| control@271 | 248.9 | 11.2% | 17.4 | 13.6 | 12.66 | 50.4 | 52 |
| control@314 | 286.1 | 12.9% | 27.9 | 23.6 | 17.42 | 65.8 | 44 |
| control@577 | 251.4 | 11.3% | 15.3 | 6.7 | 6.66 | 42.2 | 107 |
| control@863 | 283.9 | 12.8% | 25.6 | 24.7 | 26.48 | 79.4 | 17 |
| control@1024 | 278.7 | 12.5% | 35.7 | 34.3 | 16.82 | 100.1 | 64 |
| control@1729 | 295.9 | 13.3% | 41.3 | 41.6 | 36.74 | 93.4 | 2 |
| geometry@42 | 246.6 | 11.1% | 25.7 | 18.9 | 11.72 | 71.1 | 70 |
| geometry@1337 | 244.3 | 11.0% | 25.8 | 21.6 | 18.07 | 72.9 | 58 |
| geometry@2718 | 243.0 | 10.9% | 18.5 | 8.2 | 2.56 | 41.2 | 107 |
| geometry@7 | 258.6 | 11.6% | 37.3 | 37.5 | 32.75 | 79.8 | 10 |
| geometry@13 | 308.6 | 13.9% | 51.0 | 51.4 | 46.90 | 89.1 | 0 |
| geometry@101 | 311.6 | 14.0% | 28.3 | 28.1 | 23.20 | 70.4 | 14 |
| geometry@271 | 241.6 | 10.9% | 16.0 | 11.6 | 0.02 | 47.9 | 145 |
| geometry@314 | 243.8 | 11.0% | 31.8 | 31.6 | 22.34 | 64.8 | 20 |
| geometry@577 | 247.7 | 11.1% | 20.1 | 18.0 | 10.20 | 50.6 | 63 |
| geometry@863 | 255.6 | 11.5% | 18.2 | 14.2 | 6.65 | 64.9 | 87 |
| geometry@1024 | 244.3 | 11.0% | 12.2 | 7.3 | 1.94 | 30.8 | 132 |
| geometry@1729 | 246.5 | 11.1% | 31.5 | 29.4 | 17.12 | 69.3 | 43 |

Training-target contamination: 0 of 1,914 night rows read above 1 MW (max 0.0 MW); dropped from fit: True.

## BE — catboost, source `energy_renewable`

n_train 21,333 · n_holdout 720 (daylight 480 / shoulder 116 / night 124) · incumbent version 20260201_222022

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 976.2 | 26.9% | 1.8 | 2.2 | 0.00 | 0.0 | 0 |
| control@42 | 447.6 | 12.3% | 19.8 | 0.7 | -11.33 | 14.6 | 149 |
| control@1337 | 458.2 | 12.6% | 29.0 | -15.8 | -19.31 | 29.7 | 150 |
| control@2718 | 445.7 | 12.3% | 22.4 | -5.4 | -14.18 | 18.2 | 156 |
| control@7 | 449.4 | 12.4% | 18.8 | -0.2 | -7.73 | 22.3 | 133 |
| control@13 | 448.9 | 12.4% | 15.1 | 2.7 | -6.88 | 21.2 | 138 |
| control@101 | 447.0 | 12.3% | 14.1 | 4.4 | -2.07 | 20.8 | 126 |
| control@271 | 464.7 | 12.8% | 24.6 | 26.5 | 16.88 | 37.2 | 1 |
| control@314 | 451.3 | 12.4% | 22.3 | 5.3 | -2.10 | 27.3 | 105 |
| control@577 | 463.1 | 12.8% | 22.0 | 22.4 | 11.91 | 31.3 | 47 |
| control@863 | 441.4 | 12.2% | 19.8 | -10.1 | -13.17 | 17.5 | 162 |
| control@1024 | 433.3 | 11.9% | 14.5 | -2.0 | -5.33 | 26.2 | 127 |
| control@1729 | 440.2 | 12.1% | 18.3 | -4.8 | -6.14 | 26.1 | 141 |
| geometry@42 | 451.3 | 12.4% | 17.7 | 17.3 | 10.74 | 37.1 | 86 |
| geometry@1337 | 437.9 | 12.1% | 14.7 | 7.5 | -5.89 | 21.7 | 151 |
| geometry@2718 | 435.6 | 12.0% | 26.3 | -15.3 | -19.85 | 31.0 | 188 |
| geometry@7 | 433.3 | 11.9% | 19.5 | 13.6 | -10.94 | 8.6 | 139 |
| geometry@13 | 444.2 | 12.2% | 17.6 | 13.9 | 1.53 | 19.6 | 84 |
| geometry@101 | 455.5 | 12.6% | 12.9 | 13.0 | 3.05 | 17.7 | 69 |
| geometry@271 | 440.3 | 12.1% | 10.4 | 1.1 | -2.27 | 34.2 | 127 |
| geometry@314 | 433.2 | 11.9% | 13.7 | -2.2 | -10.24 | 22.4 | 140 |
| geometry@577 | 447.1 | 12.3% | 15.5 | 12.7 | -1.76 | 22.5 | 93 |
| geometry@863 | 438.5 | 12.1% | 23.2 | -13.9 | -27.82 | 21.1 | 190 |
| geometry@1024 | 436.8 | 12.0% | 17.0 | -4.6 | -12.87 | 21.5 | 161 |
| geometry@1729 | 451.7 | 12.5% | 16.6 | 18.4 | 7.87 | 31.3 | 20 |

Training-target contamination: 0 of 7,902 night rows read above 1 MW (max 0.1 MW); dropped from fit: True.

## DE — catboost, source `energy_renewable`

n_train 4,807 · n_holdout 720 (daylight 480 / shoulder 110 / night 130) · incumbent version 20260223_193822

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 5,806.6 | 23.4% | 20.0 | 57.7 | 0.00 | 0.0 | 0 |
| control@42 | 3,507.3 | 14.2% | 309.6 | 356.1 | 276.62 | 674.9 | 11 |
| control@1337 | 3,301.4 | 13.3% | 125.3 | 47.3 | -38.66 | 271.2 | 125 |
| control@2718 | 3,196.7 | 12.9% | 229.2 | 259.0 | 36.07 | 289.7 | 68 |
| control@7 | 3,353.0 | 13.5% | 120.2 | 107.1 | 26.11 | 327.9 | 83 |
| control@13 | 3,981.9 | 16.1% | 325.8 | 366.6 | 239.47 | 922.2 | 26 |
| control@101 | 3,259.6 | 13.2% | 206.4 | 208.7 | 119.98 | 449.9 | 56 |
| control@271 | 3,657.1 | 14.8% | 274.5 | 326.4 | 206.89 | 668.7 | 20 |
| control@314 | 3,511.2 | 14.2% | 321.7 | 372.0 | 167.83 | 580.6 | 29 |
| control@577 | 3,898.3 | 15.7% | 559.8 | 621.5 | 428.24 | 934.9 | 0 |
| control@863 | 4,297.3 | 17.3% | 523.7 | 585.4 | 476.66 | 1,018.1 | 0 |
| control@1024 | 3,439.6 | 13.9% | 336.4 | 383.0 | 332.20 | 873.5 | 16 |
| control@1729 | 3,376.9 | 13.6% | 198.4 | 214.4 | 164.75 | 507.1 | 26 |
| geometry@42 | 3,259.5 | 13.2% | 230.0 | 264.8 | 131.14 | 630.3 | 56 |
| geometry@1337 | 3,265.4 | 13.2% | 247.6 | 286.4 | 174.40 | 537.4 | 38 |
| geometry@2718 | 3,179.8 | 12.8% | 229.6 | 284.8 | 195.75 | 394.7 | 8 |
| geometry@7 | 3,589.5 | 14.5% | 317.5 | 371.5 | 299.93 | 877.5 | 4 |
| geometry@13 | 3,669.2 | 14.8% | 282.0 | 321.2 | 250.43 | 541.3 | 2 |
| geometry@101 | 3,965.7 | 16.0% | 265.0 | 267.6 | 116.65 | 651.8 | 63 |
| geometry@271 | 3,231.0 | 13.0% | 224.5 | 271.2 | 160.95 | 581.4 | 27 |
| geometry@314 | 3,258.3 | 13.1% | 238.2 | 238.0 | 15.86 | 498.3 | 98 |
| geometry@577 | 3,466.0 | 14.0% | 344.7 | 399.8 | 286.51 | 1,011.7 | 22 |
| geometry@863 | 3,422.9 | 13.8% | 311.5 | 369.4 | 229.87 | 534.2 | 10 |
| geometry@1024 | 3,151.6 | 12.7% | 210.5 | 171.0 | 84.71 | 583.2 | 85 |
| geometry@1729 | 3,341.7 | 13.5% | 195.7 | 220.2 | 105.31 | 555.3 | 58 |

Training-target contamination: 4 of 1,957 night rows read above 1 MW (max 1.7 MW); dropped from fit: True.

## FR — catboost, source `energy_renewable`

n_train 29,668 · n_holdout 720 (daylight 480 / shoulder 60 / night 180) · incumbent version 20260201_222014

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,645.0 | 16.0% | 40.6 | 129.1 | 34.82 | 245.9 | 0 |
| control@42 | 1,208.2 | 11.8% | 82.8 | 116.6 | 17.94 | 167.3 | 75 |
| control@1337 | 1,185.3 | 11.5% | 91.4 | 107.0 | 30.83 | 181.7 | 59 |
| control@2718 | 1,157.1 | 11.3% | 107.5 | 105.4 | 25.92 | 213.0 | 62 |
| control@7 | 1,169.3 | 11.4% | 80.4 | 136.3 | 27.03 | 137.6 | 56 |
| control@13 | 1,187.4 | 11.6% | 81.7 | 119.1 | 44.31 | 239.6 | 50 |
| control@101 | 1,147.2 | 11.2% | 93.7 | 94.1 | 5.03 | 158.5 | 96 |
| control@271 | 1,134.3 | 11.0% | 107.0 | 96.2 | 13.78 | 129.1 | 80 |
| control@314 | 1,163.4 | 11.3% | 111.6 | 66.6 | -2.78 | 110.2 | 98 |
| control@577 | 1,147.7 | 11.2% | 87.4 | 109.5 | 24.64 | 193.6 | 68 |
| control@863 | 1,105.1 | 10.8% | 98.9 | 119.4 | 24.15 | 151.2 | 66 |
| control@1024 | 1,166.2 | 11.4% | 102.8 | 86.9 | 1.47 | 156.1 | 107 |
| control@1729 | 1,107.0 | 10.8% | 71.5 | 123.9 | 34.53 | 171.7 | 59 |
| geometry@42 | 1,096.7 | 10.7% | 107.1 | 172.4 | -7.24 | 89.5 | 107 |
| geometry@1337 | 1,181.4 | 11.5% | 78.3 | 170.2 | 2.98 | 139.2 | 98 |
| geometry@2718 | 1,201.4 | 11.7% | 83.4 | 164.4 | 4.98 | 152.6 | 95 |
| geometry@7 | 1,182.3 | 11.5% | 88.6 | 133.7 | -31.10 | 111.2 | 133 |
| geometry@13 | 1,120.1 | 10.9% | 86.9 | 165.0 | 2.26 | 125.9 | 93 |
| geometry@101 | 1,120.4 | 10.9% | 85.4 | 142.8 | -22.66 | 91.1 | 109 |
| geometry@271 | 1,187.3 | 11.6% | 84.5 | 139.0 | -33.29 | 109.8 | 126 |
| geometry@314 | 1,123.7 | 10.9% | 98.7 | 156.2 | 7.91 | 112.9 | 81 |
| geometry@577 | 1,132.0 | 11.0% | 100.8 | 127.7 | -32.94 | 106.9 | 130 |
| geometry@863 | 1,130.6 | 11.0% | 84.6 | 163.6 | 8.81 | 164.0 | 83 |
| geometry@1024 | 1,158.4 | 11.3% | 104.3 | 117.8 | -25.59 | 102.8 | 117 |
| geometry@1729 | 1,114.8 | 10.9% | 93.4 | 145.6 | -7.92 | 128.0 | 106 |

Training-target contamination: 488 of 11,614 night rows read above 1 MW (max 439.3 MW); dropped from fit: True.
