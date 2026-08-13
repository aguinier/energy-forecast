# ABL-338 — solar non-negativity and solar geometry: held-out A/B

Generated 2026-08-13T14:05:54 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-07-13 .. 2026-08-11**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. Night hours are reported in MW, never as a percentage: their denominator is ~0.

## AT — catboost, source `energy_renewable`

n_train 5,423 · n_holdout 720 (daylight 448 / shoulder 92 / night 180) · incumbent version 20260112_165237

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 573.1 | 25.6% | 1.1 | 4.5 | 0.00 | 0.0 | 0 |
| control@42 | 324.8 | 14.5% | 14.3 | 15.4 | 5.86 | 46.5 | 91 |
| control@1337 | 317.7 | 14.2% | 13.0 | 9.6 | 1.97 | 39.9 | 108 |
| control@2718 | 351.4 | 15.7% | 44.9 | 48.8 | 39.78 | 97.5 | 0 |
| control@7 | 329.5 | 14.7% | 20.9 | 24.7 | 9.24 | 32.8 | 15 |
| control@13 | 336.9 | 15.0% | 47.1 | 50.9 | 27.04 | 54.2 | 0 |
| control@101 | 321.9 | 14.4% | 12.7 | 14.8 | 7.07 | 41.7 | 50 |
| control@271 | 333.7 | 14.9% | 36.9 | 40.7 | 33.17 | 78.8 | 0 |
| control@314 | 332.7 | 14.8% | 20.5 | 21.4 | 0.84 | 37.3 | 102 |
| control@577 | 330.5 | 14.8% | 25.6 | 29.5 | 13.66 | 34.5 | 0 |
| control@863 | 342.5 | 15.3% | 47.2 | 51.0 | 47.01 | 81.2 | 0 |
| control@1024 | 353.6 | 15.8% | 50.5 | 54.3 | 47.83 | 102.8 | 0 |
| control@1729 | 334.5 | 14.9% | 28.6 | 32.4 | 18.70 | 52.2 | 8 |
| geometry@42 | 334.7 | 14.9% | 25.9 | 28.9 | 13.16 | 66.3 | 49 |
| geometry@1337 | 316.8 | 14.1% | 13.2 | 4.7 | -5.14 | 36.6 | 174 |
| geometry@2718 | 325.1 | 14.5% | 16.9 | 19.5 | 6.87 | 36.4 | 73 |
| geometry@7 | 355.3 | 15.9% | 57.0 | 60.9 | 55.20 | 101.8 | 0 |
| geometry@13 | 347.2 | 15.5% | 45.2 | 49.0 | 37.98 | 81.9 | 0 |
| geometry@101 | 358.0 | 16.0% | 9.6 | 9.9 | -1.74 | 31.2 | 149 |
| geometry@271 | 313.5 | 14.0% | 19.5 | 18.2 | 2.42 | 51.0 | 125 |
| geometry@314 | 310.0 | 13.8% | 14.0 | 7.6 | -3.63 | 30.1 | 152 |
| geometry@577 | 327.1 | 14.6% | 24.1 | 27.8 | 17.39 | 53.4 | 9 |
| geometry@863 | 322.6 | 14.4% | 15.1 | 15.8 | -2.79 | 23.7 | 142 |
| geometry@1024 | 330.1 | 14.7% | 14.6 | 16.9 | 8.39 | 37.4 | 35 |
| geometry@1729 | 319.4 | 14.3% | 18.5 | 21.7 | 11.02 | 46.1 | 55 |

Training-target contamination: 0 of 2,094 night rows read above 1 MW (max 0.0 MW); dropped from fit: True.

## BE — catboost, source `energy_renewable`

n_train 22,053 · n_holdout 720 (daylight 465 / shoulder 88 / night 167) · incumbent version 20260201_222022

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,090.1 | 32.9% | 3.3 | 5.1 | 0.00 | 0.0 | 0 |
| control@42 | 567.0 | 17.1% | 18.6 | -7.0 | -18.11 | 37.5 | 188 |
| control@1337 | 570.7 | 17.2% | 23.0 | 3.6 | -8.66 | 28.0 | 153 |
| control@2718 | 570.0 | 17.2% | 21.4 | -3.1 | -2.80 | 82.2 | 156 |
| control@7 | 561.9 | 16.9% | 18.5 | -5.7 | -11.81 | 12.1 | 183 |
| control@13 | 566.8 | 17.1% | 16.8 | -4.9 | -8.96 | 24.9 | 160 |
| control@101 | 561.5 | 16.9% | 20.4 | -5.8 | -12.97 | 26.8 | 183 |
| control@271 | 568.6 | 17.1% | 17.4 | -8.8 | -12.43 | 45.1 | 198 |
| control@314 | 571.4 | 17.2% | 19.1 | -7.5 | -10.27 | 42.2 | 188 |
| control@577 | 561.3 | 16.9% | 17.2 | 2.5 | -0.91 | 43.9 | 126 |
| control@863 | 570.3 | 17.2% | 19.5 | -10.4 | -12.40 | 21.8 | 200 |
| control@1024 | 566.0 | 17.1% | 17.0 | -8.0 | -10.21 | 15.4 | 196 |
| control@1729 | 574.6 | 17.3% | 22.6 | -15.6 | -14.98 | 39.9 | 202 |
| geometry@42 | 539.1 | 16.3% | 18.8 | -6.9 | -14.11 | 24.5 | 191 |
| geometry@1337 | 558.4 | 16.8% | 13.1 | 2.2 | -9.28 | 30.7 | 158 |
| geometry@2718 | 529.3 | 16.0% | 16.3 | 7.6 | -6.49 | 21.9 | 145 |
| geometry@7 | 558.3 | 16.8% | 18.9 | -6.7 | -17.97 | 37.0 | 197 |
| geometry@13 | 541.5 | 16.3% | 15.5 | -3.5 | -12.19 | 13.0 | 187 |
| geometry@101 | 563.1 | 17.0% | 16.5 | -4.4 | -10.37 | 42.0 | 186 |
| geometry@271 | 551.7 | 16.6% | 23.0 | -17.4 | -19.31 | 36.0 | 225 |
| geometry@314 | 547.2 | 16.5% | 15.5 | -4.6 | -9.04 | 38.6 | 161 |
| geometry@577 | 542.8 | 16.4% | 12.4 | -6.8 | -18.35 | 3.8 | 218 |
| geometry@863 | 544.1 | 16.4% | 19.8 | -11.0 | -16.89 | 56.8 | 210 |
| geometry@1024 | 552.1 | 16.6% | 22.8 | -17.6 | -15.02 | 26.5 | 217 |
| geometry@1729 | 556.1 | 16.8% | 19.0 | 3.1 | -7.05 | 36.9 | 147 |

Training-target contamination: 0 of 8,026 night rows read above 1 MW (max 0.1 MW); dropped from fit: True.

## DE — catboost, source `energy_renewable`

n_train 5,527 · n_holdout 720 (daylight 457 / shoulder 99 / night 164) · incumbent version 20260223_193822

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 6,066.6 | 23.0% | 33.5 | 66.2 | 0.00 | 0.0 | 0 |
| control@42 | 3,587.6 | 13.6% | 332.8 | 346.0 | 240.59 | 710.5 | 6 |
| control@1337 | 3,381.8 | 12.8% | 180.9 | 170.8 | 66.70 | 522.7 | 71 |
| control@2718 | 3,693.9 | 14.0% | 417.7 | 434.2 | 366.82 | 842.2 | 23 |
| control@7 | 3,243.4 | 12.3% | 192.9 | 150.0 | -24.88 | 228.7 | 130 |
| control@13 | 3,914.5 | 14.8% | 396.5 | 429.7 | 336.97 | 623.0 | 3 |
| control@101 | 3,380.0 | 12.8% | 156.7 | 150.0 | 31.92 | 267.1 | 78 |
| control@271 | 3,569.6 | 13.5% | 422.0 | 453.6 | 421.83 | 761.8 | 1 |
| control@314 | 3,805.9 | 14.4% | 390.7 | 416.5 | 246.44 | 669.3 | 21 |
| control@577 | 3,704.7 | 14.1% | 268.9 | 292.6 | 158.45 | 549.1 | 15 |
| control@863 | 4,510.0 | 17.1% | 767.5 | 801.2 | 764.23 | 1,158.0 | 0 |
| control@1024 | 3,381.3 | 12.8% | 145.8 | 117.2 | -31.43 | 290.4 | 136 |
| control@1729 | 3,933.9 | 14.9% | 429.0 | 462.7 | 404.10 | 754.3 | 0 |
| geometry@42 | 3,740.3 | 14.2% | 451.1 | 484.7 | 381.71 | 686.3 | 0 |
| geometry@1337 | 3,348.8 | 12.7% | 146.6 | 100.9 | -48.50 | 324.9 | 152 |
| geometry@2718 | 3,327.4 | 12.6% | 122.4 | 80.7 | -36.31 | 184.0 | 162 |
| geometry@7 | 4,055.5 | 15.4% | 457.6 | 487.5 | 449.14 | 889.9 | 4 |
| geometry@13 | 4,199.8 | 15.9% | 383.4 | 417.0 | 292.28 | 620.0 | 1 |
| geometry@101 | 3,618.7 | 13.7% | 213.1 | 220.3 | 137.00 | 572.5 | 75 |
| geometry@271 | 3,565.2 | 13.5% | 563.8 | 597.5 | 595.14 | 886.2 | 0 |
| geometry@314 | 3,352.4 | 12.7% | 168.0 | 112.5 | 111.07 | 416.0 | 45 |
| geometry@577 | 3,577.4 | 13.6% | 407.4 | 438.8 | 410.54 | 804.3 | 0 |
| geometry@863 | 3,274.8 | 12.4% | 108.5 | 58.9 | -30.54 | 309.3 | 135 |
| geometry@1024 | 3,490.7 | 13.2% | 260.5 | 264.7 | 158.46 | 641.9 | 49 |
| geometry@1729 | 3,481.8 | 13.2% | 265.0 | 286.0 | 151.31 | 511.3 | 25 |

Training-target contamination: 4 of 2,087 night rows read above 1 MW (max 1.7 MW); dropped from fit: True.

## FR — catboost, source `energy_renewable`

n_train 30,359 · n_holdout 720 (daylight 435 / shoulder 99 / night 186) · incumbent version 20260201_222014

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 2,368.6 | 22.3% | 50.8 | 270.4 | 34.88 | 321.5 | 0 |
| control@42 | 1,530.0 | 14.4% | 95.4 | 175.4 | 11.26 | 104.6 | 82 |
| control@1337 | 1,471.8 | 13.8% | 95.4 | 165.7 | 15.92 | 121.4 | 75 |
| control@2718 | 1,496.8 | 14.1% | 91.6 | 172.9 | 19.95 | 105.1 | 60 |
| control@7 | 1,472.1 | 13.8% | 93.0 | 170.8 | 10.38 | 166.1 | 73 |
| control@13 | 1,478.5 | 13.9% | 91.0 | 173.5 | 15.88 | 127.4 | 70 |
| control@101 | 1,514.6 | 14.2% | 103.2 | 160.1 | 20.38 | 92.8 | 50 |
| control@271 | 1,486.4 | 14.0% | 89.0 | 174.4 | 25.02 | 180.7 | 59 |
| control@314 | 1,479.8 | 13.9% | 80.6 | 208.3 | 26.55 | 119.1 | 54 |
| control@577 | 1,410.7 | 13.3% | 97.1 | 168.1 | 8.83 | 142.6 | 84 |
| control@863 | 1,502.0 | 14.1% | 96.4 | 163.5 | 12.48 | 126.5 | 75 |
| control@1024 | 1,498.7 | 14.1% | 100.0 | 182.1 | 23.89 | 98.3 | 62 |
| control@1729 | 1,464.0 | 13.8% | 84.3 | 184.5 | 29.86 | 227.5 | 60 |
| geometry@42 | 1,324.8 | 12.5% | 90.9 | 167.3 | -13.84 | 102.3 | 127 |
| geometry@1337 | 1,417.6 | 13.3% | 77.4 | 217.4 | 30.83 | 100.9 | 4 |
| geometry@2718 | 1,311.0 | 12.3% | 98.6 | 164.0 | -8.38 | 95.7 | 110 |
| geometry@7 | 1,373.7 | 12.9% | 99.5 | 160.5 | -14.89 | 66.0 | 121 |
| geometry@13 | 1,339.9 | 12.6% | 88.6 | 190.6 | 1.77 | 88.2 | 94 |
| geometry@101 | 1,341.1 | 12.6% | 91.2 | 176.0 | -10.92 | 76.3 | 112 |
| geometry@271 | 1,411.5 | 13.3% | 83.3 | 188.8 | 3.75 | 68.7 | 90 |
| geometry@314 | 1,374.2 | 12.9% | 85.1 | 195.4 | -9.82 | 75.8 | 110 |
| geometry@577 | 1,385.2 | 13.0% | 89.9 | 172.8 | -16.03 | 68.9 | 124 |
| geometry@863 | 1,395.5 | 13.1% | 96.7 | 164.4 | 8.99 | 78.4 | 74 |
| geometry@1024 | 1,373.7 | 12.9% | 83.3 | 199.9 | 19.21 | 78.4 | 45 |
| geometry@1729 | 1,347.9 | 12.7% | 100.7 | 169.2 | -11.13 | 75.2 | 117 |

Training-target contamination: 517 of 11,794 night rows read above 1 MW (max 439.3 MW); dropped from fit: True.
