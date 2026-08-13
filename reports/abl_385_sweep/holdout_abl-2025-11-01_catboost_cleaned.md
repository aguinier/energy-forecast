# ABL-338 — solar non-negativity and solar geometry: held-out A/B

Generated 2026-08-13T14:22:00 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-07-13 .. 2026-08-11**, training from 2025-11-01 up to the holdout start.

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

n_train 5,733 · n_holdout 720 (daylight 465 / shoulder 88 / night 167) · incumbent version 20260201_222022

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,090.1 | 32.9% | 3.3 | 5.1 | 0.00 | 0.0 | 0 |
| control@42 | 599.3 | 18.1% | 20.7 | 12.1 | -3.13 | 42.4 | 146 |
| control@1337 | 600.8 | 18.1% | 21.0 | -6.3 | -22.22 | 21.5 | 206 |
| control@2718 | 604.4 | 18.2% | 15.3 | -2.8 | -4.75 | 36.3 | 161 |
| control@7 | 598.9 | 18.1% | 25.1 | 23.7 | 8.82 | 56.7 | 50 |
| control@13 | 578.3 | 17.4% | 21.0 | 15.8 | 0.16 | 49.4 | 119 |
| control@101 | 578.2 | 17.4% | 21.8 | 13.5 | 3.79 | 87.8 | 124 |
| control@271 | 575.8 | 17.4% | 16.5 | 7.0 | -4.39 | 61.6 | 145 |
| control@314 | 592.1 | 17.8% | 19.4 | 18.8 | 7.80 | 49.6 | 49 |
| control@577 | 625.9 | 18.9% | 35.9 | 37.3 | 22.97 | 67.5 | 20 |
| control@863 | 644.2 | 19.4% | 37.1 | 34.0 | 23.42 | 63.9 | 58 |
| control@1024 | 610.0 | 18.4% | 18.2 | 12.8 | 3.24 | 67.4 | 116 |
| control@1729 | 620.4 | 18.7% | 46.2 | 44.9 | 27.12 | 84.3 | 32 |
| geometry@42 | 571.2 | 17.2% | 23.7 | -7.5 | -24.07 | 17.0 | 213 |
| geometry@1337 | 616.6 | 18.6% | 25.1 | 12.0 | 1.93 | 66.0 | 108 |
| geometry@2718 | 556.6 | 16.8% | 16.6 | 9.9 | -7.68 | 34.8 | 146 |
| geometry@7 | 662.3 | 20.0% | 87.6 | 89.3 | 67.92 | 118.4 | 10 |
| geometry@13 | 678.2 | 20.4% | 52.9 | 52.2 | 32.54 | 102.6 | 33 |
| geometry@101 | 657.1 | 19.8% | 46.9 | 40.3 | 7.84 | 72.1 | 97 |
| geometry@271 | 577.4 | 17.4% | 16.9 | -5.6 | -10.19 | 35.6 | 191 |
| geometry@314 | 601.3 | 18.1% | 29.5 | 21.3 | -5.31 | 38.6 | 119 |
| geometry@577 | 604.3 | 18.2% | 34.2 | 32.2 | 16.54 | 62.3 | 56 |
| geometry@863 | 566.6 | 17.1% | 17.8 | 8.0 | -17.20 | 33.8 | 170 |
| geometry@1024 | 573.3 | 17.3% | 27.6 | 6.8 | -8.49 | 93.0 | 160 |
| geometry@1729 | 604.2 | 18.2% | 22.7 | -0.2 | -8.17 | 40.3 | 150 |

Training-target contamination: 0 of 2,181 night rows read above 1 MW (max 0.0 MW); dropped from fit: True.

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

n_train 5,545 · n_holdout 720 (daylight 435 / shoulder 99 / night 186) · incumbent version 20260201_222014

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 2,368.6 | 22.3% | 50.8 | 270.4 | 34.88 | 321.5 | 0 |
| control@42 | 2,024.3 | 19.0% | 341.0 | 584.4 | 366.44 | 546.4 | 0 |
| control@1337 | 1,729.2 | 16.3% | 84.2 | 235.0 | 76.23 | 252.3 | 4 |
| control@2718 | 1,907.2 | 17.9% | 162.1 | 396.3 | 212.23 | 366.4 | 0 |
| control@7 | 1,780.6 | 16.7% | 120.7 | 346.5 | 192.79 | 483.3 | 0 |
| control@13 | 1,862.2 | 17.5% | 149.9 | 385.3 | 209.87 | 352.0 | 0 |
| control@101 | 2,014.5 | 18.9% | 193.1 | 435.3 | 245.41 | 403.0 | 0 |
| control@271 | 1,978.0 | 18.6% | 183.1 | 425.4 | 251.92 | 367.9 | 0 |
| control@314 | 2,056.5 | 19.3% | 170.8 | 408.9 | 244.85 | 401.9 | 0 |
| control@577 | 1,745.2 | 16.4% | 107.4 | 330.7 | 155.35 | 426.1 | 0 |
| control@863 | 1,787.8 | 16.8% | 133.4 | 326.3 | 120.05 | 416.5 | 1 |
| control@1024 | 1,768.7 | 16.6% | 86.8 | 283.3 | 109.66 | 308.0 | 1 |
| control@1729 | 1,810.1 | 17.0% | 98.7 | 302.7 | 115.22 | 304.2 | 3 |
| geometry@42 | 1,785.8 | 16.8% | 97.3 | 308.7 | 143.35 | 452.8 | 0 |
| geometry@1337 | 1,764.0 | 16.6% | 63.0 | 246.7 | 128.05 | 230.2 | 0 |
| geometry@2718 | 2,126.4 | 20.0% | 131.3 | 369.2 | 243.61 | 364.1 | 0 |
| geometry@7 | 1,775.9 | 16.7% | 93.9 | 252.0 | 101.55 | 352.1 | 0 |
| geometry@13 | 1,974.1 | 18.6% | 132.9 | 347.4 | 177.85 | 441.2 | 0 |
| geometry@101 | 1,668.6 | 15.7% | 76.8 | 262.2 | 115.58 | 271.8 | 0 |
| geometry@271 | 1,874.8 | 17.6% | 95.2 | 315.5 | 203.18 | 289.4 | 0 |
| geometry@314 | 2,071.3 | 19.5% | 146.5 | 362.6 | 217.09 | 508.0 | 0 |
| geometry@577 | 1,968.5 | 18.5% | 156.2 | 393.9 | 237.14 | 535.6 | 0 |
| geometry@863 | 2,151.5 | 20.2% | 193.3 | 434.1 | 305.52 | 501.3 | 0 |
| geometry@1024 | 1,974.5 | 18.6% | 116.1 | 346.0 | 212.99 | 375.4 | 0 |
| geometry@1729 | 1,851.9 | 17.4% | 158.3 | 392.6 | 257.45 | 386.4 | 0 |

Training-target contamination: 183 of 2,270 night rows read above 1 MW (max 285.9 MW); dropped from fit: True.
