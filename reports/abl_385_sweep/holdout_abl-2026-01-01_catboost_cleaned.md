# ABL-338 — solar non-negativity and solar geometry: held-out A/B

Generated 2026-08-13T14:25:26 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-07-13 .. 2026-08-11**, training from 2026-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. Night hours are reported in MW, never as a percentage: their denominator is ~0.

## AT — catboost, source `energy_renewable`

n_train 4,268 · n_holdout 720 (daylight 448 / shoulder 92 / night 180) · incumbent version 20260112_165237

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 573.1 | 25.6% | 1.1 | 4.5 | 0.00 | 0.0 | 0 |
| control@42 | 324.0 | 14.5% | 20.7 | 22.4 | 7.39 | 60.7 | 72 |
| control@1337 | 329.2 | 14.7% | 27.2 | 29.4 | 13.34 | 72.3 | 47 |
| control@2718 | 322.8 | 14.4% | 30.1 | 32.7 | 16.51 | 61.5 | 42 |
| control@7 | 331.7 | 14.8% | 17.0 | 18.4 | 6.58 | 56.5 | 55 |
| control@13 | 334.9 | 14.9% | 17.9 | 13.5 | 2.28 | 73.1 | 132 |
| control@101 | 323.1 | 14.4% | 23.8 | 27.1 | 5.76 | 52.2 | 71 |
| control@271 | 319.0 | 14.2% | 26.1 | 24.1 | 3.00 | 58.5 | 118 |
| control@314 | 331.3 | 14.8% | 22.7 | 23.6 | 8.08 | 66.5 | 72 |
| control@577 | 348.2 | 15.5% | 37.4 | 38.5 | 19.66 | 82.4 | 59 |
| control@863 | 342.8 | 15.3% | 16.7 | 14.7 | 2.81 | 56.2 | 98 |
| control@1024 | 323.1 | 14.4% | 20.3 | 19.5 | -1.04 | 54.1 | 124 |
| control@1729 | 322.2 | 14.4% | 15.7 | 15.0 | 6.60 | 54.2 | 89 |
| geometry@42 | 307.4 | 13.7% | 22.5 | 21.8 | 2.03 | 60.0 | 115 |
| geometry@1337 | 320.5 | 14.3% | 14.5 | 12.5 | 4.16 | 32.8 | 98 |
| geometry@2718 | 310.5 | 13.9% | 14.1 | 15.2 | 2.26 | 35.0 | 90 |
| geometry@7 | 318.8 | 14.2% | 23.4 | 13.9 | 0.85 | 81.4 | 150 |
| geometry@13 | 313.6 | 14.0% | 11.7 | 1.5 | -2.81 | 32.6 | 171 |
| geometry@101 | 305.3 | 13.6% | 13.1 | 4.3 | 5.12 | 34.9 | 106 |
| geometry@271 | 301.3 | 13.4% | 10.4 | 10.2 | -3.68 | 27.6 | 156 |
| geometry@314 | 313.8 | 14.0% | 13.8 | 7.6 | -4.84 | 40.3 | 163 |
| geometry@577 | 318.1 | 14.2% | 24.4 | 26.7 | 15.20 | 56.4 | 43 |
| geometry@863 | 306.3 | 13.7% | 21.8 | 20.2 | 7.21 | 55.9 | 93 |
| geometry@1024 | 307.3 | 13.7% | 25.1 | 21.9 | 11.78 | 67.0 | 74 |
| geometry@1729 | 310.4 | 13.9% | 22.9 | 19.8 | -1.03 | 45.6 | 123 |

Training-target contamination: 0 of 1,487 night rows read above 1 MW (max 0.0 MW); dropped from fit: True.

## BE — catboost, source `energy_renewable`

n_train 4,268 · n_holdout 720 (daylight 465 / shoulder 88 / night 167) · incumbent version 20260201_222022

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,090.1 | 32.9% | 3.3 | 5.1 | 0.00 | 0.0 | 0 |
| control@42 | 574.7 | 17.3% | 41.6 | 40.8 | 18.63 | 80.0 | 44 |
| control@1337 | 573.3 | 17.3% | 28.9 | 29.0 | 1.99 | 75.8 | 105 |
| control@2718 | 623.5 | 18.8% | 55.0 | 56.5 | 45.29 | 117.5 | 16 |
| control@7 | 577.9 | 17.4% | 32.4 | 33.9 | 16.78 | 54.7 | 28 |
| control@13 | 583.7 | 17.6% | 50.7 | 51.5 | 20.11 | 80.0 | 44 |
| control@101 | 562.7 | 17.0% | 41.8 | 42.4 | 15.76 | 75.2 | 37 |
| control@271 | 602.9 | 18.2% | 55.4 | 56.2 | 32.72 | 185.4 | 35 |
| control@314 | 562.8 | 17.0% | 24.7 | 25.3 | 10.17 | 90.3 | 68 |
| control@577 | 604.1 | 18.2% | 50.9 | 51.7 | 36.43 | 96.9 | 21 |
| control@863 | 615.9 | 18.6% | 47.6 | 49.4 | 27.96 | 79.1 | 0 |
| control@1024 | 599.6 | 18.1% | 45.5 | 41.6 | 19.01 | 120.7 | 48 |
| control@1729 | 617.0 | 18.6% | 64.4 | 66.0 | 46.35 | 117.7 | 11 |
| geometry@42 | 571.9 | 17.2% | 33.7 | 20.8 | 9.62 | 110.7 | 105 |
| geometry@1337 | 598.7 | 18.0% | 52.4 | 54.0 | 36.79 | 125.8 | 10 |
| geometry@2718 | 570.8 | 17.2% | 23.9 | 22.9 | 6.08 | 54.5 | 82 |
| geometry@7 | 603.5 | 18.2% | 62.3 | 63.6 | 34.21 | 131.3 | 29 |
| geometry@13 | 574.4 | 17.3% | 49.5 | 48.8 | 11.14 | 69.5 | 82 |
| geometry@101 | 568.5 | 17.1% | 33.6 | 32.7 | 12.24 | 74.3 | 74 |
| geometry@271 | 560.7 | 16.9% | 32.5 | 33.0 | 11.06 | 61.5 | 48 |
| geometry@314 | 585.8 | 17.7% | 51.6 | 52.5 | 34.77 | 112.4 | 23 |
| geometry@577 | 579.9 | 17.5% | 30.7 | 25.1 | 10.07 | 84.8 | 88 |
| geometry@863 | 570.6 | 17.2% | 22.0 | 11.3 | -9.39 | 84.5 | 158 |
| geometry@1024 | 588.9 | 17.8% | 45.6 | 45.3 | 36.65 | 99.2 | 27 |
| geometry@1729 | 573.5 | 17.3% | 31.2 | 31.3 | 15.00 | 76.0 | 43 |

Training-target contamination: 0 of 1,387 night rows read above 1 MW (max 0.0 MW); dropped from fit: True.

## DE — catboost, source `energy_renewable`

n_train 4,265 · n_holdout 720 (daylight 457 / shoulder 99 / night 164) · incumbent version 20260223_193822

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 6,066.6 | 23.0% | 33.5 | 66.2 | 0.00 | 0.0 | 0 |
| control@42 | 3,368.3 | 12.8% | 153.2 | 134.8 | -0.53 | 306.7 | 107 |
| control@1337 | 3,336.5 | 12.7% | 167.2 | 117.2 | -64.86 | 449.1 | 153 |
| control@2718 | 3,365.0 | 12.8% | 124.1 | 101.8 | -61.84 | 230.5 | 137 |
| control@7 | 3,507.8 | 13.3% | 361.8 | 363.6 | 219.13 | 710.7 | 47 |
| control@13 | 3,307.2 | 12.5% | 147.2 | 115.8 | 3.31 | 283.2 | 107 |
| control@101 | 3,225.4 | 12.2% | 255.5 | 244.6 | 117.80 | 694.3 | 83 |
| control@271 | 3,467.7 | 13.2% | 199.5 | 188.6 | 96.61 | 402.0 | 61 |
| control@314 | 3,539.3 | 13.4% | 500.7 | 524.3 | 288.26 | 959.5 | 45 |
| control@577 | 3,616.3 | 13.7% | 314.4 | 307.1 | 241.20 | 775.2 | 55 |
| control@863 | 3,438.7 | 13.0% | 281.4 | 270.5 | 230.36 | 684.5 | 42 |
| control@1024 | 3,705.1 | 14.1% | 337.4 | 365.2 | 235.46 | 713.4 | 40 |
| control@1729 | 3,323.4 | 12.6% | 201.2 | 192.5 | 117.09 | 597.1 | 81 |
| geometry@42 | 3,191.7 | 12.1% | 141.4 | 126.7 | 24.30 | 366.2 | 100 |
| geometry@1337 | 3,512.1 | 13.3% | 331.4 | 355.6 | 154.47 | 568.4 | 59 |
| geometry@2718 | 3,385.8 | 12.8% | 345.1 | 360.6 | 113.88 | 669.8 | 66 |
| geometry@7 | 3,531.1 | 13.4% | 530.6 | 549.1 | 271.48 | 593.9 | 25 |
| geometry@13 | 3,371.0 | 12.8% | 288.9 | 300.4 | 166.64 | 494.7 | 38 |
| geometry@101 | 3,330.1 | 12.6% | 128.4 | 93.9 | 43.59 | 461.8 | 120 |
| geometry@271 | 3,401.5 | 12.9% | 288.9 | 306.2 | 205.19 | 659.7 | 35 |
| geometry@314 | 3,376.5 | 12.8% | 267.5 | 282.8 | 217.86 | 599.2 | 21 |
| geometry@577 | 3,271.4 | 12.4% | 177.7 | 130.4 | 47.04 | 334.5 | 79 |
| geometry@863 | 3,189.5 | 12.1% | 133.4 | 131.5 | -13.73 | 243.6 | 111 |
| geometry@1024 | 3,384.5 | 12.8% | 337.8 | 356.3 | 161.12 | 656.7 | 45 |
| geometry@1729 | 3,342.4 | 12.7% | 176.2 | 114.0 | 42.67 | 582.8 | 118 |

Training-target contamination: 4 of 1,399 night rows read above 1 MW (max 1.7 MW); dropped from fit: True.

## FR — catboost, source `energy_renewable`

n_train 4,129 · n_holdout 720 (daylight 435 / shoulder 99 / night 186) · incumbent version 20260201_222014

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 2,368.6 | 22.3% | 50.8 | 270.4 | 34.88 | 321.5 | 0 |
| control@42 | 1,615.9 | 15.2% | 111.4 | 294.3 | 109.31 | 440.5 | 0 |
| control@1337 | 1,695.1 | 15.9% | 95.0 | 271.5 | 59.34 | 315.5 | 28 |
| control@2718 | 1,647.0 | 15.5% | 109.4 | 321.6 | 118.25 | 371.7 | 0 |
| control@7 | 1,593.7 | 15.0% | 87.6 | 236.5 | 77.63 | 219.8 | 0 |
| control@13 | 1,695.8 | 15.9% | 117.6 | 336.7 | 154.05 | 310.4 | 0 |
| control@101 | 1,633.4 | 15.4% | 90.9 | 279.8 | 109.89 | 279.3 | 0 |
| control@271 | 1,755.0 | 16.5% | 148.1 | 351.2 | 198.26 | 726.9 | 0 |
| control@314 | 1,758.4 | 16.5% | 129.6 | 337.7 | 174.98 | 523.5 | 0 |
| control@577 | 1,857.2 | 17.5% | 185.8 | 414.8 | 234.05 | 529.5 | 0 |
| control@863 | 1,734.7 | 16.3% | 103.6 | 306.0 | 102.72 | 302.4 | 0 |
| control@1024 | 1,700.9 | 16.0% | 94.9 | 272.5 | 122.04 | 524.9 | 0 |
| control@1729 | 1,724.5 | 16.2% | 110.4 | 331.4 | 186.93 | 539.5 | 0 |
| geometry@42 | 1,651.7 | 15.5% | 97.9 | 281.6 | 148.84 | 489.4 | 0 |
| geometry@1337 | 1,646.9 | 15.5% | 101.6 | 303.7 | 78.02 | 368.2 | 6 |
| geometry@2718 | 1,698.3 | 16.0% | 128.4 | 333.8 | 124.96 | 454.6 | 4 |
| geometry@7 | 1,708.1 | 16.1% | 148.7 | 372.2 | 184.92 | 674.7 | 0 |
| geometry@13 | 1,647.7 | 15.5% | 157.3 | 358.2 | 152.68 | 425.0 | 10 |
| geometry@101 | 1,659.9 | 15.6% | 90.7 | 290.5 | 113.66 | 355.6 | 0 |
| geometry@271 | 1,749.1 | 16.4% | 135.1 | 348.2 | 148.38 | 284.4 | 0 |
| geometry@314 | 1,635.5 | 15.4% | 110.4 | 332.3 | 165.61 | 380.9 | 0 |
| geometry@577 | 1,665.9 | 15.7% | 140.1 | 323.9 | 133.08 | 371.8 | 2 |
| geometry@863 | 1,704.1 | 16.0% | 100.0 | 282.4 | 84.54 | 308.5 | 18 |
| geometry@1024 | 1,673.4 | 15.7% | 126.6 | 311.9 | 172.05 | 389.6 | 7 |
| geometry@1729 | 1,683.0 | 15.8% | 109.7 | 270.3 | 95.08 | 402.1 | 2 |

Training-target contamination: 114 of 1,487 night rows read above 1 MW (max 285.9 MW); dropped from fit: True.
