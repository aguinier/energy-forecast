# ABL-338 — solar non-negativity and solar geometry: held-out A/B

Generated 2026-08-13T13:14:52 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-02-13 .. 2026-03-14**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. Night hours are reported in MW, never as a percentage: their denominator is ~0.

## AT — catboost, source `energy_renewable`

n_train 1,851 · n_holdout 692 (daylight 313 / shoulder 76 / night 303) · incumbent version 20260112_165237

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 598.9 | 48.2% | 159.4 | 159.3 | 10.50 | 1,011.0 | 0 |
| control@42 | 926.0 | 74.5% | 40.4 | 40.7 | 18.34 | 230.6 | 0 |
| control@1337 | 921.7 | 74.1% | 39.6 | 39.9 | 18.64 | 245.2 | 0 |
| control@2718 | 932.5 | 75.0% | 43.2 | 43.6 | 24.26 | 252.6 | 0 |
| control@7 | 905.2 | 72.8% | 39.6 | 40.0 | 17.90 | 195.7 | 1 |
| control@13 | 899.5 | 72.3% | 43.5 | 43.8 | 16.50 | 287.3 | 12 |
| control@101 | 903.9 | 72.7% | 41.9 | 42.2 | 16.03 | 263.0 | 3 |
| control@271 | 938.1 | 75.4% | 46.4 | 46.8 | 23.07 | 253.5 | 0 |
| control@314 | 895.8 | 72.0% | 35.5 | 35.8 | 14.11 | 224.8 | 13 |
| control@577 | 919.4 | 73.9% | 47.8 | 48.2 | 23.36 | 231.2 | 0 |
| control@863 | 919.2 | 73.9% | 37.8 | 38.1 | 19.12 | 192.9 | 0 |
| control@1024 | 928.8 | 74.7% | 45.3 | 45.7 | 23.41 | 179.0 | 0 |
| control@1729 | 904.2 | 72.7% | 34.1 | 30.9 | 13.21 | 234.1 | 60 |
| geometry@42 | 872.1 | 70.1% | 33.7 | 34.1 | 19.14 | 170.4 | 0 |
| geometry@1337 | 884.5 | 71.1% | 31.6 | 32.0 | 19.74 | 153.8 | 0 |
| geometry@2718 | 891.3 | 71.7% | 42.7 | 43.1 | 26.30 | 135.7 | 0 |
| geometry@7 | 863.6 | 69.4% | 40.1 | 40.5 | 28.63 | 133.4 | 0 |
| geometry@13 | 864.9 | 69.5% | 32.7 | 33.0 | 21.86 | 162.7 | 0 |
| geometry@101 | 835.4 | 67.2% | 34.8 | 35.1 | 21.08 | 200.0 | 6 |
| geometry@271 | 851.4 | 68.5% | 27.7 | 28.0 | 17.42 | 163.8 | 0 |
| geometry@314 | 846.5 | 68.1% | 30.8 | 31.2 | 19.77 | 165.9 | 0 |
| geometry@577 | 861.5 | 69.3% | 33.1 | 33.5 | 21.46 | 161.2 | 0 |
| geometry@863 | 841.6 | 67.7% | 27.1 | 27.4 | 16.44 | 116.5 | 0 |
| geometry@1024 | 863.3 | 69.4% | 23.5 | 23.7 | 11.36 | 178.7 | 8 |
| geometry@1729 | 863.3 | 69.4% | 36.4 | 36.7 | 19.73 | 160.2 | 0 |

Training-target contamination: 0 of 955 night rows read above 1 MW (max 0.0 MW); dropped from fit: True.

## BE — catboost, source `energy_renewable`

n_train 18,481 · n_holdout 692 (daylight 306 / shoulder 80 / night 306) · incumbent version 20260201_222022

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,327.6 | 61.6% | 130.6 | 124.7 | 52.55 | 2,967.3 | 0 |
| control@42 | 561.0 | 26.0% | 45.7 | 46.3 | 13.84 | 182.9 | 89 |
| control@1337 | 560.8 | 26.0% | 43.3 | 48.3 | 11.08 | 189.9 | 134 |
| control@2718 | 551.0 | 25.6% | 39.6 | 35.8 | 7.63 | 110.6 | 130 |
| control@7 | 571.6 | 26.5% | 46.6 | 46.9 | 14.68 | 267.4 | 104 |
| control@13 | 573.6 | 26.6% | 44.1 | 50.0 | 11.65 | 192.2 | 96 |
| control@101 | 547.6 | 25.4% | 51.2 | 54.3 | 11.29 | 162.3 | 116 |
| control@271 | 597.5 | 27.7% | 50.6 | 55.9 | 15.67 | 279.2 | 97 |
| control@314 | 579.6 | 26.9% | 42.3 | 47.2 | 12.45 | 175.0 | 104 |
| control@577 | 595.8 | 27.6% | 54.2 | 58.2 | 13.77 | 255.2 | 110 |
| control@863 | 551.6 | 25.6% | 51.3 | 51.8 | 15.06 | 262.0 | 115 |
| control@1024 | 581.5 | 27.0% | 46.0 | 50.0 | 16.01 | 225.0 | 89 |
| control@1729 | 571.9 | 26.5% | 50.2 | 54.0 | 14.36 | 237.4 | 81 |
| geometry@42 | 578.9 | 26.8% | 37.1 | 40.6 | 10.30 | 146.3 | 123 |
| geometry@1337 | 565.8 | 26.2% | 36.9 | 40.5 | 12.86 | 118.7 | 100 |
| geometry@2718 | 569.5 | 26.4% | 42.3 | 46.2 | 8.18 | 82.1 | 119 |
| geometry@7 | 553.0 | 25.6% | 34.4 | 39.7 | 6.43 | 109.2 | 116 |
| geometry@13 | 587.9 | 27.3% | 39.8 | 42.2 | 10.84 | 101.1 | 113 |
| geometry@101 | 549.1 | 25.5% | 34.2 | 37.0 | 10.53 | 144.3 | 127 |
| geometry@271 | 580.7 | 26.9% | 37.7 | 40.6 | 15.71 | 129.3 | 91 |
| geometry@314 | 569.2 | 26.4% | 42.7 | 46.3 | 11.75 | 122.9 | 80 |
| geometry@577 | 575.6 | 26.7% | 50.7 | 53.3 | 16.36 | 171.4 | 85 |
| geometry@863 | 581.8 | 27.0% | 42.0 | 44.1 | 12.53 | 163.6 | 105 |
| geometry@1024 | 570.5 | 26.5% | 40.9 | 47.1 | 12.12 | 125.8 | 107 |
| geometry@1729 | 548.0 | 25.4% | 36.8 | 39.5 | 18.11 | 136.3 | 77 |

Training-target contamination: 0 of 6,989 night rows read above 1 MW (max 0.1 MW); dropped from fit: True.

## DE — catboost, source `energy_renewable`

n_train 1,958 · n_holdout 693 (daylight 311 / shoulder 73 / night 309) · incumbent version 20260223_193822

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 7,765.0 | 47.6% | 708.1 | 701.5 | 43.39 | 2,552.9 | 0 |
| control@42 | 11,763.0 | 72.1% | 593.4 | 622.2 | 439.00 | 701.6 | 0 |
| control@1337 | 11,211.8 | 68.7% | 476.0 | 502.6 | 282.26 | 466.3 | 7 |
| control@2718 | 11,471.5 | 70.3% | 524.5 | 555.0 | 386.88 | 610.3 | 0 |
| control@7 | 10,881.6 | 66.7% | 513.4 | 543.4 | 360.99 | 622.2 | 0 |
| control@13 | 11,791.9 | 72.3% | 584.6 | 615.0 | 385.71 | 629.4 | 0 |
| control@101 | 11,271.5 | 69.1% | 470.8 | 501.3 | 394.40 | 634.8 | 0 |
| control@271 | 11,213.5 | 68.7% | 404.9 | 435.4 | 226.99 | 495.0 | 0 |
| control@314 | 11,604.4 | 71.1% | 620.2 | 650.6 | 410.86 | 807.1 | 0 |
| control@577 | 11,157.5 | 68.4% | 493.9 | 524.4 | 325.75 | 580.3 | 0 |
| control@863 | 10,971.3 | 67.3% | 409.7 | 439.6 | 240.27 | 655.9 | 7 |
| control@1024 | 10,976.2 | 67.3% | 415.4 | 443.2 | 271.53 | 462.0 | 7 |
| control@1729 | 11,290.3 | 69.2% | 539.1 | 569.2 | 341.12 | 659.2 | 0 |
| geometry@42 | 10,726.1 | 65.7% | 525.7 | 556.0 | 471.73 | 905.1 | 0 |
| geometry@1337 | 11,236.5 | 68.9% | 505.0 | 533.3 | 414.64 | 789.2 | 3 |
| geometry@2718 | 10,546.8 | 64.6% | 450.7 | 481.1 | 379.51 | 622.1 | 0 |
| geometry@7 | 10,811.5 | 66.3% | 404.8 | 435.2 | 277.24 | 483.1 | 0 |
| geometry@13 | 10,153.0 | 62.2% | 365.5 | 396.0 | 284.08 | 546.4 | 2 |
| geometry@101 | 10,657.7 | 65.3% | 466.2 | 496.7 | 321.02 | 690.8 | 5 |
| geometry@271 | 10,470.6 | 64.2% | 431.9 | 462.2 | 338.91 | 550.7 | 0 |
| geometry@314 | 11,098.7 | 68.0% | 637.7 | 668.1 | 573.43 | 794.8 | 0 |
| geometry@577 | 10,750.8 | 65.9% | 566.2 | 596.6 | 489.61 | 1,028.0 | 0 |
| geometry@863 | 10,762.1 | 66.0% | 488.9 | 519.4 | 435.07 | 992.3 | 0 |
| geometry@1024 | 11,161.3 | 68.4% | 492.0 | 522.1 | 390.06 | 718.2 | 6 |
| geometry@1729 | 10,199.2 | 62.5% | 358.5 | 388.9 | 271.73 | 423.8 | 0 |

Training-target contamination: 0 of 1,047 night rows read above 1 MW (max 0.1 MW); dropped from fit: True.

## FR — catboost, source `energy_renewable`

n_train 26,874 · n_holdout 692 (daylight 314 / shoulder 78 / night 300) · incumbent version 20260201_222014

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 2,551.8 | 40.1% | 852.9 | 959.0 | 97.38 | 3,275.8 | 0 |
| control@42 | 939.6 | 14.8% | 213.3 | 373.2 | 19.17 | 904.4 | 151 |
| control@1337 | 918.0 | 14.4% | 173.2 | 352.5 | 16.27 | 543.8 | 147 |
| control@2718 | 966.4 | 15.2% | 182.3 | 342.6 | 14.10 | 576.4 | 163 |
| control@7 | 941.1 | 14.8% | 199.7 | 351.8 | 30.75 | 581.3 | 141 |
| control@13 | 937.3 | 14.7% | 176.5 | 329.0 | 6.71 | 528.0 | 170 |
| control@101 | 929.6 | 14.6% | 154.9 | 327.1 | 2.88 | 823.2 | 193 |
| control@271 | 968.9 | 15.2% | 205.5 | 365.1 | 27.39 | 706.2 | 129 |
| control@314 | 974.0 | 15.3% | 173.1 | 333.1 | 18.04 | 594.6 | 154 |
| control@577 | 940.4 | 14.8% | 195.4 | 355.1 | 25.18 | 589.6 | 133 |
| control@863 | 944.1 | 14.9% | 201.8 | 371.9 | 20.76 | 798.9 | 152 |
| control@1024 | 970.9 | 15.3% | 196.2 | 373.9 | 12.61 | 461.1 | 170 |
| control@1729 | 951.1 | 15.0% | 202.6 | 358.4 | 10.92 | 670.8 | 153 |
| geometry@42 | 913.3 | 14.4% | 123.1 | 279.7 | 26.48 | 397.8 | 115 |
| geometry@1337 | 924.3 | 14.5% | 165.3 | 324.6 | 27.04 | 500.1 | 123 |
| geometry@2718 | 908.6 | 14.3% | 148.2 | 297.3 | 10.13 | 685.1 | 161 |
| geometry@7 | 904.0 | 14.2% | 174.8 | 316.5 | 18.18 | 488.6 | 132 |
| geometry@13 | 916.2 | 14.4% | 184.6 | 343.8 | 24.42 | 881.0 | 148 |
| geometry@101 | 932.5 | 14.7% | 165.0 | 322.5 | 27.28 | 462.7 | 120 |
| geometry@271 | 914.2 | 14.4% | 158.4 | 321.8 | 21.77 | 602.9 | 124 |
| geometry@314 | 909.1 | 14.3% | 156.9 | 319.8 | 27.46 | 548.2 | 99 |
| geometry@577 | 920.9 | 14.5% | 150.3 | 301.5 | 13.69 | 563.0 | 148 |
| geometry@863 | 948.8 | 14.9% | 162.7 | 317.9 | 30.76 | 590.6 | 127 |
| geometry@1024 | 939.8 | 14.8% | 138.7 | 317.8 | 11.29 | 641.1 | 173 |
| geometry@1729 | 932.0 | 14.7% | 212.7 | 395.2 | 16.75 | 432.5 | 161 |

Training-target contamination: 430 of 10,642 night rows read above 1 MW (max 439.3 MW); dropped from fit: True.
