# ABL-338 — solar non-negativity and solar geometry: held-out A/B

Generated 2026-08-13T13:24:21 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-03-15 .. 2026-04-13**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. Night hours are reported in MW, never as a percentage: their denominator is ~0.

## AT — catboost, source `energy_renewable`

n_train 2,543 · n_holdout 720 (daylight 374 / shoulder 82 / night 264) · incumbent version 20260112_165237

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 873.5 | 62.7% | 0.3 | 0.0 | 0.00 | 0.0 | 0 |
| control@42 | 1,018.4 | 73.2% | 12.0 | 12.3 | 9.75 | 19.0 | 0 |
| control@1337 | 1,001.5 | 71.9% | 3.8 | 1.8 | 0.03 | 9.6 | 164 |
| control@2718 | 1,017.0 | 73.1% | 6.0 | 6.1 | 4.87 | 21.6 | 10 |
| control@7 | 1,020.4 | 73.3% | 10.3 | 10.5 | 8.13 | 30.3 | 0 |
| control@13 | 1,039.9 | 74.7% | 15.2 | 15.4 | 14.18 | 25.1 | 0 |
| control@101 | 1,007.9 | 72.4% | 13.0 | 13.2 | 9.47 | 25.1 | 0 |
| control@271 | 992.2 | 71.3% | 12.1 | 12.3 | 8.57 | 22.9 | 0 |
| control@314 | 1,002.4 | 72.0% | 11.2 | 11.4 | 6.49 | 15.9 | 0 |
| control@577 | 1,009.9 | 72.5% | 5.6 | 5.1 | 1.80 | 15.6 | 124 |
| control@863 | 1,002.7 | 72.0% | 7.0 | 7.1 | 3.92 | 18.2 | 33 |
| control@1024 | 1,011.2 | 72.6% | 11.0 | 11.1 | 10.05 | 29.0 | 3 |
| control@1729 | 1,008.0 | 72.4% | 6.9 | -1.0 | -3.53 | 20.3 | 219 |
| geometry@42 | 996.2 | 71.6% | 7.9 | 8.1 | 5.40 | 13.9 | 7 |
| geometry@1337 | 1,004.5 | 72.2% | 10.1 | 10.4 | 7.86 | 21.0 | 0 |
| geometry@2718 | 967.8 | 69.5% | 7.8 | 8.0 | 5.38 | 15.7 | 19 |
| geometry@7 | 971.2 | 69.8% | 10.0 | 9.9 | 4.78 | 21.8 | 44 |
| geometry@13 | 990.6 | 71.2% | 9.5 | 9.7 | 5.97 | 18.1 | 12 |
| geometry@101 | 972.9 | 69.9% | 6.5 | 6.6 | 5.28 | 22.6 | 11 |
| geometry@271 | 994.7 | 71.5% | 7.5 | 7.7 | 3.84 | 14.0 | 4 |
| geometry@314 | 975.2 | 70.0% | 18.0 | 18.2 | 15.56 | 29.9 | 0 |
| geometry@577 | 976.2 | 70.1% | 3.8 | 3.4 | 2.01 | 17.3 | 84 |
| geometry@863 | 979.6 | 70.4% | 13.0 | 13.2 | 9.23 | 34.1 | 0 |
| geometry@1024 | 993.8 | 71.4% | 6.5 | 6.4 | 1.73 | 11.5 | 107 |
| geometry@1729 | 970.0 | 69.7% | 10.2 | 10.5 | 7.30 | 29.2 | 0 |

Training-target contamination: 0 of 1,258 night rows read above 1 MW (max 0.0 MW); dropped from fit: True.

## BE — catboost, source `energy_renewable`

n_train 19,173 · n_holdout 720 (daylight 375 / shoulder 83 / night 262) · incumbent version 20260201_222022

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,528.4 | 48.7% | 4.4 | 0.8 | 0.00 | 0.0 | 0 |
| control@42 | 622.5 | 19.8% | 27.6 | 18.6 | 3.69 | 73.0 | 130 |
| control@1337 | 664.0 | 21.2% | 31.0 | 12.2 | -6.56 | 59.4 | 193 |
| control@2718 | 624.0 | 19.9% | 29.4 | 6.2 | -5.43 | 54.3 | 197 |
| control@7 | 663.1 | 21.1% | 26.1 | 1.7 | -9.06 | 74.1 | 215 |
| control@13 | 646.3 | 20.6% | 25.9 | 10.9 | -2.48 | 82.7 | 168 |
| control@101 | 646.7 | 20.6% | 26.7 | 8.1 | 0.61 | 81.3 | 157 |
| control@271 | 654.9 | 20.9% | 30.6 | 22.1 | 2.02 | 62.0 | 147 |
| control@314 | 639.4 | 20.4% | 25.7 | 9.8 | -2.07 | 71.2 | 180 |
| control@577 | 632.9 | 20.2% | 26.4 | 13.9 | -3.83 | 61.5 | 183 |
| control@863 | 660.0 | 21.0% | 31.3 | 0.8 | -7.36 | 110.8 | 207 |
| control@1024 | 633.3 | 20.2% | 32.4 | 11.5 | -4.59 | 81.3 | 177 |
| control@1729 | 624.7 | 19.9% | 28.5 | 9.0 | -4.70 | 78.5 | 193 |
| geometry@42 | 666.0 | 21.2% | 24.4 | 1.3 | -9.14 | 38.2 | 213 |
| geometry@1337 | 672.2 | 21.4% | 25.6 | 13.8 | -1.57 | 64.8 | 162 |
| geometry@2718 | 673.0 | 21.5% | 26.5 | 8.6 | -3.67 | 77.4 | 172 |
| geometry@7 | 672.6 | 21.4% | 29.1 | 12.4 | -5.95 | 63.2 | 194 |
| geometry@13 | 676.8 | 21.6% | 28.2 | 15.6 | -6.91 | 72.8 | 192 |
| geometry@101 | 662.1 | 21.1% | 28.6 | 15.1 | -3.30 | 45.4 | 152 |
| geometry@271 | 685.7 | 21.9% | 29.7 | 9.0 | -1.70 | 100.2 | 185 |
| geometry@314 | 663.3 | 21.1% | 24.1 | 10.6 | -3.30 | 46.2 | 164 |
| geometry@577 | 673.8 | 21.5% | 26.0 | 9.0 | -7.66 | 47.4 | 186 |
| geometry@863 | 676.4 | 21.6% | 24.9 | 5.2 | -3.03 | 77.9 | 170 |
| geometry@1024 | 716.1 | 22.8% | 34.1 | 24.9 | -0.52 | 83.4 | 160 |
| geometry@1729 | 708.2 | 22.6% | 26.3 | 14.9 | 2.11 | 78.8 | 142 |

Training-target contamination: 0 of 7,295 night rows read above 1 MW (max 0.1 MW); dropped from fit: True.

## DE — catboost, source `energy_renewable`

n_train 2,649 · n_holdout 720 (daylight 377 / shoulder 83 / night 260) · incumbent version 20260223_193822

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 8,629.4 | 42.5% | 46.1 | 17.0 | 0.06 | 1.1 | 0 |
| control@42 | 10,000.0 | 49.3% | 406.8 | 468.4 | 309.06 | 507.3 | 0 |
| control@1337 | 9,880.9 | 48.7% | 340.1 | 399.9 | 277.91 | 501.0 | 0 |
| control@2718 | 9,877.8 | 48.7% | 337.8 | 396.6 | 219.81 | 566.1 | 0 |
| control@7 | 10,220.5 | 50.4% | 317.8 | 379.4 | 214.69 | 631.7 | 0 |
| control@13 | 9,586.3 | 47.2% | 409.4 | 464.9 | 284.24 | 685.9 | 3 |
| control@101 | 9,647.0 | 47.5% | 294.9 | 347.4 | 171.81 | 534.5 | 5 |
| control@271 | 9,950.2 | 49.0% | 275.3 | 333.3 | 199.47 | 475.4 | 5 |
| control@314 | 9,939.7 | 49.0% | 257.2 | 314.6 | 212.32 | 477.1 | 0 |
| control@577 | 9,602.0 | 47.3% | 242.3 | 293.8 | 174.65 | 464.7 | 8 |
| control@863 | 9,542.3 | 47.0% | 299.5 | 357.5 | 168.61 | 361.7 | 0 |
| control@1024 | 10,562.7 | 52.1% | 412.0 | 473.6 | 350.22 | 599.3 | 0 |
| control@1729 | 9,949.1 | 49.0% | 253.5 | 306.8 | 173.03 | 365.7 | 2 |
| geometry@42 | 9,647.7 | 47.5% | 580.7 | 642.3 | 489.09 | 794.2 | 0 |
| geometry@1337 | 9,426.5 | 46.5% | 419.0 | 477.2 | 346.13 | 786.6 | 0 |
| geometry@2718 | 8,749.4 | 43.1% | 321.3 | 381.4 | 238.39 | 566.9 | 8 |
| geometry@7 | 8,950.4 | 44.1% | 367.9 | 429.5 | 317.42 | 469.7 | 0 |
| geometry@13 | 8,611.7 | 42.4% | 325.1 | 386.7 | 277.08 | 638.6 | 0 |
| geometry@101 | 9,455.7 | 46.6% | 423.8 | 480.9 | 357.82 | 821.1 | 7 |
| geometry@271 | 9,517.8 | 46.9% | 363.5 | 424.0 | 268.32 | 554.9 | 0 |
| geometry@314 | 9,365.0 | 46.1% | 341.3 | 402.9 | 283.16 | 481.3 | 2 |
| geometry@577 | 9,197.7 | 45.3% | 344.8 | 405.1 | 329.40 | 657.0 | 0 |
| geometry@863 | 8,752.0 | 43.1% | 308.8 | 366.1 | 225.53 | 735.3 | 5 |
| geometry@1024 | 9,769.7 | 48.1% | 541.4 | 603.0 | 491.86 | 868.5 | 0 |
| geometry@1729 | 9,114.2 | 44.9% | 394.6 | 456.2 | 300.83 | 573.1 | 0 |

Training-target contamination: 2 of 1,356 night rows read above 1 MW (max 1.7 MW); dropped from fit: True.

## FR — catboost, source `energy_renewable`

n_train 27,551 · n_holdout 720 (daylight 370 / shoulder 80 / night 270) · incumbent version 20260201_222014

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 2,690.1 | 30.8% | 73.6 | 184.4 | 14.19 | 259.8 | 0 |
| control@42 | 1,149.7 | 13.1% | 122.6 | 290.5 | 14.29 | 227.6 | 120 |
| control@1337 | 1,193.2 | 13.6% | 92.2 | 240.2 | -11.71 | 216.9 | 158 |
| control@2718 | 1,205.1 | 13.8% | 97.9 | 305.4 | 53.18 | 291.7 | 51 |
| control@7 | 1,265.3 | 14.5% | 92.1 | 262.6 | 25.61 | 245.0 | 90 |
| control@13 | 1,267.4 | 14.5% | 97.9 | 255.0 | 33.92 | 333.9 | 97 |
| control@101 | 1,217.0 | 13.9% | 117.1 | 275.3 | 27.12 | 339.2 | 113 |
| control@271 | 1,189.5 | 13.6% | 93.0 | 231.7 | -26.20 | 219.3 | 188 |
| control@314 | 1,151.9 | 13.2% | 93.3 | 236.2 | 6.47 | 231.2 | 124 |
| control@577 | 1,191.5 | 13.6% | 92.5 | 244.2 | 10.44 | 209.0 | 112 |
| control@863 | 1,187.9 | 13.6% | 110.9 | 300.5 | 19.66 | 277.4 | 109 |
| control@1024 | 1,209.8 | 13.8% | 102.7 | 254.3 | -1.02 | 199.9 | 141 |
| control@1729 | 1,155.7 | 13.2% | 85.1 | 262.5 | 7.99 | 182.2 | 149 |
| geometry@42 | 1,121.1 | 12.8% | 105.7 | 278.7 | 11.55 | 185.0 | 110 |
| geometry@1337 | 1,208.0 | 13.8% | 102.7 | 243.8 | 49.74 | 220.1 | 61 |
| geometry@2718 | 1,118.0 | 12.8% | 89.0 | 276.6 | 23.57 | 154.3 | 91 |
| geometry@7 | 1,157.8 | 13.2% | 85.2 | 273.0 | 9.39 | 204.3 | 117 |
| geometry@13 | 1,180.3 | 13.5% | 109.6 | 239.8 | 18.49 | 143.0 | 90 |
| geometry@101 | 1,166.2 | 13.3% | 91.7 | 252.5 | 25.16 | 171.3 | 88 |
| geometry@271 | 1,143.1 | 13.1% | 109.9 | 292.7 | 35.58 | 238.6 | 63 |
| geometry@314 | 1,123.7 | 12.8% | 99.0 | 287.9 | 43.28 | 204.5 | 53 |
| geometry@577 | 1,158.1 | 13.2% | 106.1 | 282.9 | 4.17 | 142.0 | 126 |
| geometry@863 | 1,157.6 | 13.2% | 89.0 | 259.6 | 12.39 | 189.1 | 117 |
| geometry@1024 | 1,175.5 | 13.4% | 93.6 | 281.0 | 20.12 | 200.0 | 103 |
| geometry@1729 | 1,133.1 | 13.0% | 96.0 | 277.7 | 8.66 | 150.2 | 117 |

Training-target contamination: 445 of 10,942 night rows read above 1 MW (max 439.3 MW); dropped from fit: True.
