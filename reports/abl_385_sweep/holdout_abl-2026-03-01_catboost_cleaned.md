# ABL-338 — solar non-negativity and solar geometry: held-out A/B

Generated 2026-08-13T14:30:39 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-07-13 .. 2026-08-11**, training from 2026-03-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. Night hours are reported in MW, never as a percentage: their denominator is ~0.

## AT — catboost, source `energy_renewable`

n_train 2,880 · n_holdout 720 (daylight 448 / shoulder 92 / night 180) · incumbent version 20260112_165237

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 573.1 | 25.6% | 1.1 | 4.5 | 0.00 | 0.0 | 0 |
| control@42 | 327.2 | 14.6% | 37.1 | 40.7 | 18.54 | 76.4 | 20 |
| control@1337 | 325.7 | 14.5% | 28.5 | 32.1 | 19.14 | 71.6 | 19 |
| control@2718 | 328.9 | 14.7% | 25.6 | 28.2 | 16.97 | 101.1 | 64 |
| control@7 | 343.1 | 15.3% | 40.0 | 42.9 | 24.05 | 111.7 | 25 |
| control@13 | 320.5 | 14.3% | 42.3 | 44.0 | 24.68 | 156.0 | 52 |
| control@101 | 323.5 | 14.4% | 46.8 | 50.1 | 21.28 | 116.6 | 56 |
| control@271 | 310.1 | 13.8% | 20.6 | 19.6 | 9.17 | 64.3 | 66 |
| control@314 | 333.2 | 14.9% | 54.4 | 57.8 | 29.22 | 137.9 | 41 |
| control@577 | 325.9 | 14.5% | 54.8 | 55.3 | 38.74 | 118.7 | 49 |
| control@863 | 331.4 | 14.8% | 52.0 | 54.6 | 33.45 | 164.2 | 25 |
| control@1024 | 332.0 | 14.8% | 22.3 | 20.6 | 3.84 | 99.4 | 127 |
| control@1729 | 336.6 | 15.0% | 27.3 | 29.2 | 14.66 | 56.8 | 53 |
| geometry@42 | 306.9 | 13.7% | 25.8 | 22.8 | 7.72 | 66.6 | 92 |
| geometry@1337 | 322.5 | 14.4% | 27.8 | 29.6 | 10.82 | 80.2 | 67 |
| geometry@2718 | 307.9 | 13.7% | 30.5 | 28.0 | 17.06 | 83.8 | 99 |
| geometry@7 | 313.0 | 14.0% | 18.8 | 17.2 | 6.10 | 72.8 | 94 |
| geometry@13 | 303.2 | 13.5% | 21.6 | 19.5 | 8.73 | 55.2 | 73 |
| geometry@101 | 313.0 | 14.0% | 32.2 | 35.8 | 12.95 | 59.7 | 36 |
| geometry@271 | 314.0 | 14.0% | 28.5 | 28.9 | 21.57 | 74.3 | 37 |
| geometry@314 | 314.6 | 14.0% | 19.8 | 18.5 | 4.77 | 108.0 | 118 |
| geometry@577 | 310.7 | 13.9% | 30.2 | 30.7 | 11.73 | 102.7 | 70 |
| geometry@863 | 330.9 | 14.8% | 50.9 | 52.0 | 36.03 | 111.6 | 34 |
| geometry@1024 | 313.2 | 14.0% | 39.6 | 39.0 | 15.45 | 74.1 | 64 |
| geometry@1729 | 308.8 | 13.8% | 44.5 | 44.1 | 21.36 | 147.4 | 66 |

Training-target contamination: 0 of 836 night rows read above 1 MW (max 0.0 MW); dropped from fit: True.

## BE — catboost, source `energy_renewable`

n_train 2,880 · n_holdout 720 (daylight 465 / shoulder 88 / night 167) · incumbent version 20260201_222022

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,090.1 | 32.9% | 3.3 | 5.1 | 0.00 | 0.0 | 0 |
| control@42 | 582.2 | 17.6% | 75.6 | 76.5 | 40.88 | 128.5 | 26 |
| control@1337 | 606.4 | 18.3% | 88.3 | 89.5 | 41.58 | 113.9 | 21 |
| control@2718 | 601.7 | 18.1% | 83.9 | 85.7 | 64.48 | 148.1 | 6 |
| control@7 | 615.4 | 18.6% | 87.7 | 88.7 | 60.64 | 147.0 | 19 |
| control@13 | 597.2 | 18.0% | 65.2 | 67.0 | 32.61 | 97.9 | 14 |
| control@101 | 563.2 | 17.0% | 60.4 | 61.5 | 35.34 | 154.2 | 19 |
| control@271 | 598.0 | 18.0% | 123.5 | 123.8 | 95.07 | 259.7 | 38 |
| control@314 | 619.7 | 18.7% | 121.5 | 123.3 | 95.77 | 184.8 | 0 |
| control@577 | 617.6 | 18.6% | 77.6 | 79.3 | 65.82 | 167.5 | 13 |
| control@863 | 605.8 | 18.3% | 78.7 | 80.4 | 54.98 | 136.2 | 0 |
| control@1024 | 601.9 | 18.1% | 95.7 | 97.5 | 54.24 | 122.9 | 6 |
| control@1729 | 557.8 | 16.8% | 91.1 | 92.9 | 62.30 | 140.7 | 6 |
| geometry@42 | 576.4 | 17.4% | 43.5 | 35.1 | 11.65 | 112.3 | 117 |
| geometry@1337 | 566.2 | 17.1% | 53.1 | 52.9 | 31.96 | 205.9 | 50 |
| geometry@2718 | 571.3 | 17.2% | 87.9 | 89.3 | 62.92 | 330.8 | 13 |
| geometry@7 | 610.7 | 18.4% | 103.3 | 105.1 | 68.82 | 164.9 | 0 |
| geometry@13 | 578.6 | 17.4% | 60.1 | 61.9 | 30.13 | 138.8 | 13 |
| geometry@101 | 574.9 | 17.3% | 76.6 | 78.4 | 52.48 | 140.5 | 4 |
| geometry@271 | 587.0 | 17.7% | 107.3 | 108.9 | 78.58 | 196.6 | 12 |
| geometry@314 | 564.2 | 17.0% | 80.5 | 79.4 | 43.55 | 145.7 | 42 |
| geometry@577 | 551.3 | 16.6% | 78.2 | 80.0 | 57.22 | 110.7 | 4 |
| geometry@863 | 601.5 | 18.1% | 95.1 | 96.9 | 58.66 | 205.5 | 9 |
| geometry@1024 | 618.0 | 18.6% | 121.4 | 121.5 | 85.79 | 265.9 | 15 |
| geometry@1729 | 615.9 | 18.6% | 146.4 | 148.2 | 112.75 | 241.1 | 0 |

Training-target contamination: 0 of 731 night rows read above 1 MW (max 0.0 MW); dropped from fit: True.

## DE — catboost, source `energy_renewable`

n_train 2,878 · n_holdout 720 (daylight 457 / shoulder 99 / night 164) · incumbent version 20260223_193822

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 6,066.6 | 23.0% | 33.5 | 66.2 | 0.00 | 0.0 | 0 |
| control@42 | 3,394.9 | 12.9% | 408.5 | 435.2 | 229.92 | 776.8 | 36 |
| control@1337 | 3,311.0 | 12.6% | 375.1 | 396.8 | 209.78 | 902.5 | 49 |
| control@2718 | 3,492.3 | 13.2% | 180.2 | 198.6 | 53.15 | 608.3 | 61 |
| control@7 | 3,238.6 | 12.3% | 325.3 | 331.4 | 30.07 | 366.1 | 84 |
| control@13 | 3,514.4 | 13.3% | 505.4 | 512.8 | 301.43 | 1,667.5 | 55 |
| control@101 | 3,412.2 | 12.9% | 360.0 | 364.2 | 184.90 | 1,124.0 | 59 |
| control@271 | 3,534.5 | 13.4% | 363.3 | 388.4 | 222.19 | 876.1 | 21 |
| control@314 | 3,355.5 | 12.7% | 312.1 | 321.9 | 190.41 | 826.5 | 44 |
| control@577 | 3,526.5 | 13.4% | 618.7 | 637.8 | 407.60 | 1,159.6 | 33 |
| control@863 | 3,591.2 | 13.6% | 383.5 | 410.1 | 237.37 | 724.9 | 25 |
| control@1024 | 3,305.1 | 12.5% | 257.5 | 234.4 | 27.59 | 479.4 | 109 |
| control@1729 | 3,274.5 | 12.4% | 239.8 | 243.9 | 7.62 | 428.5 | 105 |
| geometry@42 | 3,296.4 | 12.5% | 207.1 | 214.2 | -30.38 | 359.7 | 130 |
| geometry@1337 | 3,251.2 | 12.3% | 293.9 | 266.7 | 67.01 | 821.2 | 98 |
| geometry@2718 | 3,338.9 | 12.7% | 248.3 | 244.7 | -13.70 | 303.8 | 120 |
| geometry@7 | 3,687.5 | 14.0% | 689.2 | 712.7 | 578.91 | 1,835.9 | 25 |
| geometry@13 | 3,307.6 | 12.5% | 224.6 | 226.9 | 73.08 | 512.2 | 66 |
| geometry@101 | 3,189.2 | 12.1% | 230.6 | 227.2 | 94.71 | 562.7 | 66 |
| geometry@271 | 3,117.9 | 11.8% | 296.9 | 283.4 | 125.25 | 631.6 | 67 |
| geometry@314 | 3,392.7 | 12.9% | 418.8 | 432.7 | 226.07 | 794.5 | 33 |
| geometry@577 | 3,129.3 | 11.9% | 334.3 | 347.8 | 133.03 | 798.7 | 71 |
| geometry@863 | 3,353.4 | 12.7% | 258.1 | 277.9 | 119.68 | 813.2 | 47 |
| geometry@1024 | 3,484.0 | 13.2% | 487.1 | 514.5 | 328.28 | 1,146.8 | 34 |
| geometry@1729 | 3,339.8 | 12.7% | 327.7 | 337.4 | 199.30 | 842.8 | 39 |

Training-target contamination: 2 of 731 night rows read above 1 MW (max 1.1 MW); dropped from fit: True.

## FR — catboost, source `energy_renewable`

n_train 2,808 · n_holdout 720 (daylight 435 / shoulder 99 / night 186) · incumbent version 20260201_222014

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 2,368.6 | 22.3% | 50.8 | 270.4 | 34.88 | 321.5 | 0 |
| control@42 | 1,833.4 | 17.2% | 179.8 | 385.5 | 214.85 | 503.6 | 0 |
| control@1337 | 1,922.8 | 18.1% | 359.9 | 603.8 | 340.47 | 893.4 | 0 |
| control@2718 | 1,776.6 | 16.7% | 141.3 | 364.7 | 183.71 | 718.3 | 0 |
| control@7 | 1,847.3 | 17.4% | 169.1 | 379.9 | 185.59 | 697.5 | 4 |
| control@13 | 2,044.2 | 19.2% | 467.1 | 711.0 | 509.78 | 832.6 | 0 |
| control@101 | 1,722.7 | 16.2% | 204.6 | 418.3 | 164.72 | 577.2 | 1 |
| control@271 | 1,840.8 | 17.3% | 302.9 | 545.4 | 319.84 | 1,035.2 | 0 |
| control@314 | 1,994.8 | 18.7% | 274.5 | 515.4 | 294.33 | 691.5 | 0 |
| control@577 | 1,840.2 | 17.3% | 293.0 | 533.6 | 303.32 | 661.1 | 0 |
| control@863 | 1,775.8 | 16.7% | 219.3 | 454.6 | 226.20 | 627.8 | 0 |
| control@1024 | 1,672.5 | 15.7% | 142.7 | 357.7 | 136.64 | 467.6 | 8 |
| control@1729 | 1,732.1 | 16.3% | 233.9 | 473.1 | 209.28 | 790.5 | 10 |
| geometry@42 | 1,707.8 | 16.1% | 182.6 | 400.5 | 240.13 | 914.0 | 0 |
| geometry@1337 | 1,830.8 | 17.2% | 260.7 | 502.8 | 253.41 | 688.2 | 0 |
| geometry@2718 | 1,781.0 | 16.7% | 213.1 | 415.6 | 123.21 | 486.6 | 26 |
| geometry@7 | 1,876.2 | 17.6% | 355.2 | 597.0 | 322.43 | 954.3 | 2 |
| geometry@13 | 1,679.7 | 15.8% | 158.5 | 362.0 | 182.24 | 664.3 | 1 |
| geometry@101 | 1,756.5 | 16.5% | 192.9 | 423.4 | 225.92 | 527.3 | 0 |
| geometry@271 | 1,735.8 | 16.3% | 186.1 | 378.0 | 147.36 | 587.8 | 15 |
| geometry@314 | 1,765.5 | 16.6% | 189.5 | 421.5 | 197.64 | 482.7 | 0 |
| geometry@577 | 1,572.8 | 14.8% | 131.1 | 308.8 | 86.29 | 383.4 | 18 |
| geometry@863 | 1,682.3 | 15.8% | 177.9 | 395.8 | 157.10 | 772.6 | 10 |
| geometry@1024 | 1,797.1 | 16.9% | 245.4 | 479.7 | 266.17 | 798.4 | 0 |
| geometry@1729 | 1,737.1 | 16.3% | 194.3 | 415.9 | 216.95 | 570.8 | 0 |

Training-target contamination: 72 of 852 night rows read above 1 MW (max 285.6 MW); dropped from fit: True.
