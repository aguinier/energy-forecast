# ABL-338 — solar non-negativity and solar geometry: held-out A/B

Generated 2026-08-13T13:35:16 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-04-14 .. 2026-05-13**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. Night hours are reported in MW, never as a percentage: their denominator is ~0.

## AT — catboost, source `energy_renewable`

n_train 3,263 · n_holdout 720 (daylight 422 / shoulder 86 / night 212) · incumbent version 20260112_165237

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 728.4 | 34.2% | 0.4 | 0.2 | 0.00 | 0.0 | 0 |
| control@42 | 630.7 | 29.6% | 20.7 | 20.3 | 13.26 | 65.8 | 50 |
| control@1337 | 584.6 | 27.5% | 20.0 | 20.4 | 12.01 | 47.2 | 46 |
| control@2718 | 607.8 | 28.6% | 13.7 | 10.7 | 7.87 | 49.5 | 85 |
| control@7 | 720.6 | 33.8% | 18.4 | 16.4 | 11.76 | 55.7 | 58 |
| control@13 | 595.6 | 28.0% | 17.2 | 16.8 | 7.01 | 52.5 | 67 |
| control@101 | 612.3 | 28.8% | 10.8 | 2.1 | 1.58 | 32.5 | 128 |
| control@271 | 607.7 | 28.5% | 13.3 | 9.9 | 7.26 | 49.6 | 77 |
| control@314 | 653.0 | 30.7% | 15.1 | 13.7 | 5.19 | 54.8 | 97 |
| control@577 | 627.9 | 29.5% | 12.4 | 12.7 | 7.86 | 34.5 | 42 |
| control@863 | 679.5 | 31.9% | 35.0 | 35.4 | 27.60 | 79.4 | 10 |
| control@1024 | 618.1 | 29.0% | 14.9 | 13.6 | 11.46 | 47.6 | 50 |
| control@1729 | 611.6 | 28.7% | 10.4 | 7.6 | 2.53 | 39.8 | 114 |
| geometry@42 | 595.3 | 28.0% | 26.3 | 26.5 | 16.15 | 61.4 | 37 |
| geometry@1337 | 607.0 | 28.5% | 23.5 | 23.8 | 16.09 | 52.2 | 28 |
| geometry@2718 | 590.8 | 27.8% | 20.9 | 20.8 | 16.08 | 50.4 | 37 |
| geometry@7 | 593.3 | 27.9% | 21.7 | 21.6 | 11.90 | 49.9 | 41 |
| geometry@13 | 593.7 | 27.9% | 23.3 | 23.3 | 11.29 | 44.9 | 37 |
| geometry@101 | 596.9 | 28.0% | 29.8 | 30.2 | 24.75 | 58.6 | 6 |
| geometry@271 | 587.9 | 27.6% | 21.8 | 22.2 | 11.19 | 57.3 | 47 |
| geometry@314 | 568.9 | 26.7% | 20.7 | 21.1 | 11.36 | 35.8 | 16 |
| geometry@577 | 598.4 | 28.1% | 23.1 | 19.9 | 11.37 | 57.1 | 51 |
| geometry@863 | 628.2 | 29.5% | 31.2 | 31.3 | 22.04 | 64.4 | 33 |
| geometry@1024 | 566.6 | 26.6% | 11.2 | 10.7 | 7.32 | 44.6 | 73 |
| geometry@1729 | 606.8 | 28.5% | 28.4 | 28.9 | 19.92 | 60.0 | 12 |

Training-target contamination: 0 of 1,522 night rows read above 1 MW (max 0.0 MW); dropped from fit: True.

## BE — catboost, source `energy_renewable`

n_train 19,893 · n_holdout 720 (daylight 436 / shoulder 85 / night 199) · incumbent version 20260201_222022

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,335.9 | 40.4% | 2.7 | 0.3 | 0.00 | 0.0 | 0 |
| control@42 | 629.2 | 19.0% | 32.2 | 5.6 | -11.23 | 57.3 | 167 |
| control@1337 | 653.3 | 19.8% | 25.0 | 14.8 | 2.41 | 58.0 | 133 |
| control@2718 | 638.1 | 19.3% | 29.4 | -1.1 | -7.02 | 57.0 | 164 |
| control@7 | 685.8 | 20.8% | 23.0 | 13.5 | 2.28 | 48.1 | 123 |
| control@13 | 687.7 | 20.8% | 28.5 | -4.7 | -14.76 | 63.9 | 201 |
| control@101 | 655.1 | 19.8% | 26.9 | -9.2 | -19.45 | 42.8 | 196 |
| control@271 | 659.1 | 20.0% | 20.7 | 2.1 | -6.36 | 36.3 | 171 |
| control@314 | 633.3 | 19.2% | 25.2 | 8.7 | -5.08 | 56.6 | 153 |
| control@577 | 636.4 | 19.3% | 24.6 | 15.9 | -1.85 | 52.4 | 125 |
| control@863 | 639.2 | 19.3% | 30.5 | 5.9 | -6.82 | 49.7 | 145 |
| control@1024 | 645.7 | 19.5% | 27.8 | 14.4 | 5.92 | 63.8 | 118 |
| control@1729 | 658.9 | 19.9% | 24.6 | 20.0 | 6.46 | 50.5 | 92 |
| geometry@42 | 640.3 | 19.4% | 21.4 | 10.3 | 3.38 | 60.4 | 127 |
| geometry@1337 | 640.1 | 19.4% | 23.0 | 20.2 | 1.74 | 46.9 | 111 |
| geometry@2718 | 695.8 | 21.1% | 21.8 | 14.8 | 3.35 | 60.6 | 94 |
| geometry@7 | 647.2 | 19.6% | 22.5 | 18.5 | 0.09 | 44.9 | 124 |
| geometry@13 | 650.3 | 19.7% | 22.5 | 20.0 | 4.27 | 53.3 | 101 |
| geometry@101 | 668.5 | 20.2% | 25.2 | 11.3 | -6.82 | 50.3 | 164 |
| geometry@271 | 672.8 | 20.4% | 21.7 | 15.3 | 2.33 | 59.2 | 123 |
| geometry@314 | 655.3 | 19.8% | 21.8 | 12.6 | 1.96 | 46.8 | 116 |
| geometry@577 | 685.9 | 20.8% | 24.0 | 23.0 | -0.17 | 37.8 | 117 |
| geometry@863 | 661.0 | 20.0% | 24.5 | 10.9 | -6.24 | 64.4 | 121 |
| geometry@1024 | 640.3 | 19.4% | 23.1 | 20.5 | 2.94 | 51.6 | 121 |
| geometry@1729 | 675.3 | 20.4% | 29.2 | 11.9 | -8.33 | 50.1 | 121 |

Training-target contamination: 0 of 7,557 night rows read above 1 MW (max 0.1 MW); dropped from fit: True.

## DE — catboost, source `energy_renewable`

n_train 3,367 · n_holdout 720 (daylight 433 / shoulder 90 / night 197) · incumbent version 20260223_193822

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 7,125.5 | 31.0% | 29.6 | 17.1 | 0.01 | 0.4 | 0 |
| control@42 | 5,293.1 | 23.0% | 321.4 | 355.2 | 116.47 | 567.8 | 58 |
| control@1337 | 5,143.3 | 22.4% | 353.1 | 398.4 | 178.75 | 560.0 | 29 |
| control@2718 | 5,208.7 | 22.7% | 273.8 | 311.9 | 134.68 | 594.6 | 39 |
| control@7 | 5,449.1 | 23.7% | 276.0 | 303.4 | 109.16 | 642.6 | 59 |
| control@13 | 5,106.1 | 22.2% | 191.2 | 227.2 | 75.57 | 406.2 | 77 |
| control@101 | 5,053.5 | 22.0% | 335.6 | 380.6 | 218.55 | 725.5 | 26 |
| control@271 | 4,946.3 | 21.5% | 226.8 | 240.6 | 68.58 | 289.3 | 71 |
| control@314 | 5,648.0 | 24.6% | 299.5 | 321.8 | 147.39 | 708.4 | 67 |
| control@577 | 5,443.3 | 23.7% | 255.1 | 288.4 | 119.62 | 490.7 | 67 |
| control@863 | 5,214.8 | 22.7% | 278.7 | 300.2 | 131.86 | 423.6 | 59 |
| control@1024 | 5,640.3 | 24.5% | 198.4 | 238.9 | 93.03 | 433.3 | 42 |
| control@1729 | 5,594.5 | 24.3% | 169.8 | 194.3 | 82.07 | 490.1 | 59 |
| geometry@42 | 5,348.9 | 23.3% | 295.7 | 319.7 | 126.88 | 513.1 | 72 |
| geometry@1337 | 5,198.9 | 22.6% | 328.3 | 366.3 | 193.52 | 672.8 | 50 |
| geometry@2718 | 5,328.5 | 23.2% | 288.4 | 326.0 | 173.88 | 496.4 | 45 |
| geometry@7 | 4,911.2 | 21.4% | 278.3 | 319.0 | 99.63 | 417.3 | 43 |
| geometry@13 | 4,997.0 | 21.7% | 288.4 | 323.3 | 124.77 | 597.2 | 64 |
| geometry@101 | 5,403.6 | 23.5% | 422.3 | 468.5 | 280.28 | 792.6 | 1 |
| geometry@271 | 5,098.0 | 22.2% | 247.4 | 291.8 | 134.72 | 413.5 | 23 |
| geometry@314 | 5,588.4 | 24.3% | 303.8 | 345.4 | 169.34 | 494.1 | 33 |
| geometry@577 | 5,055.2 | 22.0% | 254.0 | 284.5 | 139.14 | 523.6 | 57 |
| geometry@863 | 4,893.5 | 21.3% | 311.1 | 347.1 | 108.29 | 650.3 | 80 |
| geometry@1024 | 4,538.1 | 19.7% | 274.4 | 314.4 | 118.17 | 447.1 | 56 |
| geometry@1729 | 4,923.7 | 21.4% | 297.2 | 343.5 | 174.93 | 534.4 | 25 |

Training-target contamination: 4 of 1,616 night rows read above 1 MW (max 1.7 MW); dropped from fit: True.

## FR — catboost, source `energy_renewable`

n_train 28,252 · n_holdout 720 (daylight 423 / shoulder 75 / night 222) · incumbent version 20260201_222014

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,909.8 | 22.9% | 27.4 | 167.2 | 4.72 | 225.4 | 0 |
| control@42 | 946.6 | 11.3% | 64.9 | 194.2 | 34.85 | 189.5 | 75 |
| control@1337 | 944.7 | 11.3% | 68.5 | 196.2 | 26.93 | 162.7 | 64 |
| control@2718 | 937.6 | 11.2% | 67.6 | 190.0 | 27.51 | 182.7 | 77 |
| control@7 | 946.3 | 11.3% | 80.5 | 221.1 | 55.00 | 281.6 | 42 |
| control@13 | 949.2 | 11.4% | 72.8 | 200.3 | 42.18 | 252.2 | 79 |
| control@101 | 973.2 | 11.7% | 78.0 | 190.4 | 30.73 | 177.4 | 66 |
| control@271 | 933.5 | 11.2% | 65.6 | 196.1 | 26.10 | 160.3 | 74 |
| control@314 | 938.8 | 11.2% | 78.0 | 201.3 | 21.62 | 187.3 | 79 |
| control@577 | 976.6 | 11.7% | 79.4 | 188.7 | 27.92 | 177.1 | 73 |
| control@863 | 940.8 | 11.3% | 73.0 | 216.7 | 31.11 | 168.4 | 72 |
| control@1024 | 997.4 | 12.0% | 89.5 | 184.0 | 25.75 | 152.6 | 72 |
| control@1729 | 955.6 | 11.5% | 90.0 | 207.0 | 43.84 | 236.1 | 52 |
| geometry@42 | 921.4 | 11.0% | 85.9 | 246.7 | 31.46 | 148.9 | 66 |
| geometry@1337 | 902.5 | 10.8% | 84.4 | 238.9 | 45.35 | 184.8 | 28 |
| geometry@2718 | 902.8 | 10.8% | 72.7 | 215.7 | 25.40 | 178.9 | 73 |
| geometry@7 | 917.0 | 11.0% | 80.8 | 232.1 | 43.07 | 167.9 | 42 |
| geometry@13 | 938.9 | 11.3% | 83.9 | 237.7 | 52.09 | 303.3 | 47 |
| geometry@101 | 941.2 | 11.3% | 70.7 | 212.8 | 25.09 | 154.0 | 80 |
| geometry@271 | 920.6 | 11.0% | 73.5 | 222.3 | 31.28 | 157.1 | 60 |
| geometry@314 | 914.1 | 11.0% | 87.4 | 220.7 | 30.12 | 189.5 | 80 |
| geometry@577 | 928.2 | 11.1% | 83.7 | 239.7 | 41.72 | 255.2 | 56 |
| geometry@863 | 898.1 | 10.8% | 77.8 | 206.8 | 36.46 | 183.9 | 57 |
| geometry@1024 | 924.0 | 11.1% | 88.3 | 246.2 | 33.99 | 178.1 | 57 |
| geometry@1729 | 933.1 | 11.2% | 86.1 | 247.4 | 47.11 | 149.0 | 46 |

Training-target contamination: 464 of 11,212 night rows read above 1 MW (max 439.3 MW); dropped from fit: True.
