# ABL-338 — solar non-negativity and solar geometry: held-out A/B

Generated 2026-08-13T13:39:04 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-04-14 .. 2026-05-13**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. Night hours are reported in MW, never as a percentage: their denominator is ~0.

## AT — xgboost, source `energy_renewable`

n_train 3,263 · n_holdout 720 (daylight 422 / shoulder 86 / night 212) · incumbent version 20260112_165237

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 728.4 | 34.2% | 0.4 | 0.2 | 0.00 | 0.0 | 0 |
| control@42 | 498.0 | 23.4% | 6.1 | 6.2 | 3.12 | 10.9 | 74 |
| control@1337 | 532.4 | 25.0% | 5.5 | 5.8 | 1.84 | 4.9 | 69 |
| control@2718 | 520.5 | 24.4% | 2.6 | 2.9 | 0.91 | 5.8 | 42 |
| control@7 | 514.8 | 24.2% | 1.7 | 1.6 | 0.74 | 16.6 | 68 |
| control@13 | 515.0 | 24.2% | 2.6 | 2.0 | 0.64 | 5.2 | 99 |
| control@101 | 519.6 | 24.4% | 3.5 | 3.8 | 1.06 | 5.5 | 65 |
| control@271 | 512.5 | 24.1% | 5.0 | 5.2 | 2.14 | 6.8 | 59 |
| control@314 | 528.4 | 24.8% | 1.6 | 0.9 | 0.23 | 8.7 | 116 |
| control@577 | 509.4 | 23.9% | 5.7 | 5.9 | 2.12 | 76.5 | 92 |
| control@863 | 520.6 | 24.5% | 3.8 | 3.7 | 0.95 | 5.3 | 50 |
| control@1024 | 518.1 | 24.3% | 1.8 | 1.5 | 0.41 | 2.0 | 72 |
| control@1729 | 512.1 | 24.1% | 5.1 | 4.5 | 2.96 | 44.7 | 64 |
| geometry@42 | 492.9 | 23.2% | 3.1 | 1.6 | -0.45 | 13.0 | 190 |
| geometry@1337 | 497.5 | 23.4% | 6.2 | 6.2 | 3.64 | 15.5 | 80 |
| geometry@2718 | 508.2 | 23.9% | 6.2 | 5.8 | 3.90 | 15.7 | 91 |
| geometry@7 | 495.0 | 23.3% | 2.2 | 1.7 | 0.39 | 7.7 | 140 |
| geometry@13 | 502.6 | 23.6% | 3.4 | 3.4 | 1.51 | 13.0 | 57 |
| geometry@101 | 500.9 | 23.5% | 4.6 | 4.6 | 3.02 | 16.6 | 48 |
| geometry@271 | 486.3 | 22.8% | 3.7 | 3.8 | 1.93 | 10.6 | 51 |
| geometry@314 | 514.6 | 24.2% | 4.6 | 4.5 | 2.47 | 19.3 | 76 |
| geometry@577 | 501.9 | 23.6% | 3.8 | 2.0 | 0.30 | 13.2 | 104 |
| geometry@863 | 502.3 | 23.6% | 3.9 | 3.5 | 0.67 | 42.1 | 148 |
| geometry@1024 | 505.8 | 23.8% | 4.4 | 3.9 | 1.34 | 15.0 | 30 |
| geometry@1729 | 500.9 | 23.5% | 1.7 | 1.4 | 0.27 | 4.5 | 81 |

Training-target contamination: 0 of 1,522 night rows read above 1 MW (max 0.0 MW); dropped from fit: True.

## BE — xgboost, source `energy_renewable`

n_train 19,893 · n_holdout 720 (daylight 436 / shoulder 85 / night 199) · incumbent version 20260201_222022

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,335.9 | 40.4% | 2.7 | 0.3 | 0.00 | 0.0 | 0 |
| control@42 | 743.6 | 22.5% | 12.9 | -9.2 | -9.09 | 2.0 | 226 |
| control@1337 | 757.4 | 22.9% | 10.8 | 12.1 | 9.14 | 9.2 | 0 |
| control@2718 | 789.9 | 23.9% | 11.7 | -1.5 | -2.20 | 39.6 | 181 |
| control@7 | 768.0 | 23.2% | 6.9 | -1.7 | 0.23 | 20.1 | 116 |
| control@13 | 726.0 | 22.0% | 5.6 | 6.6 | 3.54 | 3.7 | 0 |
| control@101 | 824.6 | 25.0% | 8.5 | -0.5 | -1.14 | 8.9 | 154 |
| control@271 | 746.3 | 22.6% | 6.3 | 2.2 | -1.17 | 4.7 | 108 |
| control@314 | 765.2 | 23.2% | 5.3 | -0.2 | -2.00 | 1.6 | 178 |
| control@577 | 749.1 | 22.7% | 8.1 | 9.2 | 4.87 | 6.7 | 0 |
| control@863 | 748.8 | 22.7% | 4.7 | 1.9 | 0.88 | 6.2 | 60 |
| control@1024 | 738.6 | 22.4% | 4.3 | 4.2 | 2.33 | 3.4 | 14 |
| control@1729 | 748.4 | 22.7% | 6.2 | 5.9 | 2.43 | 9.0 | 48 |
| geometry@42 | 744.6 | 22.5% | 4.5 | 1.2 | -1.01 | 2.7 | 151 |
| geometry@1337 | 772.5 | 23.4% | 4.2 | 0.8 | 0.56 | 2.2 | 88 |
| geometry@2718 | 779.8 | 23.6% | 8.9 | -0.8 | 0.59 | 10.4 | 103 |
| geometry@7 | 763.3 | 23.1% | 7.9 | 0.6 | -6.94 | 4.5 | 222 |
| geometry@13 | 768.4 | 23.3% | 8.1 | -1.0 | -4.10 | 4.8 | 98 |
| geometry@101 | 755.2 | 22.9% | 9.2 | 1.2 | -0.80 | 28.5 | 113 |
| geometry@271 | 741.4 | 22.4% | 8.1 | 10.2 | 5.47 | 5.5 | 0 |
| geometry@314 | 753.1 | 22.8% | 8.1 | -3.3 | -2.39 | 2.7 | 176 |
| geometry@577 | 752.1 | 22.8% | 5.7 | 6.9 | 1.90 | 2.6 | 1 |
| geometry@863 | 751.2 | 22.7% | 5.1 | 5.3 | 3.28 | 4.0 | 1 |
| geometry@1024 | 761.0 | 23.0% | 7.1 | 1.1 | -1.40 | 11.3 | 113 |
| geometry@1729 | 761.6 | 23.1% | 5.4 | 1.1 | -0.14 | 12.9 | 89 |

Training-target contamination: 0 of 7,557 night rows read above 1 MW (max 0.1 MW); dropped from fit: True.

## DE — xgboost, source `energy_renewable`

n_train 3,367 · n_holdout 720 (daylight 433 / shoulder 90 / night 197) · incumbent version 20260223_193822

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 7,125.5 | 31.0% | 29.6 | 17.1 | 0.01 | 0.4 | 0 |
| control@42 | 4,166.5 | 18.1% | 69.8 | 96.9 | 7.78 | 94.2 | 106 |
| control@1337 | 4,359.6 | 19.0% | 69.4 | 46.4 | -26.19 | 11.3 | 235 |
| control@2718 | 4,374.0 | 19.0% | 54.3 | 70.1 | -3.55 | 39.1 | 105 |
| control@7 | 4,233.6 | 18.4% | 71.7 | 74.5 | -29.56 | 12.8 | 216 |
| control@13 | 4,184.4 | 18.2% | 73.2 | 83.1 | -13.19 | 23.6 | 166 |
| control@101 | 4,660.0 | 20.3% | 75.8 | 85.4 | -15.93 | 18.4 | 170 |
| control@271 | 4,334.6 | 18.9% | 78.9 | 94.3 | -5.75 | 23.3 | 138 |
| control@314 | 4,523.9 | 19.7% | 67.1 | 91.3 | 2.53 | 96.5 | 135 |
| control@577 | 4,672.7 | 20.3% | 71.8 | 46.7 | -42.35 | -0.8 | 257 |
| control@863 | 4,674.8 | 20.3% | 70.3 | 84.7 | -17.11 | 20.5 | 185 |
| control@1024 | 4,514.7 | 19.6% | 74.6 | 26.9 | -40.68 | 9.2 | 250 |
| control@1729 | 4,447.9 | 19.3% | 78.3 | 84.9 | -26.83 | 2.7 | 236 |
| geometry@42 | 4,396.1 | 19.1% | 66.7 | 6.3 | -51.21 | 14.0 | 237 |
| geometry@1337 | 4,733.6 | 20.6% | 45.5 | 71.2 | -8.61 | 22.4 | 134 |
| geometry@2718 | 4,754.0 | 20.7% | 37.2 | 45.2 | -16.54 | 25.9 | 188 |
| geometry@7 | 4,311.6 | 18.8% | 42.0 | 29.4 | -36.87 | 6.5 | 226 |
| geometry@13 | 4,442.6 | 19.3% | 53.8 | 62.3 | -19.94 | 18.1 | 225 |
| geometry@101 | 4,636.6 | 20.2% | 40.7 | 52.9 | -10.48 | 24.1 | 120 |
| geometry@271 | 4,580.9 | 19.9% | 45.3 | 54.3 | -19.55 | 52.1 | 189 |
| geometry@314 | 4,611.2 | 20.1% | 32.0 | 52.1 | -16.18 | 23.7 | 181 |
| geometry@577 | 4,759.4 | 20.7% | 41.9 | 44.2 | -20.69 | 8.6 | 218 |
| geometry@863 | 4,649.1 | 20.2% | 43.2 | 42.0 | -26.60 | 12.2 | 217 |
| geometry@1024 | 4,711.8 | 20.5% | 45.4 | 53.8 | -11.27 | 45.2 | 163 |
| geometry@1729 | 4,573.0 | 19.9% | 55.5 | 66.3 | -21.81 | 23.3 | 196 |

Training-target contamination: 4 of 1,616 night rows read above 1 MW (max 1.7 MW); dropped from fit: True.

## FR — xgboost, source `energy_renewable`

n_train 28,252 · n_holdout 720 (daylight 423 / shoulder 75 / night 222) · incumbent version 20260201_222014

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,909.8 | 22.9% | 27.4 | 167.2 | 4.72 | 225.4 | 0 |
| control@42 | 952.6 | 11.4% | 27.6 | 180.9 | 15.05 | 190.1 | 59 |
| control@1337 | 924.3 | 11.1% | 23.4 | 185.1 | 11.92 | 178.3 | 45 |
| control@2718 | 959.9 | 11.5% | 32.1 | 192.2 | 9.92 | 195.2 | 63 |
| control@7 | 949.3 | 11.4% | 23.0 | 177.6 | 10.60 | 193.3 | 0 |
| control@13 | 946.0 | 11.3% | 29.1 | 183.9 | 13.41 | 185.4 | 0 |
| control@101 | 916.4 | 11.0% | 24.1 | 177.6 | 8.02 | 195.9 | 108 |
| control@271 | 927.2 | 11.1% | 25.2 | 178.9 | 11.24 | 191.2 | 4 |
| control@314 | 969.5 | 11.6% | 23.4 | 176.5 | 11.93 | 202.1 | 39 |
| control@577 | 921.0 | 11.0% | 25.6 | 182.4 | 8.24 | 186.6 | 75 |
| control@863 | 961.0 | 11.5% | 25.1 | 179.1 | 6.39 | 191.3 | 93 |
| control@1024 | 941.4 | 11.3% | 24.5 | 178.1 | 9.45 | 195.2 | 117 |
| control@1729 | 947.4 | 11.4% | 30.0 | 182.4 | 9.24 | 213.1 | 81 |
| geometry@42 | 901.9 | 10.8% | 22.6 | 189.3 | 9.90 | 110.7 | 0 |
| geometry@1337 | 933.4 | 11.2% | 23.6 | 189.4 | 6.29 | 78.3 | 10 |
| geometry@2718 | 899.9 | 10.8% | 24.7 | 190.4 | 6.59 | 84.9 | 0 |
| geometry@7 | 907.8 | 10.9% | 24.5 | 191.3 | 4.38 | 55.7 | 45 |
| geometry@13 | 923.4 | 11.1% | 22.2 | 184.0 | 4.27 | 75.2 | 25 |
| geometry@101 | 907.1 | 10.9% | 22.8 | 184.1 | 3.99 | 51.1 | 52 |
| geometry@271 | 919.4 | 11.0% | 24.7 | 190.4 | 8.00 | 105.6 | 0 |
| geometry@314 | 934.3 | 11.2% | 22.1 | 186.3 | 6.46 | 95.6 | 0 |
| geometry@577 | 913.9 | 11.0% | 23.5 | 190.0 | 4.61 | 48.4 | 52 |
| geometry@863 | 945.0 | 11.3% | 24.2 | 185.9 | 0.10 | 76.5 | 86 |
| geometry@1024 | 906.7 | 10.9% | 22.6 | 187.2 | 3.70 | 58.8 | 59 |
| geometry@1729 | 893.9 | 10.7% | 24.0 | 189.1 | 7.12 | 73.9 | 0 |

Training-target contamination: 464 of 11,212 night rows read above 1 MW (max 439.3 MW); dropped from fit: True.
