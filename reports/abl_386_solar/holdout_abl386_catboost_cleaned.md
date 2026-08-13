# ABL-338 — solar non-negativity and solar geometry: held-out A/B

Generated 2026-08-13T12:51:35 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-04-30 .. 2026-06-12**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. Night hours are
reported in MW, never as a percentage: their denominator is ~0.

## AT — catboost, source `energy_renewable`

n_train 3,647 · n_holdout 1,056 (daylight 670 / shoulder 119 / night 267) · incumbent version 20260112_165237

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 781.9 | 37.2% | 1.0 | 0.3 | 0.00 | 0.0 | 0 |
| control_noholiday@42 | 720.1 | 34.3% | 55.9 | 56.9 | 51.03 | 99.8 | 10 |
| control_noholiday@1337 | 627.2 | 29.9% | 30.6 | 31.4 | 27.20 | 89.9 | 14 |
| control_noholiday@2718 | 691.1 | 32.9% | 36.2 | 35.3 | 34.31 | 98.7 | 41 |
| geometry_noholiday@42 | 601.7 | 28.7% | 41.4 | 42.0 | 31.02 | 76.9 | 11 |
| geometry_noholiday@1337 | 568.9 | 27.1% | 44.1 | 45.3 | 27.96 | 95.4 | 36 |
| geometry_noholiday@2718 | 620.0 | 29.5% | 38.3 | 39.5 | 26.27 | 76.3 | 28 |
| control@42 | 660.0 | 31.4% | 56.5 | 57.7 | 44.90 | 125.5 | 0 |
| control@1337 | 647.2 | 30.8% | 20.0 | 20.4 | 19.56 | 54.4 | 31 |
| control@2718 | 654.4 | 31.2% | 20.1 | 20.8 | 15.37 | 63.1 | 53 |
| geometry@42 | 630.1 | 30.0% | 36.5 | 37.7 | 31.30 | 83.5 | 1 |
| geometry@1337 | 637.5 | 30.4% | 34.1 | 35.3 | 27.87 | 86.3 | 16 |
| geometry@2718 | 589.1 | 28.1% | 25.4 | 25.8 | 19.10 | 67.3 | 60 |

Training-target contamination: 0 of 1,647 night rows read above 1 MW (max 0.0 MW); dropped from fit: True.

## BE — catboost, source `energy_renewable`

n_train 20,277 · n_holdout 1,056 (daylight 681 / shoulder 143 / night 232) · incumbent version 20260201_222022

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,425.5 | 49.2% | 1.5 | 0.7 | 0.00 | 0.0 | 0 |
| control_noholiday@42 | 563.4 | 19.4% | 23.0 | -6.1 | -15.38 | 47.6 | 225 |
| control_noholiday@1337 | 546.9 | 18.9% | 22.8 | 0.6 | -11.82 | 49.8 | 191 |
| control_noholiday@2718 | 564.4 | 19.5% | 28.2 | -3.2 | -14.24 | 124.3 | 220 |
| geometry_noholiday@42 | 549.7 | 19.0% | 21.7 | 6.5 | -11.36 | 66.2 | 241 |
| geometry_noholiday@1337 | 549.9 | 19.0% | 21.3 | -3.5 | -13.09 | 54.6 | 232 |
| geometry_noholiday@2718 | 556.4 | 19.2% | 18.0 | 6.0 | -11.81 | 42.2 | 225 |
| control@42 | 560.1 | 19.3% | 20.5 | 8.0 | -5.82 | 62.4 | 187 |
| control@1337 | 576.9 | 19.9% | 26.6 | 3.6 | -8.40 | 80.2 | 208 |
| control@2718 | 552.0 | 19.0% | 22.9 | -11.5 | -15.20 | 54.4 | 250 |
| geometry@42 | 573.1 | 19.8% | 24.4 | -13.2 | -12.42 | 37.8 | 257 |
| geometry@1337 | 561.6 | 19.4% | 21.9 | -3.3 | -11.33 | 35.5 | 235 |
| geometry@2718 | 581.9 | 20.1% | 18.8 | -1.1 | -11.06 | 80.1 | 251 |

Training-target contamination: 0 of 7,670 night rows read above 1 MW (max 0.1 MW); dropped from fit: True.

## DE — catboost, source `energy_renewable`

n_train 3,751 · n_holdout 1,056 (daylight 686 / shoulder 142 / night 228) · incumbent version 20260223_193822

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 6,509.3 | 29.5% | 14.8 | 13.9 | 0.00 | 0.4 | 0 |
| control_noholiday@42 | 4,144.5 | 18.8% | 345.0 | 372.2 | 326.73 | 561.5 | 0 |
| control_noholiday@1337 | 3,758.1 | 17.0% | 155.7 | 106.6 | -7.38 | 274.5 | 173 |
| control_noholiday@2718 | 4,278.8 | 19.4% | 332.7 | 359.9 | 194.31 | 548.3 | 1 |
| geometry_noholiday@42 | 3,999.7 | 18.1% | 368.2 | 395.4 | 278.58 | 510.9 | 0 |
| geometry_noholiday@1337 | 4,524.6 | 20.5% | 374.8 | 402.0 | 311.73 | 726.7 | 2 |
| geometry_noholiday@2718 | 4,467.3 | 20.2% | 433.4 | 460.6 | 272.12 | 663.3 | 3 |
| control@42 | 4,443.3 | 20.1% | 294.9 | 305.8 | 185.04 | 650.5 | 50 |
| control@1337 | 4,131.5 | 18.7% | 234.1 | 251.5 | 131.84 | 575.0 | 41 |
| control@2718 | 4,815.6 | 21.8% | 396.6 | 417.0 | 321.64 | 815.6 | 13 |
| geometry@42 | 4,286.4 | 19.4% | 470.1 | 497.2 | 357.46 | 809.9 | 5 |
| geometry@1337 | 4,224.5 | 19.1% | 222.7 | 246.7 | 159.43 | 758.9 | 30 |
| geometry@2718 | 4,838.2 | 21.9% | 258.7 | 275.8 | 175.32 | 675.4 | 44 |

Training-target contamination: 4 of 1,729 night rows read above 1 MW (max 1.7 MW); dropped from fit: True.

## FR — catboost, source `energy_renewable`

n_train 28,636 · n_holdout 1,056 (daylight 656 / shoulder 123 / night 277) · incumbent version 20260201_222014

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 2,128.1 | 25.6% | 29.9 | 146.2 | 7.28 | 251.0 | 0 |
| control_noholiday@42 | 1,023.7 | 12.3% | 87.4 | 208.9 | 50.18 | 198.4 | 48 |
| control_noholiday@1337 | 1,025.1 | 12.3% | 94.8 | 215.4 | 55.76 | 223.5 | 35 |
| control_noholiday@2718 | 1,002.5 | 12.0% | 95.9 | 198.0 | 49.47 | 150.5 | 15 |
| geometry_noholiday@42 | 989.1 | 11.9% | 104.7 | 223.5 | 37.95 | 143.2 | 42 |
| geometry_noholiday@1337 | 983.8 | 11.8% | 94.2 | 229.0 | 33.95 | 168.3 | 74 |
| geometry_noholiday@2718 | 977.6 | 11.7% | 94.7 | 220.9 | 37.25 | 158.1 | 44 |
| control@42 | 1,034.1 | 12.4% | 84.7 | 205.6 | 39.88 | 142.5 | 55 |
| control@1337 | 1,000.5 | 12.0% | 90.7 | 181.5 | 52.52 | 181.9 | 29 |
| control@2718 | 1,021.4 | 12.3% | 91.1 | 176.9 | 47.14 | 206.6 | 38 |
| geometry@42 | 1,003.4 | 12.1% | 83.8 | 196.6 | 19.32 | 120.2 | 85 |
| geometry@1337 | 996.6 | 12.0% | 108.0 | 220.2 | 33.92 | 157.7 | 45 |
| geometry@2718 | 1,004.8 | 12.1% | 104.9 | 195.6 | 33.35 | 122.9 | 56 |

Training-target contamination: 464 of 11,337 night rows read above 1 MW (max 439.3 MW); dropped from fit: True.
