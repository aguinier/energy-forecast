# ABL-338 — solar non-negativity and solar geometry: held-out A/B

Generated 2026-08-13T14:30:49 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-07-13 .. 2026-08-11**, training from 2026-03-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. Night hours are reported in MW, never as a percentage: their denominator is ~0.

## AT — xgboost, source `energy_renewable`

n_train 2,880 · n_holdout 720 (daylight 448 / shoulder 92 / night 180) · incumbent version 20260112_165237

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 573.1 | 25.6% | 1.1 | 4.5 | 0.00 | 0.0 | 0 |
| control@42 | 298.9 | 13.3% | 2.3 | 3.5 | 0.12 | 5.8 | 128 |
| control@1337 | 287.7 | 12.8% | 2.0 | 5.7 | 1.31 | 1.4 | 0 |
| control@2718 | 288.8 | 12.9% | 2.9 | 5.4 | 1.34 | 9.3 | 63 |
| control@7 | 291.2 | 13.0% | 2.5 | 6.0 | 1.97 | 3.3 | 0 |
| control@13 | 296.9 | 13.3% | 2.4 | 5.0 | 0.87 | 6.0 | 75 |
| control@101 | 298.5 | 13.3% | 2.1 | 5.5 | 2.00 | 2.2 | 1 |
| control@271 | 305.3 | 13.6% | 2.9 | 6.5 | 2.65 | 4.0 | 0 |
| control@314 | 297.1 | 13.3% | 2.7 | 4.9 | 0.29 | 9.2 | 119 |
| control@577 | 282.8 | 12.6% | 2.2 | 4.4 | 0.18 | 4.8 | 93 |
| control@863 | 293.5 | 13.1% | 2.3 | 5.4 | 0.94 | 5.5 | 37 |
| control@1024 | 297.8 | 13.3% | 2.5 | 5.4 | 1.39 | 5.9 | 0 |
| control@1729 | 296.3 | 13.2% | 1.6 | 4.6 | 0.42 | 3.6 | 66 |
| geometry@42 | 283.6 | 12.7% | 2.7 | 6.0 | 0.86 | 6.0 | 44 |
| geometry@1337 | 289.9 | 12.9% | 1.9 | 3.2 | -0.77 | 2.1 | 159 |
| geometry@2718 | 284.1 | 12.7% | 1.9 | 3.8 | -0.07 | 12.0 | 113 |
| geometry@7 | 288.3 | 12.9% | 2.2 | 3.4 | -1.20 | 3.8 | 183 |
| geometry@13 | 288.0 | 12.9% | 2.1 | 5.2 | 1.55 | 1.8 | 8 |
| geometry@101 | 285.1 | 12.7% | 2.5 | 6.1 | 1.96 | 2.0 | 0 |
| geometry@271 | 288.8 | 12.9% | 1.7 | 4.5 | 0.51 | 1.6 | 26 |
| geometry@314 | 280.1 | 12.5% | 2.3 | 6.1 | 1.04 | 1.5 | 0 |
| geometry@577 | 284.1 | 12.7% | 2.5 | 5.2 | 1.88 | 2.2 | 0 |
| geometry@863 | 286.5 | 12.8% | 2.6 | 6.3 | 1.97 | 2.3 | 0 |
| geometry@1024 | 293.1 | 13.1% | 2.1 | 5.5 | 1.25 | 3.3 | 18 |
| geometry@1729 | 287.8 | 12.8% | 2.1 | 5.3 | 0.19 | 3.3 | 93 |

Training-target contamination: 0 of 836 night rows read above 1 MW (max 0.0 MW); dropped from fit: True.

## BE — xgboost, source `energy_renewable`

n_train 2,880 · n_holdout 720 (daylight 465 / shoulder 88 / night 167) · incumbent version 20260201_222022

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,090.1 | 32.9% | 3.3 | 5.1 | 0.00 | 0.0 | 0 |
| control@42 | 600.9 | 18.1% | 15.4 | 16.0 | 1.88 | 5.2 | 24 |
| control@1337 | 598.4 | 18.0% | 16.1 | 12.6 | -2.95 | 4.0 | 132 |
| control@2718 | 605.3 | 18.2% | 25.0 | 26.8 | 12.69 | 14.2 | 0 |
| control@7 | 581.9 | 17.5% | 17.5 | 19.3 | 4.73 | 9.4 | 1 |
| control@13 | 586.0 | 17.7% | 20.7 | 22.5 | 10.10 | 10.9 | 0 |
| control@101 | 580.3 | 17.5% | 26.2 | 28.0 | 13.93 | 16.7 | 0 |
| control@271 | 603.3 | 18.2% | 19.4 | 21.2 | 7.65 | 7.8 | 0 |
| control@314 | 596.3 | 18.0% | 28.8 | 30.6 | 17.49 | 30.7 | 0 |
| control@577 | 595.2 | 17.9% | 18.8 | 20.6 | 6.72 | 7.1 | 0 |
| control@863 | 589.1 | 17.8% | 20.1 | 21.9 | 7.11 | 8.3 | 0 |
| control@1024 | 601.7 | 18.1% | 23.3 | 25.1 | 11.57 | 12.0 | 0 |
| control@1729 | 599.6 | 18.1% | 12.9 | 13.7 | 0.78 | 3.0 | 35 |
| geometry@42 | 578.8 | 17.4% | 6.8 | 7.1 | 1.27 | 3.5 | 29 |
| geometry@1337 | 579.9 | 17.5% | 23.2 | 25.0 | 18.13 | 18.2 | 0 |
| geometry@2718 | 577.3 | 17.4% | 7.6 | 9.4 | 3.49 | 4.4 | 0 |
| geometry@7 | 573.1 | 17.3% | 4.9 | 6.3 | 1.24 | 6.8 | 43 |
| geometry@13 | 565.1 | 17.0% | 9.6 | 11.4 | 6.21 | 7.9 | 0 |
| geometry@101 | 575.1 | 17.3% | 5.0 | 6.6 | 0.80 | 2.5 | 50 |
| geometry@271 | 578.9 | 17.5% | 10.0 | 11.8 | 6.16 | 6.6 | 0 |
| geometry@314 | 570.9 | 17.2% | 6.5 | 8.2 | 2.68 | 5.8 | 7 |
| geometry@577 | 577.5 | 17.4% | 7.4 | 9.0 | 2.31 | 14.8 | 21 |
| geometry@863 | 566.3 | 17.1% | 13.5 | 15.3 | 7.52 | 7.7 | 0 |
| geometry@1024 | 565.8 | 17.1% | 16.9 | 18.6 | 15.03 | 15.0 | 0 |
| geometry@1729 | 574.9 | 17.3% | 10.8 | 12.6 | 6.51 | 6.8 | 0 |

Training-target contamination: 0 of 731 night rows read above 1 MW (max 0.0 MW); dropped from fit: True.

## DE — xgboost, source `energy_renewable`

n_train 2,878 · n_holdout 720 (daylight 457 / shoulder 99 / night 164) · incumbent version 20260223_193822

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 6,066.6 | 23.0% | 33.5 | 66.2 | 0.00 | 0.0 | 0 |
| control@42 | 3,035.2 | 11.5% | 77.3 | 12.4 | -51.82 | 56.1 | 152 |
| control@1337 | 3,072.5 | 11.7% | 44.9 | 36.3 | -18.53 | 73.3 | 111 |
| control@2718 | 3,030.6 | 11.5% | 51.6 | 80.4 | 7.35 | 64.2 | 61 |
| control@7 | 2,946.2 | 11.2% | 66.7 | 37.4 | -29.97 | 79.7 | 116 |
| control@13 | 3,010.9 | 11.4% | 55.6 | 32.4 | -25.78 | 48.7 | 114 |
| control@101 | 2,865.4 | 10.9% | 27.1 | 43.6 | -4.66 | 29.5 | 81 |
| control@271 | 3,029.3 | 11.5% | 55.8 | 23.6 | -33.73 | 26.3 | 150 |
| control@314 | 3,030.9 | 11.5% | 44.3 | 45.7 | -24.67 | 29.9 | 152 |
| control@577 | 2,966.3 | 11.3% | 65.4 | 44.3 | -29.05 | 64.2 | 132 |
| control@863 | 2,908.2 | 11.0% | 55.1 | 44.5 | -21.66 | 60.4 | 99 |
| control@1024 | 3,059.1 | 11.6% | 55.7 | 50.8 | -21.06 | 90.6 | 122 |
| control@1729 | 3,046.8 | 11.6% | 52.1 | 17.8 | -33.63 | 22.8 | 152 |
| geometry@42 | 3,005.9 | 11.4% | 40.1 | 13.1 | -27.05 | 23.1 | 163 |
| geometry@1337 | 2,982.1 | 11.3% | 28.3 | 35.4 | -4.65 | 50.2 | 114 |
| geometry@2718 | 2,978.6 | 11.3% | 54.0 | 4.5 | -33.17 | 26.1 | 130 |
| geometry@7 | 2,924.3 | 11.1% | 33.3 | 27.4 | -15.28 | 25.7 | 115 |
| geometry@13 | 2,908.3 | 11.0% | 31.9 | 26.2 | -18.76 | 19.6 | 86 |
| geometry@101 | 2,931.6 | 11.1% | 46.0 | 17.6 | -23.80 | 64.0 | 121 |
| geometry@271 | 2,957.2 | 11.2% | 57.6 | -6.0 | -39.10 | 20.0 | 146 |
| geometry@314 | 2,892.2 | 11.0% | 53.9 | 1.3 | -34.50 | 34.4 | 163 |
| geometry@577 | 2,881.2 | 10.9% | 46.4 | 13.0 | -23.16 | 29.8 | 143 |
| geometry@863 | 2,896.8 | 11.0% | 53.4 | 24.1 | -36.33 | 58.9 | 143 |
| geometry@1024 | 2,988.7 | 11.3% | 47.0 | 12.3 | -26.01 | 42.0 | 120 |
| geometry@1729 | 2,973.1 | 11.3% | 58.0 | 3.0 | -34.73 | 49.2 | 134 |

Training-target contamination: 2 of 731 night rows read above 1 MW (max 1.1 MW); dropped from fit: True.

## FR — xgboost, source `energy_renewable`

n_train 2,808 · n_holdout 720 (daylight 435 / shoulder 99 / night 186) · incumbent version 20260201_222014

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 2,368.6 | 22.3% | 50.8 | 270.4 | 34.88 | 321.5 | 0 |
| control@42 | 1,581.6 | 14.9% | 45.8 | 279.1 | 38.72 | 316.2 | 0 |
| control@1337 | 1,579.9 | 14.8% | 53.3 | 283.4 | 37.74 | 333.0 | 5 |
| control@2718 | 1,612.8 | 15.2% | 54.7 | 288.5 | 42.70 | 337.6 | 14 |
| control@7 | 1,474.3 | 13.9% | 47.8 | 277.7 | 34.46 | 303.9 | 45 |
| control@13 | 1,639.1 | 15.4% | 45.8 | 275.9 | 40.89 | 334.5 | 16 |
| control@101 | 1,577.4 | 14.8% | 45.2 | 279.9 | 41.43 | 317.7 | 2 |
| control@271 | 1,575.6 | 14.8% | 49.0 | 280.3 | 29.56 | 330.3 | 67 |
| control@314 | 1,569.1 | 14.7% | 48.2 | 284.9 | 39.46 | 321.2 | 0 |
| control@577 | 1,648.8 | 15.5% | 55.5 | 292.5 | 37.65 | 306.2 | 24 |
| control@863 | 1,544.6 | 14.5% | 48.0 | 280.7 | 37.04 | 319.5 | 6 |
| control@1024 | 1,516.5 | 14.3% | 44.4 | 278.5 | 36.47 | 320.6 | 30 |
| control@1729 | 1,605.2 | 15.1% | 48.8 | 282.9 | 39.49 | 305.5 | 0 |
| geometry@42 | 1,589.8 | 14.9% | 42.6 | 274.2 | 38.97 | 299.7 | 16 |
| geometry@1337 | 1,507.9 | 14.2% | 55.3 | 291.0 | 40.51 | 296.7 | 8 |
| geometry@2718 | 1,555.0 | 14.6% | 49.4 | 282.7 | 43.29 | 310.1 | 0 |
| geometry@7 | 1,450.7 | 13.6% | 50.6 | 285.2 | 42.55 | 301.8 | 18 |
| geometry@13 | 1,611.1 | 15.1% | 44.6 | 275.1 | 35.60 | 287.2 | 0 |
| geometry@101 | 1,554.9 | 14.6% | 39.4 | 272.2 | 34.17 | 275.7 | 0 |
| geometry@271 | 1,565.3 | 14.7% | 53.7 | 287.6 | 39.91 | 291.2 | 0 |
| geometry@314 | 1,597.1 | 15.0% | 60.4 | 294.3 | 45.57 | 295.8 | 20 |
| geometry@577 | 1,518.1 | 14.3% | 41.7 | 273.4 | 37.57 | 287.4 | 6 |
| geometry@863 | 1,571.1 | 14.8% | 50.2 | 281.4 | 36.93 | 267.9 | 0 |
| geometry@1024 | 1,547.5 | 14.5% | 42.3 | 276.9 | 39.76 | 295.6 | 0 |
| geometry@1729 | 1,537.6 | 14.5% | 45.2 | 276.8 | 37.44 | 292.6 | 11 |

Training-target contamination: 72 of 852 night rows read above 1 MW (max 285.6 MW); dropped from fit: True.
