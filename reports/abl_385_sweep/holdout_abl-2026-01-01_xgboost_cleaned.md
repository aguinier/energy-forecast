# ABL-338 — solar non-negativity and solar geometry: held-out A/B

Generated 2026-08-13T14:28:24 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-07-13 .. 2026-08-11**, training from 2026-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. Night hours are reported in MW, never as a percentage: their denominator is ~0.

## AT — xgboost, source `energy_renewable`

n_train 4,268 · n_holdout 720 (daylight 448 / shoulder 92 / night 180) · incumbent version 20260112_165237

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 573.1 | 25.6% | 1.1 | 4.5 | 0.00 | 0.0 | 0 |
| control@42 | 305.5 | 13.6% | 2.1 | 5.2 | 0.72 | 2.1 | 4 |
| control@1337 | 295.0 | 13.2% | 2.1 | 3.9 | -0.20 | 6.2 | 116 |
| control@2718 | 286.9 | 12.8% | 3.9 | 6.3 | 0.77 | 9.2 | 114 |
| control@7 | 290.0 | 12.9% | 2.9 | 5.7 | 0.87 | 9.9 | 77 |
| control@13 | 298.7 | 13.3% | 3.5 | 6.4 | 0.75 | 5.2 | 44 |
| control@101 | 296.9 | 13.3% | 3.9 | 6.6 | 1.28 | 12.4 | 94 |
| control@271 | 296.9 | 13.3% | 2.7 | 5.3 | 0.66 | 9.1 | 106 |
| control@314 | 299.2 | 13.4% | 2.6 | 4.6 | 0.39 | 6.4 | 93 |
| control@577 | 297.3 | 13.3% | 2.7 | 5.6 | 0.13 | 1.4 | 69 |
| control@863 | 286.2 | 12.8% | 2.5 | 4.8 | 0.47 | 6.9 | 27 |
| control@1024 | 293.4 | 13.1% | 3.7 | 5.0 | -0.35 | 5.5 | 119 |
| control@1729 | 299.9 | 13.4% | 2.6 | 5.9 | 1.40 | 1.8 | 0 |
| geometry@42 | 291.0 | 13.0% | 2.2 | 2.9 | 0.43 | 1.2 | 12 |
| geometry@1337 | 284.7 | 12.7% | 1.9 | 3.8 | 1.01 | 1.1 | 0 |
| geometry@2718 | 282.2 | 12.6% | 4.1 | 7.6 | 4.24 | 4.3 | 0 |
| geometry@7 | 287.0 | 12.8% | 3.1 | 1.7 | -1.19 | 1.7 | 182 |
| geometry@13 | 290.2 | 13.0% | 1.7 | 3.0 | 0.07 | 4.9 | 96 |
| geometry@101 | 278.9 | 12.4% | 2.1 | 4.0 | 0.23 | 5.9 | 112 |
| geometry@271 | 286.4 | 12.8% | 2.2 | 1.9 | 0.05 | 1.7 | 144 |
| geometry@314 | 286.0 | 12.8% | 2.0 | 3.8 | 0.26 | 2.4 | 52 |
| geometry@577 | 284.3 | 12.7% | 2.6 | 2.6 | -1.11 | 3.1 | 171 |
| geometry@863 | 282.9 | 12.6% | 2.3 | 3.3 | 0.97 | 7.0 | 58 |
| geometry@1024 | 287.1 | 12.8% | 2.1 | 2.9 | -0.79 | 2.6 | 153 |
| geometry@1729 | 296.1 | 13.2% | 6.2 | 10.0 | 6.37 | 6.4 | 0 |

Training-target contamination: 0 of 1,487 night rows read above 1 MW (max 0.0 MW); dropped from fit: True.

## BE — xgboost, source `energy_renewable`

n_train 4,268 · n_holdout 720 (daylight 465 / shoulder 88 / night 167) · incumbent version 20260201_222022

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,090.1 | 32.9% | 3.3 | 5.1 | 0.00 | 0.0 | 0 |
| control@42 | 606.1 | 18.3% | 13.6 | 15.4 | 11.09 | 11.4 | 0 |
| control@1337 | 577.7 | 17.4% | 6.5 | 5.7 | -0.80 | 6.0 | 87 |
| control@2718 | 586.2 | 17.7% | 10.4 | 12.2 | 5.99 | 6.3 | 0 |
| control@7 | 579.1 | 17.5% | 13.2 | 15.0 | 9.85 | 10.6 | 0 |
| control@13 | 588.6 | 17.7% | 12.6 | 14.4 | 10.00 | 12.2 | 0 |
| control@101 | 578.5 | 17.4% | 6.4 | 7.9 | 4.68 | 5.3 | 0 |
| control@271 | 590.5 | 17.8% | 15.0 | 16.8 | 10.65 | 10.6 | 0 |
| control@314 | 585.8 | 17.7% | 10.0 | 11.8 | 5.45 | 9.7 | 0 |
| control@577 | 580.5 | 17.5% | 12.3 | 14.1 | 7.96 | 8.8 | 0 |
| control@863 | 568.6 | 17.1% | 9.6 | 11.4 | 4.18 | 4.8 | 0 |
| control@1024 | 589.3 | 17.8% | 11.9 | 13.7 | 7.57 | 8.2 | 0 |
| control@1729 | 598.6 | 18.0% | 10.5 | 12.3 | 5.91 | 6.5 | 0 |
| geometry@42 | 568.0 | 17.1% | 6.8 | 8.3 | 5.64 | 6.4 | 0 |
| geometry@1337 | 558.4 | 16.8% | 16.7 | 18.5 | 15.79 | 15.8 | 0 |
| geometry@2718 | 575.0 | 17.3% | 18.7 | 20.5 | 16.65 | 16.7 | 0 |
| geometry@7 | 567.2 | 17.1% | 14.3 | 16.0 | 12.31 | 13.2 | 0 |
| geometry@13 | 581.3 | 17.5% | 7.2 | 8.7 | 5.78 | 5.8 | 0 |
| geometry@101 | 562.6 | 17.0% | 9.8 | 11.3 | 7.91 | 7.9 | 0 |
| geometry@271 | 573.5 | 17.3% | 5.4 | 6.5 | 4.12 | 8.8 | 3 |
| geometry@314 | 567.6 | 17.1% | 5.8 | 5.4 | 2.40 | 4.9 | 29 |
| geometry@577 | 574.2 | 17.3% | 9.5 | 11.2 | 7.57 | 9.2 | 0 |
| geometry@863 | 565.4 | 17.0% | 17.9 | 19.7 | 15.74 | 15.7 | 0 |
| geometry@1024 | 570.1 | 17.2% | 3.3 | 2.9 | -0.07 | 2.4 | 63 |
| geometry@1729 | 582.4 | 17.6% | 18.4 | 20.2 | 16.68 | 16.7 | 0 |

Training-target contamination: 0 of 1,387 night rows read above 1 MW (max 0.0 MW); dropped from fit: True.

## DE — xgboost, source `energy_renewable`

n_train 4,265 · n_holdout 720 (daylight 457 / shoulder 99 / night 164) · incumbent version 20260223_193822

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 6,066.6 | 23.0% | 33.5 | 66.2 | 0.00 | 0.0 | 0 |
| control@42 | 3,126.2 | 11.9% | 44.3 | 28.4 | -22.74 | 80.4 | 165 |
| control@1337 | 3,130.4 | 11.9% | 38.6 | 68.0 | 4.45 | 40.8 | 75 |
| control@2718 | 3,066.6 | 11.6% | 52.5 | 69.3 | -5.21 | 119.8 | 110 |
| control@7 | 3,094.7 | 11.7% | 59.9 | 71.0 | 0.93 | 96.7 | 96 |
| control@13 | 2,994.4 | 11.4% | 50.0 | 29.7 | -21.08 | 108.7 | 158 |
| control@101 | 3,088.0 | 11.7% | 50.0 | 70.9 | 8.65 | 117.0 | 84 |
| control@271 | 3,023.8 | 11.5% | 67.0 | 88.7 | -8.02 | 53.0 | 147 |
| control@314 | 3,061.4 | 11.6% | 54.8 | 43.6 | -22.09 | 67.9 | 142 |
| control@577 | 3,069.9 | 11.6% | 43.8 | 52.5 | -12.32 | 94.1 | 118 |
| control@863 | 3,010.8 | 11.4% | 60.8 | 27.8 | -19.46 | 97.1 | 119 |
| control@1024 | 3,033.5 | 11.5% | 68.0 | 36.2 | -22.30 | 129.1 | 119 |
| control@1729 | 3,056.0 | 11.6% | 60.8 | 68.5 | -7.46 | 89.3 | 137 |
| geometry@42 | 2,967.0 | 11.3% | 16.3 | 23.9 | -3.80 | 10.8 | 128 |
| geometry@1337 | 3,151.6 | 12.0% | 41.7 | 34.7 | -16.65 | 78.8 | 140 |
| geometry@2718 | 3,046.9 | 11.6% | 46.0 | 52.7 | -4.60 | 63.7 | 113 |
| geometry@7 | 3,019.9 | 11.5% | 41.3 | 20.3 | -15.57 | 87.1 | 129 |
| geometry@13 | 3,028.9 | 11.5% | 33.7 | 10.8 | -25.75 | 18.4 | 194 |
| geometry@101 | 3,054.0 | 11.6% | 38.3 | 22.5 | -0.13 | 60.4 | 123 |
| geometry@271 | 3,047.8 | 11.6% | 33.6 | 50.3 | -6.52 | 34.5 | 137 |
| geometry@314 | 3,068.7 | 11.6% | 37.4 | 18.5 | -15.82 | 33.6 | 118 |
| geometry@577 | 2,975.7 | 11.3% | 24.9 | 17.1 | -10.63 | 23.0 | 185 |
| geometry@863 | 3,043.9 | 11.5% | 36.8 | 28.0 | -8.33 | 38.4 | 79 |
| geometry@1024 | 3,008.1 | 11.4% | 38.9 | 15.6 | -11.10 | 57.6 | 124 |
| geometry@1729 | 3,025.7 | 11.5% | 49.4 | 12.0 | -29.75 | 41.3 | 147 |

Training-target contamination: 4 of 1,399 night rows read above 1 MW (max 1.7 MW); dropped from fit: True.

## FR — xgboost, source `energy_renewable`

n_train 4,129 · n_holdout 720 (daylight 435 / shoulder 99 / night 186) · incumbent version 20260201_222014

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 2,368.6 | 22.3% | 50.8 | 270.4 | 34.88 | 321.5 | 0 |
| control@42 | 1,623.3 | 15.3% | 26.4 | 253.9 | 32.15 | 310.4 | 40 |
| control@1337 | 1,633.6 | 15.4% | 34.4 | 261.8 | 31.08 | 256.9 | 33 |
| control@2718 | 1,732.4 | 16.3% | 36.0 | 268.2 | 34.18 | 280.0 | 48 |
| control@7 | 1,630.0 | 15.3% | 31.8 | 259.7 | 32.78 | 285.6 | 29 |
| control@13 | 1,582.9 | 14.9% | 25.7 | 250.4 | 29.11 | 292.5 | 32 |
| control@101 | 1,536.2 | 14.4% | 29.6 | 258.8 | 32.03 | 281.0 | 3 |
| control@271 | 1,664.3 | 15.6% | 28.6 | 256.3 | 32.68 | 294.2 | 0 |
| control@314 | 1,585.6 | 14.9% | 33.9 | 266.4 | 33.42 | 288.2 | 7 |
| control@577 | 1,624.1 | 15.3% | 34.0 | 262.0 | 33.67 | 279.5 | 15 |
| control@863 | 1,658.3 | 15.6% | 30.0 | 262.8 | 35.36 | 287.8 | 11 |
| control@1024 | 1,769.8 | 16.6% | 25.4 | 245.3 | 33.93 | 286.5 | 0 |
| control@1729 | 1,669.6 | 15.7% | 28.9 | 261.8 | 30.34 | 285.3 | 77 |
| geometry@42 | 1,616.1 | 15.2% | 27.7 | 235.0 | 24.04 | 208.9 | 58 |
| geometry@1337 | 1,598.5 | 15.0% | 30.5 | 236.6 | 25.12 | 199.7 | 10 |
| geometry@2718 | 1,652.5 | 15.5% | 30.2 | 242.0 | 26.24 | 224.1 | 18 |
| geometry@7 | 1,499.2 | 14.1% | 29.7 | 240.8 | 27.06 | 167.4 | 0 |
| geometry@13 | 1,469.8 | 13.8% | 27.9 | 239.5 | 22.26 | 191.1 | 47 |
| geometry@101 | 1,580.1 | 14.9% | 28.2 | 240.2 | 27.08 | 195.3 | 4 |
| geometry@271 | 1,578.2 | 14.8% | 34.5 | 240.1 | 22.89 | 189.5 | 47 |
| geometry@314 | 1,550.8 | 14.6% | 31.5 | 237.5 | 23.00 | 203.0 | 48 |
| geometry@577 | 1,547.7 | 14.5% | 29.9 | 233.0 | 22.22 | 195.0 | 18 |
| geometry@863 | 1,575.6 | 14.8% | 32.9 | 247.4 | 29.43 | 199.1 | 0 |
| geometry@1024 | 1,582.3 | 14.9% | 28.3 | 239.8 | 31.58 | 184.5 | 0 |
| geometry@1729 | 1,664.1 | 15.6% | 29.7 | 238.0 | 24.22 | 211.2 | 50 |

Training-target contamination: 114 of 1,487 night rows read above 1 MW (max 285.9 MW); dropped from fit: True.
