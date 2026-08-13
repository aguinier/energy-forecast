# ABL-338 — solar non-negativity and solar geometry: held-out A/B

Generated 2026-08-13T13:59:18 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-06-13 .. 2026-07-12**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. Night hours are reported in MW, never as a percentage: their denominator is ~0.

## AT — xgboost, source `energy_renewable`

n_train 4,703 · n_holdout 720 (daylight 480 / shoulder 60 / night 180) · incumbent version 20260112_165237

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 502.3 | 22.6% | 0.1 | 0.4 | 0.00 | 0.0 | 0 |
| control@42 | 258.6 | 11.6% | 3.5 | -1.4 | -2.09 | 3.2 | 143 |
| control@1337 | 264.7 | 11.9% | 1.3 | 1.2 | -0.43 | 6.5 | 159 |
| control@2718 | 274.0 | 12.3% | 3.1 | 3.5 | 2.94 | 5.3 | 0 |
| control@7 | 271.8 | 12.2% | 0.9 | 0.6 | 0.10 | 1.4 | 96 |
| control@13 | 260.1 | 11.7% | 1.2 | 0.4 | -0.71 | 1.7 | 158 |
| control@101 | 271.2 | 12.2% | 3.9 | 4.3 | 3.26 | 3.5 | 0 |
| control@271 | 262.6 | 11.8% | 3.7 | 4.0 | 2.72 | 9.3 | 19 |
| control@314 | 276.2 | 12.4% | 4.2 | 4.4 | 1.92 | 5.6 | 54 |
| control@577 | 274.2 | 12.3% | 3.8 | 4.2 | 2.16 | 7.8 | 0 |
| control@863 | 268.4 | 12.1% | 4.2 | 4.6 | 2.53 | 5.8 | 1 |
| control@1024 | 276.3 | 12.4% | 3.6 | 4.0 | 3.28 | 6.1 | 0 |
| control@1729 | 279.2 | 12.6% | 8.8 | 9.2 | 6.30 | 37.9 | 4 |
| geometry@42 | 259.5 | 11.7% | 10.5 | 10.9 | 10.70 | 10.8 | 0 |
| geometry@1337 | 267.4 | 12.0% | 0.6 | 0.9 | 0.73 | 0.7 | 0 |
| geometry@2718 | 269.3 | 12.1% | 3.6 | 4.0 | 3.96 | 4.6 | 0 |
| geometry@7 | 265.6 | 12.0% | 2.3 | 2.7 | 2.56 | 3.8 | 0 |
| geometry@13 | 261.1 | 11.7% | 0.7 | 0.7 | 0.61 | 1.2 | 90 |
| geometry@101 | 259.6 | 11.7% | 1.6 | -0.3 | -0.55 | 1.1 | 96 |
| geometry@271 | 269.6 | 12.1% | 5.8 | 6.2 | 6.16 | 7.1 | 0 |
| geometry@314 | 264.3 | 11.9% | 5.1 | 5.5 | 5.47 | 6.5 | 0 |
| geometry@577 | 265.3 | 11.9% | 1.0 | 0.6 | 0.66 | 1.4 | 96 |
| geometry@863 | 264.3 | 11.9% | 2.5 | 2.9 | 2.31 | 2.9 | 0 |
| geometry@1024 | 262.9 | 11.8% | 3.8 | 3.4 | 1.69 | 8.5 | 72 |
| geometry@1729 | 266.6 | 12.0% | 0.7 | 1.1 | 1.08 | 1.1 | 0 |

Training-target contamination: 0 of 1,914 night rows read above 1 MW (max 0.0 MW); dropped from fit: True.

## BE — xgboost, source `energy_renewable`

n_train 21,333 · n_holdout 720 (daylight 480 / shoulder 116 / night 124) · incumbent version 20260201_222022

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 976.2 | 26.9% | 1.8 | 2.2 | 0.00 | 0.0 | 0 |
| control@42 | 466.8 | 12.9% | 9.1 | 10.6 | 7.65 | 7.8 | 0 |
| control@1337 | 485.3 | 13.4% | 17.1 | 18.9 | 16.20 | 16.2 | 0 |
| control@2718 | 448.9 | 12.4% | 5.1 | 6.3 | 4.74 | 5.0 | 0 |
| control@7 | 475.7 | 13.1% | 14.9 | 16.7 | 14.38 | 15.9 | 0 |
| control@13 | 471.5 | 13.0% | 20.7 | 22.6 | 20.05 | 20.0 | 0 |
| control@101 | 462.1 | 12.7% | 20.5 | 22.3 | 19.77 | 236.1 | 0 |
| control@271 | 457.5 | 12.6% | 9.7 | 11.3 | 8.78 | 9.3 | 0 |
| control@314 | 473.0 | 13.0% | 5.5 | 6.7 | 5.04 | 5.5 | 0 |
| control@577 | 481.4 | 13.3% | 24.6 | 26.5 | 24.45 | 24.4 | 0 |
| control@863 | 465.2 | 12.8% | 17.4 | 19.3 | 17.51 | 17.5 | 0 |
| control@1024 | 460.3 | 12.7% | 15.3 | 17.2 | 15.10 | 15.1 | 0 |
| control@1729 | 459.5 | 12.7% | 9.2 | 10.8 | 8.65 | 8.9 | 0 |
| geometry@42 | 456.3 | 12.6% | 8.8 | 10.6 | 8.54 | 8.7 | 0 |
| geometry@1337 | 494.4 | 13.6% | 18.5 | 20.4 | 18.20 | 18.2 | 0 |
| geometry@2718 | 473.6 | 13.1% | 22.2 | 24.1 | 21.18 | 21.2 | 0 |
| geometry@7 | 466.5 | 12.9% | 21.9 | 23.8 | 21.17 | 21.2 | 0 |
| geometry@13 | 451.6 | 12.4% | 9.2 | 11.0 | 8.67 | 8.7 | 0 |
| geometry@101 | 461.5 | 12.7% | 8.5 | 10.3 | 8.17 | 20.5 | 0 |
| geometry@271 | 464.5 | 12.8% | 11.0 | 12.9 | 10.68 | 10.7 | 0 |
| geometry@314 | 479.6 | 13.2% | 6.0 | 7.7 | 4.96 | 5.5 | 0 |
| geometry@577 | 467.2 | 12.9% | 13.3 | 15.2 | 12.85 | 12.9 | 0 |
| geometry@863 | 444.6 | 12.3% | 4.8 | 6.2 | 3.45 | 3.6 | 0 |
| geometry@1024 | 459.4 | 12.7% | 12.4 | 14.2 | 12.05 | 12.4 | 6 |
| geometry@1729 | 448.4 | 12.4% | 5.6 | 7.2 | 5.49 | 5.5 | 0 |

Training-target contamination: 0 of 7,902 night rows read above 1 MW (max 0.1 MW); dropped from fit: True.

## DE — xgboost, source `energy_renewable`

n_train 4,807 · n_holdout 720 (daylight 480 / shoulder 110 / night 130) · incumbent version 20260223_193822

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 5,806.6 | 23.4% | 20.0 | 57.7 | 0.00 | 0.0 | 0 |
| control@42 | 2,850.4 | 11.5% | 144.9 | 206.6 | 42.20 | 48.4 | 0 |
| control@1337 | 3,163.6 | 12.8% | 236.3 | 298.0 | 163.93 | 173.5 | 0 |
| control@2718 | 2,924.1 | 11.8% | 151.7 | 207.2 | 34.59 | 59.7 | 0 |
| control@7 | 2,917.7 | 11.8% | 74.3 | 127.4 | 7.89 | 27.5 | 35 |
| control@13 | 2,935.0 | 11.8% | 150.5 | 211.4 | 37.75 | 40.9 | 0 |
| control@101 | 2,855.4 | 11.5% | 93.8 | 146.3 | 4.29 | 9.5 | 6 |
| control@271 | 2,962.3 | 12.0% | 156.1 | 217.2 | 54.83 | 65.4 | 0 |
| control@314 | 2,916.6 | 11.8% | 89.6 | 146.4 | 17.89 | 26.7 | 24 |
| control@577 | 2,885.8 | 11.6% | 144.9 | 199.5 | 10.16 | 16.8 | 12 |
| control@863 | 2,936.5 | 11.8% | 136.1 | 194.5 | 40.83 | 56.6 | 0 |
| control@1024 | 2,787.6 | 11.2% | 103.4 | 147.1 | -8.24 | 11.3 | 89 |
| control@1729 | 2,876.8 | 11.6% | 184.3 | 227.3 | 2.58 | 32.8 | 89 |
| geometry@42 | 2,976.9 | 12.0% | 89.9 | 148.5 | 47.31 | 52.3 | 0 |
| geometry@1337 | 2,943.1 | 11.9% | 63.8 | 123.2 | 52.20 | 53.0 | 0 |
| geometry@2718 | 3,054.9 | 12.3% | 132.3 | 192.0 | 57.55 | 58.2 | 0 |
| geometry@7 | 2,938.9 | 11.9% | 51.9 | 109.4 | 48.94 | 49.0 | 0 |
| geometry@13 | 2,976.7 | 12.0% | 93.9 | 154.8 | 69.09 | 69.8 | 0 |
| geometry@101 | 3,027.4 | 12.2% | 43.8 | 96.9 | 28.41 | 71.2 | 1 |
| geometry@271 | 2,885.9 | 11.6% | 28.7 | 73.6 | 17.12 | 25.8 | 6 |
| geometry@314 | 2,957.9 | 11.9% | 21.2 | 66.0 | 6.48 | 12.0 | 31 |
| geometry@577 | 3,021.2 | 12.2% | 117.5 | 177.1 | 78.63 | 87.8 | 0 |
| geometry@863 | 2,902.7 | 11.7% | 51.6 | 91.1 | -4.82 | 37.7 | 120 |
| geometry@1024 | 2,986.7 | 12.1% | 48.7 | 105.5 | 47.13 | 47.1 | 0 |
| geometry@1729 | 2,983.5 | 12.0% | 31.8 | 78.4 | 22.65 | 23.4 | 0 |

Training-target contamination: 4 of 1,957 night rows read above 1 MW (max 1.7 MW); dropped from fit: True.

## FR — xgboost, source `energy_renewable`

n_train 29,668 · n_holdout 720 (daylight 480 / shoulder 60 / night 180) · incumbent version 20260201_222014

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,645.0 | 16.0% | 40.6 | 129.1 | 34.82 | 245.9 | 0 |
| control@42 | 1,254.1 | 12.2% | 23.2 | 142.8 | 29.66 | 205.6 | 8 |
| control@1337 | 1,251.3 | 12.2% | 24.2 | 146.0 | 29.29 | 201.5 | 20 |
| control@2718 | 1,210.1 | 11.8% | 29.3 | 160.3 | 52.36 | 230.1 | 0 |
| control@7 | 1,395.1 | 13.6% | 22.2 | 143.0 | 29.59 | 191.4 | 0 |
| control@13 | 1,282.2 | 12.5% | 29.4 | 167.5 | 51.91 | 243.9 | 0 |
| control@101 | 1,292.8 | 12.6% | 26.8 | 144.0 | 33.40 | 187.7 | 21 |
| control@271 | 1,282.1 | 12.5% | 24.3 | 146.4 | 28.83 | 220.5 | 17 |
| control@314 | 1,257.5 | 12.2% | 23.0 | 150.6 | 37.59 | 220.4 | 0 |
| control@577 | 1,243.9 | 12.1% | 23.7 | 142.6 | 27.23 | 201.0 | 70 |
| control@863 | 1,220.0 | 11.9% | 31.5 | 128.6 | 13.38 | 199.6 | 95 |
| control@1024 | 1,192.6 | 11.6% | 21.8 | 148.4 | 30.07 | 192.3 | 0 |
| control@1729 | 1,491.0 | 14.5% | 25.5 | 143.5 | 27.09 | 214.8 | 26 |
| geometry@42 | 1,274.7 | 12.4% | 28.8 | 163.7 | 12.43 | 99.5 | 6 |
| geometry@1337 | 1,315.5 | 12.8% | 35.5 | 176.9 | 25.00 | 82.7 | 0 |
| geometry@2718 | 1,336.2 | 13.0% | 26.3 | 156.7 | 10.77 | 91.1 | 0 |
| geometry@7 | 1,291.5 | 12.6% | 34.4 | 178.5 | 19.76 | 64.4 | 0 |
| geometry@13 | 1,245.7 | 12.1% | 27.8 | 152.9 | 5.99 | 47.7 | 22 |
| geometry@101 | 1,206.8 | 11.7% | 25.7 | 160.8 | 17.59 | 60.0 | 0 |
| geometry@271 | 1,207.1 | 11.7% | 26.2 | 158.8 | 14.90 | 93.8 | 0 |
| geometry@314 | 1,238.6 | 12.1% | 25.6 | 155.8 | 8.27 | 56.0 | 2 |
| geometry@577 | 1,329.2 | 12.9% | 27.1 | 164.0 | 14.11 | 50.5 | 0 |
| geometry@863 | 1,254.5 | 12.2% | 29.4 | 167.2 | 16.95 | 63.3 | 0 |
| geometry@1024 | 1,286.5 | 12.5% | 24.8 | 156.8 | 13.34 | 70.9 | 1 |
| geometry@1729 | 1,326.8 | 12.9% | 24.5 | 155.7 | 9.23 | 53.9 | 0 |

Training-target contamination: 488 of 11,614 night rows read above 1 MW (max 439.3 MW); dropped from fit: True.
