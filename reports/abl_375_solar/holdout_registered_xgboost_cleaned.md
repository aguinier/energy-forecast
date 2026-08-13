# ABL-338 — solar non-negativity and solar geometry: held-out A/B

Generated 2026-08-13T10:27:50 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-04-30 .. 2026-06-12**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. Night hours are
reported in MW, never as a percentage: their denominator is ~0.

## AT — xgboost, source `energy_renewable`

n_train 3,647 · n_holdout 1,056 (daylight 670 / shoulder 119 / night 267) · incumbent version 20260112_165237

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 781.9 | 37.2% | 1.0 | 0.3 | 0.00 | 0.0 | 0 |
| control@42 | 524.7 | 25.0% | 2.3 | 0.2 | -1.34 | 1.9 | 288 |
| control@1337 | 514.5 | 24.5% | 3.1 | 3.0 | -0.18 | 5.0 | 227 |
| control@2718 | 514.2 | 24.5% | 4.2 | 4.5 | 1.93 | 11.7 | 104 |
| geometry@42 | 494.0 | 23.5% | 3.9 | 4.2 | 3.42 | 15.1 | 36 |
| geometry@1337 | 499.9 | 23.8% | 3.2 | 3.2 | 1.68 | 8.7 | 141 |
| geometry@2718 | 494.7 | 23.6% | 7.1 | 8.0 | 7.06 | 23.6 | 34 |

Training-target contamination: 0 of 1,647 night rows read above 1 MW (max 0.0 MW); dropped from fit: True.

## BE — xgboost, source `energy_renewable`

n_train 20,277 · n_holdout 1,056 (daylight 681 / shoulder 143 / night 232) · incumbent version 20260201_222022

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,425.5 | 49.2% | 1.5 | 0.7 | 0.00 | 0.0 | 0 |
| control@42 | 635.5 | 21.9% | 8.4 | -0.6 | -5.74 | 6.3 | 175 |
| control@1337 | 643.1 | 22.2% | 8.2 | 2.3 | 0.02 | 38.2 | 190 |
| control@2718 | 627.2 | 21.6% | 8.0 | 9.6 | 7.17 | 7.2 | 0 |
| geometry@42 | 598.8 | 20.7% | 2.3 | 3.1 | 0.56 | 1.9 | 72 |
| geometry@1337 | 597.5 | 20.6% | 4.1 | 3.0 | 0.05 | 8.5 | 49 |
| geometry@2718 | 622.1 | 21.5% | 5.8 | 0.7 | -1.81 | 12.6 | 198 |

Training-target contamination: 0 of 7,670 night rows read above 1 MW (max 0.1 MW); dropped from fit: True.

## DE — xgboost, source `energy_renewable`

n_train 3,751 · n_holdout 1,056 (daylight 686 / shoulder 142 / night 228) · incumbent version 20260223_193822

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 6,509.3 | 29.5% | 14.8 | 13.9 | 0.00 | 0.4 | 0 |
| control@42 | 3,856.8 | 17.5% | 49.8 | -8.7 | -25.73 | 51.3 | 155 |
| control@1337 | 3,958.1 | 17.9% | 67.5 | -34.4 | -53.34 | 1.2 | 332 |
| control@2718 | 3,928.7 | 17.8% | 44.8 | -5.3 | -35.64 | 15.3 | 199 |
| geometry@42 | 4,362.6 | 19.8% | 81.5 | -42.9 | -63.19 | 40.1 | 189 |
| geometry@1337 | 4,161.8 | 18.9% | 40.1 | -0.3 | -24.57 | 24.5 | 267 |
| geometry@2718 | 4,224.2 | 19.1% | 56.7 | -12.8 | -45.87 | 14.8 | 153 |

Training-target contamination: 4 of 1,729 night rows read above 1 MW (max 1.7 MW); dropped from fit: True.

## FR — xgboost, source `energy_renewable`

n_train 28,636 · n_holdout 1,056 (daylight 656 / shoulder 123 / night 277) · incumbent version 20260201_222014

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 2,128.1 | 25.6% | 29.9 | 146.2 | 7.28 | 251.0 | 0 |
| control@42 | 979.8 | 11.8% | 33.0 | 184.5 | 36.81 | 192.2 | 0 |
| control@1337 | 982.4 | 11.8% | 47.6 | 202.3 | 45.62 | 223.1 | 0 |
| control@2718 | 980.0 | 11.8% | 33.9 | 182.2 | 29.82 | 208.0 | 0 |
| geometry@42 | 961.1 | 11.5% | 36.2 | 188.8 | 22.44 | 129.8 | 0 |
| geometry@1337 | 965.4 | 11.6% | 53.8 | 208.5 | 42.56 | 112.2 | 0 |
| geometry@2718 | 997.9 | 12.0% | 57.1 | 214.0 | 46.67 | 108.7 | 0 |

Training-target contamination: 464 of 11,337 night rows read above 1 MW (max 439.3 MW); dropped from fit: True.
