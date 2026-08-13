# Held-out A/B — price (ABL-393 reuse of the ABL-338 harness)

Generated 2026-08-13T21:55:18 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-04-30 .. 2026-06-12**, training from 2021-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `price` has no band structure, so one all-hours row is the result.

## AT / price — xgboost, one fixed table for `price`

n_train 35,525 · n_holdout 1,056 · incumbent version 20260202_144224

| arm | n | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,056 | 32.4 | 30.8% | 59.3 | -5.1 | 95.0 | 107 |
| _constant_causal_ | 1,056 | 46.6 | 44.4% | 73.6 | 35.5 | 135.5 | 0 |
| _constant_oracle_ | 1,056 | 43.5 | 41.3% | 67.3 | 19.2 | 119.3 | 0 |
| _climatology_causal_ | 1,056 | 43.8 | 41.6% | 67.3 | 35.4 | 135.5 | 0 |
| _climatology_oracle_ | 1,056 | 26.1 | 24.8% | 44.2 | -1.2 | 98.8 | 0 |
| control@101 | 1,056 | 20.7 | 19.7% | 31.8 | -8.2 | 91.8 | 88 |
| control@103 | 1,056 | 20.3 | 19.3% | 31.6 | -7.8 | 92.2 | 85 |
| control@107 | 1,056 | 20.9 | 19.9% | 31.9 | -8.2 | 91.9 | 78 |
| control@109 | 1,056 | 20.0 | 19.1% | 31.0 | -6.8 | 93.3 | 71 |
| control@113 | 1,056 | 20.3 | 19.3% | 30.8 | -7.0 | 93.0 | 76 |
| control@127 | 1,056 | 20.7 | 19.7% | 31.4 | -7.5 | 92.5 | 71 |
| control@131 | 1,056 | 20.9 | 19.9% | 31.5 | -9.4 | 90.7 | 77 |
| control@137 | 1,056 | 21.6 | 20.5% | 32.5 | -9.6 | 90.4 | 74 |
| control_noholiday@101 | 1,056 | 20.6 | 19.6% | 32.2 | -7.4 | 92.7 | 71 |
| control_noholiday@103 | 1,056 | 20.5 | 19.5% | 32.1 | -7.0 | 93.0 | 57 |
| control_noholiday@107 | 1,056 | 21.7 | 20.7% | 33.1 | -9.0 | 91.0 | 73 |
| control_noholiday@109 | 1,056 | 21.5 | 20.4% | 33.1 | -9.9 | 90.2 | 77 |
| control_noholiday@113 | 1,056 | 21.4 | 20.3% | 33.0 | -9.2 | 90.8 | 77 |
| control_noholiday@127 | 1,056 | 21.2 | 20.2% | 32.9 | -8.0 | 92.0 | 78 |
| control_noholiday@131 | 1,056 | 21.4 | 20.3% | 32.4 | -8.2 | 91.8 | 65 |
| control_noholiday@137 | 1,056 | 20.5 | 19.5% | 32.1 | -6.0 | 94.0 | 66 |

Holiday subsets of the holdout (holiday 96 · holiday_affected 288 · ordinary 768). `holiday_affected` is a holiday, a bridge day, or within one day of a holiday — the rows these four features can distinguish from an ordinary day at all. A holiday effect that is real here is diluted by the row counts in the all-hours table above.

| arm | holiday MAE | holiday_affected MAE | ordinary MAE |
|---|---:|---:|---:|
| _seasonal-naive D-7_ | 56.1 | 35.0 | 31.3 |
| _constant_causal_ | 76.3 | 57.4 | 42.6 |
| _constant_oracle_ | 70.3 | 52.9 | 39.9 |
| _climatology_causal_ | 75.1 | 54.7 | 39.7 |
| _climatology_oracle_ | 40.8 | 30.6 | 24.4 |
| control@101 | 32.9 | 24.3 | 19.4 |
| control@103 | 32.9 | 24.3 | 18.8 |
| control@107 | 31.6 | 24.8 | 19.5 |
| control@109 | 31.5 | 23.2 | 18.9 |
| control@113 | 31.9 | 22.8 | 19.4 |
| control@127 | 30.0 | 23.5 | 19.6 |
| control@131 | 31.2 | 23.7 | 19.8 |
| control@137 | 33.3 | 25.6 | 20.1 |
| control_noholiday@101 | 34.7 | 25.3 | 18.8 |
| control_noholiday@103 | 34.1 | 24.5 | 19.0 |
| control_noholiday@107 | 37.8 | 27.3 | 19.6 |
| control_noholiday@109 | 34.4 | 25.3 | 20.1 |
| control_noholiday@113 | 37.0 | 26.4 | 19.5 |
| control_noholiday@127 | 37.4 | 26.0 | 19.5 |
| control_noholiday@131 | 36.1 | 25.7 | 19.8 |
| control_noholiday@137 | 33.8 | 24.7 | 18.9 |

ABL-337 night screen: not applicable to price.

## BE / price — xgboost, one fixed table for `price`

n_train 37,883 · n_holdout 1,056 · incumbent version 20260404_185535

| arm | n | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,056 | 35.9 | 37.1% | 61.8 | -5.7 | 86.3 | 109 |
| _constant_causal_ | 1,056 | 42.4 | 43.9% | 66.7 | 28.5 | 120.6 | 0 |
| _constant_oracle_ | 1,056 | 40.4 | 41.7% | 62.4 | 16.3 | 108.3 | 0 |
| _climatology_causal_ | 1,056 | 37.5 | 38.8% | 58.7 | 28.5 | 120.6 | 0 |
| _climatology_oracle_ | 1,056 | 25.9 | 26.8% | 42.8 | 0.3 | 92.4 | 0 |
| control@101 | 1,056 | 20.8 | 21.5% | 31.7 | -9.0 | 83.1 | 81 |
| control@103 | 1,056 | 20.5 | 21.2% | 31.9 | -8.0 | 84.0 | 72 |
| control@107 | 1,056 | 21.5 | 22.2% | 32.7 | -9.8 | 82.3 | 76 |
| control@109 | 1,056 | 20.7 | 21.4% | 32.1 | -8.9 | 83.1 | 79 |
| control@113 | 1,056 | 21.0 | 21.7% | 32.4 | -9.0 | 83.0 | 77 |
| control@127 | 1,056 | 21.3 | 22.0% | 32.7 | -8.5 | 83.5 | 75 |
| control@131 | 1,056 | 21.3 | 22.0% | 32.5 | -9.1 | 82.9 | 78 |
| control@137 | 1,056 | 21.6 | 22.3% | 33.2 | -10.5 | 81.5 | 79 |
| control_noholiday@101 | 1,056 | 20.9 | 21.6% | 32.8 | -7.4 | 84.6 | 71 |
| control_noholiday@103 | 1,056 | 22.0 | 22.8% | 34.1 | -10.3 | 81.8 | 74 |
| control_noholiday@107 | 1,056 | 20.6 | 21.3% | 32.6 | -9.0 | 83.0 | 79 |
| control_noholiday@109 | 1,056 | 21.5 | 22.2% | 33.4 | -9.6 | 82.4 | 77 |
| control_noholiday@113 | 1,056 | 21.5 | 22.3% | 33.6 | -9.4 | 82.6 | 80 |
| control_noholiday@127 | 1,056 | 21.2 | 21.9% | 32.8 | -8.5 | 83.5 | 77 |
| control_noholiday@131 | 1,056 | 21.7 | 22.4% | 33.5 | -9.6 | 82.5 | 73 |
| control_noholiday@137 | 1,056 | 21.3 | 22.1% | 33.4 | -9.4 | 82.6 | 74 |

Holiday subsets of the holdout (holiday 96 · holiday_affected 240 · ordinary 816). `holiday_affected` is a holiday, a bridge day, or within one day of a holiday — the rows these four features can distinguish from an ordinary day at all. A holiday effect that is real here is diluted by the row counts in the all-hours table above.

| arm | holiday MAE | holiday_affected MAE | ordinary MAE |
|---|---:|---:|---:|
| _seasonal-naive D-7_ | 52.5 | 35.4 | 36.0 |
| _constant_causal_ | 74.3 | 60.3 | 37.2 |
| _constant_oracle_ | 70.7 | 56.7 | 35.6 |
| _climatology_causal_ | 68.4 | 53.7 | 32.8 |
| _climatology_oracle_ | 40.3 | 29.7 | 24.8 |
| control@101 | 37.0 | 25.4 | 19.5 |
| control@103 | 35.9 | 25.3 | 19.1 |
| control@107 | 38.9 | 28.1 | 19.5 |
| control@109 | 38.5 | 26.2 | 19.1 |
| control@113 | 38.8 | 27.5 | 19.1 |
| control@127 | 39.0 | 27.9 | 19.3 |
| control@131 | 37.5 | 26.6 | 19.8 |
| control@137 | 38.6 | 27.8 | 19.8 |
| control_noholiday@101 | 35.4 | 26.6 | 19.2 |
| control_noholiday@103 | 37.8 | 27.7 | 20.3 |
| control_noholiday@107 | 34.9 | 25.8 | 19.1 |
| control_noholiday@109 | 37.0 | 26.9 | 19.9 |
| control_noholiday@113 | 36.0 | 26.9 | 19.9 |
| control_noholiday@127 | 34.6 | 26.6 | 19.6 |
| control_noholiday@131 | 35.1 | 26.7 | 20.2 |
| control_noholiday@137 | 38.0 | 27.6 | 19.5 |

ABL-337 night screen: not applicable to price.

## DE / price — xgboost, one fixed table for `price`

n_train 35,270 · n_holdout 1,056 · incumbent version 20260404_185523

| arm | n | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,056 | 35.2 | 34.1% | 62.5 | -6.5 | 91.6 | 134 |
| _constant_causal_ | 1,056 | 44.3 | 42.9% | 69.9 | 26.1 | 124.2 | 0 |
| _constant_oracle_ | 1,056 | 43.2 | 41.9% | 66.8 | 16.2 | 114.3 | 0 |
| _climatology_causal_ | 1,056 | 38.5 | 37.2% | 60.2 | 26.1 | 124.2 | 0 |
| _climatology_oracle_ | 1,056 | 26.7 | 25.8% | 44.4 | 1.5 | 99.6 | 0 |
| control@101 | 1,056 | 27.5 | 26.6% | 39.1 | -20.3 | 77.8 | 105 |
| control@103 | 1,056 | 26.7 | 25.8% | 37.5 | -19.1 | 79.0 | 99 |
| control@107 | 1,056 | 26.1 | 25.3% | 36.6 | -18.6 | 79.5 | 99 |
| control@109 | 1,056 | 26.7 | 25.9% | 37.2 | -19.5 | 78.6 | 102 |
| control@113 | 1,056 | 26.7 | 25.9% | 38.1 | -18.5 | 79.6 | 100 |
| control@127 | 1,056 | 26.6 | 25.8% | 37.4 | -18.2 | 79.9 | 90 |
| control@131 | 1,056 | 27.7 | 26.9% | 39.0 | -20.0 | 78.1 | 101 |
| control@137 | 1,056 | 26.1 | 25.3% | 37.8 | -18.1 | 80.0 | 104 |
| control_noholiday@101 | 1,056 | 25.4 | 24.6% | 36.5 | -16.6 | 81.5 | 95 |
| control_noholiday@103 | 1,056 | 25.2 | 24.5% | 37.2 | -15.9 | 82.2 | 91 |
| control_noholiday@107 | 1,056 | 27.0 | 26.1% | 38.6 | -18.8 | 79.3 | 105 |
| control_noholiday@109 | 1,056 | 25.7 | 24.9% | 37.6 | -17.2 | 80.9 | 97 |
| control_noholiday@113 | 1,056 | 25.4 | 24.6% | 37.4 | -16.5 | 81.6 | 92 |
| control_noholiday@127 | 1,056 | 26.4 | 25.5% | 38.5 | -17.9 | 80.2 | 92 |
| control_noholiday@131 | 1,056 | 26.0 | 25.2% | 37.2 | -17.2 | 80.9 | 89 |
| control_noholiday@137 | 1,056 | 26.1 | 25.3% | 37.7 | -17.8 | 80.3 | 96 |

Holiday subsets of the holdout (holiday 72 · holiday_affected 216 · ordinary 840). `holiday_affected` is a holiday, a bridge day, or within one day of a holiday — the rows these four features can distinguish from an ordinary day at all. A holiday effect that is real here is diluted by the row counts in the all-hours table above.

| arm | holiday MAE | holiday_affected MAE | ordinary MAE |
|---|---:|---:|---:|
| _seasonal-naive D-7_ | 62.2 | 36.1 | 34.9 |
| _constant_causal_ | 77.8 | 60.4 | 40.1 |
| _constant_oracle_ | 74.7 | 58.4 | 39.3 |
| _climatology_causal_ | 72.5 | 54.0 | 34.5 |
| _climatology_oracle_ | 45.0 | 32.4 | 25.2 |
| control@101 | 46.9 | 37.2 | 25.0 |
| control@103 | 50.0 | 37.5 | 23.9 |
| control@107 | 45.8 | 34.8 | 23.8 |
| control@109 | 45.4 | 35.8 | 24.4 |
| control@113 | 46.1 | 36.7 | 24.1 |
| control@127 | 46.9 | 36.7 | 24.0 |
| control@131 | 49.1 | 38.6 | 24.9 |
| control@137 | 42.3 | 34.0 | 24.1 |
| control_noholiday@101 | 41.5 | 32.9 | 23.5 |
| control_noholiday@103 | 41.3 | 33.4 | 23.1 |
| control_noholiday@107 | 38.3 | 33.9 | 25.2 |
| control_noholiday@109 | 40.8 | 32.3 | 24.0 |
| control_noholiday@113 | 40.4 | 32.6 | 23.6 |
| control_noholiday@127 | 41.0 | 34.6 | 24.2 |
| control_noholiday@131 | 40.9 | 34.2 | 23.9 |
| control_noholiday@137 | 41.0 | 33.6 | 24.2 |

ABL-337 night screen: not applicable to price.

## FR / price — xgboost, one fixed table for `price`

n_train 37,331 · n_holdout 1,056 · incumbent version 20260404_185529

| arm | n | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,056 | 41.1 | 84.6% | 64.5 | -2.3 | 41.4 | 179 |
| _constant_causal_ | 1,056 | 77.1 | 158.9% | 92.2 | 75.1 | 118.9 | 0 |
| _constant_oracle_ | 1,056 | 41.1 | 84.7% | 54.7 | -11.7 | 32.1 | 0 |
| _climatology_causal_ | 1,056 | 76.1 | 156.9% | 90.2 | 75.1 | 118.9 | 0 |
| _climatology_oracle_ | 1,056 | 25.7 | 52.9% | 42.4 | -2.9 | 40.8 | 220 |
| control@101 | 1,056 | 19.1 | 39.3% | 31.5 | -2.9 | 40.8 | 144 |
| control@103 | 1,056 | 18.6 | 38.4% | 31.6 | -2.7 | 41.1 | 142 |
| control@107 | 1,056 | 18.8 | 38.7% | 31.4 | -3.1 | 40.7 | 146 |
| control@109 | 1,056 | 18.6 | 38.3% | 31.3 | -2.7 | 41.1 | 143 |
| control@113 | 1,056 | 18.7 | 38.6% | 31.1 | -3.0 | 40.7 | 154 |
| control@127 | 1,056 | 18.9 | 39.0% | 31.4 | -2.7 | 41.1 | 140 |
| control@131 | 1,056 | 18.5 | 38.1% | 30.8 | -3.1 | 40.6 | 150 |
| control@137 | 1,056 | 18.5 | 38.0% | 31.3 | -3.7 | 40.1 | 153 |
| control_noholiday@101 | 1,056 | 19.4 | 40.0% | 32.5 | -2.2 | 41.5 | 120 |
| control_noholiday@103 | 1,056 | 19.2 | 39.6% | 32.2 | -1.7 | 42.1 | 126 |
| control_noholiday@107 | 1,056 | 19.2 | 39.5% | 32.4 | -0.8 | 43.0 | 104 |
| control_noholiday@109 | 1,056 | 19.6 | 40.4% | 32.8 | -2.5 | 41.3 | 130 |
| control_noholiday@113 | 1,056 | 19.8 | 40.8% | 32.8 | -1.6 | 42.2 | 114 |
| control_noholiday@127 | 1,056 | 19.3 | 39.8% | 32.3 | -1.6 | 42.1 | 116 |
| control_noholiday@131 | 1,056 | 19.3 | 39.9% | 32.7 | -2.4 | 41.3 | 127 |
| control_noholiday@137 | 1,056 | 19.4 | 40.0% | 32.7 | -1.7 | 42.1 | 113 |

Holiday subsets of the holdout (holiday 96 · holiday_affected 288 · ordinary 768). `holiday_affected` is a holiday, a bridge day, or within one day of a holiday — the rows these four features can distinguish from an ordinary day at all. A holiday effect that is real here is diluted by the row counts in the all-hours table above.

| arm | holiday MAE | holiday_affected MAE | ordinary MAE |
|---|---:|---:|---:|
| _seasonal-naive D-7_ | 71.8 | 50.4 | 37.6 |
| _constant_causal_ | 100.4 | 87.7 | 73.1 |
| _constant_oracle_ | 50.2 | 41.5 | 40.9 |
| _climatology_causal_ | 100.4 | 86.8 | 72.1 |
| _climatology_oracle_ | 29.5 | 26.2 | 25.4 |
| control@101 | 29.1 | 20.8 | 18.5 |
| control@103 | 27.9 | 20.5 | 17.9 |
| control@107 | 28.0 | 20.2 | 18.3 |
| control@109 | 27.6 | 20.3 | 17.9 |
| control@113 | 28.9 | 21.2 | 17.8 |
| control@127 | 28.7 | 21.1 | 18.1 |
| control@131 | 27.8 | 20.3 | 17.8 |
| control@137 | 27.0 | 19.9 | 17.9 |
| control_noholiday@101 | 32.5 | 22.3 | 18.3 |
| control_noholiday@103 | 32.7 | 22.2 | 18.1 |
| control_noholiday@107 | 33.7 | 22.6 | 17.9 |
| control_noholiday@109 | 32.8 | 22.5 | 18.5 |
| control_noholiday@113 | 34.4 | 23.3 | 18.5 |
| control_noholiday@127 | 32.6 | 22.5 | 18.1 |
| control_noholiday@131 | 33.1 | 22.5 | 18.2 |
| control_noholiday@137 | 33.1 | 22.6 | 18.2 |

ABL-337 night screen: not applicable to price.
