# Held-out A/B — price (ABL-393 reuse of the ABL-338 harness)

Generated 2026-08-13T21:51:08 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-04-30 .. 2026-06-12**, training from 2021-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `price` has no band structure, so one all-hours row is the result.

## AT / price — catboost, one fixed table for `price`

n_train 35,525 · n_holdout 1,056 · incumbent version 20260202_144224

| arm | n | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,056 | 32.4 | 30.8% | 59.3 | -5.1 | 95.0 | 107 |
| _constant_causal_ | 1,056 | 46.6 | 44.4% | 73.6 | 35.5 | 135.5 | 0 |
| _constant_oracle_ | 1,056 | 43.5 | 41.3% | 67.3 | 19.2 | 119.3 | 0 |
| _climatology_causal_ | 1,056 | 43.8 | 41.6% | 67.3 | 35.4 | 135.5 | 0 |
| _climatology_oracle_ | 1,056 | 26.1 | 24.8% | 44.2 | -1.2 | 98.8 | 0 |
| control@101 | 1,056 | 19.4 | 18.5% | 29.9 | -6.3 | 93.8 | 85 |
| control@103 | 1,056 | 18.5 | 17.6% | 29.4 | -4.7 | 95.4 | 74 |
| control@107 | 1,056 | 19.6 | 18.6% | 30.5 | -7.8 | 92.3 | 88 |
| control@109 | 1,056 | 19.7 | 18.8% | 29.8 | -6.7 | 93.3 | 86 |
| control@113 | 1,056 | 19.0 | 18.0% | 30.1 | -5.5 | 94.6 | 81 |
| control@127 | 1,056 | 19.4 | 18.4% | 30.2 | -6.7 | 93.3 | 88 |
| control@131 | 1,056 | 19.4 | 18.4% | 30.3 | -6.0 | 94.0 | 88 |
| control@137 | 1,056 | 19.7 | 18.8% | 30.3 | -6.4 | 93.6 | 83 |
| control_noholiday@101 | 1,056 | 20.4 | 19.4% | 31.7 | -7.6 | 92.4 | 85 |
| control_noholiday@103 | 1,056 | 20.5 | 19.5% | 32.1 | -7.2 | 92.8 | 84 |
| control_noholiday@107 | 1,056 | 20.0 | 19.0% | 31.8 | -5.9 | 94.1 | 76 |
| control_noholiday@109 | 1,056 | 20.3 | 19.3% | 32.0 | -6.7 | 93.4 | 76 |
| control_noholiday@113 | 1,056 | 19.2 | 18.3% | 31.3 | -4.7 | 95.4 | 76 |
| control_noholiday@127 | 1,056 | 20.4 | 19.4% | 32.1 | -7.2 | 92.9 | 77 |
| control_noholiday@131 | 1,056 | 20.2 | 19.2% | 32.2 | -7.2 | 92.8 | 83 |
| control_noholiday@137 | 1,056 | 19.8 | 18.8% | 32.3 | -6.1 | 93.9 | 73 |

Holiday subsets of the holdout (holiday 96 · holiday_affected 288 · ordinary 768). `holiday_affected` is a holiday, a bridge day, or within one day of a holiday — the rows these four features can distinguish from an ordinary day at all. A holiday effect that is real here is diluted by the row counts in the all-hours table above.

| arm | holiday MAE | holiday_affected MAE | ordinary MAE |
|---|---:|---:|---:|
| _seasonal-naive D-7_ | 56.1 | 35.0 | 31.3 |
| _constant_causal_ | 76.3 | 57.4 | 42.6 |
| _constant_oracle_ | 70.3 | 52.9 | 39.9 |
| _climatology_causal_ | 75.1 | 54.7 | 39.7 |
| _climatology_oracle_ | 40.8 | 30.6 | 24.4 |
| control@101 | 32.0 | 22.4 | 18.3 |
| control@103 | 31.1 | 21.0 | 17.6 |
| control@107 | 30.7 | 21.6 | 18.8 |
| control@109 | 32.0 | 22.5 | 18.7 |
| control@113 | 30.5 | 21.1 | 18.2 |
| control@127 | 32.6 | 22.3 | 18.2 |
| control@131 | 31.7 | 22.2 | 18.3 |
| control@137 | 32.3 | 22.6 | 18.7 |
| control_noholiday@101 | 34.3 | 24.4 | 18.9 |
| control_noholiday@103 | 34.9 | 25.3 | 18.7 |
| control_noholiday@107 | 37.2 | 24.2 | 18.4 |
| control_noholiday@109 | 36.6 | 25.3 | 18.5 |
| control_noholiday@113 | 37.0 | 24.2 | 17.4 |
| control_noholiday@127 | 37.7 | 25.1 | 18.6 |
| control_noholiday@131 | 36.2 | 24.1 | 18.8 |
| control_noholiday@137 | 36.9 | 24.9 | 17.8 |

ABL-337 night screen: not applicable to price.

## BE / price — catboost, one fixed table for `price`

n_train 37,883 · n_holdout 1,056 · incumbent version 20260404_185535

| arm | n | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,056 | 35.9 | 37.1% | 61.8 | -5.7 | 86.3 | 109 |
| _constant_causal_ | 1,056 | 42.4 | 43.9% | 66.7 | 28.5 | 120.6 | 0 |
| _constant_oracle_ | 1,056 | 40.4 | 41.7% | 62.4 | 16.3 | 108.3 | 0 |
| _climatology_causal_ | 1,056 | 37.5 | 38.8% | 58.7 | 28.5 | 120.6 | 0 |
| _climatology_oracle_ | 1,056 | 25.9 | 26.8% | 42.8 | 0.3 | 92.4 | 0 |
| control@101 | 1,056 | 19.2 | 19.9% | 31.1 | -7.3 | 84.7 | 78 |
| control@103 | 1,056 | 19.2 | 19.9% | 30.2 | -7.3 | 84.7 | 78 |
| control@107 | 1,056 | 19.8 | 20.4% | 30.9 | -8.6 | 83.4 | 85 |
| control@109 | 1,056 | 18.5 | 19.2% | 29.6 | -7.1 | 84.9 | 81 |
| control@113 | 1,056 | 19.2 | 19.9% | 31.1 | -6.6 | 85.4 | 71 |
| control@127 | 1,056 | 18.5 | 19.1% | 30.2 | -5.9 | 86.1 | 76 |
| control@131 | 1,056 | 19.3 | 19.9% | 30.6 | -6.6 | 85.5 | 72 |
| control@137 | 1,056 | 19.4 | 20.0% | 30.2 | -7.7 | 84.4 | 85 |
| control_noholiday@101 | 1,056 | 19.4 | 20.1% | 32.1 | -6.1 | 85.9 | 75 |
| control_noholiday@103 | 1,056 | 19.3 | 20.0% | 31.7 | -7.3 | 84.8 | 76 |
| control_noholiday@107 | 1,056 | 19.3 | 20.0% | 31.7 | -6.9 | 85.1 | 77 |
| control_noholiday@109 | 1,056 | 20.0 | 20.7% | 31.3 | -8.7 | 83.4 | 79 |
| control_noholiday@113 | 1,056 | 19.0 | 19.7% | 31.1 | -6.9 | 85.1 | 82 |
| control_noholiday@127 | 1,056 | 19.7 | 20.3% | 31.3 | -7.6 | 84.4 | 75 |
| control_noholiday@131 | 1,056 | 19.4 | 20.0% | 31.3 | -6.4 | 85.6 | 76 |
| control_noholiday@137 | 1,056 | 19.7 | 20.4% | 31.9 | -6.5 | 85.6 | 68 |

Holiday subsets of the holdout (holiday 96 · holiday_affected 240 · ordinary 816). `holiday_affected` is a holiday, a bridge day, or within one day of a holiday — the rows these four features can distinguish from an ordinary day at all. A holiday effect that is real here is diluted by the row counts in the all-hours table above.

| arm | holiday MAE | holiday_affected MAE | ordinary MAE |
|---|---:|---:|---:|
| _seasonal-naive D-7_ | 52.5 | 35.4 | 36.0 |
| _constant_causal_ | 74.3 | 60.3 | 37.2 |
| _constant_oracle_ | 70.7 | 56.7 | 35.6 |
| _climatology_causal_ | 68.4 | 53.7 | 32.8 |
| _climatology_oracle_ | 40.3 | 29.7 | 24.8 |
| control@101 | 35.1 | 24.1 | 17.8 |
| control@103 | 34.2 | 24.2 | 17.8 |
| control@107 | 35.3 | 24.8 | 18.3 |
| control@109 | 33.0 | 23.0 | 17.2 |
| control@113 | 34.8 | 24.1 | 17.8 |
| control@127 | 32.7 | 23.2 | 17.1 |
| control@131 | 35.1 | 24.1 | 17.8 |
| control@137 | 34.2 | 25.2 | 17.6 |
| control_noholiday@101 | 33.4 | 23.8 | 18.2 |
| control_noholiday@103 | 33.4 | 25.2 | 17.6 |
| control_noholiday@107 | 32.7 | 24.5 | 17.8 |
| control_noholiday@109 | 32.0 | 24.7 | 18.7 |
| control_noholiday@113 | 33.4 | 25.0 | 17.3 |
| control_noholiday@127 | 33.1 | 24.8 | 18.2 |
| control_noholiday@131 | 32.1 | 24.3 | 17.9 |
| control_noholiday@137 | 33.3 | 24.6 | 18.3 |

ABL-337 night screen: not applicable to price.

## DE / price — catboost, one fixed table for `price`

n_train 35,270 · n_holdout 1,056 · incumbent version 20260404_185523

| arm | n | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,056 | 35.2 | 34.1% | 62.5 | -6.5 | 91.6 | 134 |
| _constant_causal_ | 1,056 | 44.3 | 42.9% | 69.9 | 26.1 | 124.2 | 0 |
| _constant_oracle_ | 1,056 | 43.2 | 41.9% | 66.8 | 16.2 | 114.3 | 0 |
| _climatology_causal_ | 1,056 | 38.5 | 37.2% | 60.2 | 26.1 | 124.2 | 0 |
| _climatology_oracle_ | 1,056 | 26.7 | 25.8% | 44.4 | 1.5 | 99.6 | 0 |
| control@101 | 1,056 | 25.1 | 24.3% | 36.5 | -17.6 | 80.5 | 102 |
| control@103 | 1,056 | 25.4 | 24.6% | 36.4 | -18.2 | 79.9 | 108 |
| control@107 | 1,056 | 24.9 | 24.1% | 36.1 | -16.7 | 81.4 | 99 |
| control@109 | 1,056 | 24.2 | 23.4% | 35.4 | -16.1 | 82.0 | 96 |
| control@113 | 1,056 | 24.3 | 23.6% | 35.8 | -16.5 | 81.6 | 105 |
| control@127 | 1,056 | 24.2 | 23.4% | 35.6 | -15.8 | 82.3 | 100 |
| control@131 | 1,056 | 24.0 | 23.2% | 35.4 | -15.4 | 82.7 | 96 |
| control@137 | 1,056 | 24.4 | 23.6% | 35.8 | -15.9 | 82.2 | 104 |
| control_noholiday@101 | 1,056 | 23.6 | 22.8% | 35.2 | -13.7 | 84.4 | 91 |
| control_noholiday@103 | 1,056 | 23.9 | 23.1% | 35.4 | -13.9 | 84.2 | 92 |
| control_noholiday@107 | 1,056 | 24.2 | 23.4% | 35.7 | -14.6 | 83.5 | 87 |
| control_noholiday@109 | 1,056 | 25.0 | 24.2% | 36.6 | -15.9 | 82.2 | 101 |
| control_noholiday@113 | 1,056 | 24.3 | 23.5% | 35.5 | -15.3 | 82.8 | 98 |
| control_noholiday@127 | 1,056 | 24.5 | 23.7% | 36.4 | -14.8 | 83.3 | 95 |
| control_noholiday@131 | 1,056 | 23.8 | 23.1% | 36.0 | -13.3 | 84.8 | 94 |
| control_noholiday@137 | 1,056 | 24.4 | 23.6% | 35.9 | -14.8 | 83.3 | 93 |

Holiday subsets of the holdout (holiday 72 · holiday_affected 216 · ordinary 840). `holiday_affected` is a holiday, a bridge day, or within one day of a holiday — the rows these four features can distinguish from an ordinary day at all. A holiday effect that is real here is diluted by the row counts in the all-hours table above.

| arm | holiday MAE | holiday_affected MAE | ordinary MAE |
|---|---:|---:|---:|
| _seasonal-naive D-7_ | 62.2 | 36.1 | 34.9 |
| _constant_causal_ | 77.8 | 60.4 | 40.1 |
| _constant_oracle_ | 74.7 | 58.4 | 39.3 |
| _climatology_causal_ | 72.5 | 54.0 | 34.5 |
| _climatology_oracle_ | 45.0 | 32.4 | 25.2 |
| control@101 | 45.6 | 31.8 | 23.3 |
| control@103 | 46.6 | 33.3 | 23.4 |
| control@107 | 47.3 | 33.1 | 22.8 |
| control@109 | 46.9 | 32.0 | 22.1 |
| control@113 | 45.3 | 32.6 | 22.2 |
| control@127 | 46.2 | 32.6 | 22.0 |
| control@131 | 45.3 | 32.2 | 21.9 |
| control@137 | 46.4 | 32.9 | 22.2 |
| control_noholiday@101 | 45.0 | 30.7 | 21.7 |
| control_noholiday@103 | 44.2 | 30.7 | 22.2 |
| control_noholiday@107 | 44.7 | 32.9 | 21.9 |
| control_noholiday@109 | 44.0 | 32.2 | 23.1 |
| control_noholiday@113 | 44.2 | 31.7 | 22.4 |
| control_noholiday@127 | 43.3 | 30.3 | 23.0 |
| control_noholiday@131 | 44.5 | 31.6 | 21.8 |
| control_noholiday@137 | 45.2 | 32.6 | 22.3 |

ABL-337 night screen: not applicable to price.

## FR / price — catboost, one fixed table for `price`

n_train 37,331 · n_holdout 1,056 · incumbent version 20260404_185529

| arm | n | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,056 | 41.1 | 84.6% | 64.5 | -2.3 | 41.4 | 179 |
| _constant_causal_ | 1,056 | 77.1 | 158.9% | 92.2 | 75.1 | 118.9 | 0 |
| _constant_oracle_ | 1,056 | 41.1 | 84.7% | 54.7 | -11.7 | 32.1 | 0 |
| _climatology_causal_ | 1,056 | 76.1 | 156.9% | 90.2 | 75.1 | 118.9 | 0 |
| _climatology_oracle_ | 1,056 | 25.7 | 52.9% | 42.4 | -2.9 | 40.8 | 220 |
| control@101 | 1,056 | 19.2 | 39.5% | 31.6 | -2.4 | 41.3 | 164 |
| control@103 | 1,056 | 19.2 | 39.5% | 31.5 | -2.7 | 41.1 | 168 |
| control@107 | 1,056 | 19.4 | 40.0% | 31.8 | -2.5 | 41.3 | 166 |
| control@109 | 1,056 | 19.6 | 40.4% | 31.9 | -2.4 | 41.3 | 171 |
| control@113 | 1,056 | 19.7 | 40.7% | 31.9 | -3.2 | 40.5 | 180 |
| control@127 | 1,056 | 19.5 | 40.2% | 31.9 | -3.2 | 40.6 | 173 |
| control@131 | 1,056 | 20.0 | 41.3% | 32.3 | -3.3 | 40.5 | 170 |
| control@137 | 1,056 | 20.3 | 41.8% | 33.4 | -0.3 | 43.4 | 117 |
| control_noholiday@101 | 1,056 | 19.9 | 40.9% | 32.9 | -2.1 | 41.7 | 148 |
| control_noholiday@103 | 1,056 | 19.7 | 40.7% | 32.6 | -3.1 | 40.7 | 145 |
| control_noholiday@107 | 1,056 | 20.1 | 41.4% | 32.8 | -2.7 | 41.0 | 145 |
| control_noholiday@109 | 1,056 | 19.7 | 40.5% | 32.5 | -1.4 | 42.3 | 157 |
| control_noholiday@113 | 1,056 | 21.0 | 43.2% | 34.9 | 0.6 | 44.4 | 97 |
| control_noholiday@127 | 1,056 | 19.3 | 39.9% | 32.3 | -3.1 | 40.7 | 164 |
| control_noholiday@131 | 1,056 | 20.1 | 41.3% | 33.1 | -1.4 | 42.3 | 139 |
| control_noholiday@137 | 1,056 | 19.6 | 40.5% | 32.3 | -2.4 | 41.4 | 153 |

Holiday subsets of the holdout (holiday 96 · holiday_affected 288 · ordinary 768). `holiday_affected` is a holiday, a bridge day, or within one day of a holiday — the rows these four features can distinguish from an ordinary day at all. A holiday effect that is real here is diluted by the row counts in the all-hours table above.

| arm | holiday MAE | holiday_affected MAE | ordinary MAE |
|---|---:|---:|---:|
| _seasonal-naive D-7_ | 71.8 | 50.4 | 37.6 |
| _constant_causal_ | 100.4 | 87.7 | 73.1 |
| _constant_oracle_ | 50.2 | 41.5 | 40.9 |
| _climatology_causal_ | 100.4 | 86.8 | 72.1 |
| _climatology_oracle_ | 29.5 | 26.2 | 25.4 |
| control@101 | 28.3 | 20.3 | 18.8 |
| control@103 | 28.9 | 20.3 | 18.8 |
| control@107 | 28.7 | 20.8 | 18.9 |
| control@109 | 29.4 | 21.5 | 18.9 |
| control@113 | 30.7 | 22.2 | 18.8 |
| control@127 | 28.7 | 20.4 | 19.2 |
| control@131 | 29.4 | 21.5 | 19.5 |
| control@137 | 30.4 | 22.3 | 19.5 |
| control_noholiday@101 | 33.2 | 22.4 | 18.9 |
| control_noholiday@103 | 31.9 | 21.9 | 18.9 |
| control_noholiday@107 | 33.7 | 23.1 | 19.0 |
| control_noholiday@109 | 32.5 | 22.6 | 18.5 |
| control_noholiday@113 | 36.0 | 24.0 | 19.9 |
| control_noholiday@127 | 30.2 | 20.9 | 18.8 |
| control_noholiday@131 | 33.7 | 22.8 | 19.0 |
| control_noholiday@137 | 31.8 | 22.0 | 18.8 |

ABL-337 night screen: not applicable to price.
