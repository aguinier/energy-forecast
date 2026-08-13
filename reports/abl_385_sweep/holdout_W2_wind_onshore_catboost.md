# Held-out A/B — wind_onshore (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T13:29:49 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-03-15 .. 2026-04-13**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `wind_onshore` has no band structure, so one all-hours row is the result.

## AT / wind_onshore — catboost, source `energy_renewable`

n_train 2,438 · n_holdout 720 · incumbent version 20260112_165238

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,480.2 | 96.6% | 1,797.3 | 57.3 | 1,590.1 | 0 |
| control@42 | 598.8 | 39.1% | 802.2 | -351.7 | 1,181.1 | 0 |
| control@1337 | 611.1 | 39.9% | 800.4 | -357.6 | 1,175.2 | 0 |
| control@2718 | 617.6 | 40.3% | 812.5 | -382.1 | 1,150.7 | 0 |
| control@7 | 600.0 | 39.1% | 800.5 | -351.8 | 1,181.0 | 0 |
| control@13 | 610.3 | 39.8% | 802.7 | -355.1 | 1,177.7 | 0 |
| control@101 | 626.0 | 40.8% | 833.0 | -381.3 | 1,151.5 | 0 |
| control@271 | 640.9 | 41.8% | 847.8 | -386.5 | 1,146.3 | 0 |
| control@314 | 600.5 | 39.2% | 791.3 | -346.7 | 1,186.2 | 0 |
| control@577 | 629.1 | 41.0% | 821.0 | -341.4 | 1,191.5 | 0 |
| control@863 | 599.6 | 39.1% | 785.2 | -341.1 | 1,191.7 | 0 |
| control@1024 | 626.1 | 40.8% | 815.3 | -365.4 | 1,167.4 | 0 |
| control@1729 | 629.9 | 41.1% | 834.4 | -368.3 | 1,164.6 | 0 |

ABL-337 night screen: not applicable to wind_onshore.

## BE / wind_onshore — catboost, source `energy_renewable`

n_train 19,108 · n_holdout 720 · incumbent version 20260201_222020

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,011.2 | 110.2% | 1,313.0 | 97.2 | 1,015.0 | 0 |
| control@42 | 308.6 | 33.6% | 384.3 | 93.0 | 1,010.9 | 0 |
| control@1337 | 316.2 | 34.5% | 397.9 | 91.8 | 1,009.7 | 0 |
| control@2718 | 315.2 | 34.3% | 400.4 | 94.2 | 1,012.1 | 0 |
| control@7 | 330.1 | 36.0% | 416.7 | 121.0 | 1,038.9 | 0 |
| control@13 | 325.7 | 35.5% | 413.8 | 115.4 | 1,033.2 | 0 |
| control@101 | 310.8 | 33.9% | 399.2 | 80.5 | 998.3 | 2 |
| control@271 | 309.2 | 33.7% | 392.5 | 87.4 | 1,005.3 | 0 |
| control@314 | 309.8 | 33.8% | 400.6 | 69.9 | 987.8 | 0 |
| control@577 | 315.8 | 34.4% | 407.1 | 84.1 | 1,001.9 | 0 |
| control@863 | 306.8 | 33.4% | 396.7 | 92.7 | 1,010.6 | 0 |
| control@1024 | 320.2 | 34.9% | 402.1 | 92.2 | 1,010.1 | 0 |
| control@1729 | 313.6 | 34.2% | 400.6 | 68.9 | 986.7 | 0 |

ABL-337 night screen: not applicable to wind_onshore.

## DE / wind_onshore — catboost, source `energy_renewable`

n_train 4,373 · n_holdout 720 · incumbent version 20260201_222000

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 12,854.4 | 99.5% | 16,005.5 | 773.9 | 13,692.0 | 0 |
| control@42 | 4,812.1 | 37.3% | 5,987.6 | 1,601.6 | 14,519.6 | 0 |
| control@1337 | 5,106.4 | 39.5% | 6,173.6 | 1,468.6 | 14,386.6 | 0 |
| control@2718 | 4,947.1 | 38.3% | 6,079.0 | 1,808.2 | 14,726.2 | 0 |
| control@7 | 4,659.8 | 36.1% | 5,813.5 | 1,670.0 | 14,588.0 | 0 |
| control@13 | 4,986.0 | 38.6% | 6,130.1 | 1,876.8 | 14,794.9 | 0 |
| control@101 | 4,903.4 | 38.0% | 6,053.7 | 1,790.7 | 14,708.7 | 0 |
| control@271 | 4,929.2 | 38.2% | 6,019.0 | 1,841.5 | 14,759.5 | 0 |
| control@314 | 4,812.9 | 37.3% | 5,910.3 | 1,635.2 | 14,553.3 | 0 |
| control@577 | 4,884.9 | 37.8% | 6,095.0 | 1,533.4 | 14,451.4 | 0 |
| control@863 | 4,763.6 | 36.9% | 5,983.8 | 1,431.1 | 14,349.1 | 0 |
| control@1024 | 4,877.1 | 37.8% | 6,010.6 | 1,522.6 | 14,440.7 | 0 |
| control@1729 | 4,949.9 | 38.3% | 6,141.6 | 1,541.9 | 14,459.9 | 0 |

ABL-337 night screen: not applicable to wind_onshore.

## FR / wind_onshore — catboost, source `energy_renewable`

n_train 27,996 · n_holdout 720 · incumbent version 20260201_222010

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 3,899.8 | 75.1% | 5,080.0 | 513.0 | 5,704.1 | 0 |
| control@42 | 2,047.7 | 39.4% | 2,556.1 | 724.6 | 5,915.7 | 0 |
| control@1337 | 2,045.0 | 39.4% | 2,548.7 | 721.7 | 5,912.8 | 0 |
| control@2718 | 2,076.3 | 40.0% | 2,604.5 | 736.8 | 5,927.9 | 0 |
| control@7 | 2,087.8 | 40.2% | 2,588.8 | 772.7 | 5,963.7 | 0 |
| control@13 | 2,018.7 | 38.9% | 2,522.8 | 701.3 | 5,892.4 | 0 |
| control@101 | 1,983.8 | 38.2% | 2,509.2 | 727.4 | 5,918.5 | 0 |
| control@271 | 1,931.0 | 37.2% | 2,485.3 | 666.6 | 5,857.7 | 0 |
| control@314 | 2,063.0 | 39.7% | 2,560.3 | 723.2 | 5,914.2 | 0 |
| control@577 | 2,020.2 | 38.9% | 2,543.8 | 737.5 | 5,928.6 | 0 |
| control@863 | 1,992.7 | 38.4% | 2,552.2 | 773.2 | 5,964.3 | 0 |
| control@1024 | 2,063.2 | 39.7% | 2,553.2 | 732.0 | 5,923.1 | 0 |
| control@1729 | 2,037.1 | 39.2% | 2,578.8 | 808.0 | 5,999.1 | 0 |

ABL-337 night screen: not applicable to wind_onshore.
