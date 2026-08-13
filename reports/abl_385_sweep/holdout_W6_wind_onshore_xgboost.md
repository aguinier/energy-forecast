# Held-out A/B — wind_onshore (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T14:11:56 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-07-13 .. 2026-08-11**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `wind_onshore` has no band structure, so one all-hours row is the result.

## AT / wind_onshore — xgboost, source `energy_renewable`

n_train 5,318 · n_holdout 720 · incumbent version 20260112_165238

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 773.3 | 104.7% | 991.2 | 76.0 | 814.9 | 0 |
| control@42 | 439.8 | 59.5% | 546.9 | 131.6 | 870.5 | 0 |
| control@1337 | 425.5 | 57.6% | 533.1 | 139.7 | 878.6 | 0 |
| control@2718 | 438.5 | 59.3% | 543.1 | 136.8 | 875.7 | 0 |
| control@7 | 445.5 | 60.3% | 550.2 | 157.2 | 896.1 | 0 |
| control@13 | 438.5 | 59.3% | 543.4 | 141.6 | 880.5 | 0 |
| control@101 | 440.3 | 59.6% | 548.2 | 143.4 | 882.3 | 0 |
| control@271 | 439.4 | 59.5% | 544.2 | 139.9 | 878.8 | 0 |
| control@314 | 435.5 | 58.9% | 534.2 | 166.0 | 904.9 | 0 |
| control@577 | 434.2 | 58.8% | 534.6 | 135.9 | 874.8 | 0 |
| control@863 | 452.0 | 61.2% | 564.4 | 165.8 | 904.7 | 0 |
| control@1024 | 442.5 | 59.9% | 546.4 | 132.1 | 871.0 | 0 |
| control@1729 | 435.3 | 58.9% | 537.1 | 141.6 | 880.5 | 0 |

ABL-337 night screen: not applicable to wind_onshore.

## BE / wind_onshore — xgboost, source `energy_renewable`

n_train 21,988 · n_holdout 720 · incumbent version 20260201_222020

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 376.1 | 83.8% | 481.7 | 23.3 | 472.2 | 0 |
| control@42 | 172.0 | 38.3% | 227.0 | 8.6 | 457.5 | 1 |
| control@1337 | 172.0 | 38.3% | 226.2 | 9.0 | 457.9 | 0 |
| control@2718 | 174.3 | 38.8% | 229.4 | 15.4 | 464.3 | 1 |
| control@7 | 168.3 | 37.5% | 222.7 | 4.2 | 453.1 | 0 |
| control@13 | 172.4 | 38.4% | 230.7 | 16.3 | 465.2 | 1 |
| control@101 | 172.6 | 38.5% | 227.7 | 14.5 | 463.4 | 1 |
| control@271 | 169.8 | 37.8% | 225.2 | 11.2 | 460.1 | 2 |
| control@314 | 171.0 | 38.1% | 225.2 | 11.0 | 459.9 | 2 |
| control@577 | 172.3 | 38.4% | 226.8 | 6.9 | 455.8 | 1 |
| control@863 | 174.1 | 38.8% | 229.2 | 11.9 | 460.8 | 2 |
| control@1024 | 172.1 | 38.3% | 227.6 | 10.6 | 459.5 | 2 |
| control@1729 | 171.7 | 38.2% | 229.7 | 11.1 | 460.0 | 0 |

ABL-337 night screen: not applicable to wind_onshore.

## DE / wind_onshore — xgboost, source `energy_renewable`

n_train 7,253 · n_holdout 720 · incumbent version 20260201_222000

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 6,138.1 | 69.9% | 8,120.2 | 854.7 | 9,634.7 | 0 |
| control@42 | 2,914.3 | 33.2% | 3,676.4 | 1,337.6 | 10,117.6 | 0 |
| control@1337 | 2,905.9 | 33.1% | 3,657.1 | 1,233.7 | 10,013.7 | 0 |
| control@2718 | 2,917.4 | 33.2% | 3,667.9 | 1,214.9 | 9,994.9 | 0 |
| control@7 | 2,810.3 | 32.0% | 3,526.2 | 1,165.5 | 9,945.5 | 0 |
| control@13 | 2,908.2 | 33.1% | 3,711.0 | 1,230.1 | 10,010.1 | 0 |
| control@101 | 2,884.6 | 32.9% | 3,658.1 | 1,180.7 | 9,960.7 | 0 |
| control@271 | 2,922.7 | 33.3% | 3,733.8 | 1,325.4 | 10,105.3 | 0 |
| control@314 | 2,950.6 | 33.6% | 3,722.4 | 1,243.7 | 10,023.7 | 0 |
| control@577 | 2,924.4 | 33.3% | 3,734.4 | 1,317.5 | 10,097.4 | 0 |
| control@863 | 2,911.2 | 33.2% | 3,689.2 | 1,111.0 | 9,891.0 | 0 |
| control@1024 | 2,796.9 | 31.9% | 3,577.0 | 1,123.2 | 9,903.2 | 0 |
| control@1729 | 2,814.1 | 32.1% | 3,592.0 | 1,093.2 | 9,873.2 | 0 |

ABL-337 night screen: not applicable to wind_onshore.

## FR / wind_onshore — xgboost, source `energy_renewable`

n_train 30,876 · n_holdout 720 · incumbent version 20260201_222010

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,753.9 | 54.4% | 2,328.9 | 175.1 | 3,400.8 | 0 |
| control@42 | 1,099.0 | 34.1% | 1,519.4 | 451.9 | 3,677.6 | 0 |
| control@1337 | 1,140.9 | 35.4% | 1,558.2 | 558.9 | 3,784.6 | 0 |
| control@2718 | 1,116.9 | 34.6% | 1,524.3 | 528.4 | 3,754.1 | 0 |
| control@7 | 1,190.2 | 36.9% | 1,633.3 | 660.5 | 3,886.1 | 0 |
| control@13 | 1,088.8 | 33.8% | 1,520.2 | 493.0 | 3,718.7 | 0 |
| control@101 | 1,098.4 | 34.1% | 1,576.8 | 506.0 | 3,731.6 | 0 |
| control@271 | 1,114.9 | 34.6% | 1,534.2 | 526.8 | 3,752.5 | 0 |
| control@314 | 1,073.7 | 33.3% | 1,459.5 | 437.8 | 3,663.5 | 0 |
| control@577 | 1,126.4 | 34.9% | 1,587.9 | 510.8 | 3,736.4 | 0 |
| control@863 | 1,111.7 | 34.5% | 1,552.4 | 506.6 | 3,732.2 | 0 |
| control@1024 | 1,096.7 | 34.0% | 1,524.0 | 489.9 | 3,715.5 | 0 |
| control@1729 | 1,122.5 | 34.8% | 1,515.4 | 529.0 | 3,754.7 | 0 |

ABL-337 night screen: not applicable to wind_onshore.
