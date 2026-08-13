# Held-out A/B — wind_onshore (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T14:10:29 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-07-13 .. 2026-08-11**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `wind_onshore` has no band structure, so one all-hours row is the result.

## AT / wind_onshore — catboost, source `energy_renewable`

n_train 5,318 · n_holdout 720 · incumbent version 20260112_165238

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 773.3 | 104.7% | 991.2 | 76.0 | 814.9 | 0 |
| control@42 | 427.0 | 57.8% | 540.9 | 102.4 | 841.3 | 0 |
| control@1337 | 428.3 | 58.0% | 531.8 | 99.0 | 837.9 | 0 |
| control@2718 | 420.7 | 56.9% | 529.3 | 111.3 | 850.2 | 0 |
| control@7 | 413.5 | 56.0% | 524.3 | 91.2 | 830.1 | 0 |
| control@13 | 415.9 | 56.3% | 531.4 | 85.6 | 824.5 | 0 |
| control@101 | 423.3 | 57.3% | 525.1 | 101.9 | 840.8 | 0 |
| control@271 | 426.6 | 57.7% | 531.1 | 110.9 | 849.8 | 0 |
| control@314 | 430.0 | 58.2% | 539.1 | 112.7 | 851.7 | 0 |
| control@577 | 425.2 | 57.5% | 537.6 | 118.2 | 857.1 | 0 |
| control@863 | 424.0 | 57.4% | 532.7 | 104.8 | 843.7 | 0 |
| control@1024 | 415.0 | 56.2% | 532.4 | 93.3 | 832.3 | 0 |
| control@1729 | 427.2 | 57.8% | 532.9 | 107.9 | 846.8 | 0 |

ABL-337 night screen: not applicable to wind_onshore.

## BE / wind_onshore — catboost, source `energy_renewable`

n_train 21,988 · n_holdout 720 · incumbent version 20260201_222020

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 376.1 | 83.8% | 481.7 | 23.3 | 472.2 | 0 |
| control@42 | 187.9 | 41.9% | 244.4 | 30.6 | 479.5 | 0 |
| control@1337 | 187.2 | 41.7% | 240.9 | 22.0 | 470.9 | 1 |
| control@2718 | 188.1 | 41.9% | 243.6 | 23.7 | 472.6 | 1 |
| control@7 | 187.9 | 41.9% | 245.1 | 25.2 | 474.1 | 0 |
| control@13 | 188.9 | 42.1% | 244.7 | 28.2 | 477.1 | 0 |
| control@101 | 189.8 | 42.3% | 241.9 | 21.5 | 470.4 | 0 |
| control@271 | 183.5 | 40.9% | 237.2 | 18.6 | 467.5 | 0 |
| control@314 | 186.8 | 41.6% | 242.5 | 28.2 | 477.1 | 1 |
| control@577 | 191.9 | 42.8% | 246.3 | 29.0 | 477.9 | 0 |
| control@863 | 188.2 | 41.9% | 243.4 | 25.1 | 474.0 | 0 |
| control@1024 | 184.4 | 41.1% | 237.6 | 17.0 | 465.9 | 1 |
| control@1729 | 190.3 | 42.4% | 244.9 | 22.6 | 471.5 | 0 |

ABL-337 night screen: not applicable to wind_onshore.

## DE / wind_onshore — catboost, source `energy_renewable`

n_train 7,253 · n_holdout 720 · incumbent version 20260201_222000

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 6,138.1 | 69.9% | 8,120.2 | 854.7 | 9,634.7 | 0 |
| control@42 | 2,776.4 | 31.6% | 3,473.7 | 754.5 | 9,534.5 | 0 |
| control@1337 | 2,739.9 | 31.2% | 3,448.8 | 818.3 | 9,598.3 | 0 |
| control@2718 | 2,899.6 | 33.0% | 3,609.6 | 915.1 | 9,695.1 | 0 |
| control@7 | 2,800.4 | 31.9% | 3,482.4 | 703.1 | 9,483.0 | 0 |
| control@13 | 2,812.6 | 32.0% | 3,476.1 | 667.3 | 9,447.3 | 0 |
| control@101 | 2,765.3 | 31.5% | 3,444.2 | 685.4 | 9,465.4 | 0 |
| control@271 | 2,871.5 | 32.7% | 3,592.7 | 770.4 | 9,550.4 | 0 |
| control@314 | 2,801.3 | 31.9% | 3,546.1 | 773.1 | 9,553.1 | 0 |
| control@577 | 2,797.6 | 31.9% | 3,492.9 | 752.9 | 9,532.9 | 0 |
| control@863 | 2,785.1 | 31.7% | 3,479.8 | 694.9 | 9,474.9 | 0 |
| control@1024 | 2,850.6 | 32.5% | 3,543.3 | 889.6 | 9,669.6 | 0 |
| control@1729 | 2,738.9 | 31.2% | 3,470.8 | 725.1 | 9,505.1 | 0 |

ABL-337 night screen: not applicable to wind_onshore.

## FR / wind_onshore — catboost, source `energy_renewable`

n_train 30,876 · n_holdout 720 · incumbent version 20260201_222010

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,753.9 | 54.4% | 2,328.9 | 175.1 | 3,400.8 | 0 |
| control@42 | 1,108.1 | 34.4% | 1,505.2 | 398.1 | 3,623.7 | 0 |
| control@1337 | 1,077.5 | 33.4% | 1,467.7 | 369.6 | 3,595.2 | 0 |
| control@2718 | 1,086.7 | 33.7% | 1,462.4 | 348.9 | 3,574.5 | 0 |
| control@7 | 1,087.3 | 33.7% | 1,475.6 | 370.2 | 3,595.8 | 0 |
| control@13 | 1,105.0 | 34.3% | 1,530.8 | 395.6 | 3,621.2 | 0 |
| control@101 | 1,094.0 | 33.9% | 1,489.7 | 390.9 | 3,616.5 | 0 |
| control@271 | 1,092.9 | 33.9% | 1,479.0 | 396.0 | 3,621.7 | 0 |
| control@314 | 1,091.7 | 33.8% | 1,460.3 | 347.1 | 3,572.8 | 0 |
| control@577 | 1,113.3 | 34.5% | 1,514.0 | 436.9 | 3,662.6 | 0 |
| control@863 | 1,092.3 | 33.9% | 1,497.9 | 396.6 | 3,622.2 | 0 |
| control@1024 | 1,087.5 | 33.7% | 1,470.6 | 404.2 | 3,629.9 | 0 |
| control@1729 | 1,089.2 | 33.8% | 1,478.0 | 391.9 | 3,617.6 | 0 |

ABL-337 night screen: not applicable to wind_onshore.
