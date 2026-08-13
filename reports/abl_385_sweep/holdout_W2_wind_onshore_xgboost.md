# Held-out A/B — wind_onshore (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T13:31:11 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-03-15 .. 2026-04-13**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `wind_onshore` has no band structure, so one all-hours row is the result.

## AT / wind_onshore — xgboost, source `energy_renewable`

n_train 2,438 · n_holdout 720 · incumbent version 20260112_165238

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,480.2 | 96.6% | 1,797.3 | 57.3 | 1,590.1 | 0 |
| control@42 | 458.6 | 29.9% | 574.5 | -171.5 | 1,361.3 | 0 |
| control@1337 | 437.6 | 28.6% | 544.5 | -164.5 | 1,368.3 | 0 |
| control@2718 | 482.6 | 31.5% | 596.8 | -168.8 | 1,364.1 | 0 |
| control@7 | 445.6 | 29.1% | 559.2 | -149.9 | 1,382.9 | 0 |
| control@13 | 454.8 | 29.7% | 563.8 | -171.5 | 1,361.3 | 0 |
| control@101 | 456.2 | 29.8% | 571.3 | -180.0 | 1,352.8 | 0 |
| control@271 | 455.9 | 29.7% | 567.5 | -192.0 | 1,340.9 | 0 |
| control@314 | 451.5 | 29.5% | 574.3 | -179.6 | 1,353.3 | 0 |
| control@577 | 475.0 | 31.0% | 604.5 | -228.4 | 1,304.4 | 0 |
| control@863 | 467.7 | 30.5% | 585.5 | -175.2 | 1,357.6 | 0 |
| control@1024 | 460.1 | 30.0% | 577.3 | -172.5 | 1,360.3 | 0 |
| control@1729 | 464.3 | 30.3% | 572.1 | -170.7 | 1,362.1 | 0 |

ABL-337 night screen: not applicable to wind_onshore.

## BE / wind_onshore — xgboost, source `energy_renewable`

n_train 19,108 · n_holdout 720 · incumbent version 20260201_222020

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,011.2 | 110.2% | 1,313.0 | 97.2 | 1,015.0 | 0 |
| control@42 | 319.9 | 34.8% | 401.6 | 71.7 | 989.5 | 0 |
| control@1337 | 312.7 | 34.1% | 390.9 | 70.2 | 988.1 | 0 |
| control@2718 | 300.8 | 32.8% | 375.9 | 69.5 | 987.3 | 0 |
| control@7 | 311.8 | 34.0% | 388.7 | 44.8 | 962.7 | 0 |
| control@13 | 316.0 | 34.4% | 386.1 | 69.6 | 987.5 | 0 |
| control@101 | 306.6 | 33.4% | 381.2 | 80.4 | 998.2 | 0 |
| control@271 | 316.2 | 34.4% | 390.7 | 83.9 | 1,001.7 | 0 |
| control@314 | 312.9 | 34.1% | 389.3 | 79.5 | 997.3 | 0 |
| control@577 | 319.9 | 34.9% | 394.2 | 82.2 | 1,000.1 | 0 |
| control@863 | 310.8 | 33.9% | 391.4 | 63.3 | 981.1 | 0 |
| control@1024 | 317.4 | 34.6% | 397.7 | 62.0 | 979.8 | 0 |
| control@1729 | 309.2 | 33.7% | 387.4 | 83.8 | 1,001.6 | 0 |

ABL-337 night screen: not applicable to wind_onshore.

## DE / wind_onshore — xgboost, source `energy_renewable`

n_train 4,373 · n_holdout 720 · incumbent version 20260201_222000

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 12,854.4 | 99.5% | 16,005.5 | 773.9 | 13,692.0 | 0 |
| control@42 | 4,797.1 | 37.1% | 6,142.8 | 2,011.0 | 14,929.1 | 0 |
| control@1337 | 4,736.2 | 36.7% | 6,045.5 | 2,008.0 | 14,926.0 | 0 |
| control@2718 | 4,868.0 | 37.7% | 6,131.8 | 1,757.4 | 14,675.5 | 0 |
| control@7 | 4,673.9 | 36.2% | 5,985.6 | 1,946.7 | 14,864.7 | 0 |
| control@13 | 4,875.5 | 37.7% | 6,275.1 | 2,229.2 | 15,147.3 | 0 |
| control@101 | 4,702.0 | 36.4% | 6,107.5 | 2,029.9 | 14,948.0 | 0 |
| control@271 | 4,711.6 | 36.5% | 6,047.3 | 1,884.6 | 14,802.6 | 0 |
| control@314 | 4,717.1 | 36.5% | 6,079.7 | 1,988.9 | 14,906.9 | 0 |
| control@577 | 4,817.2 | 37.3% | 6,155.4 | 2,070.1 | 14,988.2 | 0 |
| control@863 | 4,790.1 | 37.1% | 6,150.4 | 1,921.2 | 14,839.3 | 0 |
| control@1024 | 4,878.2 | 37.8% | 6,277.1 | 2,035.9 | 14,953.9 | 0 |
| control@1729 | 4,769.4 | 36.9% | 6,013.4 | 2,069.5 | 14,987.5 | 0 |

ABL-337 night screen: not applicable to wind_onshore.

## FR / wind_onshore — xgboost, source `energy_renewable`

n_train 27,996 · n_holdout 720 · incumbent version 20260201_222010

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 3,899.8 | 75.1% | 5,080.0 | 513.0 | 5,704.1 | 0 |
| control@42 | 2,034.6 | 39.2% | 2,537.7 | 717.0 | 5,908.1 | 0 |
| control@1337 | 2,023.9 | 39.0% | 2,542.6 | 764.8 | 5,955.9 | 0 |
| control@2718 | 2,013.9 | 38.8% | 2,525.3 | 656.0 | 5,847.1 | 0 |
| control@7 | 1,985.5 | 38.2% | 2,514.2 | 815.9 | 6,007.0 | 0 |
| control@13 | 1,999.2 | 38.5% | 2,544.6 | 760.1 | 5,951.1 | 0 |
| control@101 | 1,987.7 | 38.3% | 2,547.6 | 802.5 | 5,993.6 | 0 |
| control@271 | 2,018.3 | 38.9% | 2,536.1 | 797.8 | 5,988.9 | 0 |
| control@314 | 2,000.1 | 38.5% | 2,602.5 | 865.1 | 6,056.2 | 0 |
| control@577 | 1,956.2 | 37.7% | 2,495.5 | 732.9 | 5,924.0 | 0 |
| control@863 | 1,988.2 | 38.3% | 2,511.5 | 728.8 | 5,919.9 | 0 |
| control@1024 | 2,025.3 | 39.0% | 2,581.9 | 822.9 | 6,014.0 | 0 |
| control@1729 | 2,022.9 | 39.0% | 2,555.5 | 690.6 | 5,881.7 | 0 |

ABL-337 night screen: not applicable to wind_onshore.
