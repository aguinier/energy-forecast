# Held-out A/B — wind_onshore (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T13:49:59 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-05-14 .. 2026-06-12**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `wind_onshore` has no band structure, so one all-hours row is the result.

## AT / wind_onshore — catboost, source `energy_renewable`

n_train 3,878 · n_holdout 720 · incumbent version 20260112_165238

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,043.3 | 118.7% | 1,354.0 | 51.0 | 930.0 | 0 |
| control@42 | 487.3 | 55.4% | 643.8 | 192.1 | 1,071.1 | 0 |
| control@1337 | 505.5 | 57.5% | 665.5 | 194.7 | 1,073.8 | 0 |
| control@2718 | 497.9 | 56.6% | 647.1 | 168.3 | 1,047.3 | 0 |
| control@7 | 501.9 | 57.1% | 649.3 | 191.5 | 1,070.5 | 0 |
| control@13 | 498.6 | 56.7% | 641.1 | 200.0 | 1,079.1 | 0 |
| control@101 | 495.7 | 56.4% | 638.8 | 180.0 | 1,059.0 | 0 |
| control@271 | 507.0 | 57.7% | 649.1 | 183.0 | 1,062.0 | 0 |
| control@314 | 506.7 | 57.6% | 657.1 | 207.1 | 1,086.1 | 0 |
| control@577 | 491.1 | 55.9% | 643.6 | 197.0 | 1,076.1 | 0 |
| control@863 | 502.1 | 57.1% | 638.1 | 189.3 | 1,068.4 | 0 |
| control@1024 | 510.6 | 58.1% | 649.9 | 196.2 | 1,075.2 | 0 |
| control@1729 | 502.4 | 57.2% | 662.0 | 190.7 | 1,069.7 | 0 |

ABL-337 night screen: not applicable to wind_onshore.

## BE / wind_onshore — catboost, source `energy_renewable`

n_train 20,548 · n_holdout 720 · incumbent version 20260201_222020

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 538.1 | 79.8% | 702.2 | -97.4 | 576.6 | 0 |
| control@42 | 261.3 | 38.8% | 332.0 | 173.4 | 847.4 | 0 |
| control@1337 | 264.8 | 39.3% | 338.5 | 182.6 | 856.6 | 0 |
| control@2718 | 257.3 | 38.2% | 331.3 | 163.6 | 837.6 | 0 |
| control@7 | 256.8 | 38.1% | 329.3 | 165.2 | 839.2 | 0 |
| control@13 | 257.2 | 38.2% | 326.4 | 174.1 | 848.1 | 0 |
| control@101 | 255.0 | 37.8% | 327.0 | 170.3 | 844.3 | 0 |
| control@271 | 255.9 | 38.0% | 329.7 | 157.0 | 831.0 | 0 |
| control@314 | 260.1 | 38.6% | 333.9 | 175.4 | 849.4 | 0 |
| control@577 | 261.7 | 38.8% | 334.5 | 170.8 | 844.8 | 0 |
| control@863 | 259.1 | 38.4% | 333.6 | 170.5 | 844.5 | 0 |
| control@1024 | 251.2 | 37.3% | 325.4 | 168.2 | 842.2 | 0 |
| control@1729 | 253.6 | 37.6% | 325.3 | 152.8 | 826.8 | 0 |

ABL-337 night screen: not applicable to wind_onshore.

## DE / wind_onshore — catboost, source `energy_renewable`

n_train 5,813 · n_holdout 720 · incumbent version 20260201_222000

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 6,528.1 | 80.1% | 8,542.7 | -63.1 | 8,086.5 | 0 |
| control@42 | 2,698.3 | 33.1% | 3,247.3 | 509.9 | 8,659.5 | 0 |
| control@1337 | 2,641.4 | 32.4% | 3,220.5 | 462.4 | 8,612.0 | 0 |
| control@2718 | 2,666.1 | 32.7% | 3,212.1 | 591.1 | 8,740.7 | 0 |
| control@7 | 2,683.9 | 32.9% | 3,269.9 | 662.7 | 8,812.3 | 0 |
| control@13 | 2,626.0 | 32.2% | 3,198.2 | 522.9 | 8,672.5 | 0 |
| control@101 | 2,585.8 | 31.7% | 3,107.3 | 492.1 | 8,641.7 | 0 |
| control@271 | 2,697.9 | 33.1% | 3,252.9 | 806.3 | 8,955.9 | 0 |
| control@314 | 2,653.5 | 32.6% | 3,214.3 | 708.1 | 8,857.6 | 0 |
| control@577 | 2,664.2 | 32.7% | 3,196.8 | 752.3 | 8,901.9 | 0 |
| control@863 | 2,597.0 | 31.9% | 3,133.0 | 545.4 | 8,695.0 | 0 |
| control@1024 | 2,638.3 | 32.4% | 3,214.4 | 547.6 | 8,697.2 | 0 |
| control@1729 | 2,674.9 | 32.8% | 3,227.1 | 472.9 | 8,622.5 | 0 |

ABL-337 night screen: not applicable to wind_onshore.

## FR / wind_onshore — catboost, source `energy_renewable`

n_train 29,436 · n_holdout 720 · incumbent version 20260201_222010

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 3,126.2 | 83.5% | 3,957.4 | -131.3 | 3,611.6 | 0 |
| control@42 | 1,316.1 | 35.2% | 1,642.5 | 567.1 | 4,310.1 | 0 |
| control@1337 | 1,289.7 | 34.5% | 1,606.4 | 558.2 | 4,301.2 | 0 |
| control@2718 | 1,364.7 | 36.5% | 1,701.6 | 682.4 | 4,425.4 | 0 |
| control@7 | 1,348.4 | 36.0% | 1,670.5 | 650.2 | 4,393.2 | 0 |
| control@13 | 1,317.9 | 35.2% | 1,643.6 | 533.8 | 4,276.7 | 0 |
| control@101 | 1,334.0 | 35.6% | 1,667.2 | 688.2 | 4,431.2 | 0 |
| control@271 | 1,342.1 | 35.9% | 1,677.2 | 661.9 | 4,404.8 | 0 |
| control@314 | 1,326.5 | 35.4% | 1,646.3 | 639.2 | 4,382.2 | 0 |
| control@577 | 1,297.8 | 34.7% | 1,623.8 | 573.9 | 4,316.8 | 0 |
| control@863 | 1,359.0 | 36.3% | 1,684.5 | 642.0 | 4,384.9 | 0 |
| control@1024 | 1,364.6 | 36.5% | 1,703.0 | 626.0 | 4,369.0 | 0 |
| control@1729 | 1,329.3 | 35.5% | 1,652.2 | 656.2 | 4,399.2 | 0 |

ABL-337 night screen: not applicable to wind_onshore.
