# Held-out A/B — wind_onshore (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T13:20:48 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-02-13 .. 2026-03-14**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `wind_onshore` has no band structure, so one all-hours row is the result.

## AT / wind_onshore — xgboost, source `energy_renewable`

n_train 1,746 · n_holdout 692 · incumbent version 20260112_165238

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,004.7 | 98.4% | 1,315.4 | -115.5 | 905.6 | 0 |
| control@42 | 532.2 | 52.1% | 760.5 | -169.8 | 851.3 | 0 |
| control@1337 | 530.8 | 52.0% | 743.7 | -130.0 | 891.2 | 0 |
| control@2718 | 536.6 | 52.5% | 740.1 | -127.4 | 893.7 | 0 |
| control@7 | 546.1 | 53.5% | 752.5 | -154.7 | 866.5 | 0 |
| control@13 | 534.8 | 52.4% | 754.5 | -150.3 | 870.8 | 0 |
| control@101 | 533.5 | 52.2% | 760.0 | -181.9 | 839.2 | 0 |
| control@271 | 547.0 | 53.6% | 771.3 | -195.3 | 825.8 | 0 |
| control@314 | 526.3 | 51.5% | 705.2 | -57.4 | 963.8 | 0 |
| control@577 | 530.9 | 52.0% | 737.6 | -138.5 | 882.6 | 0 |
| control@863 | 533.8 | 52.3% | 750.5 | -156.0 | 865.1 | 0 |
| control@1024 | 526.1 | 51.5% | 732.9 | -136.8 | 884.4 | 0 |
| control@1729 | 535.4 | 52.4% | 747.5 | -151.9 | 869.2 | 0 |

ABL-337 night screen: not applicable to wind_onshore.

## BE / wind_onshore — xgboost, source `energy_renewable`

n_train 18,416 · n_holdout 692 · incumbent version 20260201_222020

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 988.8 | 86.7% | 1,211.5 | -12.1 | 1,128.2 | 0 |
| control@42 | 411.3 | 36.1% | 513.7 | -58.6 | 1,081.7 | 0 |
| control@1337 | 404.8 | 35.5% | 503.8 | -60.0 | 1,080.3 | 0 |
| control@2718 | 394.6 | 34.6% | 498.6 | -39.1 | 1,101.1 | 0 |
| control@7 | 407.2 | 35.7% | 510.7 | -63.0 | 1,077.3 | 0 |
| control@13 | 426.5 | 37.4% | 541.2 | -116.5 | 1,023.8 | 0 |
| control@101 | 407.5 | 35.7% | 511.5 | -71.2 | 1,069.1 | 0 |
| control@271 | 391.8 | 34.4% | 484.4 | -25.4 | 1,114.9 | 0 |
| control@314 | 423.6 | 37.2% | 527.2 | -66.2 | 1,074.1 | 0 |
| control@577 | 413.0 | 36.2% | 516.8 | -61.1 | 1,079.2 | 0 |
| control@863 | 401.0 | 35.2% | 502.2 | -50.3 | 1,089.9 | 0 |
| control@1024 | 385.9 | 33.8% | 481.4 | -19.1 | 1,121.2 | 0 |
| control@1729 | 400.6 | 35.1% | 502.7 | -56.7 | 1,083.6 | 0 |

ABL-337 night screen: not applicable to wind_onshore.

## DE / wind_onshore — xgboost, source `energy_renewable`

n_train 3,680 · n_holdout 693 · incumbent version 20260201_222000

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 10,945.1 | 76.2% | 14,130.3 | -840.7 | 13,528.0 | 0 |
| control@42 | 4,341.2 | 30.2% | 5,669.3 | 1,774.0 | 16,142.8 | 0 |
| control@1337 | 4,359.8 | 30.3% | 5,756.9 | 2,249.2 | 16,617.9 | 0 |
| control@2718 | 4,420.5 | 30.8% | 5,811.9 | 2,075.0 | 16,443.8 | 0 |
| control@7 | 4,221.6 | 29.4% | 5,535.9 | 1,955.8 | 16,324.5 | 0 |
| control@13 | 4,352.0 | 30.3% | 5,729.1 | 2,203.6 | 16,572.3 | 0 |
| control@101 | 4,344.8 | 30.2% | 5,763.4 | 2,145.9 | 16,514.6 | 0 |
| control@271 | 4,270.5 | 29.7% | 5,604.0 | 2,040.9 | 16,409.6 | 0 |
| control@314 | 4,280.2 | 29.8% | 5,606.6 | 2,138.2 | 16,506.9 | 0 |
| control@577 | 4,317.2 | 30.0% | 5,667.6 | 2,097.8 | 16,466.5 | 0 |
| control@863 | 4,282.1 | 29.8% | 5,595.2 | 1,871.4 | 16,240.2 | 0 |
| control@1024 | 4,395.9 | 30.6% | 5,844.1 | 2,107.0 | 16,475.8 | 0 |
| control@1729 | 4,303.8 | 30.0% | 5,602.5 | 2,079.7 | 16,448.4 | 0 |

ABL-337 night screen: not applicable to wind_onshore.

## FR / wind_onshore — xgboost, source `energy_renewable`

n_train 27,304 · n_holdout 692 · incumbent version 20260201_222010

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 4,395.1 | 69.5% | 5,403.3 | 740.5 | 7,066.9 | 0 |
| control@42 | 1,925.9 | 30.4% | 2,495.0 | 356.4 | 6,682.7 | 0 |
| control@1337 | 1,953.9 | 30.9% | 2,496.4 | 312.1 | 6,638.5 | 0 |
| control@2718 | 1,894.3 | 29.9% | 2,425.5 | 445.6 | 6,772.0 | 0 |
| control@7 | 1,943.6 | 30.7% | 2,512.5 | 467.2 | 6,793.6 | 0 |
| control@13 | 1,932.3 | 30.5% | 2,489.9 | 398.3 | 6,724.7 | 0 |
| control@101 | 1,918.8 | 30.3% | 2,508.0 | 456.4 | 6,782.8 | 0 |
| control@271 | 1,958.6 | 31.0% | 2,557.8 | 435.4 | 6,761.8 | 0 |
| control@314 | 1,933.3 | 30.6% | 2,501.4 | 282.7 | 6,609.1 | 0 |
| control@577 | 1,933.9 | 30.6% | 2,492.9 | 420.7 | 6,747.1 | 0 |
| control@863 | 1,927.5 | 30.5% | 2,487.4 | 403.7 | 6,730.1 | 0 |
| control@1024 | 1,922.9 | 30.4% | 2,465.0 | 342.2 | 6,668.5 | 0 |
| control@1729 | 1,925.6 | 30.4% | 2,502.2 | 374.4 | 6,700.8 | 0 |

ABL-337 night screen: not applicable to wind_onshore.
