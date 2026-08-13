# Held-out A/B — wind_onshore (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T13:51:25 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-05-14 .. 2026-06-12**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `wind_onshore` has no band structure, so one all-hours row is the result.

## AT / wind_onshore — xgboost, source `energy_renewable`

n_train 3,878 · n_holdout 720 · incumbent version 20260112_165238

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,043.3 | 118.7% | 1,354.0 | 51.0 | 930.0 | 0 |
| control@42 | 545.2 | 62.0% | 706.3 | 274.4 | 1,153.5 | 0 |
| control@1337 | 579.0 | 65.9% | 719.9 | 269.2 | 1,148.3 | 0 |
| control@2718 | 572.9 | 65.2% | 732.3 | 278.0 | 1,157.1 | 0 |
| control@7 | 577.3 | 65.7% | 733.7 | 295.1 | 1,174.1 | 0 |
| control@13 | 577.5 | 65.7% | 722.9 | 258.8 | 1,137.8 | 0 |
| control@101 | 565.1 | 64.3% | 710.5 | 259.5 | 1,138.6 | 0 |
| control@271 | 596.9 | 67.9% | 735.3 | 263.7 | 1,142.8 | 0 |
| control@314 | 569.5 | 64.8% | 722.7 | 275.8 | 1,154.9 | 0 |
| control@577 | 574.8 | 65.4% | 736.1 | 300.1 | 1,179.1 | 0 |
| control@863 | 575.9 | 65.5% | 730.9 | 283.7 | 1,162.8 | 0 |
| control@1024 | 557.2 | 63.4% | 717.9 | 261.8 | 1,140.8 | 0 |
| control@1729 | 554.4 | 63.1% | 706.4 | 251.3 | 1,130.3 | 0 |

ABL-337 night screen: not applicable to wind_onshore.

## BE / wind_onshore — xgboost, source `energy_renewable`

n_train 20,548 · n_holdout 720 · incumbent version 20260201_222020

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 538.1 | 79.8% | 702.2 | -97.4 | 576.6 | 0 |
| control@42 | 232.2 | 34.5% | 299.1 | 127.3 | 801.3 | 0 |
| control@1337 | 235.6 | 35.0% | 302.7 | 135.6 | 809.6 | 0 |
| control@2718 | 240.5 | 35.7% | 310.5 | 145.0 | 819.0 | 0 |
| control@7 | 236.6 | 35.1% | 301.5 | 128.3 | 802.3 | 0 |
| control@13 | 229.4 | 34.0% | 297.2 | 122.2 | 796.2 | 0 |
| control@101 | 239.6 | 35.5% | 307.9 | 146.6 | 820.6 | 0 |
| control@271 | 232.3 | 34.5% | 298.3 | 127.1 | 801.1 | 0 |
| control@314 | 240.8 | 35.7% | 310.8 | 146.4 | 820.4 | 0 |
| control@577 | 232.5 | 34.5% | 297.1 | 111.0 | 785.0 | 0 |
| control@863 | 234.7 | 34.8% | 300.3 | 133.3 | 807.3 | 0 |
| control@1024 | 242.8 | 36.0% | 312.9 | 141.4 | 815.4 | 0 |
| control@1729 | 230.5 | 34.2% | 293.0 | 95.3 | 769.3 | 0 |

ABL-337 night screen: not applicable to wind_onshore.

## DE / wind_onshore — xgboost, source `energy_renewable`

n_train 5,813 · n_holdout 720 · incumbent version 20260201_222000

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 6,528.1 | 80.1% | 8,542.7 | -63.1 | 8,086.5 | 0 |
| control@42 | 2,566.0 | 31.5% | 3,122.7 | 987.6 | 9,137.2 | 0 |
| control@1337 | 2,579.2 | 31.6% | 3,131.2 | 979.6 | 9,129.2 | 0 |
| control@2718 | 2,570.0 | 31.5% | 3,129.3 | 898.7 | 9,048.3 | 0 |
| control@7 | 2,496.9 | 30.6% | 3,014.8 | 916.6 | 9,066.2 | 0 |
| control@13 | 2,469.5 | 30.3% | 3,001.2 | 931.6 | 9,081.2 | 0 |
| control@101 | 2,514.7 | 30.9% | 3,038.5 | 860.7 | 9,010.3 | 0 |
| control@271 | 2,495.7 | 30.6% | 3,019.1 | 833.2 | 8,982.8 | 0 |
| control@314 | 2,451.7 | 30.1% | 2,969.7 | 751.1 | 8,900.7 | 0 |
| control@577 | 2,627.0 | 32.2% | 3,141.0 | 1,145.2 | 9,294.8 | 0 |
| control@863 | 2,477.9 | 30.4% | 3,029.7 | 810.5 | 8,960.1 | 0 |
| control@1024 | 2,524.1 | 31.0% | 3,100.3 | 757.1 | 8,906.7 | 0 |
| control@1729 | 2,560.6 | 31.4% | 3,118.8 | 897.2 | 9,046.7 | 0 |

ABL-337 night screen: not applicable to wind_onshore.

## FR / wind_onshore — xgboost, source `energy_renewable`

n_train 29,436 · n_holdout 720 · incumbent version 20260201_222010

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 3,126.2 | 83.5% | 3,957.4 | -131.3 | 3,611.6 | 0 |
| control@42 | 1,430.7 | 38.2% | 1,814.7 | 860.4 | 4,603.4 | 0 |
| control@1337 | 1,427.3 | 38.1% | 1,816.9 | 882.5 | 4,625.5 | 0 |
| control@2718 | 1,418.2 | 37.9% | 1,794.9 | 814.4 | 4,557.3 | 0 |
| control@7 | 1,474.0 | 39.4% | 1,834.9 | 887.6 | 4,630.6 | 0 |
| control@13 | 1,401.4 | 37.4% | 1,776.6 | 841.3 | 4,584.2 | 0 |
| control@101 | 1,433.7 | 38.3% | 1,794.5 | 832.2 | 4,575.2 | 0 |
| control@271 | 1,457.9 | 39.0% | 1,843.8 | 869.6 | 4,612.6 | 0 |
| control@314 | 1,450.4 | 38.8% | 1,828.3 | 882.1 | 4,625.1 | 0 |
| control@577 | 1,446.6 | 38.6% | 1,826.6 | 899.2 | 4,642.2 | 0 |
| control@863 | 1,464.2 | 39.1% | 1,844.4 | 858.1 | 4,601.1 | 0 |
| control@1024 | 1,452.6 | 38.8% | 1,821.1 | 909.8 | 4,652.7 | 0 |
| control@1729 | 1,395.4 | 37.3% | 1,753.1 | 810.2 | 4,553.1 | 0 |

ABL-337 night screen: not applicable to wind_onshore.
