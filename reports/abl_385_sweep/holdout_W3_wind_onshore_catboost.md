# Held-out A/B — wind_onshore (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T13:40:19 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-04-14 .. 2026-05-13**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `wind_onshore` has no band structure, so one all-hours row is the result.

## AT / wind_onshore — catboost, source `energy_renewable`

n_train 3,158 · n_holdout 720 · incumbent version 20260112_165238

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,026.3 | 98.0% | 1,289.8 | -5.0 | 1,042.5 | 0 |
| control@42 | 539.3 | 51.5% | 724.3 | 9.8 | 1,057.2 | 0 |
| control@1337 | 544.1 | 52.0% | 714.8 | 8.7 | 1,056.1 | 0 |
| control@2718 | 543.1 | 51.9% | 728.3 | 29.1 | 1,076.5 | 0 |
| control@7 | 558.6 | 53.3% | 743.4 | 31.7 | 1,079.1 | 0 |
| control@13 | 539.8 | 51.5% | 713.4 | 8.7 | 1,056.2 | 0 |
| control@101 | 542.4 | 51.8% | 727.2 | 23.7 | 1,071.1 | 0 |
| control@271 | 553.5 | 52.8% | 732.0 | 7.3 | 1,054.7 | 0 |
| control@314 | 546.1 | 52.1% | 724.6 | 21.6 | 1,069.0 | 0 |
| control@577 | 552.9 | 52.8% | 730.6 | 36.9 | 1,084.3 | 0 |
| control@863 | 536.7 | 51.2% | 723.2 | 25.9 | 1,073.3 | 0 |
| control@1024 | 545.3 | 52.1% | 712.5 | 54.0 | 1,101.5 | 0 |
| control@1729 | 553.7 | 52.9% | 726.8 | -6.1 | 1,041.3 | 0 |

ABL-337 night screen: not applicable to wind_onshore.

## BE / wind_onshore — catboost, source `energy_renewable`

n_train 19,828 · n_holdout 720 · incumbent version 20260201_222020

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 520.6 | 90.3% | 693.4 | -1.3 | 575.3 | 0 |
| control@42 | 235.4 | 40.8% | 295.4 | 148.5 | 725.0 | 0 |
| control@1337 | 238.3 | 41.3% | 293.0 | 152.3 | 728.8 | 0 |
| control@2718 | 236.8 | 41.1% | 297.1 | 147.8 | 724.4 | 0 |
| control@7 | 249.2 | 43.2% | 308.1 | 179.8 | 756.3 | 0 |
| control@13 | 244.6 | 42.4% | 305.0 | 160.4 | 737.0 | 0 |
| control@101 | 231.9 | 40.2% | 290.5 | 150.3 | 726.8 | 0 |
| control@271 | 239.4 | 41.5% | 300.2 | 148.7 | 725.3 | 0 |
| control@314 | 245.6 | 42.6% | 306.7 | 162.9 | 739.5 | 0 |
| control@577 | 233.1 | 40.4% | 291.3 | 153.3 | 729.9 | 0 |
| control@863 | 228.5 | 39.6% | 286.2 | 139.7 | 716.2 | 0 |
| control@1024 | 248.1 | 43.0% | 308.0 | 169.7 | 746.3 | 0 |
| control@1729 | 249.0 | 43.2% | 312.1 | 180.2 | 756.8 | 0 |

ABL-337 night screen: not applicable to wind_onshore.

## DE / wind_onshore — catboost, source `energy_renewable`

n_train 5,093 · n_holdout 720 · incumbent version 20260201_222000

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 8,106.4 | 85.4% | 9,962.9 | -53.9 | 9,439.8 | 0 |
| control@42 | 3,087.5 | 32.5% | 4,161.2 | 899.3 | 10,393.0 | 0 |
| control@1337 | 2,994.0 | 31.5% | 3,983.2 | 707.7 | 10,201.4 | 0 |
| control@2718 | 2,917.8 | 30.7% | 3,917.9 | 623.6 | 10,117.2 | 0 |
| control@7 | 2,957.1 | 31.1% | 3,924.7 | 547.5 | 10,041.1 | 0 |
| control@13 | 2,867.3 | 30.2% | 3,856.0 | 533.3 | 10,027.0 | 0 |
| control@101 | 3,015.8 | 31.8% | 4,031.2 | 591.4 | 10,085.0 | 0 |
| control@271 | 2,912.9 | 30.7% | 3,891.8 | 728.8 | 10,222.5 | 0 |
| control@314 | 2,893.9 | 30.5% | 3,903.6 | 412.5 | 9,906.2 | 0 |
| control@577 | 2,972.8 | 31.3% | 4,031.1 | 650.8 | 10,144.5 | 0 |
| control@863 | 2,940.4 | 31.0% | 3,954.5 | 550.3 | 10,043.9 | 0 |
| control@1024 | 2,843.0 | 29.9% | 3,833.9 | 367.6 | 9,861.3 | 0 |
| control@1729 | 2,980.7 | 31.4% | 4,023.8 | 608.7 | 10,102.4 | 0 |

ABL-337 night screen: not applicable to wind_onshore.

## FR / wind_onshore — catboost, source `energy_renewable`

n_train 28,716 · n_holdout 720 · incumbent version 20260201_222010

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 2,627.8 | 63.8% | 3,538.0 | -134.2 | 3,982.4 | 0 |
| control@42 | 1,485.7 | 36.1% | 1,964.5 | 250.1 | 4,366.7 | 0 |
| control@1337 | 1,481.2 | 36.0% | 1,940.6 | 332.1 | 4,448.7 | 0 |
| control@2718 | 1,490.5 | 36.2% | 1,944.3 | 260.3 | 4,376.9 | 0 |
| control@7 | 1,468.0 | 35.7% | 1,937.1 | 267.9 | 4,384.5 | 0 |
| control@13 | 1,475.9 | 35.9% | 1,960.9 | 237.8 | 4,354.4 | 0 |
| control@101 | 1,470.2 | 35.7% | 1,935.8 | 215.4 | 4,332.0 | 0 |
| control@271 | 1,697.3 | 41.2% | 2,186.8 | 494.4 | 4,611.0 | 0 |
| control@314 | 1,444.1 | 35.1% | 1,894.7 | 256.0 | 4,372.6 | 0 |
| control@577 | 1,442.4 | 35.0% | 1,899.7 | 232.4 | 4,349.0 | 0 |
| control@863 | 1,533.7 | 37.3% | 2,018.4 | 255.5 | 4,372.1 | 0 |
| control@1024 | 1,466.3 | 35.6% | 1,927.4 | 308.8 | 4,425.4 | 0 |
| control@1729 | 1,435.5 | 34.9% | 1,899.0 | 234.8 | 4,351.4 | 0 |

ABL-337 night screen: not applicable to wind_onshore.
