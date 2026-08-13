# Held-out A/B — wind_onshore (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T14:01:39 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-06-13 .. 2026-07-12**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `wind_onshore` has no band structure, so one all-hours row is the result.

## AT / wind_onshore — xgboost, source `energy_renewable`

n_train 4,598 · n_holdout 720 · incumbent version 20260112_165238

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 816.9 | 102.8% | 1,119.1 | -51.7 | 742.9 | 0 |
| control@42 | 471.9 | 59.4% | 619.5 | 243.9 | 1,038.5 | 0 |
| control@1337 | 479.7 | 60.4% | 621.4 | 243.6 | 1,038.2 | 0 |
| control@2718 | 496.4 | 62.5% | 628.4 | 229.8 | 1,024.4 | 0 |
| control@7 | 491.9 | 61.9% | 633.7 | 250.3 | 1,044.9 | 0 |
| control@13 | 479.5 | 60.3% | 621.8 | 222.7 | 1,017.3 | 0 |
| control@101 | 493.2 | 62.1% | 627.2 | 230.1 | 1,024.6 | 0 |
| control@271 | 511.2 | 64.3% | 634.8 | 264.0 | 1,058.6 | 0 |
| control@314 | 523.8 | 65.9% | 645.4 | 275.1 | 1,069.7 | 0 |
| control@577 | 490.8 | 61.8% | 624.5 | 225.3 | 1,019.9 | 0 |
| control@863 | 511.9 | 64.4% | 642.0 | 257.3 | 1,051.9 | 0 |
| control@1024 | 468.3 | 58.9% | 606.8 | 234.1 | 1,028.7 | 0 |
| control@1729 | 484.6 | 61.0% | 611.4 | 236.3 | 1,030.9 | 0 |

ABL-337 night screen: not applicable to wind_onshore.

## BE / wind_onshore — xgboost, source `energy_renewable`

n_train 21,268 · n_holdout 720 · incumbent version 20260201_222020

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 434.9 | 107.3% | 588.8 | 99.6 | 504.9 | 0 |
| control@42 | 214.5 | 52.9% | 267.3 | 136.9 | 542.1 | 0 |
| control@1337 | 210.4 | 51.9% | 265.9 | 123.8 | 529.0 | 0 |
| control@2718 | 211.8 | 52.3% | 269.0 | 124.1 | 529.4 | 0 |
| control@7 | 206.1 | 50.9% | 262.3 | 112.6 | 517.9 | 0 |
| control@13 | 208.3 | 51.4% | 263.3 | 122.4 | 527.6 | 0 |
| control@101 | 213.4 | 52.7% | 269.2 | 130.0 | 535.2 | 0 |
| control@271 | 199.8 | 49.3% | 255.8 | 119.2 | 524.4 | 0 |
| control@314 | 207.0 | 51.1% | 261.6 | 118.2 | 523.4 | 0 |
| control@577 | 205.3 | 50.7% | 261.5 | 111.0 | 516.2 | 0 |
| control@863 | 208.5 | 51.4% | 260.2 | 117.9 | 523.2 | 0 |
| control@1024 | 207.7 | 51.2% | 264.0 | 120.1 | 525.4 | 0 |
| control@1729 | 206.2 | 50.9% | 260.9 | 118.7 | 523.9 | 0 |

ABL-337 night screen: not applicable to wind_onshore.

## DE / wind_onshore — xgboost, source `energy_renewable`

n_train 6,533 · n_holdout 720 · incumbent version 20260201_222000

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 9,947.6 | 102.7% | 12,535.5 | -445.5 | 9,239.7 | 0 |
| control@42 | 2,941.5 | 30.4% | 3,854.6 | 1,327.7 | 11,012.8 | 0 |
| control@1337 | 2,898.9 | 29.9% | 3,768.4 | 1,250.9 | 10,936.1 | 0 |
| control@2718 | 3,077.2 | 31.8% | 4,037.5 | 1,372.1 | 11,057.3 | 0 |
| control@7 | 2,967.3 | 30.6% | 3,884.5 | 1,337.0 | 11,022.2 | 0 |
| control@13 | 2,914.0 | 30.1% | 3,804.2 | 1,277.5 | 10,962.7 | 0 |
| control@101 | 2,939.9 | 30.4% | 3,799.4 | 1,430.1 | 11,115.3 | 0 |
| control@271 | 2,939.5 | 30.4% | 3,826.9 | 1,294.0 | 10,979.2 | 0 |
| control@314 | 3,000.5 | 31.0% | 3,881.4 | 1,373.8 | 11,059.0 | 0 |
| control@577 | 3,043.9 | 31.4% | 4,001.2 | 1,468.1 | 11,153.3 | 0 |
| control@863 | 2,912.2 | 30.1% | 3,870.8 | 1,385.3 | 11,070.4 | 0 |
| control@1024 | 2,978.6 | 30.8% | 3,898.8 | 1,212.9 | 10,898.1 | 0 |
| control@1729 | 3,019.5 | 31.2% | 3,921.5 | 1,370.0 | 11,055.2 | 0 |

ABL-337 night screen: not applicable to wind_onshore.

## FR / wind_onshore — xgboost, source `energy_renewable`

n_train 30,156 · n_holdout 720 · incumbent version 20260201_222010

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,626.1 | 55.8% | 2,134.5 | 216.1 | 3,131.9 | 0 |
| control@42 | 1,040.1 | 35.7% | 1,389.0 | 438.6 | 3,354.3 | 0 |
| control@1337 | 1,019.8 | 35.0% | 1,368.5 | 387.0 | 3,302.8 | 0 |
| control@2718 | 1,025.4 | 35.2% | 1,343.8 | 456.4 | 3,372.2 | 0 |
| control@7 | 1,033.2 | 35.4% | 1,376.9 | 448.2 | 3,364.0 | 0 |
| control@13 | 1,019.5 | 35.0% | 1,350.2 | 343.4 | 3,259.2 | 0 |
| control@101 | 1,048.4 | 36.0% | 1,390.1 | 439.6 | 3,355.4 | 0 |
| control@271 | 1,001.2 | 34.3% | 1,317.3 | 344.8 | 3,260.6 | 0 |
| control@314 | 1,041.0 | 35.7% | 1,364.7 | 425.5 | 3,341.3 | 0 |
| control@577 | 1,016.8 | 34.9% | 1,354.1 | 380.4 | 3,296.2 | 0 |
| control@863 | 1,025.9 | 35.2% | 1,370.2 | 394.5 | 3,310.3 | 0 |
| control@1024 | 1,054.3 | 36.2% | 1,373.0 | 435.6 | 3,351.4 | 0 |
| control@1729 | 1,059.9 | 36.4% | 1,399.9 | 475.6 | 3,391.4 | 0 |

ABL-337 night screen: not applicable to wind_onshore.
