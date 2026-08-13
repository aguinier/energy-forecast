# Held-out A/B — wind_onshore (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T14:00:17 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-06-13 .. 2026-07-12**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `wind_onshore` has no band structure, so one all-hours row is the result.

## AT / wind_onshore — catboost, source `energy_renewable`

n_train 4,598 · n_holdout 720 · incumbent version 20260112_165238

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 816.9 | 102.8% | 1,119.1 | -51.7 | 742.9 | 0 |
| control@42 | 441.9 | 55.6% | 594.3 | 134.8 | 929.4 | 0 |
| control@1337 | 455.0 | 57.3% | 611.4 | 154.4 | 949.0 | 0 |
| control@2718 | 438.3 | 55.2% | 589.5 | 129.9 | 924.5 | 0 |
| control@7 | 458.1 | 57.6% | 622.0 | 157.5 | 952.0 | 0 |
| control@13 | 463.1 | 58.3% | 616.8 | 164.9 | 959.5 | 0 |
| control@101 | 438.6 | 55.2% | 581.8 | 144.9 | 939.5 | 0 |
| control@271 | 451.4 | 56.8% | 606.1 | 156.0 | 950.6 | 0 |
| control@314 | 441.5 | 55.6% | 592.1 | 144.9 | 939.5 | 0 |
| control@577 | 448.2 | 56.4% | 605.0 | 154.3 | 948.9 | 0 |
| control@863 | 456.6 | 57.5% | 610.2 | 160.5 | 955.1 | 0 |
| control@1024 | 456.0 | 57.4% | 609.6 | 140.7 | 935.3 | 0 |
| control@1729 | 468.8 | 59.0% | 632.7 | 151.9 | 946.5 | 0 |

ABL-337 night screen: not applicable to wind_onshore.

## BE / wind_onshore — catboost, source `energy_renewable`

n_train 21,268 · n_holdout 720 · incumbent version 20260201_222020

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 434.9 | 107.3% | 588.8 | 99.6 | 504.9 | 0 |
| control@42 | 221.3 | 54.6% | 285.7 | 140.0 | 545.3 | 0 |
| control@1337 | 213.1 | 52.6% | 276.6 | 134.0 | 539.3 | 0 |
| control@2718 | 217.1 | 53.6% | 282.1 | 135.8 | 541.1 | 0 |
| control@7 | 214.5 | 52.9% | 281.3 | 125.9 | 531.2 | 0 |
| control@13 | 217.3 | 53.6% | 285.6 | 132.0 | 537.3 | 0 |
| control@101 | 215.6 | 53.2% | 280.4 | 130.8 | 536.0 | 0 |
| control@271 | 209.0 | 51.6% | 271.0 | 123.3 | 528.6 | 0 |
| control@314 | 213.3 | 52.6% | 277.8 | 132.4 | 537.6 | 0 |
| control@577 | 220.5 | 54.4% | 283.9 | 137.5 | 542.8 | 1 |
| control@863 | 219.3 | 54.1% | 281.9 | 137.1 | 542.3 | 0 |
| control@1024 | 222.7 | 55.0% | 290.4 | 144.9 | 550.2 | 0 |
| control@1729 | 213.6 | 52.7% | 277.0 | 131.5 | 536.7 | 0 |

ABL-337 night screen: not applicable to wind_onshore.

## DE / wind_onshore — catboost, source `energy_renewable`

n_train 6,533 · n_holdout 720 · incumbent version 20260201_222000

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 9,947.6 | 102.7% | 12,535.5 | -445.5 | 9,239.7 | 0 |
| control@42 | 2,736.5 | 28.3% | 3,570.4 | 712.7 | 10,397.9 | 0 |
| control@1337 | 2,635.2 | 27.2% | 3,530.1 | 607.2 | 10,292.4 | 0 |
| control@2718 | 2,719.0 | 28.1% | 3,545.9 | 690.9 | 10,376.1 | 0 |
| control@7 | 2,676.1 | 27.6% | 3,562.0 | 810.4 | 10,495.6 | 0 |
| control@13 | 2,630.3 | 27.2% | 3,518.1 | 761.3 | 10,446.4 | 0 |
| control@101 | 2,661.5 | 27.5% | 3,490.3 | 793.4 | 10,478.6 | 0 |
| control@271 | 2,676.7 | 27.6% | 3,497.2 | 652.9 | 10,338.1 | 0 |
| control@314 | 2,718.2 | 28.1% | 3,581.1 | 898.1 | 10,583.2 | 0 |
| control@577 | 2,649.9 | 27.4% | 3,530.1 | 739.0 | 10,424.2 | 0 |
| control@863 | 2,660.2 | 27.5% | 3,526.3 | 585.2 | 10,270.4 | 0 |
| control@1024 | 2,704.2 | 27.9% | 3,563.6 | 840.7 | 10,525.9 | 0 |
| control@1729 | 2,687.5 | 27.7% | 3,563.9 | 512.2 | 10,197.3 | 0 |

ABL-337 night screen: not applicable to wind_onshore.

## FR / wind_onshore — catboost, source `energy_renewable`

n_train 30,156 · n_holdout 720 · incumbent version 20260201_222010

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,626.1 | 55.8% | 2,134.5 | 216.1 | 3,131.9 | 0 |
| control@42 | 948.3 | 32.5% | 1,241.3 | 223.7 | 3,139.5 | 0 |
| control@1337 | 1,007.1 | 34.5% | 1,283.0 | 266.9 | 3,182.7 | 0 |
| control@2718 | 965.0 | 33.1% | 1,247.6 | 258.7 | 3,174.5 | 0 |
| control@7 | 1,017.2 | 34.9% | 1,309.6 | 322.5 | 3,238.3 | 0 |
| control@13 | 983.9 | 33.7% | 1,269.3 | 267.4 | 3,183.2 | 0 |
| control@101 | 951.4 | 32.6% | 1,237.4 | 250.1 | 3,165.9 | 0 |
| control@271 | 952.8 | 32.7% | 1,254.4 | 250.1 | 3,165.9 | 0 |
| control@314 | 963.7 | 33.0% | 1,251.9 | 216.2 | 3,131.9 | 0 |
| control@577 | 981.4 | 33.7% | 1,275.8 | 299.7 | 3,215.5 | 0 |
| control@863 | 982.4 | 33.7% | 1,263.3 | 257.1 | 3,172.9 | 0 |
| control@1024 | 945.6 | 32.4% | 1,233.6 | 231.9 | 3,147.7 | 0 |
| control@1729 | 982.7 | 33.7% | 1,266.7 | 241.9 | 3,157.7 | 0 |

ABL-337 night screen: not applicable to wind_onshore.
