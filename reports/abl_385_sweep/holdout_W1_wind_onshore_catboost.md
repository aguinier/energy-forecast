# Held-out A/B — wind_onshore (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T13:19:46 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-02-13 .. 2026-03-14**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `wind_onshore` has no band structure, so one all-hours row is the result.

## AT / wind_onshore — catboost, source `energy_renewable`

n_train 1,746 · n_holdout 692 · incumbent version 20260112_165238

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,004.7 | 98.4% | 1,315.4 | -115.5 | 905.6 | 0 |
| control@42 | 537.8 | 52.7% | 698.0 | -146.3 | 874.8 | 0 |
| control@1337 | 561.4 | 55.0% | 719.6 | -110.2 | 910.9 | 0 |
| control@2718 | 550.4 | 53.9% | 711.4 | -103.3 | 917.8 | 0 |
| control@7 | 556.6 | 54.5% | 716.5 | -123.1 | 898.1 | 0 |
| control@13 | 547.7 | 53.6% | 695.1 | -74.6 | 946.5 | 0 |
| control@101 | 544.1 | 53.3% | 681.9 | -60.3 | 960.9 | 0 |
| control@271 | 547.4 | 53.6% | 699.5 | -92.8 | 928.4 | 0 |
| control@314 | 556.4 | 54.5% | 708.1 | -107.7 | 913.5 | 0 |
| control@577 | 537.8 | 52.7% | 695.5 | -114.2 | 907.0 | 0 |
| control@863 | 518.6 | 50.8% | 664.8 | -85.8 | 935.3 | 0 |
| control@1024 | 542.7 | 53.1% | 699.1 | -97.3 | 923.9 | 0 |
| control@1729 | 546.2 | 53.5% | 700.2 | -115.3 | 905.8 | 0 |

ABL-337 night screen: not applicable to wind_onshore.

## BE / wind_onshore — catboost, source `energy_renewable`

n_train 18,416 · n_holdout 692 · incumbent version 20260201_222020

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 988.8 | 86.7% | 1,211.5 | -12.1 | 1,128.2 | 0 |
| control@42 | 392.1 | 34.4% | 491.2 | 23.7 | 1,164.0 | 0 |
| control@1337 | 408.5 | 35.8% | 507.4 | 33.0 | 1,173.3 | 0 |
| control@2718 | 443.9 | 38.9% | 553.2 | -17.4 | 1,122.9 | 0 |
| control@7 | 371.0 | 32.5% | 460.9 | 42.9 | 1,183.2 | 0 |
| control@13 | 450.5 | 39.5% | 555.0 | 19.5 | 1,159.7 | 0 |
| control@101 | 438.4 | 38.4% | 542.1 | 14.7 | 1,155.0 | 0 |
| control@271 | 403.4 | 35.4% | 494.8 | 76.7 | 1,216.9 | 0 |
| control@314 | 389.8 | 34.2% | 479.2 | 64.1 | 1,204.4 | 0 |
| control@577 | 471.9 | 41.4% | 582.8 | -17.9 | 1,122.3 | 0 |
| control@863 | 390.4 | 34.2% | 478.3 | 52.2 | 1,192.4 | 0 |
| control@1024 | 372.6 | 32.7% | 459.2 | 85.3 | 1,225.6 | 0 |
| control@1729 | 464.1 | 40.7% | 569.0 | 18.3 | 1,158.5 | 0 |

ABL-337 night screen: not applicable to wind_onshore.

## DE / wind_onshore — catboost, source `energy_renewable`

n_train 3,680 · n_holdout 693 · incumbent version 20260201_222000

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 10,945.1 | 76.2% | 14,130.3 | -840.7 | 13,528.0 | 0 |
| control@42 | 4,484.3 | 31.2% | 5,611.9 | 1,848.5 | 16,217.2 | 0 |
| control@1337 | 4,485.2 | 31.2% | 5,456.2 | 2,146.6 | 16,515.4 | 0 |
| control@2718 | 4,362.6 | 30.4% | 5,380.4 | 1,460.6 | 15,829.3 | 0 |
| control@7 | 4,248.3 | 29.6% | 5,291.5 | 1,612.2 | 15,981.0 | 0 |
| control@13 | 4,633.9 | 32.2% | 5,686.3 | 2,448.9 | 16,817.6 | 0 |
| control@101 | 4,326.7 | 30.1% | 5,447.1 | 1,553.5 | 15,922.3 | 0 |
| control@271 | 4,217.3 | 29.4% | 5,293.1 | 1,780.4 | 16,149.2 | 0 |
| control@314 | 4,368.8 | 30.4% | 5,349.4 | 1,303.1 | 15,671.8 | 0 |
| control@577 | 4,384.5 | 30.5% | 5,381.2 | 1,557.9 | 15,926.7 | 0 |
| control@863 | 4,535.7 | 31.6% | 5,626.5 | 1,893.2 | 16,261.9 | 0 |
| control@1024 | 4,430.4 | 30.8% | 5,424.9 | 1,603.8 | 15,972.5 | 0 |
| control@1729 | 4,356.0 | 30.3% | 5,419.8 | 1,540.8 | 15,909.6 | 0 |

ABL-337 night screen: not applicable to wind_onshore.

## FR / wind_onshore — catboost, source `energy_renewable`

n_train 27,304 · n_holdout 692 · incumbent version 20260201_222010

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 4,395.1 | 69.5% | 5,403.3 | 740.5 | 7,066.9 | 0 |
| control@42 | 2,080.0 | 32.9% | 2,756.1 | 485.0 | 6,811.4 | 0 |
| control@1337 | 2,049.4 | 32.4% | 2,738.9 | 522.5 | 6,848.8 | 0 |
| control@2718 | 2,069.4 | 32.7% | 2,731.2 | 465.0 | 6,791.4 | 0 |
| control@7 | 2,092.4 | 33.1% | 2,734.1 | 453.4 | 6,779.7 | 0 |
| control@13 | 2,052.4 | 32.4% | 2,720.0 | 574.9 | 6,901.2 | 0 |
| control@101 | 2,034.3 | 32.2% | 2,673.7 | 506.8 | 6,833.2 | 0 |
| control@271 | 2,072.0 | 32.8% | 2,743.0 | 599.0 | 6,925.3 | 0 |
| control@314 | 2,071.6 | 32.7% | 2,730.9 | 603.8 | 6,930.2 | 0 |
| control@577 | 2,060.1 | 32.6% | 2,709.7 | 588.5 | 6,914.8 | 0 |
| control@863 | 2,049.1 | 32.4% | 2,715.3 | 497.3 | 6,823.7 | 0 |
| control@1024 | 2,084.5 | 32.9% | 2,691.1 | 556.4 | 6,882.8 | 0 |
| control@1729 | 2,096.7 | 33.1% | 2,759.7 | 594.0 | 6,920.3 | 0 |

ABL-337 night screen: not applicable to wind_onshore.
