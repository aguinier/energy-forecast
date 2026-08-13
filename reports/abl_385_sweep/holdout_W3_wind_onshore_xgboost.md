# Held-out A/B — wind_onshore (ABL-385 reuse of the ABL-338 harness)

Generated 2026-08-13T13:42:04 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-04-14 .. 2026-05-13**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `wind_onshore` has no band structure, so one all-hours row is the result.

## AT / wind_onshore — xgboost, source `energy_renewable`

n_train 3,158 · n_holdout 720 · incumbent version 20260112_165238

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,026.3 | 98.0% | 1,289.8 | -5.0 | 1,042.5 | 0 |
| control@42 | 550.4 | 52.5% | 763.3 | 91.1 | 1,138.6 | 0 |
| control@1337 | 548.5 | 52.4% | 756.9 | 102.9 | 1,150.3 | 0 |
| control@2718 | 554.6 | 52.9% | 758.3 | 118.8 | 1,166.3 | 0 |
| control@7 | 544.7 | 52.0% | 753.1 | 61.8 | 1,109.2 | 0 |
| control@13 | 553.5 | 52.8% | 763.7 | 120.1 | 1,167.5 | 0 |
| control@101 | 543.7 | 51.9% | 758.2 | 139.5 | 1,186.9 | 0 |
| control@271 | 547.6 | 52.3% | 751.9 | 97.2 | 1,144.6 | 0 |
| control@314 | 564.9 | 53.9% | 774.5 | 132.4 | 1,179.8 | 0 |
| control@577 | 551.5 | 52.6% | 758.2 | 127.8 | 1,175.2 | 0 |
| control@863 | 538.5 | 51.4% | 741.4 | 90.8 | 1,138.2 | 0 |
| control@1024 | 540.3 | 51.6% | 751.4 | 83.6 | 1,131.1 | 0 |
| control@1729 | 534.7 | 51.1% | 731.7 | 109.4 | 1,156.8 | 0 |

ABL-337 night screen: not applicable to wind_onshore.

## BE / wind_onshore — xgboost, source `energy_renewable`

n_train 19,828 · n_holdout 720 · incumbent version 20260201_222020

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 520.6 | 90.3% | 693.4 | -1.3 | 575.3 | 0 |
| control@42 | 235.6 | 40.9% | 298.3 | 134.7 | 711.2 | 0 |
| control@1337 | 226.0 | 39.2% | 290.1 | 117.5 | 694.1 | 0 |
| control@2718 | 244.2 | 42.4% | 305.6 | 144.3 | 720.9 | 0 |
| control@7 | 236.5 | 41.0% | 296.7 | 134.8 | 711.4 | 0 |
| control@13 | 226.9 | 39.4% | 286.9 | 131.0 | 707.6 | 0 |
| control@101 | 223.6 | 38.8% | 284.7 | 127.8 | 704.3 | 0 |
| control@271 | 218.3 | 37.9% | 279.9 | 109.6 | 686.2 | 0 |
| control@314 | 226.8 | 39.3% | 289.4 | 113.9 | 690.5 | 0 |
| control@577 | 231.7 | 40.2% | 290.0 | 128.8 | 705.4 | 0 |
| control@863 | 221.5 | 38.4% | 282.7 | 115.8 | 692.4 | 0 |
| control@1024 | 241.3 | 41.8% | 304.9 | 142.0 | 718.6 | 0 |
| control@1729 | 223.8 | 38.8% | 284.4 | 116.6 | 693.1 | 0 |

ABL-337 night screen: not applicable to wind_onshore.

## DE / wind_onshore — xgboost, source `energy_renewable`

n_train 5,093 · n_holdout 720 · incumbent version 20260201_222000

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 8,106.4 | 85.4% | 9,962.9 | -53.9 | 9,439.8 | 0 |
| control@42 | 3,309.8 | 34.9% | 4,474.8 | 991.0 | 10,484.6 | 0 |
| control@1337 | 3,231.2 | 34.0% | 4,455.3 | 832.2 | 10,325.8 | 0 |
| control@2718 | 3,282.5 | 34.6% | 4,383.6 | 1,071.0 | 10,564.7 | 0 |
| control@7 | 3,098.4 | 32.6% | 4,290.3 | 952.4 | 10,446.1 | 0 |
| control@13 | 3,247.5 | 34.2% | 4,424.4 | 805.2 | 10,298.9 | 0 |
| control@101 | 3,216.2 | 33.9% | 4,454.7 | 942.1 | 10,435.7 | 0 |
| control@271 | 3,148.2 | 33.2% | 4,293.2 | 998.9 | 10,492.6 | 0 |
| control@314 | 3,231.8 | 34.0% | 4,397.1 | 941.5 | 10,435.2 | 0 |
| control@577 | 3,131.5 | 33.0% | 4,304.1 | 949.0 | 10,442.7 | 0 |
| control@863 | 3,250.6 | 34.2% | 4,455.7 | 870.1 | 10,363.8 | 0 |
| control@1024 | 3,212.6 | 33.8% | 4,467.2 | 984.5 | 10,478.2 | 0 |
| control@1729 | 3,313.8 | 34.9% | 4,483.0 | 1,120.9 | 10,614.6 | 0 |

ABL-337 night screen: not applicable to wind_onshore.

## FR / wind_onshore — xgboost, source `energy_renewable`

n_train 28,716 · n_holdout 720 · incumbent version 20260201_222010

| arm | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 2,627.8 | 63.8% | 3,538.0 | -134.2 | 3,982.4 | 0 |
| control@42 | 1,499.1 | 36.4% | 1,992.0 | 395.0 | 4,511.6 | 0 |
| control@1337 | 1,512.6 | 36.7% | 2,028.0 | 449.2 | 4,565.8 | 0 |
| control@2718 | 1,448.4 | 35.2% | 1,931.4 | 413.9 | 4,530.5 | 0 |
| control@7 | 1,487.7 | 36.1% | 1,959.4 | 442.9 | 4,559.5 | 0 |
| control@13 | 1,457.3 | 35.4% | 1,959.2 | 372.6 | 4,489.2 | 0 |
| control@101 | 1,483.8 | 36.0% | 1,968.4 | 410.7 | 4,527.3 | 0 |
| control@271 | 1,587.4 | 38.6% | 2,072.9 | 514.6 | 4,631.2 | 0 |
| control@314 | 1,524.2 | 37.0% | 2,034.6 | 417.9 | 4,534.5 | 0 |
| control@577 | 1,496.9 | 36.4% | 1,960.2 | 442.6 | 4,559.2 | 0 |
| control@863 | 1,589.0 | 38.6% | 2,060.9 | 517.5 | 4,634.1 | 0 |
| control@1024 | 1,502.8 | 36.5% | 1,992.3 | 439.4 | 4,556.0 | 0 |
| control@1729 | 1,585.6 | 38.5% | 2,068.9 | 601.7 | 4,718.3 | 0 |

ABL-337 night screen: not applicable to wind_onshore.
