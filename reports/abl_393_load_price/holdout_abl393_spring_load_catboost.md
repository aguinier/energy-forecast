# Held-out A/B — load (ABL-393 reuse of the ABL-338 harness)

Generated 2026-08-13T21:44:29 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-04-30 .. 2026-06-12**, training from 2021-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `load` has no band structure, so one all-hours row is the result.

## AT / load — catboost, one fixed table for `load`

n_train 20,300 · n_holdout 1,056 · incumbent version 20260201_221635

| arm | n | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,056 | 375.9 | 6.3% | 623.0 | -17.6 | 5,963.1 | 0 |
| _constant_causal_ | 1,056 | 1,012.7 | 16.9% | 1,232.2 | 802.5 | 6,783.2 | 0 |
| _constant_oracle_ | 1,056 | 822.8 | 13.8% | 945.8 | -142.1 | 5,838.6 | 0 |
| _climatology_causal_ | 1,056 | 846.3 | 14.2% | 1,102.8 | 802.6 | 6,783.3 | 0 |
| _climatology_oracle_ | 1,056 | 531.1 | 8.9% | 794.7 | 298.5 | 6,279.2 | 0 |
| control@101 | 1,056 | 163.5 | 2.7% | 221.1 | 32.9 | 6,013.6 | 0 |
| control@103 | 1,056 | 165.9 | 2.8% | 225.3 | 28.7 | 6,009.4 | 0 |
| control@107 | 1,056 | 169.3 | 2.8% | 228.8 | 38.5 | 6,019.2 | 0 |
| control@109 | 1,056 | 177.8 | 3.0% | 243.4 | 47.4 | 6,028.1 | 0 |
| control@113 | 1,056 | 174.5 | 2.9% | 238.9 | 40.5 | 6,021.2 | 0 |
| control@127 | 1,056 | 177.9 | 3.0% | 244.0 | 37.3 | 6,018.0 | 0 |
| control@131 | 1,056 | 162.6 | 2.7% | 224.3 | 33.6 | 6,014.3 | 0 |
| control@137 | 1,056 | 172.4 | 2.9% | 238.0 | 34.2 | 6,014.9 | 0 |
| control_noholiday@101 | 1,056 | 210.5 | 3.5% | 306.7 | 78.1 | 6,058.8 | 0 |
| control_noholiday@103 | 1,056 | 208.9 | 3.5% | 308.2 | 89.0 | 6,069.7 | 0 |
| control_noholiday@107 | 1,056 | 211.9 | 3.5% | 310.3 | 80.0 | 6,060.6 | 0 |
| control_noholiday@109 | 1,056 | 204.7 | 3.4% | 307.8 | 71.4 | 6,052.1 | 0 |
| control_noholiday@113 | 1,056 | 202.3 | 3.4% | 296.2 | 77.5 | 6,058.2 | 0 |
| control_noholiday@127 | 1,056 | 209.2 | 3.5% | 312.0 | 71.3 | 6,052.0 | 0 |
| control_noholiday@131 | 1,056 | 211.7 | 3.5% | 311.3 | 72.0 | 6,052.7 | 0 |
| control_noholiday@137 | 1,056 | 209.1 | 3.5% | 307.4 | 71.1 | 6,051.8 | 0 |

Holiday subsets of the holdout (holiday 96 · holiday_affected 288 · ordinary 768). `holiday_affected` is a holiday, a bridge day, or within one day of a holiday — the rows these four features can distinguish from an ordinary day at all. A holiday effect that is real here is diluted by the row counts in the all-hours table above.

| arm | holiday MAE | holiday_affected MAE | ordinary MAE |
|---|---:|---:|---:|
| _seasonal-naive D-7_ | 1,165.0 | 531.1 | 317.6 |
| _constant_causal_ | 1,564.9 | 1,162.4 | 956.6 |
| _constant_oracle_ | 640.3 | 822.4 | 822.9 |
| _climatology_causal_ | 1,565.0 | 1,021.3 | 780.7 |
| _climatology_oracle_ | 1,070.6 | 682.4 | 474.4 |
| control@101 | 250.8 | 215.9 | 143.8 |
| control@103 | 247.0 | 226.8 | 143.1 |
| control@107 | 258.7 | 223.9 | 148.8 |
| control@109 | 299.1 | 235.8 | 156.0 |
| control@113 | 260.9 | 224.4 | 155.7 |
| control@127 | 280.6 | 238.1 | 155.3 |
| control@131 | 241.5 | 215.9 | 142.7 |
| control@137 | 274.4 | 229.8 | 150.8 |
| control_noholiday@101 | 480.1 | 309.4 | 173.3 |
| control_noholiday@103 | 509.2 | 308.4 | 171.5 |
| control_noholiday@107 | 513.1 | 318.2 | 172.0 |
| control_noholiday@109 | 510.1 | 314.0 | 163.7 |
| control_noholiday@113 | 492.9 | 310.0 | 162.0 |
| control_noholiday@127 | 543.5 | 331.3 | 163.4 |
| control_noholiday@131 | 523.2 | 328.3 | 168.0 |
| control_noholiday@137 | 551.0 | 332.4 | 162.8 |

ABL-337 night screen: not applicable to load.

## BE / load — catboost, one fixed table for `load`

n_train 46,629 · n_holdout 1,056 · incumbent version 20260404_185533

| arm | n | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,056 | 430.4 | 4.8% | 599.0 | 27.9 | 9,028.5 | 0 |
| _constant_causal_ | 1,056 | 1,008.9 | 11.2% | 1,186.5 | 314.8 | 9,315.4 | 0 |
| _constant_oracle_ | 1,056 | 977.6 | 10.9% | 1,145.4 | -56.3 | 8,944.3 | 0 |
| _climatology_causal_ | 1,056 | 595.6 | 6.6% | 754.3 | 314.8 | 9,315.4 | 0 |
| _climatology_oracle_ | 1,056 | 490.3 | 5.4% | 681.1 | 176.6 | 9,177.2 | 0 |
| control@101 | 1,056 | 188.5 | 2.1% | 248.3 | -12.6 | 8,988.0 | 0 |
| control@103 | 1,056 | 193.7 | 2.2% | 253.6 | -14.4 | 8,986.2 | 0 |
| control@107 | 1,056 | 192.7 | 2.1% | 254.1 | -9.0 | 8,991.6 | 0 |
| control@109 | 1,056 | 194.4 | 2.2% | 256.7 | -19.5 | 8,981.1 | 0 |
| control@113 | 1,056 | 195.0 | 2.2% | 259.1 | -19.5 | 8,981.2 | 0 |
| control@127 | 1,056 | 195.8 | 2.2% | 258.0 | -15.3 | 8,985.3 | 0 |
| control@131 | 1,056 | 194.6 | 2.2% | 259.9 | -21.8 | 8,978.8 | 0 |
| control@137 | 1,056 | 191.0 | 2.1% | 252.9 | -15.9 | 8,984.7 | 0 |
| control_noholiday@101 | 1,056 | 219.6 | 2.4% | 310.6 | -9.5 | 8,991.1 | 0 |
| control_noholiday@103 | 1,056 | 221.0 | 2.5% | 308.0 | -4.2 | 8,996.4 | 0 |
| control_noholiday@107 | 1,056 | 219.0 | 2.4% | 311.6 | -1.2 | 8,999.4 | 0 |
| control_noholiday@109 | 1,056 | 221.3 | 2.5% | 313.8 | -3.3 | 8,997.4 | 0 |
| control_noholiday@113 | 1,056 | 220.3 | 2.4% | 313.3 | -14.1 | 8,986.6 | 0 |
| control_noholiday@127 | 1,056 | 219.5 | 2.4% | 308.8 | -16.7 | 8,983.9 | 0 |
| control_noholiday@131 | 1,056 | 220.2 | 2.4% | 313.0 | -10.1 | 8,990.5 | 0 |
| control_noholiday@137 | 1,056 | 222.0 | 2.5% | 313.2 | -23.3 | 8,977.4 | 0 |

Holiday subsets of the holdout (holiday 96 · holiday_affected 240 · ordinary 816). `holiday_affected` is a holiday, a bridge day, or within one day of a holiday — the rows these four features can distinguish from an ordinary day at all. A holiday effect that is real here is diluted by the row counts in the all-hours table above.

| arm | holiday MAE | holiday_affected MAE | ordinary MAE |
|---|---:|---:|---:|
| _seasonal-naive D-7_ | 808.6 | 563.0 | 391.4 |
| _constant_causal_ | 1,010.9 | 1,000.7 | 1,011.3 |
| _constant_oracle_ | 777.2 | 889.6 | 1,003.5 |
| _climatology_causal_ | 905.1 | 718.8 | 559.4 |
| _climatology_oracle_ | 778.3 | 594.2 | 459.7 |
| control@101 | 248.3 | 223.0 | 178.3 |
| control@103 | 250.9 | 235.2 | 181.5 |
| control@107 | 259.8 | 233.3 | 180.7 |
| control@109 | 267.7 | 238.4 | 181.5 |
| control@113 | 253.0 | 235.3 | 183.1 |
| control@127 | 244.7 | 232.9 | 184.9 |
| control@131 | 278.3 | 243.4 | 180.2 |
| control@137 | 252.3 | 235.6 | 177.9 |
| control_noholiday@101 | 434.9 | 330.8 | 186.9 |
| control_noholiday@103 | 420.2 | 324.2 | 190.7 |
| control_noholiday@107 | 429.8 | 323.5 | 188.3 |
| control_noholiday@109 | 444.8 | 335.0 | 187.8 |
| control_noholiday@113 | 421.6 | 328.0 | 188.6 |
| control_noholiday@127 | 415.8 | 317.4 | 190.7 |
| control_noholiday@131 | 423.4 | 328.7 | 188.3 |
| control_noholiday@137 | 414.7 | 331.9 | 189.7 |

ABL-337 night screen: not applicable to load.

## DE / load — catboost, one fixed table for `load`

n_train 46,749 · n_holdout 1,056 · incumbent version 20260404_185521

| arm | n | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,056 | 3,458.0 | 7.1% | 5,314.5 | 17.7 | 48,831.0 | 0 |
| _constant_causal_ | 1,056 | 7,472.1 | 15.3% | 9,005.4 | 5,552.4 | 54,365.7 | 0 |
| _constant_oracle_ | 1,056 | 6,221.1 | 12.7% | 7,221.0 | -1,369.5 | 47,443.8 | 0 |
| _climatology_causal_ | 1,056 | 6,190.2 | 12.7% | 8,275.8 | 5,552.7 | 54,366.0 | 0 |
| _climatology_oracle_ | 1,056 | 4,550.4 | 9.3% | 6,181.6 | 2,157.9 | 50,971.1 | 0 |
| control@101 | 1,056 | 1,418.5 | 2.9% | 1,868.8 | 179.7 | 48,993.0 | 0 |
| control@103 | 1,056 | 1,422.6 | 2.9% | 1,884.7 | 266.1 | 49,079.4 | 0 |
| control@107 | 1,056 | 1,407.7 | 2.9% | 1,879.5 | 255.0 | 49,068.3 | 0 |
| control@109 | 1,056 | 1,397.1 | 2.9% | 1,841.6 | 179.2 | 48,992.4 | 0 |
| control@113 | 1,056 | 1,402.8 | 2.9% | 1,875.9 | 203.6 | 49,016.9 | 0 |
| control@127 | 1,056 | 1,412.8 | 2.9% | 1,853.5 | 154.5 | 48,967.7 | 0 |
| control@131 | 1,056 | 1,458.7 | 3.0% | 1,935.8 | 244.8 | 49,058.1 | 0 |
| control@137 | 1,056 | 1,395.0 | 2.9% | 1,846.9 | 254.4 | 49,067.7 | 0 |
| control_noholiday@101 | 1,056 | 1,678.2 | 3.4% | 2,438.4 | 457.3 | 49,270.5 | 0 |
| control_noholiday@103 | 1,056 | 1,684.7 | 3.5% | 2,444.5 | 484.7 | 49,298.0 | 0 |
| control_noholiday@107 | 1,056 | 1,663.1 | 3.4% | 2,395.2 | 467.4 | 49,280.7 | 0 |
| control_noholiday@109 | 1,056 | 1,660.9 | 3.4% | 2,400.5 | 521.4 | 49,334.7 | 0 |
| control_noholiday@113 | 1,056 | 1,652.2 | 3.4% | 2,381.2 | 472.2 | 49,285.5 | 0 |
| control_noholiday@127 | 1,056 | 1,654.6 | 3.4% | 2,390.1 | 473.5 | 49,286.8 | 0 |
| control_noholiday@131 | 1,056 | 1,656.4 | 3.4% | 2,378.1 | 468.1 | 49,281.4 | 0 |
| control_noholiday@137 | 1,056 | 1,675.5 | 3.4% | 2,429.8 | 462.9 | 49,276.2 | 0 |

Holiday subsets of the holdout (holiday 72 · holiday_affected 216 · ordinary 840). `holiday_affected` is a holiday, a bridge day, or within one day of a holiday — the rows these four features can distinguish from an ordinary day at all. A holiday effect that is real here is diluted by the row counts in the all-hours table above.

| arm | holiday MAE | holiday_affected MAE | ordinary MAE |
|---|---:|---:|---:|
| _seasonal-naive D-7_ | 11,553.2 | 5,803.2 | 2,854.9 |
| _constant_causal_ | 12,846.7 | 9,729.1 | 6,891.8 |
| _constant_oracle_ | 5,929.1 | 6,013.8 | 6,274.5 |
| _climatology_causal_ | 12,847.0 | 9,316.5 | 5,386.2 |
| _climatology_oracle_ | 9,452.1 | 6,381.5 | 4,079.5 |
| control@101 | 1,978.4 | 1,540.2 | 1,387.2 |
| control@103 | 1,940.3 | 1,550.1 | 1,389.8 |
| control@107 | 1,955.6 | 1,560.9 | 1,368.3 |
| control@109 | 1,922.4 | 1,471.8 | 1,377.9 |
| control@113 | 1,777.8 | 1,427.4 | 1,396.4 |
| control@127 | 1,985.2 | 1,558.3 | 1,375.4 |
| control@131 | 2,073.0 | 1,572.1 | 1,429.5 |
| control@137 | 1,922.7 | 1,481.6 | 1,372.7 |
| control_noholiday@101 | 4,952.6 | 2,710.9 | 1,412.7 |
| control_noholiday@103 | 5,041.0 | 2,733.6 | 1,415.0 |
| control_noholiday@107 | 4,790.9 | 2,680.2 | 1,401.6 |
| control_noholiday@109 | 4,759.3 | 2,646.0 | 1,407.7 |
| control_noholiday@113 | 4,691.3 | 2,628.9 | 1,401.0 |
| control_noholiday@127 | 4,826.8 | 2,647.2 | 1,399.4 |
| control_noholiday@131 | 4,672.7 | 2,649.8 | 1,400.9 |
| control_noholiday@137 | 4,958.4 | 2,759.1 | 1,396.9 |

ABL-337 night screen: not applicable to load.

## FR / load — catboost, one fixed table for `load`

n_train 46,627 · n_holdout 1,056 · incumbent version 20260404_185525

| arm | n | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,056 | 2,211.0 | 5.2% | 2,970.7 | 141.5 | 42,327.7 | 0 |
| _constant_causal_ | 1,056 | 7,564.2 | 17.9% | 9,008.9 | 7,494.0 | 49,680.2 | 0 |
| _constant_oracle_ | 1,056 | 4,258.0 | 10.1% | 5,038.9 | 624.8 | 42,811.0 | 0 |
| _climatology_causal_ | 1,056 | 7,499.2 | 17.8% | 8,422.4 | 7,494.0 | 49,680.2 | 0 |
| _climatology_oracle_ | 1,056 | 2,647.2 | 6.3% | 3,707.2 | 1,060.7 | 43,246.8 | 0 |
| control@101 | 1,056 | 1,104.0 | 2.6% | 1,512.8 | 3.5 | 42,189.7 | 0 |
| control@103 | 1,056 | 1,100.7 | 2.6% | 1,503.5 | -63.7 | 42,122.5 | 0 |
| control@107 | 1,056 | 1,143.3 | 2.7% | 1,566.5 | -161.0 | 42,025.2 | 0 |
| control@109 | 1,056 | 1,156.6 | 2.7% | 1,560.1 | -25.7 | 42,160.5 | 0 |
| control@113 | 1,056 | 1,094.8 | 2.6% | 1,504.2 | -68.1 | 42,118.1 | 0 |
| control@127 | 1,056 | 1,095.1 | 2.6% | 1,515.7 | -41.3 | 42,144.9 | 0 |
| control@131 | 1,056 | 1,139.7 | 2.7% | 1,550.3 | -139.8 | 42,046.4 | 0 |
| control@137 | 1,056 | 1,092.6 | 2.6% | 1,491.3 | -6.4 | 42,179.8 | 0 |
| control_noholiday@101 | 1,056 | 1,243.5 | 2.9% | 1,787.0 | -81.0 | 42,105.2 | 0 |
| control_noholiday@103 | 1,056 | 1,214.0 | 2.9% | 1,765.9 | -89.7 | 42,096.5 | 0 |
| control_noholiday@107 | 1,056 | 1,220.4 | 2.9% | 1,779.4 | -82.3 | 42,103.9 | 0 |
| control_noholiday@109 | 1,056 | 1,235.1 | 2.9% | 1,778.3 | -115.5 | 42,070.7 | 0 |
| control_noholiday@113 | 1,056 | 1,240.5 | 2.9% | 1,832.8 | -39.6 | 42,146.6 | 0 |
| control_noholiday@127 | 1,056 | 1,218.5 | 2.9% | 1,763.9 | -94.8 | 42,091.4 | 0 |
| control_noholiday@131 | 1,056 | 1,269.9 | 3.0% | 1,835.8 | -85.6 | 42,100.6 | 0 |
| control_noholiday@137 | 1,056 | 1,232.1 | 2.9% | 1,791.0 | -98.8 | 42,087.3 | 0 |

Holiday subsets of the holdout (holiday 96 · holiday_affected 288 · ordinary 768). `holiday_affected` is a holiday, a bridge day, or within one day of a holiday — the rows these four features can distinguish from an ordinary day at all. A holiday effect that is real here is diluted by the row counts in the all-hours table above.

| arm | holiday MAE | holiday_affected MAE | ordinary MAE |
|---|---:|---:|---:|
| _seasonal-naive D-7_ | 4,814.9 | 3,166.6 | 1,852.6 |
| _constant_causal_ | 10,460.0 | 8,830.8 | 7,089.2 |
| _constant_oracle_ | 4,215.1 | 4,481.5 | 4,174.2 |
| _climatology_causal_ | 10,460.1 | 8,779.8 | 7,019.0 |
| _climatology_oracle_ | 4,378.5 | 3,734.5 | 2,239.5 |
| control@101 | 2,309.8 | 1,603.6 | 916.6 |
| control@103 | 2,161.9 | 1,583.6 | 919.6 |
| control@107 | 2,371.7 | 1,623.7 | 963.1 |
| control@109 | 2,232.4 | 1,598.8 | 990.8 |
| control@113 | 2,227.3 | 1,582.4 | 911.9 |
| control@127 | 2,209.4 | 1,585.8 | 911.0 |
| control@131 | 2,267.9 | 1,617.3 | 960.6 |
| control@137 | 2,247.5 | 1,559.3 | 917.6 |
| control_noholiday@101 | 3,153.3 | 1,959.6 | 975.0 |
| control_noholiday@103 | 3,064.0 | 1,916.6 | 950.5 |
| control_noholiday@107 | 3,123.2 | 1,956.5 | 944.4 |
| control_noholiday@109 | 3,108.5 | 1,961.9 | 962.5 |
| control_noholiday@113 | 3,281.0 | 2,025.8 | 946.0 |
| control_noholiday@127 | 3,072.6 | 1,926.2 | 953.1 |
| control_noholiday@131 | 3,134.0 | 2,043.6 | 979.8 |
| control_noholiday@137 | 3,033.2 | 1,988.9 | 948.4 |

ABL-337 night screen: not applicable to load.
