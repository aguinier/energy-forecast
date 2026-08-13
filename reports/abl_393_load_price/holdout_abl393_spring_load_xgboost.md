# Held-out A/B — load (ABL-393 reuse of the ABL-338 harness)

Generated 2026-08-13T21:49:02 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-04-30 .. 2026-06-12**, training from 2021-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `load` has no band structure, so one all-hours row is the result.

## AT / load — xgboost, one fixed table for `load`

n_train 20,300 · n_holdout 1,056 · incumbent version 20260201_221635

| arm | n | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,056 | 375.9 | 6.3% | 623.0 | -17.6 | 5,963.1 | 0 |
| _constant_causal_ | 1,056 | 1,012.7 | 16.9% | 1,232.2 | 802.5 | 6,783.2 | 0 |
| _constant_oracle_ | 1,056 | 822.8 | 13.8% | 945.8 | -142.1 | 5,838.6 | 0 |
| _climatology_causal_ | 1,056 | 846.3 | 14.2% | 1,102.8 | 802.6 | 6,783.3 | 0 |
| _climatology_oracle_ | 1,056 | 531.1 | 8.9% | 794.7 | 298.5 | 6,279.2 | 0 |
| control@101 | 1,056 | 163.3 | 2.7% | 227.0 | 40.9 | 6,021.6 | 0 |
| control@103 | 1,056 | 157.8 | 2.6% | 220.1 | 28.2 | 6,008.9 | 0 |
| control@107 | 1,056 | 163.2 | 2.7% | 224.6 | 34.2 | 6,014.9 | 0 |
| control@109 | 1,056 | 162.6 | 2.7% | 226.7 | 37.4 | 6,018.1 | 0 |
| control@113 | 1,056 | 164.0 | 2.7% | 232.2 | 31.1 | 6,011.8 | 0 |
| control@127 | 1,056 | 165.1 | 2.8% | 230.8 | 36.6 | 6,017.3 | 0 |
| control@131 | 1,056 | 159.4 | 2.7% | 224.5 | 22.8 | 6,003.5 | 0 |
| control@137 | 1,056 | 161.6 | 2.7% | 221.2 | 24.1 | 6,004.8 | 0 |
| control_noholiday@101 | 1,056 | 193.5 | 3.2% | 287.6 | 58.9 | 6,039.6 | 0 |
| control_noholiday@103 | 1,056 | 193.8 | 3.2% | 283.4 | 70.0 | 6,050.7 | 0 |
| control_noholiday@107 | 1,056 | 192.8 | 3.2% | 291.4 | 58.8 | 6,039.5 | 0 |
| control_noholiday@109 | 1,056 | 183.4 | 3.1% | 273.1 | 53.5 | 6,034.2 | 0 |
| control_noholiday@113 | 1,056 | 189.2 | 3.2% | 281.0 | 56.2 | 6,036.9 | 0 |
| control_noholiday@127 | 1,056 | 186.5 | 3.1% | 278.3 | 48.4 | 6,029.1 | 0 |
| control_noholiday@131 | 1,056 | 191.8 | 3.2% | 284.9 | 59.2 | 6,039.9 | 0 |
| control_noholiday@137 | 1,056 | 187.6 | 3.1% | 278.6 | 52.1 | 6,032.8 | 0 |

Holiday subsets of the holdout (holiday 96 · holiday_affected 288 · ordinary 768). `holiday_affected` is a holiday, a bridge day, or within one day of a holiday — the rows these four features can distinguish from an ordinary day at all. A holiday effect that is real here is diluted by the row counts in the all-hours table above.

| arm | holiday MAE | holiday_affected MAE | ordinary MAE |
|---|---:|---:|---:|
| _seasonal-naive D-7_ | 1,165.0 | 531.1 | 317.6 |
| _constant_causal_ | 1,564.9 | 1,162.4 | 956.6 |
| _constant_oracle_ | 640.3 | 822.4 | 822.9 |
| _climatology_causal_ | 1,565.0 | 1,021.3 | 780.7 |
| _climatology_oracle_ | 1,070.6 | 682.4 | 474.4 |
| control@101 | 250.5 | 213.9 | 144.3 |
| control@103 | 244.8 | 209.1 | 138.6 |
| control@107 | 249.3 | 213.5 | 144.4 |
| control@109 | 250.9 | 215.6 | 142.7 |
| control@113 | 263.6 | 217.6 | 143.8 |
| control@127 | 274.2 | 217.5 | 145.4 |
| control@131 | 258.4 | 220.6 | 136.4 |
| control@137 | 245.0 | 215.1 | 141.5 |
| control_noholiday@101 | 490.7 | 306.2 | 151.2 |
| control_noholiday@103 | 467.3 | 301.3 | 153.5 |
| control_noholiday@107 | 512.6 | 317.1 | 146.2 |
| control_noholiday@109 | 462.0 | 291.9 | 142.7 |
| control_noholiday@113 | 478.6 | 305.4 | 145.6 |
| control_noholiday@127 | 468.3 | 305.3 | 142.0 |
| control_noholiday@131 | 459.0 | 299.5 | 151.4 |
| control_noholiday@137 | 497.9 | 314.9 | 139.8 |

ABL-337 night screen: not applicable to load.

## BE / load — xgboost, one fixed table for `load`

n_train 46,629 · n_holdout 1,056 · incumbent version 20260404_185533

| arm | n | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,056 | 430.4 | 4.8% | 599.0 | 27.9 | 9,028.5 | 0 |
| _constant_causal_ | 1,056 | 1,008.9 | 11.2% | 1,186.5 | 314.8 | 9,315.4 | 0 |
| _constant_oracle_ | 1,056 | 977.6 | 10.9% | 1,145.4 | -56.3 | 8,944.3 | 0 |
| _climatology_causal_ | 1,056 | 595.6 | 6.6% | 754.3 | 314.8 | 9,315.4 | 0 |
| _climatology_oracle_ | 1,056 | 490.3 | 5.4% | 681.1 | 176.6 | 9,177.2 | 0 |
| control@101 | 1,056 | 200.1 | 2.2% | 266.6 | -40.5 | 8,960.1 | 0 |
| control@103 | 1,056 | 194.4 | 2.2% | 261.2 | -32.7 | 8,967.9 | 0 |
| control@107 | 1,056 | 197.2 | 2.2% | 263.2 | -37.7 | 8,962.9 | 0 |
| control@109 | 1,056 | 194.5 | 2.2% | 262.1 | -35.8 | 8,964.9 | 0 |
| control@113 | 1,056 | 199.0 | 2.2% | 264.6 | -35.7 | 8,965.0 | 0 |
| control@127 | 1,056 | 193.9 | 2.2% | 260.3 | -39.1 | 8,961.6 | 0 |
| control@131 | 1,056 | 200.1 | 2.2% | 272.8 | -39.2 | 8,961.4 | 0 |
| control@137 | 1,056 | 202.3 | 2.2% | 269.4 | -29.4 | 8,971.2 | 0 |
| control_noholiday@101 | 1,056 | 221.9 | 2.5% | 309.9 | -34.2 | 8,966.4 | 0 |
| control_noholiday@103 | 1,056 | 212.9 | 2.4% | 298.1 | -36.8 | 8,963.8 | 0 |
| control_noholiday@107 | 1,056 | 218.0 | 2.4% | 306.6 | -20.2 | 8,980.4 | 0 |
| control_noholiday@109 | 1,056 | 220.2 | 2.4% | 308.1 | -21.8 | 8,978.9 | 0 |
| control_noholiday@113 | 1,056 | 218.8 | 2.4% | 307.5 | -33.3 | 8,967.3 | 0 |
| control_noholiday@127 | 1,056 | 224.1 | 2.5% | 317.9 | -12.3 | 8,988.3 | 0 |
| control_noholiday@131 | 1,056 | 218.2 | 2.4% | 307.1 | -28.2 | 8,972.4 | 0 |
| control_noholiday@137 | 1,056 | 223.0 | 2.5% | 310.3 | -25.0 | 8,975.6 | 0 |

Holiday subsets of the holdout (holiday 96 · holiday_affected 240 · ordinary 816). `holiday_affected` is a holiday, a bridge day, or within one day of a holiday — the rows these four features can distinguish from an ordinary day at all. A holiday effect that is real here is diluted by the row counts in the all-hours table above.

| arm | holiday MAE | holiday_affected MAE | ordinary MAE |
|---|---:|---:|---:|
| _seasonal-naive D-7_ | 808.6 | 563.0 | 391.4 |
| _constant_causal_ | 1,010.9 | 1,000.7 | 1,011.3 |
| _constant_oracle_ | 777.2 | 889.6 | 1,003.5 |
| _climatology_causal_ | 905.1 | 718.8 | 559.4 |
| _climatology_oracle_ | 778.3 | 594.2 | 459.7 |
| control@101 | 299.7 | 259.8 | 182.6 |
| control@103 | 294.0 | 250.1 | 178.0 |
| control@107 | 295.8 | 251.3 | 181.2 |
| control@109 | 303.7 | 260.5 | 175.1 |
| control@113 | 312.9 | 264.2 | 179.9 |
| control@127 | 286.8 | 248.2 | 177.9 |
| control@131 | 312.2 | 268.3 | 180.1 |
| control@137 | 292.8 | 254.7 | 186.9 |
| control_noholiday@101 | 434.2 | 331.5 | 189.7 |
| control_noholiday@103 | 420.5 | 326.0 | 179.6 |
| control_noholiday@107 | 430.5 | 331.3 | 184.7 |
| control_noholiday@109 | 444.9 | 332.0 | 187.3 |
| control_noholiday@113 | 416.7 | 325.5 | 187.4 |
| control_noholiday@127 | 452.0 | 333.0 | 192.0 |
| control_noholiday@131 | 430.8 | 326.6 | 186.3 |
| control_noholiday@137 | 424.8 | 325.2 | 192.9 |

ABL-337 night screen: not applicable to load.

## DE / load — xgboost, one fixed table for `load`

n_train 46,749 · n_holdout 1,056 · incumbent version 20260404_185521

| arm | n | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,056 | 3,458.0 | 7.1% | 5,314.5 | 17.7 | 48,831.0 | 0 |
| _constant_causal_ | 1,056 | 7,472.1 | 15.3% | 9,005.4 | 5,552.4 | 54,365.7 | 0 |
| _constant_oracle_ | 1,056 | 6,221.1 | 12.7% | 7,221.0 | -1,369.5 | 47,443.8 | 0 |
| _climatology_causal_ | 1,056 | 6,190.2 | 12.7% | 8,275.8 | 5,552.7 | 54,366.0 | 0 |
| _climatology_oracle_ | 1,056 | 4,550.4 | 9.3% | 6,181.6 | 2,157.9 | 50,971.1 | 0 |
| control@101 | 1,056 | 1,349.8 | 2.8% | 1,876.9 | 94.4 | 48,907.7 | 0 |
| control@103 | 1,056 | 1,316.5 | 2.7% | 1,839.1 | 105.3 | 48,918.6 | 0 |
| control@107 | 1,056 | 1,323.6 | 2.7% | 1,810.9 | 73.9 | 48,887.2 | 0 |
| control@109 | 1,056 | 1,319.2 | 2.7% | 1,853.6 | 64.2 | 48,877.5 | 0 |
| control@113 | 1,056 | 1,345.8 | 2.8% | 1,871.0 | 149.9 | 48,963.2 | 0 |
| control@127 | 1,056 | 1,345.1 | 2.8% | 1,868.3 | 59.3 | 48,872.6 | 0 |
| control@131 | 1,056 | 1,287.9 | 2.6% | 1,803.2 | 56.2 | 48,869.5 | 0 |
| control@137 | 1,056 | 1,343.7 | 2.8% | 1,874.2 | 55.0 | 48,868.2 | 0 |
| control_noholiday@101 | 1,056 | 1,598.3 | 3.3% | 2,430.4 | 297.0 | 49,110.3 | 0 |
| control_noholiday@103 | 1,056 | 1,640.8 | 3.4% | 2,385.5 | 288.9 | 49,102.1 | 0 |
| control_noholiday@107 | 1,056 | 1,584.9 | 3.2% | 2,409.6 | 296.6 | 49,109.9 | 0 |
| control_noholiday@109 | 1,056 | 1,617.2 | 3.3% | 2,417.2 | 279.1 | 49,092.4 | 0 |
| control_noholiday@113 | 1,056 | 1,571.0 | 3.2% | 2,396.2 | 300.7 | 49,114.0 | 0 |
| control_noholiday@127 | 1,056 | 1,615.1 | 3.3% | 2,463.9 | 370.4 | 49,183.7 | 0 |
| control_noholiday@131 | 1,056 | 1,587.3 | 3.3% | 2,410.3 | 341.7 | 49,155.0 | 0 |
| control_noholiday@137 | 1,056 | 1,594.7 | 3.3% | 2,426.9 | 268.8 | 49,082.1 | 0 |

Holiday subsets of the holdout (holiday 72 · holiday_affected 216 · ordinary 840). `holiday_affected` is a holiday, a bridge day, or within one day of a holiday — the rows these four features can distinguish from an ordinary day at all. A holiday effect that is real here is diluted by the row counts in the all-hours table above.

| arm | holiday MAE | holiday_affected MAE | ordinary MAE |
|---|---:|---:|---:|
| _seasonal-naive D-7_ | 11,553.2 | 5,803.2 | 2,854.9 |
| _constant_causal_ | 12,846.7 | 9,729.1 | 6,891.8 |
| _constant_oracle_ | 5,929.1 | 6,013.8 | 6,274.5 |
| _climatology_causal_ | 12,847.0 | 9,316.5 | 5,386.2 |
| _climatology_oracle_ | 9,452.1 | 6,381.5 | 4,079.5 |
| control@101 | 2,457.4 | 1,551.8 | 1,297.9 |
| control@103 | 2,179.1 | 1,434.9 | 1,286.1 |
| control@107 | 2,142.0 | 1,475.5 | 1,284.5 |
| control@109 | 2,308.7 | 1,462.0 | 1,282.5 |
| control@113 | 2,393.9 | 1,548.5 | 1,293.6 |
| control@127 | 2,337.5 | 1,531.0 | 1,297.3 |
| control@131 | 2,026.7 | 1,416.8 | 1,254.7 |
| control@137 | 2,179.3 | 1,473.3 | 1,310.3 |
| control_noholiday@101 | 5,391.4 | 2,718.0 | 1,310.4 |
| control_noholiday@103 | 5,075.5 | 2,704.0 | 1,367.4 |
| control_noholiday@107 | 5,368.8 | 2,735.5 | 1,289.1 |
| control_noholiday@109 | 5,292.1 | 2,694.4 | 1,340.2 |
| control_noholiday@113 | 5,339.1 | 2,652.6 | 1,292.9 |
| control_noholiday@127 | 5,531.2 | 2,765.5 | 1,319.3 |
| control_noholiday@131 | 5,472.4 | 2,726.7 | 1,294.3 |
| control_noholiday@137 | 5,226.5 | 2,642.4 | 1,325.3 |

ABL-337 night screen: not applicable to load.

## FR / load — xgboost, one fixed table for `load`

n_train 46,627 · n_holdout 1,056 · incumbent version 20260404_185525

| arm | n | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,056 | 2,211.0 | 5.2% | 2,970.7 | 141.5 | 42,327.7 | 0 |
| _constant_causal_ | 1,056 | 7,564.2 | 17.9% | 9,008.9 | 7,494.0 | 49,680.2 | 0 |
| _constant_oracle_ | 1,056 | 4,258.0 | 10.1% | 5,038.9 | 624.8 | 42,811.0 | 0 |
| _climatology_causal_ | 1,056 | 7,499.2 | 17.8% | 8,422.4 | 7,494.0 | 49,680.2 | 0 |
| _climatology_oracle_ | 1,056 | 2,647.2 | 6.3% | 3,707.2 | 1,060.7 | 43,246.8 | 0 |
| control@101 | 1,056 | 1,126.7 | 2.7% | 1,567.9 | -0.6 | 42,185.6 | 0 |
| control@103 | 1,056 | 1,111.9 | 2.6% | 1,521.5 | -28.9 | 42,157.2 | 0 |
| control@107 | 1,056 | 1,113.7 | 2.6% | 1,525.4 | 52.3 | 42,238.5 | 0 |
| control@109 | 1,056 | 1,143.2 | 2.7% | 1,607.8 | -29.9 | 42,156.3 | 0 |
| control@113 | 1,056 | 1,121.7 | 2.7% | 1,552.7 | -12.3 | 42,173.9 | 0 |
| control@127 | 1,056 | 1,106.3 | 2.6% | 1,530.6 | 14.2 | 42,200.4 | 0 |
| control@131 | 1,056 | 1,098.4 | 2.6% | 1,523.0 | 30.5 | 42,216.7 | 0 |
| control@137 | 1,056 | 1,099.3 | 2.6% | 1,511.8 | -11.7 | 42,174.5 | 0 |
| control_noholiday@101 | 1,056 | 1,207.0 | 2.9% | 1,735.9 | -71.7 | 42,114.5 | 0 |
| control_noholiday@103 | 1,056 | 1,182.1 | 2.8% | 1,728.1 | -2.8 | 42,183.4 | 0 |
| control_noholiday@107 | 1,056 | 1,189.7 | 2.8% | 1,737.2 | 15.7 | 42,201.9 | 0 |
| control_noholiday@109 | 1,056 | 1,154.1 | 2.7% | 1,680.8 | 39.0 | 42,225.1 | 0 |
| control_noholiday@113 | 1,056 | 1,177.1 | 2.8% | 1,711.1 | -53.6 | 42,132.6 | 0 |
| control_noholiday@127 | 1,056 | 1,176.4 | 2.8% | 1,715.8 | 11.6 | 42,197.8 | 0 |
| control_noholiday@131 | 1,056 | 1,155.1 | 2.7% | 1,709.2 | 36.3 | 42,222.5 | 0 |
| control_noholiday@137 | 1,056 | 1,162.5 | 2.8% | 1,699.9 | 25.5 | 42,211.7 | 0 |

Holiday subsets of the holdout (holiday 96 · holiday_affected 288 · ordinary 768). `holiday_affected` is a holiday, a bridge day, or within one day of a holiday — the rows these four features can distinguish from an ordinary day at all. A holiday effect that is real here is diluted by the row counts in the all-hours table above.

| arm | holiday MAE | holiday_affected MAE | ordinary MAE |
|---|---:|---:|---:|
| _seasonal-naive D-7_ | 4,814.9 | 3,166.6 | 1,852.6 |
| _constant_causal_ | 10,460.0 | 8,830.8 | 7,089.2 |
| _constant_oracle_ | 4,215.1 | 4,481.5 | 4,174.2 |
| _climatology_causal_ | 10,460.1 | 8,779.8 | 7,019.0 |
| _climatology_oracle_ | 4,378.5 | 3,734.5 | 2,239.5 |
| control@101 | 2,673.0 | 1,732.8 | 899.4 |
| control@103 | 2,539.9 | 1,732.8 | 879.1 |
| control@107 | 2,616.5 | 1,722.2 | 885.5 |
| control@109 | 2,623.6 | 1,791.7 | 900.0 |
| control@113 | 2,665.1 | 1,776.0 | 876.3 |
| control@127 | 2,654.5 | 1,709.0 | 880.3 |
| control@131 | 2,575.4 | 1,651.9 | 890.9 |
| control@137 | 2,519.0 | 1,687.4 | 878.8 |
| control_noholiday@101 | 3,012.1 | 1,933.7 | 934.4 |
| control_noholiday@103 | 3,128.4 | 1,963.2 | 889.2 |
| control_noholiday@107 | 3,083.0 | 1,984.0 | 891.9 |
| control_noholiday@109 | 3,054.7 | 1,911.9 | 869.9 |
| control_noholiday@113 | 3,002.6 | 1,934.2 | 893.2 |
| control_noholiday@127 | 3,072.0 | 1,958.0 | 883.3 |
| control_noholiday@131 | 3,185.8 | 1,938.4 | 861.4 |
| control_noholiday@137 | 3,101.8 | 1,920.6 | 878.2 |

ABL-337 night screen: not applicable to load.
