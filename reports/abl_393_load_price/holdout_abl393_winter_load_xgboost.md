# Held-out A/B — load (ABL-393 reuse of the ABL-338 harness)

Generated 2026-08-13T22:01:59 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2025-12-06 .. 2026-01-18**, training from 2021-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `load` has no band structure, so one all-hours row is the result.

## AT / load — xgboost, one fixed table for `load`

n_train 16,848 · n_holdout 1,056 · incumbent version 20260201_221635

| arm | n | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,056 | 540.4 | 7.1% | 848.0 | -14.6 | 7,576.4 | 0 |
| _constant_causal_ | 1,056 | 1,165.1 | 15.3% | 1,481.3 | -924.1 | 6,667.0 | 0 |
| _constant_oracle_ | 1,056 | 972.1 | 12.8% | 1,168.8 | -160.2 | 7,430.9 | 0 |
| _climatology_causal_ | 1,056 | 1,031.7 | 13.6% | 1,238.6 | -924.1 | 6,667.0 | 0 |
| _climatology_oracle_ | 1,056 | 647.9 | 8.5% | 783.9 | 8.6 | 7,599.6 | 0 |
| control@101 | 1,056 | 259.1 | 3.4% | 371.8 | 26.0 | 7,617.1 | 0 |
| control@103 | 1,056 | 264.6 | 3.5% | 381.9 | 33.2 | 7,624.2 | 0 |
| control@107 | 1,056 | 266.7 | 3.5% | 391.1 | 43.8 | 7,634.9 | 0 |
| control@109 | 1,056 | 266.2 | 3.5% | 384.1 | 33.9 | 7,625.0 | 0 |
| control@113 | 1,056 | 268.8 | 3.5% | 386.3 | 37.2 | 7,628.2 | 0 |
| control@127 | 1,056 | 264.3 | 3.5% | 385.8 | 42.4 | 7,633.5 | 0 |
| control@131 | 1,056 | 269.0 | 3.5% | 385.4 | 36.1 | 7,627.2 | 0 |
| control@137 | 1,056 | 264.8 | 3.5% | 376.0 | 33.7 | 7,624.7 | 0 |
| control_noholiday@101 | 1,056 | 288.8 | 3.8% | 418.3 | 56.7 | 7,647.7 | 0 |
| control_noholiday@103 | 1,056 | 298.0 | 3.9% | 429.4 | 67.8 | 7,658.8 | 0 |
| control_noholiday@107 | 1,056 | 290.4 | 3.8% | 422.2 | 41.6 | 7,632.7 | 0 |
| control_noholiday@109 | 1,056 | 290.3 | 3.8% | 429.5 | 51.8 | 7,642.9 | 0 |
| control_noholiday@113 | 1,056 | 298.0 | 3.9% | 426.0 | 63.4 | 7,654.5 | 0 |
| control_noholiday@127 | 1,056 | 293.7 | 3.9% | 429.4 | 68.8 | 7,659.9 | 0 |
| control_noholiday@131 | 1,056 | 288.0 | 3.8% | 419.5 | 44.9 | 7,635.9 | 0 |
| control_noholiday@137 | 1,056 | 293.5 | 3.9% | 425.4 | 57.4 | 7,648.5 | 0 |

Holiday subsets of the holdout (holiday 120 · holiday_affected 312 · ordinary 744). `holiday_affected` is a holiday, a bridge day, or within one day of a holiday — the rows these four features can distinguish from an ordinary day at all. A holiday effect that is real here is diluted by the row counts in the all-hours table above.

| arm | holiday MAE | holiday_affected MAE | ordinary MAE |
|---|---:|---:|---:|
| _seasonal-naive D-7_ | 1,231.6 | 939.9 | 372.8 |
| _constant_causal_ | 621.8 | 880.3 | 1,284.5 |
| _constant_oracle_ | 844.0 | 860.6 | 1,018.9 |
| _climatology_causal_ | 528.6 | 720.1 | 1,162.4 |
| _climatology_oracle_ | 940.7 | 690.1 | 630.2 |
| control@101 | 357.4 | 389.8 | 204.3 |
| control@103 | 381.3 | 392.4 | 211.0 |
| control@107 | 377.8 | 386.7 | 216.3 |
| control@109 | 396.3 | 412.0 | 205.1 |
| control@113 | 418.7 | 408.7 | 210.1 |
| control@127 | 410.2 | 402.3 | 206.5 |
| control@131 | 363.8 | 397.1 | 215.3 |
| control@137 | 383.0 | 389.9 | 212.3 |
| control_noholiday@101 | 539.7 | 481.1 | 208.2 |
| control_noholiday@103 | 565.6 | 504.4 | 211.5 |
| control_noholiday@107 | 573.3 | 497.9 | 203.5 |
| control_noholiday@109 | 602.3 | 501.3 | 201.8 |
| control_noholiday@113 | 586.9 | 509.2 | 209.4 |
| control_noholiday@127 | 590.5 | 509.0 | 203.5 |
| control_noholiday@131 | 553.6 | 492.8 | 202.1 |
| control_noholiday@137 | 583.3 | 499.3 | 207.2 |

ABL-337 night screen: not applicable to load.

## BE / load — xgboost, one fixed table for `load`

n_train 43,176 · n_holdout 1,056 · incumbent version 20260404_185533

| arm | n | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,056 | 706.7 | 6.8% | 970.3 | -98.9 | 10,273.7 | 0 |
| _constant_causal_ | 1,056 | 1,443.2 | 13.9% | 1,780.6 | -1,121.6 | 9,251.1 | 0 |
| _constant_oracle_ | 1,056 | 1,154.1 | 11.1% | 1,384.8 | -70.3 | 10,302.4 | 0 |
| _climatology_causal_ | 1,056 | 1,196.7 | 11.5% | 1,449.4 | -1,121.6 | 9,251.1 | 0 |
| _climatology_oracle_ | 1,056 | 668.5 | 6.4% | 861.6 | -64.8 | 10,307.8 | 0 |
| control@101 | 1,056 | 258.4 | 2.5% | 344.2 | -82.3 | 10,290.3 | 0 |
| control@103 | 1,056 | 252.5 | 2.4% | 336.4 | -81.6 | 10,291.1 | 0 |
| control@107 | 1,056 | 257.4 | 2.5% | 343.7 | -94.2 | 10,278.4 | 0 |
| control@109 | 1,056 | 255.8 | 2.5% | 337.5 | -86.1 | 10,286.6 | 0 |
| control@113 | 1,056 | 257.7 | 2.5% | 344.7 | -85.3 | 10,287.3 | 0 |
| control@127 | 1,056 | 256.5 | 2.5% | 341.9 | -87.6 | 10,285.0 | 0 |
| control@131 | 1,056 | 260.8 | 2.5% | 346.1 | -89.3 | 10,283.4 | 0 |
| control@137 | 1,056 | 255.1 | 2.5% | 344.1 | -88.7 | 10,283.9 | 0 |
| control_noholiday@101 | 1,056 | 278.4 | 2.7% | 387.3 | -47.0 | 10,325.6 | 0 |
| control_noholiday@103 | 1,056 | 277.7 | 2.7% | 384.6 | -44.2 | 10,328.5 | 0 |
| control_noholiday@107 | 1,056 | 281.0 | 2.7% | 384.9 | -50.8 | 10,321.8 | 0 |
| control_noholiday@109 | 1,056 | 272.4 | 2.6% | 377.7 | -47.8 | 10,324.8 | 0 |
| control_noholiday@113 | 1,056 | 279.6 | 2.7% | 388.8 | -40.5 | 10,332.2 | 0 |
| control_noholiday@127 | 1,056 | 275.8 | 2.7% | 382.9 | -43.4 | 10,329.3 | 0 |
| control_noholiday@131 | 1,056 | 278.1 | 2.7% | 383.7 | -39.0 | 10,333.6 | 0 |
| control_noholiday@137 | 1,056 | 278.2 | 2.7% | 383.0 | -32.2 | 10,340.5 | 0 |

Holiday subsets of the holdout (holiday 48 · holiday_affected 144 · ordinary 912). `holiday_affected` is a holiday, a bridge day, or within one day of a holiday — the rows these four features can distinguish from an ordinary day at all. A holiday effect that is real here is diluted by the row counts in the all-hours table above.

| arm | holiday MAE | holiday_affected MAE | ordinary MAE |
|---|---:|---:|---:|
| _seasonal-naive D-7_ | 818.6 | 522.6 | 735.8 |
| _constant_causal_ | 604.3 | 992.6 | 1,514.4 |
| _constant_oracle_ | 946.2 | 965.3 | 1,183.9 |
| _climatology_causal_ | 413.0 | 702.9 | 1,274.7 |
| _climatology_oracle_ | 923.9 | 506.3 | 694.2 |
| control@101 | 332.5 | 300.2 | 251.8 |
| control@103 | 322.5 | 273.1 | 249.3 |
| control@107 | 356.7 | 304.7 | 249.9 |
| control@109 | 340.7 | 294.5 | 249.7 |
| control@113 | 342.1 | 296.2 | 251.6 |
| control@127 | 331.2 | 291.6 | 251.0 |
| control@131 | 380.7 | 336.6 | 248.8 |
| control@137 | 350.3 | 289.5 | 249.7 |
| control_noholiday@101 | 642.4 | 445.3 | 252.0 |
| control_noholiday@103 | 615.3 | 435.7 | 252.7 |
| control_noholiday@107 | 658.7 | 460.1 | 252.8 |
| control_noholiday@109 | 597.9 | 423.3 | 248.5 |
| control_noholiday@113 | 663.2 | 438.6 | 254.5 |
| control_noholiday@127 | 632.0 | 447.7 | 248.7 |
| control_noholiday@131 | 640.7 | 434.7 | 253.4 |
| control_noholiday@137 | 640.1 | 453.9 | 250.5 |

ABL-337 night screen: not applicable to load.

## DE / load — xgboost, one fixed table for `load`

n_train 43,296 · n_holdout 1,056 · incumbent version 20260404_185521

| arm | n | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,056 | 4,726.9 | 8.3% | 6,611.6 | -37.6 | 56,669.9 | 0 |
| _constant_causal_ | 1,056 | 7,707.1 | 13.6% | 9,460.1 | -2,515.6 | 54,191.8 | 0 |
| _constant_oracle_ | 1,056 | 7,600.5 | 13.4% | 9,163.9 | -901.1 | 55,806.4 | 0 |
| _climatology_causal_ | 1,056 | 5,840.1 | 10.3% | 7,148.6 | -2,515.6 | 54,191.8 | 0 |
| _climatology_oracle_ | 1,056 | 5,442.4 | 9.6% | 6,637.2 | -1,184.1 | 55,523.3 | 0 |
| control@101 | 1,056 | 1,335.3 | 2.4% | 1,864.8 | -367.8 | 56,339.6 | 0 |
| control@103 | 1,056 | 1,335.6 | 2.4% | 1,841.6 | -267.7 | 56,439.7 | 0 |
| control@107 | 1,056 | 1,333.1 | 2.4% | 1,859.0 | -356.2 | 56,351.2 | 0 |
| control@109 | 1,056 | 1,303.6 | 2.3% | 1,798.1 | -250.4 | 56,457.0 | 0 |
| control@113 | 1,056 | 1,330.5 | 2.3% | 1,812.4 | -391.1 | 56,316.3 | 0 |
| control@127 | 1,056 | 1,299.9 | 2.3% | 1,778.9 | -292.0 | 56,415.5 | 0 |
| control@131 | 1,056 | 1,363.0 | 2.4% | 1,875.5 | -352.3 | 56,355.1 | 0 |
| control@137 | 1,056 | 1,365.3 | 2.4% | 1,879.7 | -321.8 | 56,385.6 | 0 |
| control_noholiday@101 | 1,056 | 1,410.6 | 2.5% | 1,976.0 | -321.6 | 56,385.9 | 0 |
| control_noholiday@103 | 1,056 | 1,433.2 | 2.5% | 2,028.3 | -400.3 | 56,307.2 | 0 |
| control_noholiday@107 | 1,056 | 1,419.9 | 2.5% | 1,963.1 | -441.0 | 56,266.4 | 0 |
| control_noholiday@109 | 1,056 | 1,412.0 | 2.5% | 1,979.1 | -308.7 | 56,398.7 | 0 |
| control_noholiday@113 | 1,056 | 1,390.3 | 2.5% | 1,934.6 | -277.0 | 56,430.5 | 0 |
| control_noholiday@127 | 1,056 | 1,362.5 | 2.4% | 1,870.6 | -341.7 | 56,365.8 | 0 |
| control_noholiday@131 | 1,056 | 1,373.1 | 2.4% | 1,932.4 | -300.2 | 56,407.2 | 0 |
| control_noholiday@137 | 1,056 | 1,387.4 | 2.4% | 1,971.9 | -370.7 | 56,336.7 | 0 |

Holiday subsets of the holdout (holiday 72 · holiday_affected 168 · ordinary 888). `holiday_affected` is a holiday, a bridge day, or within one day of a holiday — the rows these four features can distinguish from an ordinary day at all. A holiday effect that is real here is diluted by the row counts in the all-hours table above.

| arm | holiday MAE | holiday_affected MAE | ordinary MAE |
|---|---:|---:|---:|
| _seasonal-naive D-7_ | 6,937.3 | 5,761.2 | 4,531.2 |
| _constant_causal_ | 5,826.5 | 5,491.7 | 8,126.2 |
| _constant_oracle_ | 7,288.3 | 6,456.1 | 7,817.0 |
| _climatology_causal_ | 5,755.9 | 4,567.2 | 6,080.9 |
| _climatology_oracle_ | 6,976.6 | 5,397.5 | 5,450.9 |
| control@101 | 1,280.0 | 1,371.9 | 1,328.3 |
| control@103 | 1,233.2 | 1,247.5 | 1,352.3 |
| control@107 | 1,309.3 | 1,385.5 | 1,323.1 |
| control@109 | 994.4 | 1,240.7 | 1,315.4 |
| control@113 | 1,318.6 | 1,371.8 | 1,322.7 |
| control@127 | 1,258.8 | 1,424.0 | 1,276.4 |
| control@131 | 1,231.9 | 1,453.6 | 1,345.9 |
| control@137 | 1,336.1 | 1,437.9 | 1,351.6 |
| control_noholiday@101 | 1,883.9 | 1,836.0 | 1,330.1 |
| control_noholiday@103 | 1,728.0 | 1,950.6 | 1,335.3 |
| control_noholiday@107 | 1,824.0 | 1,856.6 | 1,337.3 |
| control_noholiday@109 | 1,802.0 | 1,778.2 | 1,342.7 |
| control_noholiday@113 | 1,840.4 | 1,789.2 | 1,314.9 |
| control_noholiday@127 | 1,815.1 | 1,868.6 | 1,266.8 |
| control_noholiday@131 | 1,719.9 | 1,776.3 | 1,296.9 |
| control_noholiday@137 | 1,893.6 | 1,969.1 | 1,277.4 |

ABL-337 night screen: not applicable to load.

## FR / load — xgboost, one fixed table for `load`

n_train 43,200 · n_holdout 1,031 · incumbent version 20260404_185525

| arm | n | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,031 | 8,005.8 | 12.8% | 9,788.2 | 73.8 | 62,572.2 | 0 |
| _constant_causal_ | 1,031 | 13,551.1 | 21.7% | 15,971.7 | -13,320.7 | 49,177.7 | 0 |
| _constant_oracle_ | 1,031 | 6,866.3 | 11.0% | 8,841.0 | -713.6 | 61,784.8 | 0 |
| _climatology_causal_ | 1,031 | 13,586.7 | 21.7% | 15,755.4 | -13,318.5 | 49,179.9 | 0 |
| _climatology_oracle_ | 1,031 | 6,338.8 | 10.1% | 8,396.7 | -593.3 | 61,905.1 | 0 |
| control@101 | 1,031 | 3,108.1 | 5.0% | 4,107.0 | 101.1 | 62,599.5 | 0 |
| control@103 | 1,031 | 3,081.1 | 4.9% | 4,106.1 | 115.6 | 62,614.0 | 0 |
| control@107 | 1,031 | 3,123.6 | 5.0% | 4,142.4 | -78.3 | 62,420.1 | 0 |
| control@109 | 1,031 | 3,135.4 | 5.0% | 4,174.7 | 77.8 | 62,576.2 | 0 |
| control@113 | 1,031 | 3,076.4 | 4.9% | 4,162.5 | 195.6 | 62,694.0 | 0 |
| control@127 | 1,031 | 3,019.0 | 4.8% | 4,048.2 | 64.2 | 62,562.6 | 0 |
| control@131 | 1,031 | 3,145.9 | 5.0% | 4,155.1 | 110.5 | 62,609.0 | 0 |
| control@137 | 1,031 | 3,126.8 | 5.0% | 4,137.5 | 144.7 | 62,643.1 | 0 |
| control_noholiday@101 | 1,031 | 3,073.6 | 4.9% | 4,083.4 | -85.9 | 62,412.5 | 0 |
| control_noholiday@103 | 1,031 | 3,010.5 | 4.8% | 4,009.2 | -22.7 | 62,475.7 | 0 |
| control_noholiday@107 | 1,031 | 3,074.9 | 4.9% | 4,137.4 | -9.1 | 62,489.3 | 0 |
| control_noholiday@109 | 1,031 | 3,050.2 | 4.9% | 4,110.0 | 94.9 | 62,593.3 | 0 |
| control_noholiday@113 | 1,031 | 3,063.4 | 4.9% | 4,099.1 | -41.8 | 62,456.6 | 0 |
| control_noholiday@127 | 1,031 | 3,150.1 | 5.0% | 4,242.1 | -4.1 | 62,494.3 | 0 |
| control_noholiday@131 | 1,031 | 3,072.0 | 4.9% | 4,119.5 | 75.1 | 62,573.6 | 0 |
| control_noholiday@137 | 1,031 | 3,106.0 | 5.0% | 4,189.5 | -108.5 | 62,389.9 | 0 |

Holiday subsets of the holdout (holiday 24 · holiday_affected 119 · ordinary 912). `holiday_affected` is a holiday, a bridge day, or within one day of a holiday — the rows these four features can distinguish from an ordinary day at all. A holiday effect that is real here is diluted by the row counts in the all-hours table above.

| arm | holiday MAE | holiday_affected MAE | ordinary MAE |
|---|---:|---:|---:|
| _seasonal-naive D-7_ | 3,528.3 | 6,527.5 | 8,198.7 |
| _constant_causal_ | 10,936.7 | 15,914.8 | 13,242.7 |
| _constant_oracle_ | 2,545.5 | 5,449.0 | 7,051.2 |
| _climatology_causal_ | 10,936.7 | 15,896.2 | 13,285.3 |
| _climatology_oracle_ | 2,394.5 | 4,546.2 | 6,572.7 |
| control@101 | 3,833.4 | 2,930.8 | 3,131.2 |
| control@103 | 3,635.8 | 2,781.9 | 3,120.1 |
| control@107 | 3,256.7 | 2,741.2 | 3,173.5 |
| control@109 | 3,603.9 | 2,841.9 | 3,173.7 |
| control@113 | 3,780.3 | 2,785.9 | 3,114.4 |
| control@127 | 3,527.1 | 2,693.6 | 3,061.4 |
| control@131 | 3,624.3 | 2,899.0 | 3,178.1 |
| control@137 | 3,680.6 | 2,955.1 | 3,149.2 |
| control_noholiday@101 | 3,724.3 | 2,683.9 | 3,124.4 |
| control_noholiday@103 | 3,370.6 | 2,554.9 | 3,070.0 |
| control_noholiday@107 | 2,987.1 | 2,583.7 | 3,139.0 |
| control_noholiday@109 | 2,953.1 | 2,396.6 | 3,135.5 |
| control_noholiday@113 | 3,556.2 | 2,708.1 | 3,109.8 |
| control_noholiday@127 | 3,194.1 | 2,591.8 | 3,222.9 |
| control_noholiday@131 | 3,822.4 | 2,694.0 | 3,121.3 |
| control_noholiday@137 | 3,087.3 | 2,677.3 | 3,162.0 |

ABL-337 night screen: not applicable to load.
