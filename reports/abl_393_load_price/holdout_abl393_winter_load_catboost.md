# Held-out A/B — load (ABL-393 reuse of the ABL-338 harness)

Generated 2026-08-13T21:56:58 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2025-12-06 .. 2026-01-18**, training from 2021-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. `load` has no band structure, so one all-hours row is the result.

## AT / load — catboost, one fixed table for `load`

n_train 16,848 · n_holdout 1,056 · incumbent version 20260201_221635

| arm | n | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,056 | 540.4 | 7.1% | 848.0 | -14.6 | 7,576.4 | 0 |
| _constant_causal_ | 1,056 | 1,165.1 | 15.3% | 1,481.3 | -924.1 | 6,667.0 | 0 |
| _constant_oracle_ | 1,056 | 972.1 | 12.8% | 1,168.8 | -160.2 | 7,430.9 | 0 |
| _climatology_causal_ | 1,056 | 1,031.7 | 13.6% | 1,238.6 | -924.1 | 6,667.0 | 0 |
| _climatology_oracle_ | 1,056 | 647.9 | 8.5% | 783.9 | 8.6 | 7,599.6 | 0 |
| control@101 | 1,056 | 248.4 | 3.3% | 363.2 | -2.6 | 7,588.5 | 0 |
| control@103 | 1,056 | 256.1 | 3.4% | 372.7 | 44.8 | 7,635.9 | 0 |
| control@107 | 1,056 | 242.1 | 3.2% | 360.8 | 21.7 | 7,612.8 | 0 |
| control@109 | 1,056 | 240.4 | 3.2% | 354.0 | 17.2 | 7,608.2 | 0 |
| control@113 | 1,056 | 233.9 | 3.1% | 349.8 | 13.4 | 7,604.5 | 0 |
| control@127 | 1,056 | 251.0 | 3.3% | 366.3 | 20.1 | 7,611.1 | 0 |
| control@131 | 1,056 | 238.1 | 3.1% | 351.6 | 22.7 | 7,613.7 | 0 |
| control@137 | 1,056 | 248.6 | 3.3% | 360.8 | 28.3 | 7,619.4 | 0 |
| control_noholiday@101 | 1,056 | 288.2 | 3.8% | 436.2 | 57.0 | 7,648.1 | 0 |
| control_noholiday@103 | 1,056 | 298.6 | 3.9% | 451.7 | 59.8 | 7,650.8 | 0 |
| control_noholiday@107 | 1,056 | 291.8 | 3.8% | 436.8 | 35.0 | 7,626.1 | 0 |
| control_noholiday@109 | 1,056 | 290.1 | 3.8% | 439.5 | 70.8 | 7,661.9 | 0 |
| control_noholiday@113 | 1,056 | 281.3 | 3.7% | 432.6 | 48.2 | 7,639.3 | 0 |
| control_noholiday@127 | 1,056 | 297.9 | 3.9% | 446.8 | 62.2 | 7,653.3 | 0 |
| control_noholiday@131 | 1,056 | 294.2 | 3.9% | 442.5 | 63.4 | 7,654.4 | 0 |
| control_noholiday@137 | 1,056 | 285.9 | 3.8% | 428.7 | 52.6 | 7,643.6 | 0 |

Holiday subsets of the holdout (holiday 120 · holiday_affected 312 · ordinary 744). `holiday_affected` is a holiday, a bridge day, or within one day of a holiday — the rows these four features can distinguish from an ordinary day at all. A holiday effect that is real here is diluted by the row counts in the all-hours table above.

| arm | holiday MAE | holiday_affected MAE | ordinary MAE |
|---|---:|---:|---:|
| _seasonal-naive D-7_ | 1,231.6 | 939.9 | 372.8 |
| _constant_causal_ | 621.8 | 880.3 | 1,284.5 |
| _constant_oracle_ | 844.0 | 860.6 | 1,018.9 |
| _climatology_causal_ | 528.6 | 720.1 | 1,162.4 |
| _climatology_oracle_ | 940.7 | 690.1 | 630.2 |
| control@101 | 249.4 | 354.0 | 204.2 |
| control@103 | 270.7 | 352.0 | 215.8 |
| control@107 | 282.9 | 349.2 | 197.2 |
| control@109 | 263.8 | 339.0 | 199.1 |
| control@113 | 259.1 | 335.5 | 191.3 |
| control@127 | 247.5 | 354.1 | 207.7 |
| control@131 | 250.0 | 333.9 | 197.9 |
| control@137 | 278.3 | 343.5 | 208.8 |
| control_noholiday@101 | 585.8 | 513.5 | 193.7 |
| control_noholiday@103 | 628.8 | 521.0 | 205.4 |
| control_noholiday@107 | 568.4 | 502.3 | 203.5 |
| control_noholiday@109 | 587.9 | 503.3 | 200.6 |
| control_noholiday@113 | 574.8 | 489.9 | 193.8 |
| control_noholiday@127 | 607.2 | 514.6 | 207.1 |
| control_noholiday@131 | 599.1 | 513.3 | 202.3 |
| control_noholiday@137 | 548.7 | 485.6 | 202.2 |

ABL-337 night screen: not applicable to load.

## BE / load — catboost, one fixed table for `load`

n_train 43,176 · n_holdout 1,056 · incumbent version 20260404_185533

| arm | n | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,056 | 706.7 | 6.8% | 970.3 | -98.9 | 10,273.7 | 0 |
| _constant_causal_ | 1,056 | 1,443.2 | 13.9% | 1,780.6 | -1,121.6 | 9,251.1 | 0 |
| _constant_oracle_ | 1,056 | 1,154.1 | 11.1% | 1,384.8 | -70.3 | 10,302.4 | 0 |
| _climatology_causal_ | 1,056 | 1,196.7 | 11.5% | 1,449.4 | -1,121.6 | 9,251.1 | 0 |
| _climatology_oracle_ | 1,056 | 668.5 | 6.4% | 861.6 | -64.8 | 10,307.8 | 0 |
| control@101 | 1,056 | 230.8 | 2.2% | 303.6 | -73.1 | 10,299.5 | 0 |
| control@103 | 1,056 | 240.7 | 2.3% | 317.0 | -94.9 | 10,277.7 | 0 |
| control@107 | 1,056 | 233.2 | 2.2% | 306.7 | -80.4 | 10,292.2 | 0 |
| control@109 | 1,056 | 234.6 | 2.3% | 308.4 | -85.5 | 10,287.2 | 0 |
| control@113 | 1,056 | 237.3 | 2.3% | 312.2 | -91.3 | 10,281.3 | 0 |
| control@127 | 1,056 | 237.7 | 2.3% | 312.6 | -92.4 | 10,280.2 | 0 |
| control@131 | 1,056 | 231.7 | 2.2% | 307.0 | -80.1 | 10,292.5 | 0 |
| control@137 | 1,056 | 238.4 | 2.3% | 314.9 | -83.6 | 10,289.0 | 0 |
| control_noholiday@101 | 1,056 | 273.7 | 2.6% | 379.1 | -32.8 | 10,339.9 | 0 |
| control_noholiday@103 | 1,056 | 265.2 | 2.6% | 374.2 | -27.8 | 10,344.9 | 0 |
| control_noholiday@107 | 1,056 | 266.7 | 2.6% | 370.8 | -26.4 | 10,346.3 | 0 |
| control_noholiday@109 | 1,056 | 265.9 | 2.6% | 370.8 | -26.6 | 10,346.0 | 0 |
| control_noholiday@113 | 1,056 | 267.0 | 2.6% | 373.9 | -29.7 | 10,342.9 | 0 |
| control_noholiday@127 | 1,056 | 269.4 | 2.6% | 377.1 | -37.8 | 10,334.8 | 0 |
| control_noholiday@131 | 1,056 | 262.7 | 2.5% | 368.9 | -27.5 | 10,345.1 | 0 |
| control_noholiday@137 | 1,056 | 267.6 | 2.6% | 375.5 | -32.5 | 10,340.1 | 0 |

Holiday subsets of the holdout (holiday 48 · holiday_affected 144 · ordinary 912). `holiday_affected` is a holiday, a bridge day, or within one day of a holiday — the rows these four features can distinguish from an ordinary day at all. A holiday effect that is real here is diluted by the row counts in the all-hours table above.

| arm | holiday MAE | holiday_affected MAE | ordinary MAE |
|---|---:|---:|---:|
| _seasonal-naive D-7_ | 818.6 | 522.6 | 735.8 |
| _constant_causal_ | 604.3 | 992.6 | 1,514.4 |
| _constant_oracle_ | 946.2 | 965.3 | 1,183.9 |
| _climatology_causal_ | 413.0 | 702.9 | 1,274.7 |
| _climatology_oracle_ | 923.9 | 506.3 | 694.2 |
| control@101 | 232.9 | 238.2 | 229.7 |
| control@103 | 267.4 | 250.4 | 239.2 |
| control@107 | 275.4 | 258.6 | 229.1 |
| control@109 | 255.4 | 247.4 | 232.6 |
| control@113 | 262.8 | 256.7 | 234.3 |
| control@127 | 278.2 | 261.5 | 233.9 |
| control@131 | 256.9 | 247.0 | 229.3 |
| control@137 | 286.6 | 262.8 | 234.5 |
| control_noholiday@101 | 694.0 | 433.7 | 248.5 |
| control_noholiday@103 | 731.8 | 437.5 | 238.0 |
| control_noholiday@107 | 710.1 | 431.9 | 240.6 |
| control_noholiday@109 | 724.0 | 435.9 | 239.1 |
| control_noholiday@113 | 691.3 | 448.3 | 238.4 |
| control_noholiday@127 | 724.3 | 450.1 | 240.9 |
| control_noholiday@131 | 717.5 | 442.2 | 234.3 |
| control_noholiday@137 | 707.5 | 443.5 | 239.9 |

ABL-337 night screen: not applicable to load.

## DE / load — catboost, one fixed table for `load`

n_train 43,296 · n_holdout 1,056 · incumbent version 20260404_185521

| arm | n | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,056 | 4,726.9 | 8.3% | 6,611.6 | -37.6 | 56,669.9 | 0 |
| _constant_causal_ | 1,056 | 7,707.1 | 13.6% | 9,460.1 | -2,515.6 | 54,191.8 | 0 |
| _constant_oracle_ | 1,056 | 7,600.5 | 13.4% | 9,163.9 | -901.1 | 55,806.4 | 0 |
| _climatology_causal_ | 1,056 | 5,840.1 | 10.3% | 7,148.6 | -2,515.6 | 54,191.8 | 0 |
| _climatology_oracle_ | 1,056 | 5,442.4 | 9.6% | 6,637.2 | -1,184.1 | 55,523.3 | 0 |
| control@101 | 1,056 | 1,188.3 | 2.1% | 1,598.6 | -265.2 | 56,442.3 | 0 |
| control@103 | 1,056 | 1,189.5 | 2.1% | 1,582.1 | -262.0 | 56,445.4 | 0 |
| control@107 | 1,056 | 1,170.4 | 2.1% | 1,583.5 | -173.7 | 56,533.7 | 0 |
| control@109 | 1,056 | 1,179.9 | 2.1% | 1,584.6 | -222.9 | 56,484.5 | 0 |
| control@113 | 1,056 | 1,147.5 | 2.0% | 1,533.1 | -208.3 | 56,499.1 | 0 |
| control@127 | 1,056 | 1,135.0 | 2.0% | 1,520.0 | -98.0 | 56,609.4 | 0 |
| control@131 | 1,056 | 1,193.5 | 2.1% | 1,580.1 | -249.6 | 56,457.8 | 0 |
| control@137 | 1,056 | 1,164.6 | 2.1% | 1,563.4 | -141.6 | 56,565.9 | 0 |
| control_noholiday@101 | 1,056 | 1,239.6 | 2.2% | 1,747.1 | -213.0 | 56,494.4 | 0 |
| control_noholiday@103 | 1,056 | 1,226.6 | 2.2% | 1,706.7 | -147.1 | 56,560.4 | 0 |
| control_noholiday@107 | 1,056 | 1,234.2 | 2.2% | 1,709.5 | -190.4 | 56,517.0 | 0 |
| control_noholiday@109 | 1,056 | 1,225.2 | 2.2% | 1,705.2 | -100.5 | 56,606.9 | 0 |
| control_noholiday@113 | 1,056 | 1,279.0 | 2.3% | 1,766.6 | -240.4 | 56,467.0 | 0 |
| control_noholiday@127 | 1,056 | 1,227.2 | 2.2% | 1,757.3 | -167.8 | 56,539.6 | 0 |
| control_noholiday@131 | 1,056 | 1,238.6 | 2.2% | 1,743.1 | -210.1 | 56,497.3 | 0 |
| control_noholiday@137 | 1,056 | 1,232.3 | 2.2% | 1,736.1 | -144.9 | 56,562.6 | 0 |

Holiday subsets of the holdout (holiday 72 · holiday_affected 168 · ordinary 888). `holiday_affected` is a holiday, a bridge day, or within one day of a holiday — the rows these four features can distinguish from an ordinary day at all. A holiday effect that is real here is diluted by the row counts in the all-hours table above.

| arm | holiday MAE | holiday_affected MAE | ordinary MAE |
|---|---:|---:|---:|
| _seasonal-naive D-7_ | 6,937.3 | 5,761.2 | 4,531.2 |
| _constant_causal_ | 5,826.5 | 5,491.7 | 8,126.2 |
| _constant_oracle_ | 7,288.3 | 6,456.1 | 7,817.0 |
| _climatology_causal_ | 5,755.9 | 4,567.2 | 6,080.9 |
| _climatology_oracle_ | 6,976.6 | 5,397.5 | 5,450.9 |
| control@101 | 1,639.3 | 1,442.4 | 1,140.2 |
| control@103 | 1,990.8 | 1,626.4 | 1,106.9 |
| control@107 | 1,538.1 | 1,408.0 | 1,125.4 |
| control@109 | 1,772.1 | 1,522.9 | 1,115.0 |
| control@113 | 1,805.9 | 1,528.0 | 1,075.6 |
| control@127 | 1,603.1 | 1,322.9 | 1,099.4 |
| control@131 | 2,114.5 | 1,700.4 | 1,097.6 |
| control@137 | 1,701.6 | 1,470.5 | 1,106.7 |
| control_noholiday@101 | 2,093.1 | 2,004.2 | 1,095.0 |
| control_noholiday@103 | 2,169.2 | 1,910.3 | 1,097.2 |
| control_noholiday@107 | 2,045.8 | 1,869.7 | 1,113.9 |
| control_noholiday@109 | 2,136.4 | 1,936.0 | 1,090.8 |
| control_noholiday@113 | 2,106.2 | 1,975.9 | 1,147.1 |
| control_noholiday@127 | 2,060.3 | 1,893.3 | 1,101.2 |
| control_noholiday@131 | 1,956.0 | 1,884.8 | 1,116.3 |
| control_noholiday@137 | 2,282.3 | 1,938.1 | 1,098.7 |

ABL-337 night screen: not applicable to load.

## FR / load — catboost, one fixed table for `load`

n_train 43,200 · n_holdout 1,031 · incumbent version 20260404_185525

| arm | n | MAE | WAPE | RMSE | bias | mean pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 1,031 | 8,005.8 | 12.8% | 9,788.2 | 73.8 | 62,572.2 | 0 |
| _constant_causal_ | 1,031 | 13,551.1 | 21.7% | 15,971.7 | -13,320.7 | 49,177.7 | 0 |
| _constant_oracle_ | 1,031 | 6,866.3 | 11.0% | 8,841.0 | -713.6 | 61,784.8 | 0 |
| _climatology_causal_ | 1,031 | 13,586.7 | 21.7% | 15,755.4 | -13,318.5 | 49,179.9 | 0 |
| _climatology_oracle_ | 1,031 | 6,338.8 | 10.1% | 8,396.7 | -593.3 | 61,905.1 | 0 |
| control@101 | 1,031 | 2,929.9 | 4.7% | 4,066.2 | 362.4 | 62,860.8 | 0 |
| control@103 | 1,031 | 2,895.9 | 4.6% | 3,992.4 | 384.2 | 62,882.6 | 0 |
| control@107 | 1,031 | 2,922.4 | 4.7% | 3,980.8 | 172.2 | 62,670.6 | 0 |
| control@109 | 1,031 | 2,969.7 | 4.8% | 4,022.4 | 200.5 | 62,698.9 | 0 |
| control@113 | 1,031 | 2,966.5 | 4.7% | 4,064.7 | 234.8 | 62,733.2 | 0 |
| control@127 | 1,031 | 2,935.2 | 4.7% | 4,040.1 | 249.4 | 62,747.8 | 0 |
| control@131 | 1,031 | 2,949.2 | 4.7% | 4,060.2 | 176.6 | 62,675.0 | 0 |
| control@137 | 1,031 | 2,900.3 | 4.6% | 4,007.7 | 341.4 | 62,839.8 | 0 |
| control_noholiday@101 | 1,031 | 2,907.2 | 4.7% | 3,981.8 | 265.3 | 62,763.7 | 0 |
| control_noholiday@103 | 1,031 | 2,841.2 | 4.5% | 3,891.3 | 261.8 | 62,760.2 | 0 |
| control_noholiday@107 | 1,031 | 2,868.5 | 4.6% | 3,918.7 | 308.8 | 62,807.2 | 0 |
| control_noholiday@109 | 1,031 | 2,878.9 | 4.6% | 3,945.7 | 307.4 | 62,805.8 | 0 |
| control_noholiday@113 | 1,031 | 2,919.2 | 4.7% | 4,007.7 | 375.9 | 62,874.3 | 0 |
| control_noholiday@127 | 1,031 | 2,936.3 | 4.7% | 3,996.3 | 231.2 | 62,729.6 | 0 |
| control_noholiday@131 | 1,031 | 2,841.6 | 4.5% | 3,914.5 | 313.4 | 62,811.8 | 0 |
| control_noholiday@137 | 1,031 | 2,857.6 | 4.6% | 3,939.2 | 215.6 | 62,714.0 | 0 |

Holiday subsets of the holdout (holiday 24 · holiday_affected 119 · ordinary 912). `holiday_affected` is a holiday, a bridge day, or within one day of a holiday — the rows these four features can distinguish from an ordinary day at all. A holiday effect that is real here is diluted by the row counts in the all-hours table above.

| arm | holiday MAE | holiday_affected MAE | ordinary MAE |
|---|---:|---:|---:|
| _seasonal-naive D-7_ | 3,528.3 | 6,527.5 | 8,198.7 |
| _constant_causal_ | 10,936.7 | 15,914.8 | 13,242.7 |
| _constant_oracle_ | 2,545.5 | 5,449.0 | 7,051.2 |
| _climatology_causal_ | 10,936.7 | 15,896.2 | 13,285.3 |
| _climatology_oracle_ | 2,394.5 | 4,546.2 | 6,572.7 |
| control@101 | 2,531.7 | 2,271.7 | 3,015.8 |
| control@103 | 2,884.6 | 2,243.4 | 2,981.1 |
| control@107 | 2,724.7 | 2,538.0 | 2,972.6 |
| control@109 | 2,710.2 | 2,430.0 | 3,040.1 |
| control@113 | 2,659.6 | 2,440.4 | 3,035.1 |
| control@127 | 2,679.2 | 2,390.2 | 3,006.3 |
| control@131 | 2,689.9 | 2,420.9 | 3,018.2 |
| control@137 | 2,689.8 | 2,452.7 | 2,958.7 |
| control_noholiday@101 | 3,356.1 | 2,316.3 | 2,984.3 |
| control_noholiday@103 | 3,290.3 | 2,250.2 | 2,918.3 |
| control_noholiday@107 | 3,194.4 | 2,293.0 | 2,943.6 |
| control_noholiday@109 | 3,315.8 | 2,219.8 | 2,964.9 |
| control_noholiday@113 | 3,590.6 | 2,476.1 | 2,977.1 |
| control_noholiday@127 | 3,242.1 | 2,699.6 | 2,967.2 |
| control_noholiday@131 | 3,018.3 | 2,234.1 | 2,920.9 |
| control_noholiday@137 | 3,061.5 | 2,355.2 | 2,923.1 |

ABL-337 night screen: not applicable to load.
