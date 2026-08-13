# ABL-386 verdict: **MIXED** -> recommend **REPORT**

## primary: `geometry` (31) vs `geometry_noholiday` (27)

verdict **MIXED** - sum(d) -1 over 8 cells, 3 disjoint (1 favour keeping, 2 favour excluding)

| cell | daylight MAE, holidays (31/29) | no holidays (27/25) | effect | spread hol / nohol | ranges | d |
|---|---:|---:|---:|---:|---|---:|
| catboost/AT | 618.9 (589.1-637.5) | 596.9 (568.9-620.0) | +3.70% | 7.82% / 8.56% | overlapping | +0 |
| catboost/BE | 572.2 (561.6-581.9) | 552.0 (549.7-556.4) | +3.66% | 3.55% / 1.23% | **disjoint** | -1 |
| catboost/DE | 4,449.7 (4,224.5-4,838.2) | 4,330.5 (3,999.7-4,524.6) | +2.75% | 13.79% / 12.12% | overlapping | +0 |
| catboost/FR | 1,001.6 (996.6-1,004.8) | 983.5 (977.6-989.1) | +1.84% | 0.82% / 1.17% | **disjoint** | -1 |
| xgboost/AT | 496.2 (494.0-499.9) | 505.0 (502.5-510.1) | -1.75% | 1.18% / 1.50% | **disjoint** | +1 |
| xgboost/BE | 606.1 (597.5-622.1) | 604.3 (587.8-619.2) | +0.31% | 4.05% / 5.20% | overlapping | +0 |
| xgboost/DE | 4,249.6 (4,161.8-4,362.6) | 4,307.8 (4,209.5-4,381.6) | -1.35% | 4.73% / 3.99% | overlapping | +0 |
| xgboost/FR | 974.8 (961.1-997.9) | 974.3 (959.8-987.0) | +0.05% | 3.77% / 2.78% | overlapping | +0 |

## replicate: `control` (29) vs `control_noholiday` (25)

verdict **NO_EFFECT** - sum(d) -1 over 8 cells, 1 disjoint (0 favour keeping, 1 favour excluding)

| cell | daylight MAE, holidays (31/29) | no holidays (27/25) | effect | spread hol / nohol | ranges | d |
|---|---:|---:|---:|---:|---|---:|
| catboost/AT | 653.9 (647.2-660.0) | 679.5 (627.2-720.1) | -3.77% | 1.95% / 13.67% | overlapping | +0 |
| catboost/BE | 563.0 (552.0-576.9) | 558.2 (546.9-564.4) | +0.86% | 4.42% / 3.14% | overlapping | +0 |
| catboost/DE | 4,463.5 (4,131.5-4,815.6) | 4,060.5 (3,758.1-4,278.8) | +9.93% | 15.33% / 12.82% | overlapping | +0 |
| catboost/FR | 1,018.7 (1,000.5-1,034.1) | 1,017.1 (1,002.5-1,025.1) | +0.16% | 3.29% / 2.22% | overlapping | +0 |
| xgboost/AT | 517.8 (514.2-524.7) | 527.0 (523.0-530.2) | -1.74% | 2.03% / 1.38% | overlapping | +0 |
| xgboost/BE | 635.2 (627.2-643.1) | 613.5 (604.3-624.2) | +3.54% | 2.50% / 3.24% | **disjoint** | -1 |
| xgboost/DE | 3,914.6 (3,856.8-3,958.1) | 3,969.6 (3,909.1-4,032.2) | -1.39% | 2.59% / 3.10% | overlapping | +0 |
| xgboost/FR | 980.7 (979.8-982.4) | 988.3 (975.9-1,010.7) | -0.76% | 0.26% / 3.52% | overlapping | +0 |

## Night guardrail (registered as |night mean|, MW)

| cell | holidays | no holidays | change from excluding | nohol seed spread | pass |
|---|---:|---:|---:|---:|---|
| catboost/AT | 26.1 | 28.4 | +2.3 | 4.8 | PASS |
| catboost/BE | 11.6 | 12.1 | +0.5 | 1.7 | PASS |
| catboost/DE | 230.7 | 287.5 | +56.7 | 39.6 | **FAIL** |
| catboost/FR | 28.9 | 36.4 | +7.5 | 4.0 | **FAIL** |
| xgboost/AT | 4.1 | 3.6 | -0.5 | 2.3 | PASS |
| xgboost/BE | 0.8 | 0.8 | -0.0 | 1.7 | PASS |
| xgboost/DE | 44.5 | 39.8 | -4.8 | 18.8 | PASS |
| xgboost/FR | 37.2 | 32.5 | -4.7 | 21.8 | PASS |

_Registered mapping: recommend exclusion on parsimony only if no cell shows disjoint HELP; otherwise name what a further read must measure._
