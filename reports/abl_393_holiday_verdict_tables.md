# ABL-393 verdict — do the four holiday features help load and price?

**load: HELP -> KEEP** · **price: NO_EFFECT -> EXCLUDE**

Registered rule: `experiments/ABL393/config.json`. Paired by seed over 8 seeds [101, 103, 107, 109, 113, 127, 131, 137]; `delta = 100 * (holidays - no_holidays) / no_holidays` on all-hours MAE, so **negative means the holiday features are better**. A cell is material at 7/8 seeds agreeing in sign (two-sided sign test p <= 0.0703).

## load/spring — **HELP**

sum(d) +8 over 8 cells, 8 material (8 favour keeping, 0 favour excluding)

| cell | n | MAE holidays (MW) | no holidays | paired delta (mean +- sd) | range | k/8 | sign p | own seed spread hol/nohol | d |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost/AT | 1,056 | 170.47 | 208.54 | -18.21% +- 3.90 | -23.19% .. -13.18% | 8/8 | 0.0078 | 8.94% / 4.57% | +1 |
| catboost/BE | 1,056 | 193.21 | 220.37 | -12.32% +- 1.18 | -14.17% .. -10.81% | 8/8 | 0.0078 | 3.78% / 1.37% | +1 |
| catboost/DE | 1,056 | 1,414.39 | 1,665.71 | -15.08% +- 1.41 | -16.74% .. -11.93% | 8/8 | 0.0078 | 4.51% / 1.95% | +1 |
| catboost/FR | 1,056 | 1,115.83 | 1,234.26 | -9.59% +- 2.15 | -11.74% .. -6.32% | 8/8 | 0.0078 | 5.74% / 4.53% | +1 |
| xgboost/AT | 1,056 | 162.12 | 189.83 | -14.56% +- 2.54 | -18.58% .. -11.35% | 8/8 | 0.0078 | 4.49% / 5.48% | +1 |
| xgboost/BE | 1,056 | 197.69 | 219.62 | -9.97% +- 1.74 | -13.47% .. -8.28% | 8/8 | 0.0078 | 4.25% / 5.11% | +1 |
| xgboost/DE | 1,056 | 1,328.93 | 1,601.16 | -16.99% +- 1.86 | -19.76% .. -14.34% | 8/8 | 0.0078 | 4.66% / 4.36% | +1 |
| xgboost/FR | 1,056 | 1,115.16 | 1,175.50 | -5.12% +- 1.82 | -6.65% .. -0.94% | 8/8 | 0.0078 | 4.02% / 4.50% | +1 |

### secondary (reported, never gating): `holiday` rows — would read **HELP**

sum(d) +8, 8 material. A holiday is a few days in a 44-day window, so this is where the mechanism lives and the all-hours table above is where it is diluted.

| cell | n | MAE holidays (MW) | no holidays | paired delta | k/8 | sign p |
|---|---:|---:|---:|---:|---:|---:|
| catboost/AT | 96 | 264.13 | 515.38 | -48.71% | 8/8 | 0.0078 |
| catboost/BE | 96 | 256.87 | 425.66 | -39.64% | 8/8 | 0.0078 |
| catboost/DE | 72 | 1,944.43 | 4,836.62 | -59.77% | 8/8 | 0.0078 |
| catboost/FR | 96 | 2,253.50 | 3,121.24 | -27.77% | 8/8 | 0.0078 |
| xgboost/AT | 96 | 254.56 | 479.54 | -46.81% | 8/8 | 0.0078 |
| xgboost/BE | 96 | 299.73 | 431.80 | -30.52% | 8/8 | 0.0078 |
| xgboost/DE | 72 | 2,253.06 | 5,337.12 | -57.77% | 8/8 | 0.0078 |
| xgboost/FR | 96 | 2,608.38 | 3,080.05 | -15.26% | 8/8 | 0.0078 |

### secondary (reported, never gating): `holiday_affected` rows — would read **HELP**

sum(d) +8, 8 material. A holiday is a few days in a 44-day window, so this is where the mechanism lives and the all-hours table above is where it is diluted.

| cell | n | MAE holidays (MW) | no holidays | paired delta | k/8 | sign p |
|---|---:|---:|---:|---:|---:|---:|
| catboost/AT | 288 | 226.35 | 319.00 | -29.00% | 8/8 | 0.0078 |
| catboost/BE | 240 | 234.66 | 327.46 | -28.32% | 8/8 | 0.0078 |
| catboost/DE | 216 | 1,520.28 | 2,681.95 | -43.30% | 8/8 | 0.0078 |
| catboost/FR | 288 | 1,594.32 | 1,972.39 | -19.13% | 8/8 | 0.0078 |
| xgboost/AT | 288 | 215.37 | 305.19 | -29.38% | 8/8 | 0.0078 |
| xgboost/BE | 240 | 257.13 | 328.88 | -21.80% | 8/8 | 0.0078 |
| xgboost/DE | 216 | 1,486.72 | 2,704.89 | -45.02% | 8/8 | 0.0078 |
| xgboost/FR | 288 | 1,725.48 | 1,943.01 | -11.18% | 8/8 | 0.0078 |

### Where the all-hours gain lands

Arithmetic on the two registered subsets, not a further test: `holiday_affected` and `ordinary` partition the holdout and MAE x n is a sum of absolute errors, so the two gains add to the total exactly. This is the internal check on the headline — if these four features are doing what the mechanism says, the gain has to land on the rows they can distinguish, and a gain spread evenly over ordinary rows would be an effect in search of an explanation.

| cell | holiday-affected share of rows | ...of the gain | total error saved (MAE x n, MW x h) |
|---|---:|---:|---:|
| catboost/AT | 27.3% | 66.4% | 40,197 |
| catboost/BE | 22.7% | 77.6% | 28,684 |
| catboost/DE | 20.5% | 94.5% | 265,389 |
| catboost/FR | 27.3% | 87.1% | 125,058 |
| xgboost/AT | 27.3% | 88.4% | 29,268 |
| xgboost/BE | 22.7% | 74.4% | 23,159 |
| xgboost/DE | 20.5% | 91.5% | 287,476 |
| xgboost/FR | 27.3% | 98.3% | 63,723 |

### What the fitted model is worth against no model at all

ABL-381/ABL-389. `constant_*` is a flat line (fit-window mean / gate-window median), `climatology_*` the same per hour of day. Positive skill means the holiday arm beats the free predictor, as a share of that predictor's own error. **Check each `n`** — a climatology is 24 levels and can be partially measurable, and two MAEs scored on different rows are not the same measurement.

| cell | holiday arm MAE | vs D-7 | vs constant_causal | vs constant_oracle | vs climatology_causal | vs climatology_oracle |
|---|---:|---:|---:|---:|---:|---:|
| catboost/AT | 170.47 | +54.6% | +83.2% | +79.3% | +79.9% | +67.9% |
| catboost/BE | 193.21 | +55.1% | +80.8% | +80.2% | +67.6% | +60.6% |
| catboost/DE | 1,414.39 | +59.1% | +81.1% | +77.3% | +77.2% | +68.9% |
| catboost/FR | 1,115.83 | +49.5% | +85.2% | +73.8% | +85.1% | +57.8% |
| xgboost/AT | 162.12 | +56.9% | +84.0% | +80.3% | +80.8% | +69.5% |
| xgboost/BE | 197.69 | +54.1% | +80.4% | +79.8% | +66.8% | +59.7% |
| xgboost/DE | 1,328.93 | +61.6% | +82.2% | +78.6% | +78.5% | +70.8% |
| xgboost/FR | 1,115.16 | +49.6% | +85.3% | +73.8% | +85.1% | +57.9% |

## price/spring — **NO_EFFECT**

sum(d) +2 over 8 cells, 2 material (2 favour keeping, 0 favour excluding)

| cell | n | MAE holidays (EUR/MWh) | no holidays | paired delta (mean +- sd) | range | k/8 | sign p | own seed spread hol/nohol | d |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost/AT | 1,056 | 19.33 | 20.09 | -3.74% +- 2.82 | -9.40% .. -0.23% | 8/8 | 0.0078 | 6.16% / 6.09% | +1 |
| catboost/BE | 1,056 | 19.15 | 19.49 | -1.72% +- 3.30 | -7.38% .. +2.17% | 6/8 | 0.2891 | 6.41% / 5.13% | +0 |
| catboost/DE | 1,056 | 24.55 | 24.20 | +1.49% +- 3.48 | -3.32% .. +6.41% | 3/8 | 0.7266 | 5.89% / 5.90% | +0 |
| catboost/FR | 1,056 | 19.61 | 19.92 | -1.50% +- 2.94 | -5.93% .. +3.30% | 6/8 | 0.2891 | 5.71% / 8.21% | +0 |
| xgboost/AT | 1,056 | 20.68 | 21.10 | -1.93% +- 3.69 | -6.71% .. +5.23% | 6/8 | 0.2891 | 7.37% / 5.76% | +0 |
| xgboost/BE | 1,056 | 21.09 | 21.35 | -1.16% +- 3.37 | -6.98% .. +4.23% | 5/8 | 0.7266 | 5.34% / 6.70% | +0 |
| xgboost/DE | 1,056 | 26.77 | 25.91 | +3.40% +- 3.88 | -3.40% .. +8.11% | 2/8 | 0.2891 | 6.26% / 6.72% | +0 |
| xgboost/FR | 1,056 | 18.71 | 19.41 | -3.58% +- 1.61 | -5.46% .. -1.71% | 8/8 | 0.0078 | 3.37% / 3.34% | +1 |

### secondary (reported, never gating): `holiday` rows — would read **MIXED**

sum(d) +0, 8 material. A holiday is a few days in a 44-day window, so this is where the mechanism lives and the all-hours table above is where it is diluted.

| cell | n | MAE holidays (EUR/MWh) | no holidays | paired delta | k/8 | sign p |
|---|---:|---:|---:|---:|---:|---:|
| catboost/AT | 96 | 31.62 | 36.34 | -12.90% | 8/8 | 0.0078 |
| catboost/BE | 96 | 34.30 | 32.92 | +4.21% | 1/8 | 0.0703 |
| catboost/DE | 72 | 46.20 | 44.38 | +4.11% | 0/8 | 0.0078 |
| catboost/FR | 96 | 29.32 | 32.88 | -10.67% | 8/8 | 0.0078 |
| xgboost/AT | 96 | 31.91 | 35.66 | -10.27% | 8/8 | 0.0078 |
| xgboost/BE | 96 | 38.03 | 36.09 | +5.53% | 1/8 | 0.0703 |
| xgboost/DE | 72 | 46.57 | 40.65 | +14.59% | 0/8 | 0.0078 |
| xgboost/FR | 96 | 28.11 | 33.09 | -15.02% | 8/8 | 0.0078 |

### secondary (reported, never gating): `holiday_affected` rows — would read **MIXED**

sum(d) +2, 6 material. A holiday is a few days in a 44-day window, so this is where the mechanism lives and the all-hours table above is where it is diluted.

| cell | n | MAE holidays (EUR/MWh) | no holidays | paired delta | k/8 | sign p |
|---|---:|---:|---:|---:|---:|---:|
| catboost/AT | 288 | 21.97 | 24.70 | -11.05% | 8/8 | 0.0078 |
| catboost/BE | 240 | 24.08 | 24.60 | -2.09% | 5/8 | 0.7266 |
| catboost/DE | 216 | 32.55 | 31.58 | +3.16% | 1/8 | 0.0703 |
| catboost/FR | 288 | 21.17 | 22.46 | -5.66% | 7/8 | 0.0703 |
| xgboost/AT | 288 | 24.02 | 25.64 | -6.18% | 7/8 | 0.0703 |
| xgboost/BE | 240 | 26.88 | 26.86 | +0.15% | 4/8 | 1.0000 |
| xgboost/DE | 216 | 36.41 | 33.45 | +8.88% | 0/8 | 0.0078 |
| xgboost/FR | 288 | 20.54 | 22.56 | -8.95% | 8/8 | 0.0078 |

### Where the all-hours gain lands

Arithmetic on the two registered subsets, not a further test: `holiday_affected` and `ordinary` partition the holdout and MAE x n is a sum of absolute errors, so the two gains add to the total exactly. This is the internal check on the headline — if these four features are doing what the mechanism says, the gain has to land on the rows they can distinguish, and a gain spread evenly over ordinary rows would be an effect in search of an explanation.

| cell | holiday-affected share of rows | ...of the gain | total error saved (MAE x n, EUR/MWh x h) |
|---|---:|---:|---:|
| catboost/AT | 27.3% | 98.2% | 802 |
| catboost/BE | 22.7% | 34.5% | 363 |
| catboost/DE | 20.5% | **net loss** | -367 |
| catboost/FR | 27.3% | 113.6% | 326 |
| xgboost/AT | 27.3% | 105.2% | 443 |
| xgboost/BE | 22.7% | -1.7% | 274 |
| xgboost/DE | 20.5% | **net loss** | -911 |
| xgboost/FR | 27.3% | 79.0% | 737 |

### What the fitted model is worth against no model at all

ABL-381/ABL-389. `constant_*` is a flat line (fit-window mean / gate-window median), `climatology_*` the same per hour of day. Positive skill means the holiday arm beats the free predictor, as a share of that predictor's own error. **Check each `n`** — a climatology is 24 levels and can be partially measurable, and two MAEs scored on different rows are not the same measurement.

| cell | holiday arm MAE | vs D-7 | vs constant_causal | vs constant_oracle | vs climatology_causal | vs climatology_oracle |
|---|---:|---:|---:|---:|---:|---:|
| catboost/AT | 19.33 | +40.2% | +58.5% | +55.5% | +55.8% | +25.9% |
| catboost/BE | 19.15 | +46.6% | +54.9% | +52.6% | +49.0% | +26.2% |
| catboost/DE | 24.55 | +30.2% | +44.5% | +43.2% | +36.2% | +8.0% |
| catboost/FR | 19.61 | +52.2% | +74.6% | +52.3% | +74.2% | +23.6% |
| xgboost/AT | 20.68 | +36.1% | +55.7% | +52.4% | +52.8% | +20.7% |
| xgboost/BE | 21.09 | +41.2% | +50.3% | +47.8% | +43.8% | +18.7% |
| xgboost/DE | 26.77 | +23.8% | +39.5% | +38.1% | +30.4% | -0.4% |
| xgboost/FR | 18.71 | +54.4% | +75.7% | +54.5% | +75.4% | +27.1% |

**A model-free predictor chosen with hindsight beats the fitted model in 1 cell(s) here.** That changes nothing above — this issue has no gate and the holiday effect is measured within the model, not against these — but it bounds what the cell is worth:

- xgboost/DE: holiday arm 26.77 vs `climatology_oracle` 26.67 EUR/MWh (-0.4%)

## load/winter — **HELP**

sum(d) +4 over 8 cells, 8 material (6 favour keeping, 2 favour excluding) · *registered replication: can weaken the load verdict, never create it*

| cell | n | MAE holidays (MW) | no holidays | paired delta (mean +- sd) | range | k/8 | sign p | own seed spread hol/nohol | d |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost/AT | 1,056 | 244.82 | 291.00 | -15.87% +- 2.03 | -19.07% .. -13.06% | 8/8 | 0.0078 | 9.06% / 5.96% | +1 |
| catboost/BE | 1,056 | 235.55 | 267.28 | -11.86% +- 1.83 | -15.67% .. -9.23% | 8/8 | 0.0078 | 4.20% / 4.13% | +1 |
| catboost/DE | 1,056 | 1,171.09 | 1,237.84 | -5.37% +- 2.44 | -10.28% .. -3.02% | 8/8 | 0.0078 | 5.00% / 4.34% | +1 |
| catboost/FR | 1,031 | 2,933.65 | 2,881.32 | +1.83% +- 1.21 | -0.04% .. +3.79% | 1/8 | 0.0703 | 2.52% / 3.30% | -1 |
| xgboost/AT | 1,056 | 265.42 | 292.61 | -9.28% +- 1.47 | -11.21% .. -6.60% | 8/8 | 0.0078 | 3.73% / 3.43% | +1 |
| xgboost/BE | 1,056 | 256.77 | 277.65 | -7.52% +- 1.07 | -9.06% .. -6.09% | 8/8 | 0.0078 | 3.21% / 3.12% | +1 |
| xgboost/DE | 1,056 | 1,333.27 | 1,398.64 | -4.65% +- 2.43 | -7.68% .. -0.74% | 8/8 | 0.0078 | 4.91% / 5.05% | +1 |
| xgboost/FR | 1,031 | 3,102.04 | 3,075.10 | +0.90% +- 2.22 | -4.16% .. +2.79% | 1/8 | 0.0703 | 4.09% / 4.54% | -1 |

### secondary (reported, never gating): `holiday` rows — would read **HELP**

sum(d) +6, 8 material. A holiday is a few days in a 44-day window, so this is where the mechanism lives and the all-hours table above is where it is diluted.

| cell | n | MAE holidays (MW) | no holidays | paired delta | k/8 | sign p |
|---|---:|---:|---:|---:|---:|---:|
| catboost/AT | 120 | 262.72 | 587.59 | -55.18% | 8/8 | 0.0078 |
| catboost/BE | 48 | 264.44 | 712.57 | -62.89% | 8/8 | 0.0078 |
| catboost/DE | 72 | 1,770.68 | 2,106.16 | -15.70% | 7/8 | 0.0703 |
| catboost/FR | 24 | 2,696.22 | 3,258.62 | -17.02% | 8/8 | 0.0078 |
| xgboost/AT | 120 | 386.09 | 574.41 | -32.81% | 8/8 | 0.0078 |
| xgboost/BE | 48 | 344.58 | 636.29 | -45.82% | 8/8 | 0.0078 |
| xgboost/DE | 72 | 1,245.31 | 1,813.36 | -31.32% | 8/8 | 0.0078 |
| xgboost/FR | 24 | 3,617.76 | 3,336.89 | +9.08% | 1/8 | 0.0703 |

### secondary (reported, never gating): `holiday_affected` rows — would read **HELP**

sum(d) +5, 7 material. A holiday is a few days in a 44-day window, so this is where the mechanism lives and the all-hours table above is where it is diluted.

| cell | n | MAE holidays (MW) | no holidays | paired delta | k/8 | sign p |
|---|---:|---:|---:|---:|---:|---:|
| catboost/AT | 312 | 345.13 | 505.44 | -31.70% | 8/8 | 0.0078 |
| catboost/BE | 144 | 252.83 | 440.39 | -42.59% | 8/8 | 0.0078 |
| catboost/DE | 168 | 1,502.70 | 1,926.54 | -21.95% | 8/8 | 0.0078 |
| catboost/FR | 119 | 2,398.41 | 2,355.53 | +2.19% | 4/8 | 1.0000 |
| xgboost/AT | 312 | 397.35 | 499.38 | -20.42% | 8/8 | 0.0078 |
| xgboost/BE | 144 | 298.30 | 442.42 | -32.53% | 8/8 | 0.0078 |
| xgboost/DE | 168 | 1,366.61 | 1,853.08 | -26.15% | 8/8 | 0.0078 |
| xgboost/FR | 119 | 2,828.69 | 2,611.29 | +8.44% | 0/8 | 0.0078 |

### Where the all-hours gain lands

Arithmetic on the two registered subsets, not a further test: `holiday_affected` and `ordinary` partition the holdout and MAE x n is a sum of absolute errors, so the two gains add to the total exactly. This is the internal check on the headline — if these four features are doing what the mechanism says, the gain has to land on the rows they can distinguish, and a gain spread evenly over ordinary rows would be an effect in search of an explanation.

| cell | holiday-affected share of rows | ...of the gain | total error saved (MAE x n, MW x h) |
|---|---:|---:|---:|
| catboost/AT | 29.5% | 102.6% | 48,764 |
| catboost/BE | 13.6% | 80.6% | 33,509 |
| catboost/DE | 15.9% | 101.0% | 70,479 |
| catboost/FR | 11.5% | **net loss** | -53,954 |
| xgboost/AT | 29.5% | 110.9% | 28,710 |
| xgboost/BE | 13.6% | 94.1% | 22,050 |
| xgboost/DE | 15.9% | 118.4% | 69,028 |
| xgboost/FR | 11.5% | **net loss** | -27,773 |

### What the fitted model is worth against no model at all

ABL-381/ABL-389. `constant_*` is a flat line (fit-window mean / gate-window median), `climatology_*` the same per hour of day. Positive skill means the holiday arm beats the free predictor, as a share of that predictor's own error. **Check each `n`** — a climatology is 24 levels and can be partially measurable, and two MAEs scored on different rows are not the same measurement.

| cell | holiday arm MAE | vs D-7 | vs constant_causal | vs constant_oracle | vs climatology_causal | vs climatology_oracle |
|---|---:|---:|---:|---:|---:|---:|
| catboost/AT | 244.82 | +54.7% | +79.0% | +74.8% | +76.3% | +62.2% |
| catboost/BE | 235.55 | +66.7% | +83.7% | +79.6% | +80.3% | +64.8% |
| catboost/DE | 1,171.09 | +75.2% | +84.8% | +84.6% | +79.9% | +78.5% |
| catboost/FR | 2,933.65 | +63.4% | +78.4% | +57.3% | +78.4% | +53.7% |
| xgboost/AT | 265.42 | +50.9% | +77.2% | +72.7% | +74.3% | +59.0% |
| xgboost/BE | 256.77 | +63.7% | +82.2% | +77.8% | +78.5% | +61.6% |
| xgboost/DE | 1,333.27 | +71.8% | +82.7% | +82.5% | +77.2% | +75.5% |
| xgboost/FR | 3,102.04 | +61.3% | +77.1% | +54.8% | +77.2% | +51.1% |

## Type verdicts

- **load**: spring **HELP**, winter replication **HELP** — no direction disagreement, spring stands. Verdict **HELP** -> **KEEP**.
- **price**: spring **NO_EFFECT** -> **EXCLUDE**. Winter was not registered for price: AT and DE are 67.3% covered there behind 1,651 h and 1,309 h ingest holes.

_load: Registered mapping. get_feature_columns() should keep the four names for this type. THIS IS A FINDING AND NOT A PROMOTION: no serving-registry change and no retrain follows from this issue - that is the CEO's decision. Report the per-country size, and that 24 countries of this type serve without them today._

_price: Registered mapping's action, unchanged: exclude the four names from the PRICE list on parsimony. The mapping's second clause is about src/features.py's 'high impact for LOAD forecasting' comment and does not apply here - load read HELP, so that comment is vindicated, not refuted._
