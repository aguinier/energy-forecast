# ABL-410 — renewable-family scorecard under each candidate truth

Generated: 2026-08-13 20:46 UTC
Replica: `C:\Code\able\data\energy_dashboard.db` (9,432,453,120 bytes), SQLite `mode=ro`.
Sidecar: `C:\Code\able\data\forecasts_local.db`
Target window: 2026-07-11 00:00:00 -> 2026-08-10 00:00:00 (exclusive).
Selection: the scorecard's own — latest vintage per country + target + model + horizon band, top-of-hour actuals only, no aggregation.
Models: the `PRODUCTION_MODELS` registry snapshot, renewable family only.
All figures out-of-sample with respect to the stored forecasts (they were issued before their target); the *models* were fitted on windows this run does not know, so no in-sample claim is made either way.

## Pooled WAPE by truth — common instants only

Restricted to the (country, target) pairs both tables carry, so the three columns score the identical sample. This is the comparable table.

| type | model | country | n frozen | WAPE frozen strict | WAPE frozen null-aware | n generation | WAPE generation | mean actual frozen | mean actual generation |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| renewable | catboost | BE | 2,760 | 41.57% | 41.57% | 2,760 | 50.94% | 3628.27 | 3379.67 |
| renewable | catboost | DE | 2,760 | 47.34% | 47.34% | 2,760 | 49.55% | 35046.46 | 33772.01 |
| renewable | catboost | FR | 1,688 | 26.04% | 26.04% | 1,688 | 32.82% | 14292.89 | 13497.81 |
| renewable | catboost | (pooled) | 7,208 | 42.97% | 42.97% | 7,208 | 46.61% | 18156.03 | 17386.65 |
| solar | catboost | BE | 2,760 | 23.42% | 23.42% | 2,760 | 23.42% | 2258.63 | 2258.63 |
| solar | catboost | DE | 2,760 | 62.37% | 62.37% | 2,760 | 62.37% | 17619.14 | 17619.14 |
| solar | catboost | FR | 1,688 | 21.49% | 21.49% | 1,688 | 21.49% | 6523.05 | 6523.05 |
| solar | catboost | (pooled) | 7,208 | 51.85% | 51.85% | 7,208 | 51.85% | 9138.95 | 9138.95 |
| wind_onshore | catboost | BE | 2,760 | 192.95% | 192.95% | 2,760 | 192.95% | 450.01 | 450.01 |
| wind_onshore | catboost | DE | 2,760 | 61.48% | 61.48% | 2,760 | 61.48% | 8419.06 | 8419.06 |
| wind_onshore | catboost | FR | 1,688 | 127.14% | 127.14% | 1,688 | 127.14% | 2807.05 | 2807.05 |
| wind_onshore | catboost | (pooled) | 7,208 | 77.72% | 77.72% | 7,208 | 77.72% | 4053.40 | 4053.40 |
| wind_offshore | xgboost | BE | 2,760 | 162.08% | 162.08% | 2,760 | 161.58% | 466.67 | 462.40 |
| wind_offshore | xgboost | FR | 1,688 | 108.66% | 108.66% | 1,688 | 108.66% | 425.04 | 425.01 |
| wind_offshore | xgboost | (pooled) | 4,448 | 142.97% | 142.97% | 4,448 | 142.75% | 450.87 | 448.21 |
| biomass | xgboost | BE | 2,760 | 69.39% | 69.39% | 2,760 | 69.76% | 202.12 | 207.38 |
| biomass | xgboost | FR | 1,688 | 3.89% | 3.89% | 1,688 | 3.89% | 284.67 | 284.67 |
| biomass | xgboost | (pooled) | 4,448 | 39.08% | 39.08% | 4,448 | 39.70% | 233.45 | 236.71 |
| hydro_total | xgboost | BE | 2,760 | 92.21% | 92.21% | 2,760 | 14274.51% | 145.66 | 1.26 |
| hydro_total | xgboost | FR | 1,688 | 15.00% | 15.00% | 1,688 | 28.70% | 4193.36 | 3458.02 |
| hydro_total | xgboost | (pooled) | 4,448 | 19.15% | 19.15% | 4,448 | 37.18% | 1681.75 | 1313.09 |

## Pooled WAPE by truth — every instant each table carries

Each truth on its own coverage — this is what each surface actually publishes, and the n columns are the reason the two are not the same measurement.

| type | model | country | n frozen | WAPE frozen strict | WAPE frozen null-aware | n generation | WAPE generation | mean actual frozen | mean actual generation |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| renewable | catboost | BE | 2,760 | 41.57% | 41.57% | 2,760 | 50.94% | 3628.27 | 3379.67 |
| renewable | catboost | DE | 2,760 | 47.34% | 47.34% | 2,760 | 49.55% | 35046.46 | 33772.01 |
| renewable | catboost | FR | 2,760 | 22.19% | 22.19% | 1,688 | 32.82% | 15285.56 | 13497.81 |
| renewable | catboost | (pooled) | 8,280 | 39.82% | 39.82% | 7,208 | 46.61% | 17986.77 | 17386.65 |
| solar | catboost | BE | 2,760 | 23.42% | 23.42% | 2,760 | 23.42% | 2258.63 | 2258.63 |
| solar | catboost | DE | 2,760 | 62.37% | 62.37% | 2,760 | 62.37% | 17619.14 | 17619.14 |
| solar | catboost | FR | 2,760 | 19.89% | 19.89% | 1,688 | 21.49% | 6699.35 | 6523.05 |
| solar | catboost | (pooled) | 8,280 | 48.35% | 48.35% | 7,208 | 51.85% | 8859.04 | 9138.95 |
| wind_onshore | catboost | BE | 2,760 | 192.95% | 192.95% | 2,760 | 192.95% | 450.01 | 450.01 |
| wind_onshore | catboost | DE | 2,760 | 61.48% | 61.48% | 2,760 | 61.48% | 8419.06 | 8419.06 |
| wind_onshore | catboost | FR | 2,760 | 100.37% | 100.37% | 1,688 | 127.14% | 3329.30 | 2807.05 |
| wind_onshore | catboost | (pooled) | 8,280 | 76.94% | 76.94% | 7,208 | 77.72% | 4066.12 | 4053.40 |
| wind_offshore | xgboost | BE | 2,760 | 162.08% | 162.08% | 2,760 | 161.58% | 466.67 | 462.40 |
| wind_offshore | xgboost | FR | 2,760 | 77.74% | 77.74% | 1,688 | 108.66% | 569.99 | 425.01 |
| wind_offshore | xgboost | (pooled) | 5,520 | 115.71% | 115.71% | 4,448 | 142.75% | 518.33 | 448.21 |
| biomass | xgboost | BE | 2,760 | 69.39% | 69.39% | 2,760 | 69.76% | 202.12 | 207.38 |
| biomass | xgboost | FR | 2,760 | 4.14% | 4.14% | 1,688 | 3.89% | 283.26 | 284.67 |
| biomass | xgboost | (pooled) | 5,520 | 31.31% | 31.31% | 4,448 | 39.70% | 242.69 | 236.71 |
| hydro_total | xgboost | BE | 2,760 | 92.21% | 92.21% | 2,760 | 14274.51% | 145.66 | 1.26 |
| hydro_total | xgboost | FR | 2,760 | 15.94% | 15.94% | 1,688 | 28.70% | 4344.45 | 3458.02 |
| hydro_total | xgboost | (pooled) | 5,520 | 18.42% | 18.42% | 4,448 | 37.18% | 2245.06 | 1313.09 |


## Hydro, leg by leg

The same stored `hydro_total` forecast scored against each component series on its own, on the common instants. `slope` and `corr` are the model against that leg: a model that tracks the store rather than the river shows it here.

| country | leg | n | mean MW | WAPE | slope | corr |
|---|---|---:|---:|---:|---:|---:|
| BE | frozen run | 2,760 | 0.98 | 18333.94% | -7.712 | -0.086 |
| BE | frozen reservoir | 2,760 | 144.68 | 93.24% | 0.498 | 0.845 |
| BE | generation run | 2,760 | 1.26 | 14274.51% | -9.024 | -0.124 |
| BE | generation reservoir | 0 | n/a | not measured | n/a | n/a |
| BE | generation pumped | 2,760 | -74.14 | 97.55% | 0.242 | 0.722 |
| FR | frozen run | 1,688 | 2326.02 | 89.89% | 3.810 | 0.720 |
| FR | frozen reservoir | 1,688 | 1867.34 | 136.53% | 0.913 | 0.884 |
| FR | generation run | 1,688 | 2326.02 | 89.89% | 3.810 | 0.720 |
| FR | generation reservoir | 1,688 | 1132.00 | 290.18% | 2.045 | 0.872 |
| FR | generation pumped | 1,688 | 6.39 | 337.74% | 0.829 | 0.838 |

## Vintage counts

| forecast / model | generated timestamps | run-days |
|---|---:|---:|
| renewable/catboost | 720 | 31 |
| solar/catboost | 720 | 31 |
| wind_onshore/catboost | 720 | 31 |
| wind_offshore/xgboost | 480 | 31 |
| biomass/xgboost | 480 | 31 |
| hydro_total/xgboost | 480 | 31 |
