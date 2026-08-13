# Forecast quality scorecard

Generated: 2026-08-13 20:47 UTC
Target window: 2026-07-11 00:00:00 → 2026-08-10 00:00:00 (exclusive)
Sample: 158,907 selected forecast rows; 152,229 paired actual rows
Selection: latest vintage per country + target + model + horizon band
Load actual guard: `load_mw > 0 (load only)`
Net-position gate: `src/evaluation/net_position.py (not duplicated here)`

## Vintage counts

| forecast / model | generated timestamps | run-days |
|---|---:|---:|
| load/catboost | 5,040 | 31 |
| price/catboost | 4,560 | 31 |
| renewable/catboost | 720 | 31 |
| solar/catboost | 720 | 31 |
| wind_onshore/catboost | 720 | 31 |
| wind_offshore/xgboost | 480 | 31 |
| biomass/xgboost | 480 | 31 |
| hydro_total/xgboost | 480 | 31 |
| net_position/chronos-2-V010 | 18 | 15 |

## ABL-128 probe reproduction

The direction of the CEO probe reproduces. Exact values below are the current replica under the explicit latest-per-band rule; differences are findings, not adjusted away.

| type | reference WAPE | measured WAPE | reference D−7 | measured D−7 | reference bias | measured bias |
|---|---:|---:|---:|---:|---:|---:|
| load | 9.4% | 9.3% | 5.9% | 5.9% | 1.6% | 2.1% |
| price | 33.2% | 34.3% | 27.3% | 27.8% | -17.3% | -16.0% |
| solar | 53.6% | 51.8% | 26.0% | 25.5% | -50.7% | -47.6% |
| wind_onshore | 73.5% | 77.7% | 75.0% | 76.0% | 28.1% | 35.7% |

Load reproduces within 0.1 percentage point on WAPE/D−7 and 0.1 point on TSO (reference 4.0%, measured 4.1%). Price, solar, and wind do not reproduce exactly; the scorecard preserves the disagreement.

## Scoring truth

Which statement of the actual each type is scored against. Since ABL-410 this is the same table the dashboard publishes against, so one model and window has one WAPE across both surfaces. Training source is a separate, unchanged decision (`db.RENEWABLE_TYPE_SOURCE_TABLE`, still `energy_renewable`); where the two disagree about the target, part of the WAPE below is target mismatch rather than model error.

| type | table | value |
|---|---|---|
| load | `energy_load` | `load_mw` |
| price | `energy_price` | `price_eur_mwh` |
| renewable | `energy_generation` | `CASE WHEN solar_mw IS NULL AND wind_onshore_mw IS NULL AND wind_offshore_mw IS NULL AND hydro_run_mw IS NULL AND hydro_reservoir_mw IS NULL AND biomass_mw IS NULL AND geothermal_mw IS NULL AND marine_mw IS NULL AND other_renewable_mw IS NULL THEN NULL ELSE COALESCE(solar_mw, 0) + COALESCE(wind_onshore_mw, 0) + COALESCE(wind_offshore_mw, 0) + COALESCE(hydro_run_mw, 0) + COALESCE(hydro_reservoir_mw, 0) + COALESCE(biomass_mw, 0) + COALESCE(geothermal_mw, 0) + COALESCE(marine_mw, 0) + COALESCE(other_renewable_mw, 0) END` |
| solar | `energy_generation` | `solar_mw` |
| wind_onshore | `energy_generation` | `wind_onshore_mw` |
| wind_offshore | `energy_generation` | `wind_offshore_mw` |
| biomass | `energy_generation` | `biomass_mw` |
| hydro_total | `energy_generation` | `CASE WHEN hydro_run_mw IS NULL AND hydro_reservoir_mw IS NULL THEN NULL ELSE COALESCE(hydro_run_mw, 0) + COALESCE(hydro_reservoir_mw, 0) END` |
| net_position | `net_position` | `net_position_mw` |

## Pooled score

Skill is `100 × (1 − model WAPE / baseline WAPE)` on the exact same pairs. `mean actual` is the level of WAPE's own denominator over the scored pairs; a percentage against a near-zero mean is arithmetic, not quality.

| type | model | horizon | n | mean actual | WAPE | MAE | bias | slope | corr | D−7 WAPE / skill | persistence WAPE / skill | TSO WAPE / skill |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| load | catboost | 12-24h | 12,595 | 10740.2 | 8.1% | 866.0 | 2.5% | 1.0 | 1.0 | 6.0% / -34.3% | 14.4% / 43.9% | 4.2% / -92.4% |
| load | catboost | 2-12h | 5,037 | 9832.2 | 6.7% | 656.5 | 2.0% | 1.0 | 1.0 | 4.7% / -40.6% | 21.2% / 68.6% | 3.2% / -108.6% |
| load | catboost | 24-36h | 15,113 | 10687.9 | 9.6% | 1029.9 | 2.0% | 1.0 | 1.0 | 5.9% / -63.8% | 14.9% / 35.4% | 4.0% / -138.6% |
| load | catboost | 36-48h | 15,113 | 10687.9 | 9.7% | 1039.3 | 2.0% | 1.0 | 1.0 | 5.9% / -65.3% | 17.2% / 43.4% | 4.0% / -140.8% |
| load | catboost | 48-64h | 10,076 | 11115.7 | 10.8% | 1201.1 | 1.8% | 1.0 | 1.0 | 6.4% / -69.3% | 15.1% / 28.4% | 4.4% / -145.1% |
| load | catboost | all | 57,934 | 10699.3 | 9.3% | 994.0 | 2.1% | 1.0 | 1.0 | 5.9% / -57.3% | 15.9% / 41.7% | 4.1% / -128.2% |
| price | catboost | 12-24h | 11,400 | 95.6 | 33.7% | 32.2 | -13.6% | 0.7 | 0.8 | 29.8% / -13.0% | 76.7% / 56.1% | Not measured / Not measured |
| price | catboost | 2-12h | 4,560 | 114.4 | 30.2% | 34.5 | -23.1% | 0.7 | 0.8 | 21.7% / -39.1% | 48.7% / 38.1% | Not measured / Not measured |
| price | catboost | 24-36h | 13,680 | 103.0 | 33.3% | 34.3 | -16.8% | 0.6 | 0.8 | 27.3% / -22.0% | 55.1% / 39.6% | Not measured / Not measured |
| price | catboost | 36-48h | 13,680 | 103.0 | 35.4% | 36.5 | -16.5% | 0.6 | 0.8 | 27.3% / -29.7% | 71.2% / 50.2% | Not measured / Not measured |
| price | catboost | 48-64h | 9,120 | 97.4 | 37.5% | 36.5 | -12.9% | 0.6 | 0.8 | 30.6% / -22.5% | 60.4% / 37.9% | Not measured / Not measured |
| price | catboost | all | 52,440 | 101.4 | 34.3% | 34.9 | -16.0% | 0.6 | 0.8 | 27.8% / -23.4% | 64.1% / 46.4% | Not measured / Not measured |
| renewable | catboost | 12-24h | 1,565 | 18590.2 | 45.9% | 8526.1 | -21.9% | 0.3 | 0.7 | 20.1% / -133.3% | 72.2% / 36.5% | 16.4% / -179.0% |
| renewable | catboost | 2-12h | 624 | 11190.7 | 42.4% | 4745.7 | 14.7% | 0.6 | 0.8 | 30.9% / -32.8% | 34.1% / -24.2% | 28.2% / -50.2% |
| renewable | catboost | 24-36h | 1,881 | 17052.9 | 46.9% | 8003.8 | -15.8% | 0.3 | 0.7 | 21.7% / -119.2% | 39.7% / -18.6% | 18.2% / -158.5% |
| renewable | catboost | 36-48h | 1,881 | 17052.9 | 46.9% | 8004.7 | -15.7% | 0.3 | 0.7 | 21.7% / -119.0% | 76.4% / 38.4% | 18.2% / -158.5% |
| renewable | catboost | 48-64h | 1,257 | 19962.9 | 47.8% | 9548.1 | -24.3% | 0.3 | 0.7 | 19.1% / -156.4% | 40.4% / -20.1% | 15.4% / -211.5% |
| renewable | catboost | all | 7,208 | 17386.6 | 46.6% | 8104.7 | -17.2% | 0.3 | 0.7 | 21.3% / -121.9% | 56.5% / 17.1% | 17.8% / -162.5% |
| solar | catboost | 12-24h | 1,565 | 10519.9 | 51.6% | 5427.6 | -47.7% | 0.4 | 0.9 | 25.5% / -109.7% | 135.2% / 61.8% | 5.9% / -777.5% |
| solar | catboost | 2-12h | 624 | 2922.9 | 47.8% | 1396.5 | -34.9% | 0.4 | 0.9 | 29.6% / -62.4% | 100.4% / 52.4% | 7.2% / -564.7% |
| solar | catboost | 24-36h | 1,881 | 8756.0 | 51.9% | 4545.6 | -47.7% | 0.4 | 0.9 | 25.5% / -109.9% | 65.7% / 20.4% | 5.9% / -777.6% |
| solar | catboost | 36-48h | 1,881 | 8756.0 | 51.9% | 4548.4 | -47.7% | 0.4 | 0.9 | 25.5% / -110.0% | 159.7% / 67.2% | 5.9% / -778.1% |
| solar | catboost | 48-64h | 1,257 | 11651.6 | 52.5% | 6112.7 | -48.8% | 0.3 | 0.9 | 24.9% / -117.5% | 62.5% / 14.6% | 5.8% / -811.3% |
| solar | catboost | all | 7,208 | 9138.9 | 51.8% | 4738.5 | -47.6% | 0.4 | 0.9 | 25.5% / -110.0% | 107.0% / 51.1% | 5.9% / -777.8% |
| wind_onshore | catboost | 12-24h | 1,565 | 3949.2 | 80.0% | 3161.0 | 36.7% | 0.4 | 0.6 | 79.9% / 5.6% | 64.5% / -24.1% | 13.0% / -516.4% |
| wind_onshore | catboost | 2-12h | 624 | 4017.3 | 70.4% | 2829.8 | 33.3% | 0.5 | 0.6 | 64.3% / -5.9% | 42.9% / -64.3% | 10.8% / -551.2% |
| wind_onshore | catboost | 24-36h | 1,881 | 4082.3 | 78.1% | 3188.5 | 36.0% | 0.4 | 0.5 | 74.9% / 1.5% | 64.7% / -20.1% | 12.9% / -505.6% |
| wind_onshore | catboost | 36-48h | 1,881 | 4082.3 | 77.4% | 3159.2 | 35.4% | 0.4 | 0.6 | 74.9% / 2.4% | 78.9% / 2.5% | 12.9% / -500.1% |
| wind_onshore | catboost | 48-64h | 1,257 | 4114.6 | 78.4% | 3225.3 | 35.5% | 0.5 | 0.6 | 80.0% / 7.0% | 73.8% / -4.5% | 13.9% / -463.8% |
| wind_onshore | catboost | all | 7,208 | 4053.4 | 77.7% | 3150.2 | 35.7% | 0.5 | 0.6 | 76.0% / 3.1% | 68.1% / -13.5% | 12.9% / -501.8% |
| wind_offshore | xgboost | 12-24h | 965 | 415.6 | 148.1% | 623.4 | 114.0% | 0.2 | 0.2 | 98.1% / -52.1% | 92.3% / -61.3% | 29.3% / -406.2% |
| wind_offshore | xgboost | 2-12h | 384 | 418.0 | 147.1% | 620.9 | 122.5% | 0.1 | 0.1 | 89.0% / -82.3% | 88.9% / -65.4% | 24.5% / -499.1% |
| wind_offshore | xgboost | 24-36h | 1,161 | 457.2 | 139.1% | 643.2 | 99.3% | 0.0 | 0.0 | 97.4% / -44.3% | 99.2% / -40.7% | 27.4% / -408.2% |
| wind_offshore | xgboost | 36-48h | 1,161 | 457.2 | 139.4% | 644.5 | 99.0% | -0.0 | -0.0 | 97.4% / -45.1% | 114.8% / -22.2% | 27.4% / -409.2% |
| wind_offshore | xgboost | 48-64h | 777 | 476.6 | 145.2% | 700.1 | 88.8% | -0.1 | -0.1 | 100.8% / -45.4% | 104.6% / -37.8% | 28.6% / -407.7% |
| wind_offshore | xgboost | all | 4,448 | 448.2 | 142.8% | 647.3 | 102.1% | 0.0 | 0.0 | 97.6% / -49.0% | 102.1% / -40.2% | 27.8% / -414.4% |
| biomass | xgboost | 12-24h | 965 | 236.2 | 39.6% | 93.6 | -37.7% | 2.6 | 0.9 | 6.2% / -666.3% | 5.4% / -643.4% | Not measured / Not measured |
| biomass | xgboost | 2-12h | 384 | 238.8 | 39.7% | 94.8 | -38.5% | 2.6 | 1.0 | 6.0% / -692.7% | 4.3% / -820.4% | Not measured / Not measured |
| biomass | xgboost | 24-36h | 1,161 | 236.9 | 39.7% | 94.0 | -37.6% | 2.6 | 0.9 | 6.4% / -644.0% | 5.4% / -648.9% | Not measured / Not measured |
| biomass | xgboost | 36-48h | 1,161 | 236.9 | 39.7% | 94.1 | -37.6% | 2.6 | 0.9 | 6.4% / -644.5% | 6.3% / -557.6% | Not measured / Not measured |
| biomass | xgboost | 48-64h | 777 | 235.9 | 39.8% | 93.8 | -37.0% | 2.5 | 0.9 | 6.6% / -622.0% | 6.3% / -559.9% | Not measured / Not measured |
| biomass | xgboost | all | 4,448 | 236.7 | 39.7% | 94.0 | -37.6% | 2.6 | 0.9 | 6.4% / -648.9% | 5.7% / -616.6% | Not measured / Not measured |
| hydro_total | xgboost | 12-24h | 965 | 1253.4 | 35.5% | 445.1 | 34.6% | 1.2 | 1.0 | 13.5% / -205.9% | 43.5% / 17.6% | Not measured / Not measured |
| hydro_total | xgboost | 2-12h | 384 | 1290.9 | 26.3% | 338.9 | 24.9% | 1.1 | 1.0 | 15.3% / -90.2% | 49.8% / 47.3% | Not measured / Not measured |
| hydro_total | xgboost | 24-36h | 1,161 | 1329.6 | 37.7% | 501.7 | 36.7% | 1.3 | 1.0 | 13.3% / -226.1% | 36.2% / -7.7% | Not measured / Not measured |
| hydro_total | xgboost | 36-48h | 1,161 | 1329.6 | 37.7% | 501.4 | 36.7% | 1.3 | 1.0 | 13.3% / -223.5% | 41.3% / 4.4% | Not measured / Not measured |
| hydro_total | xgboost | 48-64h | 777 | 1348.8 | 42.7% | 576.1 | 41.9% | 1.3 | 1.0 | 12.3% / -289.1% | 30.1% / -51.7% | Not measured / Not measured |
| hydro_total | xgboost | all | 4,448 | 1313.1 | 37.2% | 488.3 | 36.2% | 1.2 | 1.0 | 13.3% / -218.5% | 39.3% / 2.2% | Not measured / Not measured |
| net_position | chronos-2-V010 | 24-36h | 90 | 461.7 | 41.1% | 674.8 | -6.1% | 0.7 | 0.9 | 37.0% / -11.1% | 43.1% / 4.5% | Not measured / Not measured |
| net_position | chronos-2-V010 | 36-48h | 2,166 | -68.7 | 79.9% | 1397.6 | 3.9% | 0.3 | 0.5 | 49.0% / -63.1% | 48.6% / -64.4% | Not measured / Not measured |
| net_position | chronos-2-V010 | 48-64h | 4,631 | 63.6 | 74.8% | 1338.5 | 3.3% | 0.4 | 0.6 | 56.7% / -32.1% | 57.7% / -29.7% | Not measured / Not measured |
| net_position | chronos-2-V010 | all | 6,887 | 27.2 | 76.0% | 1348.4 | 3.4% | 0.3 | 0.6 | 54.0% / -40.6% | 54.7% / -38.9% | Not measured / Not measured |

## Country × horizon detail

Rows with no paired observations say **Not measured**; zero is never substituted.

| type | model | country | horizon | n | mean actual | WAPE | bias | slope | corr | D−7 skill | persistence skill | TSO skill |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| load | catboost | BG | 12-24h | 600 | 3944.4 | 5.9% | -3.8% | 0.8 | 0.9 | 3.5% | 55.7% | -112.7% |
| load | catboost | BG | 2-12h | 240 | 3421.6 | 4.1% | -1.8% | 0.8 | 0.9 | -2.5% | 77.3% | -63.6% |
| load | catboost | BG | 24-36h | 720 | 3844.4 | 6.3% | -3.1% | 0.7 | 0.8 | -6.1% | 45.4% | -134.8% |
| load | catboost | BG | 36-48h | 720 | 3844.4 | 6.8% | -3.5% | 0.8 | 0.8 | -14.2% | 54.4% | -152.8% |
| load | catboost | BG | 48-64h | 480 | 4055.8 | 7.2% | -3.9% | 0.7 | 0.7 | -7.2% | 23.8% | -162.4% |
| load | catboost | CH | 12-24h | 600 | 6662.9 | 6.6% | -0.1% | 0.5 | 0.7 | 20.5% | 43.8% | -83.3% |
| load | catboost | CH | 2-12h | 240 | 6131.6 | 7.4% | 0.3% | 0.4 | 0.6 | 13.8% | 42.2% | -127.7% |
| load | catboost | CH | 24-36h | 720 | 6580.4 | 6.7% | -0.1% | 0.5 | 0.7 | 18.1% | 38.8% | -89.9% |
| load | catboost | CH | 36-48h | 720 | 6580.4 | 6.6% | -0.1% | 0.5 | 0.7 | 19.0% | 44.3% | -87.8% |
| load | catboost | CH | 48-64h | 480 | 6804.9 | 6.1% | -0.2% | 0.4 | 0.6 | 23.4% | 37.5% | -67.7% |
| load | catboost | CZ | 12-24h | 600 | 6406.8 | 6.8% | 0.4% | 0.7 | 0.9 | -56.3% | 64.3% | -363.8% |
| load | catboost | CZ | 2-12h | 240 | 5526.2 | 7.1% | 4.0% | 0.8 | 0.9 | -71.4% | 59.0% | -364.1% |
| load | catboost | CZ | 24-36h | 720 | 6235.5 | 8.9% | 0.8% | 0.7 | 0.8 | -103.2% | 29.8% | -495.0% |
| load | catboost | CZ | 36-48h | 720 | 6235.5 | 9.0% | 0.7% | 0.7 | 0.8 | -105.4% | 59.3% | -501.3% |
| load | catboost | CZ | 48-64h | 480 | 6590.2 | 9.5% | -0.6% | 0.6 | 0.7 | -109.9% | 27.9% | -534.3% |
| load | catboost | DE | 12-24h | 600 | 50340.0 | 7.8% | 2.7% | 0.6 | 0.8 | -82.0% | 36.5% | -104.1% |
| load | catboost | DE | 2-12h | 240 | 46635.0 | 7.5% | 3.7% | 0.8 | 0.8 | -83.9% | 50.6% | -102.8% |
| load | catboost | DE | 24-36h | 720 | 49596.9 | 10.7% | 2.9% | 0.4 | 0.5 | -142.8% | 21.4% | -183.0% |
| load | catboost | DE | 36-48h | 720 | 49596.9 | 10.8% | 2.8% | 0.4 | 0.5 | -145.7% | 39.3% | -186.4% |
| load | catboost | DE | 48-64h | 480 | 51077.8 | 11.9% | 2.0% | 0.2 | 0.2 | -161.4% | 26.0% | -212.1% |
| load | catboost | EE | 12-24h | 600 | 720.3 | 13.0% | 2.0% | 0.2 | 0.4 | 15.1% | 27.6% | 8.2% |
| load | catboost | EE | 2-12h | 240 | 688.1 | 10.4% | 2.9% | 0.6 | 0.6 | 12.3% | 27.5% | -5.8% |
| load | catboost | EE | 24-36h | 720 | 712.2 | 12.8% | 1.6% | 0.3 | 0.4 | 13.0% | 24.0% | 3.0% |
| load | catboost | EE | 36-48h | 720 | 712.2 | 12.5% | 1.6% | 0.3 | 0.4 | 14.6% | 34.6% | 4.7% |
| load | catboost | EE | 48-64h | 480 | 724.2 | 13.3% | 0.3% | 0.1 | 0.2 | 16.7% | 25.2% | 9.4% |
| load | catboost | ES | 12-24h | 600 | 30753.0 | 6.8% | 0.9% | 0.7 | 0.8 | -86.5% | 47.3% | -447.9% |
| load | catboost | ES | 2-12h | 240 | 27361.9 | 6.4% | 0.4% | 0.5 | 0.6 | -156.2% | 77.4% | -540.8% |
| load | catboost | ES | 24-36h | 720 | 30883.2 | 8.9% | 0.1% | 0.5 | 0.6 | -152.7% | 43.5% | -642.3% |
| load | catboost | ES | 36-48h | 720 | 30883.2 | 9.0% | 0.1% | 0.5 | 0.6 | -155.8% | 39.4% | -651.5% |
| load | catboost | ES | 48-64h | 480 | 32643.9 | 9.7% | -0.2% | 0.1 | 0.2 | -144.2% | 33.7% | -652.5% |
| load | catboost | FI | 12-24h | 599 | 8491.6 | 2.9% | -0.5% | 0.7 | 0.9 | 23.8% | 58.9% | -22.5% |
| load | catboost | FI | 2-12h | 239 | 7991.7 | 2.8% | 0.4% | 0.8 | 0.9 | 25.4% | 60.0% | -30.1% |
| load | catboost | FI | 24-36h | 717 | 8417.0 | 3.5% | -0.6% | 0.7 | 0.8 | 7.0% | 35.2% | -52.1% |
| load | catboost | FI | 36-48h | 717 | 8417.0 | 3.6% | -0.7% | 0.7 | 0.8 | 5.0% | 53.2% | -55.3% |
| load | catboost | FI | 48-64h | 478 | 8629.6 | 3.6% | -1.3% | 0.6 | 0.7 | 1.9% | 29.3% | -55.0% |
| load | catboost | GR | 12-24h | 600 | 7189.8 | 7.9% | 0.4% | 0.6 | 0.9 | 24.5% | 48.5% | -191.9% |
| load | catboost | GR | 2-12h | 240 | 6048.9 | 7.1% | 4.4% | 0.7 | 0.8 | 18.1% | 70.2% | -174.3% |
| load | catboost | GR | 24-36h | 720 | 7074.1 | 9.9% | 0.4% | 0.5 | 0.7 | 6.9% | 38.0% | -268.9% |
| load | catboost | GR | 36-48h | 720 | 7074.1 | 10.0% | 0.4% | 0.5 | 0.7 | 5.5% | 43.6% | -274.5% |
| load | catboost | GR | 48-64h | 480 | 7586.7 | 11.5% | -1.2% | 0.3 | 0.4 | -1.2% | 29.8% | -324.4% |
| load | catboost | HR | 12-24h | 600 | 2282.1 | 7.0% | -1.8% | 0.7 | 0.9 | 39.2% | 64.0% | -140.0% |
| load | catboost | HR | 2-12h | 240 | 1926.1 | 5.6% | -0.2% | 0.7 | 0.8 | 30.1% | 89.0% | -135.8% |
| load | catboost | HR | 24-36h | 720 | 2297.6 | 8.3% | -2.6% | 0.7 | 0.8 | 26.4% | 65.1% | -202.7% |
| load | catboost | HR | 36-48h | 720 | 2297.6 | 8.3% | -2.5% | 0.7 | 0.8 | 26.8% | 56.8% | -201.1% |
| load | catboost | HR | 48-64h | 480 | 2483.3 | 9.7% | -3.5% | 0.5 | 0.7 | 23.1% | 37.0% | -234.2% |
| load | catboost | HU | 12-24h | 600 | 4569.5 | 6.8% | 0.7% | 0.8 | 0.8 | 39.6% | 51.3% | -62.2% |
| load | catboost | HU | 2-12h | 240 | 4241.5 | 6.4% | -1.0% | 0.6 | 0.6 | 15.7% | 75.6% | -33.8% |
| load | catboost | HU | 24-36h | 720 | 4621.3 | 8.7% | 0.4% | 0.6 | 0.7 | 18.7% | 42.7% | -93.8% |
| load | catboost | HU | 36-48h | 720 | 4621.3 | 8.8% | 0.3% | 0.6 | 0.7 | 17.9% | 43.2% | -95.6% |
| load | catboost | HU | 48-64h | 480 | 4811.3 | 10.3% | 0.6% | 0.5 | 0.6 | 14.6% | 21.4% | -136.1% |
| load | catboost | IT | 12-24h | 600 | 38675.0 | 8.8% | 2.9% | 0.7 | 0.8 | -18.4% | 43.6% | -224.1% |
| load | catboost | IT | 2-12h | 240 | 33687.3 | 8.1% | 2.7% | 0.8 | 0.8 | -34.1% | 74.5% | -301.0% |
| load | catboost | IT | 24-36h | 720 | 38470.3 | 10.3% | 1.7% | 0.6 | 0.7 | -38.9% | 45.2% | -275.2% |
| load | catboost | IT | 36-48h | 720 | 38470.3 | 10.2% | 1.7% | 0.6 | 0.7 | -38.6% | 46.8% | -274.4% |
| load | catboost | IT | 48-64h | 480 | 40861.8 | 11.0% | 1.1% | 0.3 | 0.4 | -38.2% | 40.3% | -262.4% |
| load | catboost | LT | 12-24h | 600 | 1327.9 | 8.9% | 0.5% | 0.5 | 0.7 | 15.3% | 42.9% | 41.5% |
| load | catboost | LT | 2-12h | 240 | 1271.7 | 7.6% | 0.8% | 0.7 | 0.8 | -14.1% | 57.0% | 50.7% |
| load | catboost | LT | 24-36h | 720 | 1299.9 | 9.3% | 0.2% | 0.5 | 0.6 | 1.4% | 48.2% | 32.4% |
| load | catboost | LT | 36-48h | 720 | 1299.9 | 9.3% | 0.1% | 0.5 | 0.6 | 1.2% | 51.0% | 32.3% |
| load | catboost | LT | 48-64h | 480 | 1314.0 | 10.4% | -0.3% | 0.3 | 0.4 | 4.0% | 43.6% | 20.3% |
| load | catboost | LV | 12-24h | 597 | 774.1 | 6.5% | -0.4% | 0.8 | 0.8 | -3.7% | 56.3% | -43.0% |
| load | catboost | LV | 2-12h | 238 | 694.1 | 6.1% | -1.1% | 0.8 | 0.9 | -8.4% | 64.8% | -37.9% |
| load | catboost | LV | 24-36h | 717 | 752.5 | 7.3% | -0.1% | 0.7 | 0.8 | -21.7% | 47.0% | -71.4% |
| load | catboost | LV | 36-48h | 717 | 752.5 | 7.5% | -0.4% | 0.7 | 0.8 | -24.4% | 59.8% | -75.2% |
| load | catboost | LV | 48-64h | 479 | 781.6 | 6.9% | -0.5% | 0.6 | 0.8 | -12.7% | 47.9% | -65.6% |
| load | catboost | NL | 12-24h | 600 | 7270.0 | 39.6% | 34.7% | 0.0 | 0.2 | -111.0% | 33.7% | -2.4% |
| load | catboost | NL | 2-12h | 240 | 9605.2 | 8.1% | -0.4% | 0.0 | 0.1 | -47.4% | 20.7% | 35.8% |
| load | catboost | NL | 24-36h | 720 | 7827.2 | 31.9% | 26.9% | 0.1 | 0.3 | -103.0% | -6.2% | 2.4% |
| load | catboost | NL | 36-48h | 720 | 7827.2 | 32.0% | 27.0% | 0.1 | 0.3 | -103.5% | 39.4% | 2.1% |
| load | catboost | NL | 48-64h | 480 | 6938.2 | 48.8% | 46.2% | 0.1 | 0.7 | -113.6% | -12.8% | -4.6% |
| load | catboost | NO | 12-24h | 599 | 12319.7 | 2.0% | -0.1% | 0.8 | 0.9 | 16.7% | 57.0% | -13.9% |
| load | catboost | NO | 2-12h | 240 | 11648.8 | 1.9% | 0.4% | 0.9 | 0.9 | 21.1% | 81.0% | -34.0% |
| load | catboost | NO | 24-36h | 719 | 12320.3 | 2.5% | -0.5% | 0.8 | 0.8 | -6.8% | 56.8% | -32.8% |
| load | catboost | NO | 36-48h | 719 | 12320.3 | 2.6% | -0.5% | 0.7 | 0.8 | -9.3% | 51.8% | -36.0% |
| load | catboost | NO | 48-64h | 479 | 12656.7 | 2.9% | -1.0% | 0.5 | 0.6 | -27.8% | 36.2% | -40.3% |
| load | catboost | PL | 12-24h | 600 | 17270.0 | 7.2% | 0.6% | 0.7 | 0.8 | -45.9% | 44.2% | -97.0% |
| load | catboost | PL | 2-12h | 240 | 15299.7 | 6.5% | 2.4% | 0.7 | 0.8 | -54.4% | 73.1% | -102.3% |
| load | catboost | PL | 24-36h | 720 | 17028.9 | 9.8% | 0.5% | 0.6 | 0.7 | -103.3% | 37.5% | -182.2% |
| load | catboost | PL | 36-48h | 720 | 17028.9 | 9.9% | 0.3% | 0.5 | 0.6 | -106.0% | 39.8% | -185.9% |
| load | catboost | PL | 48-64h | 480 | 17893.6 | 10.2% | -0.9% | 0.4 | 0.4 | -102.2% | 32.0% | -186.7% |
| load | catboost | PT | 12-24h | 600 | 5915.6 | 6.8% | 1.1% | 0.7 | 0.8 | -62.9% | 57.0% | -420.9% |
| load | catboost | PT | 2-12h | 240 | 5243.1 | 5.5% | 0.4% | 0.6 | 0.7 | -44.9% | 84.4% | -366.4% |
| load | catboost | PT | 24-36h | 720 | 5983.5 | 8.5% | 1.4% | 0.6 | 0.7 | -107.1% | 53.4% | -580.0% |
| load | catboost | PT | 36-48h | 720 | 5983.5 | 8.7% | 1.3% | 0.6 | 0.7 | -112.1% | 50.2% | -596.5% |
| load | catboost | PT | 48-64h | 480 | 6353.7 | 10.0% | 1.6% | 0.3 | 0.3 | -137.2% | 34.8% | -684.8% |
| load | catboost | RO | 12-24h | 600 | 5466.6 | 8.7% | 1.3% | 0.6 | 0.8 | 1.3% | 51.6% | -261.7% |
| load | catboost | RO | 2-12h | 240 | 5214.7 | 8.1% | 0.2% | 0.3 | 0.4 | -33.0% | 72.1% | -359.6% |
| load | catboost | RO | 24-36h | 720 | 5493.1 | 10.8% | 0.8% | 0.4 | 0.6 | -28.1% | 43.5% | -378.4% |
| load | catboost | RO | 36-48h | 720 | 5493.1 | 10.8% | 0.7% | 0.4 | 0.5 | -29.1% | 41.9% | -381.9% |
| load | catboost | RO | 48-64h | 480 | 5632.3 | 12.0% | 0.9% | 0.4 | 0.5 | -26.7% | 28.4% | -384.8% |
| load | catboost | SE | 12-24h | 600 | 10972.7 | 5.1% | 2.2% | 0.6 | 0.8 | 17.5% | 41.5% | 16.8% |
| load | catboost | SE | 2-12h | 240 | 10067.1 | 5.4% | 3.6% | 0.7 | 0.8 | 12.9% | 61.7% | 5.2% |
| load | catboost | SE | 24-36h | 720 | 10890.7 | 5.3% | 2.1% | 0.6 | 0.8 | 13.9% | 45.4% | 14.8% |
| load | catboost | SE | 36-48h | 720 | 10890.7 | 5.3% | 2.1% | 0.6 | 0.8 | 12.5% | 47.6% | 13.4% |
| load | catboost | SE | 48-64h | 480 | 11302.5 | 5.7% | 1.5% | 0.3 | 0.5 | 6.9% | 30.9% | 11.4% |
| load | catboost | SI | 12-24h | 600 | 1427.9 | 12.5% | 2.0% | 0.5 | 0.6 | -112.1% | 30.7% | -150.7% |
| load | catboost | SI | 2-12h | 240 | 1218.5 | 14.7% | 9.3% | 0.5 | 0.6 | -189.0% | 44.4% | -276.7% |
| load | catboost | SI | 24-36h | 720 | 1394.3 | 14.3% | 2.8% | 0.4 | 0.5 | -147.6% | 15.6% | -195.8% |
| load | catboost | SI | 36-48h | 720 | 1394.3 | 14.6% | 2.8% | 0.4 | 0.5 | -154.4% | 29.8% | -203.9% |
| load | catboost | SI | 48-64h | 480 | 1482.2 | 14.2% | 0.4% | 0.2 | 0.3 | -136.3% | 12.9% | -174.4% |
| load | catboost | SK | 12-24h | 600 | 2712.6 | 5.7% | -0.3% | 0.7 | 0.8 | 9.0% | 37.2% | -140.5% |
| load | catboost | SK | 2-12h | 240 | 2470.1 | 5.6% | 0.5% | 0.7 | 0.8 | -12.2% | 65.4% | -168.2% |
| load | catboost | SK | 24-36h | 720 | 2674.4 | 7.4% | -0.2% | 0.6 | 0.7 | -20.7% | 31.1% | -210.8% |
| load | catboost | SK | 36-48h | 720 | 2674.4 | 7.5% | -0.3% | 0.6 | 0.6 | -22.7% | 37.0% | -215.9% |
| load | catboost | SK | 48-64h | 480 | 2776.5 | 8.0% | -0.6% | 0.4 | 0.5 | -20.6% | 22.6% | -218.9% |
| price | catboost | AT | 12-24h | 600 | 119.1 | 29.3% | -14.3% | 0.6 | 0.8 | -28.6% | 58.2% | Not measured |
| price | catboost | AT | 2-12h | 240 | 147.9 | 26.0% | -22.8% | 0.4 | 0.4 | -108.4% | 24.1% | Not measured |
| price | catboost | AT | 24-36h | 720 | 127.9 | 29.2% | -16.7% | 0.5 | 0.8 | -47.4% | 40.2% | Not measured |
| price | catboost | AT | 36-48h | 720 | 127.9 | 30.8% | -15.3% | 0.5 | 0.7 | -55.7% | 50.5% | Not measured |
| price | catboost | AT | 48-64h | 480 | 117.9 | 34.3% | -9.8% | 0.5 | 0.7 | -40.7% | 43.5% | Not measured |
| price | catboost | BG | 12-24h | 600 | 108.4 | 32.7% | -15.7% | 0.6 | 0.8 | -13.2% | 58.8% | Not measured |
| price | catboost | BG | 2-12h | 240 | 131.5 | 27.0% | -21.8% | 0.5 | 0.7 | -45.7% | 37.2% | Not measured |
| price | catboost | BG | 24-36h | 720 | 118.9 | 31.0% | -18.2% | 0.6 | 0.8 | -20.8% | 34.3% | Not measured |
| price | catboost | BG | 36-48h | 720 | 118.9 | 33.6% | -18.7% | 0.6 | 0.7 | -30.6% | 56.3% | Not measured |
| price | catboost | BG | 48-64h | 480 | 112.6 | 35.2% | -14.8% | 0.6 | 0.8 | -18.0% | 34.2% | Not measured |
| price | catboost | CH | 12-24h | 600 | 122.0 | 22.6% | -8.0% | 0.5 | 0.8 | -12.2% | 51.9% | Not measured |
| price | catboost | CH | 2-12h | 240 | 141.9 | 16.1% | -11.5% | 0.4 | 0.5 | -43.2% | 40.6% | Not measured |
| price | catboost | CH | 24-36h | 720 | 128.4 | 21.5% | -9.3% | 0.5 | 0.8 | -23.1% | 36.2% | Not measured |
| price | catboost | CH | 36-48h | 720 | 128.4 | 22.8% | -8.5% | 0.5 | 0.7 | -30.4% | 49.4% | Not measured |
| price | catboost | CH | 48-64h | 480 | 121.6 | 26.0% | -6.4% | 0.5 | 0.8 | -23.3% | 36.0% | Not measured |
| price | catboost | CZ | 12-24h | 600 | 110.3 | 31.7% | -11.5% | 0.5 | 0.8 | -23.2% | 61.9% | Not measured |
| price | catboost | CZ | 2-12h | 240 | 140.1 | 25.3% | -21.6% | 0.2 | 0.3 | -62.1% | 42.9% | Not measured |
| price | catboost | CZ | 24-36h | 720 | 119.5 | 30.8% | -14.7% | 0.5 | 0.8 | -38.3% | 47.5% | Not measured |
| price | catboost | CZ | 36-48h | 720 | 119.5 | 33.2% | -14.6% | 0.4 | 0.7 | -49.0% | 54.3% | Not measured |
| price | catboost | CZ | 48-64h | 480 | 109.2 | 37.3% | -9.4% | 0.5 | 0.8 | -40.2% | 47.4% | Not measured |
| price | catboost | EE | 12-24h | 600 | 32.8 | 83.8% | -26.3% | 0.3 | 0.5 | 27.3% | 53.9% | Not measured |
| price | catboost | EE | 2-12h | 240 | 34.9 | 89.5% | -39.0% | 0.2 | 0.4 | 39.2% | 57.6% | Not measured |
| price | catboost | EE | 24-36h | 720 | 37.8 | 86.8% | -36.1% | 0.2 | 0.4 | 25.7% | 43.7% | Not measured |
| price | catboost | EE | 36-48h | 720 | 37.8 | 92.0% | -35.6% | 0.1 | 0.3 | 21.3% | 46.3% | Not measured |
| price | catboost | EE | 48-64h | 480 | 39.2 | 88.9% | -38.8% | 0.2 | 0.3 | 13.9% | 22.1% | Not measured |
| price | catboost | FI | 12-24h | 600 | 12.5 | 70.1% | -48.8% | 0.3 | 0.4 | 7.7% | 8.9% | Not measured |
| price | catboost | FI | 2-12h | 240 | 12.1 | 72.6% | -53.0% | 0.4 | 0.4 | 5.1% | 5.5% | Not measured |
| price | catboost | FI | 24-36h | 720 | 12.7 | 76.9% | -54.3% | 0.2 | 0.2 | 0.7% | 12.7% | Not measured |
| price | catboost | FI | 36-48h | 720 | 12.7 | 80.8% | -53.1% | 0.1 | 0.2 | -4.3% | 10.0% | Not measured |
| price | catboost | FI | 48-64h | 480 | 13.0 | 79.2% | -58.0% | 0.1 | 0.2 | -1.7% | 8.3% | Not measured |
| price | catboost | GR | 12-24h | 600 | 103.3 | 31.9% | -11.3% | 0.7 | 0.8 | -10.9% | 64.9% | Not measured |
| price | catboost | GR | 2-12h | 240 | 129.0 | 27.7% | -22.5% | 0.5 | 0.7 | -66.3% | 38.8% | Not measured |
| price | catboost | GR | 24-36h | 720 | 114.4 | 29.4% | -13.6% | 0.6 | 0.8 | -15.0% | 40.3% | Not measured |
| price | catboost | GR | 36-48h | 720 | 114.4 | 31.1% | -13.4% | 0.6 | 0.8 | -21.7% | 64.1% | Not measured |
| price | catboost | GR | 48-64h | 480 | 107.0 | 32.9% | -7.9% | 0.6 | 0.8 | -6.2% | 38.1% | Not measured |
| price | catboost | HR | 12-24h | 600 | 120.4 | 28.9% | -1.5% | 0.6 | 0.8 | -4.2% | 62.9% | Not measured |
| price | catboost | HR | 2-12h | 240 | 143.3 | 20.5% | -14.2% | 0.4 | 0.5 | -26.6% | 59.6% | Not measured |
| price | catboost | HR | 24-36h | 720 | 129.6 | 27.0% | -4.8% | 0.6 | 0.8 | -12.9% | 51.7% | Not measured |
| price | catboost | HR | 36-48h | 720 | 129.6 | 30.3% | -4.9% | 0.5 | 0.7 | -26.5% | 57.4% | Not measured |
| price | catboost | HR | 48-64h | 480 | 122.8 | 34.4% | 1.2% | 0.5 | 0.7 | -20.8% | 43.8% | Not measured |
| price | catboost | HU | 12-24h | 600 | 120.2 | 28.7% | -5.9% | 0.6 | 0.8 | -11.7% | 64.3% | Not measured |
| price | catboost | HU | 2-12h | 240 | 145.6 | 24.1% | -21.7% | 0.4 | 0.5 | -73.8% | 53.1% | Not measured |
| price | catboost | HU | 24-36h | 720 | 129.8 | 27.5% | -9.2% | 0.6 | 0.8 | -20.9% | 53.2% | Not measured |
| price | catboost | HU | 36-48h | 720 | 129.8 | 30.6% | -9.2% | 0.6 | 0.7 | -34.8% | 57.7% | Not measured |
| price | catboost | HU | 48-64h | 480 | 121.9 | 34.0% | -2.5% | 0.6 | 0.8 | -21.3% | 48.2% | Not measured |
| price | catboost | IT | 12-24h | 600 | 165.9 | 11.1% | -6.6% | 0.5 | 0.8 | 12.3% | 52.5% | Not measured |
| price | catboost | IT | 2-12h | 240 | 170.5 | 9.5% | -7.8% | 0.5 | 0.6 | -9.3% | 57.5% | Not measured |
| price | catboost | IT | 24-36h | 720 | 170.6 | 12.1% | -8.4% | 0.5 | 0.7 | -0.5% | 42.4% | Not measured |
| price | catboost | IT | 36-48h | 720 | 170.6 | 13.0% | -8.7% | 0.4 | 0.7 | -8.4% | 40.7% | Not measured |
| price | catboost | IT | 48-64h | 480 | 170.7 | 14.5% | -8.9% | 0.4 | 0.7 | -6.1% | 30.2% | Not measured |
| price | catboost | LT | 12-24h | 600 | 57.9 | 66.0% | -10.5% | 0.4 | 0.6 | 9.4% | 50.9% | Not measured |
| price | catboost | LT | 2-12h | 240 | 70.1 | 67.7% | -26.2% | 0.2 | 0.5 | 16.9% | 31.7% | Not measured |
| price | catboost | LT | 24-36h | 720 | 65.2 | 65.1% | -19.5% | 0.3 | 0.6 | 5.8% | 32.2% | Not measured |
| price | catboost | LT | 36-48h | 720 | 65.2 | 66.5% | -17.4% | 0.3 | 0.5 | 3.7% | 47.9% | Not measured |
| price | catboost | LT | 48-64h | 480 | 62.8 | 63.9% | -14.4% | 0.4 | 0.6 | -2.7% | 31.6% | Not measured |
| price | catboost | LV | 12-24h | 600 | 57.0 | 66.5% | -14.4% | 0.4 | 0.6 | 8.1% | 51.9% | Not measured |
| price | catboost | LV | 2-12h | 240 | 70.1 | 67.4% | -33.0% | 0.2 | 0.5 | 17.2% | 31.9% | Not measured |
| price | catboost | LV | 24-36h | 720 | 64.5 | 66.6% | -22.7% | 0.3 | 0.5 | 2.9% | 30.7% | Not measured |
| price | catboost | LV | 36-48h | 720 | 64.5 | 67.9% | -19.9% | 0.3 | 0.5 | 1.1% | 47.8% | Not measured |
| price | catboost | LV | 48-64h | 480 | 61.7 | 64.4% | -17.8% | 0.4 | 0.6 | -4.8% | 33.1% | Not measured |
| price | catboost | NL | 12-24h | 600 | 100.5 | 48.5% | -37.8% | 0.5 | 0.8 | -102.3% | 48.0% | Not measured |
| price | catboost | NL | 2-12h | 240 | 140.4 | 43.9% | -43.1% | 0.4 | 0.5 | -252.0% | -36.5% | Not measured |
| price | catboost | NL | 24-36h | 720 | 111.1 | 47.4% | -38.9% | 0.5 | 0.8 | -133.1% | 19.7% | Not measured |
| price | catboost | NL | 36-48h | 720 | 111.1 | 49.3% | -39.9% | 0.5 | 0.7 | -142.4% | 37.0% | Not measured |
| price | catboost | NL | 48-64h | 480 | 96.5 | 51.8% | -37.4% | 0.5 | 0.8 | -99.1% | 35.6% | Not measured |
| price | catboost | NO | 12-24h | 600 | 79.0 | 27.2% | -14.6% | 0.2 | 0.4 | -56.3% | 8.8% | Not measured |
| price | catboost | NO | 2-12h | 240 | 90.0 | 28.3% | -27.6% | 0.3 | 0.2 | -126.7% | -193.9% | Not measured |
| price | catboost | NO | 24-36h | 720 | 81.8 | 29.3% | -19.1% | 0.1 | 0.3 | -83.8% | -33.1% | Not measured |
| price | catboost | NO | 36-48h | 720 | 81.8 | 31.2% | -18.8% | 0.0 | 0.0 | -95.9% | -16.0% | Not measured |
| price | catboost | NO | 48-64h | 480 | 77.6 | 31.7% | -13.2% | 0.0 | 0.1 | -77.0% | -2.6% | Not measured |
| price | catboost | PL | 12-24h | 600 | 116.8 | 45.6% | -38.0% | 0.4 | 0.8 | -58.0% | 42.6% | Not measured |
| price | catboost | PL | 2-12h | 240 | 136.1 | 45.2% | -44.2% | 0.2 | 0.3 | -191.3% | 28.9% | Not measured |
| price | catboost | PL | 24-36h | 720 | 124.5 | 45.0% | -38.8% | 0.4 | 0.7 | -77.9% | 25.0% | Not measured |
| price | catboost | PL | 36-48h | 720 | 124.5 | 46.6% | -39.5% | 0.4 | 0.7 | -83.9% | 35.0% | Not measured |
| price | catboost | PL | 48-64h | 480 | 118.7 | 47.0% | -37.7% | 0.4 | 0.8 | -52.0% | 20.9% | Not measured |
| price | catboost | RO | 12-24h | 600 | 118.4 | 29.6% | -9.6% | 0.6 | 0.8 | -6.3% | 64.6% | Not measured |
| price | catboost | RO | 2-12h | 240 | 142.2 | 24.1% | -20.4% | 0.4 | 0.5 | -41.6% | 56.6% | Not measured |
| price | catboost | RO | 24-36h | 720 | 128.3 | 28.1% | -12.1% | 0.6 | 0.8 | -15.1% | 52.4% | Not measured |
| price | catboost | RO | 36-48h | 720 | 128.3 | 30.8% | -12.0% | 0.5 | 0.7 | -25.8% | 60.3% | Not measured |
| price | catboost | RO | 48-64h | 480 | 121.3 | 33.5% | -7.4% | 0.6 | 0.8 | -16.4% | 46.0% | Not measured |
| price | catboost | SE | 12-24h | 600 | 30.0 | 60.5% | -32.1% | 0.4 | 0.6 | -9.6% | 38.2% | Not measured |
| price | catboost | SE | 2-12h | 240 | 33.4 | 63.7% | -39.5% | 0.3 | 0.5 | -5.7% | 24.6% | Not measured |
| price | catboost | SE | 24-36h | 720 | 32.1 | 62.0% | -36.5% | 0.3 | 0.5 | -14.8% | 30.8% | Not measured |
| price | catboost | SE | 36-48h | 720 | 32.1 | 63.9% | -34.5% | 0.3 | 0.4 | -18.3% | 36.0% | Not measured |
| price | catboost | SE | 48-64h | 480 | 31.4 | 63.1% | -34.4% | 0.3 | 0.5 | -24.7% | 29.4% | Not measured |
| price | catboost | SI | 12-24h | 600 | 123.5 | 26.8% | 1.6% | 0.6 | 0.8 | -0.5% | 64.3% | Not measured |
| price | catboost | SI | 2-12h | 240 | 148.1 | 18.8% | -8.9% | 0.5 | 0.4 | -41.7% | 58.8% | Not measured |
| price | catboost | SI | 24-36h | 720 | 132.2 | 24.5% | -1.2% | 0.6 | 0.8 | -6.1% | 56.2% | Not measured |
| price | catboost | SI | 36-48h | 720 | 132.2 | 27.7% | -2.4% | 0.5 | 0.7 | -20.0% | 58.6% | Not measured |
| price | catboost | SI | 48-64h | 480 | 124.2 | 31.5% | 2.8% | 0.6 | 0.8 | -8.7% | 51.5% | Not measured |
| price | catboost | SK | 12-24h | 600 | 118.6 | 30.7% | -14.8% | 0.5 | 0.8 | -19.1% | 62.0% | Not measured |
| price | catboost | SK | 2-12h | 240 | 146.0 | 27.4% | -24.6% | 0.3 | 0.4 | -82.1% | 43.1% | Not measured |
| price | catboost | SK | 24-36h | 720 | 128.8 | 30.7% | -18.0% | 0.5 | 0.8 | -32.2% | 47.4% | Not measured |
| price | catboost | SK | 36-48h | 720 | 128.8 | 32.1% | -16.9% | 0.5 | 0.8 | -38.4% | 55.8% | Not measured |
| price | catboost | SK | 48-64h | 480 | 120.2 | 35.6% | -13.7% | 0.5 | 0.8 | -26.2% | 47.1% | Not measured |
| renewable | catboost | BE | 12-24h | 600 | 3674.4 | 43.6% | 34.7% | 0.6 | 0.9 | -33.8% | 52.6% | -373.3% |
| renewable | catboost | BE | 2-12h | 240 | 1716.7 | 114.3% | 113.7% | 0.5 | 0.7 | -157.4% | -131.4% | -619.8% |
| renewable | catboost | BE | 24-36h | 720 | 3297.8 | 52.6% | 44.2% | 0.5 | 0.9 | -47.6% | 0.8% | -423.4% |
| renewable | catboost | BE | 36-48h | 720 | 3297.8 | 52.8% | 44.3% | 0.5 | 0.9 | -48.2% | 46.7% | -425.6% |
| renewable | catboost | BE | 48-64h | 480 | 4088.3 | 41.7% | 31.8% | 0.5 | 0.8 | -23.6% | 20.9% | -373.3% |
| renewable | catboost | DE | 12-24h | 600 | 36185.1 | 50.1% | -38.7% | 0.0 | 0.1 | -163.5% | 31.2% | -237.0% |
| renewable | catboost | DE | 2-12h | 240 | 22341.8 | 35.2% | -0.8% | 0.0 | 0.2 | -12.0% | -3.0% | -40.6% |
| renewable | catboost | DE | 24-36h | 720 | 33101.7 | 49.5% | -33.0% | 0.0 | 0.1 | -139.5% | -26.1% | -204.4% |
| renewable | catboost | DE | 36-48h | 720 | 33101.7 | 49.5% | -33.0% | 0.0 | 0.1 | -139.4% | 36.7% | -204.3% |
| renewable | catboost | DE | 48-64h | 480 | 38481.7 | 53.2% | -42.3% | 0.0 | 0.1 | -203.8% | -34.9% | -288.7% |
| renewable | catboost | FR | 12-24h | 365 | 14186.2 | 29.1% | 24.3% | 0.6 | 0.8 | -42.7% | 52.8% | -11.3% |
| renewable | catboost | FR | 2-12h | 144 | 8395.6 | 49.7% | 49.7% | 1.1 | 0.7 | -124.1% | -73.9% | -6.8% |
| renewable | catboost | FR | 24-36h | 441 | 13307.9 | 34.3% | 30.1% | 0.6 | 0.8 | -60.9% | 4.8% | -17.5% |
| renewable | catboost | FR | 36-48h | 441 | 13307.9 | 34.3% | 30.2% | 0.6 | 0.8 | -58.9% | 42.2% | -17.7% |
| renewable | catboost | FR | 48-64h | 297 | 15689.6 | 28.9% | 23.4% | 0.5 | 0.7 | -36.2% | 21.9% | -17.2% |
| solar | catboost | BE | 12-24h | 600 | 2597.4 | 22.9% | -15.1% | 0.8 | 0.9 | 30.5% | 83.0% | -150.0% |
| solar | catboost | BE | 2-12h | 240 | 731.7 | 22.9% | -17.7% | 0.9 | 1.0 | 26.4% | 77.1% | -98.2% |
| solar | catboost | BE | 24-36h | 720 | 2164.5 | 23.5% | -14.7% | 0.8 | 0.9 | 28.7% | 66.4% | -156.4% |
| solar | catboost | BE | 36-48h | 720 | 2164.5 | 23.7% | -14.7% | 0.8 | 0.9 | 28.0% | 85.1% | -158.7% |
| solar | catboost | BE | 48-64h | 480 | 2880.9 | 23.7% | -8.1% | 0.8 | 0.9 | 28.7% | 64.4% | -167.0% |
| solar | catboost | DE | 12-24h | 600 | 20262.0 | 62.2% | -61.5% | 0.3 | 0.9 | -156.7% | 53.5% | -1169.9% |
| solar | catboost | DE | 2-12h | 240 | 6077.2 | 51.6% | -46.3% | 0.4 | 1.0 | -75.5% | 48.3% | -751.3% |
| solar | catboost | DE | 24-36h | 720 | 16885.0 | 62.4% | -61.3% | 0.3 | 1.0 | -157.4% | 1.2% | -1173.4% |
| solar | catboost | DE | 36-48h | 720 | 16885.0 | 62.4% | -61.3% | 0.3 | 1.0 | -157.4% | 60.9% | -1173.6% |
| solar | catboost | DE | 48-64h | 480 | 22288.9 | 63.9% | -63.4% | 0.3 | 1.0 | -171.6% | -7.7% | -1248.2% |
| solar | catboost | FR | 12-24h | 365 | 7528.9 | 20.8% | -5.4% | 0.9 | 0.9 | 34.3% | 85.3% | -149.3% |
| solar | catboost | FR | 2-12h | 144 | 1317.8 | 41.1% | 36.1% | 1.2 | 1.0 | -38.7% | 60.5% | -248.9% |
| solar | catboost | FR | 24-36h | 441 | 6245.5 | 21.7% | -6.2% | 0.9 | 0.9 | 38.3% | 70.7% | -153.6% |
| solar | catboost | FR | 36-48h | 441 | 6245.5 | 21.8% | -6.3% | 0.9 | 0.9 | 38.0% | 86.2% | -154.1% |
| solar | catboost | FR | 48-64h | 297 | 8634.7 | 20.2% | -10.0% | 0.8 | 0.9 | 42.3% | 72.5% | -142.8% |
| wind_onshore | catboost | BE | 12-24h | 600 | 433.6 | 202.7% | 195.2% | 0.3 | 0.3 | -140.4% | -132.9% | -790.7% |
| wind_onshore | catboost | BE | 2-12h | 240 | 377.3 | 246.9% | 242.9% | 0.2 | 0.2 | -161.0% | -177.7% | -923.1% |
| wind_onshore | catboost | BE | 24-36h | 720 | 454.6 | 189.7% | 184.5% | 0.3 | 0.3 | -126.7% | -118.4% | -774.1% |
| wind_onshore | catboost | BE | 36-48h | 720 | 454.6 | 191.2% | 184.7% | 0.2 | 0.2 | -128.5% | -102.9% | -780.9% |
| wind_onshore | catboost | BE | 48-64h | 480 | 493.2 | 168.5% | 161.3% | 0.3 | 0.3 | -112.1% | -107.2% | -711.4% |
| wind_onshore | catboost | DE | 12-24h | 600 | 8238.2 | 62.6% | 10.0% | -0.1 | -0.2 | 23.4% | 1.0% | -436.2% |
| wind_onshore | catboost | DE | 2-12h | 240 | 8329.5 | 55.9% | 10.0% | -0.1 | -0.2 | 13.4% | -36.5% | -479.5% |
| wind_onshore | catboost | DE | 24-36h | 720 | 8469.3 | 62.3% | 11.0% | -0.1 | -0.3 | 18.1% | 5.3% | -427.4% |
| wind_onshore | catboost | DE | 36-48h | 720 | 8469.3 | 61.6% | 10.6% | -0.1 | -0.2 | 19.0% | 22.1% | -421.6% |
| wind_onshore | catboost | DE | 48-64h | 480 | 8539.2 | 61.6% | 9.7% | -0.0 | -0.0 | 24.6% | 18.3% | -378.9% |
| wind_onshore | catboost | FR | 12-24h | 365 | 2677.8 | 135.8% | 129.2% | 0.0 | 0.0 | -106.3% | -111.5% | -697.0% |
| wind_onshore | catboost | FR | 2-12h | 144 | 2896.9 | 101.6% | 99.9% | 0.2 | 0.2 | -100.7% | -142.8% | -652.4% |
| wind_onshore | catboost | FR | 24-36h | 441 | 2842.7 | 126.0% | 118.9% | 0.1 | 0.1 | -92.0% | -141.5% | -692.7% |
| wind_onshore | catboost | FR | 36-48h | 441 | 2842.7 | 124.6% | 117.3% | 0.1 | 0.1 | -89.4% | -73.5% | -683.6% |
| wind_onshore | catboost | FR | 48-64h | 297 | 2816.4 | 135.3% | 126.0% | 0.1 | 0.1 | -104.0% | -115.5% | -691.2% |
| wind_offshore | xgboost | BE | 12-24h | 600 | 435.3 | 162.6% | 128.2% | 0.2 | 0.2 | -56.8% | -81.8% | -471.2% |
| wind_offshore | xgboost | BE | 2-12h | 240 | 396.8 | 190.5% | 169.6% | 0.2 | 0.2 | -109.5% | -92.9% | -575.1% |
| wind_offshore | xgboost | BE | 24-36h | 720 | 469.9 | 156.9% | 116.5% | 0.0 | 0.0 | -49.6% | -44.6% | -477.8% |
| wind_offshore | xgboost | BE | 36-48h | 720 | 469.9 | 158.4% | 116.0% | 0.0 | 0.0 | -51.1% | -30.4% | -483.5% |
| wind_offshore | xgboost | BE | 48-64h | 480 | 506.5 | 160.1% | 94.3% | -0.1 | -0.1 | -45.1% | -44.9% | -499.0% |
| wind_offshore | xgboost | FR | 12-24h | 365 | 383.2 | 120.5% | 86.8% | -0.0 | -0.0 | -31.0% | -24.5% | -291.7% |
| wind_offshore | xgboost | FR | 2-12h | 144 | 453.3 | 82.7% | 52.5% | 0.2 | 0.2 | 1.4% | -11.1% | -332.6% |
| wind_offshore | xgboost | FR | 24-36h | 441 | 436.6 | 107.3% | 68.7% | -0.1 | -0.1 | -20.7% | -30.8% | -286.5% |
| wind_offshore | xgboost | FR | 36-48h | 441 | 436.6 | 105.3% | 68.6% | -0.1 | -0.1 | -18.3% | -3.1% | -279.2% |
| wind_offshore | xgboost | FR | 48-64h | 297 | 428.4 | 116.0% | 78.3% | -0.1 | -0.1 | -46.5% | -19.8% | -260.2% |
| biomass | xgboost | BE | 12-24h | 600 | 206.8 | 69.8% | -69.8% | 0.3 | 0.5 | -800.0% | -790.4% | Not measured |
| biomass | xgboost | BE | 2-12h | 240 | 209.3 | 69.8% | -69.8% | 0.4 | 0.6 | -829.1% | -1031.7% | Not measured |
| biomass | xgboost | BE | 24-36h | 720 | 207.5 | 69.8% | -69.8% | 0.2 | 0.3 | -776.1% | -771.2% | Not measured |
| biomass | xgboost | BE | 36-48h | 720 | 207.5 | 69.7% | -69.7% | 0.2 | 0.3 | -775.8% | -661.3% | Not measured |
| biomass | xgboost | BE | 48-64h | 480 | 206.6 | 69.7% | -69.7% | 0.1 | 0.2 | -751.5% | -689.6% | Not measured |
| biomass | xgboost | FR | 12-24h | 365 | 284.5 | 3.6% | 0.7% | 0.2 | 0.2 | -40.7% | -51.2% | Not measured |
| biomass | xgboost | FR | 2-12h | 144 | 287.9 | 3.2% | -0.4% | 0.4 | 0.3 | -28.0% | -53.8% | Not measured |
| biomass | xgboost | FR | 24-36h | 441 | 284.7 | 3.9% | 0.6% | 0.1 | 0.0 | -43.8% | -80.4% | Not measured |
| biomass | xgboost | FR | 36-48h | 441 | 284.7 | 4.0% | 0.7% | -0.0 | -0.0 | -47.9% | -63.4% | Not measured |
| biomass | xgboost | FR | 48-64h | 297 | 283.2 | 4.4% | 1.5% | -0.2 | -0.2 | -51.4% | -49.1% | Not measured |
| hydro_total | xgboost | BE | 12-24h | 600 | 1.3 | 13302.6% | 13302.6% | -8.8 | -0.1 | -12973.1% | -16108.2% | Not measured |
| hydro_total | xgboost | BE | 2-12h | 240 | 1.6 | 7949.6% | 7949.6% | -8.5 | -0.3 | -8409.7% | -9857.9% | Not measured |
| hydro_total | xgboost | BE | 24-36h | 720 | 1.2 | 14497.7% | 14497.7% | -8.9 | -0.1 | -13904.6% | -13165.1% | Not measured |
| hydro_total | xgboost | BE | 36-48h | 720 | 1.2 | 14525.3% | 14525.3% | -8.8 | -0.1 | -13931.2% | -10718.7% | Not measured |
| hydro_total | xgboost | BE | 48-64h | 480 | 1.1 | 19631.7% | 19631.7% | -7.2 | -0.1 | -17579.4% | -13044.0% | Not measured |
| hydro_total | xgboost | FR | 12-24h | 365 | 3311.7 | 26.9% | 26.0% | 1.5 | 0.9 | -104.8% | 38.0% | Not measured |
| hydro_total | xgboost | FR | 2-12h | 144 | 3439.7 | 20.1% | 18.7% | 0.7 | 0.5 | -26.6% | 59.6% | Not measured |
| hydro_total | xgboost | FR | 24-36h | 441 | 3498.4 | 29.3% | 28.3% | 1.5 | 0.9 | -126.2% | 17.0% | Not measured |
| hydro_total | xgboost | FR | 36-48h | 441 | 3498.4 | 29.3% | 28.3% | 1.4 | 0.9 | -123.4% | 26.9% | Not measured |
| hydro_total | xgboost | FR | 48-64h | 297 | 3526.9 | 33.1% | 32.3% | 1.5 | 0.9 | -166.1% | -15.8% | Not measured |
| net_position | chronos-2-V010 | AT | 24-36h | 5 | -1107.2 | 25.1% | -15.6% | 0.4 | 0.9 | -24.1% | 59.6% | Not measured |
| net_position | chronos-2-V010 | AT | 36-48h | 115 | -1592.2 | 86.4% | 69.1% | 0.0 | 0.1 | -82.5% | -53.5% | Not measured |
| net_position | chronos-2-V010 | AT | 48-64h | 245 | -715.1 | 90.4% | 41.0% | 0.1 | 0.3 | -9.1% | -14.6% | Not measured |
| net_position | chronos-2-V010 | BE | 24-36h | 5 | -4083.1 | 36.2% | 36.2% | 0.2 | 0.8 | -336.6% | -100.3% | Not measured |
| net_position | chronos-2-V010 | BE | 36-48h | 115 | -3576.8 | 70.1% | 68.8% | 0.4 | 0.2 | -171.8% | -182.8% | Not measured |
| net_position | chronos-2-V010 | BE | 48-64h | 245 | -3350.2 | 73.4% | 68.6% | 0.2 | 0.1 | -170.7% | -87.4% | Not measured |
| net_position | chronos-2-V010 | BG | 24-36h | 5 | 398.4 | 97.2% | 97.2% | 0.4 | 0.7 | -48.7% | -91.2% | Not measured |
| net_position | chronos-2-V010 | BG | 36-48h | 115 | 157.7 | 60.4% | -20.1% | 0.5 | 0.8 | 4.7% | 2.5% | Not measured |
| net_position | chronos-2-V010 | BG | 48-64h | 245 | 509.3 | 59.1% | -14.4% | 0.5 | 0.8 | 7.6% | 2.9% | Not measured |
| net_position | chronos-2-V010 | CZ | 24-36h | 5 | 2406.6 | 14.6% | -14.6% | -0.7 | -0.8 | 31.4% | -158.4% | Not measured |
| net_position | chronos-2-V010 | CZ | 36-48h | 115 | 1605.8 | 86.9% | -78.7% | 0.2 | 0.2 | -246.0% | -146.3% | Not measured |
| net_position | chronos-2-V010 | CZ | 48-64h | 245 | 970.5 | 91.9% | -81.1% | 0.4 | 0.4 | -105.2% | -88.1% | Not measured |
| net_position | chronos-2-V010 | DE | 36-48h | 96 | -5168.5 | 90.1% | 65.9% | 0.1 | 0.2 | -58.8% | -22.0% | Not measured |
| net_position | chronos-2-V010 | DE | 48-64h | 221 | 91.0 | 79.0% | 39.2% | 0.4 | 0.7 | -25.4% | 6.0% | Not measured |
| net_position | chronos-2-V010 | EE | 24-36h | 5 | -416.4 | 3.7% | -1.6% | 1.5 | 1.0 | 87.9% | 66.0% | Not measured |
| net_position | chronos-2-V010 | EE | 36-48h | 115 | -337.0 | 79.5% | 78.2% | 0.2 | 0.2 | -203.8% | -209.7% | Not measured |
| net_position | chronos-2-V010 | EE | 48-64h | 245 | -281.8 | 65.8% | 61.4% | 0.5 | 0.6 | -52.5% | -112.1% | Not measured |
| net_position | chronos-2-V010 | ES | 24-36h | 5 | 209.9 | 134.4% | 134.4% | 1.6 | 0.8 | 59.1% | 65.5% | Not measured |
| net_position | chronos-2-V010 | ES | 36-48h | 115 | 345.2 | 54.0% | 8.8% | 0.6 | 0.9 | 25.7% | 22.0% | Not measured |
| net_position | chronos-2-V010 | ES | 48-64h | 245 | 2443.7 | 68.7% | -48.0% | 0.3 | 0.5 | -51.9% | -52.5% | Not measured |
| net_position | chronos-2-V010 | FI | 24-36h | 5 | 1552.0 | 12.5% | -6.3% | -0.5 | -0.9 | -36.0% | 33.0% | Not measured |
| net_position | chronos-2-V010 | FI | 36-48h | 115 | 224.1 | 70.3% | -17.4% | 0.3 | 0.7 | 51.6% | 45.3% | Not measured |
| net_position | chronos-2-V010 | FI | 48-64h | 245 | 61.0 | 93.9% | -37.4% | 0.3 | 0.5 | 43.9% | 36.7% | Not measured |
| net_position | chronos-2-V010 | FR | 24-36h | 5 | 9582.2 | 19.0% | -19.0% | 0.1 | 0.4 | -371.9% | -265.1% | Not measured |
| net_position | chronos-2-V010 | FR | 36-48h | 115 | 7914.8 | 77.1% | -70.7% | -0.2 | -0.1 | -346.9% | -348.8% | Not measured |
| net_position | chronos-2-V010 | FR | 48-64h | 245 | 5739.8 | 72.3% | -56.4% | 0.1 | 0.1 | -61.8% | -60.2% | Not measured |
| net_position | chronos-2-V010 | GR | 36-48h | 0 | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured |
| net_position | chronos-2-V010 | GR | 48-64h | 0 | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured |
| net_position | chronos-2-V010 | HR | 24-36h | 5 | -355.1 | 39.9% | -39.9% | 0.3 | 0.7 | 78.8% | 25.8% | Not measured |
| net_position | chronos-2-V010 | HR | 36-48h | 115 | -472.7 | 84.2% | 59.4% | 0.3 | 0.4 | -14.9% | -6.4% | Not measured |
| net_position | chronos-2-V010 | HR | 48-64h | 245 | -1013.4 | 65.8% | 56.2% | 0.1 | 0.1 | -72.2% | -83.1% | Not measured |
| net_position | chronos-2-V010 | HU | 24-36h | 5 | -588.7 | 122.9% | -122.9% | 0.6 | 1.0 | 25.8% | -81.0% | Not measured |
| net_position | chronos-2-V010 | HU | 36-48h | 115 | -1678.9 | 75.0% | 60.5% | 0.4 | 0.5 | -54.0% | -200.0% | Not measured |
| net_position | chronos-2-V010 | HU | 48-64h | 245 | -1829.0 | 61.0% | 42.5% | 0.4 | 0.6 | -23.8% | -108.0% | Not measured |
| net_position | chronos-2-V010 | LT | 24-36h | 5 | 95.2 | 161.3% | -161.3% | 0.9 | 1.0 | -2.2% | 53.6% | Not measured |
| net_position | chronos-2-V010 | LT | 36-48h | 115 | -262.6 | 78.2% | 69.2% | 0.4 | 0.7 | 8.5% | -23.3% | Not measured |
| net_position | chronos-2-V010 | LT | 48-64h | 245 | -224.6 | 67.7% | 42.6% | 0.6 | 0.7 | 34.7% | 21.4% | Not measured |
| net_position | chronos-2-V010 | LV | 24-36h | 5 | -245.7 | 44.4% | -44.4% | -0.2 | -0.1 | -168.8% | -80.9% | Not measured |
| net_position | chronos-2-V010 | LV | 36-48h | 115 | -308.0 | 77.3% | 74.2% | 0.4 | 0.5 | -55.0% | -90.3% | Not measured |
| net_position | chronos-2-V010 | LV | 48-64h | 245 | -271.0 | 71.3% | 61.5% | 0.4 | 0.6 | -23.0% | -34.7% | Not measured |
| net_position | chronos-2-V010 | NL | 24-36h | 5 | -2219.4 | 102.1% | 102.1% | 2.0 | 0.6 | -75.1% | 21.3% | Not measured |
| net_position | chronos-2-V010 | NL | 36-48h | 115 | 1674.9 | 111.5% | -34.5% | -0.2 | -0.3 | -3.0% | -10.2% | Not measured |
| net_position | chronos-2-V010 | NL | 48-64h | 245 | 2127.4 | 90.2% | -43.4% | -0.1 | -0.1 | 4.6% | -2.1% | Not measured |
| net_position | chronos-2-V010 | PL | 24-36h | 5 | 2682.2 | 93.1% | -93.1% | 0.5 | 0.6 | 3.4% | 31.1% | Not measured |
| net_position | chronos-2-V010 | PL | 36-48h | 115 | 1085.7 | 89.0% | -63.7% | 0.0 | 0.1 | 4.4% | -14.7% | Not measured |
| net_position | chronos-2-V010 | PL | 48-64h | 245 | 280.9 | 85.5% | -19.0% | 0.2 | 0.5 | 12.4% | 7.5% | Not measured |
| net_position | chronos-2-V010 | PT | 24-36h | 5 | -1229.9 | 42.9% | 42.9% | 0.2 | 0.3 | -10.1% | 19.7% | Not measured |
| net_position | chronos-2-V010 | PT | 36-48h | 115 | -1817.1 | 53.9% | 49.1% | 0.9 | 0.8 | -78.7% | -63.1% | Not measured |
| net_position | chronos-2-V010 | PT | 48-64h | 245 | -2724.6 | 62.5% | 54.4% | 0.4 | 0.4 | -154.0% | -121.1% | Not measured |
| net_position | chronos-2-V010 | RO | 24-36h | 5 | 893.2 | 51.4% | -51.4% | 0.8 | 0.9 | 67.2% | 13.7% | Not measured |
| net_position | chronos-2-V010 | RO | 36-48h | 115 | 184.6 | 91.4% | -24.9% | 0.2 | 0.4 | 22.1% | 5.6% | Not measured |
| net_position | chronos-2-V010 | RO | 48-64h | 245 | -145.1 | 89.3% | -11.5% | 0.2 | 0.4 | 7.3% | -25.9% | Not measured |
| net_position | chronos-2-V010 | SI | 24-36h | 5 | -321.5 | 46.6% | -32.7% | -0.4 | -0.8 | 23.6% | 7.2% | Not measured |
| net_position | chronos-2-V010 | SI | 36-48h | 115 | -453.0 | 72.6% | 69.3% | -0.1 | -0.0 | -258.5% | -139.3% | Not measured |
| net_position | chronos-2-V010 | SI | 48-64h | 245 | -347.3 | 85.3% | 79.0% | 0.2 | 0.2 | -151.5% | -104.2% | Not measured |
| net_position | chronos-2-V010 | SK | 24-36h | 5 | 1058.7 | 18.8% | -18.8% | 1.8 | 0.9 | -3.1% | -33.2% | Not measured |
| net_position | chronos-2-V010 | SK | 36-48h | 115 | 326.9 | 83.7% | -77.2% | 0.3 | 0.6 | 2.9% | -55.7% | Not measured |
| net_position | chronos-2-V010 | SK | 48-64h | 245 | -109.9 | 59.9% | 14.9% | 0.5 | 0.8 | 18.3% | -39.5% | Not measured |

## Correctness notes

- Both `T` and space timestamp separators are parsed before joining.
- **Renewable-family figures are not comparable across the 2026-08-13 boundary.** ABL-410 moved their scoring truth from `energy_renewable` to `energy_generation` to match what the dashboard publishes. Eight of the fifteen live pairs are unchanged to the digit; `renewable` and `hydro_total` move, and BE `hydro_total` moves by two orders of magnitude because its frozen actual folded in pumped storage. The before/after is in `reports/abl_410_scoring_truth.md`.
- `energy_generation` has an open FR ingest gap (ABL-318 §3): no rows 2026-06-30 23:45 → 2026-07-22 14:15. FR sample sizes in any window overlapping it are correspondingly smaller.
- **Pooled rows are denominator-weighted across countries, so they move when coverage does.** On the 2026-07-11 → 2026-08-10 window the FR gap above cost 1,072 hours of the best-forecast country in most types: pooled `solar` went 48.35% → 51.85% and pooled `wind_onshore` 76.94% → 77.72% while **every country's own figure was unchanged to the digit**. Read a pooled move against the country × horizon detail before reading it as quality.
- `load_mw > 0` is applied only to load. Measured zero remains valid for every other type.
- GR net position is excluded by name: actuals are fabricated exact zeros, not measurements (ABL-35: every published row since 2025-10-01 is 0.0 while GR moved a median 1,142 MW across its borders); row deletion pending on ABL-67
- D−7 and persistence use only stored actual observations. Missing source rows remain missing.
- Net-position persistence reuses the promotion evaluator's day-ahead publication cutoff.
- TSO comparisons use the latest stored TSO series; the database does not retain an issued-vintage archive for reconstruction.
- The separate net-position promotion gate remains authoritative and is not reproduced here.
