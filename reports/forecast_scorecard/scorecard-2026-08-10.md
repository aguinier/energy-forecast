# Forecast quality scorecard

Generated: 2026-08-10 09:20 UTC
Target window: 2026-07-11 00:00:00 → 2026-08-10 00:00:00 (exclusive)
Sample: 158,907 selected forecast rows; 158,639 paired actual rows
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
| solar | 53.6% | 48.4% | 26.0% | 24.6% | -50.7% | -43.3% |
| wind_onshore | 73.5% | 76.9% | 75.0% | 71.6% | 28.1% | 38.3% |

Load reproduces within 0.1 percentage point on WAPE/D−7 and 0.1 point on TSO (reference 4.0%, measured 4.1%). Price, solar, and wind do not reproduce exactly; the scorecard preserves the disagreement.

## Pooled score

Skill is `100 × (1 − model WAPE / baseline WAPE)` on the exact same pairs.

| type | model | horizon | n | WAPE | MAE | bias | slope | corr | D−7 WAPE / skill | persistence WAPE / skill | TSO WAPE / skill |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| load | catboost | 12-24h | 12,594 | 8.1% | 867.6 | 2.5% | 1.0 | 1.0 | 6.0% / -34.4% | 14.4% / 43.8% | 4.2% / -91.6% |
| load | catboost | 2-12h | 5,037 | 6.7% | 658.1 | 2.0% | 1.0 | 1.0 | 4.8% / -40.7% | 21.3% / 68.5% | 3.2% / -107.8% |
| load | catboost | 24-36h | 15,106 | 9.7% | 1032.1 | 2.0% | 1.0 | 1.0 | 5.9% / -64.0% | 14.9% / 35.4% | 4.1% / -137.8% |
| load | catboost | 36-48h | 15,106 | 9.7% | 1041.3 | 2.0% | 1.0 | 1.0 | 5.9% / -65.4% | 17.2% / 43.3% | 4.1% / -139.9% |
| load | catboost | 48-64h | 10,069 | 10.8% | 1203.1 | 1.8% | 1.0 | 1.0 | 6.4% / -69.4% | 15.1% / 28.3% | 4.4% / -144.1% |
| load | catboost | all | 57,912 | 9.3% | 995.9 | 2.1% | 1.0 | 1.0 | 5.9% / -57.3% | 15.9% / 41.6% | 4.1% / -127.3% |
| price | catboost | 12-24h | 11,400 | 33.7% | 32.2 | -13.6% | 0.7 | 0.8 | 29.8% / -13.0% | 76.7% / 56.1% | Not measured / Not measured |
| price | catboost | 2-12h | 4,560 | 30.2% | 34.5 | -23.1% | 0.7 | 0.8 | 21.7% / -39.1% | 48.7% / 38.1% | Not measured / Not measured |
| price | catboost | 24-36h | 13,680 | 33.3% | 34.3 | -16.8% | 0.6 | 0.8 | 27.3% / -22.0% | 55.1% / 39.6% | Not measured / Not measured |
| price | catboost | 36-48h | 13,680 | 35.4% | 36.5 | -16.5% | 0.6 | 0.8 | 27.3% / -29.7% | 71.2% / 50.2% | Not measured / Not measured |
| price | catboost | 48-64h | 9,120 | 37.5% | 36.5 | -12.9% | 0.6 | 0.8 | 30.6% / -22.5% | 60.4% / 37.9% | Not measured / Not measured |
| price | catboost | all | 52,440 | 34.3% | 34.9 | -16.0% | 0.6 | 0.8 | 27.8% / -23.4% | 64.1% / 46.4% | Not measured / Not measured |
| renewable | catboost | 12-24h | 1,800 | 40.1% | 7604.3 | -20.4% | 0.3 | 0.7 | 18.7% / -114.9% | 58.6% / 31.6% | 20.3% / -97.7% |
| renewable | catboost | 2-12h | 720 | 37.6% | 4394.2 | 11.1% | 0.6 | 0.8 | 28.8% / -30.7% | 36.9% / -2.0% | 33.0% / -14.2% |
| renewable | catboost | 24-36h | 2,160 | 39.9% | 7058.6 | -16.0% | 0.4 | 0.7 | 19.8% / -101.8% | 36.0% / -10.7% | 23.1% / -72.4% |
| renewable | catboost | 36-48h | 2,160 | 39.9% | 7070.1 | -15.8% | 0.4 | 0.7 | 19.8% / -102.1% | 60.5% / 34.0% | 23.1% / -72.7% |
| renewable | catboost | 48-64h | 1,440 | 40.1% | 8319.4 | -23.6% | 0.3 | 0.7 | 17.2% / -133.3% | 35.1% / -14.2% | 20.3% / -97.3% |
| renewable | catboost | all | 8,280 | 39.9% | 7167.8 | -17.0% | 0.3 | 0.7 | 19.5% / -104.4% | 47.4% / 15.8% | 22.5% / -77.4% |
| solar | catboost | 12-24h | 1,800 | 48.2% | 4909.6 | -43.5% | 0.4 | 0.8 | 24.6% / -95.9% | 135.7% / 64.5% | 5.9% / -720.8% |
| solar | catboost | 2-12h | 720 | 46.7% | 1279.3 | -30.0% | 0.5 | 0.9 | 28.8% / -62.6% | 100.8% / 53.6% | 7.3% / -538.4% |
| solar | catboost | 24-36h | 2,160 | 48.5% | 4115.2 | -43.4% | 0.4 | 0.9 | 24.6% / -96.9% | 66.1% / 26.7% | 5.9% / -717.3% |
| solar | catboost | 36-48h | 2,160 | 48.5% | 4117.4 | -43.4% | 0.4 | 0.9 | 24.6% / -97.0% | 159.6% / 69.6% | 5.9% / -717.7% |
| solar | catboost | 48-64h | 1,440 | 48.4% | 5507.9 | -44.6% | 0.4 | 0.8 | 24.1% / -100.9% | 63.0% / 23.1% | 5.8% / -740.8% |
| solar | catboost | all | 8,280 | 48.4% | 4284.1 | -43.3% | 0.4 | 0.9 | 24.6% / -96.5% | 107.1% / 54.9% | 5.9% / -717.4% |
| wind_onshore | catboost | 12-24h | 1,800 | 79.8% | 3147.2 | 40.3% | 0.4 | 0.5 | 75.1% / -6.2% | 63.4% / -25.9% | 13.3% / -501.0% |
| wind_onshore | catboost | 2-12h | 720 | 68.1% | 2776.0 | 33.3% | 0.5 | 0.6 | 62.0% / -9.8% | 41.2% / -65.2% | 10.3% / -563.7% |
| wind_onshore | catboost | 24-36h | 2,160 | 77.1% | 3159.4 | 38.1% | 0.4 | 0.5 | 70.6% / -9.1% | 61.5% / -25.4% | 13.0% / -493.0% |
| wind_onshore | catboost | 36-48h | 2,160 | 76.4% | 3131.9 | 38.0% | 0.4 | 0.5 | 70.6% / -8.1% | 75.9% / -0.7% | 13.0% / -487.9% |
| wind_onshore | catboost | 48-64h | 1,440 | 78.6% | 3230.2 | 39.1% | 0.5 | 0.6 | 74.9% / -4.9% | 69.9% / -12.5% | 14.3% / -447.5% |
| wind_onshore | catboost | all | 8,280 | 76.9% | 3128.6 | 38.3% | 0.4 | 0.5 | 71.6% / -7.5% | 65.4% / -17.7% | 13.1% / -489.4% |
| wind_offshore | xgboost | 12-24h | 1,200 | 117.0% | 571.1 | 83.3% | 0.2 | 0.2 | 91.2% / -28.3% | 80.9% / -44.7% | 27.0% / -333.7% |
| wind_offshore | xgboost | 2-12h | 480 | 109.6% | 564.5 | 79.7% | 0.1 | 0.1 | 79.1% / -38.6% | 69.9% / -56.8% | 22.6% / -384.7% |
| wind_offshore | xgboost | 24-36h | 1,440 | 114.4% | 602.7 | 72.6% | -0.0 | -0.0 | 89.3% / -28.2% | 85.0% / -34.6% | 25.5% / -348.3% |
| wind_offshore | xgboost | 36-48h | 1,440 | 114.7% | 604.1 | 72.8% | -0.0 | -0.0 | 89.3% / -28.5% | 100.2% / -14.5% | 25.5% / -349.3% |
| wind_offshore | xgboost | 48-64h | 960 | 120.6% | 642.3 | 69.4% | -0.1 | -0.1 | 94.2% / -28.1% | 93.8% / -28.5% | 26.9% / -347.7% |
| wind_offshore | xgboost | all | 5,520 | 115.7% | 599.8 | 74.9% | 0.0 | 0.0 | 89.7% / -29.1% | 88.5% / -30.8% | 25.8% / -348.1% |
| biomass | xgboost | 12-24h | 1,200 | 29.0% | 68.3 | -25.8% | 1.4 | 0.8 | 10.5% / -176.2% | 8.0% / -262.8% | Not measured / Not measured |
| biomass | xgboost | 2-12h | 480 | 28.7% | 68.5 | -26.6% | 1.4 | 0.8 | 10.2% / -181.5% | 7.0% / -307.8% | Not measured / Not measured |
| biomass | xgboost | 24-36h | 1,440 | 29.2% | 68.8 | -25.6% | 1.4 | 0.8 | 10.7% / -171.9% | 8.7% / -237.1% | Not measured / Not measured |
| biomass | xgboost | 36-48h | 1,440 | 29.2% | 68.8 | -25.6% | 1.4 | 0.8 | 10.7% / -172.1% | 9.8% / -199.7% | Not measured / Not measured |
| biomass | xgboost | 48-64h | 960 | 29.6% | 69.2 | -24.9% | 1.4 | 0.8 | 11.0% / -168.4% | 10.0% / -195.6% | Not measured / Not measured |
| biomass | xgboost | all | 5,520 | 29.2% | 68.7 | -25.6% | 1.4 | 0.8 | 10.7% / -173.1% | 8.9% / -228.3% | Not measured / Not measured |
| hydro_total | xgboost | 12-24h | 1,200 | 17.1% | 359.4 | 7.2% | 1.0 | 1.0 | 16.5% / -3.7% | 75.6% / 77.3% | Not measured / Not measured |
| hydro_total | xgboost | 2-12h | 480 | 15.4% | 326.3 | 3.4% | 0.9 | 1.0 | 17.3% / 10.8% | 105.8% / 85.4% | Not measured / Not measured |
| hydro_total | xgboost | 24-36h | 1,440 | 18.6% | 424.3 | 5.5% | 1.0 | 1.0 | 17.6% / -5.8% | 62.6% / 70.3% | Not measured / Not measured |
| hydro_total | xgboost | 36-48h | 1,440 | 18.8% | 428.6 | 5.6% | 1.0 | 1.0 | 17.6% / -6.8% | 69.1% / 72.9% | Not measured / Not measured |
| hydro_total | xgboost | 48-64h | 960 | 20.5% | 485.5 | 6.8% | 1.0 | 1.0 | 17.7% / -15.9% | 44.2% / 53.7% | Not measured / Not measured |
| hydro_total | xgboost | all | 5,520 | 18.4% | 413.5 | 5.9% | 1.0 | 1.0 | 17.3% / -6.2% | 67.1% / 72.6% | Not measured / Not measured |
| net_position | chronos-2-V010 | 24-36h | 90 | 41.1% | 674.8 | -6.1% | 0.7 | 0.9 | 37.0% / -11.1% | 43.1% / 4.5% | Not measured / Not measured |
| net_position | chronos-2-V010 | 36-48h | 2,166 | 79.9% | 1397.6 | 3.9% | 0.3 | 0.5 | 49.0% / -63.1% | 48.6% / -64.4% | Not measured / Not measured |
| net_position | chronos-2-V010 | 48-64h | 4,631 | 74.8% | 1338.5 | 3.3% | 0.4 | 0.6 | 56.7% / -32.1% | 57.7% / -29.7% | Not measured / Not measured |
| net_position | chronos-2-V010 | all | 6,887 | 76.0% | 1348.4 | 3.4% | 0.3 | 0.6 | 54.0% / -40.6% | 54.7% / -38.9% | Not measured / Not measured |

## Country × horizon detail

Rows with no paired observations say **Not measured**; zero is never substituted.

| type | model | country | horizon | n | WAPE | bias | slope | corr | D−7 skill | persistence skill | TSO skill |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| load | catboost | BG | 12-24h | 600 | 6.8% | -2.3% | 0.7 | 0.8 | -2.9% | 50.2% | -59.2% |
| load | catboost | BG | 2-12h | 240 | 5.0% | -0.5% | 0.8 | 0.8 | -12.6% | 73.4% | -36.1% |
| load | catboost | BG | 24-36h | 717 | 7.4% | -1.7% | 0.7 | 0.8 | -17.7% | 39.1% | -85.0% |
| load | catboost | BG | 36-48h | 717 | 7.8% | -2.1% | 0.7 | 0.8 | -24.0% | 49.5% | -95.0% |
| load | catboost | BG | 48-64h | 477 | 8.2% | -2.5% | 0.6 | 0.6 | -15.5% | 19.8% | -97.5% |
| load | catboost | CH | 12-24h | 600 | 6.6% | -0.1% | 0.5 | 0.7 | 20.5% | 43.8% | -83.3% |
| load | catboost | CH | 2-12h | 240 | 7.4% | 0.3% | 0.4 | 0.6 | 13.8% | 42.2% | -127.7% |
| load | catboost | CH | 24-36h | 719 | 6.7% | -0.1% | 0.5 | 0.7 | 18.2% | 38.7% | -89.5% |
| load | catboost | CH | 36-48h | 719 | 6.6% | -0.1% | 0.5 | 0.7 | 19.1% | 44.3% | -87.5% |
| load | catboost | CH | 48-64h | 479 | 6.1% | -0.3% | 0.4 | 0.6 | 23.5% | 37.5% | -67.2% |
| load | catboost | CZ | 12-24h | 600 | 6.8% | 0.4% | 0.7 | 0.9 | -56.3% | 64.3% | -363.8% |
| load | catboost | CZ | 2-12h | 240 | 7.1% | 4.0% | 0.8 | 0.9 | -71.4% | 59.0% | -364.1% |
| load | catboost | CZ | 24-36h | 720 | 8.9% | 0.8% | 0.7 | 0.8 | -103.2% | 29.8% | -495.0% |
| load | catboost | CZ | 36-48h | 720 | 9.0% | 0.7% | 0.7 | 0.8 | -105.4% | 59.3% | -501.3% |
| load | catboost | CZ | 48-64h | 480 | 9.5% | -0.6% | 0.6 | 0.7 | -109.9% | 27.9% | -534.3% |
| load | catboost | DE | 12-24h | 600 | 7.8% | 2.7% | 0.6 | 0.8 | -82.0% | 36.6% | -104.2% |
| load | catboost | DE | 2-12h | 240 | 7.5% | 3.7% | 0.8 | 0.8 | -83.9% | 50.6% | -102.8% |
| load | catboost | DE | 24-36h | 720 | 10.7% | 2.9% | 0.4 | 0.5 | -142.8% | 21.5% | -183.3% |
| load | catboost | DE | 36-48h | 720 | 10.8% | 2.8% | 0.4 | 0.5 | -145.7% | 39.3% | -186.7% |
| load | catboost | DE | 48-64h | 480 | 11.9% | 2.0% | 0.2 | 0.2 | -161.5% | 26.0% | -212.6% |
| load | catboost | EE | 12-24h | 600 | 13.0% | 2.0% | 0.2 | 0.4 | 15.1% | 27.6% | 8.2% |
| load | catboost | EE | 2-12h | 240 | 10.4% | 2.9% | 0.6 | 0.6 | 12.3% | 27.5% | -5.8% |
| load | catboost | EE | 24-36h | 720 | 12.8% | 1.6% | 0.3 | 0.4 | 13.0% | 24.0% | 3.0% |
| load | catboost | EE | 36-48h | 720 | 12.5% | 1.6% | 0.3 | 0.4 | 14.6% | 34.6% | 4.7% |
| load | catboost | EE | 48-64h | 480 | 13.3% | 0.3% | 0.1 | 0.2 | 16.7% | 25.2% | 9.4% |
| load | catboost | ES | 12-24h | 600 | 6.8% | 0.9% | 0.7 | 0.8 | -86.5% | 47.3% | -447.9% |
| load | catboost | ES | 2-12h | 240 | 6.4% | 0.4% | 0.5 | 0.6 | -156.2% | 77.4% | -540.8% |
| load | catboost | ES | 24-36h | 720 | 8.9% | 0.1% | 0.5 | 0.6 | -152.7% | 43.5% | -642.3% |
| load | catboost | ES | 36-48h | 720 | 9.0% | 0.1% | 0.5 | 0.6 | -155.8% | 39.4% | -651.5% |
| load | catboost | ES | 48-64h | 480 | 9.7% | -0.2% | 0.1 | 0.2 | -144.2% | 33.7% | -652.5% |
| load | catboost | FI | 12-24h | 599 | 2.9% | -0.5% | 0.7 | 0.9 | 23.8% | 58.9% | -22.5% |
| load | catboost | FI | 2-12h | 239 | 2.8% | 0.4% | 0.8 | 0.9 | 25.4% | 60.0% | -30.1% |
| load | catboost | FI | 24-36h | 717 | 3.5% | -0.6% | 0.7 | 0.8 | 7.0% | 35.2% | -52.1% |
| load | catboost | FI | 36-48h | 717 | 3.6% | -0.7% | 0.7 | 0.8 | 5.0% | 53.2% | -55.3% |
| load | catboost | FI | 48-64h | 478 | 3.6% | -1.3% | 0.6 | 0.7 | 1.9% | 29.3% | -55.0% |
| load | catboost | GR | 12-24h | 600 | 7.9% | 0.4% | 0.6 | 0.9 | 24.5% | 48.5% | -191.9% |
| load | catboost | GR | 2-12h | 240 | 7.1% | 4.4% | 0.7 | 0.8 | 18.1% | 70.2% | -174.3% |
| load | catboost | GR | 24-36h | 720 | 9.9% | 0.4% | 0.5 | 0.7 | 6.9% | 38.0% | -268.9% |
| load | catboost | GR | 36-48h | 720 | 10.0% | 0.4% | 0.5 | 0.7 | 5.5% | 43.6% | -274.5% |
| load | catboost | GR | 48-64h | 480 | 11.5% | -1.2% | 0.3 | 0.4 | -1.2% | 29.8% | -324.4% |
| load | catboost | HR | 12-24h | 600 | 7.0% | -1.8% | 0.7 | 0.9 | 39.2% | 64.0% | -140.2% |
| load | catboost | HR | 2-12h | 240 | 5.6% | -0.2% | 0.7 | 0.8 | 30.1% | 89.0% | -136.0% |
| load | catboost | HR | 24-36h | 720 | 8.3% | -2.6% | 0.7 | 0.8 | 26.4% | 65.1% | -202.9% |
| load | catboost | HR | 36-48h | 720 | 8.3% | -2.5% | 0.7 | 0.8 | 26.8% | 56.8% | -201.3% |
| load | catboost | HR | 48-64h | 480 | 9.7% | -3.5% | 0.5 | 0.7 | 23.1% | 37.0% | -234.4% |
| load | catboost | HU | 12-24h | 600 | 6.8% | 0.7% | 0.8 | 0.8 | 39.6% | 51.3% | -62.2% |
| load | catboost | HU | 2-12h | 240 | 6.4% | -1.0% | 0.6 | 0.6 | 15.7% | 75.6% | -33.8% |
| load | catboost | HU | 24-36h | 720 | 8.7% | 0.4% | 0.6 | 0.7 | 18.7% | 42.7% | -93.8% |
| load | catboost | HU | 36-48h | 720 | 8.8% | 0.3% | 0.6 | 0.7 | 17.9% | 43.2% | -95.6% |
| load | catboost | HU | 48-64h | 480 | 10.3% | 0.6% | 0.5 | 0.6 | 14.6% | 21.4% | -136.1% |
| load | catboost | IT | 12-24h | 600 | 8.8% | 2.9% | 0.7 | 0.8 | -18.4% | 43.6% | -223.8% |
| load | catboost | IT | 2-12h | 240 | 8.1% | 2.8% | 0.8 | 0.8 | -34.1% | 74.5% | -299.7% |
| load | catboost | IT | 24-36h | 720 | 10.3% | 1.7% | 0.6 | 0.7 | -38.9% | 45.2% | -274.9% |
| load | catboost | IT | 36-48h | 720 | 10.2% | 1.7% | 0.6 | 0.7 | -38.5% | 46.8% | -274.0% |
| load | catboost | IT | 48-64h | 480 | 11.0% | 1.1% | 0.3 | 0.4 | -38.2% | 40.3% | -262.2% |
| load | catboost | LT | 12-24h | 600 | 8.9% | 0.5% | 0.5 | 0.7 | 15.3% | 42.9% | 41.5% |
| load | catboost | LT | 2-12h | 240 | 7.6% | 0.8% | 0.7 | 0.8 | -14.1% | 57.0% | 50.7% |
| load | catboost | LT | 24-36h | 720 | 9.3% | 0.2% | 0.5 | 0.6 | 1.4% | 48.2% | 32.4% |
| load | catboost | LT | 36-48h | 720 | 9.3% | 0.1% | 0.5 | 0.6 | 1.2% | 51.0% | 32.3% |
| load | catboost | LT | 48-64h | 480 | 10.4% | -0.3% | 0.3 | 0.4 | 4.0% | 43.6% | 20.3% |
| load | catboost | LV | 12-24h | 597 | 6.5% | -0.4% | 0.8 | 0.8 | -3.7% | 56.3% | -43.0% |
| load | catboost | LV | 2-12h | 238 | 6.1% | -1.1% | 0.8 | 0.9 | -8.4% | 64.8% | -37.9% |
| load | catboost | LV | 24-36h | 717 | 7.3% | -0.1% | 0.7 | 0.8 | -21.7% | 47.0% | -71.4% |
| load | catboost | LV | 36-48h | 717 | 7.5% | -0.4% | 0.7 | 0.8 | -24.4% | 59.8% | -75.2% |
| load | catboost | LV | 48-64h | 479 | 6.9% | -0.5% | 0.6 | 0.8 | -12.7% | 47.9% | -65.6% |
| load | catboost | NL | 12-24h | 600 | 39.6% | 34.7% | 0.0 | 0.2 | -111.0% | 33.7% | -2.4% |
| load | catboost | NL | 2-12h | 240 | 8.1% | -0.4% | 0.0 | 0.1 | -47.4% | 20.7% | 35.8% |
| load | catboost | NL | 24-36h | 720 | 31.9% | 26.9% | 0.1 | 0.3 | -103.0% | -6.2% | 2.4% |
| load | catboost | NL | 36-48h | 720 | 32.0% | 27.0% | 0.1 | 0.3 | -103.5% | 39.4% | 2.1% |
| load | catboost | NL | 48-64h | 480 | 48.8% | 46.2% | 0.1 | 0.7 | -113.6% | -12.8% | -4.6% |
| load | catboost | NO | 12-24h | 598 | 2.0% | -0.1% | 0.8 | 0.9 | 16.6% | 57.1% | -13.9% |
| load | catboost | NO | 2-12h | 240 | 1.9% | 0.4% | 0.9 | 0.9 | 21.1% | 81.0% | -34.0% |
| load | catboost | NO | 24-36h | 718 | 2.5% | -0.5% | 0.8 | 0.8 | -6.8% | 56.9% | -32.7% |
| load | catboost | NO | 36-48h | 718 | 2.6% | -0.5% | 0.7 | 0.8 | -9.4% | 51.8% | -35.9% |
| load | catboost | NO | 48-64h | 478 | 2.9% | -1.0% | 0.5 | 0.6 | -28.0% | 36.1% | -40.2% |
| load | catboost | PL | 12-24h | 600 | 7.2% | 0.6% | 0.7 | 0.8 | -45.9% | 44.2% | -97.0% |
| load | catboost | PL | 2-12h | 240 | 6.5% | 2.4% | 0.7 | 0.8 | -54.4% | 73.1% | -102.3% |
| load | catboost | PL | 24-36h | 720 | 9.8% | 0.5% | 0.6 | 0.7 | -103.3% | 37.5% | -182.2% |
| load | catboost | PL | 36-48h | 720 | 9.9% | 0.3% | 0.5 | 0.6 | -106.0% | 39.8% | -185.9% |
| load | catboost | PL | 48-64h | 480 | 10.2% | -0.9% | 0.4 | 0.4 | -102.2% | 32.0% | -186.7% |
| load | catboost | PT | 12-24h | 600 | 6.9% | 1.2% | 0.7 | 0.8 | -61.4% | 56.8% | -410.1% |
| load | catboost | PT | 2-12h | 240 | 5.5% | 0.4% | 0.6 | 0.7 | -44.0% | 84.4% | -371.8% |
| load | catboost | PT | 24-36h | 720 | 8.5% | 1.4% | 0.6 | 0.7 | -105.1% | 53.2% | -564.1% |
| load | catboost | PT | 36-48h | 720 | 8.7% | 1.3% | 0.6 | 0.7 | -110.0% | 50.0% | -579.8% |
| load | catboost | PT | 48-64h | 480 | 10.0% | 1.6% | 0.3 | 0.3 | -133.7% | 34.6% | -653.5% |
| load | catboost | RO | 12-24h | 600 | 8.7% | 1.3% | 0.6 | 0.8 | 1.3% | 51.6% | -261.7% |
| load | catboost | RO | 2-12h | 240 | 8.1% | 0.2% | 0.3 | 0.4 | -33.0% | 72.1% | -359.6% |
| load | catboost | RO | 24-36h | 720 | 10.8% | 0.8% | 0.4 | 0.6 | -28.1% | 43.5% | -378.4% |
| load | catboost | RO | 36-48h | 720 | 10.8% | 0.7% | 0.4 | 0.5 | -29.1% | 41.9% | -381.9% |
| load | catboost | RO | 48-64h | 480 | 12.0% | 0.9% | 0.4 | 0.5 | -26.7% | 28.4% | -384.8% |
| load | catboost | SE | 12-24h | 600 | 5.1% | 2.2% | 0.6 | 0.8 | 17.5% | 41.5% | 16.8% |
| load | catboost | SE | 2-12h | 240 | 5.4% | 3.6% | 0.7 | 0.8 | 12.9% | 61.7% | 5.2% |
| load | catboost | SE | 24-36h | 718 | 5.3% | 2.1% | 0.6 | 0.8 | 14.0% | 45.4% | 14.8% |
| load | catboost | SE | 36-48h | 718 | 5.4% | 2.1% | 0.6 | 0.8 | 12.6% | 47.6% | 13.4% |
| load | catboost | SE | 48-64h | 478 | 5.7% | 1.5% | 0.3 | 0.5 | 6.9% | 30.5% | 11.3% |
| load | catboost | SI | 12-24h | 600 | 12.3% | 1.9% | 0.5 | 0.6 | -110.0% | 30.8% | -145.1% |
| load | catboost | SI | 2-12h | 240 | 14.7% | 9.3% | 0.5 | 0.6 | -191.5% | 44.5% | -282.2% |
| load | catboost | SI | 24-36h | 720 | 14.1% | 2.7% | 0.4 | 0.5 | -145.7% | 16.2% | -190.1% |
| load | catboost | SI | 36-48h | 720 | 14.5% | 2.7% | 0.4 | 0.5 | -152.7% | 29.9% | -198.3% |
| load | catboost | SI | 48-64h | 480 | 14.1% | 0.3% | 0.3 | 0.3 | -133.7% | 12.9% | -166.9% |
| load | catboost | SK | 12-24h | 600 | 5.7% | -0.3% | 0.7 | 0.8 | 9.0% | 37.2% | -140.5% |
| load | catboost | SK | 2-12h | 240 | 5.6% | 0.5% | 0.7 | 0.8 | -12.2% | 65.4% | -168.2% |
| load | catboost | SK | 24-36h | 720 | 7.4% | -0.2% | 0.6 | 0.7 | -20.7% | 31.1% | -210.8% |
| load | catboost | SK | 36-48h | 720 | 7.5% | -0.3% | 0.6 | 0.6 | -22.7% | 37.0% | -215.9% |
| load | catboost | SK | 48-64h | 480 | 8.0% | -0.6% | 0.4 | 0.5 | -20.6% | 22.6% | -218.9% |
| price | catboost | AT | 12-24h | 600 | 29.3% | -14.3% | 0.6 | 0.8 | -28.6% | 58.2% | Not measured |
| price | catboost | AT | 2-12h | 240 | 26.0% | -22.8% | 0.4 | 0.4 | -108.4% | 24.1% | Not measured |
| price | catboost | AT | 24-36h | 720 | 29.2% | -16.7% | 0.5 | 0.8 | -47.4% | 40.2% | Not measured |
| price | catboost | AT | 36-48h | 720 | 30.8% | -15.3% | 0.5 | 0.7 | -55.7% | 50.5% | Not measured |
| price | catboost | AT | 48-64h | 480 | 34.3% | -9.8% | 0.5 | 0.7 | -40.7% | 43.5% | Not measured |
| price | catboost | BG | 12-24h | 600 | 32.7% | -15.7% | 0.6 | 0.8 | -13.2% | 58.8% | Not measured |
| price | catboost | BG | 2-12h | 240 | 27.0% | -21.8% | 0.5 | 0.7 | -45.7% | 37.2% | Not measured |
| price | catboost | BG | 24-36h | 720 | 31.0% | -18.2% | 0.6 | 0.8 | -20.8% | 34.3% | Not measured |
| price | catboost | BG | 36-48h | 720 | 33.6% | -18.7% | 0.6 | 0.7 | -30.6% | 56.3% | Not measured |
| price | catboost | BG | 48-64h | 480 | 35.2% | -14.8% | 0.6 | 0.8 | -18.0% | 34.2% | Not measured |
| price | catboost | CH | 12-24h | 600 | 22.6% | -8.0% | 0.5 | 0.8 | -12.2% | 51.9% | Not measured |
| price | catboost | CH | 2-12h | 240 | 16.1% | -11.5% | 0.4 | 0.5 | -43.2% | 40.6% | Not measured |
| price | catboost | CH | 24-36h | 720 | 21.5% | -9.3% | 0.5 | 0.8 | -23.1% | 36.2% | Not measured |
| price | catboost | CH | 36-48h | 720 | 22.8% | -8.5% | 0.5 | 0.7 | -30.4% | 49.4% | Not measured |
| price | catboost | CH | 48-64h | 480 | 26.0% | -6.4% | 0.5 | 0.8 | -23.3% | 36.0% | Not measured |
| price | catboost | CZ | 12-24h | 600 | 31.7% | -11.5% | 0.5 | 0.8 | -23.2% | 61.9% | Not measured |
| price | catboost | CZ | 2-12h | 240 | 25.3% | -21.6% | 0.2 | 0.3 | -62.1% | 42.9% | Not measured |
| price | catboost | CZ | 24-36h | 720 | 30.8% | -14.7% | 0.5 | 0.8 | -38.3% | 47.5% | Not measured |
| price | catboost | CZ | 36-48h | 720 | 33.2% | -14.6% | 0.4 | 0.7 | -49.0% | 54.3% | Not measured |
| price | catboost | CZ | 48-64h | 480 | 37.3% | -9.4% | 0.5 | 0.8 | -40.2% | 47.4% | Not measured |
| price | catboost | EE | 12-24h | 600 | 83.8% | -26.3% | 0.3 | 0.5 | 27.3% | 53.9% | Not measured |
| price | catboost | EE | 2-12h | 240 | 89.5% | -39.0% | 0.2 | 0.4 | 39.2% | 57.6% | Not measured |
| price | catboost | EE | 24-36h | 720 | 86.8% | -36.1% | 0.2 | 0.4 | 25.7% | 43.7% | Not measured |
| price | catboost | EE | 36-48h | 720 | 92.0% | -35.6% | 0.1 | 0.3 | 21.3% | 46.3% | Not measured |
| price | catboost | EE | 48-64h | 480 | 88.9% | -38.8% | 0.2 | 0.3 | 13.9% | 22.1% | Not measured |
| price | catboost | FI | 12-24h | 600 | 70.1% | -48.8% | 0.3 | 0.4 | 7.7% | 8.9% | Not measured |
| price | catboost | FI | 2-12h | 240 | 72.6% | -53.0% | 0.4 | 0.4 | 5.1% | 5.5% | Not measured |
| price | catboost | FI | 24-36h | 720 | 76.9% | -54.3% | 0.2 | 0.2 | 0.7% | 12.7% | Not measured |
| price | catboost | FI | 36-48h | 720 | 80.8% | -53.1% | 0.1 | 0.2 | -4.3% | 10.0% | Not measured |
| price | catboost | FI | 48-64h | 480 | 79.2% | -58.0% | 0.1 | 0.2 | -1.7% | 8.3% | Not measured |
| price | catboost | GR | 12-24h | 600 | 31.9% | -11.3% | 0.7 | 0.8 | -10.9% | 64.9% | Not measured |
| price | catboost | GR | 2-12h | 240 | 27.7% | -22.5% | 0.5 | 0.7 | -66.3% | 38.8% | Not measured |
| price | catboost | GR | 24-36h | 720 | 29.4% | -13.6% | 0.6 | 0.8 | -15.0% | 40.3% | Not measured |
| price | catboost | GR | 36-48h | 720 | 31.1% | -13.4% | 0.6 | 0.8 | -21.7% | 64.1% | Not measured |
| price | catboost | GR | 48-64h | 480 | 32.9% | -7.9% | 0.6 | 0.8 | -6.2% | 38.1% | Not measured |
| price | catboost | HR | 12-24h | 600 | 28.9% | -1.5% | 0.6 | 0.8 | -4.2% | 62.9% | Not measured |
| price | catboost | HR | 2-12h | 240 | 20.5% | -14.2% | 0.4 | 0.5 | -26.6% | 59.6% | Not measured |
| price | catboost | HR | 24-36h | 720 | 27.0% | -4.8% | 0.6 | 0.8 | -12.9% | 51.7% | Not measured |
| price | catboost | HR | 36-48h | 720 | 30.3% | -4.9% | 0.5 | 0.7 | -26.5% | 57.4% | Not measured |
| price | catboost | HR | 48-64h | 480 | 34.4% | 1.2% | 0.5 | 0.7 | -20.8% | 43.8% | Not measured |
| price | catboost | HU | 12-24h | 600 | 28.7% | -5.9% | 0.6 | 0.8 | -11.7% | 64.3% | Not measured |
| price | catboost | HU | 2-12h | 240 | 24.1% | -21.7% | 0.4 | 0.5 | -73.8% | 53.1% | Not measured |
| price | catboost | HU | 24-36h | 720 | 27.5% | -9.2% | 0.6 | 0.8 | -20.9% | 53.2% | Not measured |
| price | catboost | HU | 36-48h | 720 | 30.6% | -9.2% | 0.6 | 0.7 | -34.8% | 57.7% | Not measured |
| price | catboost | HU | 48-64h | 480 | 34.0% | -2.5% | 0.6 | 0.8 | -21.3% | 48.2% | Not measured |
| price | catboost | IT | 12-24h | 600 | 11.1% | -6.6% | 0.5 | 0.8 | 12.3% | 52.5% | Not measured |
| price | catboost | IT | 2-12h | 240 | 9.5% | -7.8% | 0.5 | 0.6 | -9.3% | 57.5% | Not measured |
| price | catboost | IT | 24-36h | 720 | 12.1% | -8.4% | 0.5 | 0.7 | -0.5% | 42.4% | Not measured |
| price | catboost | IT | 36-48h | 720 | 13.0% | -8.7% | 0.4 | 0.7 | -8.4% | 40.7% | Not measured |
| price | catboost | IT | 48-64h | 480 | 14.5% | -8.9% | 0.4 | 0.7 | -6.1% | 30.2% | Not measured |
| price | catboost | LT | 12-24h | 600 | 66.0% | -10.5% | 0.4 | 0.6 | 9.4% | 50.9% | Not measured |
| price | catboost | LT | 2-12h | 240 | 67.7% | -26.2% | 0.2 | 0.5 | 16.9% | 31.7% | Not measured |
| price | catboost | LT | 24-36h | 720 | 65.1% | -19.5% | 0.3 | 0.6 | 5.8% | 32.2% | Not measured |
| price | catboost | LT | 36-48h | 720 | 66.5% | -17.4% | 0.3 | 0.5 | 3.7% | 47.9% | Not measured |
| price | catboost | LT | 48-64h | 480 | 63.9% | -14.4% | 0.4 | 0.6 | -2.7% | 31.6% | Not measured |
| price | catboost | LV | 12-24h | 600 | 66.5% | -14.4% | 0.4 | 0.6 | 8.1% | 51.9% | Not measured |
| price | catboost | LV | 2-12h | 240 | 67.4% | -33.0% | 0.2 | 0.5 | 17.2% | 31.9% | Not measured |
| price | catboost | LV | 24-36h | 720 | 66.6% | -22.7% | 0.3 | 0.5 | 2.9% | 30.7% | Not measured |
| price | catboost | LV | 36-48h | 720 | 67.9% | -19.9% | 0.3 | 0.5 | 1.1% | 47.8% | Not measured |
| price | catboost | LV | 48-64h | 480 | 64.4% | -17.8% | 0.4 | 0.6 | -4.8% | 33.1% | Not measured |
| price | catboost | NL | 12-24h | 600 | 48.5% | -37.8% | 0.5 | 0.8 | -102.3% | 48.0% | Not measured |
| price | catboost | NL | 2-12h | 240 | 43.9% | -43.1% | 0.4 | 0.5 | -252.0% | -36.5% | Not measured |
| price | catboost | NL | 24-36h | 720 | 47.4% | -38.9% | 0.5 | 0.8 | -133.1% | 19.7% | Not measured |
| price | catboost | NL | 36-48h | 720 | 49.3% | -39.9% | 0.5 | 0.7 | -142.4% | 37.0% | Not measured |
| price | catboost | NL | 48-64h | 480 | 51.8% | -37.4% | 0.5 | 0.8 | -99.1% | 35.6% | Not measured |
| price | catboost | NO | 12-24h | 600 | 27.2% | -14.6% | 0.2 | 0.4 | -56.3% | 8.8% | Not measured |
| price | catboost | NO | 2-12h | 240 | 28.3% | -27.6% | 0.3 | 0.2 | -126.7% | -193.9% | Not measured |
| price | catboost | NO | 24-36h | 720 | 29.3% | -19.1% | 0.1 | 0.3 | -83.8% | -33.1% | Not measured |
| price | catboost | NO | 36-48h | 720 | 31.2% | -18.8% | 0.0 | 0.0 | -95.9% | -16.0% | Not measured |
| price | catboost | NO | 48-64h | 480 | 31.7% | -13.2% | 0.0 | 0.1 | -77.0% | -2.6% | Not measured |
| price | catboost | PL | 12-24h | 600 | 45.6% | -38.0% | 0.4 | 0.8 | -58.0% | 42.6% | Not measured |
| price | catboost | PL | 2-12h | 240 | 45.2% | -44.2% | 0.2 | 0.3 | -191.3% | 28.9% | Not measured |
| price | catboost | PL | 24-36h | 720 | 45.0% | -38.8% | 0.4 | 0.7 | -77.9% | 25.0% | Not measured |
| price | catboost | PL | 36-48h | 720 | 46.6% | -39.5% | 0.4 | 0.7 | -83.9% | 35.0% | Not measured |
| price | catboost | PL | 48-64h | 480 | 47.0% | -37.7% | 0.4 | 0.8 | -52.0% | 20.9% | Not measured |
| price | catboost | RO | 12-24h | 600 | 29.6% | -9.6% | 0.6 | 0.8 | -6.3% | 64.6% | Not measured |
| price | catboost | RO | 2-12h | 240 | 24.1% | -20.4% | 0.4 | 0.5 | -41.6% | 56.6% | Not measured |
| price | catboost | RO | 24-36h | 720 | 28.1% | -12.1% | 0.6 | 0.8 | -15.1% | 52.4% | Not measured |
| price | catboost | RO | 36-48h | 720 | 30.8% | -12.0% | 0.5 | 0.7 | -25.8% | 60.3% | Not measured |
| price | catboost | RO | 48-64h | 480 | 33.5% | -7.4% | 0.6 | 0.8 | -16.4% | 46.0% | Not measured |
| price | catboost | SE | 12-24h | 600 | 60.5% | -32.1% | 0.4 | 0.6 | -9.6% | 38.2% | Not measured |
| price | catboost | SE | 2-12h | 240 | 63.7% | -39.5% | 0.3 | 0.5 | -5.7% | 24.6% | Not measured |
| price | catboost | SE | 24-36h | 720 | 62.0% | -36.5% | 0.3 | 0.5 | -14.8% | 30.8% | Not measured |
| price | catboost | SE | 36-48h | 720 | 63.9% | -34.5% | 0.3 | 0.4 | -18.3% | 36.0% | Not measured |
| price | catboost | SE | 48-64h | 480 | 63.1% | -34.4% | 0.3 | 0.5 | -24.7% | 29.4% | Not measured |
| price | catboost | SI | 12-24h | 600 | 26.8% | 1.6% | 0.6 | 0.8 | -0.5% | 64.3% | Not measured |
| price | catboost | SI | 2-12h | 240 | 18.8% | -8.9% | 0.5 | 0.4 | -41.7% | 58.8% | Not measured |
| price | catboost | SI | 24-36h | 720 | 24.5% | -1.2% | 0.6 | 0.8 | -6.1% | 56.2% | Not measured |
| price | catboost | SI | 36-48h | 720 | 27.7% | -2.4% | 0.5 | 0.7 | -20.0% | 58.6% | Not measured |
| price | catboost | SI | 48-64h | 480 | 31.5% | 2.8% | 0.6 | 0.8 | -8.7% | 51.5% | Not measured |
| price | catboost | SK | 12-24h | 600 | 30.7% | -14.8% | 0.5 | 0.8 | -19.1% | 62.0% | Not measured |
| price | catboost | SK | 2-12h | 240 | 27.4% | -24.6% | 0.3 | 0.4 | -82.1% | 43.1% | Not measured |
| price | catboost | SK | 24-36h | 720 | 30.7% | -18.0% | 0.5 | 0.8 | -32.2% | 47.4% | Not measured |
| price | catboost | SK | 36-48h | 720 | 32.1% | -16.9% | 0.5 | 0.8 | -38.4% | 55.8% | Not measured |
| price | catboost | SK | 48-64h | 480 | 35.6% | -13.7% | 0.5 | 0.8 | -26.2% | 47.1% | Not measured |
| renewable | catboost | BE | 12-24h | 600 | 36.8% | 27.7% | 0.6 | 0.9 | -20.9% | 48.0% | -180.5% |
| renewable | catboost | BE | 2-12h | 240 | 104.3% | 103.6% | 0.5 | 0.7 | -143.4% | -35.0% | -466.2% |
| renewable | catboost | BE | 24-36h | 720 | 43.2% | 34.3% | 0.6 | 0.9 | -32.9% | 15.6% | -181.0% |
| renewable | catboost | BE | 36-48h | 720 | 43.2% | 34.5% | 0.6 | 0.9 | -32.9% | 42.9% | -180.8% |
| renewable | catboost | BE | 48-64h | 480 | 32.4% | 22.2% | 0.5 | 0.8 | -6.5% | 30.3% | -119.3% |
| renewable | catboost | DE | 12-24h | 600 | 48.8% | -40.3% | 0.0 | 0.1 | -171.3% | 22.7% | -184.9% |
| renewable | catboost | DE | 2-12h | 240 | 34.5% | -4.2% | 0.0 | 0.2 | -13.2% | -6.1% | -24.7% |
| renewable | catboost | DE | 24-36h | 720 | 47.0% | -35.7% | 0.0 | 0.1 | -142.4% | -32.1% | -140.6% |
| renewable | catboost | DE | 36-48h | 720 | 47.0% | -35.6% | 0.0 | 0.1 | -142.2% | 29.7% | -140.4% |
| renewable | catboost | DE | 48-64h | 480 | 50.2% | -44.6% | 0.0 | 0.1 | -210.4% | -41.5% | -192.3% |
| renewable | catboost | FR | 12-24h | 600 | 20.4% | 14.8% | 0.7 | 0.9 | -17.6% | 54.5% | 30.7% |
| renewable | catboost | FR | 2-12h | 240 | 32.8% | 29.7% | 0.8 | 0.7 | -46.8% | 17.2% | 31.1% |
| renewable | catboost | FR | 24-36h | 720 | 22.8% | 17.0% | 0.7 | 0.8 | -29.6% | 31.8% | 31.0% |
| renewable | catboost | FR | 36-48h | 720 | 23.1% | 17.4% | 0.7 | 0.8 | -31.3% | 45.7% | 30.1% |
| renewable | catboost | FR | 48-64h | 480 | 19.2% | 12.7% | 0.5 | 0.7 | -18.1% | 39.1% | 33.6% |
| solar | catboost | BE | 12-24h | 600 | 22.9% | -15.1% | 0.8 | 0.9 | 30.5% | 83.0% | -150.0% |
| solar | catboost | BE | 2-12h | 240 | 22.9% | -17.7% | 0.9 | 1.0 | 26.4% | 77.1% | -98.2% |
| solar | catboost | BE | 24-36h | 720 | 23.5% | -14.7% | 0.8 | 0.9 | 28.7% | 66.4% | -156.4% |
| solar | catboost | BE | 36-48h | 720 | 23.7% | -14.7% | 0.8 | 0.9 | 28.0% | 85.1% | -158.7% |
| solar | catboost | BE | 48-64h | 480 | 23.7% | -8.1% | 0.8 | 0.9 | 28.7% | 64.4% | -167.0% |
| solar | catboost | DE | 12-24h | 600 | 62.2% | -61.5% | 0.3 | 0.9 | -156.7% | 53.5% | -1173.9% |
| solar | catboost | DE | 2-12h | 240 | 51.6% | -46.3% | 0.4 | 1.0 | -75.4% | 48.3% | -751.3% |
| solar | catboost | DE | 24-36h | 720 | 62.4% | -61.3% | 0.3 | 1.0 | -157.4% | 1.2% | -1177.4% |
| solar | catboost | DE | 36-48h | 720 | 62.4% | -61.3% | 0.3 | 1.0 | -157.4% | 60.9% | -1177.5% |
| solar | catboost | DE | 48-64h | 480 | 63.9% | -63.4% | 0.3 | 1.0 | -171.6% | -7.7% | -1253.1% |
| solar | catboost | FR | 12-24h | 600 | 19.8% | -5.5% | 0.9 | 1.0 | 13.0% | 86.0% | -169.0% |
| solar | catboost | FR | 2-12h | 240 | 38.0% | 34.1% | 1.2 | 1.0 | -54.9% | 63.7% | -259.2% |
| solar | catboost | FR | 24-36h | 720 | 20.2% | -6.0% | 0.9 | 1.0 | 11.2% | 72.1% | -166.6% |
| solar | catboost | FR | 36-48h | 720 | 20.2% | -6.0% | 0.9 | 1.0 | 11.1% | 87.3% | -166.8% |
| solar | catboost | FR | 48-64h | 480 | 17.8% | -9.4% | 0.9 | 0.9 | 21.2% | 74.9% | -142.5% |
| wind_onshore | catboost | BE | 12-24h | 600 | 202.6% | 195.1% | 0.3 | 0.3 | -140.3% | -132.7% | -788.7% |
| wind_onshore | catboost | BE | 2-12h | 240 | 246.7% | 242.7% | 0.2 | 0.2 | -160.7% | -177.4% | -920.6% |
| wind_onshore | catboost | BE | 24-36h | 720 | 189.6% | 184.4% | 0.3 | 0.3 | -126.6% | -118.3% | -772.0% |
| wind_onshore | catboost | BE | 36-48h | 720 | 191.0% | 184.6% | 0.2 | 0.2 | -128.4% | -102.8% | -778.7% |
| wind_onshore | catboost | BE | 48-64h | 480 | 168.4% | 161.2% | 0.3 | 0.3 | -112.0% | -107.1% | -709.4% |
| wind_onshore | catboost | DE | 12-24h | 600 | 62.6% | 10.0% | -0.1 | -0.2 | 23.4% | 1.0% | -436.2% |
| wind_onshore | catboost | DE | 2-12h | 240 | 55.9% | 10.0% | -0.1 | -0.2 | 13.4% | -36.5% | -479.5% |
| wind_onshore | catboost | DE | 24-36h | 720 | 62.3% | 11.0% | -0.1 | -0.3 | 18.1% | 5.3% | -427.4% |
| wind_onshore | catboost | DE | 36-48h | 720 | 61.6% | 10.6% | -0.1 | -0.2 | 19.0% | 22.1% | -421.6% |
| wind_onshore | catboost | DE | 48-64h | 480 | 61.6% | 9.7% | -0.0 | -0.0 | 24.6% | 18.3% | -379.0% |
| wind_onshore | catboost | FR | 12-24h | 600 | 107.7% | 97.8% | 0.1 | 0.1 | -89.8% | -77.8% | -567.4% |
| wind_onshore | catboost | FR | 2-12h | 240 | 77.8% | 66.1% | 0.2 | 0.2 | -48.3% | -111.9% | -661.3% |
| wind_onshore | catboost | FR | 24-36h | 720 | 99.1% | 86.2% | 0.1 | 0.1 | -79.2% | -109.5% | -569.2% |
| wind_onshore | catboost | FR | 36-48h | 720 | 98.1% | 87.0% | 0.1 | 0.1 | -77.5% | -49.9% | -563.0% |
| wind_onshore | catboost | FR | 48-64h | 480 | 109.0% | 96.8% | 0.1 | 0.1 | -91.9% | -102.4% | -532.3% |
| wind_offshore | xgboost | BE | 12-24h | 600 | 163.2% | 128.5% | 0.2 | 0.2 | -58.6% | -83.6% | -489.0% |
| wind_offshore | xgboost | BE | 2-12h | 240 | 191.3% | 170.3% | 0.2 | 0.2 | -112.4% | -94.4% | -593.8% |
| wind_offshore | xgboost | BE | 24-36h | 720 | 157.3% | 116.6% | 0.0 | 0.0 | -51.1% | -45.8% | -494.8% |
| wind_offshore | xgboost | BE | 36-48h | 720 | 158.9% | 116.2% | 0.0 | 0.0 | -52.6% | -31.2% | -500.8% |
| wind_offshore | xgboost | BE | 48-64h | 480 | 160.6% | 94.2% | -0.1 | -0.1 | -46.5% | -46.0% | -517.5% |
| wind_offshore | xgboost | FR | 12-24h | 600 | 79.2% | 46.3% | 0.2 | 0.2 | 3.0% | -6.5% | -200.1% |
| wind_offshore | xgboost | FR | 2-12h | 240 | 57.7% | 22.2% | 0.3 | 0.4 | 20.0% | -11.5% | -196.6% |
| wind_offshore | xgboost | FR | 24-36h | 720 | 79.3% | 36.5% | 0.0 | 0.0 | -2.9% | -19.7% | -220.2% |
| wind_offshore | xgboost | FR | 36-48h | 720 | 78.5% | 37.4% | 0.0 | 0.0 | -1.8% | 5.5% | -217.0% |
| wind_offshore | xgboost | FR | 48-64h | 480 | 83.7% | 46.5% | 0.0 | 0.1 | -4.7% | -6.1% | -201.1% |
| biomass | xgboost | BE | 12-24h | 600 | 67.3% | -66.7% | 0.1 | 0.3 | -238.0% | -318.0% | Not measured |
| biomass | xgboost | BE | 2-12h | 240 | 67.3% | -66.7% | 0.1 | 0.4 | -239.3% | -369.2% | Not measured |
| biomass | xgboost | BE | 24-36h | 720 | 67.0% | -66.5% | 0.1 | 0.4 | -227.4% | -282.4% | Not measured |
| biomass | xgboost | BE | 36-48h | 720 | 67.0% | -66.5% | 0.1 | 0.4 | -227.3% | -244.5% | Not measured |
| biomass | xgboost | BE | 48-64h | 480 | 67.0% | -66.4% | 0.0 | 0.3 | -222.3% | -248.2% | Not measured |
| biomass | xgboost | FR | 12-24h | 600 | 3.7% | 1.2% | 0.2 | 0.2 | 13.3% | -40.7% | Not measured |
| biomass | xgboost | FR | 2-12h | 240 | 3.2% | 0.0% | 0.2 | 0.2 | 15.6% | -45.9% | Not measured |
| biomass | xgboost | FR | 24-36h | 720 | 4.2% | 1.4% | -0.1 | -0.1 | 2.4% | -50.2% | Not measured |
| biomass | xgboost | FR | 36-48h | 720 | 4.3% | 1.5% | -0.1 | -0.1 | 1.2% | -27.5% | Not measured |
| biomass | xgboost | FR | 48-64h | 480 | 4.8% | 2.5% | -0.2 | -0.2 | -5.7% | -23.6% | Not measured |
| hydro_total | xgboost | BE | 12-24h | 600 | 103.6% | 44.5% | 0.5 | 0.9 | -168.2% | 71.5% | Not measured |
| hydro_total | xgboost | BE | 2-12h | 240 | 376.6% | 312.5% | 0.3 | 0.3 | -152.0% | 86.1% | Not measured |
| hydro_total | xgboost | BE | 24-36h | 720 | 89.7% | 19.6% | 0.5 | 0.8 | -121.8% | 71.1% | Not measured |
| hydro_total | xgboost | BE | 36-48h | 720 | 89.8% | 19.8% | 0.5 | 0.8 | -121.9% | 67.2% | Not measured |
| hydro_total | xgboost | BE | 48-64h | 480 | 68.7% | -0.5% | 0.5 | 0.9 | -111.7% | 48.8% | Not measured |
| hydro_total | xgboost | FR | 12-24h | 600 | 14.6% | 6.1% | 0.8 | 0.9 | 8.2% | 78.3% | Not measured |
| hydro_total | xgboost | FR | 2-12h | 240 | 12.8% | 1.1% | 0.5 | 0.6 | 21.9% | 85.2% | Not measured |
| hydro_total | xgboost | FR | 24-36h | 720 | 16.1% | 5.0% | 0.8 | 0.9 | 3.9% | 70.2% | Not measured |
| hydro_total | xgboost | FR | 36-48h | 720 | 16.3% | 5.1% | 0.8 | 0.9 | 2.7% | 73.7% | Not measured |
| hydro_total | xgboost | FR | 48-64h | 480 | 18.2% | 7.2% | 0.8 | 0.9 | -7.3% | 54.5% | Not measured |
| net_position | chronos-2-V010 | AT | 24-36h | 5 | 25.1% | -15.6% | 0.4 | 0.9 | -24.1% | 59.6% | Not measured |
| net_position | chronos-2-V010 | AT | 36-48h | 115 | 86.4% | 69.1% | 0.0 | 0.1 | -82.5% | -53.5% | Not measured |
| net_position | chronos-2-V010 | AT | 48-64h | 245 | 90.4% | 41.0% | 0.1 | 0.3 | -9.1% | -14.6% | Not measured |
| net_position | chronos-2-V010 | BE | 24-36h | 5 | 36.2% | 36.2% | 0.2 | 0.8 | -336.6% | -100.3% | Not measured |
| net_position | chronos-2-V010 | BE | 36-48h | 115 | 70.1% | 68.8% | 0.4 | 0.2 | -171.8% | -182.8% | Not measured |
| net_position | chronos-2-V010 | BE | 48-64h | 245 | 73.4% | 68.6% | 0.2 | 0.1 | -170.7% | -87.4% | Not measured |
| net_position | chronos-2-V010 | BG | 24-36h | 5 | 97.2% | 97.2% | 0.4 | 0.7 | -48.7% | -91.2% | Not measured |
| net_position | chronos-2-V010 | BG | 36-48h | 115 | 60.4% | -20.1% | 0.5 | 0.8 | 4.7% | 2.5% | Not measured |
| net_position | chronos-2-V010 | BG | 48-64h | 245 | 59.1% | -14.4% | 0.5 | 0.8 | 7.6% | 2.9% | Not measured |
| net_position | chronos-2-V010 | CZ | 24-36h | 5 | 14.6% | -14.6% | -0.7 | -0.8 | 31.4% | -158.4% | Not measured |
| net_position | chronos-2-V010 | CZ | 36-48h | 115 | 86.9% | -78.7% | 0.2 | 0.2 | -246.0% | -146.3% | Not measured |
| net_position | chronos-2-V010 | CZ | 48-64h | 245 | 91.9% | -81.1% | 0.4 | 0.4 | -105.2% | -88.1% | Not measured |
| net_position | chronos-2-V010 | DE | 36-48h | 96 | 90.1% | 65.9% | 0.1 | 0.2 | -58.8% | -22.0% | Not measured |
| net_position | chronos-2-V010 | DE | 48-64h | 221 | 79.0% | 39.2% | 0.4 | 0.7 | -25.4% | 6.0% | Not measured |
| net_position | chronos-2-V010 | EE | 24-36h | 5 | 3.7% | -1.6% | 1.5 | 1.0 | 87.9% | 66.0% | Not measured |
| net_position | chronos-2-V010 | EE | 36-48h | 115 | 79.5% | 78.2% | 0.2 | 0.2 | -203.8% | -209.7% | Not measured |
| net_position | chronos-2-V010 | EE | 48-64h | 245 | 65.8% | 61.4% | 0.5 | 0.6 | -52.5% | -112.1% | Not measured |
| net_position | chronos-2-V010 | ES | 24-36h | 5 | 134.4% | 134.4% | 1.6 | 0.8 | 59.1% | 65.5% | Not measured |
| net_position | chronos-2-V010 | ES | 36-48h | 115 | 54.0% | 8.8% | 0.6 | 0.9 | 25.7% | 22.0% | Not measured |
| net_position | chronos-2-V010 | ES | 48-64h | 245 | 68.7% | -48.0% | 0.3 | 0.5 | -51.9% | -52.5% | Not measured |
| net_position | chronos-2-V010 | FI | 24-36h | 5 | 12.5% | -6.3% | -0.5 | -0.9 | -36.0% | 33.0% | Not measured |
| net_position | chronos-2-V010 | FI | 36-48h | 115 | 70.3% | -17.4% | 0.3 | 0.7 | 51.6% | 45.3% | Not measured |
| net_position | chronos-2-V010 | FI | 48-64h | 245 | 93.9% | -37.4% | 0.3 | 0.5 | 43.9% | 36.7% | Not measured |
| net_position | chronos-2-V010 | FR | 24-36h | 5 | 19.0% | -19.0% | 0.1 | 0.4 | -371.9% | -265.1% | Not measured |
| net_position | chronos-2-V010 | FR | 36-48h | 115 | 77.1% | -70.7% | -0.2 | -0.1 | -346.9% | -348.8% | Not measured |
| net_position | chronos-2-V010 | FR | 48-64h | 245 | 72.3% | -56.4% | 0.1 | 0.1 | -61.8% | -60.2% | Not measured |
| net_position | chronos-2-V010 | GR | 36-48h | 0 | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured |
| net_position | chronos-2-V010 | GR | 48-64h | 0 | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured |
| net_position | chronos-2-V010 | HR | 24-36h | 5 | 39.9% | -39.9% | 0.3 | 0.7 | 78.8% | 25.8% | Not measured |
| net_position | chronos-2-V010 | HR | 36-48h | 115 | 84.2% | 59.4% | 0.3 | 0.4 | -14.9% | -6.4% | Not measured |
| net_position | chronos-2-V010 | HR | 48-64h | 245 | 65.8% | 56.2% | 0.1 | 0.1 | -72.2% | -83.1% | Not measured |
| net_position | chronos-2-V010 | HU | 24-36h | 5 | 122.9% | -122.9% | 0.6 | 1.0 | 25.8% | -81.0% | Not measured |
| net_position | chronos-2-V010 | HU | 36-48h | 115 | 75.0% | 60.5% | 0.4 | 0.5 | -54.0% | -200.0% | Not measured |
| net_position | chronos-2-V010 | HU | 48-64h | 245 | 61.0% | 42.5% | 0.4 | 0.6 | -23.8% | -108.0% | Not measured |
| net_position | chronos-2-V010 | LT | 24-36h | 5 | 161.3% | -161.3% | 0.9 | 1.0 | -2.2% | 53.6% | Not measured |
| net_position | chronos-2-V010 | LT | 36-48h | 115 | 78.2% | 69.2% | 0.4 | 0.7 | 8.5% | -23.3% | Not measured |
| net_position | chronos-2-V010 | LT | 48-64h | 245 | 67.7% | 42.6% | 0.6 | 0.7 | 34.7% | 21.4% | Not measured |
| net_position | chronos-2-V010 | LV | 24-36h | 5 | 44.4% | -44.4% | -0.2 | -0.1 | -168.8% | -80.9% | Not measured |
| net_position | chronos-2-V010 | LV | 36-48h | 115 | 77.3% | 74.2% | 0.4 | 0.5 | -55.0% | -90.3% | Not measured |
| net_position | chronos-2-V010 | LV | 48-64h | 245 | 71.3% | 61.5% | 0.4 | 0.6 | -23.0% | -34.7% | Not measured |
| net_position | chronos-2-V010 | NL | 24-36h | 5 | 102.1% | 102.1% | 2.0 | 0.6 | -75.1% | 21.3% | Not measured |
| net_position | chronos-2-V010 | NL | 36-48h | 115 | 111.5% | -34.5% | -0.2 | -0.3 | -3.0% | -10.2% | Not measured |
| net_position | chronos-2-V010 | NL | 48-64h | 245 | 90.2% | -43.4% | -0.1 | -0.1 | 4.6% | -2.1% | Not measured |
| net_position | chronos-2-V010 | PL | 24-36h | 5 | 93.1% | -93.1% | 0.5 | 0.6 | 3.4% | 31.1% | Not measured |
| net_position | chronos-2-V010 | PL | 36-48h | 115 | 89.2% | -63.6% | 0.0 | 0.1 | 4.4% | -14.8% | Not measured |
| net_position | chronos-2-V010 | PL | 48-64h | 245 | 85.5% | -19.0% | 0.2 | 0.5 | 12.4% | 7.5% | Not measured |
| net_position | chronos-2-V010 | PT | 24-36h | 5 | 42.9% | 42.9% | 0.2 | 0.3 | -10.1% | 19.7% | Not measured |
| net_position | chronos-2-V010 | PT | 36-48h | 115 | 53.9% | 49.1% | 0.9 | 0.8 | -78.7% | -63.1% | Not measured |
| net_position | chronos-2-V010 | PT | 48-64h | 245 | 62.5% | 54.4% | 0.4 | 0.4 | -154.0% | -121.1% | Not measured |
| net_position | chronos-2-V010 | RO | 24-36h | 5 | 51.4% | -51.4% | 0.8 | 0.9 | 67.2% | 13.7% | Not measured |
| net_position | chronos-2-V010 | RO | 36-48h | 115 | 91.4% | -24.9% | 0.2 | 0.4 | 22.1% | 5.6% | Not measured |
| net_position | chronos-2-V010 | RO | 48-64h | 245 | 89.3% | -11.5% | 0.2 | 0.4 | 7.3% | -25.9% | Not measured |
| net_position | chronos-2-V010 | SI | 24-36h | 5 | 46.6% | -32.7% | -0.4 | -0.8 | 23.6% | 7.2% | Not measured |
| net_position | chronos-2-V010 | SI | 36-48h | 115 | 72.6% | 69.3% | -0.1 | -0.0 | -258.5% | -139.3% | Not measured |
| net_position | chronos-2-V010 | SI | 48-64h | 245 | 85.3% | 79.0% | 0.2 | 0.2 | -151.5% | -104.2% | Not measured |
| net_position | chronos-2-V010 | SK | 24-36h | 5 | 18.8% | -18.8% | 1.8 | 0.9 | -3.1% | -33.2% | Not measured |
| net_position | chronos-2-V010 | SK | 36-48h | 115 | 83.7% | -77.2% | 0.3 | 0.6 | 2.9% | -55.7% | Not measured |
| net_position | chronos-2-V010 | SK | 48-64h | 245 | 59.9% | 14.9% | 0.5 | 0.8 | 18.3% | -39.5% | Not measured |

## Correctness notes

- Both `T` and space timestamp separators are parsed before joining.
- `load_mw > 0` is applied only to load. Measured zero remains valid for every other type.
- GR net position is excluded by name: actuals are fabricated exact zeros, not measurements (ABL-35: every published row since 2025-10-01 is 0.0 while GR moved a median 1,142 MW across its borders); row deletion pending on ABL-67
- D−7 and persistence use only stored actual observations. Missing source rows remain missing.
- Net-position persistence reuses the promotion evaluator's day-ahead publication cutoff.
- TSO comparisons use the latest stored TSO series; the database does not retain an issued-vintage archive for reconstruction.
- The separate net-position promotion gate remains authoritative and is not reproduced here.
