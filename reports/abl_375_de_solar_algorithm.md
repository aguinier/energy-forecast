# ABL-375 — DE solar: XGBoost vs the serving CatBoost configuration

Generated 2026-08-13T10:32:59. Registration: `experiments/ABL375/config.json`, committed before the first fit.

Both arms are **refits on the identically truncated window** — never the live
artifacts, which are fitted through roughly today and would score in-sample.
ABL-338's solar geometry is on **both** arms: `src/features.py` appends it to
every solar fit unconditionally on `origin/main`, so the geometry arm is what a
routine retrain would actually produce.

Cells are **seed-mean (min–max)** over seeds [42, 1337, 2718]. MAE in MW.
Night is MW only: its denominator is ~0.

## The registered read

Holdout **2026-04-30 .. 2026-06-12** (registered confirmatory), n = 1,056 hours. Never fitted or scored for
this comparison before the registration — the gap between ABL-338's two
committed holdouts.

| country | n_train | daylight n | CatBoost+geom daylight | XGBoost+geom daylight | gap | verdict |
|---|---:|---:|---:|---:|---:|---|
| AT | 3,647 | 670 | 618.9 (589.1–637.5) | 496.2 (494.0–499.9) | +19.8% | PASS (xgboost) |
| BE | 20,277 | 681 | 572.2 (561.6–581.9) | 606.1 (597.5–622.1) | -5.9% | FAIL (catboost) |
| DE **(primary)** | 3,751 | 686 | 4,449.7 (4,224.5–4,838.2) | 4,249.6 (4,161.8–4,362.6) | +4.5% | AMBIGUOUS (xgboost) |
| FR | 28,636 | 656 | 1,001.6 (996.6–1,004.8) | 974.8 (961.1–997.9) | +2.7% | AMBIGUOUS (xgboost) |

### All bands, all arms, registered window

#### AT — n_holdout 1,056 (daylight 670 / shoulder 119 / night 267)

| arm | features | daylight MAE | daylight WAPE | shoulder MAE | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | – | 781.9 | – | 1.0 | – | – | – |
| catboost+control | 29 | 653.9 (647.2–660.0) | 31.10 (30.80–31.40)% | 32.2 (20.0–56.5) | 26.60 (15.40–44.90) | 81.0 (54.4–125.5) | 53 |
| catboost+geometry | 31 | 618.9 (589.1–637.5) | 29.50 (28.10–30.40)% | 32.0 (25.4–36.5) | 26.10 (19.10–31.30) | 79.0 (67.3–86.3) | 60 |
| xgboost+control | 29 | 517.8 (514.2–524.7) | 24.70 (24.50–25.00)% | 3.2 (2.3–4.2) | 0.10 (-1.30–1.90) | 6.2 (1.9–11.7) | 288 |
| xgboost+geometry | 31 | 496.2 (494.0–499.9) | 23.60 (23.50–23.80)% | 4.7 (3.2–7.1) | 4.10 (1.70–7.10) | 15.8 (8.7–23.6) | 141 |

ABL-337 contamination: 0 of 1,647 fit-window night rows read above 1 MW (max 0.0 MW), dropped from the fit and never from the score.

#### BE — n_holdout 1,056 (daylight 681 / shoulder 143 / night 232)

| arm | features | daylight MAE | daylight WAPE | shoulder MAE | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | – | 1,425.5 | – | 1.5 | – | – | – |
| catboost+control | 29 | 563.0 (552.0–576.9) | 19.40 (19.00–19.90)% | 23.3 (20.5–26.6) | -9.80 (-15.20–-5.80) | 65.6 (54.4–80.2) | 250 |
| catboost+geometry | 31 | 572.2 (561.6–581.9) | 19.70 (19.40–20.10)% | 21.7 (18.8–24.4) | -11.60 (-12.40–-11.10) | 51.1 (35.5–80.1) | 257 |
| xgboost+control | 29 | 635.2 (627.2–643.1) | 21.90 (21.60–22.20)% | 8.2 (8.0–8.4) | 0.50 (-5.70–7.20) | 17.2 (6.3–38.2) | 190 |
| xgboost+geometry | 31 | 606.1 (597.5–622.1) | 20.90 (20.60–21.50)% | 4.1 (2.3–5.8) | -0.40 (-1.80–0.60) | 7.7 (1.9–12.6) | 198 |

ABL-337 contamination: 0 of 7,670 fit-window night rows read above 1 MW (max 0.1 MW), dropped from the fit and never from the score.

#### DE — n_holdout 1,056 (daylight 686 / shoulder 142 / night 228)

| arm | features | daylight MAE | daylight WAPE | shoulder MAE | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | – | 6,509.3 | – | 14.8 | – | – | – |
| catboost+control | 29 | 4,463.5 (4,131.5–4,815.6) | 20.20 (18.70–21.80)% | 308.5 (234.1–396.6) | 212.80 (131.80–321.60) | 680.4 (575.0–815.6) | 50 |
| catboost+geometry | 31 | 4,449.7 (4,224.5–4,838.2) | 20.20 (19.10–21.90)% | 317.2 (222.7–470.1) | 230.70 (159.40–357.50) | 748.1 (675.4–809.9) | 44 |
| xgboost+control | 29 | 3,914.6 (3,856.8–3,958.1) | 17.70 (17.50–17.90)% | 54.0 (44.8–67.5) | -38.20 (-53.30–-25.70) | 22.6 (1.2–51.3) | 332 |
| xgboost+geometry | 31 | 4,249.6 (4,161.8–4,362.6) | 19.30 (18.90–19.80)% | 59.4 (40.1–81.5) | -44.50 (-63.20–-24.60) | 26.5 (14.8–40.1) | 267 |

ABL-337 contamination: 4 of 1,729 fit-window night rows read above 1 MW (max 1.7 MW), dropped from the fit and never from the score.

#### FR — n_holdout 1,056 (daylight 656 / shoulder 123 / night 277)

| arm | features | daylight MAE | daylight WAPE | shoulder MAE | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | – | 2,128.1 | – | 29.9 | – | – | – |
| catboost+control | 29 | 1,018.7 (1,000.5–1,034.1) | 12.20 (12.00–12.40)% | 88.9 (84.7–91.1) | 46.50 (39.90–52.50) | 177.0 (142.5–206.6) | 55 |
| catboost+geometry | 31 | 1,001.6 (996.6–1,004.8) | 12.00 (12.00–12.10)% | 98.9 (83.8–108.0) | 28.90 (19.30–33.90) | 133.6 (120.2–157.7) | 85 |
| xgboost+control | 29 | 980.7 (979.8–982.4) | 11.80 (11.80–11.80)% | 38.2 (33.0–47.6) | 37.40 (29.80–45.60) | 207.7 (192.2–223.1) | 0 |
| xgboost+geometry | 31 | 974.8 (961.1–997.9) | 11.70 (11.50–12.00)% | 49.0 (36.2–57.1) | 37.20 (22.40–46.70) | 116.9 (108.7–129.8) | 0 |

ABL-337 contamination: 464 of 11,337 fit-window night rows read above 1 MW (max 439.3 MW), dropped from the fit and never from the score.

## Already-observed windows (EXPLORATORY — not a second gate)

Fitted and scored under ABL-338 before this registration existed. Seeing them
is what created the hypothesis, so they cannot confirm it. Single seed each.

### summer — 2026-06-13 .. 2026-08-11 (EXPLORATORY)

| country | CatBoost+geom daylight | XGBoost+geom daylight | gap | CatBoost+geom night | XGBoost+geom night |
|---|---:|---:|---:|---:|---:|
| AT | – | – | one algorithm only | – | – |
| BE | 493.8 | 518.6 | -5.0% | 1.10 | 14.10 |
| DE | 3,694.8 | 3,076.3 | +16.7% | 453.80 | 12.10 |
| FR | 1,285.1 | 1,468.6 | -14.3% | -2.60 | 7.80 |

**DE re-fitted at the registered seeds on this already-observed window (POST-HOC, POST-HOC).** Read for reference only: the registered verdict is the confirmatory window's and is not revised here.

| arm | daylight MAE (mean, min–max) | seed spread | shoulder MAE | night mean pred |
|---|---:|---:|---:|---:|
| catboost+control | 3,633.0 (3,504.5–3,852.2) | 9.57% | 215.2 (157.4–284.9) | 90.20 (23.60–220.10) |
| catboost+geometry | 3,602.1 (3,443.5–3,694.8) | 6.98% | 379.9 (300.8–501.4) | 306.20 (200.50–453.80) |
| xgboost+control | 3,199.0 (3,173.3–3,238.6) | 2.04% | 182.5 (143.3–215.1) | 57.00 (32.90–94.40) |
| xgboost+geometry | 3,120.9 (3,076.3–3,202.2) | 4.03% | 77.3 (25.4–127.2) | 19.10 (10.80–34.40) |

Geometry-arm gap +13.4% favouring xgboost; seed ranges disjoint. Would read PASS had this been the registered window - it was not.


### spring — 2026-03-01 .. 2026-04-29 (EXPLORATORY)

| country | CatBoost+geom daylight | XGBoost+geom daylight | gap | CatBoost+geom night | XGBoost+geom night |
|---|---:|---:|---:|---:|---:|
| AT | – | – | one algorithm only | – | – |
| BE | 684.7 | 675.9 | +1.3% | 9.20 | -0.20 |
| DE | 8,866.6 | 7,634.9 | +13.9% | 190.20 | 52.70 |
| FR | 1,051.1 | 1,054.7 | -0.3% | 5.60 | 13.90 |

**DE re-fitted at the registered seeds on this already-observed window (POST-HOC, POST-HOC).** Read for reference only: the registered verdict is the confirmatory window's and is not revised here.

| arm | daylight MAE (mean, min–max) | seed spread | shoulder MAE | night mean pred |
|---|---:|---:|---:|---:|
| catboost+control | 10,367.8 (10,160.2–10,541.9) | 3.68% | 345.8 (307.0–390.8) | 213.30 (161.70–247.70) |
| catboost+geometry | 9,069.0 (8,866.6–9,285.2) | 4.62% | 341.9 (233.3–416.3) | 212.70 (154.60–293.30) |
| xgboost+control | 7,903.9 (7,766.9–8,109.7) | 4.34% | 185.6 (140.6–208.3) | 63.80 (14.70–98.40) |
| xgboost+geometry | 7,770.6 (7,634.9–7,954.9) | 4.12% | 158.2 (99.0–243.4) | 104.90 (52.70–197.90) |

Geometry-arm gap +14.3% favouring xgboost; seed ranges disjoint. Would read PASS had this been the registered window - it was not.

