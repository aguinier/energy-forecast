# ABL-437 — the amended ladder read, applied to every graded ABL-316 pair

Generated: 2026-08-14T01:29:21Z. Registration: `experiments/ABL437/config.json`, committed before this read existed.

Levelling: **`fit_window` → `trailing_28d`**. Arithmetic over the stored records plus the two trailing references recomputed on the same rows — no refit, no new model, replica opened read-only.
Replica: `C:\Code\able\data\energy_dashboard.db` (9,432,453,120 bytes).

**No committed record is edited by this read.** It is a new document, on the ABL-418 retro-grade precedent.

## 1. The row set is proved, not assumed

Each cell's scored rows are rebuilt from ABL-348's eight registered run instants — the harness's own `schedule_vintages` and `horizon_band`, latest vintage per (target, band) — and then **checked by recomputing that cell's published `constant_causal` and `climatology_causal` WAPE *and* MAE from it**, to 1e-09. A constant and a 24-bucket climatology agreeing on two statistics each is the row set; one agreeing alone would not be.

**113 of 113 cells reconstructed.** Every cell.

**93 of them came back on the schedule alone; 20 needed the harness's own feature build.** Where a gate row was dropped for a NaN feature, only the feature vector knows which — and because `finite_training_rows` runs *before* `select_latest_challenger_per_band`, a dropped vintage does not merely shrink n, it can promote the next-latest vintage into the band and move that row's issue instant. Under a reference levelled on `generated_at` that is not a detail, so those pairs are rebuilt through `RenewableFeatureBuilder` rather than estimated, on the feature list each record names for itself (`meta.feature_columns`, whose absence dates the read). Every one of them then reproduces its published references to the same tolerance.

## 2. Which pairs the amendment moves

**11 pairs move, 28 hold.**

| tranche | pair | published | amended | what changed | flip margin, tightest-widest |
|---|---|:---:|:---:|---|---:|
| 2a | HU solar | U(+) | **U** | now fails G3 | 3.60-9.69pp |
| 2a | PL solar | A | **B** | now fails G3 | 0.36-1.13pp |
| 2a | SI solar | A | **B** | now fails G3 | 16.11-23.14pp |
| 2a | SK solar | A | **B** | now fails G3 | 2.29-8.01pp |
| 2b | IT wind_onshore | U(+) | **U** | now fails G2, G3 | 22.64-34.70pp |
| 2c | ES solar | U(+) | **U** | now fails G3 | 13.90-18.34pp |
| 2c | HR solar | U(+) | **U** | now fails G3 | 20.72-23.83pp |
| 2c | IT solar | U(+) | **U** | now fails G3 | 6.01-10.19pp |
| 2d | LT solar | A | **B** | now fails G3 | 3.65-6.87pp |
| 2d | SE solar | A | **B** | now fails G3 | 2.97-8.65pp |
| 2f | CH wind_onshore | A | **B** | now fails G2, G3 | 0.52-12.89pp |

**Read the last column before reading the letter.** ABL-418 registers G2 and G3 as sign tests — `skill > 0` — where G1 carries a readability floor (7.51% wind, 10.65% solar at k=1). So a G2/G3 flip can sit far inside the margin at which one seed can resolve anything, and several of these do. A flip on a sub-1pp margin means **not demonstrated**, not *measured worse*. Widening G2/G3 to a floor test would be a second registration change on top of this one and is not made here; the margin is printed instead.

## 3. Every cell, both levellings

`c` = constant, `clim` = climatology. `28d` is the ABL-437 trailing reference; `causal` is the fit-window one, kept and reported. `inflation` is each causal reference's WAPE over the oracle constant's — the residual mis-levelling, which the amendment reduces rather than removes.

| tranche | pair | band | n | challenger | D-7 | c causal | c 28d | c oracle | clim causal | clim 28d | clim oracle | inflation causal / 28d | published | amended |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 1b | BG solar | 24-36h | 720 | 18.9% | 24.4% | 75.3% | 75.2% | 73.5% | 42.0% | 20.7% | 19.2% | 2.5% / 2.3% | A | A |
| 1b | BG solar | 36-48h | 720 | 18.6% | 24.4% | 75.3% | 75.1% | 73.5% | 42.0% | 20.8% | 19.2% | 2.5% / 2.1% | A | A |
| 1b | BG solar | 48-64h | 510 | 20.0% | 25.0% | 68.2% | 62.7% | 63.8% | 41.3% | 22.1% | 20.4% | 6.8% / -1.8% | A | A |
| 1b | CH solar | 24-36h | 720 | 8.2% | 12.7% | 95.1% | 100.0% | 94.6% | 37.5% | 10.2% | 9.0% | 0.5% / 5.7% | A | A |
| 1b | CH solar | 36-48h | 720 | 8.0% | 12.7% | 95.1% | 100.1% | 94.6% | 37.5% | 10.2% | 9.0% | 0.5% / 5.7% | A | A |
| 1b | CH solar | 48-64h | 510 | 8.4% | 12.5% | 86.0% | 81.0% | 87.9% | 36.6% | 9.8% | 8.7% | -2.2% / -7.9% | A | A |
| 2a | BG solar | 24-36h | 720 | 19.6% | 24.4% | 75.3% | 75.2% | 73.5% | 41.9% | 20.7% | 19.2% | 2.4% / 2.3% | A | A |
| 2a | BG solar | 36-48h | 720 | 19.5% | 24.4% | 75.3% | 75.1% | 73.5% | 41.9% | 20.8% | 19.2% | 2.4% / 2.1% | A | A |
| 2a | BG solar | 48-64h | 510 | 20.8% | 25.0% | 68.1% | 62.7% | 63.8% | 41.2% | 22.1% | 20.4% | 6.8% / -1.8% | A | A |
| 2a | CH solar | 24-36h | 720 | 7.7% | 12.7% | 94.9% | 100.0% | 94.6% | 39.7% | 10.2% | 9.0% | 0.3% / 5.7% | A | A |
| 2a | CH solar | 36-48h | 720 | 7.5% | 12.7% | 94.9% | 100.1% | 94.6% | 39.7% | 10.2% | 9.0% | 0.3% / 5.7% | A | A |
| 2a | CH solar | 48-64h | 510 | 8.0% | 12.5% | 86.3% | 81.0% | 87.9% | 39.1% | 9.8% | 8.7% | -1.8% / -7.9% | A | A |
| 2a | CZ solar | 24-36h | 720 | 12.9% | 24.0% | 93.9% | 98.3% | 92.8% | 29.3% | 17.3% | 15.9% | 1.2% / 6.0% | A | A |
| 2a | CZ solar | 36-48h | 720 | 12.9% | 24.0% | 93.9% | 98.3% | 92.8% | 29.3% | 17.4% | 15.9% | 1.2% / 6.0% | A | A |
| 2a | CZ solar | 48-64h | 510 | 14.0% | 24.0% | 85.8% | 84.3% | 89.9% | 28.4% | 17.3% | 16.1% | -4.5% / -6.2% | A | A |
| 2a | HU solar | 24-36h | 720 | 17.3% | 18.2% | 95.7% | 98.3% | 95.0% | 30.9% | 15.8% | 14.2% | 0.7% / 3.5% | U(+) | U |
| 2a | HU solar | 36-48h | 720 | 17.3% | 18.2% | 95.7% | 98.3% | 95.0% | 30.9% | 15.9% | 14.2% | 0.7% / 3.5% | U(+) | U |
| 2a | HU solar | 48-64h | 510 | 16.5% | 17.9% | 88.6% | 87.2% | 91.4% | 29.8% | 16.0% | 14.3% | -3.0% / -4.6% | U(+) | U |
| 2a | PL solar | 24-36h | 720 | 17.3% | 26.0% | 92.6% | 95.1% | 92.2% | 28.1% | 17.1% | 15.4% | 0.5% / 3.2% | A | B |
| 2a | PL solar | 36-48h | 720 | 17.4% | 26.0% | 92.6% | 95.1% | 92.2% | 28.1% | 17.2% | 15.4% | 0.5% / 3.2% | A | B |
| 2a | PL solar | 48-64h | 510 | 16.3% | 24.5% | 85.9% | 83.9% | 88.1% | 27.1% | 16.2% | 14.6% | -2.4% / -4.8% | A | B |
| 2a | RO solar | 24-36h | 720 | 18.8% | 24.3% | 96.3% | 100.5% | 95.8% | 43.4% | 21.9% | 19.9% | 0.5% / 4.9% | A | A |
| 2a | RO solar | 36-48h | 720 | 18.7% | 24.3% | 96.3% | 100.4% | 95.8% | 43.4% | 21.9% | 19.9% | 0.5% / 4.8% | A | A |
| 2a | RO solar | 48-64h | 510 | 19.2% | 25.0% | 92.3% | 91.3% | 93.0% | 42.4% | 22.7% | 20.4% | -0.8% / -1.9% | A | A |
| 2a | SI solar | 24-36h | 720 | 17.9% | 21.6% | 95.0% | 100.1% | 93.8% | 35.1% | 15.4% | 13.0% | 1.2% / 6.7% | A | B |
| 2a | SI solar | 36-48h | 720 | 18.1% | 21.6% | 95.0% | 100.1% | 93.8% | 35.1% | 15.4% | 13.0% | 1.2% / 6.6% | A | B |
| 2a | SI solar | 48-64h | 510 | 18.7% | 21.2% | 86.6% | 84.9% | 90.1% | 34.3% | 15.1% | 12.8% | -3.9% / -5.8% | A | B |
| 2a | SK solar | 24-36h | 715 | 16.3% | 18.8% | 97.1% | 101.9% | 95.3% | 32.6% | 15.2% | 13.1% | 2.0% / 7.0% | A | B |
| 2a | SK solar | 36-48h | 715 | 16.4% | 18.8% | 97.1% | 101.9% | 95.3% | 32.6% | 15.2% | 13.1% | 2.0% / 6.9% | A | B |
| 2a | SK solar | 48-64h | 507 | 15.1% | 18.3% | 89.8% | 88.9% | 93.1% | 31.6% | 14.7% | 12.5% | -3.5% / -4.5% | A | B |
| 2b | ES wind_onshore | 24-36h | 720 | 54.3% | 41.0% | 62.1% | 44.0% | 41.5% | 55.1% | 29.8% | 27.5% | 49.6% / 6.0% | C | C |
| 2b | ES wind_onshore | 36-48h | 720 | 54.2% | 41.0% | 62.1% | 43.9% | 41.5% | 55.1% | 29.8% | 27.5% | 49.6% / 5.9% | C | C |
| 2b | ES wind_onshore | 48-64h | 510 | 52.4% | 38.5% | 63.9% | 46.6% | 44.2% | 53.1% | 28.0% | 26.1% | 44.6% / 5.4% | C | C |
| 2b | FI wind_onshore | 24-36h | 711 | 41.1% | 59.6% | 54.6% | 53.7% | 53.5% | 53.4% | 52.6% | 51.4% | 2.2% / 0.5% | A | A |
| 2b | FI wind_onshore | 36-48h | 711 | 43.3% | 59.6% | 54.6% | 54.2% | 53.5% | 53.4% | 52.8% | 51.4% | 2.2% / 1.3% | A | A |
| 2b | FI wind_onshore | 48-64h | 504 | 46.2% | 54.9% | 52.0% | 53.5% | 51.5% | 50.7% | 52.5% | 49.9% | 0.8% / 3.8% | A | A |
| 2b | GR wind_onshore | 24-36h | 720 | 29.6% | 63.8% | 53.2% | 55.3% | 51.7% | 53.4% | 55.7% | 51.4% | 2.9% / 7.0% | A | A |
| 2b | GR wind_onshore | 36-48h | 720 | 29.6% | 63.8% | 53.2% | 56.1% | 51.7% | 53.4% | 55.9% | 51.4% | 2.9% / 8.4% | A | A |
| 2b | GR wind_onshore | 48-64h | 510 | 30.2% | 58.9% | 50.9% | 53.9% | 49.0% | 50.8% | 53.4% | 48.7% | 3.9% / 10.0% | A | A |
| 2b | IT wind_onshore | 24-36h | 716 | 71.4% | 70.6% | 92.0% | 57.8% | 52.3% | 90.7% | 53.0% | 45.1% | 75.8% / 10.5% | U(+) | U |
| 2b | IT wind_onshore | 36-48h | 715 | 71.2% | 70.6% | 91.9% | 58.0% | 52.3% | 90.6% | 53.0% | 45.1% | 75.7% / 11.0% | U(+) | U |
| 2b | IT wind_onshore | 48-64h | 505 | 66.8% | 67.2% | 82.8% | 53.8% | 50.8% | 83.3% | 49.9% | 42.4% | 63.2% / 6.0% | U(+) | U |
| 2b | NO wind_onshore | 24-36h | 720 | 51.4% | 61.0% | 59.7% | 44.0% | 42.4% | 59.4% | 43.9% | 41.9% | 40.7% / 3.8% | B | B |
| 2b | NO wind_onshore | 36-48h | 720 | 51.6% | 61.0% | 59.7% | 44.2% | 42.4% | 59.4% | 43.9% | 41.9% | 40.7% / 4.1% | B | B |
| 2b | NO wind_onshore | 48-64h | 510 | 51.8% | 61.6% | 57.7% | 44.9% | 43.1% | 58.0% | 44.7% | 42.4% | 33.9% / 4.2% | B | B |
| 2b | PL wind_onshore | 24-36h | 720 | 54.1% | 92.8% | 61.1% | 60.1% | 51.2% | 59.7% | 59.1% | 47.4% | 19.5% / 17.4% | A | A |
| 2b | PL wind_onshore | 36-48h | 720 | 52.5% | 92.8% | 61.1% | 60.9% | 51.2% | 59.7% | 59.3% | 47.4% | 19.5% / 19.0% | A | A |
| 2b | PL wind_onshore | 48-64h | 510 | 51.4% | 94.4% | 63.9% | 64.8% | 52.3% | 61.5% | 63.5% | 48.2% | 22.3% / 24.0% | A | A |
| 2b | PT wind_onshore | 24-36h | 720 | 68.2% | 49.6% | 101.4% | 55.1% | 50.2% | 101.1% | 48.1% | 39.4% | 102.1% / 9.9% | C | C |
| 2b | PT wind_onshore | 36-48h | 720 | 68.7% | 49.6% | 101.4% | 55.5% | 50.2% | 101.1% | 48.2% | 39.4% | 102.1% / 10.6% | C | C |
| 2b | PT wind_onshore | 48-64h | 510 | 61.0% | 46.6% | 93.1% | 53.5% | 49.3% | 87.3% | 42.4% | 35.9% | 88.7% / 8.5% | C | C |
| 2b | SE wind_onshore | 24-36h | 720 | 30.2% | 53.5% | 43.7% | 36.8% | 36.5% | 42.7% | 36.7% | 35.5% | 19.7% / 1.0% | A | A |
| 2b | SE wind_onshore | 36-48h | 720 | 30.2% | 53.5% | 43.7% | 37.1% | 36.5% | 42.7% | 36.8% | 35.5% | 19.7% / 1.7% | A | A |
| 2b | SE wind_onshore | 48-64h | 510 | 30.3% | 52.8% | 44.4% | 36.9% | 36.2% | 42.9% | 37.2% | 35.3% | 22.7% / 2.0% | A | A |
| 2c | ES solar | 24-36h | 720 | 11.4% | 11.7% | 89.8% | 91.6% | 89.8% | 35.4% | 10.0% | 8.7% | 0.0% / 2.1% | U(+) | U |
| 2c | ES solar | 36-48h | 720 | 11.4% | 11.7% | 89.8% | 91.6% | 89.8% | 35.4% | 10.0% | 8.7% | 0.0% / 2.0% | U(+) | U |
| 2c | ES solar | 48-64h | 510 | 11.0% | 11.1% | 77.8% | 70.2% | 77.1% | 35.2% | 9.3% | 8.4% | 1.0% / -8.9% | U(+) | U |
| 2c | GR solar | 24-36h | 720 | 20.8% | 10.2% | 98.3% | 102.0% | 97.1% | 46.3% | 12.5% | 8.6% | 1.2% / 5.0% | C | C |
| 2c | GR solar | 36-48h | 720 | 20.8% | 10.2% | 98.3% | 101.8% | 97.1% | 46.3% | 12.6% | 8.6% | 1.2% / 4.8% | C | C |
| 2c | GR solar | 48-64h | 510 | 20.6% | 10.3% | 93.3% | 92.0% | 95.1% | 46.5% | 13.7% | 8.9% | -2.0% / -3.3% | C | C |
| 2c | HR solar | 24-36h | 720 | 15.0% | 16.2% | 96.0% | 99.5% | 96.0% | 39.3% | 12.1% | 9.3% | 0.0% / 3.6% | U(+) | U |
| 2c | HR solar | 36-48h | 720 | 15.1% | 16.2% | 96.0% | 99.4% | 96.0% | 39.3% | 12.2% | 9.3% | 0.0% / 3.6% | U(+) | U |
| 2c | HR solar | 48-64h | 510 | 14.8% | 16.2% | 89.0% | 85.9% | 91.5% | 38.8% | 12.3% | 9.3% | -2.8% / -6.2% | U(+) | U |
| 2c | IT solar | 24-36h | 720 | 6.6% | 7.0% | 98.9% | 102.9% | 97.2% | 29.6% | 6.1% | 4.3% | 1.8% / 5.9% | U(+) | U |
| 2c | IT solar | 36-48h | 720 | 6.7% | 7.0% | 98.9% | 102.9% | 97.2% | 29.6% | 6.1% | 4.3% | 1.8% / 5.9% | U(+) | U |
| 2c | IT solar | 48-64h | 510 | 6.0% | 6.6% | 88.2% | 86.0% | 93.0% | 28.8% | 5.7% | 4.1% | -5.1% / -7.5% | U(+) | U |
| 2c | PT solar | 24-36h | 720 | 14.5% | 13.1% | 97.2% | 100.6% | 96.4% | 36.5% | 15.4% | 13.6% | 0.8% / 4.4% | C | C |
| 2c | PT solar | 36-48h | 720 | 14.9% | 13.1% | 97.2% | 100.5% | 96.4% | 36.5% | 15.4% | 13.6% | 0.8% / 4.3% | C | C |
| 2c | PT solar | 48-64h | 510 | 15.0% | 13.0% | 81.0% | 73.9% | 87.1% | 36.3% | 15.0% | 13.6% | -7.0% / -15.1% | C | C |
| 2d | EE solar | 48-64h | 388 | 25.1% | 35.3% | 80.4% | 80.8% | 81.4% | 29.1% | 25.1% | 23.3% | -1.2% / -0.7% | A | A |
| 2d | FI solar | 48-64h | 453 | 24.0% | 38.0% | 82.5% | 75.6% | 77.3% | 45.1% | 24.1% | 22.6% | 6.7% / -2.2% | A | A |
| 2d | LT solar | 24-36h | 720 | 20.9% | 30.6% | 91.2% | 93.8% | 91.1% | 45.8% | 19.6% | 18.2% | 0.1% / 2.9% | A | B |
| 2d | LT solar | 36-48h | 720 | 20.8% | 30.6% | 91.2% | 93.8% | 91.1% | 45.8% | 19.6% | 18.2% | 0.1% / 3.0% | A | B |
| 2d | LT solar | 48-64h | 510 | 19.8% | 29.4% | 90.9% | 89.5% | 90.3% | 44.8% | 19.1% | 17.5% | 0.6% / -0.9% | A | B |
| 2d | LV solar | 24-36h | 708 | 29.6% | 47.8% | 89.9% | 95.6% | 89.2% | 39.8% | 40.4% | 33.9% | 0.8% / 7.1% | A | A |
| 2d | LV solar | 36-48h | 708 | 29.5% | 47.8% | 89.9% | 95.9% | 89.2% | 39.8% | 40.7% | 33.9% | 0.8% / 7.5% | A | A |
| 2d | LV solar | 48-64h | 506 | 32.2% | 47.8% | 90.4% | 94.5% | 90.2% | 36.0% | 43.0% | 34.2% | 0.2% / 4.7% | A | A |
| 2d | NL solar | 24-36h | 720 | 37.0% | 45.0% | 103.2% | 106.3% | 95.7% | 34.8% | 35.6% | 32.6% | 7.8% / 11.0% | B | B |
| 2d | NL solar | 36-48h | 720 | 37.7% | 45.0% | 103.2% | 106.5% | 95.7% | 34.8% | 35.7% | 32.6% | 7.8% / 11.3% | B | B |
| 2d | NL solar | 48-64h | 510 | 36.8% | 45.2% | 85.2% | 85.9% | 91.4% | 34.8% | 36.1% | 32.6% | -6.8% / -6.0% | B | B |
| 2d | SE solar | 24-36h | 720 | 21.2% | 23.9% | 94.6% | 100.9% | 94.0% | 40.9% | 19.5% | 18.5% | 0.7% / 7.4% | A | B |
| 2d | SE solar | 36-48h | 720 | 20.8% | 23.9% | 94.6% | 100.9% | 94.0% | 40.9% | 19.5% | 18.5% | 0.7% / 7.4% | A | B |
| 2d | SE solar | 48-64h | 510 | 19.8% | 23.7% | 87.5% | 84.6% | 90.3% | 40.7% | 19.2% | 17.9% | -3.1% / -6.3% | A | B |
| 2e | CZ wind_onshore | 24-36h | 720 | 44.8% | 86.4% | 54.8% | 54.1% | 46.1% | 53.4% | 52.2% | 42.7% | 18.8% / 17.3% | A | A |
| 2e | CZ wind_onshore | 36-48h | 720 | 45.0% | 86.4% | 54.8% | 54.5% | 46.1% | 53.4% | 52.4% | 42.7% | 18.8% / 18.3% | A | A |
| 2e | CZ wind_onshore | 48-64h | 510 | 47.4% | 86.1% | 57.8% | 57.7% | 47.8% | 56.5% | 56.8% | 44.2% | 20.8% / 20.7% | A | A |
| 2e | EE wind_onshore | 24-36h | 685 | 42.7% | 85.8% | 73.0% | 64.5% | 62.2% | 73.1% | 63.3% | 59.7% | 17.3% / 3.7% | A | A |
| 2e | EE wind_onshore | 36-48h | 684 | 42.4% | 85.7% | 72.9% | 64.1% | 62.2% | 73.0% | 63.3% | 59.6% | 17.2% / 3.1% | A | A |
| 2e | EE wind_onshore | 48-64h | 475 | 46.0% | 86.7% | 73.5% | 66.0% | 64.2% | 73.6% | 66.1% | 61.7% | 14.4% / 2.7% | A | A |
| 2e | HR wind_onshore | 24-36h | 720 | 74.1% | 97.7% | 92.0% | 76.2% | 69.2% | 91.0% | 75.6% | 65.7% | 32.9% / 10.1% | A | A |
| 2e | HR wind_onshore | 36-48h | 720 | 68.6% | 97.7% | 92.0% | 76.3% | 69.2% | 91.0% | 75.6% | 65.7% | 32.9% / 10.3% | A | A |
| 2e | HR wind_onshore | 48-64h | 510 | 60.8% | 88.5% | 89.3% | 72.6% | 65.0% | 85.3% | 69.6% | 60.2% | 37.5% / 11.7% | A | A |
| 2e | HU wind_onshore | 24-36h | 720 | 104.9% | 124.2% | 103.1% | 82.8% | 72.1% | 102.7% | 83.6% | 70.5% | 43.0% / 14.8% | B | B |
| 2e | HU wind_onshore | 36-48h | 720 | 105.4% | 124.2% | 103.1% | 83.0% | 72.1% | 102.7% | 83.6% | 70.5% | 43.0% / 15.1% | B | B |
| 2e | HU wind_onshore | 48-64h | 510 | 103.9% | 124.5% | 99.0% | 80.7% | 71.3% | 99.7% | 84.6% | 69.6% | 38.8% / 13.2% | B | B |
| 2e | LT wind_onshore | 24-36h | 720 | 56.4% | 100.5% | 90.4% | 79.5% | 66.2% | 88.7% | 75.8% | 61.4% | 36.4% / 19.9% | A | A |
| 2e | LT wind_onshore | 36-48h | 720 | 56.1% | 100.5% | 90.4% | 79.8% | 66.2% | 88.7% | 76.0% | 61.4% | 36.4% / 20.4% | A | A |
| 2e | LT wind_onshore | 48-64h | 510 | 61.0% | 99.2% | 94.1% | 83.2% | 67.9% | 93.1% | 80.6% | 62.8% | 38.7% / 22.6% | A | A |
| 2e | LV wind_onshore | 24-36h | 708 | 89.0% | 97.5% | 72.1% | 77.5% | 69.6% | 70.4% | 75.2% | 67.2% | 3.6% / 11.4% | B | B |
| 2e | LV wind_onshore | 36-48h | 708 | 90.3% | 97.5% | 72.1% | 77.7% | 69.6% | 70.4% | 75.2% | 67.2% | 3.6% / 11.6% | U | U |
| 2e | LV wind_onshore | 48-64h | 506 | 90.9% | 97.1% | 71.0% | 77.2% | 68.3% | 69.6% | 75.7% | 66.6% | 3.9% / 12.9% | U | U |
| 2e | NL wind_onshore | 24-36h | 720 | 78.0% | 94.9% | 225.5% | 90.7% | 73.8% | 225.8% | 90.9% | 72.4% | 205.4% / 22.8% | A | A |
| 2e | NL wind_onshore | 36-48h | 720 | 82.2% | 94.9% | 225.5% | 91.8% | 73.8% | 225.8% | 91.3% | 72.4% | 205.4% / 24.4% | A | A |
| 2e | NL wind_onshore | 48-64h | 510 | 82.2% | 94.8% | 217.9% | 92.1% | 74.6% | 220.2% | 93.2% | 72.7% | 192.2% / 23.5% | A | A |
| 2e | RO wind_onshore | 24-36h | 720 | 79.5% | 103.7% | 93.1% | 74.5% | 71.3% | 92.4% | 72.4% | 68.3% | 30.5% / 4.4% | B | B |
| 2e | RO wind_onshore | 36-48h | 720 | 80.3% | 103.7% | 93.1% | 74.7% | 71.3% | 92.4% | 72.5% | 68.3% | 30.5% / 4.7% | B | B |
| 2e | RO wind_onshore | 48-64h | 510 | 79.8% | 98.6% | 91.0% | 73.6% | 69.7% | 90.8% | 71.9% | 67.4% | 30.6% / 5.6% | B | B |
| 2f | BG wind_onshore | 24-36h | 720 | 56.9% | 93.8% | 82.8% | 67.1% | 63.8% | 81.0% | 66.1% | 62.5% | 29.8% / 5.2% | A | A |
| 2f | BG wind_onshore | 36-48h | 720 | 56.8% | 93.8% | 82.8% | 67.1% | 63.8% | 81.0% | 66.2% | 62.5% | 29.8% / 5.2% | A | A |
| 2f | BG wind_onshore | 48-64h | 510 | 57.8% | 89.3% | 86.9% | 65.9% | 60.7% | 82.7% | 65.1% | 60.0% | 43.3% / 8.7% | A | A |
| 2f | CH wind_onshore | 24-36h | 720 | 47.4% | 59.3% | 79.1% | 43.9% | 40.3% | 77.8% | 42.0% | 38.2% | 96.2% / 9.0% | A | B |
| 2f | CH wind_onshore | 36-48h | 720 | 45.0% | 59.3% | 79.1% | 43.9% | 40.3% | 77.8% | 42.1% | 38.2% | 96.2% / 9.0% | A | B |
| 2f | CH wind_onshore | 48-64h | 510 | 44.3% | 59.8% | 78.4% | 44.1% | 40.0% | 73.5% | 42.7% | 37.9% | 96.1% / 10.3% | A | B |

## 4. Source records, unchanged

| tranche | scope | source table | record | SHA-256 |
|---|---|---|---|---|
| 1b | `abl316-t1b` | `energy_generation` | `experiments/ABL348/results_abl381_tranche1b.json` | `6ff1629cc4525683…` |
| 2a | `abl316-t2a` | `energy_renewable` | `experiments/ABL348/results_abl405_tranche2a.json` | `895e1259c0da3921…` |
| 2b | `abl406-tranche2b` | `energy_generation` | `experiments/ABL348/results_abl406_tranche2b.json` | `972eea5fe8880668…` |
| 2c | `abl316-t2c` | `energy_generation` | `experiments/ABL348/results_abl419_tranche2c.json` | `fe25b86c98304059…` |
| 2d | `abl316-t2d` | `energy_generation` | `experiments/ABL348/results_abl421_tranche2d.json` | `ebbc4c448dbd5614…` |
| 2e | `abl417-tranche2e` | `energy_generation` | `experiments/ABL348/results_abl417_tranche2e.json` | `1225905d091b4417…` |
| 2f | `abl435-tranche2f` | `energy_generation` | `experiments/ABL348/results_abl435_tranche2f.json` | `70c6669b17cf74a4…` |

Read-only. This script writes to no path any gate read owns.
