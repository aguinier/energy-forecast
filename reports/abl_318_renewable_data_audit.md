# ABL-318 — per-country data audit: solar, wind_onshore, wind_offshore

Generated `2026-08-12T18:28:07+00:00` by `scripts/audit_renewable_availability.py`.
Replica `C:/Code/able/data/energy_dashboard.db` (9.43 GB), `energy_generation` current to `2026-08-12 13:30:00`. Read-only (`file:...?mode=ro`); no write of any kind touched the replica.

Verdicts are set on **`energy_generation`** (NaN-preserving, covers 2021-01-01 → now for all 24 countries), not on `energy_renewable` (the table training reads today). Columns prefixed `ren_` census `energy_renewable` so the cost of the status-quo source is visible.

## Precedent check

The run detector must re-find the known contamination or its negatives mean nothing:

```
OK   ABL-188: DE/solar energy_renewable -- longest zero run 2025-09-08 22:00:00 -> 2025-11-14 15:45:00 (6408 rows, 1601.75h); 1 zero run(s) >= 24h totalling 6408 rows
OK   ABL-198/199/200: BE/wind_offshore energy_renewable -- longest zero run 2025-11-15 20:00:00 -> 2025-11-17 17:00:00 (46 rows, 45.00h); 9 zero run(s) >= 24h totalling 320 rows
```

## Verdict table (24 countries × 3 streams = 72 rows)

| country | stream | verdict | first actual | last actual | rows | non-null | % zero | peak MW | longest zero-run (h) | longest gap (h) | note |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| AT | solar | `ALREADY-COVERED` | 2021-01-01 00:00 | 2026-08-12 12:15 | 196,754 | 196,754 | 48.14 | 5,408.0 | 16.8 | 0.0 | n=196754 observations, 2050d, 48.1% exact zeros, longest gap 0h, no suspect constant run >= 24h |
| AT | wind_onshore | `ALREADY-COVERED` | 2021-01-01 00:00 | 2026-08-12 12:15 | 196,754 | 196,754 | 0.00 | 3,892.0 | 0.0 | 0.0 | n=196754 observations, 2050d, 0.0% exact zeros, longest gap 0h, no suspect constant run >= 24h |
| AT | wind_offshore | `EXCLUDE-NO-DATA` | - | - | 196,754 | 0 | 0.00 | - | 0.0 | 0.0 | 196754 rows present for this country but energy_generation.wind_offshore_mw is NULL in every one -- ENTSO-E does not report this production type here |
| BE | solar | `ALREADY-COVERED` | 2021-01-01 00:00 | 2026-08-12 12:00 | 49,189 | 49,189 | 43.74 | 8,537.2 | 15.0 | 0.0 | n=49189 observations, 2050d, 43.7% exact zeros, longest gap 0h, no suspect constant run >= 24h |
| BE | wind_onshore | `ALREADY-COVERED` | 2021-01-01 00:00 | 2026-08-12 12:00 | 49,189 | 49,189 | 0.00 | 2,889.8 | 0.0 | 0.0 | n=49189 observations, 2050d, 0.0% exact zeros, longest gap 0h, no suspect constant run >= 24h |
| BE | wind_offshore | `ALREADY-COVERED` | 2021-01-01 00:00 | 2026-08-12 12:00 | 49,189 | 49,189 | 0.00 | 2,203.4 | 0.0 | 0.0 | n=49189 observations, 2050d, 0.0% exact zeros, longest gap 0h, no suspect constant run >= 24h |
| BG | solar | `TRAIN` | 2021-01-01 00:00 | 2026-08-12 07:00 | 49,182 | 49,182 | 17.83 | 4,408.1 | 14.0 | 0.0 | n=49182 observations, 2049d, 17.8% exact zeros, longest gap 0h, no suspect constant run >= 24h |
| BG | wind_onshore | `TRAIN` | 2021-01-01 00:00 | 2026-08-12 07:00 | 49,182 | 49,182 | 0.07 | 737.7 | 4.0 | 0.0 | n=49182 observations, 2049d, 0.1% exact zeros, longest gap 0h, no suspect constant run >= 24h |
| BG | wind_offshore | `EXCLUDE-NO-DATA` | - | - | 49,182 | 0 | 0.00 | - | 0.0 | 0.0 | 49182 rows present for this country but energy_generation.wind_offshore_mw is NULL in every one -- ENTSO-E does not report this production type here |
| CH | solar | `TRAIN` | 2021-01-01 00:00 | 2026-08-12 12:00 | 49,189 | 49,189 | 39.22 | 4,477.0 | 91.0 | 0.0 | n=49189 observations, 2050d, 39.2% exact zeros, longest gap 0h, no suspect constant run >= 24h |
| CH | wind_onshore | `TRAIN` | 2021-01-01 00:00 | 2026-08-12 12:00 | 49,189 | 49,189 | 1.58 | 198.6 | 719.0 | 0.0 | full window usable: n=49189, 2050d; damage is interior and small (720 rows, 1.46%; longest gap 0h) and is nulled at load by exclude_suspect_constant_runs |
| CH | wind_offshore | `EXCLUDE-NO-DATA` | - | - | 49,189 | 0 | 0.00 | - | 0.0 | 0.0 | 49189 rows present for this country but energy_generation.wind_offshore_mw is NULL in every one -- ENTSO-E does not report this production type here |
| CZ | solar | `TRAIN` | 2021-01-01 00:00 | 2026-08-12 12:45 | 104,884 | 104,883 | 39.58 | 3,554.7 | 13.8 | 0.0 | n=104883 observations, 2050d, 39.6% exact zeros, longest gap 0h, no suspect constant run >= 24h |
| CZ | wind_onshore | `TRAIN` | 2021-01-01 00:00 | 2026-08-12 12:45 | 104,884 | 104,883 | 0.00 | 304.2 | 0.0 | 0.0 | n=104883 observations, 2050d, 0.0% exact zeros, longest gap 0h, no suspect constant run >= 24h |
| CZ | wind_offshore | `EXCLUDE-NO-DATA` | - | - | 104,884 | 0 | 0.00 | - | 0.0 | 0.0 | 104884 rows present for this country but energy_generation.wind_offshore_mw is NULL in every one -- ENTSO-E does not report this production type here |
| DE | solar | `ALREADY-COVERED` | 2021-01-01 00:00 | 2026-08-12 12:45 | 196,757 | 196,756 | 0.91 | 57,939.3 | 7.5 | 0.0 | n=196756 observations, 2050d, 0.9% exact zeros, longest gap 0h, no suspect constant run >= 24h |
| DE | wind_onshore | `ALREADY-COVERED` | 2021-01-01 00:00 | 2026-08-12 12:45 | 196,757 | 196,756 | 0.00 | 48,606.7 | 0.0 | 0.0 | n=196756 observations, 2050d, 0.0% exact zeros, longest gap 0h, no suspect constant run >= 24h |
| DE | wind_offshore | `TRAIN` | 2021-01-01 00:00 | 2026-08-12 12:45 | 196,757 | 196,756 | 0.07 | 8,486.4 | 3.8 | 0.0 | n=196756 observations, 2050d, 0.1% exact zeros, longest gap 0h, no suspect constant run >= 24h. ARTIFACT ORPHANED: candidate/centroid/multipoint/production/model.joblib exist(s) but models/DE/wind_offshore/model.joblib -- the only path Forecaster.load reads -- does not, so forecast_daily.py skips it and 0 rows are served |
| EE | solar | `TRAIN` | 2021-01-01 00:00 | 2026-08-12 12:45 | 67,807 | 67,807 | 10.96 | 980.2 | 44.8 | 0.0 | n=67807 observations, 2050d, 11.0% exact zeros, longest gap 0h, no suspect constant run >= 24h |
| EE | wind_onshore | `TRAIN` | 2021-01-01 00:00 | 2026-08-12 12:45 | 67,807 | 67,619 | 0.01 | 606.6 | 0.0 | 7.2 | n=67619 observations, 2050d, 0.0% exact zeros, longest gap 7h, no suspect constant run >= 24h |
| EE | wind_offshore | `EXCLUDE-NO-DATA` | - | - | 67,807 | 0 | 0.00 | - | 0.0 | 0.0 | 67807 rows present for this country but energy_generation.wind_offshore_mw is NULL in every one -- ENTSO-E does not report this production type here |
| ES | solar | `TRAIN` | 2021-01-01 00:00 | 2026-08-12 13:00 | 160,101 | 160,101 | 0.39 | 29,672.0 | 13.0 | 35.8 | n=160101 observations, 2050d, 0.4% exact zeros, longest gap 36h, no suspect constant run >= 24h |
| ES | wind_onshore | `TRAIN` | 2021-01-01 00:00 | 2026-08-12 13:00 | 160,101 | 160,101 | 0.01 | 20,820.0 | 0.2 | 35.8 | n=160101 observations, 2050d, 0.0% exact zeros, longest gap 36h, no suspect constant run >= 24h |
| ES | wind_offshore | `EXCLUDE-NO-FLEET` | 2021-01-01 00:00 | 2026-08-12 13:00 | 160,101 | 160,101 | 100.00 | 0.0 | 11,910.0 | 35.8 | reported and never once non-zero across n=160101 observations (2021-01-01 to 2026-08-12) -- a declared-zero series, not a missing one |
| FI | solar | `TRAIN` | 2022-12-31 22:00 | 2026-08-12 12:45 | 134,104 | 116,354 | 29.96 | 1,473.5 | 21.0 | 8.2 | n=116354 observations, 1320d, 30.0% exact zeros, longest gap 8h, no suspect constant run >= 24h |
| FI | wind_onshore | `TRAIN` | 2021-01-01 00:00 | 2026-08-12 12:45 | 134,104 | 134,104 | 0.11 | 7,634.9 | 8.2 | 3.2 | n=134104 observations, 2050d, 0.1% exact zeros, longest gap 3h, no suspect constant run >= 24h |
| FI | wind_offshore | `EXCLUDE-NO-DATA` | - | - | 134,104 | 0 | 0.00 | - | 0.0 | 0.0 | 134104 rows present for this country but energy_generation.wind_offshore_mw is NULL in every one -- ENTSO-E does not report this production type here |
| FR | solar | `ALREADY-COVERED` | 2021-01-01 00:00 | 2026-08-12 12:45 | 90,301 | 90,300 | 39.55 | 22,324.8 | 14.0 | 518.5 | full window usable: n=90300, 2050d; damage is interior and small (0 rows, 0.00%; longest gap 518h) and is nulled at load by exclude_suspect_constant_runs |
| FR | wind_onshore | `ALREADY-COVERED` | 2021-01-01 00:00 | 2026-08-12 12:45 | 90,301 | 90,300 | 0.00 | 18,154.9 | 0.0 | 518.5 | full window usable: n=90300, 2050d; damage is interior and small (0 rows, 0.00%; longest gap 518h) and is nulled at load by exclude_suspect_constant_runs |
| FR | wind_offshore | `ALREADY-COVERED` | 2023-05-31 22:00 | 2026-08-12 12:45 | 90,301 | 68,751 | 0.58 | 1,898.2 | 0.0 | 518.5 | full window usable: n=68751, 1169d; damage is interior and small (0 rows, 0.00%; longest gap 518h) and is nulled at load by exclude_suspect_constant_runs |
| GR | solar | `TRAIN` | 2021-01-01 00:00 | 2026-08-12 13:00 | 56,358 | 56,356 | 42.39 | 8,628.0 | 14.0 | 0.0 | n=56356 observations, 2050d, 42.4% exact zeros, longest gap 0h, no suspect constant run >= 24h |
| GR | wind_onshore | `TRAIN` | 2021-01-01 00:00 | 2026-08-12 13:00 | 56,358 | 56,356 | 0.00 | 4,000.0 | 0.0 | 0.0 | n=56356 observations, 2050d, 0.0% exact zeros, longest gap 0h, no suspect constant run >= 24h |
| GR | wind_offshore | `EXCLUDE-NO-DATA` | - | - | 56,358 | 0 | 0.00 | - | 0.0 | 0.0 | 56358 rows present for this country but energy_generation.wind_offshore_mw is NULL in every one -- ENTSO-E does not report this production type here |
| HR | solar | `TRAIN` | 2021-01-01 00:00 | 2026-08-12 13:15 | 120,222 | 120,222 | 44.55 | 476.4 | 14.8 | 0.0 | n=120222 observations, 2050d, 44.5% exact zeros, longest gap 0h, no suspect constant run >= 24h |
| HR | wind_onshore | `TRAIN` | 2021-01-01 00:00 | 2026-08-12 13:00 | 120,222 | 120,221 | 0.28 | 1,120.0 | 4.8 | 0.0 | n=120221 observations, 2050d, 0.3% exact zeros, longest gap 0h, no suspect constant run >= 24h |
| HR | wind_offshore | `EXCLUDE-NO-DATA` | - | - | 120,222 | 0 | 0.00 | - | 0.0 | 0.0 | 120222 rows present for this country but energy_generation.wind_offshore_mw is NULL in every one -- ENTSO-E does not report this production type here |
| HU | solar | `TRAIN` | 2021-01-01 00:00 | 2026-08-12 13:00 | 196,757 | 196,757 | 23.66 | 4,339.9 | 15.0 | 0.0 | n=196757 observations, 2050d, 23.7% exact zeros, longest gap 0h, no suspect constant run >= 24h |
| HU | wind_onshore | `TRAIN` | 2021-01-01 00:00 | 2026-08-12 13:00 | 196,757 | 196,757 | 0.67 | 308.4 | 27.8 | 0.0 | n=196757 observations, 2050d, 0.7% exact zeros, longest gap 0h, no suspect constant run >= 24h |
| HU | wind_offshore | `EXCLUDE-NO-DATA` | - | - | 196,757 | 0 | 0.00 | - | 0.0 | 0.0 | 196757 rows present for this country but energy_generation.wind_offshore_mw is NULL in every one -- ENTSO-E does not report this production type here |
| IT | solar | `TRAIN` | 2021-01-01 00:00 | 2026-08-12 13:00 | 91,637 | 91,634 | 39.83 | 26,085.0 | 14.2 | 0.0 | n=91634 observations, 2050d, 39.8% exact zeros, longest gap 0h, no suspect constant run >= 24h |
| IT | wind_onshore | `TRAIN` | 2021-01-01 00:00 | 2026-08-12 13:00 | 91,637 | 91,570 | 0.00 | 9,107.0 | 0.0 | 5.2 | n=91570 observations, 2050d, 0.0% exact zeros, longest gap 5h, no suspect constant run >= 24h |
| IT | wind_offshore | `EXCLUDE-NO-FLEET` | 2022-05-26 22:00 | 2026-08-12 13:00 | 91,637 | 79,375 | 37.21 | 30.0 | 52.2 | 0.0 | reported, but all-time peak is 30.0 MW over n=79375 observations (2022-05-26 to 2026-08-12) -- below the 50 MW demonstrator threshold, not a fleet worth modelling |
| LT | solar | `TRAIN` | 2021-01-01 00:00 | 2026-08-12 13:15 | 94,797 | 94,797 | 41.79 | 1,926.0 | 17.0 | 0.0 | n=94797 observations, 2050d, 41.8% exact zeros, longest gap 0h, no suspect constant run >= 24h |
| LT | wind_onshore | `TRAIN` | 2021-01-01 00:00 | 2026-08-12 13:15 | 94,797 | 94,797 | 0.01 | 1,996.1 | 0.0 | 0.0 | n=94797 observations, 2050d, 0.0% exact zeros, longest gap 0h, no suspect constant run >= 24h |
| LT | wind_offshore | `EXCLUDE-NO-DATA` | - | - | 94,797 | 0 | 0.00 | - | 0.0 | 0.0 | 94797 rows present for this country but energy_generation.wind_offshore_mw is NULL in every one -- ENTSO-E does not report this production type here |
| LV | solar | `TRAIN` | 2023-12-31 22:00 | 2026-08-12 12:00 | 49,182 | 22,908 | 12.04 | 1,349.3 | 18.0 | 0.0 | n=22908 observations, 955d, 12.0% exact zeros, longest gap 0h, no suspect constant run >= 24h |
| LV | wind_onshore | `TRAIN` | 2021-01-01 00:00 | 2026-08-12 12:00 | 49,182 | 49,182 | 5.19 | 174.0 | 66.0 | 0.0 | n=49182 observations, 2050d, 5.2% exact zeros, longest gap 0h, no suspect constant run >= 24h |
| LV | wind_offshore | `EXCLUDE-NO-DATA` | - | - | 49,182 | 0 | 0.00 | - | 0.0 | 0.0 | 49182 rows present for this country but energy_generation.wind_offshore_mw is NULL in every one -- ENTSO-E does not report this production type here |
| NL | solar | `TRAIN` | 2021-01-01 00:00 | 2026-08-12 13:00 | 196,757 | 196,757 | 0.05 | 428.8 | 0.2 | 0.0 | n=196757 observations, 2050d, 0.0% exact zeros, longest gap 0h, no suspect constant run >= 24h |
| NL | wind_onshore | `TRAIN` | 2021-01-01 00:00 | 2026-08-12 13:00 | 196,757 | 196,757 | 0.00 | 2,565.1 | 0.0 | 0.0 | n=196757 observations, 2050d, 0.0% exact zeros, longest gap 0h, no suspect constant run >= 24h |
| NL | wind_offshore | `TRAIN` | 2021-01-01 00:00 | 2026-08-12 13:00 | 196,757 | 196,757 | 0.00 | 4,586.3 | 0.0 | 0.0 | n=196757 observations, 2050d, 0.0% exact zeros, longest gap 0h, no suspect constant run >= 24h |
| NO | solar | `EXCLUDE-NO-FLEET` | 2024-05-06 22:00 | 2026-08-12 12:45 | 84,433 | 55,108 | 52.82 | 7.5 | 22.0 | 0.0 | reported, but all-time peak is 7.5 MW over n=55108 observations (2024-05-06 to 2026-08-12) -- below the 50 MW demonstrator threshold, not a fleet worth modelling |
| NO | wind_onshore | `TRAIN` | 2021-01-01 00:00 | 2026-08-12 12:45 | 84,433 | 84,432 | 0.00 | 4,211.9 | 0.0 | 0.0 | n=84432 observations, 2050d, 0.0% exact zeros, longest gap 0h, no suspect constant run >= 24h |
| NO | wind_offshore | `EXCLUDE-NO-FLEET` | 2025-08-25 22:00 | 2026-08-12 12:45 | 84,433 | 33,755 | 5.02 | 5.5 | 21.8 | 0.0 | reported, but all-time peak is 5.5 MW over n=33755 observations (2025-08-25 to 2026-08-12) -- below the 50 MW demonstrator threshold, not a fleet worth modelling |
| PL | solar | `TRAIN` | 2021-01-01 00:00 | 2026-08-12 13:00 | 106,109 | 106,109 | 43.13 | 14,564.6 | 15.2 | 0.0 | n=106109 observations, 2050d, 43.1% exact zeros, longest gap 0h, no suspect constant run >= 24h |
| PL | wind_onshore | `TRAIN` | 2021-01-01 00:00 | 2026-08-12 13:00 | 106,109 | 106,109 | 0.00 | 8,906.4 | 0.0 | 0.0 | n=106109 observations, 2050d, 0.0% exact zeros, longest gap 0h, no suspect constant run >= 24h |
| PL | wind_offshore | `EXCLUDE-INSUFFICIENT-HISTORY` | 2026-07-05 22:00 | 2026-08-12 13:00 | 106,109 | 3,613 | 29.59 | 189.2 | 110.5 | 0.0 | real and non-trivial (peak 189.2 MW) but only 38d of history (2026-07-05 to 2026-08-12, n=3613) -- under the 365d needed to see every season once. Revisit, do not write off |
| PT | solar | `TRAIN` | 2021-01-01 00:00 | 2026-08-12 12:00 | 49,189 | 49,189 | 23.95 | 3,770.6 | 13.0 | 0.0 | n=49189 observations, 2050d, 23.9% exact zeros, longest gap 0h, no suspect constant run >= 24h |
| PT | wind_onshore | `TRAIN` | 2021-01-01 00:00 | 2026-08-12 12:00 | 49,189 | 49,189 | 0.01 | 5,062.4 | 6.0 | 0.0 | n=49189 observations, 2050d, 0.0% exact zeros, longest gap 0h, no suspect constant run >= 24h |
| PT | wind_offshore | `EXCLUDE-NO-FLEET` | 2021-01-01 00:00 | 2026-08-12 12:00 | 49,189 | 49,189 | 23.46 | 25.0 | 2,989.0 | 0.0 | reported, but all-time peak is 25.0 MW over n=49189 observations (2021-01-01 to 2026-08-12) -- below the 50 MW demonstrator threshold, not a fleet worth modelling |
| RO | solar | `TRAIN` | 2021-01-01 00:00 | 2026-08-12 13:15 | 194,291 | 194,291 | 48.13 | 3,117.0 | 16.5 | 0.0 | n=194291 observations, 2050d, 48.1% exact zeros, longest gap 0h, no suspect constant run >= 24h |
| RO | wind_onshore | `TRAIN` | 2021-01-01 00:00 | 2026-08-12 13:15 | 194,291 | 194,291 | 1.19 | 2,769.0 | 15.8 | 0.0 | n=194291 observations, 2050d, 1.2% exact zeros, longest gap 0h, no suspect constant run >= 24h |
| RO | wind_offshore | `EXCLUDE-NO-DATA` | - | - | 194,291 | 0 | 0.00 | - | 0.0 | 0.0 | 194291 rows present for this country but energy_generation.wind_offshore_mw is NULL in every one -- ENTSO-E does not report this production type here |
| SE | solar | `TRAIN` | 2021-12-14 23:00 | 2026-08-12 12:45 | 67,446 | 59,087 | 8.91 | 2,545.3 | 16.0 | 0.0 | n=59087 observations, 1702d, 8.9% exact zeros, longest gap 0h, no suspect constant run >= 24h |
| SE | wind_onshore | `TRAIN` | 2021-01-01 00:00 | 2026-08-12 12:45 | 67,446 | 67,446 | 0.00 | 13,528.2 | 0.0 | 0.0 | n=67446 observations, 2050d, 0.0% exact zeros, longest gap 0h, no suspect constant run >= 24h |
| SE | wind_offshore | `EXCLUDE-NO-DATA` | - | - | 67,446 | 0 | 0.00 | - | 0.0 | 0.0 | 67446 rows present for this country but energy_generation.wind_offshore_mw is NULL in every one -- ENTSO-E does not report this production type here |
| SI | solar | `TRAIN` | 2021-01-01 00:00 | 2026-08-12 13:15 | 55,264 | 55,264 | 0.12 | 1,122.2 | 14.0 | 5.0 | n=55264 observations, 2050d, 0.1% exact zeros, longest gap 5h, no suspect constant run >= 24h |
| SI | wind_onshore | `EXCLUDE-NO-FLEET` | 2021-01-01 00:00 | 2026-08-12 13:15 | 55,264 | 55,228 | 9.33 | 4.8 | 101.8 | 9.2 | reported, but all-time peak is 4.8 MW over n=55228 observations (2021-01-01 to 2026-08-12) -- below the 50 MW demonstrator threshold, not a fleet worth modelling |
| SI | wind_offshore | `EXCLUDE-NO-DATA` | - | - | 55,264 | 0 | 0.00 | - | 0.0 | 0.0 | 55264 rows present for this country but energy_generation.wind_offshore_mw is NULL in every one -- ENTSO-E does not report this production type here |
| SK | solar | `TRAIN` | 2021-01-01 00:00 | 2026-08-12 12:45 | 56,757 | 56,757 | 7.98 | 418.4 | 13.0 | 4.0 | n=56757 observations, 2050d, 8.0% exact zeros, longest gap 4h, no suspect constant run >= 24h |
| SK | wind_onshore | `EXCLUDE-NO-FLEET` | 2021-01-01 00:00 | 2026-08-12 12:45 | 56,757 | 56,757 | 23.91 | 3.1 | 134.0 | 4.0 | reported, but all-time peak is 3.1 MW over n=56757 observations (2021-01-01 to 2026-08-12) -- below the 50 MW demonstrator threshold, not a fleet worth modelling |
| SK | wind_offshore | `EXCLUDE-NO-DATA` | - | - | 56,757 | 0 | 0.00 | - | 0.0 | 0.0 | 56757 rows present for this country but energy_generation.wind_offshore_mw is NULL in every one -- ENTSO-E does not report this production type here |

## Status-quo source census (`energy_renewable`)

| country | stream | ren first | ren last | ren non-null | ren % zero | zero-filled rows | peak contradicted MW | dup instants | dup disagreeing |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| AT | solar | 2025-11-07 23:00 | 2026-08-12 12:15 | 27,731 | 51.22 | 127 | 1,884.0 | 1,918 | 238 |
| AT | wind_onshore | 2025-11-07 23:00 | 2026-08-12 12:15 | 27,731 | 2.18 | 469 | 3,788.0 | 1,918 | 700 |
| AT | wind_offshore | 2025-11-07 23:00 | 2026-08-12 12:15 | 27,731 | 100.00 | 0 | 0.0 | 1,918 | 0 |
| BE | solar | 2024-01-01 00:00 | 2026-08-12 12:00 | 23,157 | 43.78 | 0 | 0.0 | 479 | 0 |
| BE | wind_onshore | 2024-01-01 00:00 | 2026-08-12 12:00 | 23,157 | 0.00 | 1 | 37.4 | 479 | 65 |
| BE | wind_offshore | 2024-01-01 00:00 | 2026-08-12 12:00 | 23,157 | 9.91 | 99 | 2,175.3 | 479 | 180 |
| BG | solar | 2021-01-01 00:00 | 2026-08-12 07:00 | 49,318 | 17.87 | 44 | 896.0 | 672 | 290 |
| BG | wind_onshore | 2021-01-01 00:00 | 2026-08-12 07:00 | 49,486 | 0.07 | 1 | 135.0 | 672 | 173 |
| BG | wind_offshore | 2021-01-01 00:00 | 2026-08-12 07:00 | 49,486 | 100.00 | 0 | 0.0 | 672 | 0 |
| CH | solar | 2021-01-01 00:00 | 2026-08-12 12:00 | 11,086 | 45.70 | 6 | 946.3 | 453 | 10 |
| CH | wind_onshore | 2021-01-01 00:00 | 2026-08-12 12:00 | 11,200 | 6.70 | 7 | 55.0 | 453 | 23 |
| CH | wind_offshore | 2021-01-01 00:00 | 2026-08-12 12:00 | 11,216 | 100.00 | 0 | 0.0 | 453 | 0 |
| CZ | solar | 2025-11-09 23:00 | 2026-08-12 12:45 | 26,724 | 41.47 | 148 | 952.8 | 768 | 0 |
| CZ | wind_onshore | 2025-11-09 23:00 | 2026-08-12 12:45 | 26,724 | 0.00 | 0 | 0.0 | 768 | 0 |
| CZ | wind_offshore | 2025-11-09 23:00 | 2026-08-12 12:45 | 26,724 | 100.00 | 0 | 0.0 | 768 | 0 |
| DE | solar | 2025-09-08 22:00 | 2026-08-12 13:00 | 33,503 | 51.00 | 2,773 | 43,408.5 | 1,924 | 78 |
| DE | wind_onshore | 2025-09-08 22:00 | 2026-08-12 13:00 | 33,472 | 0.01 | 0 | 0.0 | 1,924 | 27 |
| DE | wind_offshore | 2025-09-08 22:00 | 2026-08-12 13:00 | 33,503 | 0.04 | 0 | 0.0 | 1,924 | 0 |
| EE | solar | 2025-11-16 23:00 | 2026-08-12 12:45 | 23,587 | 3.75 | 217 | 77.5 | 192 | 96 |
| EE | wind_onshore | 2025-11-09 23:00 | 2026-08-12 12:45 | 23,752 | 0.53 | 33 | 366.1 | 192 | 96 |
| EE | wind_offshore | 2025-11-09 23:00 | 2026-08-12 12:45 | 23,753 | 100.00 | 0 | 0.0 | 192 | 0 |
| ES | solar | 2025-11-08 23:00 | 2026-08-12 13:00 | 27,631 | 2.49 | 275 | 18,176.0 | 1,921 | 0 |
| ES | wind_onshore | 2025-11-08 23:00 | 2026-08-12 13:00 | 27,631 | 2.06 | 554 | 17,500.0 | 1,921 | 0 |
| ES | wind_offshore | 2025-11-08 23:00 | 2026-08-12 13:00 | 27,631 | 100.00 | 0 | 0.0 | 1,921 | 0 |
| FI | solar | 2025-10-05 22:00 | 2026-08-12 12:45 | 30,819 | 2.50 | 16 | 161.4 | 1,922 | 0 |
| FI | wind_onshore | 2025-10-05 22:00 | 2026-08-12 12:45 | 30,819 | 0.21 | 28 | 2,258.7 | 1,922 | 0 |
| FI | wind_offshore | 2025-10-05 22:00 | 2026-08-12 12:45 | 30,883 | 100.00 | 0 | 0.0 | 1,922 | 0 |
| FR | solar | 2023-01-01 00:00 | 2026-08-12 12:45 | 76,999 | 38.26 | 1 | 247.2 | 3,819 | 0 |
| FR | wind_onshore | 2023-01-01 00:00 | 2026-08-12 12:45 | 76,999 | 0.05 | 0 | 0.0 | 3,819 | 0 |
| FR | wind_offshore | 2023-01-01 00:00 | 2026-08-12 12:45 | 76,999 | 6.48 | 0 | 0.0 | 3,819 | 0 |
| GR | solar | 2025-11-07 23:00 | 2026-08-12 13:00 | 13,870 | 36.35 | 16 | 5,855.0 | 192 | 61 |
| GR | wind_onshore | 2025-11-07 23:00 | 2026-08-12 13:00 | 13,870 | 0.19 | 26 | 3,059.0 | 192 | 96 |
| GR | wind_offshore | 2025-11-07 23:00 | 2026-08-12 13:00 | 13,870 | 100.00 | 0 | 0.0 | 192 | 0 |
| HR | solar | 2025-11-07 23:00 | 2026-08-12 13:15 | 26,921 | 44.68 | 0 | 0.0 | 768 | 1 |
| HR | wind_onshore | 2025-11-07 23:00 | 2026-08-12 13:15 | 26,921 | 0.29 | 0 | 0.0 | 768 | 3 |
| HR | wind_offshore | 2025-11-07 23:00 | 2026-08-12 13:15 | 26,921 | 100.00 | 0 | 0.0 | 768 | 0 |
| HU | solar | 2025-11-16 23:00 | 2026-08-12 13:00 | 26,055 | 0.30 | 0 | 0.0 | 768 | 209 |
| HU | wind_onshore | 2025-11-16 23:00 | 2026-08-12 13:00 | 26,055 | 0.10 | 0 | 0.0 | 768 | 255 |
| HU | wind_offshore | 2025-11-09 23:00 | 2026-08-12 13:00 | 26,727 | 100.00 | 0 | 0.0 | 768 | 0 |
| IT | solar | 2025-10-05 22:00 | 2026-08-12 13:00 | 30,641 | 45.10 | 0 | 0.0 | 1,922 | 16 |
| IT | wind_onshore | 2025-10-05 22:00 | 2026-08-12 13:00 | 30,641 | 0.08 | 3 | 7,992.0 | 1,922 | 90 |
| IT | wind_offshore | 2025-10-05 22:00 | 2026-08-12 13:00 | 30,641 | 37.55 | 14 | 2.0 | 1,922 | 8 |
| LT | solar | 2025-11-09 23:00 | 2026-08-12 13:15 | 26,644 | 39.86 | 1 | 294.8 | 768 | 0 |
| LT | wind_onshore | 2025-11-09 23:00 | 2026-08-12 13:15 | 26,644 | 0.00 | 1 | 965.6 | 768 | 0 |
| LT | wind_offshore | 2025-11-09 23:00 | 2026-08-12 13:15 | 26,644 | 100.00 | 0 | 0.0 | 768 | 0 |
| LV | solar | 2025-11-08 23:00 | 2026-08-12 12:00 | 6,700 | 34.39 | 374 | 952.9 | 192 | 96 |
| LV | wind_onshore | 2025-11-08 23:00 | 2026-08-12 12:00 | 6,700 | 4.27 | 30 | 57.0 | 192 | 96 |
| LV | wind_offshore | 2025-11-08 23:00 | 2026-08-12 12:00 | 6,700 | 100.00 | 0 | 0.0 | 192 | 0 |
| NL | solar | 2025-11-09 23:00 | 2026-08-12 13:00 | 27,485 | 41.37 | 356 | 176.2 | 1,812 | 601 |
| NL | wind_onshore | 2025-11-09 23:00 | 2026-08-12 13:00 | 27,485 | 1.92 | 467 | 2,032.8 | 1,812 | 668 |
| NL | wind_offshore | 2025-11-09 23:00 | 2026-08-12 13:00 | 27,485 | 1.79 | 447 | 4,537.6 | 1,812 | 668 |
| NO | solar | 2025-11-09 23:00 | 2026-08-12 12:45 | 27,413 | 57.08 | 21 | 1.2 | 1,812 | 0 |
| NO | wind_onshore | 2025-11-09 23:00 | 2026-08-12 12:45 | 27,413 | 0.10 | 28 | 2,151.8 | 1,812 | 0 |
| NO | wind_offshore | 2025-11-09 23:00 | 2026-08-12 12:45 | 27,413 | 3.82 | 28 | 3.0 | 1,812 | 0 |
| PL | solar | 2025-10-05 22:00 | 2026-08-12 13:00 | 30,856 | 47.10 | 0 | 0.0 | 1,812 | 0 |
| PL | wind_onshore | 2025-10-05 22:00 | 2026-08-12 13:00 | 30,856 | 0.00 | 0 | 0.0 | 1,812 | 0 |
| PL | wind_offshore | 2025-10-05 22:00 | 2026-08-12 13:00 | 30,856 | 91.76 | 0 | 0.0 | 1,812 | 0 |
| PT | solar | 2025-11-08 23:00 | 2026-08-12 12:00 | 6,896 | 35.25 | 58 | 2,016.9 | 453 | 71 |
| PT | wind_onshore | 2025-11-08 23:00 | 2026-08-12 12:00 | 6,896 | 0.00 | 0 | 0.0 | 453 | 125 |
| PT | wind_offshore | 2025-11-08 23:00 | 2026-08-12 12:00 | 6,896 | 20.14 | 0 | 0.0 | 453 | 31 |
| RO | solar | 2025-10-05 22:00 | 2026-08-12 13:15 | 29,975 | 50.47 | 0 | 0.0 | 768 | 0 |
| RO | wind_onshore | 2025-10-05 22:00 | 2026-08-12 13:15 | 29,975 | 1.17 | 0 | 0.0 | 768 | 0 |
| RO | wind_offshore | 2025-10-05 22:00 | 2026-08-12 13:15 | 29,975 | 100.00 | 0 | 0.0 | 768 | 0 |
| SE | solar | 2025-10-05 22:00 | 2026-08-12 12:45 | 24,766 | 0.07 | 5 | 276.8 | 451 | 111 |
| SE | wind_onshore | 2025-10-05 22:00 | 2026-08-12 12:45 | 24,766 | 0.03 | 7 | 6,508.9 | 451 | 124 |
| SE | wind_offshore | 2025-10-05 22:00 | 2026-08-12 12:45 | 24,782 | 100.00 | 0 | 0.0 | 451 | 0 |
| SI | solar | 2025-11-16 23:00 | 2026-08-12 13:15 | 12,608 | 0.17 | 0 | 0.0 | 192 | 71 |
| SI | wind_onshore | 2025-11-08 23:00 | 2026-08-12 13:15 | 12,800 | 15.38 | 30 | 3.1 | 192 | 48 |
| SI | wind_offshore | 2025-11-08 23:00 | 2026-08-12 13:15 | 12,800 | 100.00 | 0 | 0.0 | 192 | 0 |
| SK | solar | 2025-11-16 23:00 | 2026-08-12 12:45 | 14,097 | 1.99 | 0 | 0.0 | 192 | 0 |
| SK | wind_onshore | 2025-11-07 23:00 | 2026-08-12 12:45 | 14,313 | 22.75 | 0 | 0.0 | 192 | 0 |
| SK | wind_offshore | 2025-11-07 23:00 | 2026-08-12 12:45 | 14,313 | 100.00 | 0 | 0.0 | 192 | 0 |

## Verdict counts

**solar** — `TRAIN` 23, `EXCLUDE-NO-FLEET` 1

**wind_onshore** — `TRAIN` 22, `EXCLUDE-NO-FLEET` 2

**wind_offshore** — `EXCLUDE-NO-DATA` 15, `TRAIN` 4, `EXCLUDE-NO-FLEET` 4, `EXCLUDE-INSUFFICIENT-HISTORY` 1

Full machine-readable output with every column: `reports/abl_318_renewable_data_audit.csv`.
