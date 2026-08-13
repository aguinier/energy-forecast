# ABL-376 §5 — the fit rule over a seed spread

Generated: 2026-08-13 11:53 UTC.
Seeds: `101, 103, 107, 109, 113, 127, 131, 137` — frozen in `scripts/abl376_night_seed_spread.py` before the first fit, and disjoint from the gate's seed 42.
Fit targets 2026-01-14 00:00:00 → 2026-07-11 00:00:00 (exclusive); scored on the registered gate window 2026-07-11 00:00:00 → 2026-08-10 00:00:00 (exclusive), out-of-sample by target timestamp.
Replica `C:\Code\able\data\energy_dashboard.db` (9,432,453,120 bytes), source table `energy_renewable`, opened read-only.
Night is `solar_geometry.is_night_hour` (sun below -8 deg for the whole hour); the fit drops night rows above 1 MW and **the score drops nothing**.

## Night level — the result

Mean challenger prediction over the gate's night hours, MW. Both arms scored on identical unfiltered rows.

| country | night rows | control (mean ± sd) | night-fit (mean ± sd) | paired change | seeds moved down |
|---|---:|---:|---:|---:|---:|
| FR | 462 | 43.66 ± 10.06 | 43.33 ± 19.59 | -0.33 MW | 5/8 |
| DE | 420 | -0.17 ± 43.07 | -10.02 ± 14.68 | -9.85 MW | 4/8 |
| BE | 420 | -6.51 ± 3.19 | -6.51 ± 3.19 | +0.00 MW | 0/8 |

## Daylight MAE — the effect against its own null

`paired change` is the mean of (night-fit − control) taken **within** each seed. `single-seed null` is the largest gap between two control fits that differ only by seed — what a one-seed read could have reported with nothing changed at all.

| country | daylight rows | control MAE | paired change | as % | single-seed null (max) | verdict |
|---|---:|---:|---:|---:|---:|---|
| FR | 1,243 | 1,602.1 MW | +6.0 MW | +0.38% | 70.5 MW (4.40%) | inside the null |
| DE | 1,278 | 3,784.3 MW | -22.9 MW | -0.60% | 140.0 MW (3.70%) | inside the null |
| BE | 1,311 | 580.0 MW | +0.0 MW | +0.00% | 31.3 MW (5.39%) | inside the null |

## What the rule removed from each fit

| country | night fit rows | excluded rows | excluded hours | max excluded actual |
|---|---:|---:|---:|---:|
| FR | 11,648 | 904 | 113 | 285.9 MW |
| DE | 10,952 | 32 | 4 | 1.7 MW |
| BE | 10,856 | 0 | 0 | n/a |
