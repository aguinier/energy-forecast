# ABL-376 §5 — the fit rule over a seed spread

Generated: 2026-08-13 12:11 UTC.
Seeds: `101, 103, 107, 109, 113, 127, 131, 137` — frozen in `scripts/abl376_night_seed_spread.py` before the first fit, and disjoint from the gate's seed 42.
Fit targets 2026-01-14 00:00:00 → 2026-07-11 00:00:00 (exclusive); scored on the registered gate window 2026-07-11 00:00:00 → 2026-08-10 00:00:00 (exclusive), out-of-sample by target timestamp.
Replica `C:\Code\able\data\energy_dashboard.db` (9,432,453,120 bytes), source table `energy_renewable`, opened read-only.
Feature set: **legacy25+geometry** (27 columns). **Exploratory — not the registered read.**
Night is `solar_geometry.is_night_hour` (sun below -8 deg for the whole hour); the fit drops night rows above 1 MW and **the score drops nothing**.

## Night level — the result

Mean challenger prediction over the gate's night hours, MW. Both arms scored on identical unfiltered rows.

| country | night rows | control (mean ± sd) | night-fit (mean ± sd) | paired change | seeds moved down |
|---|---:|---:|---:|---:|---:|
| FR | 462 | 58.39 ± 7.38 | 49.59 ± 19.87 | -8.81 MW | 7/8 |

## Daylight MAE — the effect against its own null

`paired change` is the mean of (night-fit − control) taken **within** each seed. `single-seed null` is the largest gap between two control fits that differ only by seed — what a one-seed read could have reported with nothing changed at all.

| country | daylight rows | control MAE | paired change | as % | single-seed null (max) | verdict |
|---|---:|---:|---:|---:|---:|---|
| FR | 1,243 | 1,599.5 MW | +3.1 MW | +0.19% | 83.0 MW (5.19%) | inside the null |

## What the rule removed from each fit

| country | night fit rows | excluded rows | excluded hours | max excluded actual |
|---|---:|---:|---:|---:|
| FR | 11,648 | 904 | 113 | 285.9 MW |
