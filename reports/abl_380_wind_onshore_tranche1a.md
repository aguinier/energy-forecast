# Serve-faithful wind retrain gate — registered scope `abl380-tranche1a`

**Disposition: PASS**

Generated: 2026-08-13 08:34 UTC
Fit targets: 2026-01-14 00:00:00 → 2026-07-11 00:00:00 (exclusive).
Out-of-sample gate targets: 2026-07-11 00:00:00 → 2026-08-10 00:00:00 (exclusive).
Baseline: literal seasonal-naive D-7. TSO is revision-contaminated context only and is not a gate criterion.
Replica: `C:\Code\able\data\energy_dashboard.db` (9,432,453,120 bytes), opened with SQLite `mode=ro`, `uri=True`.
That one file is the source of the TSO series, the contamination screen, and — since ABL-355 — the fitted target series, its lag/rolling features, the D-7 and persistence baselines, the gate actuals and the weather archive. The incumbent forecasts are the only read it does not hold alone; see the sidecar below.
Sidecar: `C:\Code\able\data\forecasts_local.db`, also opened `mode=ro`, and read for locally generated incumbent forecasts only. Where a sidecar row and a replica row carry the same vintage, the sidecar's is the one scored.

## Gate read

Registered scope `abl380-tranche1a`: BG wind_onshore, CH wind_onshore.
Target series, features, baselines and contamination screen: `energy_generation`.
Gate basis — the columns that must be simultaneously finite for a row to be scored: `challenger`, `seasonal_naive`. Comparators outside the basis are scored on their own intersection with it and carry their own n, so a comparator that does not exist for a pair reads Not measured instead of emptying the cell.
Strict full PASS requires challenger WAPE < D-7 in all 6 country × primary D+2-band cells and ≥95% of intended pairs. Result: **6/6 cells pass**.

| type | country | horizon | n | challenger WAPE | D-7 WAPE | skill vs D-7 | incumbent WAPE | MAE | bias | slope | corr | gate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| wind_onshore | BG | 24-36h | 720 | 56.9% | 93.8% | +39.3% | Not measured | 61.9 MW | -18.7% | 0.3 | 0.6 | PASS |
| wind_onshore | BG | 36-48h | 720 | 56.8% | 93.8% | +39.4% | Not measured | 61.9 MW | -17.4% | 0.3 | 0.6 | PASS |
| wind_onshore | BG | 48-64h | 510 | 57.8% | 89.3% | +35.3% | Not measured | 57.0 MW | -12.5% | 0.2 | 0.5 | PASS |
| wind_onshore | CH | 24-36h | 720 | 47.4% | 59.3% | +20.0% | Not measured | 6.1 MW | 13.2% | 0.1 | 0.1 | PASS |
| wind_onshore | CH | 36-48h | 720 | 45.0% | 59.3% | +24.1% | Not measured | 5.8 MW | 9.4% | 0.1 | 0.2 | PASS |
| wind_onshore | CH | 48-64h | 510 | 44.3% | 59.8% | +25.9% | Not measured | 5.7 MW | 13.4% | 0.1 | 0.2 | PASS |

## Per-country all-D+2 summary

Gate-basis values (actual, challenger, seasonal_naive) share one finite intersection; each comparator outside the basis is scored on its own intersection with it, and its n is given in `comparator_n` in the JSON. A comparator showing `Not measured` had no finite rows at all.

| type | country | n | challenger WAPE | D-7 WAPE | persistence WAPE | incumbent WAPE | TSO WAPE (revision-contaminated; n) |
|---|---|---:|---:|---:|---:|---:|---:|
| wind_onshore | BG | 1,950 | 57.1% | 92.7% | 80.8% | Not measured | 50.1% (n=1,950) |
| wind_onshore | CH | 1,950 | 45.7% | 59.4% | 58.8% | Not measured | 27.8% (n=1,950) |
## Training cost

Wall-clock on the rail interpreter, one pair at a time in a single process. Feature build and fit are separated because they scale on different things. Measured under whatever else this workstation was running; treat as an upper bound for sizing, not a benchmark.

| type | country | fit rows | feature build | fit | gate build + predict | pair total |
|---|---|---:|---:|---:|---:|---:|
| wind_onshore | BG | 34,176 | 44.6 s | 3.6 s | 7.2 s | **55.3 s** |
| wind_onshore | CH | 34,176 | 52.7 s | 4.3 s | 8.3 s | **65.3 s** |

Scope total across 2 pair(s): **120.6 s** (60.3 s mean per pair).

## Fit and missingness audit

Each training row was constructed by `RenewableFeatureBuilder.row(target, generated_at, generated_at)` on the measured eight-vintage schedule. Gate targets were never fitted.

| type | country | algorithm | retained / intended fit rows | unique fit targets | excluded missing | degraded lag-1d rows | artifact SHA-256 |
|---|---|---|---:|---:|---:|---:|---|
| wind_onshore | BG | catboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `eb0f63d8c5ccf395a0d80a443257dc5a91bb61620eec7b30c3b9c862037b43ea` |
| wind_onshore | CH | catboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `5d2ec407b47f727d107a2121d86b27fb4ff8e316bedfa2ca749a406c70990840` |

## Data quality and limits

- ABL-188 constant-run screening found no ≥24-hour bit-identical wind run in any fitted/scored pair; no wind row was excluded by that invariant.
- ABL-67 is net-position-only; ABL-109/111 are load-only. They do not intersect these wind targets. ABL-71's known wrong-write modes are load and net position, not wind; this is a provenance caveat, not proof that wind ingest is pristine.
- Weather rows were admitted only where `forecast_run_time <= generated_at`; missing vintages were excluded and counted, never backfilled from the future.
- TSO values come from a replacement table without first-seen vintages. They may include revisions and cannot support promotion.
- **BG wind_onshore: the TSO forecast is better than the challenger** (50.1% vs 57.1% WAPE over the same n=1,950). The gate is against D-7 and this pair clears it, but clearing D-7 is not evidence the model is good — it bounds how uninformative D-7 is. Treat the TSO series as a feature to ingest, not merely as context.
- **CH wind_onshore: the TSO forecast is better than the challenger** (27.8% vs 45.7% WAPE over the same n=1,950). The gate is against D-7 and this pair clears it, but clearing D-7 is not evidence the model is good — it bounds how uninformative D-7 is. Treat the TSO series as a feature to ingest, not merely as context.
- This is one 30-day summer holdout. It is out-of-sample by target timestamp, but not a year-round robustness claim.

## Recommendation to the CEO

The challenger clears the pre-registered D-7 bar in every served D+2 country-band cell. Preserve these experiment artifacts and ask the CEO to initiate Board review; do not promote from this issue.

No production deploy, serving-registry change, model promotion, ingest change, dashboard change, replica write, or sidecar write was performed.
