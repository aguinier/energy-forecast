# Serve-faithful wind retrain gate — registered scope `abl322-pilot`

**Disposition: PASS**

Generated: 2026-08-13 06:12 UTC
Fit targets: 2026-01-14 00:00:00 → 2026-07-11 00:00:00 (exclusive).
Out-of-sample gate targets: 2026-07-11 00:00:00 → 2026-08-10 00:00:00 (exclusive).
Baseline: literal seasonal-naive D-7. TSO is revision-contaminated context only and is not a gate criterion.
Replica: `C:\Code\able\data\energy_dashboard.db` (9,432,453,120 bytes), opened with SQLite `mode=ro`, `uri=True`.

## Gate read

Registered scope `abl322-pilot`: DE wind_offshore, NL wind_offshore.
Target series, features, baselines and contamination screen: `energy_generation`.
Gate basis — the columns that must be simultaneously finite for a row to be scored: `challenger`, `seasonal_naive`. Comparators outside the basis are scored on their own intersection with it and carry their own n, so a comparator that does not exist for a pair reads Not measured instead of emptying the cell.
Strict full PASS requires challenger WAPE < D-7 in all 6 country × primary D+2-band cells and ≥95% of intended pairs. Result: **6/6 cells pass**.

| type | country | horizon | n | challenger WAPE | D-7 WAPE | skill vs D-7 | incumbent WAPE | MAE | bias | slope | corr | gate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| wind_offshore | DE | 24-36h | 720 | 66.1% | 88.9% | +25.6% | Not measured | 1660.3 MW | -3.0% | 0.1 | 0.1 | PASS |
| wind_offshore | DE | 36-48h | 720 | 65.7% | 88.9% | +26.1% | Not measured | 1648.9 MW | -2.3% | 0.1 | 0.2 | PASS |
| wind_offshore | DE | 48-64h | 510 | 66.1% | 87.1% | +24.0% | Not measured | 1590.3 MW | -3.7% | 0.1 | 0.1 | PASS |
| wind_offshore | NL | 24-36h | 720 | 60.5% | 81.8% | +26.1% | Not measured | 686.4 MW | -14.3% | 0.2 | 0.5 | PASS |
| wind_offshore | NL | 36-48h | 720 | 61.3% | 81.8% | +25.1% | Not measured | 695.5 MW | -12.1% | 0.2 | 0.4 | PASS |
| wind_offshore | NL | 48-64h | 510 | 63.8% | 88.5% | +28.0% | Not measured | 724.5 MW | -14.5% | 0.2 | 0.4 | PASS |

## Per-country all-D+2 summary

Gate-basis values (actual, challenger, seasonal_naive) share one finite intersection; each comparator outside the basis is scored on its own intersection with it, and its n is given in `comparator_n` in the JSON. A comparator showing `Not measured` had no finite rows at all.

| type | country | n | challenger WAPE | D-7 WAPE | persistence WAPE | incumbent WAPE | TSO WAPE (revision-contaminated; n) |
|---|---|---:|---:|---:|---:|---:|---:|
| wind_offshore | DE | 1,950 | 66.0% | 88.4% | 78.6% | Not measured | 21.1% (n=1,950) |
| wind_offshore | NL | 1,950 | 61.6% | 83.5% | 92.4% | Not measured | 69.0% (n=1,950) |
## Training cost

Wall-clock on the rail interpreter, one pair at a time in a single process. Feature build and fit are separated because they scale on different things. Measured under whatever else this workstation was running; treat as an upper bound for sizing, not a benchmark.

| type | country | fit rows | feature build | fit | gate build + predict | pair total |
|---|---|---:|---:|---:|---:|---:|
| wind_offshore | DE | 34,176 | 55.9 s | 2.3 s | 7.3 s | **65.5 s** |
| wind_offshore | NL | 34,176 | 46.1 s | 2.0 s | 8.6 s | **56.8 s** |

Scope total across 2 pair(s): **122.3 s** (61.1 s mean per pair).

## Fit and missingness audit

Each training row was constructed by `RenewableFeatureBuilder.row(target, generated_at, generated_at)` on the measured eight-vintage schedule. Gate targets were never fitted.

| type | country | algorithm | retained / intended fit rows | unique fit targets | excluded missing | degraded lag-1d rows | artifact SHA-256 |
|---|---|---|---:|---:|---:|---:|---|
| wind_offshore | DE | xgboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `c7151e3c7b2e7238282ac10d87c77a361a36900570588093187c7131f428bdb8` |
| wind_offshore | NL | xgboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `5292e38bbd4268497c2ceb43e6abddf9bdf95cc60292378619c8128ba89cc7d7` |

## Data quality and limits

- ABL-188 constant-run screening found no ≥24-hour bit-identical wind run in any fitted/scored pair; no wind row was excluded by that invariant.
- ABL-67 is net-position-only; ABL-109/111 are load-only. They do not intersect these wind targets. ABL-71's known wrong-write modes are load and net position, not wind; this is a provenance caveat, not proof that wind ingest is pristine.
- Weather rows were admitted only where `forecast_run_time <= generated_at`; missing vintages were excluded and counted, never backfilled from the future.
- TSO values come from a replacement table without first-seen vintages. They may include revisions and cannot support promotion.
- **DE wind_offshore: the TSO forecast is better than the challenger** (21.1% vs 66.0% WAPE over the same n=1,950). The gate is against D-7 and this pair clears it, but clearing D-7 is not evidence the model is good — it bounds how uninformative D-7 is. Treat the TSO series as a feature to ingest, not merely as context.
- This is one 30-day summer holdout. It is out-of-sample by target timestamp, but not a year-round robustness claim.

## Recommendation to the CEO

The challenger clears the pre-registered D-7 bar in every served D+2 country-band cell. Preserve these experiment artifacts and ask the CEO to initiate Board review; do not promote from this issue.

No production deploy, serving-registry change, model promotion, ingest change, dashboard change, replica write, or sidecar write was performed.
