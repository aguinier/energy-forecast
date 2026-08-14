# Serve-faithful wind retrain gate — registered scope `abl322-pilot`

**Disposition: PASS**

Generated: 2026-08-14 00:49 UTC
Fit targets: 2026-01-14 00:00:00 → 2026-07-11 00:00:00 (exclusive).
Out-of-sample gate targets: 2026-07-11 00:00:00 → 2026-08-10 00:00:00 (exclusive).
Baseline: literal seasonal-naive D-7. TSO is revision-contaminated context only and is not a gate criterion.
Replica: `C:\Code\able\data\energy_dashboard.db` (9,432,453,120 bytes), opened with SQLite `mode=ro`, `uri=True`.
That one file is the source of the TSO series, the contamination screen, and — since ABL-355 — the fitted target series, its lag/rolling features, the D-7 and persistence baselines, the gate actuals and the weather archive. The incumbent forecasts are the only read it does not hold alone; see the sidecar below.
Sidecar: `C:\Code\able\data\forecasts_local.db`, also opened `mode=ro`, and read for locally generated incumbent forecasts only. Where a sidecar row and a replica row carry the same vintage, the sidecar's is the one scored.

## Gate read

Registered scope `abl322-pilot`: DE wind_offshore, NL wind_offshore.
Target series, features, baselines and contamination screen: `energy_generation`.
Gate basis — the columns that must be simultaneously finite for a row to be scored: `challenger`, `seasonal_naive`. Comparators outside the basis are scored on their own intersection with it and carry their own n, so a comparator that does not exist for a pair reads Not measured instead of emptying the cell.
Strict full PASS requires challenger WAPE < D-7 in all 6 country × primary D+2-band cells and ≥95% of intended pairs. Result: **6/6 cells pass**.

Model-free reference (ABL-389) — four predictors with no model in them, reported beside every cell. `constant_causal` is a flat line at the **fit-window mean**, the honest "no model" floor, using only what was knowable before the gate window opened; `constant_oracle` is a flat line at the **gate-window median**, a hindsight upper bound on what any constant could have achieved. `climatology_causal` and `climatology_oracle` are the same two forms taken **per hour of day** — the fit-window hourly mean and the gate-window hourly median — which is the tighter reference on every pair measured so far, because a constant is a climatology with one bucket. Read the pair together: the constant says whether the model predicts the *level*, the climatology says whether it predicts the level *and the daily shape*, and the gap between them is how much of this series is forced diurnal structure.

All four are **reported references and not gate criteria**: none is in the gate basis, none can move a cell's verdict, and a pair that clears its D-7 bar while losing to one still reads PASS. They are the number that qualifies the PASS — a challenger that does not beat `climatology_oracle` has not demonstrated skill beyond the average day, and a D-7 bar that `constant_causal` clears on its own was not a demanding bar. **Check each reference's own n before comparing it to the challenger.** A climatology is 24 levels, so an hour of day absent from its source window leaves those rows unscored for that column alone; scored on different rows, two WAPEs are not the same measurement. Nothing is interpolated to close that gap.

Graded disposition (ABL-418) — the registered bar is **not** re-opened. Seasonal-naive D-7 is still the gate, ABL-348's windows, bands, metric, minimum n and source are unchanged, and a cell that clears D-7 still reads PASS. What the grade adds is **what that PASS entitles the cell to**. ABL-406 measured across eight wind pairs that the gate outcome was fully predicted by whether a causal constant clears the bar on its own — five weak bars, five passes; three strong bars, three failures or ties — and that NO passed 3/3 while anti-correlated with its own target. A PASS is necessary and not sufficient.

**G1** gate: beats D-7 by more than the readability floor — ABL-385's `delta_min(k)` with `c_B = 0`, since every reference here is deterministic, which is **7.51%** for this stream at k=1. **G2** level: beats `constant_causal`. **G3** shape: beats `climatology_causal`. **G4** direction: slope > 0 and corr > 0. **A** = all four in every band (promotion-eligible, subject to any named data hold); **B** = G1 holds and one or more of G2/G3/G4 fails, named; **C** = a readable loss to D-7; **U** = the G1 margin sits inside the floor, so the cell is unreadable at one seed — **U(+)** where G2–G4 clear readably, meaning *re-read at k>1 seeds*, not *reject*.

Causal references only. The two oracle references stay reported and gate nothing: an oracle is not causally available, so losing to one bounds what a verdict means rather than voiding it. The bar-weakness flag — does `constant_causal` clear the registered D-7 bar on its own? — is reported for the same reason. Neither is on the ladder. A condition that could not be measured is not satisfied, and is named like any other failure.

| type | country | horizon | n | challenger WAPE | D-7 WAPE | skill vs D-7 | constant causal WAPE | constant oracle WAPE | climatology causal WAPE | climatology oracle WAPE | incumbent WAPE | MAE | bias | slope | corr | gate | grade |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| wind_offshore | DE | 24-36h | 720 | 66.1% | 88.9% | +25.6% | 74.4% | 62.9% | 73.7% | 62.1% | Not measured | 1660.3 MW | -3.0% | 0.1 | 0.1 | PASS | A |
| wind_offshore | DE | 36-48h | 720 | 65.7% | 88.9% | +26.1% | 74.4% | 62.9% | 73.7% | 62.1% | Not measured | 1648.9 MW | -2.3% | 0.1 | 0.2 | PASS | A |
| wind_offshore | DE | 48-64h | 510 | 66.1% | 87.1% | +24.0% | 75.8% | 62.3% | 73.6% | 61.3% | Not measured | 1590.3 MW | -3.7% | 0.1 | 0.1 | PASS | A |
| wind_offshore | NL | 24-36h | 720 | 60.5% | 81.8% | +26.1% | 90.0% | 71.7% | 87.3% | 70.9% | Not measured | 686.4 MW | -14.3% | 0.2 | 0.5 | PASS | A |
| wind_offshore | NL | 36-48h | 720 | 61.3% | 81.8% | +25.1% | 90.0% | 71.7% | 87.3% | 70.9% | Not measured | 695.5 MW | -12.1% | 0.2 | 0.4 | PASS | A |
| wind_offshore | NL | 48-64h | 510 | 63.8% | 88.5% | +28.0% | 93.0% | 73.3% | 88.4% | 72.3% | Not measured | 724.5 MW | -14.5% | 0.2 | 0.4 | PASS | A |

### Graded disposition, per pair

| pair | bands | grade | failed conditions | bar weaker than a flat line? |
|---|---|:---:|---|:---:|
| DE wind_offshore | A / A / A | **A** | — | yes |
| NL wind_offshore | A / A / A | **A** | — | no |

Reference levels used, from the same ABL-188-filtered target series the gate actuals and the D-7/persistence baselines come from — no refit, no second read, no additional upstream fetch. The hourly levels behind the climatology columns are in `results.json` in full; `h` is how many of the 24 hours of the day that level set covers, and anything below 24 means those rows were dropped from that column's n:

| type | country | constant causal | constant oracle | climatology causal | climatology oracle |
|---|---|---:|---:|---:|---:|
| wind_offshore | DE | 3263.82 MW | 2005.99 MW | 2794.83–3541.11 MW (24h) | 1463.55–2468.13 MW (24h) |
| wind_offshore | NL | 1704.09 MW | 870.25 MW | 1344.58–1909.78 MW (24h) | 616.29–1105.70 MW (24h) |

## Per-country all-D+2 summary

Gate-basis values (actual, challenger, seasonal_naive) share one finite intersection; each comparator outside the basis is scored on its own intersection with it, and its n is given in `comparator_n` in the JSON. A comparator showing `Not measured` had no finite rows at all.

| type | country | n | challenger WAPE | D-7 WAPE | persistence WAPE | constant causal WAPE | constant oracle WAPE | climatology causal WAPE | climatology oracle WAPE | incumbent WAPE | TSO WAPE (revision-contaminated; n) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wind_offshore | DE | 1,950 | 66.0% | 88.4% | 78.6% | 74.8% | 62.7% | 73.7% | 61.9% | Not measured | 21.1% (n=1,950) |
| wind_offshore | NL | 1,950 | 61.6% | 83.5% | 92.4% | 90.8% | 72.1% | 87.6% | 71.3% | Not measured | 69.0% (n=1,950) |
## Training cost

Wall-clock on the rail interpreter, one pair at a time in a single process. Feature build and fit are separated because they scale on different things. Measured under whatever else this workstation was running; treat as an upper bound for sizing, not a benchmark.

| type | country | fit rows | feature build | fit | gate build + predict | pair total |
|---|---|---:|---:|---:|---:|---:|
| wind_offshore | DE | 34,176 | 42.7 s | 1.9 s | 6.9 s | **51.6 s** |
| wind_offshore | NL | 34,176 | 40.6 s | 1.6 s | 6.8 s | **49.0 s** |

Scope total across 2 pair(s): **100.6 s** (50.3 s mean per pair).

## Fit and missingness audit

Each training row was constructed by `RenewableFeatureBuilder.row(target, generated_at, generated_at)` on the measured eight-vintage schedule. Gate targets were never fitted.

| type | country | algorithm | retained / intended fit rows | unique fit targets | excluded missing | degraded lag-1d rows | artifact SHA-256 |
|---|---|---|---:|---:|---:|---:|---|
| wind_offshore | DE | xgboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `5900d9673d96c0d513277a0405ee9de85880a8db3ddd2c8a5424f134cf79777b` |
| wind_offshore | NL | xgboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `d7d65793edd87ed59be88e65ab60b58715df6208fbc932eb876bf5f8e6449e1c` |

## Data quality and limits

- ABL-188 constant-run screening found no ≥24-hour bit-identical wind run in any fitted/scored pair; no wind row was excluded by that invariant.
- ABL-67 is net-position-only; ABL-109/111 are load-only. They do not intersect these wind targets. ABL-71's known wrong-write modes are load and net position, not wind; this is a provenance caveat, not proof that wind ingest is pristine.
- Weather rows were admitted only where `forecast_run_time <= generated_at`; missing vintages were excluded and counted, never backfilled from the future.
- TSO values come from a replacement table without first-seen vintages. They may include revisions and cannot support promotion.
- **DE wind_offshore: the TSO forecast is better than the challenger** (21.1% vs 66.0% WAPE over the same n=1,950). The gate is against D-7 and this pair clears it, but clearing D-7 is not evidence the model is good — it bounds how uninformative D-7 is. Treat the TSO series as a feature to ingest, not merely as context.
- **The challenger loses to a constant chosen with hindsight in 3 cell(s).** A flat line at the gate-window median scores better than the fitted model there, so whatever those cells earn over the D-7 bar they earn by predicting close to the level and varying little around it — and not even at the best level. This does not change any verdict above; it bounds what the verdict means:
  - DE wind_offshore 24-36h: challenger 66.11% vs oracle constant 62.86% (+3.25pp)
  - DE wind_offshore 36-48h: challenger 65.66% vs oracle constant 62.86% (+2.80pp)
  - DE wind_offshore 48-64h: challenger 66.15% vs oracle constant 62.32% (+3.83pp)
- **The challenger loses to a climatology chosen with hindsight in 3 cell(s).** An hour-of-day median — the average day, with no model and no weather in it — scores better than the fitted model there. This is the weaker claim to lose and the stronger one to win: a challenger that beats it is doing something no table of hourly averages can do. This does not change any verdict above; it bounds what the verdict means:
  - DE wind_offshore 24-36h: challenger 66.11% vs oracle climatology 62.12% (+3.99pp)
  - DE wind_offshore 36-48h: challenger 65.66% vs oracle climatology 62.12% (+3.53pp)
  - DE wind_offshore 48-64h: challenger 66.15% vs oracle climatology 61.31% (+4.84pp)
- This is one 30-day summer holdout. It is out-of-sample by target timestamp, but not a year-round robustness claim.

## Recommendation to the CEO

The challenger clears the pre-registered D-7 bar in every served D+2 country-band cell. Preserve these experiment artifacts and ask the CEO to initiate Board review; do not promote from this issue.

No production deploy, serving-registry change, model promotion, ingest change, dashboard change, replica write, or sidecar write was performed.
