# Serve-faithful wind retrain gate — registered scope `abl417-tranche2e`

**Disposition: PASS**

Generated: 2026-08-13 22:57 UTC
Fit targets: 2026-01-14 00:00:00 → 2026-07-11 00:00:00 (exclusive).
Out-of-sample gate targets: 2026-07-11 00:00:00 → 2026-08-10 00:00:00 (exclusive).
Baseline: literal seasonal-naive D-7. TSO is revision-contaminated context only and is not a gate criterion.
Replica: `C:\Code\able\data\energy_dashboard.db` (9,432,453,120 bytes), opened with SQLite `mode=ro`, `uri=True`.
That one file is the source of the TSO series, the contamination screen, and — since ABL-355 — the fitted target series, its lag/rolling features, the D-7 and persistence baselines, the gate actuals and the weather archive. The incumbent forecasts are the only read it does not hold alone; see the sidecar below.
Sidecar: `C:\Code\able\data\forecasts_local.db`, also opened `mode=ro`, and read for locally generated incumbent forecasts only. Where a sidecar row and a replica row carry the same vintage, the sidecar's is the one scored.
`ENERGY_DB_PATH` resolved to `\data\energy_dashboard.db` and was **not** read by this run. Before ABL-355 that path, not the replica, is where the fitted series would have come from.

## Gate read

Registered scope `abl417-tranche2e`: CZ wind_onshore, EE wind_onshore, HR wind_onshore, HU wind_onshore, LT wind_onshore, LV wind_onshore, NL wind_onshore, RO wind_onshore.
Target series, features, baselines and contamination screen: `energy_generation`.
Gate basis — the columns that must be simultaneously finite for a row to be scored: `challenger`, `seasonal_naive`. Comparators outside the basis are scored on their own intersection with it and carry their own n, so a comparator that does not exist for a pair reads Not measured instead of emptying the cell.
Strict full PASS requires challenger WAPE < D-7 in all 24 country × primary D+2-band cells and ≥95% of intended pairs. Result: **24/24 cells pass**.

Model-free reference (ABL-389) — four predictors with no model in them, reported beside every cell. `constant_causal` is a flat line at the **fit-window mean**, the honest "no model" floor, using only what was knowable before the gate window opened; `constant_oracle` is a flat line at the **gate-window median**, a hindsight upper bound on what any constant could have achieved. `climatology_causal` and `climatology_oracle` are the same two forms taken **per hour of day** — the fit-window hourly mean and the gate-window hourly median — which is the tighter reference on every pair measured so far, because a constant is a climatology with one bucket. Read the pair together: the constant says whether the model predicts the *level*, the climatology says whether it predicts the level *and the daily shape*, and the gap between them is how much of this series is forced diurnal structure.

All four are **reported references and not gate criteria**: none is in the gate basis, none can move a cell's verdict, and a pair that clears its D-7 bar while losing to one still reads PASS. They are the number that qualifies the PASS — a challenger that does not beat `climatology_oracle` has not demonstrated skill beyond the average day, and a D-7 bar that `constant_causal` clears on its own was not a demanding bar. **Check each reference's own n before comparing it to the challenger.** A climatology is 24 levels, so an hour of day absent from its source window leaves those rows unscored for that column alone; scored on different rows, two WAPEs are not the same measurement. Nothing is interpolated to close that gap.

Graded disposition (ABL-418) — the registered bar is **not** re-opened. Seasonal-naive D-7 is still the gate, ABL-348's windows, bands, metric, minimum n and source are unchanged, and a cell that clears D-7 still reads PASS. What the grade adds is **what that PASS entitles the cell to**. ABL-406 measured across eight wind pairs that the gate outcome was fully predicted by whether a causal constant clears the bar on its own — five weak bars, five passes; three strong bars, three failures or ties — and that NO passed 3/3 while anti-correlated with its own target. A PASS is necessary and not sufficient.

**G1** gate: beats D-7 by more than the readability floor — ABL-385's `delta_min(k)` with `c_B = 0`, since every reference here is deterministic, which is **7.51%** for this stream at k=1. **G2** level: beats `constant_causal`. **G3** shape: beats `climatology_causal`. **G4** direction: slope > 0 and corr > 0. **A** = all four in every band (promotion-eligible, subject to any named data hold); **B** = G1 holds and one or more of G2/G3/G4 fails, named; **C** = a readable loss to D-7; **U** = the G1 margin sits inside the floor, so the cell is unreadable at one seed — **U(+)** where G2–G4 clear readably, meaning *re-read at k>1 seeds*, not *reject*.

Causal references only. The two oracle references stay reported and gate nothing: an oracle is not causally available, so losing to one bounds what a verdict means rather than voiding it. The bar-weakness flag — does `constant_causal` clear the registered D-7 bar on its own? — is reported for the same reason. Neither is on the ladder. A condition that could not be measured is not satisfied, and is named like any other failure.

| type | country | horizon | n | challenger WAPE | D-7 WAPE | skill vs D-7 | constant causal WAPE | constant oracle WAPE | climatology causal WAPE | climatology oracle WAPE | incumbent WAPE | MAE | bias | slope | corr | gate | grade |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| wind_onshore | CZ | 24-36h | 720 | 44.8% | 86.4% | +48.2% | 54.8% | 46.1% | 53.4% | 42.7% | Not measured | 28.3 MW | 19.9% | 0.4 | 0.5 | PASS | A |
| wind_onshore | CZ | 36-48h | 720 | 45.0% | 86.4% | +47.9% | 54.8% | 46.1% | 53.4% | 42.7% | Not measured | 28.5 MW | 19.5% | 0.4 | 0.5 | PASS | A |
| wind_onshore | CZ | 48-64h | 510 | 47.4% | 86.1% | +44.9% | 57.8% | 47.8% | 56.5% | 44.2% | Not measured | 29.6 MW | 22.9% | 0.4 | 0.5 | PASS | A |
| wind_onshore | EE | 24-36h | 685 | 42.7% | 85.8% | +50.2% | 73.0% | 62.2% | 73.1% | 59.7% | Not measured | 44.5 MW | -16.1% | 0.4 | 0.7 | PASS | A |
| wind_onshore | EE | 36-48h | 684 | 42.4% | 85.7% | +50.6% | 72.9% | 62.2% | 73.0% | 59.6% | Not measured | 44.2 MW | -11.5% | 0.5 | 0.7 | PASS | A |
| wind_onshore | EE | 48-64h | 475 | 46.0% | 86.7% | +47.0% | 73.5% | 64.2% | 73.6% | 61.7% | Not measured | 49.0 MW | -13.8% | 0.4 | 0.7 | PASS | A |
| wind_onshore | HR | 24-36h | 720 | 74.1% | 97.7% | +24.2% | 92.0% | 69.2% | 91.0% | 65.7% | Not measured | 167.4 MW | 31.4% | 0.3 | 0.5 | PASS | A |
| wind_onshore | HR | 36-48h | 720 | 68.6% | 97.7% | +29.8% | 92.0% | 69.2% | 91.0% | 65.7% | Not measured | 155.1 MW | 18.1% | 0.3 | 0.5 | PASS | A |
| wind_onshore | HR | 48-64h | 510 | 60.8% | 88.5% | +31.3% | 89.3% | 65.0% | 85.3% | 60.2% | Not measured | 133.8 MW | 7.6% | 0.3 | 0.5 | PASS | A |
| wind_onshore | HU | 24-36h | 720 | 104.9% | 124.2% | +15.5% | 103.1% | 72.1% | 102.7% | 70.5% | Not measured | 43.7 MW | 53.9% | 0.2 | 0.2 | PASS | B — fails G2, G3 |
| wind_onshore | HU | 36-48h | 720 | 105.4% | 124.2% | +15.1% | 103.1% | 72.1% | 102.7% | 70.5% | Not measured | 43.9 MW | 55.2% | 0.2 | 0.2 | PASS | B — fails G2, G3 |
| wind_onshore | HU | 48-64h | 510 | 103.9% | 124.5% | +16.5% | 99.0% | 71.3% | 99.7% | 69.6% | Not measured | 44.9 MW | 55.9% | 0.2 | 0.2 | PASS | B — fails G2, G3 |
| wind_onshore | LT | 24-36h | 720 | 56.4% | 100.5% | +43.9% | 90.4% | 66.2% | 88.7% | 61.4% | Not measured | 215.0 MW | 25.3% | 0.5 | 0.7 | PASS | A |
| wind_onshore | LT | 36-48h | 720 | 56.1% | 100.5% | +44.1% | 90.4% | 66.2% | 88.7% | 61.4% | Not measured | 214.1 MW | 25.3% | 0.5 | 0.7 | PASS | A |
| wind_onshore | LT | 48-64h | 510 | 61.0% | 99.2% | +38.5% | 94.1% | 67.9% | 93.1% | 62.8% | Not measured | 229.8 MW | 26.9% | 0.4 | 0.6 | PASS | A |
| wind_onshore | LV | 24-36h | 708 | 89.0% | 97.5% | +8.7% | 72.1% | 69.6% | 70.4% | 67.2% | Not measured | 30.7 MW | 51.0% | 0.1 | 0.1 | PASS | B — fails G2, G3 |
| wind_onshore | LV | 36-48h | 708 | 90.3% | 97.5% | +7.4% | 72.1% | 69.6% | 70.4% | 67.2% | Not measured | 31.2 MW | 52.7% | 0.0 | 0.1 | PASS | U — fails G2, G3 |
| wind_onshore | LV | 48-64h | 506 | 90.9% | 97.1% | +6.4% | 71.0% | 68.3% | 69.6% | 66.6% | Not measured | 31.1 MW | 55.5% | 0.0 | 0.1 | PASS | U — fails G2, G3 |
| wind_onshore | NL | 24-36h | 720 | 78.0% | 94.9% | +17.8% | 225.5% | 73.8% | 225.8% | 72.4% | Not measured | 175.9 MW | 45.9% | 0.4 | 0.6 | PASS | A |
| wind_onshore | NL | 36-48h | 720 | 82.2% | 94.9% | +13.4% | 225.5% | 73.8% | 225.8% | 72.4% | Not measured | 185.3 MW | 46.4% | 0.4 | 0.5 | PASS | A |
| wind_onshore | NL | 48-64h | 510 | 82.2% | 94.8% | +13.3% | 217.9% | 74.6% | 220.2% | 72.7% | Not measured | 190.4 MW | 44.2% | 0.4 | 0.5 | PASS | A |
| wind_onshore | RO | 24-36h | 720 | 79.5% | 103.7% | +23.3% | 93.1% | 71.3% | 92.4% | 68.3% | Not measured | 391.5 MW | -27.6% | -0.0 | -0.0 | PASS | B — fails G4 |
| wind_onshore | RO | 36-48h | 720 | 80.3% | 103.7% | +22.6% | 93.1% | 71.3% | 92.4% | 68.3% | Not measured | 395.4 MW | -28.8% | -0.0 | -0.0 | PASS | B — fails G4 |
| wind_onshore | RO | 48-64h | 510 | 79.8% | 98.6% | +19.1% | 91.0% | 69.7% | 90.8% | 67.4% | Not measured | 389.6 MW | -27.5% | -0.0 | -0.1 | PASS | B — fails G4 |

### Graded disposition, per pair

| pair | bands | grade | failed conditions | bar weaker than a flat line? |
|---|---|:---:|---|:---:|
| CZ wind_onshore | A / A / A | **A** | — | yes |
| EE wind_onshore | A / A / A | **A** | — | yes |
| HR wind_onshore | A / A / A | **A** | — | yes |
| HU wind_onshore | B / B / B | **B** | G2 (beats constant_causal -- a flat line at the fit-window mean), G3 (beats climatology_causal -- an hour-of-day mean over the fit window) | yes |
| LT wind_onshore | A / A / A | **A** | — | yes |
| LV wind_onshore | B / U / U | **B** | G2 (beats constant_causal -- a flat line at the fit-window mean), G3 (beats climatology_causal -- an hour-of-day mean over the fit window) | yes |
| NL wind_onshore | A / A / A | **A** | — | no |
| RO wind_onshore | B / B / B | **B** | G4 (slope > 0 and correlation > 0) | yes |

Reference levels used, from the same ABL-188-filtered target series the gate actuals and the D-7/persistence baselines come from — no refit, no second read, no additional upstream fetch. The hourly levels behind the climatology columns are in `results.json` in full; `h` is how many of the 24 hours of the day that level set covers, and anything below 24 means those rows were dropped from that column's n:

| type | country | constant causal | constant oracle | climatology causal | climatology oracle |
|---|---|---:|---:|---:|---:|
| wind_onshore | CZ | 78.08 MW | 52.77 MW | 70.45–84.62 MW (24h) | 38.14–80.03 MW (24h) |
| wind_onshore | EE | 141.65 MW | 88.88 MW | 112.12–172.32 MW (24h) | 40.88–121.10 MW (24h) |
| wind_onshore | HR | 335.92 MW | 165.50 MW | 277.46–374.53 MW (24h) | 67.35–273.55 MW (24h) |
| wind_onshore | HU | 65.53 MW | 29.09 MW | 51.93–78.63 MW (24h) | 15.01–45.02 MW (24h) |
| wind_onshore | LT | 598.46 MW | 290.91 MW | 463.09–738.00 MW (24h) | 90.06–504.99 MW (24h) |
| wind_onshore | LV | 34.87 MW | 27.00 MW | 26.35–40.21 MW (24h) | 12.00–39.50 MW (24h) |
| wind_onshore | NL | 728.03 MW | 159.14 MW | 663.71–787.74 MW (24h) | 101.64–216.30 MW (24h) |
| wind_onshore | RO | 682.88 MW | 318.25 MW | 591.01–749.59 MW (24h) | 137.88–478.12 MW (24h) |

## Per-country all-D+2 summary

Gate-basis values (actual, challenger, seasonal_naive) share one finite intersection; each comparator outside the basis is scored on its own intersection with it, and its n is given in `comparator_n` in the JSON. A comparator showing `Not measured` had no finite rows at all.

| type | country | n | challenger WAPE | D-7 WAPE | persistence WAPE | constant causal WAPE | constant oracle WAPE | climatology causal WAPE | climatology oracle WAPE | incumbent WAPE | TSO WAPE (revision-contaminated; n) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wind_onshore | CZ | 1,950 | 45.5% | 86.3% | 68.3% | 55.5% | 46.6% | 54.2% | 43.1% | Not measured | 100.0% (n=1,950) |
| wind_onshore | EE | 1,844 | 43.4% | 86.0% | 92.9% | 73.1% | 62.7% | 73.2% | 60.2% | Not measured | 41.0% (n=1,844) |
| wind_onshore | HR | 1,950 | 68.6% | 95.3% | 85.3% | 91.3% | 68.1% | 89.6% | 64.3% | Not measured | 25.9% (n=1,950) |
| wind_onshore | HU | 1,950 | 104.8% | 124.3% | 109.1% | 102.0% | 71.9% | 101.9% | 70.3% | Not measured | 38.8% (n=1,950) |
| wind_onshore | LT | 1,950 | 57.5% | 100.2% | 92.5% | 91.4% | 66.7% | 89.8% | 61.8% | Not measured | 25.1% (n=1,950) |
| wind_onshore | LV | 1,922 | 90.0% | 97.4% | 101.1% | 71.8% | 69.3% | 70.2% | 67.0% | Not measured | 50.8% (n=1,922) |
| wind_onshore | NL | 1,950 | 80.6% | 94.9% | 106.1% | 223.5% | 74.0% | 224.3% | 72.5% | Not measured | 235.6% (n=1,950) |
| wind_onshore | RO | 1,950 | 79.9% | 102.4% | 87.4% | 92.5% | 70.9% | 91.9% | 68.1% | Not measured | 26.7% (n=1,950) |
## Training cost

Wall-clock on the rail interpreter, one pair at a time in a single process. Feature build and fit are separated because they scale on different things. Measured under whatever else this workstation was running; treat as an upper bound for sizing, not a benchmark.

| type | country | fit rows | feature build | fit | gate build + predict | pair total |
|---|---|---:|---:|---:|---:|---:|
| wind_onshore | CZ | 34,176 | 46.5 s | 3.7 s | 7.0 s | **57.3 s** |
| wind_onshore | EE | 33,784 | 44.9 s | 3.5 s | 7.0 s | **55.5 s** |
| wind_onshore | HR | 34,176 | 45.1 s | 3.6 s | 7.3 s | **56.0 s** |
| wind_onshore | HU | 34,176 | 47.3 s | 3.5 s | 7.6 s | **58.4 s** |
| wind_onshore | LT | 34,144 | 45.1 s | 3.8 s | 7.9 s | **56.7 s** |
| wind_onshore | LV | 34,176 | 46.9 s | 3.6 s | 7.2 s | **57.7 s** |
| wind_onshore | NL | 34,176 | 44.3 s | 3.5 s | 7.3 s | **55.0 s** |
| wind_onshore | RO | 34,176 | 46.1 s | 3.6 s | 8.3 s | **58.0 s** |

Scope total across 8 pair(s): **454.6 s** (56.8 s mean per pair).

## Fit and missingness audit

Each training row was constructed by `RenewableFeatureBuilder.row(target, generated_at, generated_at)` on the measured eight-vintage schedule. Gate targets were never fitted.

| type | country | algorithm | retained / intended fit rows | unique fit targets | excluded missing | degraded lag-1d rows | artifact SHA-256 |
|---|---|---|---:|---:|---:|---:|---|
| wind_onshore | CZ | catboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `649c26eeeac2a9f444b720cf465be0f1d674a50b8fb287aa2204282e0781e3c6` |
| wind_onshore | EE | catboost | 33,784 / 34,176 | 4,232 | 392 | 23,418 | `5efa9d4e44aed2a085d6105930d7c9ad8684bc80b0d68816405985dec7635d32` |
| wind_onshore | HR | catboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `72e98d5311386da7a5774301b96e2b51ba74a7b4b6994a7ebad9df347ffa2dce` |
| wind_onshore | HU | catboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `143ac511dfc19a438d2e07fb868d1d2e260e9cf3b761bb21de70ec721722266a` |
| wind_onshore | LT | catboost | 34,144 / 34,176 | 4,269 | 32 | 23,650 | `01d7e1bdef48c6603d507c750be5b887e401a4cd0021026098328e8841a9eacd` |
| wind_onshore | LV | catboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `08ae5dbd2bc906e68c4d5a42f3e6848429a0c51068d2b425a037f5054f21425c` |
| wind_onshore | NL | catboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `a3d2339fc30ad73bc43a11ad7f5739fbdd0e4f7170c1a1b2fede9d91012b70e5` |
| wind_onshore | RO | catboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `fc8971d2048ae84300c1203d75ae8bbc9d7f65d9e091153fffc3489abccc4d84` |

## Data quality and limits

- ABL-188 constant-run screening found no ≥24-hour bit-identical wind run in any fitted/scored pair; no wind row was excluded by that invariant.
- ABL-67 is net-position-only; ABL-109/111 are load-only. They do not intersect these wind targets. ABL-71's known wrong-write modes are load and net position, not wind; this is a provenance caveat, not proof that wind ingest is pristine.
- Weather rows were admitted only where `forecast_run_time <= generated_at`; missing vintages were excluded and counted, never backfilled from the future.
- TSO values come from a replacement table without first-seen vintages. They may include revisions and cannot support promotion.
- **EE wind_onshore: the TSO forecast is better than the challenger** (41.0% vs 43.4% WAPE over the same n=1,844). The gate is against D-7 and this pair clears it, but clearing D-7 is not evidence the model is good — it bounds how uninformative D-7 is. Treat the TSO series as a feature to ingest, not merely as context.
- **HR wind_onshore: the TSO forecast is better than the challenger** (25.9% vs 68.6% WAPE over the same n=1,950). The gate is against D-7 and this pair clears it, but clearing D-7 is not evidence the model is good — it bounds how uninformative D-7 is. Treat the TSO series as a feature to ingest, not merely as context.
- **HU wind_onshore: the TSO forecast is better than the challenger** (38.8% vs 104.8% WAPE over the same n=1,950). The gate is against D-7 and this pair clears it, but clearing D-7 is not evidence the model is good — it bounds how uninformative D-7 is. Treat the TSO series as a feature to ingest, not merely as context.
- **LT wind_onshore: the TSO forecast is better than the challenger** (25.1% vs 57.5% WAPE over the same n=1,950). The gate is against D-7 and this pair clears it, but clearing D-7 is not evidence the model is good — it bounds how uninformative D-7 is. Treat the TSO series as a feature to ingest, not merely as context.
- **LV wind_onshore: the TSO forecast is better than the challenger** (50.8% vs 90.0% WAPE over the same n=1,922). The gate is against D-7 and this pair clears it, but clearing D-7 is not evidence the model is good — it bounds how uninformative D-7 is. Treat the TSO series as a feature to ingest, not merely as context.
- **RO wind_onshore: the TSO forecast is better than the challenger** (26.7% vs 79.9% WAPE over the same n=1,950). The gate is against D-7 and this pair clears it, but clearing D-7 is not evidence the model is good — it bounds how uninformative D-7 is. Treat the TSO series as a feature to ingest, not merely as context.
- **The challenger loses to a constant chosen with hindsight in 13 cell(s).** A flat line at the gate-window median scores better than the fitted model there, so whatever those cells earn over the D-7 bar they earn by predicting close to the level and varying little around it — and not even at the best level. This does not change any verdict above; it bounds what the verdict means:
  - HR wind_onshore 24-36h: challenger 74.09% vs oracle constant 69.21% (+4.88pp)
  - HU wind_onshore 24-36h: challenger 104.93% vs oracle constant 72.13% (+32.81pp)
  - HU wind_onshore 36-48h: challenger 105.42% vs oracle constant 72.13% (+33.29pp)
  - HU wind_onshore 48-64h: challenger 103.90% vs oracle constant 71.34% (+32.56pp)
  - LV wind_onshore 24-36h: challenger 89.03% vs oracle constant 69.57% (+19.45pp)
  - LV wind_onshore 36-48h: challenger 90.30% vs oracle constant 69.57% (+20.73pp)
  - LV wind_onshore 48-64h: challenger 90.87% vs oracle constant 68.35% (+22.52pp)
  - NL wind_onshore 24-36h: challenger 77.97% vs oracle constant 73.85% (+4.12pp)
  - NL wind_onshore 36-48h: challenger 82.15% vs oracle constant 73.85% (+8.30pp)
  - NL wind_onshore 48-64h: challenger 82.23% vs oracle constant 74.56% (+7.67pp)
  - RO wind_onshore 24-36h: challenger 79.53% vs oracle constant 71.35% (+8.18pp)
  - RO wind_onshore 36-48h: challenger 80.32% vs oracle constant 71.35% (+8.97pp)
  - RO wind_onshore 48-64h: challenger 79.77% vs oracle constant 69.69% (+10.08pp)
- **The challenger loses to a climatology chosen with hindsight in 18 cell(s).** An hour-of-day median — the average day, with no model and no weather in it — scores better than the fitted model there. This is the weaker claim to lose and the stronger one to win: a challenger that beats it is doing something no table of hourly averages can do. This does not change any verdict above; it bounds what the verdict means:
  - CZ wind_onshore 24-36h: challenger 44.75% vs oracle climatology 42.69% (+2.07pp)
  - CZ wind_onshore 36-48h: challenger 45.03% vs oracle climatology 42.69% (+2.34pp)
  - CZ wind_onshore 48-64h: challenger 47.43% vs oracle climatology 44.17% (+3.26pp)
  - HR wind_onshore 24-36h: challenger 74.09% vs oracle climatology 65.67% (+8.42pp)
  - HR wind_onshore 36-48h: challenger 68.61% vs oracle climatology 65.67% (+2.94pp)
  - HR wind_onshore 48-64h: challenger 60.81% vs oracle climatology 60.18% (+0.64pp)
  - HU wind_onshore 24-36h: challenger 104.93% vs oracle climatology 70.51% (+34.42pp)
  - HU wind_onshore 36-48h: challenger 105.42% vs oracle climatology 70.51% (+34.90pp)
  - HU wind_onshore 48-64h: challenger 103.90% vs oracle climatology 69.64% (+34.26pp)
  - LV wind_onshore 24-36h: challenger 89.03% vs oracle climatology 67.16% (+21.87pp)
  - LV wind_onshore 36-48h: challenger 90.30% vs oracle climatology 67.16% (+23.14pp)
  - LV wind_onshore 48-64h: challenger 90.87% vs oracle climatology 66.55% (+24.31pp)
  - NL wind_onshore 24-36h: challenger 77.97% vs oracle climatology 72.37% (+5.60pp)
  - NL wind_onshore 36-48h: challenger 82.15% vs oracle climatology 72.37% (+9.78pp)
  - NL wind_onshore 48-64h: challenger 82.23% vs oracle climatology 72.69% (+9.54pp)
  - RO wind_onshore 24-36h: challenger 79.53% vs oracle climatology 68.34% (+11.19pp)
  - RO wind_onshore 36-48h: challenger 80.32% vs oracle climatology 68.34% (+11.98pp)
  - RO wind_onshore 48-64h: challenger 79.77% vs oracle climatology 67.35% (+12.41pp)
- This is one 30-day summer holdout. It is out-of-sample by target timestamp, but not a year-round robustness claim.

## Recommendation to the CEO

The challenger clears the pre-registered D-7 bar in every served D+2 country-band cell. Preserve these experiment artifacts and ask the CEO to initiate Board review; do not promote from this issue.

No production deploy, serving-registry change, model promotion, ingest change, dashboard change, replica write, or sidecar write was performed.
