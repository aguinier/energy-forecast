# ABL-421 — Serve-faithful solar retrain gate, ABL-316 tranche 2d: 6 northern countries on energy_generation at 27 features, 14 evaluable cells of 18

**Disposition: FAIL**

Generated: 2026-08-13 23:48 UTC
Fit targets: 2026-01-14 00:00:00 → 2026-07-11 00:00:00 (exclusive).
Out-of-sample gate targets: 2026-07-11 00:00:00 → 2026-08-10 00:00:00 (exclusive).
Baseline: literal seasonal-naive D-7. TSO is revision-contaminated context only and is not a gate criterion.
Replica: `C:\Code\able\data\energy_dashboard.db` (9,432,453,120 bytes), opened with SQLite `mode=ro`, `uri=True`.
That one file is the source of the TSO series, the contamination screen, and — since ABL-355 — the fitted target series, its lag/rolling features, the D-7 and persistence baselines, the gate actuals and the weather archive. The incumbent forecasts are the only read it does not hold alone; see the sidecar below.
Sidecar: `C:\Code\able\data\forecasts_local.db`, also opened `mode=ro`, and read for locally generated incumbent forecasts only. Where a sidecar row and a replica row carry the same vintage, the sidecar's is the one scored.
Target series, features, baselines and contamination screen: `energy_generation`.
Feature set: **legacy25+geometry** (27 columns), the module default -- this scope registers no feature set of its own.

## Gate read

Registered scope `abl316-t2d`: EE, FI, LT, LV, NL, SE.
Gate basis — the columns that must be simultaneously finite for a row to be scored: `challenger`, `seasonal_naive`. Comparators outside the basis are scored on their own intersection with it and carry their own n, so a comparator that does not exist for a country reads Not measured instead of emptying the cell.
Strict full PASS requires challenger WAPE < D-7 in all 14 country × primary D+2-band cells and ≥95% of intended pairs. Result: **12/14 cells pass**.

Model-free reference (ABL-389) — four predictors with no model in them, reported beside every cell. `constant_causal` is a flat line at the **fit-window mean**, the honest "no model" floor, using only what was knowable before the gate window opened; `constant_oracle` is a flat line at the **gate-window median**, a hindsight upper bound on what any constant could have achieved. `climatology_causal` and `climatology_oracle` are the same two forms taken **per hour of day** — the fit-window hourly mean and the gate-window hourly median — which is the tighter reference on every pair measured so far, because a constant is a climatology with one bucket. Read the pair together: the constant says whether the model predicts the *level*, the climatology says whether it predicts the level *and the daily shape*, and the gap between them is how much of this series is forced diurnal structure.

All four are **reported references and not gate criteria**: none is in the gate basis, none can move a cell's verdict, and a pair that clears its D-7 bar while losing to one still reads PASS. They are the number that qualifies the PASS — a challenger that does not beat `climatology_oracle` has not demonstrated skill beyond the average day, and a D-7 bar that `constant_causal` clears on its own was not a demanding bar. **Check each reference's own n before comparing it to the challenger.** A climatology is 24 levels, so an hour of day absent from its source window leaves those rows unscored for that column alone; scored on different rows, two WAPEs are not the same measurement. Nothing is interpolated to close that gap.

Graded disposition (ABL-418) — the registered bar is **not** re-opened. Seasonal-naive D-7 is still the gate, ABL-348's windows, bands, metric, minimum n and source are unchanged, and a cell that clears D-7 still reads PASS. What the grade adds is **what that PASS entitles the cell to**. ABL-406 measured across eight wind pairs that the gate outcome was fully predicted by whether a causal constant clears the bar on its own — five weak bars, five passes; three strong bars, three failures or ties — and that NO passed 3/3 while anti-correlated with its own target. A PASS is necessary and not sufficient.

**G1** gate: beats D-7 by more than the readability floor — ABL-385's `delta_min(k)` with `c_B = 0`, since every reference here is deterministic, which is **10.65%** for this stream at k=1. **G2** level: beats `constant_causal`. **G3** shape: beats `climatology_causal`. **G4** direction: slope > 0 and corr > 0. **A** = all four in every band (promotion-eligible, subject to any named data hold); **B** = G1 holds and one or more of G2/G3/G4 fails, named; **C** = a readable loss to D-7; **U** = the G1 margin sits inside the floor, so the cell is unreadable at one seed — **U(+)** where G2–G4 clear readably, meaning *re-read at k>1 seeds*, not *reject*.

Causal references only. The two oracle references stay reported and gate nothing: an oracle is not causally available, so losing to one bounds what a verdict means rather than voiding it. The bar-weakness flag — does `constant_causal` clear the registered D-7 bar on its own? — is reported for the same reason. Neither is on the ladder. A condition that could not be measured is not satisfied, and is named like any other failure.

| country | horizon | n | challenger WAPE | D-7 WAPE | skill vs D-7 | constant causal WAPE | constant oracle WAPE | climatology causal WAPE | climatology oracle WAPE | incumbent WAPE | MAE | bias | slope | corr | gate | grade |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| EE | 48-64h | 388 | 25.1% | 35.3% | +29.0% | 80.4% | 81.4% | 29.1% | 23.3% | Not measured | 62.7 MW | -1.5% | 0.8 | 0.9 | FAIL | A |
| FI | 48-64h | 453 | 24.0% | 38.0% | +36.8% | 82.5% | 77.3% | 45.1% | 22.6% | Not measured | 125.4 MW | -13.5% | 0.8 | 0.9 | FAIL | A |
| LT | 24-36h | 720 | 20.9% | 30.6% | +31.7% | 91.2% | 91.1% | 45.8% | 18.2% | Not measured | 113.8 MW | -14.3% | 0.8 | 1.0 | PASS | A |
| LT | 36-48h | 720 | 20.8% | 30.6% | +32.0% | 91.2% | 91.1% | 45.8% | 18.2% | Not measured | 113.2 MW | -14.1% | 0.8 | 1.0 | PASS | A |
| LT | 48-64h | 510 | 19.8% | 29.4% | +32.5% | 90.9% | 90.3% | 44.8% | 17.5% | Not measured | 123.6 MW | -12.2% | 0.8 | 1.0 | PASS | A |
| LV | 24-36h | 708 | 29.6% | 47.8% | +38.1% | 89.9% | 89.2% | 39.8% | 33.9% | Not measured | 86.4 MW | 6.3% | 0.9 | 0.9 | PASS | A |
| LV | 36-48h | 708 | 29.5% | 47.8% | +38.3% | 89.9% | 89.2% | 39.8% | 33.9% | Not measured | 86.0 MW | 6.3% | 0.9 | 0.9 | PASS | A |
| LV | 48-64h | 506 | 32.2% | 47.8% | +32.7% | 90.4% | 90.2% | 36.0% | 34.2% | Not measured | 98.8 MW | 13.4% | 0.9 | 0.9 | PASS | A |
| NL | 24-36h | 720 | 37.0% | 45.0% | +17.9% | 103.2% | 95.7% | 34.8% | 32.6% | Not measured | 24.6 MW | -17.2% | 0.6 | 0.9 | PASS | B — fails G3 |
| NL | 36-48h | 720 | 37.7% | 45.0% | +16.3% | 103.2% | 95.7% | 34.8% | 32.6% | Not measured | 25.0 MW | -16.7% | 0.6 | 0.9 | PASS | B — fails G3 |
| NL | 48-64h | 510 | 36.8% | 45.2% | +18.5% | 85.2% | 91.4% | 34.8% | 32.6% | Not measured | 32.3 MW | -13.5% | 0.6 | 0.8 | PASS | B — fails G3 |
| SE | 24-36h | 720 | 21.2% | 23.9% | +11.3% | 94.6% | 94.0% | 40.9% | 18.5% | Not measured | 138.0 MW | -2.1% | 0.9 | 0.9 | PASS | A |
| SE | 36-48h | 720 | 20.8% | 23.9% | +12.8% | 94.6% | 94.0% | 40.9% | 18.5% | Not measured | 135.7 MW | -2.5% | 0.9 | 1.0 | PASS | A |
| SE | 48-64h | 510 | 19.8% | 23.7% | +16.5% | 87.5% | 90.3% | 40.7% | 17.9% | Not measured | 166.3 MW | -2.9% | 0.9 | 0.9 | PASS | A |

### Graded disposition, per pair

| pair | bands | grade | failed conditions | bar weaker than a flat line? |
|---|---|:---:|---|:---:|
| EE | A | **A** | — | no |
| FI | A | **A** | — | no |
| LT | A / A / A | **A** | — | no |
| LV | A / A / A | **A** | — | no |
| NL | B / B / B | **B** | G3 (beats climatology_causal -- an hour-of-day mean over the fit window) | no |
| SE | A / A / A | **A** | — | no |

## Cells the registration declares NOT-EVALUABLE

Declared by `experiments/ABL348/config.json -> not_evaluable.pairs` **before any fit existed**, and excluded from the 14-cell bar above. ABL-348's rule: *"A pair listed here is reported NOT-EVALUABLE on the named bands. It is not a FAIL and must not be counted as one; a gate read that scores it has misread this registration."* These rows are therefore measured and shown, but carry no gate outcome and no grade, and are counted neither as passes nor as failures.

The cause is per pair and only one of the two is ours: EE's shortfall is an ABL-188 bit-identical zero run present in **both** source tables (`source_dependent: false`), so it would not be recovered by reverting the source; FI's is `energy_generation` holding fewer gate hours than `energy_renewable` (`source_dependent: **true**`), which is a cost of ABL-348's source change and a finding for whoever owns that decision rather than a fact about FI's model.

| country | horizon | n | registered min n | challenger WAPE | D-7 WAPE | skill vs D-7 | declared cause |
|---|---|---:|---:|---:|---:|---:|---|
| EE | 24-36h | 543 | 684 | 23.8% | 36.2% | +34.3% | ABL-188 excludes a 44.8h bit-identical zero run (2026-07-21 -> 2026-07-22), present identically in **both** source tables; not source-dependent |
| EE | 36-48h | 540 | 684 | 24.0% | 36.3% | +33.8% | ABL-188 excludes a 44.8h bit-identical zero run (2026-07-21 -> 2026-07-22), present identically in **both** source tables; not source-dependent |
| FI | 24-36h | 629 | 684 | 26.3% | 38.1% | +31.0% | `energy_generation` holds 663 of 720 gate hours against `energy_renewable`'s 717 (the ABL-322 s3.3 phenomenon); **source-dependent** |
| FI | 36-48h | 628 | 684 | 25.6% | 38.1% | +32.7% | `energy_generation` holds 663 of 720 gate hours against `energy_renewable`'s 717 (the ABL-322 s3.3 phenomenon); **source-dependent** |

Reference levels used, from the same ABL-188-filtered target series the gate actuals and the D-7/persistence baselines come from — no refit, no second read, no additional upstream fetch. The hourly levels behind the climatology columns are in `results.json` in full; `h` is how many of the 24 hours of the day that level set covers, and anything below 24 means those rows were dropped from that column's n:

| country | constant causal | constant oracle | climatology causal | climatology oracle |
|---|---:|---:|---:|---:|
| EE | 172.03 MW | 131.92 MW | 7.00–453.46 MW (24h) | 3.68–518.76 MW (24h) |
| FI | 227.15 MW | 351.16 MW | 2.69–609.39 MW (24h) | 0.25–1007.25 MW (24h) |
| LT | 297.96 MW | 336.51 MW | 0.00–833.72 MW (24h) | 0.00–1433.40 MW (24h) |
| LV | 228.41 MW | 184.00 MW | 1.34–588.23 MW (24h) | 0.00–687.00 MW (24h) |
| NL | 65.28 MW | 24.20 MW | -1.23–181.57 MW (24h) | -0.13–193.16 MW (24h) |
| SE | 400.05 MW | 267.26 MW | 0.88–1186.51 MW (24h) | 1.33–1913.53 MW (24h) |

## Per-country all-D+2 summary

Gate-basis values (actual, challenger, seasonal_naive) share one finite intersection; each comparator outside the basis is scored on its own intersection with it, and its n is given in `comparator_n` in the JSON. A comparator showing `Not measured` had no finite rows at all.

**Pooling caveat.** This is a reported aggregate over *all* primary bands and is not a gate read. For EE, FI it therefore pools the band(s) the registration declares NOT-EVALUABLE, so the row is not the pooled form of that country's gate cells and must not be quoted as one.

| country | n | challenger WAPE | D-7 WAPE | persistence WAPE | constant causal WAPE | constant oracle WAPE | climatology causal WAPE | climatology oracle WAPE | incumbent WAPE | TSO WAPE (revision-contaminated; n) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EE | 1,471 | 24.2% | 36.0% | 85.2% | 84.0% | 83.4% | 29.9% | 24.0% | Not measured | 18.2% (n=1,471) |
| FI | 1,710 | 25.4% | 38.1% | 78.2% | 81.4% | 77.4% | 45.7% | 22.9% | Not measured | 12.7% (n=1,710) |
| LT | 1,950 | 20.6% | 30.3% | 85.4% | 91.1% | 90.9% | 45.5% | 18.0% | Not measured | 18.2% (n=1,950) |
| LV | 1,922 | 30.3% | 47.8% | 98.4% | 90.1% | 89.5% | 38.8% | 34.0% | Not measured | 24.2% (n=1,922) |
| NL | 1,950 | 37.2% | 45.1% | 99.2% | 97.5% | 94.4% | 34.8% | 32.6% | Not measured | 2129.4% (n=1,950) |
| SE | 1,950 | 20.6% | 23.8% | 90.7% | 92.4% | 92.8% | 40.9% | 18.3% | Not measured | 7.6% (n=1,950) |

## Fit and missingness audit

Every training row was built with `RenewableFeatureBuilder.row(target, generated_at, observation_as_of=generated_at)`. Gate targets were never fitted.

| country | algorithm | retained / intended fit rows | unique fit targets | excluded missing | degraded lag-1d rows | artifact SHA-256 |
|---|---|---:|---:|---:|---:|---|
| EE | catboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `0eee3622676236b07c35b334e58d4464d7f803a909a05c01148acf73c987b6a2` |
| FI | catboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `c84e032edcb72a2d419851d7621b860c29440fa1ddfdf3d6a0514a9cb80c3bb3` |
| LT | catboost | 34,144 / 34,176 | 4,269 | 32 | 23,650 | `c447bd08d6793f8a0ab8a4ee737a63df4f7ccbe053a1d681f66474b31e76d9da` |
| LV | catboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `06555acb2a16192bb5268bc9cce422927d446c00f2e95342d9ef26a1c0272299` |
| NL | catboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `38d537f10586a1194572761e627961abad5b9eded28533fec33f3f37457ef36b` |
| SE | catboost | 34,112 / 34,176 | 4,266 | 64 | 23,642 | `88ce1d46ba756e260a959e65b3af5098827a43f764011898db6961d017065e45` |

### Physically impossible night rows (ABL-376)

Not registered for scope `abl316-t2d`. The fit saw every night row, including any whose actual the sun says is impossible.

## Data quality and limits

- ABL-188 screening found suspect solar runs for EE in `energy_generation`: `[{'start': '2026-07-21 00:00:00', 'end': '2026-07-22 20:45:00', 'value': 0.0, 'n_rows': 180, 'duration_hours': 44.75}]`. The builder nulls these before fit; see the training audit and recommendation.
- ABL-67 is net-position-only; ABL-109/111 are load-only. ABL-71's known wrong-write modes are load and net position, not solar; this is a provenance caveat, not proof that solar ingest is pristine.
- Weather rows were admitted only where `forecast_run_time <= generated_at`; missing vintages were excluded, never filled with future reanalysis.
- TSO values come from an `INSERT OR REPLACE` table without first-seen vintages. They may include revisions and cannot support promotion.
- This is one 30-day summer holdout. It is out-of-sample by target timestamp, but not year-round evidence.
- **The challenger loses to a climatology chosen with hindsight in 11 cell(s).** An hour-of-day median — the average day, with no model and no weather in it — scores better than the fitted model there. This is the weaker claim to lose and the stronger one to win: a challenger that beats it is doing something no table of hourly averages can do. This does not change any verdict above; it bounds what the verdict means:
  - EE solar 48-64h: challenger 25.05% vs oracle climatology 23.29% (+1.76pp)
  - FI solar 48-64h: challenger 23.99% vs oracle climatology 22.56% (+1.43pp)
  - LT solar 24-36h: challenger 20.94% vs oracle climatology 18.15% (+2.79pp)
  - LT solar 36-48h: challenger 20.83% vs oracle climatology 18.15% (+2.68pp)
  - LT solar 48-64h: challenger 19.82% vs oracle climatology 17.47% (+2.35pp)
  - NL solar 24-36h: challenger 36.97% vs oracle climatology 32.57% (+4.40pp)
  - NL solar 36-48h: challenger 37.66% vs oracle climatology 32.57% (+5.09pp)
  - NL solar 48-64h: challenger 36.82% vs oracle climatology 32.62% (+4.20pp)
  - SE solar 24-36h: challenger 21.16% vs oracle climatology 18.49% (+2.67pp)
  - SE solar 36-48h: challenger 20.81% vs oracle climatology 18.49% (+2.32pp)
  - SE solar 48-64h: challenger 19.75% vs oracle climatology 17.89% (+1.86pp)

## Recommendation to the CEO

Do not promote these artifacts: only 12/14 primary cells clear the registered bar. Report the losing country/bands as the finding and pursue country-specific diagnosis/model work on a fresh pre-registered split.

No production deploy, serving-registry change, model promotion, ingest change, dashboard change, replica write, or sidecar write was performed.
