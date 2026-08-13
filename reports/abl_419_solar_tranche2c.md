# ABL-419 — Serve-faithful solar retrain gate, ABL-316 tranche 2c: 5 Mediterranean countries on energy_generation at 27 features

**Disposition: FAIL**

Generated: 2026-08-13 23:03 UTC
Fit targets: 2026-01-14 00:00:00 → 2026-07-11 00:00:00 (exclusive).
Out-of-sample gate targets: 2026-07-11 00:00:00 → 2026-08-10 00:00:00 (exclusive).
Baseline: literal seasonal-naive D-7. TSO is revision-contaminated context only and is not a gate criterion.
Replica: `C:\Code\able\data\energy_dashboard.db` (9,432,453,120 bytes), opened with SQLite `mode=ro`, `uri=True`.
That one file is the source of the TSO series, the contamination screen, and — since ABL-355 — the fitted target series, its lag/rolling features, the D-7 and persistence baselines, the gate actuals and the weather archive. The incumbent forecasts are the only read it does not hold alone; see the sidecar below.
Sidecar: `C:\Code\able\data\forecasts_local.db`, also opened `mode=ro`, and read for locally generated incumbent forecasts only. Where a sidecar row and a replica row carry the same vintage, the sidecar's is the one scored.
Target series, features, baselines and contamination screen: `energy_generation`.
Feature set: **legacy25+geometry** (27 columns), the module default -- this scope registers no feature set of its own.

## Gate read

Registered scope `abl316-t2c`: ES, GR, HR, IT, PT.
Gate basis — the columns that must be simultaneously finite for a row to be scored: `challenger`, `seasonal_naive`. Comparators outside the basis are scored on their own intersection with it and carry their own n, so a comparator that does not exist for a country reads Not measured instead of emptying the cell.
Strict full PASS requires challenger WAPE < D-7 in all 15 country × primary D+2-band cells and ≥95% of intended pairs. Result: **9/15 cells pass**.

Model-free reference (ABL-389) — four predictors with no model in them, reported beside every cell. `constant_causal` is a flat line at the **fit-window mean**, the honest "no model" floor, using only what was knowable before the gate window opened; `constant_oracle` is a flat line at the **gate-window median**, a hindsight upper bound on what any constant could have achieved. `climatology_causal` and `climatology_oracle` are the same two forms taken **per hour of day** — the fit-window hourly mean and the gate-window hourly median — which is the tighter reference on every pair measured so far, because a constant is a climatology with one bucket. Read the pair together: the constant says whether the model predicts the *level*, the climatology says whether it predicts the level *and the daily shape*, and the gap between them is how much of this series is forced diurnal structure.

All four are **reported references and not gate criteria**: none is in the gate basis, none can move a cell's verdict, and a pair that clears its D-7 bar while losing to one still reads PASS. They are the number that qualifies the PASS — a challenger that does not beat `climatology_oracle` has not demonstrated skill beyond the average day, and a D-7 bar that `constant_causal` clears on its own was not a demanding bar. **Check each reference's own n before comparing it to the challenger.** A climatology is 24 levels, so an hour of day absent from its source window leaves those rows unscored for that column alone; scored on different rows, two WAPEs are not the same measurement. Nothing is interpolated to close that gap.

Graded disposition (ABL-418) — the registered bar is **not** re-opened. Seasonal-naive D-7 is still the gate, ABL-348's windows, bands, metric, minimum n and source are unchanged, and a cell that clears D-7 still reads PASS. What the grade adds is **what that PASS entitles the cell to**. ABL-406 measured across eight wind pairs that the gate outcome was fully predicted by whether a causal constant clears the bar on its own — five weak bars, five passes; three strong bars, three failures or ties — and that NO passed 3/3 while anti-correlated with its own target. A PASS is necessary and not sufficient.

**G1** gate: beats D-7 by more than the readability floor — ABL-385's `delta_min(k)` with `c_B = 0`, since every reference here is deterministic, which is **10.65%** for this stream at k=1. **G2** level: beats `constant_causal`. **G3** shape: beats `climatology_causal`. **G4** direction: slope > 0 and corr > 0. **A** = all four in every band (promotion-eligible, subject to any named data hold); **B** = G1 holds and one or more of G2/G3/G4 fails, named; **C** = a readable loss to D-7; **U** = the G1 margin sits inside the floor, so the cell is unreadable at one seed — **U(+)** where G2–G4 clear readably, meaning *re-read at k>1 seeds*, not *reject*.

Causal references only. The two oracle references stay reported and gate nothing: an oracle is not causally available, so losing to one bounds what a verdict means rather than voiding it. The bar-weakness flag — does `constant_causal` clear the registered D-7 bar on its own? — is reported for the same reason. Neither is on the ladder. A condition that could not be measured is not satisfied, and is named like any other failure.

| country | horizon | n | challenger WAPE | D-7 WAPE | skill vs D-7 | constant causal WAPE | constant oracle WAPE | climatology causal WAPE | climatology oracle WAPE | incumbent WAPE | MAE | bias | slope | corr | gate | grade |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| ES | 24-36h | 720 | 11.4% | 11.7% | +2.5% | 89.8% | 89.8% | 35.4% | 8.7% | Not measured | 1272.7 MW | -6.1% | 0.9 | 1.0 | PASS | U(+) |
| ES | 36-48h | 720 | 11.4% | 11.7% | +2.3% | 89.8% | 89.8% | 35.4% | 8.7% | Not measured | 1275.4 MW | -6.0% | 0.9 | 1.0 | PASS | U(+) |
| ES | 48-64h | 510 | 11.0% | 11.1% | +1.0% | 77.8% | 77.1% | 35.2% | 8.4% | Not measured | 1664.1 MW | -6.7% | 0.9 | 1.0 | PASS | U(+) |
| GR | 24-36h | 720 | 20.8% | 10.2% | -104.3% | 98.3% | 97.1% | 46.3% | 8.6% | Not measured | 555.5 MW | -18.2% | 0.8 | 1.0 | FAIL | C — fails G1 |
| GR | 36-48h | 720 | 20.8% | 10.2% | -104.4% | 98.3% | 97.1% | 46.3% | 8.6% | Not measured | 555.8 MW | -18.1% | 0.8 | 1.0 | FAIL | C — fails G1 |
| GR | 48-64h | 510 | 20.6% | 10.3% | -99.1% | 93.3% | 95.1% | 46.5% | 8.9% | Not measured | 682.6 MW | -18.1% | 0.8 | 1.0 | FAIL | C — fails G1 |
| HR | 24-36h | 720 | 15.0% | 16.2% | +7.8% | 96.0% | 96.0% | 39.3% | 9.3% | Not measured | 23.7 MW | -10.1% | 0.9 | 1.0 | PASS | U(+) |
| HR | 36-48h | 720 | 15.1% | 16.2% | +7.2% | 96.0% | 96.0% | 39.3% | 9.3% | Not measured | 23.9 MW | -10.2% | 0.9 | 1.0 | PASS | U(+) |
| HR | 48-64h | 510 | 14.8% | 16.2% | +8.6% | 89.0% | 91.5% | 38.8% | 9.3% | Not measured | 29.6 MW | -10.5% | 0.9 | 1.0 | PASS | U(+) |
| IT | 24-36h | 720 | 6.6% | 7.0% | +4.8% | 98.9% | 97.2% | 29.6% | 4.3% | Not measured | 426.9 MW | -2.9% | 1.0 | 1.0 | PASS | U(+) |
| IT | 36-48h | 720 | 6.7% | 7.0% | +3.6% | 98.9% | 97.2% | 29.6% | 4.3% | Not measured | 432.2 MW | -2.9% | 1.0 | 1.0 | PASS | U(+) |
| IT | 48-64h | 510 | 6.0% | 6.6% | +8.1% | 88.2% | 93.0% | 28.8% | 4.1% | Not measured | 509.2 MW | -3.0% | 1.0 | 1.0 | PASS | U(+) |
| PT | 24-36h | 720 | 14.5% | 13.1% | -10.7% | 97.2% | 96.4% | 36.5% | 13.6% | Not measured | 172.7 MW | -3.8% | 0.9 | 1.0 | FAIL | C — fails G1 |
| PT | 36-48h | 720 | 14.9% | 13.1% | -13.5% | 97.2% | 96.4% | 36.5% | 13.6% | Not measured | 177.1 MW | -3.9% | 0.9 | 1.0 | FAIL | C — fails G1 |
| PT | 48-64h | 510 | 15.0% | 13.0% | -15.6% | 81.0% | 87.1% | 36.3% | 13.6% | Not measured | 248.6 MW | -4.3% | 0.9 | 1.0 | FAIL | C — fails G1 |

### Graded disposition, per pair

| pair | bands | grade | failed conditions | bar weaker than a flat line? |
|---|---|:---:|---|:---:|
| ES | U(+) / U(+) / U(+) | **U(+)** | — | no |
| GR | C / C / C | **C** | G1 (readable loss to seasonal_naive D-7) | no |
| HR | U(+) / U(+) / U(+) | **U(+)** | — | no |
| IT | U(+) / U(+) / U(+) | **U(+)** | — | no |
| PT | C / C / C | **C** | G1 (readable loss to seasonal_naive D-7) | no |

Reference levels used, from the same ABL-188-filtered target series the gate actuals and the D-7/persistence baselines come from — no refit, no second read, no additional upstream fetch. The hourly levels behind the climatology columns are in `results.json` in full; `h` is how many of the 24 hours of the day that level set covers, and anything below 24 means those rows were dropped from that column's n:

| country | constant causal | constant oracle | climatology causal | climatology oracle |
|---|---:|---:|---:|---:|
| ES | 7222.73 MW | 7565.00 MW | 161.62–17190.76 MW (24h) | 287.00–25662.00 MW (24h) |
| GR | 1439.96 MW | 830.88 MW | 0.00–4122.90 MW (24h) | 0.00–7591.88 MW (24h) |
| HR | 97.72 MW | 68.65 MW | 0.00–268.06 MW (24h) | 0.00–435.70 MW (24h) |
| IT | 4542.37 MW | 2237.38 MW | 0.00–13935.51 MW (24h) | 0.00–18420.50 MW (24h) |
| PT | 770.03 MW | 491.15 MW | 8.88–2119.97 MW (24h) | 0.00–3272.10 MW (24h) |

## Per-country all-D+2 summary

Gate-basis values (actual, challenger, seasonal_naive) share one finite intersection; each comparator outside the basis is scored on its own intersection with it, and its n is given in `comparator_n` in the JSON. A comparator showing `Not measured` had no finite rows at all.

| country | n | challenger WAPE | D-7 WAPE | persistence WAPE | constant causal WAPE | constant oracle WAPE | climatology causal WAPE | climatology oracle WAPE | incumbent WAPE | TSO WAPE (revision-contaminated; n) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ES | 1,950 | 11.3% | 11.5% | 78.6% | 85.9% | 85.6% | 35.4% | 8.6% | Not measured | 15.2% (n=1,950) |
| GR | 1,950 | 20.7% | 10.2% | 83.5% | 96.8% | 96.5% | 46.3% | 8.7% | Not measured | 15.5% (n=1,950) |
| HR | 1,950 | 14.9% | 16.2% | 86.2% | 93.8% | 94.6% | 39.1% | 9.3% | Not measured | 9.8% (n=1,950) |
| IT | 1,950 | 6.5% | 6.8% | 87.5% | 95.5% | 95.8% | 29.3% | 4.3% | Not measured | 10.3% (n=1,950) |
| PT | 1,950 | 14.8% | 13.1% | 92.3% | 91.8% | 93.3% | 36.4% | 13.6% | Not measured | 10.0% (n=1,950) |

## Fit and missingness audit

Every training row was built with `RenewableFeatureBuilder.row(target, generated_at, observation_as_of=generated_at)`. Gate targets were never fitted.

| country | algorithm | retained / intended fit rows | unique fit targets | excluded missing | degraded lag-1d rows | artifact SHA-256 |
|---|---|---:|---:|---:|---:|---|
| ES | catboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `0f7cc49b9347dcd3ca7b526e2dce5418631c22b0d3e55f3d04895027334ce094` |
| GR | catboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `e9d5a4be5972586cd60940c17abd05c3639959a5859d468c718582e9145dd0ec` |
| HR | catboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `9dd4ea73407533060a84ab00d4afc9956b644f1055f244b67ff1d50ed521040c` |
| IT | catboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `10c58b10f724fa92996e4c79392249bdc52f17f95d2f4b1d8e642804f4fd39b9` |
| PT | catboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `392cdaf769ae87a63970dee0a16cbd6dd2bdf0d70cb578a64f595928bd8937ce` |

### Physically impossible night rows (ABL-376)

Not registered for scope `abl316-t2c`. The fit saw every night row, including any whose actual the sun says is impossible.

## Data quality and limits

- ABL-188 constant-run screening found no ≥24-hour bit-identical solar run in `energy_generation` over the registered fit/scoring interval plus 14-day feature lookback (2025-12-31 → 2026-08-10 UTC). The builder still routes solar through `exclude_suspect_constant_runs`; the invariant was verified on the actual window, not assumed from ABL-191.
- ABL-67 is net-position-only; ABL-109/111 are load-only. ABL-71's known wrong-write modes are load and net position, not solar; this is a provenance caveat, not proof that solar ingest is pristine.
- Weather rows were admitted only where `forecast_run_time <= generated_at`; missing vintages were excluded, never filled with future reanalysis.
- TSO values come from an `INSERT OR REPLACE` table without first-seen vintages. They may include revisions and cannot support promotion.
- This is one 30-day summer holdout. It is out-of-sample by target timestamp, but not year-round evidence.
- **The challenger loses to a climatology chosen with hindsight in 15 cell(s).** An hour-of-day median — the average day, with no model and no weather in it — scores better than the fitted model there. This is the weaker claim to lose and the stronger one to win: a challenger that beats it is doing something no table of hourly averages can do. This does not change any verdict above; it bounds what the verdict means:
  - ES solar 24-36h: challenger 11.39% vs oracle climatology 8.74% (+2.65pp)
  - ES solar 36-48h: challenger 11.41% vs oracle climatology 8.74% (+2.68pp)
  - ES solar 48-64h: challenger 11.03% vs oracle climatology 8.43% (+2.60pp)
  - GR solar 24-36h: challenger 20.78% vs oracle climatology 8.60% (+12.18pp)
  - GR solar 36-48h: challenger 20.79% vs oracle climatology 8.60% (+12.19pp)
  - GR solar 48-64h: challenger 20.58% vs oracle climatology 8.94% (+11.64pp)
  - HR solar 24-36h: challenger 14.95% vs oracle climatology 9.31% (+5.64pp)
  - HR solar 36-48h: challenger 15.06% vs oracle climatology 9.31% (+5.75pp)
  - HR solar 48-64h: challenger 14.79% vs oracle climatology 9.33% (+5.47pp)
  - IT solar 24-36h: challenger 6.63% vs oracle climatology 4.34% (+2.28pp)
  - IT solar 36-48h: challenger 6.71% vs oracle climatology 4.34% (+2.36pp)
  - IT solar 48-64h: challenger 6.05% vs oracle climatology 4.06% (+1.99pp)
  - PT solar 24-36h: challenger 14.49% vs oracle climatology 13.63% (+0.87pp)
  - PT solar 36-48h: challenger 14.86% vs oracle climatology 13.63% (+1.23pp)
  - PT solar 48-64h: challenger 15.00% vs oracle climatology 13.61% (+1.39pp)

## Recommendation to the CEO

Do not promote these artifacts: only 9/15 primary cells clear the registered bar. Report the losing country/bands as the finding and pursue country-specific diagnosis/model work on a fresh pre-registered split.

No production deploy, serving-registry change, model promotion, ingest change, dashboard change, replica write, or sidecar write was performed.
