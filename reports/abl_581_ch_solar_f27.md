# ABL-581 — Serve-faithful solar retrain gate, CH solar alone at 27 features on energy_generation: the ABL-525 withdrawal's route back

**Disposition: PASS**

Generated: 2026-08-27 15:26 UTC
Fit targets: 2026-01-14 00:00:00 → 2026-07-11 00:00:00 (exclusive).
Out-of-sample gate targets: 2026-07-11 00:00:00 → 2026-08-10 00:00:00 (exclusive).
Baseline: literal seasonal-naive D-7. TSO is revision-contaminated context only and is not a gate criterion.
Replica: `C:\Code\able\data\energy_dashboard.db` (10,632,605,696 bytes), opened with SQLite `mode=ro`, `uri=True`.
That one file is the source of the TSO series, the contamination screen, and — since ABL-355 — the fitted target series, its lag/rolling features, the D-7 and persistence baselines, the gate actuals and the weather archive. The incumbent forecasts are the only read it does not hold alone; see the sidecar below.
Sidecar: `C:\Code\able\data\forecasts_local.db`, also opened `mode=ro`, and read for locally generated incumbent forecasts only. Where a sidecar row and a replica row carry the same vintage, the sidecar's is the one scored.
Target series, features, baselines and contamination screen: `energy_generation`.
Feature set: **legacy25+geometry** (27 columns), the module default -- this scope registers no feature set of its own.

## Gate read

Registered scope `abl581-ch-solar-f27`: CH.
Gate basis — the columns that must be simultaneously finite for a row to be scored: `challenger`, `seasonal_naive`. Comparators outside the basis are scored on their own intersection with it and carry their own n, so a comparator that does not exist for a country reads Not measured instead of emptying the cell.
Strict full PASS requires challenger WAPE < D-7 in all 3 country × primary D+2-band cells and ≥95% of intended pairs. Result: **3/3 cells pass**.

Model-free reference (ABL-389) — four predictors with no model in them, reported beside every cell. `constant_causal` is a flat line at the **fit-window mean**, the honest "no model" floor, using only what was knowable before the gate window opened; `constant_oracle` is a flat line at the **gate-window median**, a hindsight upper bound on what any constant could have achieved. `climatology_causal` and `climatology_oracle` are the same two forms taken **per hour of day** — the fit-window hourly mean and the gate-window hourly median — which is the tighter reference on every pair measured so far, because a constant is a climatology with one bucket. Read the pair together: the constant says whether the model predicts the *level*, the climatology says whether it predicts the level *and the daily shape*, and the gap between them is how much of this series is forced diurnal structure.

All four are **reported references and not gate criteria**: none is in the gate basis, none can move a cell's verdict, and a pair that clears its D-7 bar while losing to one still reads PASS. They are the number that qualifies the PASS — a challenger that does not beat `climatology_oracle` has not demonstrated skill beyond the average day, and a D-7 bar that `constant_causal` clears on its own was not a demanding bar. **Check each reference's own n before comparing it to the challenger.** A climatology is 24 levels, so an hour of day absent from its source window leaves those rows unscored for that column alone; scored on different rows, two WAPEs are not the same measurement. Nothing is interpolated to close that gap.

**Trailing-window causal reference (ABL-437).** `constant_causal_28d` and `climatology_causal_28d` are the same flat line and the same hour-of-day mean, levelled over the **28 days ending at each row's own `generated_at`** instead of over the whole fit window. They exist because the fit window here runs winter to summer and the gate window is high summer, so on a seasonal series the fit-window mean is not an estimate of the gate window's level: a causal constant reads up to **205% worse than the correctly-levelled oracle constant** (NL `wind_onshore`, 225.54% against 73.85%), which inflates a G2/G3 pass for free. The trailing form is strictly causal by construction — same anchor, same inclusive hour-floored bound and same filtered series as the challenger's own `target_value_roll_168h_mean` feature — so it uses no information the challenger did not have. The `level inflation` column prints the residual per cell. **Which pair the grade ladder reads is registered per scope**; the fit-window pair keeps its name, its definition and every value already published.

Graded disposition (ABL-418) — the registered bar is **not** re-opened. Seasonal-naive D-7 is still the gate, ABL-348's windows, bands, metric, minimum n and source are unchanged, and a cell that clears D-7 still reads PASS. What the grade adds is **what that PASS entitles the cell to**. ABL-406 measured across eight wind pairs that the gate outcome was fully predicted by whether a causal constant clears the bar on its own — five weak bars, five passes; three strong bars, three failures or ties — and that NO passed 3/3 while anti-correlated with its own target. A PASS is necessary and not sufficient.

**G1** gate: beats D-7 by more than the readability floor — ABL-385's `delta_min(k)` with `c_B = 0`, since every reference here is deterministic, which is **10.65%** for this stream at k=1. **G2** level: beats `constant_causal_28d`. **G3** shape: beats `climatology_causal_28d`. **G4** direction: slope > 0 and corr > 0. **A** = all four in every band (promotion-eligible, subject to any named data hold); **B** = G1 holds and one or more of G2/G3/G4 fails, named; **C** = a readable loss to D-7; **U** = the G1 margin sits inside the floor, so the cell is unreadable at one seed — **U(+)** where G2–G4 clear readably, meaning *re-read at k>1 seeds*, not *reject*.

**G0** readable (ABL-434): the cell meets ABL-348's registered minimum n, assessed **before** any of G1–G4. A cell that does not grades **`X`** — not readable at the registered coverage, so nothing on the ladder below it is decidable, and not promotion-eligible. This is not a new bar: `enough_pairs` already decides the gate column, and what changes is only that the grade may no longer disagree with it. It is one-way — a coverage shortfall can only remove eligibility — and coverage that is not recorded is not coverage that holds. `X` ranks below `B` and `C`: a band that had the rows and lost readably has something definite to say.

Causal references only. The two oracle references stay reported and gate nothing: an oracle is not causally available, so losing to one bounds what a verdict means rather than voiding it. The bar-weakness flag — does `constant_causal` clear the registered D-7 bar on its own? — is reported for the same reason. Neither is on the ladder. A condition that could not be measured is not satisfied, and is named like any other failure.

**Causal levelling (ABL-437): `trailing_28d`.** G2 and G3 read `constant_causal_28d` and `climatology_causal_28d` — the flat line and the hour-of-day mean over the **28 days ending at each row's own forecast issue instant**, not over the whole fit window. The fit-window forms are levelled on 2026-01-14 → 2026-07-11 and scored on high summer, which on a seasonal series makes them a strawman: measured across every committed tranche record, a causal constant runs up to **205% worse** than the correctly-levelled oracle constant (NL `wind_onshore`), which passes G2/G3 for free. The trailing form is strictly causal — same anchor and same filtered series as the challenger's own `target_value_roll_168h_mean` feature. Both fit-window references stay **reported** beside it, and the `level inflation` column prints the residual, so nothing is discarded. **G1 is unchanged**: the registered D-7 bar is not re-opened, and no oracle is on the ladder.

**G2/G3 readability (ABL-444): `floored`.** G2 and G3 are decided against the same **10.65%** floor G1 carries, not by a bare sign test. A margin inside it is **`N` — not readable**: neither demonstrated nor refuted at k=1, an abstention rather than a failure, and not promotion-eligible, because a condition that could not be measured is not satisfied. `B` still outranks `N`, so a cell that also fails something readably reads `B`. **The margin is printed either way** — the floor decides gradeability, it does not replace the number. ABL-418 already applied this floor to G2/G3 when deciding `U` against `U(+)`; the amendment carries it to the `A`/`B` branch, where a letter could turn on 0.36% (PL solar's G3).

**Readability test (ABL-467): `delta_min`.** Every condition is decided against ABL-385's imported `delta_min` floor, **10.65%** for this stream at k=1, which is the only tool available: a single fit carries no internal estimate of its own spread.

| country | horizon | n | challenger WAPE | D-7 WAPE | skill vs D-7 | constant causal WAPE | constant causal 28d WAPE | constant oracle WAPE | climatology causal WAPE | climatology causal 28d WAPE | climatology oracle WAPE | level inflation (causal / 28d) | incumbent WAPE | MAE | bias | slope | corr | gate | grade |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| CH | 24-36h | 720 | 7.8% | 12.7% | +38.6% | 95.1% | 100.0% | 94.6% | 37.5% | 10.2% | 9.0% | 0.5% / 5.7% | Not measured | 103.6 MW | -1.5% | 1.0 | 1.0 | PASS | A |
| CH | 36-48h | 720 | 7.8% | 12.7% | +38.8% | 95.1% | 100.1% | 94.6% | 37.5% | 10.2% | 9.0% | 0.5% / 5.7% | Not measured | 103.2 MW | -1.6% | 1.0 | 1.0 | PASS | A |
| CH | 48-64h | 510 | 8.3% | 12.5% | +33.6% | 86.0% | 81.0% | 87.9% | 36.6% | 9.8% | 8.7% | -2.2% / -7.9% | Not measured | 145.7 MW | -2.3% | 1.0 | 1.0 | PASS | A |

### Graded disposition, per pair

| pair | bands | grade | failed conditions | not readable | bar weaker than a flat line? |
|---|---|:---:|---|---|:---:|
| CH | A / A / A | **A** | — | — | no |

Reference levels used, from the same ABL-188-filtered target series the gate actuals and the D-7/persistence baselines come from — no refit, no second read, no additional upstream fetch. The hourly levels behind the climatology columns are in `results.json` in full; `h` is how many of the 24 hours of the day that level set covers, and anything below 24 means those rows were dropped from that column's n:

| country | constant causal | constant oracle | climatology causal | climatology oracle | constant causal 28d |
|---|---:|---:|---:|---:|---:|
| CH | 833.37 MW | 677.22 MW | 0.73–2586.18 MW (24h) | 0.00–3695.79 MW (24h) | 1332.98–1459.22 MW (124 as-of) |

## Per-country all-D+2 summary

Gate-basis values (actual, challenger, seasonal_naive) share one finite intersection; each comparator outside the basis is scored on its own intersection with it, and its n is given in `comparator_n` in the JSON. A comparator showing `Not measured` had no finite rows at all.

| country | n | challenger WAPE | D-7 WAPE | persistence WAPE | constant causal WAPE | constant causal 28d WAPE | constant oracle WAPE | climatology causal WAPE | climatology causal 28d WAPE | climatology oracle WAPE | incumbent WAPE | TSO WAPE (revision-contaminated; n) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CH | 1,950 | 7.9% | 12.6% | 87.5% | 92.2% | 94.0% | 92.5% | 37.2% | 10.1% | 8.9% | Not measured | 7.1% (n=1,950) |

## Fit and missingness audit

Every training row was built with `RenewableFeatureBuilder.row(target, generated_at, observation_as_of=generated_at)`. Gate targets were never fitted.

| country | algorithm | retained / intended fit rows | unique fit targets | excluded missing | degraded lag-1d rows | artifact SHA-256 |
|---|---|---:|---:|---:|---:|---|
| CH | catboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `a8545851ea7e385bcad1f82a96631ecec06bb6b7d163ab66a054f0ea5b316aee` |

### Physically impossible night rows (ABL-376)

Not registered for scope `abl581-ch-solar-f27`. The fit saw every night row, including any whose actual the sun says is impossible.

## Data quality and limits

- ABL-188 constant-run screening found no ≥24-hour bit-identical solar run in `energy_generation` over the registered fit/scoring interval plus 14-day feature lookback (2025-12-31 → 2026-08-10 UTC). The builder still routes solar through `exclude_suspect_constant_runs`; the invariant was verified on the actual window, not assumed from ABL-191.
- ABL-67 is net-position-only; ABL-109/111 are load-only. ABL-71's known wrong-write modes are load and net position, not solar; this is a provenance caveat, not proof that solar ingest is pristine.
- Weather rows were admitted only where `forecast_run_time <= generated_at`; missing vintages were excluded, never filled with future reanalysis.
- TSO values come from an `INSERT OR REPLACE` table without first-seen vintages. They may include revisions and cannot support promotion.
- This is one 30-day summer holdout. It is out-of-sample by target timestamp, but not year-round evidence.

## Recommendation to the CEO

The challenger clears the pre-registered D-7 bar in every served solar D+2 country-band cell. Preserve these experiment artifacts and ask the CEO to initiate Board review; do not promote from this issue.

No production deploy, serving-registry change, model promotion, ingest change, dashboard change, replica write, or sidecar write was performed.
