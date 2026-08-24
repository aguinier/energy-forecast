# ABL-426 — Serve-faithful solar retrain gate, ABL-316 tranche 2a re-read on the registered energy_generation: 8 continental countries at 27 features

**Disposition: FAIL**

Generated: 2026-08-23 07:36 UTC
Fit targets: 2026-01-14 00:00:00 → 2026-07-11 00:00:00 (exclusive).
Out-of-sample gate targets: 2026-07-11 00:00:00 → 2026-08-10 00:00:00 (exclusive).
Baseline: literal seasonal-naive D-7. TSO is revision-contaminated context only and is not a gate criterion.
Replica: `C:\Code\able\data\energy_dashboard.db` (10,266,849,280 bytes), opened with SQLite `mode=ro`, `uri=True`.
That one file is the source of the TSO series, the contamination screen, and — since ABL-355 — the fitted target series, its lag/rolling features, the D-7 and persistence baselines, the gate actuals and the weather archive. The incumbent forecasts are the only read it does not hold alone; see the sidecar below.
Sidecar: `C:\Code\able\data\forecasts_local.db`, also opened `mode=ro`, and read for locally generated incumbent forecasts only. Where a sidecar row and a replica row carry the same vintage, the sidecar's is the one scored.
Target series, features, baselines and contamination screen: `energy_generation`.
Feature set: **legacy25+geometry** (27 columns), the module default -- this scope registers no feature set of its own.

## Gate read

Registered scope `abl316-t2a-generation`: BG, CH, CZ, HU, PL, RO, SI, SK.
Gate basis — the columns that must be simultaneously finite for a row to be scored: `challenger`, `seasonal_naive`. Comparators outside the basis are scored on their own intersection with it and carry their own n, so a comparator that does not exist for a country reads Not measured instead of emptying the cell.
Strict full PASS requires challenger WAPE < D-7 in all 24 country × primary D+2-band cells and ≥95% of intended pairs. Result: **22/24 cells pass**.

Model-free reference (ABL-389) — four predictors with no model in them, reported beside every cell. `constant_causal` is a flat line at the **fit-window mean**, the honest "no model" floor, using only what was knowable before the gate window opened; `constant_oracle` is a flat line at the **gate-window median**, a hindsight upper bound on what any constant could have achieved. `climatology_causal` and `climatology_oracle` are the same two forms taken **per hour of day** — the fit-window hourly mean and the gate-window hourly median — which is the tighter reference on every pair measured so far, because a constant is a climatology with one bucket. Read the pair together: the constant says whether the model predicts the *level*, the climatology says whether it predicts the level *and the daily shape*, and the gap between them is how much of this series is forced diurnal structure.

All four are **reported references and not gate criteria**: none is in the gate basis, none can move a cell's verdict, and a pair that clears its D-7 bar while losing to one still reads PASS. They are the number that qualifies the PASS — a challenger that does not beat `climatology_oracle` has not demonstrated skill beyond the average day, and a D-7 bar that `constant_causal` clears on its own was not a demanding bar. **Check each reference's own n before comparing it to the challenger.** A climatology is 24 levels, so an hour of day absent from its source window leaves those rows unscored for that column alone; scored on different rows, two WAPEs are not the same measurement. Nothing is interpolated to close that gap.

**Trailing-window causal reference (ABL-437).** `constant_causal_28d` and `climatology_causal_28d` are the same flat line and the same hour-of-day mean, levelled over the **28 days ending at each row's own `generated_at`** instead of over the whole fit window. They exist because the fit window here runs winter to summer and the gate window is high summer, so on a seasonal series the fit-window mean is not an estimate of the gate window's level: a causal constant reads up to **205% worse than the correctly-levelled oracle constant** (NL `wind_onshore`, 225.54% against 73.85%), which inflates a G2/G3 pass for free. The trailing form is strictly causal by construction — same anchor, same inclusive hour-floored bound and same filtered series as the challenger's own `target_value_roll_168h_mean` feature — so it uses no information the challenger did not have. The `level inflation` column prints the residual per cell. **Which pair the grade ladder reads is registered per scope**; the fit-window pair keeps its name, its definition and every value already published.

Graded disposition (ABL-418) — the registered bar is **not** re-opened. Seasonal-naive D-7 is still the gate, ABL-348's windows, bands, metric, minimum n and source are unchanged, and a cell that clears D-7 still reads PASS. What the grade adds is **what that PASS entitles the cell to**. ABL-406 measured across eight wind pairs that the gate outcome was fully predicted by whether a causal constant clears the bar on its own — five weak bars, five passes; three strong bars, three failures or ties — and that NO passed 3/3 while anti-correlated with its own target. A PASS is necessary and not sufficient.

**G1** gate: beats D-7 by more than the readability floor — ABL-385's `delta_min(k)` with `c_B = 0`, since every reference here is deterministic, which is **10.65%** for this stream at k=1. **G2** level: beats `constant_causal`. **G3** shape: beats `climatology_causal`. **G4** direction: slope > 0 and corr > 0. **A** = all four in every band (promotion-eligible, subject to any named data hold); **B** = G1 holds and one or more of G2/G3/G4 fails, named; **C** = a readable loss to D-7; **U** = the G1 margin sits inside the floor, so the cell is unreadable at one seed — **U(+)** where G2–G4 clear readably, meaning *re-read at k>1 seeds*, not *reject*.

**G0** readable (ABL-434): the cell meets ABL-348's registered minimum n, assessed **before** any of G1–G4. A cell that does not grades **`X`** — not readable at the registered coverage, so nothing on the ladder below it is decidable, and not promotion-eligible. This is not a new bar: `enough_pairs` already decides the gate column, and what changes is only that the grade may no longer disagree with it. It is one-way — a coverage shortfall can only remove eligibility — and coverage that is not recorded is not coverage that holds. `X` ranks below `B` and `C`: a band that had the rows and lost readably has something definite to say.

Causal references only. The two oracle references stay reported and gate nothing: an oracle is not causally available, so losing to one bounds what a verdict means rather than voiding it. The bar-weakness flag — does `constant_causal` clear the registered D-7 bar on its own? — is reported for the same reason. Neither is on the ladder. A condition that could not be measured is not satisfied, and is named like any other failure.

**Causal levelling (ABL-437): `fit_window`.** G2 and G3 read `constant_causal` and `climatology_causal` — levelled on the fit window, which is what this scope was registered and published under. ABL-437 measured that form to be inflated by up to 205% on a seasonal pair and re-levels it for new scopes; this scope keeps the reference its published letters were decided against, and the trailing columns are reported beside it so the difference is readable rather than asserted.

**G2/G3 readability (ABL-444): `sign_test`.** G2 and G3 are bare sign tests, `skill > 0`, which is what this scope was registered and published under. ABL-444 registers a floored form for new scopes — a margin inside the **10.65%** readability floor grades `N`, not readable — and re-reads the published scopes under it separately (`reports/abl_444_g23_floor_reread.md`) rather than restating their letters here.

**Readability test (ABL-467): `delta_min`.** Every condition is decided against ABL-385's imported `delta_min` floor, **10.65%** for this stream at k=1, which is the only tool available: a single fit carries no internal estimate of its own spread.

| country | horizon | n | challenger WAPE | D-7 WAPE | skill vs D-7 | constant causal WAPE | constant causal 28d WAPE | constant oracle WAPE | climatology causal WAPE | climatology causal 28d WAPE | climatology oracle WAPE | level inflation (causal / 28d) | incumbent WAPE | MAE | bias | slope | corr | gate | grade |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| BG | 24-36h | 720 | 20.0% | 24.4% | +18.2% | 75.3% | 75.2% | 73.5% | 42.0% | 20.7% | 19.2% | 2.5% / 2.3% | Not measured | 287.2 MW | -14.5% | 0.8 | 0.9 | PASS | A |
| BG | 36-48h | 720 | 19.7% | 24.4% | +19.1% | 75.3% | 75.1% | 73.5% | 42.0% | 20.8% | 19.2% | 2.5% / 2.1% | Not measured | 284.2 MW | -14.6% | 0.8 | 0.9 | PASS | A |
| BG | 48-64h | 510 | 21.4% | 25.0% | +14.4% | 68.2% | 62.7% | 63.8% | 41.3% | 22.1% | 20.4% | 6.8% / -1.8% | Not measured | 363.6 MW | -14.7% | 0.8 | 0.9 | PASS | A |
| CH | 24-36h | 720 | 7.8% | 12.7% | +38.6% | 95.1% | 100.0% | 94.6% | 37.5% | 10.2% | 9.0% | 0.5% / 5.7% | Not measured | 103.6 MW | -1.5% | 1.0 | 1.0 | PASS | A |
| CH | 36-48h | 720 | 7.8% | 12.7% | +38.8% | 95.1% | 100.1% | 94.6% | 37.5% | 10.2% | 9.0% | 0.5% / 5.7% | Not measured | 103.2 MW | -1.6% | 1.0 | 1.0 | PASS | A |
| CH | 48-64h | 510 | 8.3% | 12.5% | +33.6% | 86.0% | 81.0% | 87.9% | 36.6% | 9.8% | 8.7% | -2.2% / -7.9% | Not measured | 145.7 MW | -2.3% | 1.0 | 1.0 | PASS | A |
| CZ | 24-36h | 706 | 13.0% | 24.0% | +45.7% | 92.4% | 96.4% | 91.8% | 30.6% | 17.3% | 15.9% | 0.6% / 4.9% | Not measured | 127.2 MW | -2.7% | 0.9 | 1.0 | PASS | A |
| CZ | 36-48h | 706 | 13.1% | 24.0% | +45.5% | 92.4% | 96.4% | 91.8% | 30.6% | 17.3% | 15.9% | 0.6% / 4.9% | Not measured | 127.7 MW | -3.1% | 0.9 | 1.0 | PASS | A |
| CZ | 48-64h | 498 | 13.6% | 24.0% | +43.3% | 84.6% | 82.5% | 86.9% | 29.7% | 17.3% | 16.1% | -2.6% / -5.1% | Not measured | 169.5 MW | -3.1% | 0.9 | 1.0 | PASS | A |
| HU | 24-36h | 720 | 18.4% | 18.2% | -1.3% | 95.7% | 98.3% | 95.0% | 31.0% | 15.8% | 14.2% | 0.7% / 3.5% | Not measured | 244.4 MW | -13.1% | 0.8 | 1.0 | FAIL | U(+) |
| HU | 36-48h | 720 | 18.4% | 18.2% | -1.4% | 95.7% | 98.3% | 95.0% | 31.0% | 15.8% | 14.2% | 0.7% / 3.5% | Not measured | 244.5 MW | -13.0% | 0.8 | 1.0 | FAIL | U(+) |
| HU | 48-64h | 510 | 17.2% | 17.9% | +4.0% | 88.6% | 87.2% | 91.4% | 29.9% | 15.9% | 14.3% | -3.0% / -4.6% | Not measured | 280.4 MW | -9.6% | 0.9 | 1.0 | PASS | U(+) |
| PL | 24-36h | 704 | 17.1% | 26.0% | +34.1% | 90.9% | 92.8% | 90.8% | 28.3% | 17.1% | 15.4% | 0.1% / 2.2% | Not measured | 715.1 MW | -11.7% | 0.8 | 1.0 | PASS | A |
| PL | 36-48h | 704 | 17.2% | 26.0% | +33.8% | 90.9% | 92.8% | 90.8% | 28.3% | 17.2% | 15.4% | 0.1% / 2.3% | Not measured | 718.1 MW | -11.6% | 0.8 | 1.0 | PASS | A |
| PL | 48-64h | 498 | 16.1% | 24.5% | +34.4% | 84.5% | 82.0% | 85.5% | 27.3% | 16.2% | 14.6% | -1.2% / -4.2% | Not measured | 826.4 MW | -8.4% | 0.9 | 1.0 | PASS | A |
| RO | 24-36h | 720 | 18.1% | 24.3% | +25.6% | 96.3% | 100.5% | 95.8% | 43.6% | 21.9% | 19.9% | 0.5% / 4.9% | Not measured | 155.7 MW | -9.5% | 0.9 | 1.0 | PASS | A |
| RO | 36-48h | 720 | 18.1% | 24.3% | +25.5% | 96.3% | 100.4% | 95.8% | 43.6% | 21.9% | 19.9% | 0.5% / 4.8% | Not measured | 155.8 MW | -9.1% | 0.9 | 1.0 | PASS | A |
| RO | 48-64h | 510 | 18.7% | 25.0% | +25.0% | 92.3% | 91.3% | 93.0% | 42.5% | 22.7% | 20.4% | -0.8% / -1.9% | Not measured | 192.7 MW | -10.2% | 0.8 | 1.0 | PASS | A |
| SI | 24-36h | 720 | 18.4% | 21.6% | +15.2% | 94.9% | 100.1% | 93.8% | 35.8% | 15.4% | 13.0% | 1.2% / 6.7% | Not measured | 60.9 MW | -11.0% | 0.8 | 1.0 | PASS | A |
| SI | 36-48h | 720 | 18.3% | 21.6% | +15.6% | 94.9% | 100.1% | 93.8% | 35.8% | 15.4% | 13.0% | 1.2% / 6.6% | Not measured | 60.7 MW | -11.3% | 0.8 | 1.0 | PASS | A |
| SI | 48-64h | 510 | 19.0% | 21.2% | +10.5% | 86.7% | 84.9% | 90.1% | 35.0% | 15.1% | 12.8% | -3.8% / -5.8% | Not measured | 81.3 MW | -12.6% | 0.8 | 1.0 | PASS | U(+) |
| SK | 24-36h | 715 | 17.5% | 18.8% | +6.9% | 97.1% | 101.9% | 95.3% | 33.0% | 15.2% | 13.1% | 1.9% / 7.0% | Not measured | 20.1 MW | -11.0% | 0.8 | 1.0 | PASS | U(+) |
| SK | 36-48h | 715 | 17.6% | 18.8% | +6.5% | 97.1% | 101.9% | 95.3% | 33.0% | 15.2% | 13.1% | 1.9% / 6.9% | Not measured | 20.2 MW | -10.9% | 0.8 | 1.0 | PASS | U(+) |
| SK | 48-64h | 507 | 15.9% | 18.3% | +13.3% | 89.8% | 88.9% | 93.1% | 32.0% | 14.7% | 12.5% | -3.5% / -4.5% | Not measured | 23.1 MW | -7.8% | 0.9 | 1.0 | PASS | A |

### Graded disposition, per pair

| pair | bands | grade | failed conditions | not readable | bar weaker than a flat line? |
|---|---|:---:|---|---|:---:|
| BG | A / A / A | **A** | — | — | no |
| CH | A / A / A | **A** | — | — | no |
| CZ | A / A / A | **A** | — | — | no |
| HU | U(+) / U(+) / U(+) | **U(+)** | — | — | no |
| PL | A / A / A | **A** | — | — | no |
| RO | A / A / A | **A** | — | — | no |
| SI | A / A / U(+) | **U(+)** | — | — | no |
| SK | U(+) / U(+) / A | **U(+)** | — | — | no |

Reference levels used, from the same ABL-188-filtered target series the gate actuals and the D-7/persistence baselines come from — no refit, no second read, no additional upstream fetch. The hourly levels behind the climatology columns are in `results.json` in full; `h` is how many of the 24 hours of the day that level set covers, and anything below 24 means those rows were dropped from that column's n:

| country | constant causal | constant oracle | climatology causal | climatology oracle | constant causal 28d |
|---|---:|---:|---:|---:|---:|
| BG | 855.24 MW | 1087.86 MW | 2.17–1979.49 MW (24h) | 1.21–3282.79 MW (24h) | 1274.24–1457.50 MW (124 as-of) |
| CH | 833.37 MW | 677.22 MW | 0.73–2586.18 MW (24h) | 0.00–3695.79 MW (24h) | 1332.98–1459.22 MW (124 as-of) |
| CZ | 681.93 MW | 519.59 MW | 0.00–2016.85 MW (24h) | 0.00–2683.30 MW (24h) | 925.47–1025.11 MW (124 as-of) |
| HU | 939.53 MW | 614.61 MW | 0.38–2639.90 MW (24h) | 0.25–3362.72 MW (24h) | 1224.70–1344.81 MW (124 as-of) |
| PL | 3024.88 MW | 2722.84 MW | 0.00–8479.63 MW (24h) | 0.00–10590.47 MW (24h) | 3837.71–4381.55 MW (124 as-of) |
| RO | 510.41 MW | 408.12 MW | 0.00–1453.18 MW (24h) | 0.00–2424.62 MW (24h) | 793.39–876.98 MW (124 as-of) |
| SI | 219.98 MW | 122.17 MW | 0.60–649.31 MW (24h) | 1.15–977.35 MW (24h) | 313.95–344.10 MW (124 as-of) |
| SK | 78.61 MW | 37.70 MW | 0.22–244.37 MW (24h) | 0.84–352.99 MW (24h) | 108.88–114.57 MW (124 as-of) |

## Per-country all-D+2 summary

Gate-basis values (actual, challenger, seasonal_naive) share one finite intersection; each comparator outside the basis is scored on its own intersection with it, and its n is given in `comparator_n` in the JSON. A comparator showing `Not measured` had no finite rows at all.

| country | n | challenger WAPE | D-7 WAPE | persistence WAPE | constant causal WAPE | constant causal 28d WAPE | constant oracle WAPE | climatology causal WAPE | climatology causal 28d WAPE | climatology oracle WAPE | incumbent WAPE | TSO WAPE (revision-contaminated; n) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BG | 1,950 | 20.3% | 24.6% | 73.2% | 73.2% | 71.4% | 70.6% | 41.8% | 21.1% | 19.5% | Not measured | 33.2% (n=1,950) |
| CH | 1,950 | 7.9% | 12.6% | 87.5% | 92.2% | 94.0% | 92.5% | 37.2% | 10.1% | 8.9% | Not measured | 7.1% (n=1,950) |
| CZ | 1,910 | 13.2% | 24.0% | 84.7% | 90.0% | 92.0% | 90.3% | 30.3% | 17.3% | 16.0% | Not measured | 11.6% (n=1,910) |
| HU | 1,950 | 18.0% | 18.1% | 86.3% | 93.5% | 94.9% | 93.9% | 30.6% | 15.9% | 14.2% | Not measured | 14.7% (n=1,950) |
| PL | 1,906 | 16.8% | 25.5% | 84.7% | 89.0% | 89.5% | 89.2% | 28.0% | 16.9% | 15.2% | Not measured | 16.0% (n=1,906) |
| RO | 1,950 | 18.3% | 24.5% | 90.5% | 95.1% | 97.7% | 95.0% | 43.3% | 22.1% | 20.1% | Not measured | 30.2% (n=1,950) |
| SI | 1,950 | 18.5% | 21.5% | 86.6% | 92.3% | 95.3% | 92.7% | 35.6% | 15.3% | 12.9% | Not measured | 18.7% (n=1,950) |
| SK | 1,937 | 17.0% | 18.7% | 86.9% | 94.8% | 97.9% | 94.6% | 32.7% | 15.0% | 12.9% | Not measured | 14.4% (n=1,872) |

## Fit and missingness audit

Every training row was built with `RenewableFeatureBuilder.row(target, generated_at, observation_as_of=generated_at)`. Gate targets were never fitted.

| country | algorithm | retained / intended fit rows | unique fit targets | excluded missing | degraded lag-1d rows | artifact SHA-256 |
|---|---|---:|---:|---:|---:|---|
| BG | catboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `2d27995e6ca459e09e4139ac67b409dcdf1224011dae0805dff5a2e54a1b706c` |
| CH | catboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `bb71903ce1e590a0e59db2de0c3bf667e9d480d8e042737ea340903e6f1b3bd5` |
| CZ | catboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `e6295daf4b8f27e18fb17911a2cb71bffeef0c5def81abab5ba8cb4abfa81b6f` |
| HU | catboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `3a6f265e3214b5a386d6a440b6d0e065ee5aa741246f1f9ad8eba2cc9dbf7e7d` |
| PL | catboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `58df0719d492d3634238a63a627b684119ee7a6424453907e4f89279a5acddbf` |
| RO | catboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `182749b44ea69394472ecaa161e64de4c43611d51e7956458521f9ca233e2bf1` |
| SI | catboost | 34,176 / 34,176 | 4,272 | 0 | 23,674 | `07fa40d49d6457b6b22d4f8f3275feef3d210fb0749404a5aa808828e63383eb` |
| SK | catboost | 34,064 / 34,176 | 4,262 | 112 | 23,598 | `f8624c90e820a3ae5f0865ec7373fbe05e5a177cf373247a8b52d40ef6d94397` |

### Physically impossible night rows (ABL-376)

Not registered for scope `abl316-t2a-generation`. The fit saw every night row, including any whose actual the sun says is impossible.

## Data quality and limits

- ABL-188 constant-run screening found no ≥24-hour bit-identical solar run in `energy_generation` over the registered fit/scoring interval plus 14-day feature lookback (2025-12-31 → 2026-08-10 UTC). The builder still routes solar through `exclude_suspect_constant_runs`; the invariant was verified on the actual window, not assumed from ABL-191.
- ABL-67 is net-position-only; ABL-109/111 are load-only. ABL-71's known wrong-write modes are load and net position, not solar; this is a provenance caveat, not proof that solar ingest is pristine.
- Weather rows were admitted only where `forecast_run_time <= generated_at`; missing vintages were excluded, never filled with future reanalysis.
- TSO values come from an `INSERT OR REPLACE` table without first-seen vintages. They may include revisions and cannot support promotion.
- This is one 30-day summer holdout. It is out-of-sample by target timestamp, but not year-round evidence.
- **The challenger loses to a climatology chosen with hindsight in 15 cell(s).** An hour-of-day median — the average day, with no model and no weather in it — scores better than the fitted model there. This is the weaker claim to lose and the stronger one to win: a challenger that beats it is doing something no table of hourly averages can do. This does not change any verdict above; it bounds what the verdict means:
  - BG solar 24-36h: challenger 19.95% vs oracle climatology 19.15% (+0.80pp)
  - BG solar 36-48h: challenger 19.75% vs oracle climatology 19.15% (+0.59pp)
  - BG solar 48-64h: challenger 21.40% vs oracle climatology 20.38% (+1.01pp)
  - HU solar 24-36h: challenger 18.42% vs oracle climatology 14.18% (+4.24pp)
  - HU solar 36-48h: challenger 18.43% vs oracle climatology 14.18% (+4.25pp)
  - HU solar 48-64h: challenger 17.18% vs oracle climatology 14.28% (+2.90pp)
  - PL solar 24-36h: challenger 17.14% vs oracle climatology 15.40% (+1.74pp)
  - PL solar 36-48h: challenger 17.21% vs oracle climatology 15.40% (+1.81pp)
  - PL solar 48-64h: challenger 16.09% vs oracle climatology 14.61% (+1.47pp)
  - SI solar 24-36h: challenger 18.35% vs oracle climatology 13.01% (+5.34pp)
  - SI solar 36-48h: challenger 18.27% vs oracle climatology 13.01% (+5.26pp)
  - SI solar 48-64h: challenger 18.99% vs oracle climatology 12.76% (+6.23pp)
  - SK solar 24-36h: challenger 17.53% vs oracle climatology 13.11% (+4.42pp)
  - SK solar 36-48h: challenger 17.59% vs oracle climatology 13.11% (+4.48pp)
  - SK solar 48-64h: challenger 15.90% vs oracle climatology 12.52% (+3.38pp)

## Recommendation to the CEO

Do not promote these artifacts: only 22/24 primary cells clear the registered bar. Report the losing country/bands as the finding and pursue country-specific diagnosis/model work on a fresh pre-registered split.

No production deploy, serving-registry change, model promotion, ingest change, dashboard change, replica write, or sidecar write was performed.
