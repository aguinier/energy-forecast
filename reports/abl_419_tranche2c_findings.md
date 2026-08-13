# ABL-419 — ABL-316 tranche 2c: the five Mediterranean solar countries

Parent **ABL-316**. Registration **ABL-348** (`experiments/ABL348/config.json`),
unchanged. Scope `abl316-t2c` = ES, GR, HR, IT, PT on `energy_generation`.
**5 countries × 3 primary D+2 bands = 15 cells.**

**Disposition: FAIL — 9 of 15 cells clear the registered bar, and not one pair
reaches grade A.**

Machine record: `experiments/ABL348/results_abl419_tranche2c.json`
(SHA-256 `fe25b86c983040591cff48e6a84fdca255e51d2f070c658915ea4aa9d73044bf`).
Harness report: `reports/abl_419_solar_tranche2c.md`.
Generated tables: `reports/abl_419_tranche2c_tables.md` / `.json`
(`scripts/abl419_tranche2c_read.py`).
GR resolution probe: `reports/abl_419_gr_resolution.json`
(`scripts/abl419_gr_resolution_probe.py`).

> **No promotion is requested or implied.** No serving-registry change, no write
> to `forecasts`, no ingest change, no dashboard change, no replica write. The
> replica was opened `mode=ro` throughout. Promotion is a CEO-to-Board decision
> and this pack is evidence for it, not a step in it.

---

## 1. Protocol — only what is new or was checked

Everything registered — windows, bands, metric, baseline, minimum n, source
table — is **ABL-348's and is not restated here**. Twenty-eight remaining
tranches must not become twenty-eight chances to shop a window.

| | |
|---|---|
| fit targets | 2026-01-14 → 2026-07-11 (exclusive) |
| gate targets | 2026-07-11 → 2026-08-10 (exclusive), out-of-sample by target timestamp |
| metric / baseline | WAPE / literal seasonal-naive D-7, recomputed on the same table |
| source table | **`energy_generation`** — passed explicitly as `--renewable-source`, and recorded as `meta.training_source` |
| features | 27 (`legacy25+geometry`), the module default; `exclude_impossible_night: False` |
| algorithm | CatBoost, seed 42 (`config.random_seed`) |
| interpreter | `C:\Code\able\energy-forecast\.venv\Scripts\python.exe` — Python 3.14.3 |
| replica | `C:\Code\able\data\energy_dashboard.db`, **9,432,453,120 bytes**, `mode=ro` |

**Verified rather than trusted, before the first fit:**

- **The replica by size, not by path.** 9,432,453,120 bytes is the live replica
  ABL-348 registered its bars against, not the 3.0 GB partial snapshot.
- **No incumbent exists to overwrite.** `forecasts` holds solar rows for
  AT/BE/DE/FR only and **zero** for each of ES, GR, HR, IT, PT. This scope
  refits no live pair.
- **Neither NOT-EVALUABLE pair is in scope.** ABL-348 declares `EE/solar` and
  `FI/solar` NOT-EVALUABLE; neither is here, and all five carry
  `n_d7_scorable` = 720.
- **All 15 cells met their registered minimum n** (720/720/510 against
  684/684/456). Nothing is unreadable for want of rows.
- **Fit was complete for every country**: 34,176 / 34,176 intended rows
  retained, 0 excluded missing, for all five.

**The registration is six tables and `check_registration_tables(...)` checks
three.** `SCOPES`, `GATE_BASIS` and `SCOPE_OUTPUTS` abort at import if they
disagree; `FIT_RULES`, `SCOPE_TITLES` and `SCOPE_FEATURES` do not, and default
silently. All six were edited and verified by resolution. `SCOPE_FEATURES`
carries a **deliberate absence** — a row bound to `FEATURE_COLUMNS` would look
like a pin without being one, and would flip
`meta.feature_set_is_registered_for_scope` to a claim that is not true of this
registration. It cannot join the strict check either: `abl316-t2a` is
deliberately absent from it (ABL-404), so adding it aborts at import on a scope
whose absence is correct and published.

**Contamination touching this window.** ABL-67 is `net_position`-only;
ABL-109/ABL-111 are `energy_load`-only; neither table is read here. ABL-71's
known wrong-write modes are load and net position — a provenance caveat, not
proof that solar ingest is pristine. **ABL-188** constant-run screening found no
≥24-hour bit-identical solar run in `energy_generation` across
2025-12-31 → 2026-08-10 for any of the five; the invariant was verified on the
actual window rather than assumed. §4 reports a resolution change in GR that no
existing contamination issue covers.

---

## 2. The read

| pair | bar (ABL-348, pre-committed) | cells | skill vs D-7 | ladder grade | **reported** |
|---|---:|:---:|---|:---:|:---:|
| IT | **7.11%** | 3/3 PASS | +4.8 / +3.6 / +8.1% | U(+) | **U(+)** |
| GR | 10.37% | 0/3 | **−104.3 / −104.4 / −99.1%** | C — fails G1 | **C** |
| ES | 11.78% | 3/3 PASS | +2.5 / +2.3 / +1.0% | U(+) | **B** (ABL-411 hold) |
| PT | 13.09% | 0/3 | **−10.7 / −13.5 / −15.6%** | C — fails G1 | **C** |
| HR | 16.43% | 3/3 PASS | +7.8 / +7.2 / +8.6% | U(+) | **U(+)** |

**Not one pair reaches grade A, and that is the finding rather than the 9/15.**
Every pair that clears its bar clears it by less than the readability floor —
ABL-385's `delta_min(k=1)` for solar with `c_B = 0`, **10.6482%** — so ES, HR
and IT are all `U(+)`: the passes are real but **unreadable at one seed**, and
their registered disposition is *re-read at k>1 seeds*, not *accept*. GR and PT
lose readably.

### The bar-weakness flag is uninformative on solar, and this tranche shows why

ABL-406 established across eight `wind_onshore` pairs that the gate outcome was
*fully* predicted by whether `constant_causal` clears the registered bar on its
own. **That does not transfer to solar, and ABL-419 is not the first sign of
it.** The flag reads `no` for all five pairs here — and it read `no` for all
eight of tranche 2a's, which produced seven grade A's. A flat line cannot
represent a diurnal cycle, so on solar `constant_causal` scores **89.8–98.9%**
WAPE and clears no bar anywhere; the flag is `no` by construction and separates
nothing.

The reference that carries information on solar is the **hour-of-day
climatology**, exactly as `REPORTED_COMPARATORS` in the harness already argues.
And on that reference this tranche reads badly:

> **The challenger loses to a hindsight hour-of-day climatology in 15 of 15
> cells** — by +0.87pp (PT) to +12.19pp (GR). In tranche 2a the same comparison
> lost 15 of 24.

An hour-of-day median, with no model and no weather in it, beats the fitted
model on every cell of this tranche. That gates nothing — an oracle is not
causally available — but it bounds what these three `U(+)` passes mean. Against
the *causal* climatology the challenger wins everywhere (29.6–46.5% vs
6.0–20.8%), so the model is doing real work; it is the **level within the
average day** that it is not yet capturing.

### Do not average this tranche's pass rate against 2a's

2a's bars ran 18.35–26.11% plus CH at 12.67%; these run **7.11–16.43%**.
ABL-348 registered that reading in advance under
`reading_caveats_not_band_changes`: same band, materially harder task. IT's
challenger at **6.05–6.71% WAPE** is the **lowest solar WAPE in any committed
gate read** — the next best is CH at 7.51% (2a), then CZ at 12.86%, against
12.69–15.56% for the three incumbent countries under `abl376` — and it still
cannot readably beat a 7.11% bar. **A smaller margin here is a harder problem,
not a worse model.**

---

## 3. ES — the band, and a cap that binds

ES gate-window `f` = **1.3520%** of energy booked at night (ABL-396 §2),
reproduced from that issue's committed machine record rather than cited.

| band | n | all-hours WAPE (measured) | implied daylight-only | if clamped to 0 at night | D-7 bar | registered | clamped variant |
|---|---:|---:|---:|---:|---:|:---:|:---:|
| 24-36h | 720 | 11.39% | 10.18%–11.55% | 11.39%–12.74% | 11.69% | **PASS** | **indeterminate** |
| 36-48h | 720 | 11.41% | 10.20%–11.57% | 11.41%–12.77% | 11.69% | **PASS** | **indeterminate** |
| 48-64h | 510 | 11.03% | 9.81%–11.18% | 11.03%–12.38% | 11.15% | **PASS** | **indeterminate** |

**The registered verdict is a direct measurement and the band does not qualify
it**: challenger and D-7 are scored on the identical all-hours rows, so ES's
night floor cannot have moved it in either direction.

**The clamped column is serving-side, is free, and is the one new thing here.**
The ABL-337 clamp forces a zero on this same night predicate, so a *served*
version of this challenger scores in `[A, A+f]`. On all three bands that
interval **straddles the D-7 bar**. The bound cannot say whether a clamped ES
would clear it — and that is worth handing to whoever owns serving, because it
means ES's PASS is not robust to the clamp. Settling it needs an actual
daylight-only read, which this bound deliberately does not substitute for.

**ES is reported at grade B with `ABL-411 hold` named**, per this issue's
ruling. Two things about the cap that matter:

- **It binds.** ABL-418 orders grades by *severity* —
  `{"A": 0, "U": 1, "B": 2, "C": 3}` — so `U(+)` is **less severe than B** and
  the cap pulls it down. A reading that took `U(+)` for "already below B" would
  have left ES uncapped on the one read the cap was written for.
- **It costs information, so the ladder grade is printed beside it.** `U(+)`
  and `B` are both non-promotion but carry *different next steps*: `U(+)` means
  re-read at k>1 seeds, `B` means not promotion-eligible with failures named.
  ES's underlying `U(+)` is what its read actually says once ABL-411 settles.

Nothing in this read depends on whether ES's overnight MW is CSP dispatch: the
exclusion fit rule is off, and this issue changes no serving path. **ABL-403's
soft hold is discharged** — the band above bounds the only cell its 2×2 could
have moved here, and it cost nothing.

---

## 4. GR — a real data defect, and it is not the explanation

GR is the worst cell in the tranche: **20.8% against a 10.2% bar, skill −104%**.
Its error is a *level* error, not a shape error — **correlation 1.00, slope 0.8,
bias −18.2%**. The model tracks GR's diurnal cycle almost perfectly and sits
~18% under it.

**The finding: GR's `energy_generation` changes resolution inside the registered
fit window.** It is hourly (1 row/hour) through 2026-04-30 and quarter-hourly
(4 rows/hour) from mid-May 2026:

| GR, rows per hour | Dec | Jan | Feb | Mar | Apr | **May** | Jun | Jul | Aug |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `energy_generation` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **3.62** | 4.00 | 4.00 | 4.00 |

So 4,272 fit-window hours carry a `:00` row but only **1,611 carry
`:15`/`:30`/`:45`**, while all 720 gate-window hours carry all four. The model
is fitted across a resolution change and scored entirely on the far side of it.
**None of the other four countries does this** — ES/HR/IT are 4/hour throughout,
PT is 1/hour throughout.

**And it is not the mechanism.** Measured on the hours carrying all four samples
— within one sample, not across a season — the `:00` instant differs from the
true 4-sample hourly mean by **+0.021%** of level in the fit window and
**−0.021%** in the gate window. An 18.2% bias cannot come from a 0.02% one. The
cadence change is a real defect in a registered window and it does **not**
explain GR's failure.

Two further candidates, both partial and neither sufficient:

| | GR | HR | ES | PT | IT |
|---|---:|---:|---:|---:|---:|
| fit → gate level ratio | **1.857** | 1.621 | 1.547 | 1.547 | 1.419 |
| gate hours above fit-window max | **29 / 720** | 1 | 10 | 1 | 1 |

GR leads both — the level ratio matches the sign of the bias, and CatBoost is a
tree ensemble whose predictions cannot exceed the target range it was fitted on.
But **HR sits at 1.62 and passed**, and 4.0% of hours cannot produce an 18.2%
bias. GR's failure is not explained here, and this pack stops rather than
asserting a cause it has not established. Recommended as its own diagnosis
issue.

> *A trap worth recording.* Averaging **rows** rather than **hours** over a
> mixed-cadence series over-weights the quarter-hourly era — which for GR is the
> summer. Done that way GR's fit mean reads 1720.67 MW against the harness's own
> 1439.96 MW, and GR's extrapolation ratio looks unremarkable at 1.55 instead of
> 1.86. Every level above is hour-weighted and cross-checked against the
> `constant_causal` levels in this tranche's own results file, which it
> reproduces **exactly for all five countries**.

---

## 5. PT — a genuine shortfall, with a quantified partial contributor

PT loses readably (−10.7 / −13.5 / −15.6%) against a 13.09% bar, and unlike GR
it loses to the *oracle* climatology too by only +0.87 to +1.39pp — so PT's
challenger is close to the average day and its bar is simply tighter than its
skill.

One measurable contributor, which the gate-window night screen does not show.
**PT's fit window carries a night floor that its gate window does not**, and it
is source-dependent:

| PT, `f` (share of energy at night) | fit | gate |
|---|---:|---:|
| `energy_generation` (registered) | **0.4490%** — 1,444 of 1,580 night hrs > 1 MW, mean 9.35 MW | 0.0090% — 7 of 211, mean 0.38 MW |
| `energy_renewable` | 0.0010% — 0 of 1,569 | 0.0090% |

The model was fitted on a series with a ~9 MW overnight floor and scored on one
without it. **Sizing this is an upper bound and rests on one assumption that was
not measured** — that the model reproduces its fitted night floor in the gate
window. Granting that: ~211 night hours × ~8.6 MW of error against a gate
denominator of 1,191.41 MW × 720 h is **≈0.21pp** of WAPE. Real, directionally
right, and roughly a seventh of PT's 1.4–2.0pp shortfall. It does not overturn
the FAIL, and confirming it would need the artifact's own night predictions
rather than this bound.

GR, HR and IT screen at **`f` ≤ 0.003%** in the fit window and **0.0000%** in
the gate window. The zeros are stated rather than omitted, as this issue asked.

---

## 6. A defect found in tranche 2a while registering this one

`experiments/ABL348/results_abl405_tranche2a.json` records
`meta.training_source = energy_renewable`, and the harness report generated from
it says so on its face — while `reports/abl_405_tranche2a_findings.md` states
`| source table | energy_generation |`. ABL-348 lists the source table under
`voids_this_registration`. Re-read at `origin/main`, not grepped from a working
tree. **Filed separately with its sizing**; no 2a verdict is claimed to be
wrong, and the measured footprint inside the registered windows is small (gate
D-7 bar identical to 0.00pp on all eight 2a countries; gate actuals bit-identical
on seven of eight; fit window differs by 0.63–0.77% of rows).

**It is relevant here because this tranche contains the two solar pairs where
that choice would not have been small.** On ABL-348's own pre-measured bars, GR's
two tables disagree by **4.877% of level** in the gate window (mean abs diff
130.36 MW, max 1,342 MW, only 81.2% of hours bit-identical) and move the D-7 bar
**−0.71pp**; PT's by 0.317% and −0.12pp. This read was taken on
`energy_generation` as registered, and `meta.training_source` records it.

---

## 7. Recommendation to the CEO

1. **Do not promote anything from this tranche.** No pair reaches grade A. ES,
   HR and IT pass their bars but inside the readability floor; GR and PT fail
   readably.
2. **ES stays behind `ABL-411 hold`** regardless, per this issue's ruling, and
   separately its clamped-variant verdict is indeterminate — worth resolving
   before ES is considered for serving at all.
3. **The cheapest next step for ES/HR/IT is a k>1 seed re-read**, which is what
   `U(+)` registers. ABL-385's protocol already exists; three pairs × k seeds is
   a bounded spend and would convert three unreadable passes into readable ones
   or into honest ties.
4. **GR needs its own diagnosis**, and its resolution change needs an owner —
   the cadence shift is excluded as the cause of the gate failure but is a real
   defect in a registered window that no contamination issue currently covers.
5. **The solar bar-weakness flag should not be read as ABL-406 reads it on
   wind.** On solar it is `no` by construction. If a second qualifying reference
   is wanted for solar, the causal hour-of-day climatology is the one that
   discriminates.

No production deploy, serving-registry change, model promotion, ingest change,
dashboard change, replica write, or sidecar write was performed.
