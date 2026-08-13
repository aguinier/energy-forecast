# ABL-380 — ABL-316 tranche 1a: BG and CH wind_onshore on `energy_generation`

**Gate disposition: PASS, 6/6 cells.**
**BG wind_onshore: passes on merit — recommended to the CEO as evidence, not as a promotion.**
**CH wind_onshore: passes the arithmetic and carries no decision. See §4 — its margin is
level-matching, and a flat line chosen with hindsight beats it.**

Gate read: `reports/abl_380_wind_onshore_tranche1a.md` (harness-generated).
Machine record: `experiments/ABL348/results_abl380_tranche1a.json`.
Registration: `experiments/ABL348/config.json`, frozen at ABL-348 and not re-derived here.
Scope: `abl380-tranche1a`, registered in `scripts/evaluate_wind_retrain.py` at commit
`1704737`.

No promotion, no serving-registry change, no ingest change, no dashboard change, no
replica write. Promotion remains CEO-to-Board.

---

## 1. Protocol, and what was verified rather than trusted

| | |
|---|---|
| Fit window | 2026-01-14 → 2026-07-11 exclusive (178 d, 4,272 hourly targets) |
| Gate window | 2026-07-11 → 2026-08-10 exclusive (30 d, 720 hours), out-of-sample by target timestamp |
| Source table | `energy_generation`, passed explicitly, never defaulted |
| Baseline | literal seasonal-naive D-7, recomputed on the same table |
| Algorithm | catboost (`ALGORITHMS["wind_onshore"]`) |
| Metric | WAPE, per horizon band |
| Interpreter | `.venv` Python 3.14.3, catboost 1.2.10, xgboost 3.3.0 |

**Ordering.** The scope registration commit `1704737` is timestamped **2026-08-13T08:32:12Z**;
the first fit started **08:32:26Z** and the run ended **08:34:40Z**. The registration is
earlier than the first fit and the ordering is checkable in git by anyone later — the
property ABL-322 held and the reason its pass was believable.

**Which file was read.** `C:\Code\able\data\energy_dashboard.db`, **9,432,453,120 bytes** —
byte-identical in size to the file the ABL-348 bars were measured on, and not the 3.0 GB
stale snapshot at `energy-data-gathering/`. The run's own `meta.databases` records
`features_match_replica: true` and `ambient_matches_replica: true`, so the fitted series,
the baselines, the gate actuals, the TSO series and the contamination screen all came from
that one file. The worktree has no `.env`; every path was passed explicitly.

**Preconditions, measured before fitting** (`scripts/abl380_tranche_precheck.py`, read-only,
fits nothing):

| | BG | CH | registered |
|---|---:|---:|---|
| gate hours present | 720 / 720 | 720 / 720 | ≥ 684 for the 720-bands |
| ABL-188 constant runs, fit / gate / lookback | 0 / 0 / 0 | 0 / 0 / 0 | — |
| native sub-hourly rows in gate window | none | none | ABL-332: BG, CH hourly throughout |
| hours bit-identical vs `energy_renewable` | 720 / 720 | 720 / 720 | — |
| literal D-7 WAPE | **93.75%** | **59.26%** | 93.75% / 59.26% |

The frozen per-pair bars reproduce **exactly, to two decimals**, through a code path
independent of the one that set them. The harness's own per-band D-7 lands on the same
numbers for both 720-hour bands (BG 93.75%, CH 59.26%); the 48-64h band scores a 510-row
subset and so differs slightly (BG 89.32%, CH 59.81%), as the registration anticipated.

**Contamination.** ABL-67 is net-position-only, ABL-109/ABL-111 are load-only — neither
intersects these targets. ABL-71's known wrong-write modes are load and net position; that
is a provenance caveat, not proof wind ingest is pristine. **ABL-188 does not touch either
pair in any of the three windows**, measured above rather than assumed.

---

## 2. The gate read

All six cells cleared their registered minimum n and beat D-7.

| pair | band | n / min n | challenger | D-7 | relative skill | incumbent |
|---|---|---:|---:|---:|---:|---|
| BG | 24-36h | 720 / 684 | 56.86% | 93.75% | +39.3% | Not measured |
| BG | 36-48h | 720 / 684 | 56.82% | 93.75% | +39.4% | Not measured |
| BG | 48-64h | 510 / 456 | 57.76% | 89.32% | +35.3% | Not measured |
| CH | 24-36h | 720 / 684 | 47.42% | 59.26% | +20.0% | Not measured |
| CH | 36-48h | 720 / 684 | 44.99% | 59.26% | +24.1% | Not measured |
| CH | 48-64h | 510 / 456 | 44.31% | 59.81% | +25.9% | Not measured |

**The incumbent rendered `Not measured` and no cell was voided.** This is the acceptance
criterion, and it is the ABL-322 defect not recurring: both pairs hold zero rows in
`forecasts`, so under the four-way gate basis every one of these six cells would have
intersected to n=0 and the harness would have printed FAIL on a comparison that never ran.
The `abl380-tranche1a` basis is `(challenger, seasonal_naive)` — the two columns the
registered bar names — and the incumbent is still reported on its own intersection, where
it correctly reads `comparator_n: 0`.

Artifacts were written through `Forecaster.save` via `save_gate_artifact` and both carry
`training_source = 'energy_generation'`, verified by loading them back under the rail
interpreter.

---

## 3. BG wind_onshore — dispositioned

**BG passes, and unlike CH it passes on merit.** Challenger 57.07% vs D-7 92.68% over
n=1,950 across all D+2 bands.

The bar itself is weak, and I would rather say so than let +39% relative stand unqualified.
BG's D-7 baseline has **correlation −0.159 and slope −0.141** against truth in this window:
seasonal-naive on a 108.9 MW Bulgarian wind fleet is not merely uninformative, it is
slightly *anti*-correlated. The registration's `small_fleet_wind_bar_is_loose` caveat
anticipated a loose bar; this is looser than loose.

What makes BG's pass real is §4's reference: the challenger at 57.07% beats not only D-7
but the **best constant chosen with hindsight** (63.78%), by 6.9pp. A model that beats the
best possible flat line is carrying genuine dynamic information. Its correlation is 0.554.

Its calibration is poor: **slope 0.261**, bias −16.7%. The model under-responds to variation
by roughly a factor of four and runs low. That is a bias/affine-calibration opportunity on a
pair that has now demonstrated it has signal to calibrate — but calibration must be fitted
on a split this gate window is not part of, and I have not done it here.

**Recommendation for BG: evidence to the CEO. Not a promotion, and not a promotion
recommendation on this evidence alone** — see §5, which I consider the more important
result for BG than the pass.

---

## 4. CH wind_onshore — the small-denominator demonstration

This is the pair the CEO asked to be built precisely so that this could be shown once with
numbers instead of argued across ten pairs later. It shows cleanly.

**CH clears its bar and the clearance carries no information.**

Reference points, all on the same 720 gate hours:

| predictor | CH WAPE | BG WAPE |
|---|---:|---:|
| literal D-7 (the registered bar) | 59.26% | 93.75% |
| constant at the **fit-window mean** — causal, no model | 79.07% | **82.77%** |
| constant at the **gate-window median** — *oracle*, not available at forecast time | **40.29%** | 63.78% |
| **challenger** | **47.42%** | **56.86%** |

Read the CH column. **A flat line chosen with hindsight scores 40.29% and the fitted model
scores 47.42%.** The challenger is 7.1pp *worse* than a constant. Everything it earns over
the D-7 bar it earns by predicting something close to the level and varying very little
around it — and it does not even find the best level.

The supporting statistics agree and are not a separate claim: CH challenger **slope 0.094,
correlation 0.176**. The model responds to variation at about a tenth of the true amplitude.
A WAPE of 45.7% on a series whose mean is **12.9 MW** means a mean absolute error of
**5.9 MW**. The fleet's own output fell from a 21.97 MW fit-window mean to a 12.91 MW
gate-window mean; on a denominator that small, a shift in level dominates the metric and
skill barely registers in it.

**So: CH is reported, and it gates nothing.** It should not be promoted, should not be read
as evidence the catboost path works, and should not be counted as a pass when the tranche
programme's pass rate is tallied. A 59.26% bar on a ~13 MW series is not a meaningful
reference, exactly as ABL-348 registered it and as the CEO restated on this issue.

Note the BG column also matters, and it is the reason this reference is worth computing for
**every** small-fleet pair rather than for CH alone: **BG's causal constant-at-fit-mean
scores 82.77% and therefore clears BG's 93.75% D-7 bar outright.** On BG the bar is
passable with no model at all. BG survives that test because it beats the oracle constant
too; a pair that clears D-7 but not the oracle constant has demonstrated nothing.

**Proposal for the remaining 33** (CEO's call, not mine to adopt unilaterally): report the
constant-predictor reference alongside every pair. It is arithmetic on actuals already
loaded, costs no compute, requires no refit, and it is the difference between BG's pass and
CH's. It changes no registered band, bar, metric or minimum n — it is a reading aid beside
the gate, not a second gate.

---

## 5. The finding that outranks the pass: the TSO forecast beats both challengers

| pair | challenger WAPE | TSO WAPE | TSO slope | TSO corr | challenger slope | challenger corr |
|---|---:|---:|---:|---:|---:|---:|
| BG | 57.07% | **50.14%** | 1.036 | 0.818 | 0.261 | 0.554 |
| CH | 45.71% | **27.78%** | 0.539 | 0.706 | 0.094 | 0.176 |

n=1,950 for every figure. The harness flagged both automatically.

BG's TSO forecast has a **slope of 1.036 and correlation 0.818** — very nearly calibrated,
and far better conditioned than our fit. CH's TSO scores 27.78% against our 45.71% and
against the oracle constant's 40.29%.

That last comparison carries a correction worth stating plainly: **CH wind is not
unforecastable.** It would be easy to read §4 as "13 MW of fleet is noise". The TSO gets
27.78% with correlation 0.706 on the same 1,950 rows, so the signal is there and our feature
set is not capturing it. CH's problem is our model, not the physics — the *metric* is what
becomes unreliable at that denominator, not the target.

Counting the pilot, the TSO now beats a freshly-fitted challenger on **3 of the 4 pairs
measured across ABL-322 and ABL-380** (DE offshore, BG onshore, CH onshore; NL offshore was
the exception). That is a pattern rather than an anecdote, and it points at the same lever
twice: **ingesting the TSO day-ahead generation forecast as a feature is likely worth more
than model selection or tuning across the whole ABL-316 programme.**

Caveat, unchanged from ABL-195: the TSO series comes from a replacement table without
first-seen vintages, so it may contain revisions and **cannot support a promotion decision**.
It is context here. Establishing a vintage-faithful TSO feature is its own piece of work and
would need the ingest owner; I am flagging the size of the prize, not proposing to build it
inside this tranche.

---

## 6. Cost — ABL-322's sizing assumption does not hold, and the correction is upward

The issue asked for a second point against ABL-322's ≈60 s/pair seed, with the expectation
that hourly countries would come in appreciably under it. **They do not. They land on it.**

| issue | pair | resolution | algorithm | fit rows | feature build | fit | gate | pair total |
|---|---|---|---|---:|---:|---:|---:|---:|
| ABL-322 | DE wind_offshore | 15-min | xgboost | 34,176 | 55.9 s | 2.3 s | 7.3 s | **65.5 s** |
| ABL-322 | NL wind_offshore | 15-min | xgboost | 34,176 | 46.1 s | 2.0 s | 8.6 s | **56.8 s** |
| ABL-380 | BG wind_onshore | hourly | catboost | 34,176 | 44.6 s | 3.6 s | 7.2 s | **55.3 s** |
| ABL-380 | CH wind_onshore | hourly | catboost | 34,176 | 52.7 s | 4.3 s | 8.3 s | **65.3 s** |

The two ranges are the same interval: 56.8–65.5 s for the "expensive" 15-minute pairs,
55.3–65.3 s for the "cheap" hourly ones. Feature build is 44.6–52.7 s here against
46.1–55.9 s there, and remains the dominant term at **81%** of pair cost (ABL-322 measured
~85%).

**Why the prediction was wrong.** ABL-322 reasoned that hourly countries "carry a quarter of
the rows into the same aggregation". They do — but the aggregation is a one-time read, and
it is not what the 45–56 seconds is spent on. The **fit rows are 34,176 for all four pairs,
identical**, because that number is 4,272 registered hourly targets × 8 pre-registered
vintages: it is fixed by the registration and the vintage schedule, not by the country's
native resolution. The builder is invoked once per (target, vintage) regardless. Source
resolution is amortised across those 34,176 calls and does not survive into the total.

**Consequence for sizing the remaining 33: ≈60 s/pair is a median, not an upper bound, and
there is no cheap tail to look forward to.** ABL-322's 90 s planning figure still holds —
it was already a 50% pad — but its justification changes, and the residual optimism in "we
have only measured the expensive case" should be dropped.

- 33 remaining pairs × 90 s ≈ **50 minutes** of compute; at the observed 65 s maximum, ≈36 min.
- Fixed per-invocation overhead measured at **13.4 s** (134 s CLI wall-clock against 120.6 s
  of summed pair time): interpreter start, the incumbent and TSO reads, and report writing.
  Seventeen two-pair tranches would spend ~3.8 minutes total on it. **Tranche granularity is
  effectively free** — the programme can be split for review convenience without a compute
  penalty, which is the operationally useful half of this measurement.
- Unchanged caveat from ABL-322: this is the wind builder. Solar runs a different harness and
  builder and must be sized off its own measurement once ABL-379 lands.

Measured on a workstation running other work; treat as an upper bound, not a benchmark.

---

## 7. The catboost path, first exercise on new countries

The issue asked whether catboost behaves differently from the pilot's xgboost. **One
difference, and it does not matter.**

Catboost fits in **3.6–4.3 s** against xgboost's 2.0–2.3 s — roughly 2× — but the fit is
6–7% of pair cost, so a doubling moves the total by about two seconds and is invisible next
to the 45–53 s feature build. Nothing else differed: both pairs retained **34,176 / 34,176**
intended fit rows with zero exclusions, artifacts saved and reloaded cleanly through
`Forecaster.save`, and `training_source` round-tripped. No warnings, no `ModelArtifactError`,
no shape-without-level symptom of the ABL-69 interpreter trap.

I did not exercise the ABL-69 failure mode deliberately: it is an xgboost pickle
phenomenon and catboost artifacts are not subject to it. That is an untested boundary rather
than a cleared one, and the catboost intercept witness has not been checked the way
`src/challengers/v014.py` checks xgboost's.

---

## 8. What makes the remaining 33 different from these two

Honest answer: **on cost, nothing — and that is the finding.** §6 shows resolution does not
drive cost, so BG and CH are representative rather than easy, and the sizing generalises.

On readability they were the easiest pairs available, and the next tranches will not be:

1. **Neither pair needed the ABL-188 screen.** Five hits sit inside the registered windows
   for other pairs (CZ solar, CZ wind_onshore, NL solar, EE solar), so the next tranches
   will exercise the excluded-rows path that this one left at zero, and n will move.
2. **Both tables were bit-identical here in all 720 co-observed hours.** GR is registered as
   the pair where they materially disagree (81.2% identical, 4.88% of level); a GR read must
   not be treated as interchangeable between sources.
3. **EE/solar and FI/solar are declared NOT-EVALUABLE before any fit** and must be reported
   as such, never as FAIL.
4. **Small-fleet bars get looser, not tighter** — HU 125.38%, RO 104.14%, LT 100.36%,
   HR 99.58%, LV 97.11%. §4's constant-predictor reference is the cheap defence, and BG
   already shows a causal constant clearing a 93.75% bar with no model.
5. **The incumbent will read `Not measured` for all 33.** That is by construction and is
   correct; a run that reports `n=0` cells instead has the wrong gate basis registered.
6. **Solar is still blocked on ABL-379.** `evaluate_solar_retrain.py` has no scope machinery
   and cannot gate BG or CH solar at all.

---

## 9. Recommendation

1. **BG wind_onshore** — accept as a clean pre-registered PASS with the §3 and §5 caveats
   attached. Evidence to the CEO; promotion, if any, is CEO-to-Board.
2. **CH wind_onshore** — reported, gates nothing, must not be counted in a tranche pass
   tally. §4 is the durable output of having built it.
3. **Adopt the constant-predictor reference** for the remaining 33 (§4). Free, and it
   separates BG's pass from CH's.
4. **Escalate the TSO result** (§5). Three of four pairs measured say a vintage-faithful TSO
   feature outranks model work for this programme. That is a sequencing decision and a
   probable ingest dependency, so it is the CEO's.
5. **Update the tranche sizing note** (§6): ≈60 s/pair is the median, hourly is not cheaper,
   and tranche granularity costs ~13 s per invocation.

Nothing here promotes a model, changes serving, or writes to the replica.
