# ABL-437 — Pre-registered amendment: how the causal gate references are levelled

**Status: REGISTRATION. Frozen before the amended ladder was read against any pair.**
Machine record: `experiments/ABL437/config.json`. Implementation:
`src/evaluation/model_free_reference.py`, `src/evaluation/gate_grading.py`, the
`CAUSAL_LEVELLING` table in both gate harnesses. Guard:
`tests/test_abl437_causal_levelling.py`.

The amended read of the pairs already graded is a **separate document and a
separate commit** — `reports/abl_437_causal_levelling_reread.md`. That ordering is
the binding constraint on this issue and it is checkable in git.

---

## 1. The defect, and what it is not

`constant_causal` and `climatology_causal` (ABL-389) are levelled on the **fit**
window and scored on the **gate** window. ABL-348 freezes those windows as

| window | span |
|---|---|
| fit | 2026-01-14 → 2026-07-11 (178 d, winter → summer) |
| gate | 2026-07-11 → 2026-08-10 (30 d, high summer) |

So the "causal constant" is a winter-and-spring average scored against high
summer. On a seasonal series it is not an estimate of the gate window's level at
all, and G2 (level) and G3 (shape) are registered on exactly those two
references — `_SCORED_CONDITIONS` in `gate_grading.py`, now
`scored_conditions()`.

Worst band per pair, as the gap between the causal constant and the
correctly-levelled oracle constant, re-derived by import from every committed
`results_*.json` (`level_inflation`, over all 37 pairs and 137 cells):

| pair | band | `constant_causal` WAPE | `constant_oracle` WAPE | inflation |
|---|---|---:|---:|---:|
| NL `wind_onshore` | 24-36h | 225.54% | 73.85% | **205.4%** |
| PT `wind_onshore` | 24-36h | 101.38% | 50.17% | 102.1% |
| CH `wind_onshore` | 24-36h | 79.07% | 40.29% | 96.2% |
| IT `wind_onshore` | 24-36h | 92.00% | 52.32% | 75.8% |
| ES `wind_onshore` | 24-36h | 62.07% | 41.49% | 49.6% |
| BG `wind_onshore` | 48-64h | 86.90% | 60.65% | 43.3% |
| **HU `wind_onshore`** | 24-36h | 103.14% | 72.13% | **43.0%** |
| **NO `wind_onshore`** | 24-36h | 59.69% | 42.42% | **40.7%** |
| LT `wind_onshore` | 48-64h | 94.13% | 67.89% | 38.7% |
| HR `wind_onshore` | 48-64h | 89.33% | 64.96% | 37.5% |
| **RO `wind_onshore`** | 48-64h | 90.98% | 69.69% | **30.6%** |
| **SE `wind_onshore`** | 48-64h | 44.45% | 36.23% | **22.7%** |
| PL `wind_onshore` | 48-64h | 63.91% | 52.27% | 22.3% |
| CZ `wind_onshore` | 48-64h | 57.78% | 47.84% | 20.8% |
| **EE `wind_onshore`** | 24-36h | 73.02% | 62.24% | **17.3%** |
| GR / LV / FI `wind_onshore` | — | — | — | 3.9% / 3.9% / 2.2% |

**The ten rows the issue lists reproduce to the digit; there are fifteen.** HU,
NO, RO, SE and EE — bolded — are above 17% and are absent from that table, so the
amendment's reach on wind is **15 of 18 onshore pairs**, not 10. NO is the pair
ABL-406 found passing 3/3 while anti-correlated with its own target, and RO is
ABL-417's original mis-levelling finding; both were already carrying a qualifier,
and both are also carrying an inflated reference.

Solar spans **−1.2% to 7.8%** across its 19 pairs, and the reason is structural
rather than lucky: a flat line's WAPE on solar is dominated by the diurnal cycle,
so the level it sits at barely moves the number. **The constant's mis-levelling
is a wind problem.** The *climatology's* is not necessarily — see §6.

> **Why a negative is possible at all**, since a median minimises `sum|a - c|`:
> `constant_oracle` is the median of the **whole gate window**, while a cell's
> WAPE is scored on that cell's **band subset**. The oracle is therefore a
> hindsight bound over the window rather than a per-cell optimum, and on EE solar
> 48-64h the fit-window mean lands marginally better on that subset (80.42% vs
> 81.40%). It is a −1.2% effect on one cell and it does not disturb the
> diagnostic's use, but a reader comparing the two columns should know the oracle
> is not per-cell optimal.

**What this is not.** It is not a claim that any challenger is worse than
published. Every challenger WAPE, every D-7 comparison and every gate PASS stands
exactly as recorded. It is a claim about a **reference**: a flat line at the wrong
level is a weak test of whether a model predicts the level, and G2 and G3 have
been reading one.

This is the third instance of one pattern — ABL-406 (bar weakness), ABL-417
(mis-levelled `constant_causal` on RO), ABL-435 §6 (BG/CH) — and it is the reason
this is an amendment rather than a third finding.

---

## 2. The two candidate forms

### (a) A not-evaluable flag — **rejected**

Flag G2/G3 `SCOPE_NOT_EVALUABLE` where the `constant_causal / constant_oracle`
level ratio falls outside a registered band, on the ABL-421 precedent and the
net-position gate's `INCOMPLETE` rule.

Four reasons, in the order that decided it:

1. **The band cannot separate the two cases it would be used to separate.** It
   is keyed on the *reference's* mis-levelling, which is a property of the
   country's seasonality, not of the challenger. BG sits at 43% inflation and CH
   at 96%; any band that abstains on CH abstains on BG, and BG *beats* the
   correctly-levelled references where CH loses to them. The CEO's steer named
   this and it holds up on the numbers.
2. **It converts a measurable question into an abstention.** The mis-levelling is
   a number, and the number is what a reader needs. Abstention throws it away and
   leaves nothing in its place.
3. **It puts a hole in the ladder exactly where the fleet is.** Fifteen of
   eighteen `wind_onshore` pairs are above 17% inflation; on any band tight
   enough to catch NL, most of the ABL-316 wind promotion set stops having a G2
   or a G3 at all.
4. **The band would itself be an unregistered choice made after seeing the
   ratios.** Every candidate threshold is visibly a threshold *about* the
   fifteen pairs in §1, which is the shape of the thing this apparatus exists to
   prevent.

**What survives from it.** The ratio is now printed per cell, per causal
reference, as a `level inflation` column (`model_free_reference.level_inflation`).
Form (a)'s information is retained; only its abstention is dropped.

### (b) A correctly-levelled causal reference — **adopted**

Keep both conditions on the ladder and re-level the reference they read:

> `constant_causal_28d` and `climatology_causal_28d` — the same flat line and the
> same hour-of-day mean, over the **28 days ending at the row's own
> `generated_at`**.

Three reasons:

1. **It measures the thing.** G2 keeps asking "does this model predict the
   level?", now against a level estimate a forecaster could actually have formed
   at issue time.
2. **It is strictly causal by construction, not by a new argument.** The window
   is anchored at `generated_at.floor("h")`, inclusive, spanning `28*24 - 1`
   hours back, over the same ABL-188-filtered series the gate actuals and the D-7
   baseline come from — character for character the bound
   `wind_features._rolling_features` already applies to
   `target_value_roll_168h_mean`, which is **one of the challenger's own 24 input
   features**. The reference therefore uses no information the challenger did not
   have, and `test_the_anchor_matches_the_builders_own_rolling_window_bound`
   reads that bound out of `wind_features.py`'s source so the two cannot drift.
3. **It discards nothing.** All four ABL-389 references keep their names, their
   definitions and every value already published, and are still reported. The
   record gains columns; it loses none.

**Recommendation: (b).** I agree with the CEO's steer, and the argument that
decided it is (a)'s reason 1 rather than the general preference — (a) is not
merely lossy, it does not discriminate.

---

## 3. Why the oracle references stay off the ladder

Unchanged from ABL-389 and from the ABL-435 pack, which did not propose otherwise:
an oracle is levelled on the gate window itself and is not causally available, so
losing to one **bounds what a verdict means** rather than voiding it. Putting one
on the ladder would replace a strawman with an unattainable bar, which is the same
error with the sign flipped. `test_no_oracle_is_on_either_ladder` pins it for both
levellings.

The oracle constant is used here for exactly one thing: as the *denominator* of
the inflation diagnostic, which is a statement about a reference and not a
criterion for a challenger.

---

## 4. Why 28 days, and why one window for both forms

The parameter was fixed on the grounds below and committed before the amended
ladder was read against any pair. **It was not chosen from a sweep of candidate
windows against the grades each would produce, and no such sweep was run.**

- **The constant and the climatology must share one window.** A constant is a
  climatology with one bucket, and the pair's whole reading — the gap between
  them is how much of the series is forced diurnal structure (ABL-389) — breaks
  if they are levelled differently.
- **A shared window has to serve the climatology.** It needs enough samples per
  hour-of-day bucket to be a level rather than noise: 28 days gives 28 per
  bucket, 7 would give 7. A noisy climatology is a *weak* reference, which
  re-creates the strawman defect in a new place.
- **Four whole weeks** balances day-of-week composition.
- **Short enough to sit inside one season**, which is the point, and long enough
  that one storm week does not set the level.
- **Available for every gate row.** The earliest gate issue instant is
  2026-07-09 07:00 and 28 days before it is 2026-06-11, inside ABL-348's fit
  window — so no run needs data it does not already load, and the trailing window
  never reaches outside the series the builder holds.

---

## 5. What the amendment does not change

| | |
|---|---|
| **G1 and the registered bar** | Seasonal-naive D-7, unchanged, under both levellings. `test_g1_is_the_registered_bar_under_every_levelling`. |
| **ABL-348** | Windows, bands, metric, baseline, minimum n, source table and `not_evaluable` are untouched, so `voids_this_registration` is not triggered — the same reading ABL-418 took for a report-only ladder. |
| **The ladder's rules** | Letters, the readability floor, the worst-band rule, `U` over `C`, and "a condition that cannot be evaluated is not satisfied" are all unchanged. `test_the_two_levellings_grade_identically_on_identical_numbers` asserts it directly: given two reference pairs carrying the same numbers, every case grades identically under either levelling. |
| **Published letters** | Every scope with a committed record is pinned to `fit_window` in `CAUSAL_LEVELLING`, so a re-run reproduces the read it published. `reports/abl_418_retro_grade.md` regenerates **byte-identical** at this commit (`c74c9e9209f287395f87facab8d23fc82ba7dc10`), and `scripts/abl418_retro_grade.py` is pinned for the same reason. |
| **Any promotion** | Nothing here promotes anything or recommends promoting anything. |

**The default direction is the opposite of `SCOPE_FEATURES`', deliberately.** A
scope absent from `CAUSAL_LEVELLING` grades on `trailing_28d`. Inheriting the old
reference silently would give a new tranche the inflated reference on pairs nobody
has looked at yet — the defect itself. The cost is that an absence can no longer
reproduce an old read, so `test_every_published_scope_pins_its_levelling` derives
the published set from `SCOPE_OUTPUTS` **and git** and requires an explicit pin for
each, on the ABL-404 precedent. The table is **not** added to
`check_registration_tables`: three PRs were open at registration time, one already
`CONFLICTING` on that same call, and adding a required table raises at import for
every branch in flight.

---

## 6. Limits, stated before the re-read

- **A trailing window converges; it does not teleport.** On a step change at the
  gate boundary the reference still carries the old level on day 1. Measured on a
  deliberately worst-case fixture — a pure step, which real seasonality is not —
  it halves the reference's WAPE (300% → 160%) rather than removing the error.
  On ABL-348's windows its *starting* position is the last 28 days of the fit
  window, i.e. mid-June to mid-July, which is already the gate season, so the real
  residual should be much smaller. **The residual is reported per cell rather than
  asserted away**, which is where form (a)'s diagnostic went.
- **The amendment can make G3 materially harder on solar**, and that is the
  intended direction rather than a side effect: `climatology_causal` there is a
  January-to-July hour-of-day mean scored on high summer, and a trailing-28-day
  one is a much better predictor of a summer day. The constant's 0–8% figure in
  §1 says nothing about this.
- **One seed, one 30-day holdout.** Unchanged from every ABL-348 read, and a gap
  below the ABL-385 floor (7.51% wind, 10.65% solar) is not readable at one seed
  under either levelling.
- **Contamination.** This amendment touches no ingest, no source table and no
  window, so it can neither introduce nor repair ABL-71 / ABL-67 / ABL-109 /
  ABL-111. Per-pair statements stay with each read.

---

## 7. What happens to the pairs already graded

**No committed record is edited or regenerated.** The amended ladder read is a new
document, on the ABL-418 retro-grade precedent — arithmetic over the stored
results files plus the trailing references recomputed from the replica's target
series on the same rows, no refit and no new model.

`reports/abl_437_causal_levelling_reread.md` carries it, in the commit after this
one, and states which already-graded pairs the amendment moves and in which
direction. Registered here, before that read exists, as a prediction to be checked
against it:

- solar G2 should move little (0–8% inflation, and the constant is a formality
  there in any case);
- solar G3 may move materially, in the tightening direction;
- wind is where G2 should move, and the ten pairs in §1 are where to look first.

That is a prediction, not a result. What the re-read measures is what gets
reported, including where it contradicts this paragraph.

---

## 8. Evidence hygiene

- `experiments/` and `reports/` carry **zero deletions** in this commit; the only
  new paths are `experiments/ABL437/config.json` and this file.
- ABL-380's and ABL-435's committed records are byte-unchanged, proven by blob
  hash in the re-read pack rather than by inspection.
- `reports/abl_418_retro_grade.json` is deliberately **not** regenerated. A re-run
  would add two additive keys per grade (`causal_levelling`,
  `level_inflation_pct`) and change no value and no letter; leaving the committed
  bytes alone keeps the whole published set unchanged in this commit.
- Suite at this commit: **1,174 passed, 0 failed** (`.venv` Python 3.14.3),
  counted from the "N passed" line and not from the exit status.
