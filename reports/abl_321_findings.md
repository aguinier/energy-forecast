# ABL-321 — training source for individual renewable types: findings

Analysis companion to the generated A/B tables in
`reports/abl_321_source_switch.md` (window 1) and
`reports/abl_321_source_switch_w2.md` (window 2). The pre-registered protocol
and decision rule are in `experiments/ABL321/protocol.md`, both registered
before the numbers they govern were read.

**Protocol.** Replica `C:\Code\able\data\energy_dashboard.db` (9,432,453,120
bytes), read 2026-08-12, opened `file:...?mode=ro` with `uri=True`. No write of
any kind touched the replica or any sidecar. Every census count below is a
census, not a sample; where a number is a model result it says so, with window,
n and baseline.

**Contamination touching these windows.** ABL-67 is net-position-only;
ABL-109/111 are load-only; ABL-71's known wrong-write modes are load and net
position. None of them is a proof that solar/wind ingest is clean — this is a
provenance caveat, not a clean bill of health. The FR gap in §2 is not covered
by any of them and was first filed in ABL-318 §3.

---

## 1. What the decision actually is

The question is not "is `energy_generation` better at predicting?" The case for
the switch is the census evidence in §2: history, NULL-vs-0, duplicates and
coverage. The backtest's job is narrower and is the one the acceptance criteria
set — **catch a regression in the four countries that already serve**. It is a
non-inferiority check, and it is reported as one.

## 2. The census: four arguments, one exception

### History

Across the 49 trainable country/stream pairs (ABL-318):

| | median usable history | pairs reaching 365 d |
|---|---:|---:|
| `energy_generation` | **2,049 d** | **49 / 49** |
| `energy_renewable` | 277 d | **10 / 49** |

For the four countries that already serve, `energy_renewable` begins:

| country | `energy_renewable` starts | days to 2026-08-12 | `energy_generation` starts |
|---|---|---:|---|
| AT | 2025-11-07 23:00 | 278 | 2021-01-01 |
| BE | 2024-01-01 00:00 | 954 | 2021-01-01 |
| DE | 2025-09-08 22:00 | 338 | 2021-01-01 |
| FR | 2023-01-01 00:00 | 1,319 | 2021-01-01 |

**AT and DE cannot be retrained on a full seasonal cycle from
`energy_renewable`.** DE misses 365 d by 27 days; AT by 87. That is not a
hypothetical about the 39 unbuilt pairs — it constrains two of the four models
already serving.

### NULL vs 0

`energy_generation.wind_offshore_mw` is NULL in every row for 15 countries — the
honest encoding of "the TSO does not report this". The same 15 read **100.0 %
exactly 0.0** in `energy_renewable`, whose columns are `REAL DEFAULT 0` and
whose mapper initialises each to 0.0 before checking the response (ABL-188). A
trainer pointed there receives a complete, non-null, perfectly valid all-zero
series for a country with no offshore fleet, and nothing in the read path
distinguishes it from a measurement.

Same shape for `biomass`: **CH, GR, SE** are 100 % NULL in `energy_generation`
and zero-filled in `energy_renewable`. Under the switch those three correctly
become "not reported" (empty frame) rather than a fabricated flat-zero series.
That is the intended behaviour and it is also a coverage change, so it is
flagged rather than buried.

### Duplicate instants

`energy_renewable`'s UNIQUE index is on `(country_code, timestamp_utc)` as a
*string*, so one instant is storable under several spellings: **78,510
duplicate-instant rows across all 24 countries, 5,425 carrying disagreeing
values**. `energy_generation` has zero. Nothing downstream chooses between the
two answers.

### Coverage — and the one exception

Census over 2025-10-01 → 2026-08-11 (the era where both tables have broad
coverage), distinct hourly instants, all 34 countries in `energy_generation`:

| | hours |
|---|---:|
| in `energy_generation`, not in `energy_renewable` | **24,694** |
| in `energy_renewable`, not in `energy_generation` | **586** |

42:1 for `energy_generation`. Of the 586, **518 are FR** — the
2026-06-30 23:45 → 2026-07-22 14:15 gap filed as new in ABL-318 §3 — and
`energy_renewable` has real data across it (FR solar daily maxima 13.7–21.8 GW,
wind_onshore mean 3,732 MW, 0.0 % zeros). The remaining 68 are MD (50) and CY
(18), neither in the 24 supported countries.

So `energy_generation` does **not** strictly dominate, and the exception lands
inside the window this backtest scores. That is handled by common-row scoring
(§4), and the lost coverage is reported as its own consequence rather than
absorbed into a metric.

I deliberately did **not** backfill `energy_generation` from `energy_renewable`
for FR. Mixing sources inside one series would put a zero-fillable segment
inside an otherwise NULL-honest one with nothing recording which rows came from
where. Closing that gap is an ingest re-fetch, not a training-source trick.

## 3. Two things the issue's own description got wrong

I wrote the issue; both corrections are to my own text.

**"The `column_map` transfers unchanged" is false for `hydro_total`.** It is
`hydro_run_mw + hydro_reservoir_mw`, and SQL's `+` propagates NULL. For **9 of
the 24 supported countries exactly one component is 100 % NULL in
`energy_generation`** — BE, EE, FI, LT, LV, NL, SI report run-of-river and never
reservoir; GR and SE report reservoir and never run-of-river. A literal column
swap returns NULL for every row of all nine and erases them. `energy_renewable`
hides this by zero-filling the absent component. The diff uses NULL-aware
addition: sum the reported components, return NULL only when *both* are absent.
Pinned by `tests/test_renewable_training_source.py`.

**Dropping NULL rows is not optional.** Without it a not-reported stream still
arrives as a feature-complete zero series and the switch buys nothing. That is
acceptance criterion 3, and it is the property the tests pin hardest.

## 4. The two tables disagree in more ways than the issue claims

The issue frames `energy_renewable` as zero-filling. Measured per served pair on
shared hourly instants, that is only one of at least three distinct classes, and
it is not the largest:

| class | example | scale |
|---|---|---|
| zero-fill (ABL-188) | BE wind_offshore: `energy_renewable` reads exactly 0.0 where `energy_generation` reads materially non-zero | 274 h in the fit window, **91 h in the scoring window** |
| frozen low value | AT wind_onshore reads 4.0 MW (63 h) or 8.0 MW (13 h); `energy_generation` contradicts ~8 of those, up to 2,476 MW. Longest contiguous low run **18 h — under `exclude_suspect_constant_runs`' 24 h minimum**, so it is invisible to the guard | small but real |
| level/revision divergence on non-zero values | BE wind_onshore: `energy_renewable` sits systematically ~190 MW above `energy_generation` on 2,380 of 4,580 fit-window hours (means 809.6 vs 710.5), converging by May; AT solar differs on 956 h with near-symmetric sign | **the largest class**, and not previously named |

The third class is not zero-fill and is not covered by ABL-188. Which table is
right for it is not established here, and I am not claiming it is. It matters
because it — not zero-fill — dominates what the two arms actually train on for
AT and BE.

**BE wind_offshore is the only served pair whose *scoring* window differs
between the tables** (91 materially disagreeing hours, all of them
`energy_renewable`-zero against `energy_generation`-non-zero). Every other
served pair has identical truth under both definitions in the scoring window,
which is why the primary and secondary truth columns agree everywhere else — the
switch is not grading its own homework.

## 5. A merge precondition the issue did not name

`src/forecaster.py:771` constructs `RenewableFeatureBuilder` with no explicit
source, so it takes `db.py`'s default. Flipping that default therefore changes
the lag and rolling features fed to the **four already-frozen serving
artifacts** the moment it merges — without retraining them. That is train/serve
skew, not a training-data change.

Worse, it would be silent: the served artifacts
(`models/AT/solar/production/model.joblib`,
`models/BE/wind_offshore/production/model.joblib`, …) carry
`algorithm`, `feature_columns`, `model_version`, `training_metrics`, `saved_at`
— and **no training-source field at all**. Nothing in the artifact or the loader
could detect the mismatch.

This is the same failure shape as the xgboost-intercept trap in CLAUDE.md: no
crash, no failing test, just a model that quietly predicts from inputs it was
not fitted on. Two recommendations follow, neither of which is mine to make:

1. The switch must not reach serving unpaired with a retrain of the ten served
   country/stream pairs. Either land it together with the retrain, or have the
   serving path pin its source until the retrain lands.
2. `save_model` should record the training source and the loader should refuse a
   mismatch — the pattern `src/xgboost_artifact_guard.py` already establishes
   for the interpreter. That is Founding Engineer territory (serving path), so
   it is a proposal, not a patch in this branch.

## 6. VERDICT — criterion 2 fails. The switch does not land.

**Three of the ten served pairs are materially worse under `energy_generation`
in the registered decision window.** The pre-registered rule fires, and the CEO's
approval made criterion 2 a hard stop. So the source switch is withheld, and this
regression is the deliverable.

Window 1 (registered), fit 2026-01-14 → 2026-07-11, score 2026-07-11 → 2026-08-10,
D+2 bands 24–64 h, primary truth `energy_generation`, common rows only:

| pair | n | before (ren) | after (gen) | relative | vs D-7 | verdict |
|---|---:|---:|---:|---:|---:|:---|
| AT solar | 1,950 | 12.89% | 13.44% | **+4.3%** | +49.4 / +47.2 | **after WORSE** |
| DE wind_onshore | 1,950 | 51.63% | 53.50% | **+3.6%** | +33.7 / +31.3 | **after WORSE** |
| BE wind_onshore | 1,950 | 46.56% | 47.81% | **+2.7%** | +43.6 / +42.1 | **after WORSE** |
| BE solar | 1,950 | 16.76% | 16.99% | +1.4% | +49.2 / +48.5 | no material change |
| DE solar | 1,950 | 13.52% | 13.58% | +0.5% | +44.0 / +43.8 | no material change |
| FR solar | 287 | 15.05% | 15.00% | −0.3% | +35.5 / +35.8 | no material change |
| AT wind_onshore | 1,950 | 72.32% | 70.43% | −2.6% | +31.5 / +33.3 | after better |
| BE wind_offshore | 1,950 | 77.54% | 75.14% | −3.1% | +27.0 / +29.2 | after better |
| FR wind_offshore | 287 | 44.04% | 39.72% | −9.8% | +18.0 / +26.0 | after better |
| FR wind_onshore | 287 | 39.27% | 32.02% | −18.5% | +29.7 / +42.7 | after better |

Four pairs improve and three regress, but the rule is not a vote — it is a
non-inferiority check on models that already serve, and three of them lose.

**The result is not an artifact of the truth definition.** In window 1 the gate
truth is *identical* between the two tables for nine of ten pairs (verified
hour-by-hour: AT/BE/DE/FR solar and wind_onshore all differ on 0 hours in the
scoring window). The secondary-truth table is therefore numerically identical to
the primary one except BE wind_offshore, where it moves the third decimal. Both
arms are scored on the same numbers; the only thing that differs is what they
trained on. The switch is not grading its own homework here.

**Do not lean on FR.** Its two largest wins (−18.5%, −9.8%) sit on **n = 287
against 1,950** for every other pair, because the ABL-323 gap removes 279 of 720
gate hours from the after arm and common-row scoring then drops them from both.
FR's window-1 cells are 15% of the evidence the other pairs carry, and I would
not defend a decision resting on them.

### Window 2 does not rescue it, and does not test what I registered it to test

I have to report a failure of my own experimental design. Amendment 1 registered
the winter holdout on the claim that it "runs the seasonal bias the other way".
**It does not.** Measured after the fact:

| window | solar fit-window level (AT) | gate truth level (AT) | level jump |
|---|---:|---:|---:|
| 1 | 884.9 MW | 1,410.1 MW | 1.6× |
| 2 | 127.4 MW | 544.8 MW | **4.3×** |

Both windows fit low and score high; window 2 does it *harder*. Fitting
Nov 21 → Feb 15 and scoring Feb 15 → Mar 17 does not invert the seasonal
gradient, it steepens it. So window 2 cannot discriminate the level-artifact
hypothesis, and I am not going to pretend it confirms anything about it.

Worse for its usefulness, **window 2's four solar cells fail the protocol's own
baseline precondition** — "both arms must also beat D-7 to be worth discussing at
all". Skill against seasonal-naive D-7 is negative for both arms in all four:
AT −44.2 / −51.6, BE −15.9 / −15.8, DE −41.3 / −22.9, FR −32.2 / −31.3. A
30-day early-spring solar holdout where neither arm beats D-7 is not a window
that can adjudicate solar. So AT solar's window-2 "+5.1% worse" is **inadmissible
under the registered rules**, and so is DE solar's headline −13.0% improvement.
Window 2's six wind cells do clear D-7 and are admissible; none of them is
materially worse, and AT wind_onshore improves in both windows (−2.6%, −3.8%).

Two further window-2 caveats, both discovered by measurement rather than
assumption: its gate truth is **not** shared for AT solar (39.2% of hours differ)
or DE solar (99.7%), unlike window 1 — though the secondary-truth column moves
those conclusions by ≤0.2 pp, so the truth choice is not what drives them.

### What the regression actually is, as far as I can show

Not a data-quality defect in `energy_generation`. The pattern across the solar
pairs is monotone in the **fit-window level gap between the arms**:

| pair | arm A fit mean | arm B fit mean | A/B | relative WAPE change |
|---|---:|---:|---:|---:|
| AT solar | 899.8 | 884.9 | 1.017 | **+4.3%** |
| BE solar | 1,557.0 | 1,557.0 | 1.000 | +1.4% |
| DE solar | 10,936.2 | 10,992.8 | 0.995 | +0.5% |
| FR solar | 4,208.9 | 4,208.9 | 1.000 | −0.3% |

Both arms under-predict a summer gate from a winter-weighted fit; the arm whose
training series sits higher under-predicts less. AT is the only pair where arm A
trains materially higher, and AT is the pair with the largest regression. Where
the two arms train at an identical level the effect is ~0.

**And the reason AT's fit window is seasonally truncated is arm A itself.** AT's
`energy_renewable` history starts 2025-11-07, so the A/B cannot fit earlier than
that without arm A having no data. The protocol holds the fit window fixed across
arms to isolate the source variable — which is correct for attribution, and which
also means **this backtest is structurally incapable of measuring the switch's
main benefit**: that arm B has 2,049 days of history where arm A has 278. The
comparison penalises arm B for a constraint that exists only because arm A is in
the room.

That is a limitation of the test I designed, not a defence of the result. Under
the rules as registered, the switch does not land.

### What I recommend, and what is not mine to decide

The census case in §2 is untouched by this — history, NULL-vs-0, duplicates and
coverage are not things this backtest measures, and nothing here contradicts
them. What the backtest establishes is narrower and real: **on an identical short
fit window, `energy_renewable` is the better training series for AT solar and for
BE/DE wind_onshore, and the largest such gap is explained by level rather than by
data quality.**

The test that would actually settle it is one the acceptance criteria did not
ask for, and I am proposing rather than running it: fit each arm on **the history
it actually has** — arm A from 2025-11-07, arm B from 2021-01-01 — and score on
the same rows. That is the "what would ship" comparison rather than the
one-variable comparison, and it is the only way to put arm B's seasonal coverage
on the board. It needs its own pre-registration before anyone sees a number, and
it needs a CEO call, because it changes the question criterion 2 asked.

Three things I am **not** doing on my own authority: landing the switch against
a failed criterion, reinterpreting the threshold after seeing the numbers, or
declaring AT solar's regression benign because I have a story for it. The story
is a hypothesis with one supporting table, not a result.

## 7. The duplicate-instant defect is worse than the issue said

The issue filed duplicate instants as nondeterminism: `energy_renewable`'s
UNIQUE index is on `(country_code, timestamp_utc)` as a *string*, so one instant
is storable under several spellings, and the training set's content depends on
which row the query returns. That is true, and it is not the whole cost.

**The duplicates make the pre-ABL-321 source unbacktestable in winter.** They
survive into `RenewableFeatureBuilder`'s index, where a same-hour lag lookup
resolves to a two-element `Series` instead of a scalar and `float()` raises.
Window 2's first attempt died on AT/solar for exactly this reason. Measured over
each window's builder span, all ten served pairs, `energy_renewable`:

| window | builder span | duplicate rows per pair | pairs with disagreeing instants |
|---|---|---|---|
| 1 | 2025-12-31 → 2026-08-10 | **0**, every pair | 0 / 10 |
| 2 | 2025-11-07 → 2026-03-17 | 299 – 2,259 | **6 / 10** (AT wind_onshore worst, 700) |

Two consequences worth stating plainly:

- **Window 1's numbers are untouched by the fix.** The collapse branch never
  fires over its span, so the two windows remain comparable and window 1 did not
  need re-running.
- **The defect is concentrated in exactly the era where `energy_renewable` is
  the only history AT and DE have.** AT's `energy_renewable` starts 2025-11-07
  and the duplicates begin 2025-11-17. So the window where arm A has any data at
  all is the window where arm A is most corrupt. That is not an argument the
  backtest can make — it is a census fact, and it points the same way as §2.

The loader now collapses agreeing spellings to their shared value and nulls
disagreeing ones as unadjudicated, the same treatment `exclude_suspect_constant_runs`
already applies. Picking a spelling would make the training set depend on which
row the query returned; averaging would invent a value the TSO never published.
It is a **no-op for `energy_generation`**, which has zero duplicates, and that is
pinned by its own test so the fix cannot be read as tilting the A/B toward the
arm the switch moves to.

## 8. ABL-326 cross-check: the corrected gap baseline does not move this backtest

ABL-326 refuted the ABL-318 claim — mine — that FR was the only country with a
`energy_generation` gap over 7 days. Seven countries carry one; FR ranks 9th, not
1st. Since criterion 2 rests on the holdout windows being uncontaminated, I
re-ran that census scoped to the four served countries, using ABL-326's own
measure and its caution that **a NULL run and an absent row both mean "not
reported" and neither is a zero**.

`energy_generation`, missing hours against each window's full hourly grid:

| pair | W1 fit (4,272 h) | W1 gate (720 h) | W2 fit (2,064 h) | W2 gate (720 h) |
|---|---:|---:|---:|---:|
| AT solar, AT wind_onshore | 0 | 0 | 0 | 0 |
| BE solar, wind_onshore, wind_offshore | 0 | 0 | 0 | 0 |
| DE solar, DE wind_onshore | 0 | 0 | 0 | 0 |
| FR solar, wind_onshore, wind_offshore | 240 | **279** | 30 | **0** |

None of the six countries ABL-326 surfaced (BA, CY, IE, MD, ME, MK) is served, so
none is in criterion 2's scope. The only hole in a scoring window is FR's, which
is ABL-323's, already handled by common-row scoring.

**One thing I did not expect: in window 2 the coverage asymmetry reverses.**
`energy_generation` is complete for all ten pairs including FR, while
`energy_renewable` has a 27–28 h hole at 2026-02-15 14:00 in every one of them.
So in window 2 the *before* arm is the one losing coverage, and FR is measurable
there in a way it is not in window 1, where its after arm loses 279 of 720 gate
hours. The winter holdout is therefore worth more than the amendment registered
it for.

Where ABL-326 does bite is **ABL-316 sizing**, not this issue: the ABL-318
verdict table qualifies a pair on history *span*, and a span does not see an
interior hole. CY `wind_onshore` clears a 365-day bar while carrying a 995-day
hole. The tranche issues need a per-pair gap screen, not just a span check.

## 9. Operational note

`energy-forecast/.env` sets `DATABASE_PATH=C:/Code/energy-data-gathering/energy_dashboard.db`
— a path that does not exist, one directory away from the 3.0 GB stale decoy
CLAUDE.md warns about. It fails loudly (`sqlite3.OperationalError`) rather than
silently reading the decoy, so nothing here is contaminated, but every run must
pass `ENERGY_DB_PATH` explicitly. `.env` is gitignored and machine-local, so
this is a note, not a patch.

## 10. Disposition — the CEO decision, and what it did to the question

Recorded 2026-08-12 after the CEO's ruling on this issue. §1–§9 above are
unrevised; this section says what happened to them.

### The switch does not land as a global flip — and the reason is not §6

The CEO rejected the global switch on a ground I had not made explicit, and it
is the stronger one. `RENEWABLE_TYPE_SOURCE_TABLE` is read at **serve** time,
not only at training time: `src/forecaster.py:771` builds a
`RenewableFeatureBuilder` per forecast call, which reaches
`src/wind_features.py:233` -> `_load_actuals_series` -> `load_renewable_type_data`
-> `src/db.py:401`. So flipping the constant alone would feed
`energy_generation` features to ten artifacts **fitted on `energy_renewable`** —
a third state that neither arm of my backtest measured:

| state | trained on | served features from | measured |
|---|---|---|---|
| today | `energy_renewable` | `energy_renewable` | yes — arm A |
| arm B | `energy_generation` | `energy_generation` | yes |
| flip the constant alone | `energy_renewable` | `energy_generation` | **no** |

My backtest licenses neither the flip nor "accept the regression and land". It
only ever spoke to rows 1 and 2. Train/serve skew is the correct objection and
it supersedes the regression as the reason.

### The source stops being global — ABL-331

One constant forced one answer onto 49 pairs that do not have one answer. The
CEO's resolution is to make the training source a **per-artifact property**:
recorded in `model_data` at save, read back in `Forecaster.load`, threaded to
the builder at serve, defaulting to `energy_renewable` when the key is absent so
every existing artifact stays bit-identical. PR #12 already threads `source=`
down through `_load_actuals_series`; the missing link is `Forecaster` -> builder.

That is a serving-path and artifact-format change, so it is the Founding
Engineer's, not mine — filed as **ABL-331**, which becomes the new gate for
ABL-316. Consequences:

- The **39 pairs with no model** train and serve on `energy_generation`
  immediately. No incumbent exists to regress against; their gate is
  seasonal-naive D-7, which is ABL-316's design already. The §2 census case for
  them is unrefuted.
- The **10 serving pairs** keep `energy_renewable` until each is individually
  retrained and gate-read. §6's three regressions stay confined to the pairs
  they were measured on.
- **ABL-321 is released from gating duty.** ABL-316's scope question — 10 pairs
  or 49 — is answered 49, without touching the ten that already work.

### What landed

PR #12 merged as `8c83d1a` with `RENEWABLE_TYPE_SOURCE_TABLE = 'energy_renewable'`
**unchanged**, so serving behaviour is bit-identical to before. What shipped is
the part that improves both sources whichever one a given pair uses: the
`source=` parameter, the NULL-drop, the NULL-aware `hydro_total` sum, the
duplicate-instant collapse (§7), the harness, and criterion 3's empty-frame
test. 56 tests pass on `.venv` Python 3.14.3.

Criterion 1 is therefore **not** met as written and deliberately so — the CEO's
decision replaces "point the loader at `energy_generation`" with "make the
source per-artifact". Criteria 2, 3 and 4 are met.

### What is registered but not run

`experiments/ABL321/protocol.md` **Amendment 2** — the "what would ship"
comparison §6 proposed: fit each arm on the history its own source actually
provides, score on window 1's registered rows, all ten pairs. Registered in full
before any number, including a falsifiable primary prediction (Spearman's rho
between B/A history ratio and arm B's relative WAPE change, predicted negative)
and explicit fixes for Amendment 1's two self-reported design failures.

It no longer gates anything. It decides one narrower thing: whether the ten
incumbents should eventually migrate, pair by pair. Registered order:
**ABL-322 -> ABL-332 -> Amendment 2.**

### One confound checked and retired

I suspected ABL-332 (the hourly feature builder discarding `:15/:30/:45`) might
have handicapped arm B asymmetrically, which would have undermined §6 —
DE wind_onshore is both 15-minute and one of the three regressions. Measured
read-only on the replica over window 1's span, the resolution is **symmetric**:
AT, DE and FR carry 15-minute rows in *both* tables and BE is hourly in both. So
the defect costs both arms the same three-quarters of their samples, and §6's
contrast is not contaminated by it. Both arms' absolute WAPEs are pessimistic
for AT/DE/FR, and ABL-332's fix will move both — which is why Amendment 2 is
sequenced after it.
