# ABL-321 — pre-registered A/B protocol

Registered **before** any model was fitted or any metric read, so the decision
rule cannot be chosen after seeing the numbers.

## Question

Does moving `src/db.py:load_renewable_type_data` from `energy_renewable` to
`energy_generation` make any of the four already-serving countries materially
worse? This is a **non-inferiority check, not a superiority test** — the case
*for* the switch is the census evidence (history, NULL-vs-0, duplicates,
coverage), and the backtest's job is to catch a regression in what already
serves.

## Arms

| arm | source table |
|---|---|
| before | `energy_renewable` |
| after  | `energy_generation` |

The source sets the training target **and** every lag/rolling feature derived
from it. Nothing else differs: same `RenewableFeatureBuilder`, same eight
pre-registered vintage instants per target hour, same fit/gate split, same
algorithm, same baseline.

## Split

- Fit targets: 2026-01-14 → 2026-07-11 (exclusive)
- Scoring targets: 2026-07-11 → 2026-08-10 (exclusive), out-of-sample by target timestamp
- The ABL-253 registered split, reused unchanged.

## Pairs

The 10 country/stream pairs that already have a serving model: AT/BE/DE/FR
solar, AT/BE/DE/FR wind_onshore, BE/FR wind_offshore.

## Truth

Every cell is scored against **both** candidate truths on the identical rows:

- `energy_generation` — primary. NULL-preserving, zero duplicate instants, and
  the table ABL-188 used to adjudicate `energy_renewable`'s zeros as wrong.
- `energy_renewable` — secondary; what `src/evaluation/scorecard.py` scores the
  live models against today.

Scoring each arm against its own source would be circular. A conclusion is
reported as robust only where the two truths agree.

## Rows

Common rows only: (target, horizon band) pairs where both arms produced a
finite prediction and the truth is finite. Arm B loses FR coverage to the
`energy_generation` gap of 2026-06-30 23:45 → 2026-07-22 14:15 (ABL-318 §3);
letting each arm score its own row set would compare different holdouts. The
coverage loss is reported separately, not absorbed into the metric.

## Decision rule

- Metric: WAPE over the three primary D+2 bands (24-36h, 36-48h, 48-64h).
- **Materially worse := arm B's WAPE exceeds arm A's by more than 2.0 % relative.**
- If any of the four served countries is materially worse on the primary truth,
  the switch does not land, and that is the reported finding.
- Baseline: literal seasonal-naive D-7, rebuilt from each truth series. Both
  arms must also beat D-7 to be worth discussing at all.

## Out of scope

No promotion, no serving change, no registry change, no ingest change, no
replica write, no sidecar write.

---

## Amendment 1 — a second, winter holdout (registered before it was run)

Registered after reading window 1's AT/solar and BE/solar cells and **before**
fitting anything on window 2. The decision rule above is **not** changed; this
amendment tests whether the *measurement* is stable, which is a different
question from what the threshold should be.

**Why.** Window 1 fits on Jan–Jul and scores on Jul–Aug, so it trains on low
solar and scores on high solar. Both arms therefore under-predict, and the arm
whose training data sits at a higher level under-predicts less. Measured on the
fit window against the window-1 gate truth, arm A does sit higher for solar:

| pair | fit mean, arm A | fit mean, arm B | gate truth mean | A / gate | B / gate |
|---|---:|---:|---:|---:|---:|
| AT solar | 899.8 | 880.1 | 1,410.1 | 0.638 | 0.624 |
| BE solar | 1,557.0 | 1,550.5 | 2,164.5 | 0.719 | 0.716 |
| DE solar | 10,936.2 | 10,939.6 | 16,885.0 | 0.648 | 0.648 |

AT — the one pair beyond the material band — has the largest level gap of the
three, and DE — where the two arms sit at an identical level — has the smallest
WAPE difference (+0.5 %). That is what a level artifact looks like, not what a
data-quality effect looks like.

**The test.** A winter holdout runs the seasonal bias the other way: fit across
autumn/winter, score in winter. If arm B's deficit is a level artifact it should
shrink or reverse; if it is a property of the source it should persist.

- Fit targets: 2025-11-21 → 2026-02-15 (exclusive)
- Scoring targets: 2026-02-15 → 2026-03-17 (exclusive)
- Everything else unchanged.

Window 2's fit span is bounded by `energy_renewable` itself: AT's history starts
2025-11-07 and DE's 2025-09-08, so no earlier fit start is available to arm A.
The fit is therefore ~86 days against window 1's ~178 — **both arms are equally
handicapped**, so the A/B stays fair, but the absolute WAPEs are not comparable
across windows and no cross-window absolute claim is made.

**How the two windows combine.** Window 1 remains the registered decision
window. Window 2 can show that window 1's result is unstable; it cannot by
itself license the switch. Reported outcomes:

- worse in both windows → the switch does not land. Reported as the finding.
- worse in one, not the other → the evidence does not settle it on 30-day
  holdouts; the recommendation says so and does not pretend otherwise.
- not materially worse in either → non-inferiority holds on the evidence
  available, and the census arguments carry the decision.

---

## Amendment 2 — the "what would ship" comparison (registered before it was run)

Authorised by the CEO on ABL-321 (2026-08-12), with two amendments of their own,
both adopted below: **all ten pairs, not the three that regressed**, and **this
no longer gates ABL-316**. Registered before any model was fitted under it and
before any metric was read. Nothing in §6 of `reports/abl_321_findings.md` is
revised by this; that verdict stands on the window it was registered for.

### What changed, and why this is a different question

Window 1 holds the fit window fixed across arms to isolate the source variable.
That is correct for attribution and it is why §6's verdict is defensible — but
it makes the test **structurally incapable of measuring the switch's main
claimed benefit**, that `energy_generation` has years of history where
`energy_renewable` has months. Window 1 penalises arm B for a constraint that
exists only because arm A is in the room.

Amendment 2 asks the deployment question instead: **fit each arm on the history
its own source actually provides, and score on the identical registered rows.**

**This is deliberately a two-variable comparison.** Source quality and history
length move together and this design cannot separate them. That is the point —
the question is "which configuration would serve better", not "what does the
source do holding all else equal". I pre-commit now: **no result from this
amendment will be reported as attributing anything to data quality.** Window 1
is the attribution test; this one is the deployment test.

### Arms — per-arm fit start, measured on the replica before registration

Fit end, gate window, algorithm, vintages, feature builder, truth handling and
common-row rule are **window 1's, unchanged**. The only edit is the fit start,
which becomes per-arm and per-pair: the earliest instant at which that arm's own
source has a non-NULL `actual` for that pair.

Measured read-only on the live replica (9.43 GB,
`C:\Code\able\data\energy_dashboard.db`) on 2026-08-12, fixed here so it cannot
be re-resolved after seeing a number:

| pair | arm A start (`energy_renewable`) | arm B start (`energy_generation`) | A days | B days | B/A history ratio |
|---|---|---|---:|---:|---:|
| AT solar | 2025-11-07 | 2021-01-01 | 246 | 2,017 | **8.20** |
| AT wind_onshore | 2025-11-07 | 2021-01-01 | 246 | 2,017 | **8.20** |
| DE solar | 2025-09-08 | 2021-01-01 | 306 | 2,017 | **6.59** |
| DE wind_onshore | 2025-09-08 | 2021-01-01 | 306 | 2,017 | **6.59** |
| BE solar | 2024-01-01 | 2021-01-01 | 922 | 2,017 | 2.19 |
| BE wind_onshore | 2024-01-01 | 2021-01-01 | 922 | 2,017 | 2.19 |
| BE wind_offshore | 2024-01-01 | 2021-01-01 | 922 | 2,017 | 2.19 |
| FR solar | 2023-01-01 | 2021-01-01 | 1,287 | 2,017 | 1.57 |
| FR wind_onshore | 2023-01-01 | 2021-01-01 | 1,287 | 2,017 | 1.57 |
| FR wind_offshore | 2023-01-01 | 2023-05-31 | 1,287 | 1,137 | **0.88** |

Two facts this table settles in advance, both of which cut against the simple
story:

1. **The treatment is not uniform.** "2,049 days vs 277" is an AT/DE fact. BE
   gets 2.2x and FR 1.6x. Any claim that long history is the operative mechanism
   has to survive that spread.
2. **FR wind_offshore is a reversed cell** — arm A has *more* history than arm B,
   because `energy_generation` has no FR offshore before 2023-05-31. It is the
   only pair where this amendment handicaps arm B rather than helping it, and it
   is also one of window 1's largest arm-B wins (-9.8%, n=287). It is registered
   as a **control cell**, not as evidence for the switch.

### The registered prediction — falsifiable, stated before any number

If long history is what makes `energy_generation` the better training source,
then arm B's relative WAPE improvement must be **monotone in the B/A history
ratio**: AT and DE (8.2x, 6.6x) should gain most, FR wind_offshore (0.88x)
should gain least or reverse.

**Primary registered statistic: Spearman's rho between the B/A history ratio and
arm B's relative WAPE change across the ten pairs.** Predicted sign: negative
(higher ratio -> larger improvement).

**If rho is not negative, the long-history hypothesis is not supported by this
experiment, whatever the individual cells do, and I will report it that way.**
Ten pairs is n=10 for this statistic and it is underpowered; it is registered as
a directional check, not a significance test, and no p-value will be quoted from
it.

This prediction is also well-targeted: window 1's three regressions — AT solar,
DE wind_onshore, BE wind_onshore — are the three largest history ratios. The
pairs the switch has to rescue are exactly the pairs receiving the most
treatment.

### Two design failures from Amendment 1 that this one must not repeat

Both were self-reported in §6 after the fact. The fixes are registered, not
promised.

**Failure 1 — I registered a seasonal claim I had not measured.** Amendment 1
asserted the winter holdout "runs the seasonal bias the other way". It does not;
it steepens the same gradient (AT solar fit->gate level jump 1.6x in window 1,
4.3x in window 2). Window 2 could therefore not discriminate the hypothesis it
existed to test.

*Fix:* **Amendment 2 registers no seasonal claim at all.** It reuses window 1's
gate window, whose level behaviour is already measured and on the record. In its
place, the fit/gate level ratio per arm per pair is registered as a **mandatory
reported diagnostic**, and its reading is pre-committed:

> Arm B's fit window spans five full seasonal cycles by construction, so its
> fit/gate level ratio will sit closer to 1 than arm A's for most pairs. §6
> established that on this gate window the arm training at the higher level
> under-predicts less. **Therefore, if arm B wins and its level ratio is closer
> to 1, that win is confounded with the level effect and will be reported as
> confounded — not as a data-quality result.** The confound is expected; hiding
> it is what would be wrong.

**Failure 2 — the D-7 precondition was checked after the fact and four cells
were inadmissible.** Window 2's four solar cells failed "both arms must beat
D-7", so its headline numbers in both directions were void under its own rules.

*Fix, in two parts:*

- **Pre-screen, already satisfied.** Amendment 2 reuses window 1's gate window,
  where all ten pairs cleared D-7 in both arms by +18.0 to +49.4 points (§6
  table). The precondition is therefore known satisfiable for this window
  *before* fitting. This is the main reason the gate window is reused rather
  than moved.
- **Order of operations, binding.** D-7 skill is computed and cell admissibility
  is fixed **before** the A/B contrast is read. An inadmissible cell is reported
  as inadmissible and is excluded from rho.

### Decision rule — per pair, not one global verdict

The global switch is dead: ABL-331 makes the training source a per-artifact
property, so the unit of decision is now one pair. Threshold unchanged from
window 1 at **2.0 % relative WAPE**, so it cannot be accused of moving after a
result.

For each of the ten pairs, on the primary truth:

| outcome | rule |
|---|---|
| **migrate candidate** | arm B better by > 2.0 % relative, **and** both arms clear D-7, **and** both truths agree on the sign, **and** n >= 1,000 common rows |
| **hold** | abs(relative) <= 2.0 %, or the two truths disagree on the sign |
| **do not migrate** | arm B worse by > 2.0 % relative |

The n >= 1,000 clause is registered **now, before the numbers**, because in
window 1 I had to disclaim FR's two largest wins after the fact for resting on
n=287 against 1,950. FR cells are reported in full and are excluded from
carrying a migrate recommendation on their own.

**Multiplicity.** Ten cells against a 2.0 % threshold; some will cross by noise.
No cell is recommended on its own crossing alone — a migrate candidate is a
recommendation to retrain and gate-read *that pair* under ABL-331, which is
itself a further gate. This amendment recommends; it does not migrate, and it
does not promote.

### Contamination exposure — new, and larger than window 1's

Arm B now reaches back to 2021-01-01, so it trains through far more known-bad
history than window 1 did. Touching this window: **ABL-71** (prod ingest stale,
fixes undeployed), **ABL-111 / ABL-109** (zero-as-missing actual-load rows; load
only, but the same ingest path), **ABL-188** (DE solar zero-fill), **ABL-199 /
ABL-200** (BE wind_offshore zero-fill runs, ~130 since 2024, only 9 caught by
the ABL-188 invariant). **ABL-67** is net_position and does not touch these
tables. **ABL-324** (78,510 duplicate instants in `energy_renewable`) is handled
at the loader by the collapse landed in PR #12.

Registered handling: `exclude_suspect_constant_runs` stays on for **both** arms,
unchanged. **Rows dropped by the guard are counted and reported per arm per
pair** — arm B's longer window will drop more in absolute terms and that number
belongs in the report, not in a footnote. Arm B's exposure to unadjudicated
zero-fill is materially greater than arm A's and is a genuine cost of the longer
history, not a nuisance to be normalised away.

### Sequencing — this runs after ABL-332, not before

**ABL-332** (the shared renewable feature builder is hourly and discards
`:15/:30/:45`) hits this experiment. Measured read-only on the replica
2026-08-12, over window 1's span, the resolution is **symmetric across the two
tables**:

| country | `energy_generation` | `energy_renewable` |
|---|---|---|
| AT | 15-min | 15-min |
| DE | 15-min | 15-min |
| FR | 15-min | 15-min |
| BE | hourly | hourly |

So the builder defect handicaps **both arms equally** and window 1's A/B
contrast is *not* contaminated by it — a confound checked and retired rather
than assumed. But it means both arms currently train on the `:00` sub-sample
alone for AT/DE/FR, and ABL-332's fix will change both arms' inputs. Running
Amendment 2 first would produce a result that is stale the day ABL-332 lands.

**Registered order: ABL-322 (offshore pilot) -> ABL-332 -> Amendment 2.** Per the
CEO, this amendment sits behind the offshore pilot in priority and no longer
gates ABL-316.

### Harness change required

`scripts/evaluate_renewable_source_switch.py` takes one `--fit-start` shared by
both arms. Amendment 2 needs per-arm starts (`--fit-start-a` / `--fit-start-b`,
defaulting to the table above). That change is part of the run, and lands with
the run's PR.

### Out of scope — unchanged

No promotion, no serving change, no registry change, no ingest change, no
replica write. Arm B is fitted for measurement only; nothing fitted here is a
candidate artifact.
