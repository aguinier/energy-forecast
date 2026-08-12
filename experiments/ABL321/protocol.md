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

## Amendment 3 — ABL-334: re-run window 1 on the corrected builder, and the revert trigger for the ABL-332 landing

Registered **before any model was fitted under it and before any metric was
read**, per the ABL-195 / ABL-253 discipline. Committed as its own change so the
commit timestamp is checkable against the run artifacts.

**Numbering.** Amendment 2 (the "what would ship" per-arm-history test) is
registered on branch `ABL-321-amendment-2-what-would-ship` (`fdca889`) and is
**not on `main`**. This is Amendment 3 rather than 2 so the two do not collide
when both land; the gap is reserved, not skipped. Nothing here revises Amendment
2, and nothing here revises §6 of `reports/abl_321_findings.md`.

### Why window 1 is being re-run

ABL-332 established that `src/wind_features.py` floored every lag, persistence
and rolling anchor to the hour over a series that is **not** hourly for 22 of the
24 supported countries — including AT and DE, and BE for part of its history. It
therefore built features from the `:00` sub-sample and discarded `:15`, `:30`
and `:45`, while its rolling windows sliced the raw ~96-sample day: three
definitions of "an hour" in one feature row.

**Both arms of window 1 ran through that builder.** The recorded §6 numbers are
therefore a comparison of two subsampled arms, not of two source tables. The
defect is not obviously asymmetric between arms, but the per-hour correction is
not small (DE solar median 373.6 MW, p90 3,211 MW) and criterion 2 was decided on
margins of 2.7–4.3 % against a 2.0 % threshold. That is inside the range a
feature-definition change can move, so the measurement is repeated on the
corrected instrument.

### Protocol — window 1, unchanged

Fit targets 2026-01-14 → 2026-07-11 (exclusive); scoring targets 2026-07-11 →
2026-08-10 (exclusive); D+2 primary bands 24-36h, 36-48h, 48-64h; both candidate
truths; common rows only; catboost, `random_seed=42`; the eight pre-registered
vintage instants per target hour; seasonal-naive D-7 rebuilt from each truth.
Arms, pairs, truth handling, row rule and the 2.0 % material threshold are §1–§6
above, unchanged. **The only intended variable is the feature builder.** A re-run
that also changed the protocol would prove nothing.

Replica: `C:\Code\able\data\energy_dashboard.db`, opened `mode=ro`, `uri=True`,
passed explicitly on the command line — a git worktree has no `.env` and
`config.DATABASE_PATH` silently degrades to a bare, nonexistent
`\data\energy_dashboard.db` there. No replica write, no sidecar write.

### The CEO's revert trigger for the ABL-332 landing (registered verbatim, not mine to move)

ABL-332 has already merged to `main` (`48e1fde`, 2026-08-12 21:08:53Z) on a
direction argument rather than a forecast-level result. Per the CEO amendment on
ABL-334, this run is where that result gets measured, and it is the revert
trigger for that landing:

- **If ≥2 of the 10 serving pairs are materially worse post-fix** (>2.0 %
  relative WAPE, the same threshold criterion 2 used) → **revert recommendation
  on the ABL-332 merge**, brought to the CEO. Do not revert unilaterally, and
  **do not proceed to interpret criterion 2.**
- **If 0 or 1 pair regresses** → the landing is verified; continue to the
  criterion-2 re-read.

The trigger is evaluated on the comparison the CEO defined: **arm A as recorded
in ABL-321 window 1 → arm A re-run here**, on the primary truth
(`energy_generation`), aggregated over the three D+2 bands. Those recorded arm A
WAPEs are fixed here before the re-run so they cannot be re-resolved afterwards
(`experiments/ABL321/results_w1.json`, primary truth): AT solar 12.8899, BE solar
16.7571, DE solar 13.5172, FR solar 15.0542, AT wind_onshore 72.3174, BE
wind_onshore 46.5557, DE wind_onshore 51.6298, FR wind_onshore 39.2734, BE
wind_offshore 77.5366, FR wind_offshore 44.0433. n = 1,950 per pair except the
three FR cells at n = 287.

I am not moving a threshold or a comparison the CEO registered. The two
registrations below add attribution and interpretation around it; neither
changes when the trigger fires.

### Registered addition 1 — the isolation control, because the CEO's comparison is not one-variable by construction

`origin/main` has advanced **ten commits** since `de369a6`, the commit that
recorded window 1. Besides ABL-332 (`70f835e`, `59e6b56`), the commits
`b9ebb8a` and `1a133d6` (ABL-331, per-artifact training source), `981e4d6`
(ABL-337, solar serve clamp) and `ad98f53` (ABL-340, package imports) all touch
`src/db.py`, `src/baselines.py` or `scripts/`. "Recorded arm A → re-run arm A"
therefore differences **ABL-332 plus whatever else moved**, and attributing a
regression to ABL-332 that another commit caused would be a wrongful revert
recommendation.

So a third run is registered, executed regardless of outcome: **the isolation
control** — arm A at this same `origin/main` commit with ABL-332's behaviour
neutralised, by patching `db.aggregate_renewable_to_hourly` to the identity and
bypassing `wind_features._assert_hourly`. That reproduces the pre-ABL-332 builder
exactly (raw sub-hourly series in, `:00`-sample lags, raw-index rolling windows)
while holding every other code change fixed. Registered readings:

- **Control reproduces the recorded ABL-321 arm A** → the other nine commits are
  inert on this path, the CEO's comparison *is* one-variable, and the trigger
  reading stands as written with no attribution caveat.
- **Control differs from the recorded arm A** → some of the movement is not
  ABL-332's. The trigger still fires on the CEO's comparison as registered, and
  the control → post-fix difference is reported beside it as the part actually
  attributable to ABL-332, for the CEO to weigh.

The control is an attribution instrument. It is not a second chance at the
threshold.

### Registered addition 2 — the truth series is the `:00` sample, and the corrected target is the hourly mean

Stated now so it cannot read as a post-hoc excuse if the post-fix arm degrades.

`_truth_series` in `scripts/evaluate_renewable_source_switch.py` filters
`h.dt.minute == 0` and reads the tables directly. ABL-332 does not touch it, so
truth is byte-identical across all three runs — which is what makes them
comparable, and it is deliberate. But it means truth is the **instantaneous `:00`
sample**, while post-fix the builder's target is the **hourly mean**. Pre-fix,
target and truth were the same statistic (`:00`); post-fix they are not.

A post-fix arm can therefore score worse on this harness while being a better
estimator of the quantity it now targets — a scoring-convention mismatch, not an
accuracy loss. Registered rule: **if the post-fix arm degrades, I will test
whether the degradation is explained by this mismatch before attributing it to
the fix**, and I will report the mismatch whether or not it is exculpatory. This
does not suppress or delay the trigger: the trigger reads as written, and this
goes to the CEO as interpretation attached to it.

### Contamination touching this window

Unchanged from window 1 and restated rather than assumed: ABL-67 is
net-position-only; ABL-109 / ABL-111 are load-only; ABL-71's known wrong-write
modes are load and net position. None is a proof that solar/wind ingest is
pristine. ABL-318 §3's `energy_generation` gap for FR (2026-06-30 23:45 →
2026-07-22 14:15) still eats the fit tail and the first 11.6 days of FR's scoring
window, which is why the three FR cells carry n = 287 against 1,950.

### Out of scope, and what is not mine to decide

No retraining of served artifacts, no promotion, no serving change, no registry
change, no ingest change, no dashboard change, no replica or sidecar write.
`RENEWABLE_TYPE_SOURCE_TABLE` is not flipped and
`tests/test_renewable_training_source.py::test_the_switch_is_withheld_pending_a_ceo_decision`
is not edited under this amendment. A flipped criterion-2 verdict and a revert
recommendation are both **CEO decisions**; this run reports them and acts on
neither.
