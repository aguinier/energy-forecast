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
