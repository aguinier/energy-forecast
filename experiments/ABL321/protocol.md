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
