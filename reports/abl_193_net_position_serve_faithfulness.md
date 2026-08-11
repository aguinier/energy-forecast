# ABL-193 — live-vintage serve-faithfulness attestation

**Verdict: PASS for V010, V012, V014 and V016.** This is an offline input-parity
reconstruction, not the separate sidecar/prod-pushed row-overlap check.

## Protocol and result

- Live vintage: V010 `2026-08-11 06:00:46.025210` UTC; challengers
  `2026-08-11 06:00:55` UTC.
- Forecast window: `2026-08-13 00:00–23:00` UTC.
- Publication/observation cutoff: `2026-08-11 22:00:00` UTC exclusive
  (observations through 21:00); issued weather additionally bounded by each
  model vintage's `generated_at`.
- Sample: 456 median rows per model = 24 hours × 19 pre-registered gate
  countries. Baseline for the parity metric: the stored as-served sidecar
  vintage. No fitting was performed and no scored row was used to fit anything.
- Acceptance tolerance: exactly 0 MW. Measured maximum absolute delta: **0 MW
  for every model and every country**.

| Country | V010 | V012 | V014 | V016 |
|---|---:|---:|---:|---:|
| AT | 0 | 0 | 0 | 0 |
| BE | 0 | 0 | 0 | 0 |
| BG | 0 | 0 | 0 | 0 |
| CZ | 0 | 0 | 0 | 0 |
| DE | 0 | 0 | 0 | 0 |
| EE | 0 | 0 | 0 | 0 |
| ES | 0 | 0 | 0 | 0 |
| FI | 0 | 0 | 0 | 0 |
| FR | 0 | 0 | 0 | 0 |
| HR | 0 | 0 | 0 | 0 |
| HU | 0 | 0 | 0 | 0 |
| LT | 0 | 0 | 0 | 0 |
| LV | 0 | 0 | 0 | 0 |
| NL | 0 | 0 | 0 | 0 |
| PL | 0 | 0 | 0 | 0 |
| PT | 0 | 0 | 0 | 0 |
| RO | 0 | 0 | 0 | 0 |
| SI | 0 | 0 | 0 | 0 |
| SK | 0 | 0 | 0 | 0 |

Values are max |offline − stored| in MW over the country's 24 target hours.
The machine-readable artifact records row counts, cutoffs, exact covariate and
feature inputs, database identities and per-country deltas:
`experiments/net_position_serve_faithful_attestations.json`.

## Inputs reproduced

- V010: 672 observed target hours; 12 past covariates and 7 future-known
  covariates; 50-hour horizon (26-hour gap + final target-day 24 hours).
- V012: net-position actuals before the cutoff, through the shared D-7 plus
  28-day same-hour climatology implementation.
- V014: its stored per-country XGBoost artifact and feature columns, with the
  serve-window cutoffs for net position, price, TSO load/generation, cross-border
  flow and issued weather.
- V016: the stored co-run V010 vintage, `correction.json`, and latest observable
  pre-cutoff residual. BG/LT/RO correctly remain identity corrections because
  their correction fit refuses unverified historical serve parity.

## Contamination caveat

- ABL-67 touches the scope: GR's net-position actuals are fabricated zeros. GR
  is excluded by name from the pre-registered gate and from this 19-country
  attestation.
- ABL-71 remains an input-provenance risk because ingest fixes are undeployed.
  The live replica was verified at the exact canonical path, 5,905,461,248
  bytes, with 647,538 net-position rows through 2026-08-11 21:00 UTC; currency
  does not certify every row's correctness.
- ABL-111/ABL-109 do not touch this window: these models consume no actual-load
  rows.

No database was written. Replica and sidecar were both opened using SQLite
read-only URIs. The only writes are this report and the tracked attestation.
