# ABL-467 — tranche 2c's k=12 cells re-graded under the Student-t interval

Scope `abl467-t2c-regrade`, a re-grade of `abl427-t2c-reread`. Registration `experiments/ABL467/config.json`, argued in `reports/abl_467_seed_interval_readability_registration.md` and committed before this file existed.

**No refit, no replica read, no artifact, no promotion.** Arithmetic over `reports/abl_427_tranche2c_seed_reread.json` (blob pinned and verified) and ABL-419's committed slope and correlation. ABL-427's record is not edited or regenerated; this is a new scope and a new document.

**What moves:** G1 only. G2/G3 are sign tests on 77-93% margins under this scope's registered form; G4 is a sign test on the challenger's own slope and correlation and is carried from ABL-419 unchanged.

## Verdict

| pair | ABL-427 published (k=12) | **ABL-467 amended** | bands | moves |
|---|:---:|:---:|---|:---:|
| **IT** | `U` | **`U`** | `A` / `U` / `A` | no |
| **HR** | `U` | **`A`** | `A` / `A` / `A` | **yes** |

## Per cell

| pair | band | n / min | mean skill vs D-7 | 95% t-CI | t half-width | ABL-427 floor | published | **amended** | `delta_min` at k=12 |
|---|---|---:|---:|---|---:|---:|:---:|:---:|:---:|
| IT | 24-36h | 720 / 684 | +3.99% | [+0.96, +7.01]% | 3.021pp | 4.772pp | `U` | **`A`** | `A` |
| IT | 36-48h | 720 / 684 | +2.69% | [-0.37, +5.74]% | 3.052pp | 4.756pp | `U` | **`U`** | `U` |
| IT | 48-64h | 510 / 456 | +6.29% | [+3.27, +9.31]% | 3.018pp | 4.885pp | `A` | **`A`** | `A` |
| HR | 24-36h | 720 / 684 | +5.39% | [+2.23, +8.56]% | 3.164pp | 5.072pp | `A` | **`A`** | `A` |
| HR | 36-48h | 720 / 684 | +5.37% | [+2.26, +8.48]% | 3.114pp | 4.991pp | `A` | **`A`** | `A` |
| HR | 48-64h | 510 / 456 | +4.62% | [+1.47, +7.77]% | 3.152pp | 5.012pp | `U` | **`A`** | `A` |

The `delta_min` column is **this module's own floor at k=12** (3.074pp), not the floor ABL-427 registered. It agrees with the amendment on every cell — the disagreement is with ABL-427's stricter scope-level choice alone, which is the double-count the registration argues against.

## Seeds losing to the baseline outright

Recorded on every amended cell, because an interval does not show it and it is the number that should govern any serving conversation (ABL-427 §5).

| pair | band | fits losing to D-7 | sd of skill |
|---|---|:---:|---:|
| IT | 24-36h | 2 / 12 | 4.75pp |
| IT | 36-48h | 2 / 12 | 4.80pp |
| IT | 48-64h | 1 / 12 | 4.75pp |
| HR | 24-36h | 2 / 12 | 4.98pp |
| HR | 36-48h | 2 / 12 | 4.90pp |
| HR | 48-64h | 3 / 12 | 4.96pp |

## The prediction registered before this ran

```
IT_24-36h: expected A, got A
IT_36-48h: expected U, got U
IT_48-64h: expected A, got A
HR_24-36h: expected A, got A
HR_36-48h: expected A, got A
HR_48-64h: expected A, got A
IT_pair: expected U, got U
HR_pair: expected A, got A
```

## What this does not do

It promotes nothing. `A` is ABL-418 promotion-**eligibility**, which is necessary and not sufficient; promotion is a CEO-to-Board decision on an evidence pack. IT remains `U` and is not close. Every caveat in `reports/abl_467_seed_interval_readability_registration.md` §7 travels with these letters — in particular that the three bands of one country share a fit and are not three independent estimates, so a pair letter is not a joint 95% statement.
