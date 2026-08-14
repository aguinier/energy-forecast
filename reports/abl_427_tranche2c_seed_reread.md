# ABL-427 — tranche 2c `U(+)` re-read at k = 12 seeds

Scope `abl427-t2c-reread`, a re-read of `abl316-t2c` and not a re-basing of it. Registration `experiments\ABL427\config.json`, frozen before the first fit. Generated 2026-08-14 11:38 UTC in 7.9 min.

Seeds (ABL-385's registered list, in registered order): `[42, 1337, 2718, 7, 13, 101, 271, 314, 577, 863, 1024, 1729]`.

## The read

| pair | band | n | challenger WAPE (k-mean) | D-7 WAPE | skill vs D-7 | own-error margin | measured floor (95% upper) | fleet floor | grade |
|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| IT | 24-36h | 720 | 6.6785% | 6.9558% | +3.99% | +4.15% | 4.77% | 3.07% | **U(+)** |
| IT | 36-48h | 720 | 6.7689% | 6.9558% | +2.69% | +2.76% | 4.76% | 3.07% | **U(+)** |
| IT | 48-64h | 510 | 6.1703% | 6.5845% | +6.29% | +6.71% | 4.88% | 3.07% | **A** |
| HR | 24-36h | 720 | 15.3484% | 16.2233% | +5.39% | +5.70% | 5.07% | 3.07% | **A** |
| HR | 36-48h | 720 | 15.3520% | 16.2233% | +5.37% | +5.68% | 4.99% | 3.07% | **A** |
| HR | 48-64h | 510 | 15.4405% | 16.1887% | +4.62% | +4.85% | 5.01% | 3.07% | **U(+)** |

## The measured per-fit seed CV

ABL-385's fleet p90 for solar is **5.4328%**. `c_B = 0` throughout: every reference on the ladder is deterministic.

| pair | band | c_A (WAPE) | 95% CI | sd (pp) | range (pp) | c_A excl. seed 42 | vs fleet p90 |
|---|---|---:|---|---:|---:|---:|---:|
| IT | 24-36h | 4.9521% | [3.508, 8.434]% | 0.3307 | 1.1094 | 5.1833% | 0.91x |
| IT | 36-48h | 4.9353% | [3.496, 8.405]% | 0.3341 | 1.1153 | 5.1632% | 0.91x |
| IT | 48-64h | 5.0692% | [3.591, 8.633]% | 0.3128 | 0.9900 | 5.2676% | 0.93x |
| HR | 24-36h | 5.2634% | [3.729, 8.964]% | 0.8079 | 2.5710 | 5.4413% | 0.97x |
| HR | 36-48h | 5.1798% | [3.669, 8.821]% | 0.7952 | 2.4765 | 5.3869% | 0.95x |
| HR | 48-64h | 5.2018% | [3.685, 8.859]% | 0.8032 | 2.3796 | 5.2573% | 0.96x |

## Grade under each floor

| pair | band | measured (95% upper) | measured (point) | fleet p90 |
|---|---|:---:|:---:|:---:|
| IT | 24-36h | U(+) | A | A |
| IT | 36-48h | U(+) | U(+) | U(+) |
| IT | 48-64h | A | A | A |
| HR | 24-36h | A | A | A |
| HR | 36-48h | A | A | A |
| HR | 48-64h | U(+) | A | A |

**Pair grades** (worst band, ABL-418's `pair_grade`):

- **IT** — measured 95% upper: **U(+)**; measured point: U(+); fleet p90: U(+)
- **HR** — measured 95% upper: **U(+)**; measured point: A; fleet p90: A

### Disposition — what this re-read returns

ABL-418's `(+)` means *re-read at k > 1 seeds*. This issue **is** that re-read, so its output is `A` or a plain `U`; the ladder cannot know the re-read has happened and still emits `U(+)` on a k = 12 mean. A `U` below is the stronger statement — not unresolved at one seed, but still unresolved at twelve.

| pair | ABL-419 (k=1) | ABL-427 disposition (k=12) |
|---|:---:|:---:|
| **IT** | U(+) | **U** |
| **HR** | U(+) | **U** |

## The direct empirical test (reported, never gating)

Skill vs D-7 is one number per seed, so the k draws support an exact Student-t interval with no delta-method approximation at all. `delta_min` exists to give a **k = 1** read an imported margin; at k > 1 the draws are in hand. Registered rule above; this is the sensitivity.

| pair | band | mean skill | sd (pp) | 95% t-CI on mean skill | excludes 0 | seeds losing to D-7 | worst seed | seed 42 rank |
|---|---|---:|---:|---|:---:|---:|---:|---:|
| IT | 24-36h | +3.99% | 4.75 | [+0.96, +7.01]% | **yes** | 2/12 | -6.40% | 7/12 |
| IT | 36-48h | +2.69% | 4.80 | [-0.37, +5.74]% | no | 2/12 | -7.52% | 7/12 |
| IT | 48-64h | +6.29% | 4.75 | [+3.27, +9.31]% | **yes** | 1/12 | -2.86% | 5/12 |
| HR | 24-36h | +5.39% | 4.98 | [+2.23, +8.56]% | **yes** | 2/12 | -3.88% | 4/12 |
| HR | 36-48h | +5.37% | 4.90 | [+2.26, +8.48]% | **yes** | 2/12 | -4.15% | 7/12 |
| HR | 48-64h | +4.62% | 4.96 | [+1.47, +7.77]% | **yes** | 3/12 | -4.41% | 4/12 |

## Reproduction controls

### Deterministic references vs ABL-419's committed record

42 comparator cells compared: **12 absent_from_abl419_record**, **30 identical**.

`absent_from_abl419_record` is ABL-437's trailing-28d pair, which did not exist when ABL-419 was written. It is a schema addition, not a moved actual.

**No deterministic reference moved.** The gate-window actuals for IT and HR solar in `energy_generation` are unchanged since ABL-419, so nothing in this read is a revision effect.

### Seed 42 against ABL-419's published challenger

| pair | band | ABL-419 | seed 42 here | Δ (pp) |
|---|---|---:|---:|---:|
| IT | 24-36h | 6.6251% | 6.6251% | +0.0000 |
| IT | 36-48h | 6.7073% | 6.7073% | +0.0000 |
| IT | 48-64h | 6.0492% | 6.0492% | +0.0000 |
| HR | 24-36h | 14.9521% | 14.9521% | +0.0000 |
| HR | 36-48h | 15.0601% | 15.0601% | +0.0000 |
| HR | 48-64h | 14.7936% | 14.7936% | +0.0000 |
