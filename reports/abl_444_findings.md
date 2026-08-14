# ABL-444 — findings

Evidence pack for the G2/G3 readability floor. Registration:
`reports/abl_444_g23_readability_floor_registration.md` and
`experiments/ABL444/config.json`, both committed before the read. Machine record:
`reports/abl_444_g23_floor_reread.json`; rendered table
`reports/abl_444_g23_floor_reread.md`.

**Protocol.** Arithmetic over records already on disk — each tranche's committed
`results_*.json` for the challenger's own scores, slope, correlation,
`enough_pairs` and `minimum_n`, and `reports/abl_437_causal_levelling_reread.json`
plus `reports/abl_443_offshore_trailing_reread.json` for the trailing-28d
reference WAPEs on the same cells. **No refit, no new model, no database read.**
**119 cells, 41 pair-records** — tranches 1b and 2a–2f, plus the
`abl443-offshore-trailing` scope, which merged to main while this issue was open.
Every cell graded four ways, because both registered axes stay live:
`{fit_window, trailing_28d} × {sign_test, floored}`. Out-of-sample throughout: ABL-348's frozen
gate window 2026-07-11 → 2026-08-10, fit window 2026-01-14 → 2026-07-11.
`n` per cell is 388–720; the floor is `readability_floor_pct` at k=1, **10.65%
solar / 7.51% wind**.

**Contamination.** ABL-71 / ABL-67 / ABL-109 / ABL-111 touch no part of this read,
which reads no ingest, no source table and no window. ABL-396's night-contamination
hold on BG solar is live and is carried into §2 and §5.

---

## 1. The read reproduces every published letter before it changes any

All **119 cells** and **41 pair-records** come back identical to ABL-437's and
ABL-443's published and amended columns on the `sign_test` arms — through a different path,
since the challenger side here is read from each tranche's own committed record
rather than from ABL-437's. Zero disagreements. That is what makes the `floored`
column a comparison rather than a restatement.

## 2. What the floor changes

### On the published path — 2 pair-records of 41, none of them `A`

| tranche | pair | published | floored | abstains on |
|---|---|:---:|:---:|---|
| 2d | NL solar | B | **N** | G3 (−6.27 / −8.27 / −5.94) |
| 2e | HU `wind_onshore` | B | **N** | G2 and G3 (−1.75…−4.90, −2.13…−4.17) |

**The promotable set as published does not move.** Two *measured worse* verdicts
become honest abstentions; no `A` is withdrawn. Solar is unmoved here for a
structural reason rather than a lucky one — against a fit-window climatology a
solar challenger's G3 margin is tens of percent.

### On ABL-437's amended path — 11 of 41, and 5 of them from `A`

| tranche | pair | amended | floored | margin (skill %) |
|---|---|:---:|:---:|---|
| 1b | BG solar | A | **N** | G3 +8.86 / +10.56 / +9.26 — **hold, §5** |
| 2a | BG solar | A | **N** | G3 +5.29 / +6.18 / +5.69 — **hold, §5** |
| 2d | EE solar | A | **N** | G3 **+0.35** — **§3** |
| 2d | FI solar | A | **N** | G3 **+0.59** — **§3** |
| 2e | HR `wind_onshore` | A | **N** | G2 +2.80, G3 +2.02 |
| 2a | PL solar | B | **N** | G3 −1.13 / −1.11 / **−0.36** |
| 2a | SK solar | B | **N** | G3 −7.70 / −8.01 / −2.29 |
| 2d | LT solar | B | **N** | G3 −6.87 / −6.12 / −3.65 |
| 2d | NL solar | B | **N** | G3 −3.86 / −5.64 / −1.95 |
| 2d | SE solar | B | **N** | G3 −8.65 / −6.51 / −2.97 |
| offshore | DE `wind_offshore` | B | **N** | G2 +0.33 / +1.32 / +1.03, G3 +0.52 / +1.32 / **−0.47** — §3.1 |

Four more pair-records abstain in at least one band without their letter moving,
because a worse band carries the pair: 2b ES `wind_onshore` (`C`), 2c PT solar
(`C`), 2a HU solar and 2c IT solar (`U`). The full inventory is
`abl_444_g23_floor_reread.md` §2 — **57 abstaining cell-conditions across the two
levellings**, not only the ones that move a letter.

### 2.1 DE `wind_offshore` is the case the floor was registered for

ABL-443 landed on main mid-issue and had already done the diagnosis without the
machinery to act on it: its record labels all six DE margins *"not readable at one
seed"* and carries `g2_g3_floor_is_a_ladder_condition: **false**`. Under the
floored form all three of DE's bands abstain — its two shorter bands were graded
`A` on G2/G3 margins of **+0.33% to +1.32%**, and its `B` came from a G3
*failure* of **−0.47%**. So DE's published `B` and its two published band-level
`A`s rest on six margins of which not one is readable, and the pair reads `N`.
Its G1 is untouched at +24…+26pp against D-7, and ABL-436's PASS stands.

## 3. The finding this read did not go looking for: EE and FI solar

**ABL-421 declares EE and FI solar NOT-EVALUABLE on 24-36h and 36-48h, so 48-64h
is the only band either pair carries a letter on. That single cell is
simultaneously coverage-short and sub-floor.**

| pair | band | n | registered minimum | `gate.pass` | ladder, 28d sign test | floored |
|---|---|---:|---:|:---:|:---:|:---:|
| EE solar | 48-64h | **388** | 456 | **False** | A | **N** |
| FI solar | 48-64h | **453** | 456 | **False** | A | **N** |

So each pair's `A` rests on a cell that **failed the gate on coverage** and whose
G3 margin under the corrected references is **+0.35%** and **+0.59%** — 3.3% and
5.5% of the solar floor. Two independent guards were needed to see it and the
ladder carried neither: ABL-434 is the coverage one and is filed and in backlog,
ABL-444 is this one.

These are the only two coverage-short cells in the programme, and they are the
same two cells this floor abstains on. **Neither issue's fix alone makes those
letters honest** — a floored ladder still reports `N` rather than "not
evaluable on coverage", and a coverage-aware ladder still reports a +0.35%
margin as a pass. They are deliberately not folded (a floor and a coverage
minimum are different guards) and both are needed. Cross-posted to ABL-434.

FI's shortfall is also the one ABL-421 records as `source_dependent`: 663 of 720
gate hours in `energy_generation` against `energy_renewable`'s 717. That is a cost
of ABL-348's source change, not a fact about FI's model.

## 4. The two denominators no longer agree, and it is one cell

ABL-418 measured `skill` against ABL-385's own-error form over its 48 cells and
found no cell moved. Over the programme's **476** G2/G3 condition-observations,
**3** change readability status and **one cell letter moves**:

| tranche | pair | band | condition | skill | own error | floor |
|---|---|---|---|---:|---:|---:|
| 1b | BG solar | 36-48h | G3 | **+10.56** (inside) | **+11.81** (outside) | 10.65 |
| 2e | RO `wind_onshore` | 36-48h | G2 | −7.53 (outside) | −7.00 (inside) | 7.51 |
| 2f | CH `wind_onshore` | 24-36h | G2 | −7.93 (outside) | −7.35 (inside) | 7.51 |

Only BG solar 1b 36-48h moves a letter (`N` on the registered column, `A` on
ABL-385's). **No pair letter moves at all**, including BG's: its other two bands
are inside the floor under either denominator. Quote the 119-cell number, not
ABL-418's "they never disagree".

## 5. One correction to the ABL-444 description, and one hold

**CH `wind_onshore` does not reproduce at pair level.** The issue names PL solar
(0.36pp) and CH `wind_onshore` (0.52pp) as the two flips inside the floor. PL
reproduces; CH does not. Its 24-36h band fails G2 at **−7.93%** and G3 at
**−12.89%**, both outside the 7.51% wind floor, and a pair takes its worst band —
so CH stays `B`, earned on a band that fails readably. The 0.52pp is its
*tightest* band. ABL-437's flip-margin column is labelled "tightest–widest" for
exactly this reason, and a tightest-band margin reads like a pair-level one.

**BG solar's two `A → N` moves carry ABL-396's hold.** BG books 152–246 MW in
76–85% of its night hours and 25.3% of its scored gate rows are night rows. The
displacement band is far wider than the +5.29% to +10.56% margins the floor
abstains on, so BG solar's letter under any arm is qualified by data before it is
qualified by readability. It is 1b BG solar that ABL-438 published as `A × 2`.

## 6. What this does not say

- **Gradeability, not skill.** No model is better or worse for this read.
- **It cannot make a pair promotable.** `N` is only ever reached from what would
  have been `A` or `B`. Note that `N` ranks *better* than `B` on the ladder — an
  abstention is a weaker negative than a named failure — so a `B → N` move lowers
  severity while leaving the pair exactly as non-promotable. Assert the caveat on
  the `A` set, not on `GRADE_SEVERITY`.
- **Nothing published is edited.** All twelve published scopes are pinned to
  `sign_test`; this is a new document beside them, on the ABL-418 / ABL-437
  precedent.
- **Tranche 1a is absent**, for ABL-437's reason: fitted before ABL-389 existed,
  so it carries no causal reference columns and G2/G3 read *not measured* there
  under every arm.
- **It promotes nothing and recommends no promotion.** Promotion remains a
  pre-registered gate read plus a Board decision.

## 7. Follow-ups named

1. **ABL-434** (backlog) — the coverage guard. §3 is a compounding finding for it,
   not a duplicate: the two guards catch different halves of the same two cells.
2. **Promote `G23_READABILITY` into `check_registration_tables`** in both
   harnesses when the repo queue is next at zero. It is out today because adding
   a required table raises on `import` for every branch in flight and three PRs
   are open; the cost of the absence is bounded because the default is the
   conservative path.
3. **The re-read is levelling-agnostic and both arms are reported.** ABL-437 and
   ABL-443 both merged to main during this issue and the offshore scope was
   folded in; a further scope landing after this read needs the same treatment,
   which is one row in `_offshore`-shaped code rather than a new document.
