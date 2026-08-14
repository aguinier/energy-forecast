# ABL-444 — a readability floor on G2 and G3, pre-registered

**Status: PRE-REGISTERED.** Committed before the floored ladder is read against
any pair. Machine form: `experiments/ABL444/config.json`. The re-read is a
separate document and a separate commit
(`reports/abl_444_g23_floor_reread.md`), so the order is checkable in git.

Approved by the CEO on ABL-437: *"A sign test that flips on 0.36pp (PL solar) or
0.52pp (CH tightest band) is reporting noise as signal, and that is the same
failure mode as ABL-381's headline margin. I would rather the gate say 'not
readable' than award a sign."*

---

## 1. The defect, and why it is a coherence fix rather than a new bar

ABL-418 registers four conditions. **G1 carries a readability floor; G2 and G3
were registered as bare sign tests, `skill > 0`.** So a G2/G3 verdict could turn
on a margin far inside the spread one seed resolves, and the ladder reported it
with the same letter it uses for a decisive result.

The part of this that is easy to miss, and that decides the shape of the fix:
**ABL-418 already applies `readability_floor_pct` to G2 and G3.** It does it on
this same `skill vs X` column, in this same function, when it decides `U`
against `U(+)`:

```python
readable = all(skill[name] is not None and skill[name] > floor
               for name in (level, shape))
return CellGrade(grade="U", plus=bool(readable and conditions["G4"]), ...)
```

So the floor is not being invented for these two conditions. It is already
registered on them, on one branch of the ladder, and absent from the other. A
margin of +2% on G3 is "not readable, no plus" if G1 happened to be unreadable,
and "G3 holds, grade A" if G1 happened to be clear. **The amendment carries the
existing test to the other branch.** That is a narrower change than a new bar,
and it is why the floor's derivation, value and `k` handling are reused
untouched rather than re-derived.

## 2. What is registered

### 2.1 A third outcome, `N`

| | means |
|---|---|
| `A` | every condition holds, readably. Promotion-eligible, subject to any named data hold. |
| `B` | G1 holds; one or more of G2/G3/G4 **fails**, named. |
| `C` | a readable loss to the registered D-7 bar. |
| `U` | the **G1** margin is inside the floor. `U(+)` where G2–G4 clear readably. |
| **`N`** | **G1 holds readably, nothing below it fails readably, and at least one of G2/G3 has a margin at or inside the floor.** Not demonstrated *in either direction*. Not promotion-eligible. |

`N` may not wear `U`'s letter: `U` is undecided on the **gate**, which is a
different statement about a model. It may not wear `B`'s either: `B` names a
failure, and a failure is precisely what is not being claimed.

**`N` is not promotion-eligible**, on the rule ABL-418 already states and the
net-position gate's `INCOMPLETE` and ABL-421's `SCOPE_NOT_EVALUABLE` state one
level down: *a condition that could not be measured is not satisfied*. `A` still
requires every condition to hold.

### 2.2 Severity: `A < N < U < B < C`

Two orderings had to be argued, and both cut against the reading that "the
abstention is the mildest thing on the ladder".

- **A definite failure outranks an abstention.** A cell whose G4 fails and whose
  G3 is unreadable grades `B`, not `N`. There is something definite to report and
  `N` would bury it — the same argument ABL-418 used to put `B` above `U`.
- **`N` sits above `U`.** Both say *re-read at k>1 seeds*; `U`'s deferral is the
  deeper one, because an `N` cell has cleared the registered gate readably and
  only cannot resolve a condition below it, where a `U` cell cannot resolve the
  gate itself.

The consequence worth stating plainly, because a reader will meet it: a pair
takes its **worst** band, so an `A / A / N` pair grades `N`. That is intended. An
abstention in any band is enough to stop a pair being promotion-eligible, exactly
as one `U` band is.

### 2.3 G4 is untouched

G4 is a sign test on the challenger's own slope and correlation. There is no
margin to read against a floor, so G4 is never routed through the abstention
branch under either form. Registering a floor there would be a different
question — how large a slope is readable — and it is not asked here.

### 2.4 The margin prints either way

Per the CEO's binding constraint. The floor decides **gradeability**; it does not
replace the number. An `N` cell carries `skill_pct`, `own_error_margin_pct`, the
floor it was tested against and the reason string naming the margin, and the
report's per-pair table gains a **not readable** column beside **failed
conditions** so the two are never collapsed. `N` with the margin printed is
strictly more information than the `A` or `B` it replaces, not less.

## 3. The affected set, measured before this was registered

Source: every committed tranche record, through
`reports/abl_437_causal_levelling_reread.json`, plus ABL-443's offshore re-read —
**119 cells, 41 pair-records**, tranches 1b and 2a–2f and the
`abl443-offshore-trailing` scope. Arithmetic over stored WAPE only: no refit, no
re-read, no database. Floor `readability_floor_pct` at k=1, **10.65% solar /
7.51% wind**.

> The offshore pair set was added after this section was first written: ABL-443
> merged to main while this issue was open, and its own record labels all six DE
> `wind_offshore` margins *"not readable at one seed"* and carries
> `g2_g3_floor_is_a_ladder_condition: false` — the hook this issue closes. The
> counts below are over the full 41.

Both live paths are enumerated, because both stay readable and a pair can be
reported under either. `sign_test × fit_window` is what is **published today**;
`sign_test × trailing_28d` is what ABL-437 makes live when it merges.

### 3.1 The published path — 2 pair-records of 41 move, none of them `A`

| tranche | pair | published | floored | sub-floor margins (skill %) |
|---|---|:---:|:---:|---|
| 2d | NL solar | B | **N** | G3 −6.27 / −8.27 / −5.94 |
| 2e | HU `wind_onshore` | B | **N** | G2 −1.75 / −2.21 / −4.90, G3 −2.13 / −2.60 / −4.17 |

**No published `A` becomes `N`.** As published, the floor removes nothing from
the promotable set; it converts two *measured worse* verdicts into honest
abstentions. Solar is unmoved on this path for a structural reason rather than a
lucky one: against a fit-window climatology a solar challenger's G3 margin is
tens of percent (PL's is +38.30%), far outside a 10.65% floor.

### 3.2 ABL-437's amended path — 11 of 41 move, and 5 are the happier letter

The set this issue said had not been enumerated: pairs that **pass** G2/G3 on a
sub-floor margin.

| tranche | pair | amended | floored | sub-floor margin (skill %) |
|---|---|:---:|:---:|---|
| 1b | BG solar | A | **N** | G3 +8.86 / +10.56 / +9.26 |
| 2a | BG solar | A | **N** | G3 +5.29 / +6.18 / +5.69 |
| 2d | EE solar | A | **N** | G3 **+0.35** (48-64h, its only gated band) |
| 2d | FI solar | A | **N** | G3 **+0.59** (48-64h, its only gated band) |
| 2e | HR `wind_onshore` | A | **N** | G2 +2.80, G3 +2.02 (24-36h only) |

and six flipping from `B`: 2a PL solar (G3 −1.13 / −1.11 / **−0.36**), 2a SK
solar, 2d LT solar, 2d NL solar, 2d SE solar, and **DE `wind_offshore`** — whose
two shorter bands were `A` at +0.33%/+0.52% and +1.32%/+1.32% and whose `B` came
from a G3 *failure* of −0.47%, so all three of its bands abstain and none of the
three margins was ever readable.

**EE and FI solar are the sharpest case in the programme.** ABL-421 declares both
NOT-EVALUABLE on 24-36h and 36-48h, so 48-64h is the *only* band carrying a
letter — and it grades `A` on a G3 margin of +0.35% and +0.59%, which is 3.3% and
5.5% of the solar floor. 1b BG solar is the other one to read carefully: it is
ABL-438's published `A × 2` and already carries ABL-396's live
night-contamination hold, whose displacement band is far wider than the margin.

### 3.3 One correction to this issue's own framing

The issue names PL solar (0.36pp) and CH `wind_onshore` (0.52pp) as the two flips
inside the floor. **PL solar reproduces at pair level. CH `wind_onshore` does
not.**

| CH `wind_onshore`, 2f, amended references | G2 skill | G3 skill | |
|---|---:|---:|---|
| 24-36h | **−7.93%** | **−12.89%** | both outside the 7.51% floor |
| 36-48h | −2.39% | −6.92% | both inside |
| 48-64h | **−0.52%** | −3.84% | both inside |

A pair takes its worst band, so CH `wind_onshore` stays **B**: the 0.52pp quoted
is its *tightest* band, and its `B` is earned on a band that fails readably. The
finding is real at cell level and does not carry to the pair. This is the same
distinction ABL-437's own flip-margin column makes ("tightest–widest") and it is
worth restating because a tightest-band margin reads like a pair-level one.

## 4. Which denominator, measured rather than assumed

Registered on the printed **`skill vs X`** column, `100 × (1 − challenger /
reference)`. Three reasons, in order: it is what G1 is registered on; it is what
ABL-418 *already* floors G2/G3 on for the `U(+)` plus (§1), so registering the
other form would leave the two branches disagreeing about the same margin; and it
is what a reader can check against the report by eye.

ABL-385's CV is measured in the challenger's own error, so the other form —
`100 × (reference − challenger) / challenger` — has the better claim on first
principles. They agree in sign always and converge as the reference approaches
the challenger, which is exactly the region a floor operates in, so the
disagreement should be small. Measured, over all 113 cells:

| | |
|---|---:|
| G2/G3 condition-observations, both levellings | **476** |
| readability status differs between denominators | **3** |
| **cell** letters that move | **1** |
| **pair** letters that move | **0** |

**They do not agree everywhere, and the disagreement is reported rather than
rounded away.** The three are, all under the trailing references:

| tranche | pair | band | condition | skill | own error | floor |
|---|---|---|---|---:|---:|---:|
| 1b | BG solar | 36-48h | G3 | **+10.56** (inside) | **+11.81** (outside) | 10.65 |
| 2e | RO `wind_onshore` | 36-48h | G2 | −7.53 (outside) | −7.00 (inside) | 7.51 |
| 2f | CH `wind_onshore` | 24-36h | G2 | −7.93 (outside) | −7.35 (inside) | 7.51 |

Only the first moves a **cell** letter — 1b BG solar 36-48h reads `N` on the
registered column and would read `A` on ABL-385's. The other two do not, because
those cells fail G3 readably under either form. **No pair letter moves at all**,
including BG solar's: its 24-36h and 48-64h bands are inside the floor under both
denominators, so the pair abstains either way.

`own_error_margin_pct` stays on every graded cell, so a later read can check this
rather than trust it. ABL-418 measured this same sensitivity and found no cell of
its 48 moved; the programme is now 113 cells and one does. That is the reason to
report the number rather than repeat the earlier "they never disagree".

## 5. Which way the table defaults, and why the sketch's guess is not adopted

`G23_READABILITY` is registered per scope in both harnesses, alongside
`CAUSAL_LEVELLING` and composing freely with it: one table says which reference
G2/G3 read, the other how wide a margin must be before its sign means anything.
Every published scope — 6 solar, 6 wind — is pinned to `sign_test`, so **no
committed letter moves and nothing is refit**, which is the CEO's binding
constraint on the ABL-401 ruling.

**A scope registering nothing gets `floored`.** The ABL-444 design sketch guessed
the opposite, and asked for the argument either way; this is the argument.

The sketch's reasoning was that `CAUSAL_LEVELLING` defaults toward its amendment
because silently inheriting an old reference is the failure mode, whereas here
*"an abstention silently inherited by a new tranche hides a result that was in
fact measured"*. That premise does not survive contact with what the floor
asserts:

1. **A sub-floor margin is not a result that was measured.** It is a number whose
   *sign* one fit cannot resolve. That is the whole content of ABL-385, and it is
   what ABL-418 already asserts about these two conditions when it withholds the
   `U(+)` plus. Inheriting `floored` therefore hides nothing that was
   established.
2. **The margin prints either way** (§2.4), so even the weaker reading — that a
   reader wants the sign — is served. It is served *labelled*, which is the
   difference.

And the two directions are not symmetric in what they cost when wrong.
Defaulting to `sign_test` hands a new tranche — the pairs nobody has looked at
yet — a letter awarded on noise, silently. That is the defect this amends, and it
is the same shape as ABL-421's `SCOPE_NOT_EVALUABLE` defaulting toward *scoring*:
a wrong verdict rather than self-documenting degradation. Defaulting to `floored`
hands it an abstention it can resolve by re-reading at k>1 seeds, and the letter
says so.

What defaulting toward the amendment costs is that an *absence* can no longer
reproduce an old read. So every published scope is pinned explicitly, and
`tests/test_abl444_g23_readability_floor.py` derives the published set from
`SCOPE_OUTPUTS` and git rather than from a list retyped here — the ABL-404
precedent, where a pin that had to be remembered went missing across a merge for
two months.

### 5.1 Why it is not in `check_registration_tables`

Deliberate, and revisit-able. CLAUDE.md's rule is that adding a required table
raises on `import` for **every branch already in flight**, which is why ABL-429
waited for both repo queues to reach zero; three PRs are open as this lands.

The cost of the absence is bounded here in a way it is not for
`SCOPE_NOT_EVALUABLE`: a scope that forgets a row gets the **conservative** path,
not a wrong verdict. The four other scope-keyed call sites that grade a
*published* record — `abl418_retro_grade.py`, `abl437_causal_levelling_reread.py`,
`abl419_tranche2c_read.py`, `abl421_tranche2d_read.py` — name `SIGN_TEST`
explicitly rather than defaulting, including the two whose records already carry
a recorded grade and where the default is unreachable today. A default that only
*happens* to be unreachable is the ABL-404 shape.

**Follow-up, named:** promote `G23_READABILITY` into
`check_registration_tables(...)` in both harnesses when the repo queue is next at
zero.

## 6. Caveats that travel with this

- **It changes gradeability, not skill.** No model is better or worse. Some
  verdicts become honest abstentions.
- **It cannot increase the promotable set.** `N` is only ever reached from what
  would have been `A` or `B`.
- **It does not touch ABL-348's registration** — windows, bands, metric,
  baseline, minimum n, source, `not_evaluable` — so `voids_this_registration` is
  not triggered.
- **ABL-434 is not folded in.** The ladder cannot see minimum n; a floor and a
  coverage minimum are different guards and remain separate. `enough_pairs` is
  still reported beside every grade and must still be read.
- **Contamination is unchanged.** ABL-71 / ABL-67 / ABL-109 / ABL-111 touch no
  part of this, which reads no ingest, no source table and no window. ABL-396's
  night-contamination hold on BG solar is carried into the re-read unchanged —
  and BG solar 1b is one of the five pairs §3.2 moves.
- **This promotes nothing.** It amends how a grade is read. Promotion remains a
  pre-registered gate read plus a Board decision.
