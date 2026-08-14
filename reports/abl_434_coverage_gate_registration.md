# ABL-434 — the grade ladder reads the cell's minimum n before its margin

**Status: REGISTERED.** Machine form: `experiments/ABL434/config.json`. The
affected set was measured over every committed record *before* the change landed
and is stated in full in §4; it is two cells, both already reported as held.

Approved by the CEO on ABL-434: *"Scope as filed — make the ladder see minimum n.
Please do not re-open the bar or re-grade any published tranche in this issue; if
fixing the ladder changes a letter anywhere, report the list and I will take the
disposition separately."*

---

## 1. The defect

ABL-418's ladder grades a **margin**. `grade_cell` is handed a cell's `scores`
mapping and nothing else — it never sees `gate.n`, `gate.minimum_n` or
`gate.enough_pairs` — so a cell that beats seasonal-naive D-7 by more than the
readability floor while falling **short of its registered minimum n** graded `A`.

`A` means promotion-eligible. A cell the registration does not consider readable
cannot be.

Measured on `experiments/ABL348/results_abl421_tranche2d.json`, the record where
the combination first arose (every cell in tranches 1a, 1b, 2a, 2b and 2c meets
its minimum, so the ladder had never been exercised against it):

| cell | n | min n | challenger | D-7 | skill vs D-7 | `beats_d7` | `enough_pairs` | `pass` | ladder |
|---|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|
| EE/solar 48-64h | 388 | 456 | 25.05% | 35.29% | +29.01% | `true` | **`false`** | `false` | **A** |
| FI/solar 48-64h | 453 | 456 | 23.99% | 37.98% | +36.83% | `true` | **`false`** | `false` | **A** |

FI misses by **three rows**. Both margins are far outside the 10.65% solar floor,
so G1 holds comfortably and G2–G4 clear too. Neither pair has any other gated
band — their 24-36h and 36-48h cells are the ones ABL-348 declares NOT-EVALUABLE
(ABL-421) — so **both pairs graded `A` with no decidable band at all**, and
`grade: A` sat beside `pass: false` inside the same cell dict with nothing in the
record to reconcile them.

ABL-421 did not fix it there, and was right not to: editing the ladder after
seeing a result is the shopping the pre-registration apparatus exists to prevent,
and `gate_grading.py` is shared with the wind harness. It named a reporting-side
hold instead — `—` with `no band meets the registered minimum n` — which is
correct for one pack and **only for that pack**. The record still says `A`.

## 2. What is registered

### 2.1 A fifth condition, `G0`, assessed before the other four

| | asks | from |
|---|---|---|
| **`G0`** *readable* | **the cell meets ABL-348's registered minimum n for its band** | `gate.enough_pairs` |
| `G1` gate | beats `seasonal_naive` D-7 by more than the readability floor | `skill vs D-7` |
| `G2` level | beats the causal constant | already printed |
| `G3` shape | beats the causal climatology | already printed |
| `G4` direction | `slope > 0` **and** `corr > 0` | already printed |

A cell that fails `G0` grades **`X` — not readable at the registered coverage**.
It does not have the rows the registration requires, so nothing on the ladder
below it is decidable. Not promotion-eligible.

`X` may not wear an existing letter. It is not `C` — nothing lost a race. It is
not `U` — the margin is not inside the floor, and here it is three times outside
it. It is not `N` — ABL-444's abstention is about a G2/G3 *margin*, and this is
about the *cell*.

### 2.2 Severity: `A < N < U < X < B < C`

Deeper than `U` and `N`: a `U` cell has the rows and cannot resolve the margin,
an `X` cell does not have the rows. Shallower than `B` and `C`, on ABL-444's rule
that **a definite failure outranks an abstention** — a band that had the rows and
lost readably has something definite to say, and `X` at pair level would bury it.

A pair takes its worst band, so one band short of its registered n takes the pair
to `X`. That is **stricter** than ABL-421's reporting-side hold, which held a pair
only when *no* band was decidable. The two agree on every cell measured so far,
because EE and FI each have exactly one gated band; where they would differ, one
registered band short of its minimum is already enough to stop a pair being
promotion-eligible, by the same "every band" rule that makes `A` the worst of its
bands.

### 2.3 It is one-way, and unrecorded coverage is not a pass

`gated_on_coverage` can only ever replace a letter with `X`, never raise one, so
no scope can be made promotion-eligible by it. A cell with no `gate` block, or a
`gate` block with no `enough_pairs`, grades `X` with **that** named as the reason
— the same rule as everywhere else on this ladder: *a condition that could not be
measured is not satisfied*. It does not arise on any committed record; see §4.

A grade of `None` — the cell measured nothing — passes through unchanged.
`Not measured` is already the weaker statement, and overwriting it with a
coverage verdict would claim the cell was scored.

## 3. Where the gate lives, and why not in `grade_cell`

This is the load-bearing implementation choice, and it is what keeps the CEO's
second constraint (*do not re-grade any published tranche*) true by construction
rather than by care.

- **`grade_cell(scores, …)` stays a function of `scores` alone.** That is what
  makes it re-runnable over a stored record, and it is what leaves the four
  published margin-only re-reads byte-identical: ABL-418's retro-grade, ABL-437's
  levelling re-read, ABL-443's offshore re-read and ABL-444's G2/G3 floor re-read
  all call it, all reproduce, and all have their own fresh-vs-stored tests still
  passing.
- **`cell_grade(cell, …)` and `attach_grades(cells, …)` are handed a whole cell,
  so they read its coverage** — through `gated_on_coverage`, applied to a
  computed grade and to one rebuilt from a record alike. Both harnesses record
  (`attach_grades`) and render (`cell_grade`) through those two, so **no future
  tranche can write a coverage-blind `A`**, and every later reader of an
  already-written record gets the hold without keeping its own books.
- Applying it to a **rebuilt** grade is the one thing `cell_grade` does not take
  from the record, deliberately. A stored `A` on a cell whose own `gate` block
  says `enough_pairs: false` is the defect itself.

The four ungated callers are registered with a reason in
`tests/test_abl434_coverage_gate.py::MARGIN_ONLY_READERS`, and an AST sweep over
`src/` and `scripts/` fails on any unregistered one — the `tso_plausibility`
pattern. A new script that copies `grade_cell(cell["scores"], …)` fails there
rather than publishing a coverage-blind `A` two tranches later.

There is deliberately **no per-scope registration table**, unlike ABL-437's
`CAUSAL_LEVELLING` and ABL-444's `G23_READABILITY`. `enough_pairs` is already
part of ABL-348's registered `pass` rule and already decides the gate verdict;
what moves here is only whether the grade may disagree with it. A table would
only ever be used to let a scope declare its own cells covered, which is a scope
that can promote on rows it does not have.

## 4. The affected set — every letter that moves

Re-graded through `cell_grade` over **every gate record tracked in git**:
11 records, **143 cells**. Arithmetic only — no refit, no database, no model.

| | count |
|---|---:|
| cells with the coverage column absent | **0** |
| cells coverage-short (`enough_pairs: false`) | **2** |
| **cell letters that move** | **2** |
| **pair letters that move** | **2** |
| published reports regenerated | **0** |

| record | pair | band | before | after |
|---|---|---|:---:|:---:|
| `results_abl421_tranche2d.json` | EE/solar | 48-64h | `A` | **`X`** |
| `results_abl421_tranche2d.json` | FI/solar | 48-64h | `A` | **`X`** |

Pair level: **EE/solar `A` → `X`** and **FI/solar `A` → `X`**, each being its
pair's only gated band. No other pair in the programme moves, in either stream.

The disposition of those two is the CEO's, per the assignment comment. What is
worth saying beside them: both were **already reported as held** — `—` with
`no band meets the registered minimum n` — in `reports/abl_421_tranche2d_tables.md`,
and both were already carried as `-` rather than `A` in today's ABL-316
disposition. Tranche 2d's own verdict is `FAIL` and does not move: `pass` was
already `false` for both cells and `passed`/`disposition` never read a grade.

**Nothing is regenerated.** `reports/abl_418_retro_grade.*`,
`reports/abl_437_causal_levelling_reread.*`, `reports/abl_443_offshore_trailing_reread.*`,
`reports/abl_444_g23_floor_reread.*` and `reports/abl_421_tranche2d_tables.*` are
untouched, and every committed `results_*.json` is byte-unchanged. Regenerating
the 2d pack would print `X` where its ladder column now prints `A`; its
*reported* grade would stay `—`, because ABL-421's hold maps every letter to `—`
when no band is decidable. That regeneration is not done here.

## 5. Caveats

- **This is not a re-opening of the bar.** ABL-348's windows, bands, metric,
  baseline, source, `not_evaluable` list and the minimum n itself
  (`ceil(0.95 × intended_n)`) are untouched. `voids_this_registration` on ABL-348
  is not triggered.
- **A grade is still a reading of a cell, never an input to it.** No gate
  verdict, `passed`, `performance_pass` or `disposition` changes;
  `test_attaching_grades_moves_no_gate_verdict` still holds.
- **`X` is a statement about coverage, not about the model.** EE's +29.0% and
  FI's +36.8% against D-7 are real numbers on the rows that were scored, and they
  are still printed. What `X` says is that those rows are fewer than the
  registration requires for the margin to carry a decision — which for FI is a
  three-row shortfall, and is exactly why the letter is an abstention and not a
  failure.
- **Contamination.** ABL-71 / ABL-67 / ABL-109 / ABL-111 are unchanged by this
  correction, which touches no ingest, no source table, no window and no fit.
  ABL-396's night-contamination hold on BG solar and ABL-419's night-floor
  reading on EE/FI travel unchanged with their pairs.
- **The three adjacent corrections are untouched.** ABL-426 (wrong source) and
  ABL-440 (gitignored record) make a pair unreadable, which is visible; this one
  made an unreadable pair look promotable, which is not — that is why it went
  first, and it closes none of the others.
