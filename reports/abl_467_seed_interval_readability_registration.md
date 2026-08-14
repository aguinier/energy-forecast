# ABL-467 — the readability test at k > 1, pre-registered

Parent **ABL-316**. Filed by **ABL-427** §7.3 as the child issue that amendment
had to be argued in, rather than taken inside the read whose letter it moves.
Registration: `experiments/ABL467/config.json`.

**Evidence and registration only.** No promotion, no serving change, no registry
change, no refit, no replica write, no ingest change, no dashboard change. Every
number below is arithmetic over records that already exist.

---

## 0. The disclosure that has to come first

**This amendment cannot be registered blind, and pretending otherwise would be
the dishonest version of the ABL-444 pattern.** ABL-427's pack published the six
Student-t intervals and the letters they imply, in §4 of
`reports/abl_427_tranche2c_seed_reread_findings.md` (PR #80, now merged at
`dbc37af`). Its §7.3 then *predicted the outcome of this issue in writing* — "HR
resolves `A`, IT 24-36h resolves `A`, IT stays `U` overall on 36-48h" — before
this issue was assigned.

So the affected set is public, and what pre-registration can still buy here is
narrower but real:

* the **argument** is made on properties of the two estimators, not on which pair
  they favour (§1);
* the **outcome is fixed before the re-grade runs**, restated in §3 and pinned in
  `tests/test_abl467_seed_interval.py::EXPECTED_UNDER_THE_AMENDMENT`, so the
  re-grade cannot quietly return something else;
* the **alternatives are priced against the same six cells** (§4), including the
  two that would have made HR `U`, rather than being named and waved past.

The reader who wants to discount this should discount §3, which is a prediction
already on the record, and weigh §1 and §4, which are not about IT or HR.

---

## 1. The defect

ABL-418's readability floor is ABL-385's

```
delta_min(k) = 1.96 * sqrt(c_A^2 + c_B^2) / sqrt(k)      →   1.96 * c_A / sqrt(k)
```

with `c_B = 0`, since every reference on the ladder is deterministic. `c_A` is the
**fleet 90th-percentile per-fit CV** — a number imported from
`reports/abl_385_decision_margin.json`, measured over other pairs.

That import exists for one reason: **a k = 1 read has no internal estimate of its
own spread.** One fit gives one number and no way to say how much another seed
would have moved it, so the spread has to come from somewhere else. At k = 1 this
is the right and only tool, and this amendment does not touch it.

**At k > 1 the import answers the wrong question.** The fleet p90 says *how much
do fits of this stream vary*. What the grade turns on is *how much does this cell
vary*. When the cell has k fits, that is measurable, and

```
skill_j = 100 * (1 - wape_j / reference)          j = 1..k
```

are k honest draws of exactly the graded quantity. Student's t on them is the
exact small-sample test, and its degrees of freedom are already the correction for
the sd having been estimated rather than known.

Pairing a **chi-square upper bound** on `c_A` with a **z** critical value — which
is what ABL-427's own scope-level rule did — counts that same estimation
uncertainty twice: once in widening the CV to its upper confidence limit, once
again in using the critical value that assumes the CV is known.

### 1.1 The rule does not change form, only its estimator

A condition is readable iff `|margin| > half_width`. That is ABL-418's rule and
it is untouched: `CI excludes 0` and `|mean| > t * se` are the same statement.
What moves is where `half_width` comes from.

| | half-width | provenance |
|---|---|---|
| `k = 1` | `1.96 * c_A_fleet_p90 / sqrt(k)` | imported — the cell has no spread |
| `k > 1` | `t(0.975, k-1) * sd / sqrt(k)` | measured — the cell's own draws |

### 1.2 The point estimate does not move at all

Against a deterministic reference, skill is affine in WAPE, so the mean of the
per-seed skills **is** the skill of the mean WAPE. Measured on all six cells, the
two agree to under `1.3e-14` pp, and every one matches the `skill_vs_d7_pct`
ABL-427 printed. The ladder still grades the registered `skill vs X` column; only
the width it is compared against is re-estimated.

### 1.3 What licenses one set of draws for three conditions

G1, G2 and G3 have different denominators, so each gets its **own** half-width —
but all three are computed from one set of per-seed *challenger* WAPEs, because no
reference on this ladder moves when the challenger is refitted. That is the same
`c_B = 0` property the floor's `sqrt(2)` correction already rests on. **A
stochastic reference voids this registration** and is named as such in
`voids_this_registration`.

---

## 2. This is not the more permissive test

The obvious objection is that a rule change proposed by the author of the pair it
promotes will be the laxer rule. It is not, and the check is cheap.

The t half-width exceeds the unamended fleet floor whenever the cell's own seed CV
exceeds roughly `(z / t_{k-1})` of the fleet p90 — about **93% of it at k = 12**.
Against `readability_floor_pct("solar", 12) = 3.0739pp`:

| cell | t half-width | unamended floor at k=12 | ratio | which is stricter | cell CV / fleet p90 |
|---|---:|---:|---:|:---:|---:|
| IT 24-36h | 3.021 | 3.074 | 0.983 | delta_min | 0.912 |
| IT 36-48h | 3.052 | 3.074 | 0.993 | delta_min | 0.908 |
| IT 48-64h | 3.018 | 3.074 | 0.982 | delta_min | 0.933 |
| **HR 24-36h** | **3.164** | 3.074 | 1.029 | **Student-t** | 0.969 |
| **HR 36-48h** | **3.114** | 3.074 | 1.013 | **Student-t** | 0.953 |
| **HR 48-64h** | **3.152** | 3.074 | 1.026 | **Student-t** | 0.957 |

**All three HR cells are graded against a half-width wider than the unamended
floor, and all three still clear it.** HR's `A` is not bought by a laxer test; it
survives a tighter one. The direction is genuinely two-way, and
`test_the_amendment_grades_down_as_well_as_up` constructs and pins the case where
it takes an `A` away.

The two floors nearly coinciding at k = 12 is arithmetic, not a theorem: these
cells' measured CV happens to sit below the fleet p90 by about the factor by which
`t(11)` exceeds `z`. It should not be expected to hold at another k or on another
stream.

---

## 3. The affected set, and the outcome fixed before the re-grade

Only **G1** moves on these cells. ABL-427's scope registers `g23_readability:
sign_test` and its G2/G3 margins are 77–93%, orders outside any width discussed
here, so ABL-444's floored form is untouched by this read even though the
amendment covers it.

| pair | band | mean skill | t half-width | 95% t-CI | amended | ABL-427's registered floor | agrees? |
|---|---|---:|---:|---|:---:|---:|:---:|
| IT | 24-36h | +3.99% | 3.021 | [+0.96, +7.01]% | **`A`** | 4.772 → `U` | no |
| IT | 36-48h | +2.69% | 3.052 | [−0.37, +5.74]% | `U` | 4.756 → `U` | yes |
| IT | 48-64h | +6.29% | 3.018 | [+3.27, +9.31]% | `A` | 4.885 → `A` | yes |
| HR | 24-36h | +5.39% | 3.164 | [+2.23, +8.56]% | `A` | 5.072 → `A` | yes |
| HR | 36-48h | +5.37% | 3.114 | [+2.26, +8.48]% | `A` | 4.991 → `A` | yes |
| HR | 48-64h | +4.62% | 3.152 | [+1.47, +7.77]% | **`A`** | 5.012 → `U` | no |

**Pair letters under the amendment: IT `U` (36-48h unresolved), HR `A`.**

Two cells move, both from `U` to `A`, and only HR's changes a pair letter. This is
exactly ABL-427 §7.3's published prediction; it is restated here so that the
re-grade in §6 is checked against a number registered before it ran.

**A caveat that ABL-427 states and this registration inherits:** the three bands
of one country share a fit, so they are **not three independent estimates**. "HR
clears in all three bands" is not three independent 95% tests, and the pair letter
should not be read as if it were.

---

## 4. The alternatives, priced on the same six cells

The t interval assumes the per-seed skill draws are normal. With 12 draws that is
not testable to useful power — Shapiro-Wilk returns p = 0.11 to 0.55 on all six,
which is *failure to reject*, not evidence of normality. The honest response is to
price the tests that do not make the assumption.

| cell | mean | **t** | **Wilcoxon** | **bootstrap** (20k, percentile) | **sign** |
|---|---:|:---:|:---:|:---:|:---:|
| IT 24-36h | +3.99% | `A` (p=0.014) | `A` (p=0.034) | `A` [+1.26, +6.37] | `A` (10/12, p=0.039) |
| IT 36-48h | +2.69% | `U` (p=0.079) | `U` (p=0.052) | `U` [−0.02, +5.11] | **`A`** (10/12, p=0.039) |
| IT 48-64h | +6.29% | `A` (p=0.001) | `A` (p=0.002) | `A` [+3.65, +8.76] | `A` (11/12, p=0.006) |
| HR 24-36h | +5.39% | `A` (p=0.003) | `A` (p=0.007) | `A` [+2.56, +7.91] | `A` (10/12, p=0.039) |
| HR 36-48h | +5.37% | `A` (p=0.003) | `A` (p=0.007) | `A` [+2.61, +7.90] | `A` (10/12, p=0.039) |
| HR 48-64h | +4.62% | `A` (p=0.008) | `A` (p=0.012) | `A` [+1.84, +7.19] | **`U`** (9/12, p=0.146) |
| **pair** | | **IT `U` / HR `A`** | **IT `U` / HR `A`** | **IT `U` / HR `A`** | **IT `A` / HR `U`** |

**t, Wilcoxon and the bootstrap agree on all six cells.** Two of the three make no
normality assumption at all, so the assumption is stated but is **not load-bearing
on this read**.

**The sign test is the lone dissenter, and it dissents in both directions** — it
passes IT 36-48h and fails HR 48-64h, exactly inverting the pair verdicts. That is
not a tie-break to take seriously: it discards magnitude entirely, and at n = 12
its attainable p-values are so coarse (9/12 → 0.146, 10/12 → 0.039, nothing
between) that it has about three usable outcomes. It cannot distinguish IT
36-48h's +2.69% ± 4.80 from HR 48-64h's +4.62% ± 4.96 except by counting signs.

**Why t rather than Wilcoxon or the bootstrap**, given they agree here:

* **t is the exact test under its assumption**, and the assumption is the one the
  whole ladder already makes implicitly — `delta_min` is a normal-theory interval
  too, with an imported sd rather than a measured one. Choosing t is *reducing*
  the number of unstated assumptions, not adding one.
* **It is deterministic and closed-form.** A registered verdict must reproduce
  exactly. The bootstrap does not: across 10 RNG seeds IT 36-48h's lower bound
  ranged −0.117 to −0.005, verdict-stable but sitting on zero. A gate whose letter
  depends on a resampling seed is a gate ABL-385 exists to prevent.
* **It costs no dependency.** `T_CRIT_95` is pinned in the module and checked
  against `scipy` in tests, so no verdict can move on a library upgrade. Wilcoxon
  would put `scipy` on the decision path of every graded cell.
* **Wilcoxon is the standing fallback**, recorded here rather than implemented: if
  a future k > 1 read shows real skew in its draws, it is the test to amend to,
  and it agreed on all six cells here.

---

## 5. What is registered

* `SEED_READABILITY` in both gate harnesses, values `delta_min` / `student_t`.
  **Every published scope is pinned to `delta_min`**, including
  `abl427-t2c-reread` itself. An unregistered scope inherits `student_t`.
* `grade_cell` takes `seed_wapes` — the challenger's per-seed WAPEs — and
  `seed_readability`. Omitting the draws changes nothing.
* `CellGrade` gains `readability_test`, `half_width_pct` and `seed_interval`,
  **serialised only where the read was seed-decided**, on ABL-434's rule that the
  key's presence dates the read. `floor_pct` stays on both forms so the two widths
  are comparable from the record alone.

### 5.1 Passing the draws, not an interval

The issue asked for this to be decided deliberately. **The draws are passed.**

* The ladder then owns the **one** implementation of its own test. A caller
  handing in an interval could have built it one-sided, or with `z`, or with the
  wrong `df`, and `grade_cell` could not tell.
* An "interval" is really **three** intervals, one per reference, and their
  selection depends on the scope's `levelling` — so a precomputed form leaks the
  ladder's own registration into every caller.
* The draws are the raw evidence and go into the record, so the read is checkable
  by hand, which is the standard the rest of this module holds.

Two guards make the draws provably the cell's own: `len(seed_wapes) != k` raises,
and draws whose mean is not the cell's recorded challenger WAPE raise. The second
catches a paste from another cell, which is otherwise silent and would centre the
interval on one cell's mean while grading another's margin.

### 5.2 `grade_cell` and ABL-434's property

ABL-434 (PR #79, merged into `main` at `ca3c7f8` while this issue was in flight)
registers that `grade_cell` stays a function of `scores` alone, so that published
margin-only re-reads reproduce byte-for-byte. **That property is preserved.**
`seed_wapes` defaults to `None`; with it omitted the function is the one ABL-434
describes, and §5.4's 1,568 replays are measured against ABL-434's own merged
module rather than the one this branch was cut from. The draws are read off the
*cell* by `cell_grade` and `attach_grades` — the same two functions ABL-434 uses
for coverage, and for the same reason: they hold a whole cell and `grade_cell`
does not.

The two amendments compose in one place worth naming: a cell held at `X` for
coverage **keeps** the interval its margin was read against. `X` is a statement
about the rows, not a reason to discard the measurement that was taken, and a
record that dropped the width could not say what the margin had been judged
against. Pinned by
`test_a_coverage_held_cell_keeps_the_interval_its_margin_was_read_against`.

The distinction from ABL-434's coverage gate that matters: coverage is one-way and
needs no per-scope table; **this is not one-way**, so it has one.

### 5.3 Why the table is not in `check_registration_tables`

Same structural reason as `CAUSAL_LEVELLING` and `G23_READABILITY`: that check
requires every scope in the union to appear in every table it is given, so
requiring this one would force each scope to be pinned and delete the
default-toward-amendment behaviour.

The hazard is answered differently from ABL-444's, because this fall-through is
the *less* conservative direction. It is answered by **k**. A fall-through row can
only bind a read at k > 1; at k = 1 there are no degrees of freedom and
`grade_cell` uses `delta_min` whatever the table says. Published scopes are pinned
by **value** in `test_every_published_scope_pins_delta_min`, which is strictly
stronger than the presence check the call would give.

### 5.4 Blast radius, measured rather than asserted

The issue says "no other committed tranche cell is at k > 1" and asks for that to
be verified. It was, three independent ways, against `main` at `ca3c7f8` — which
**already contains ABL-427's record**, so this is a measurement rather than a
prediction:

1. **No call site anywhere passes k > 1** to `readability_floor_pct`,
   `grade_cell`, `cell_grade`, `attach_grades` or `pair_grade`.
2. **631 committed graded cell-records** across `reports/` and `experiments/`.
   **613** carry `floor_pct` of exactly `10.6482` (solar, 305) or `7.5054` (wind,
   308) — the k = 1 floors. The other **18 are all in one file**,
   `reports/abl_427_tranche2c_seed_reread.json`: six cells published under three
   candidate floors each. **No other file carries a non-k=1 floor at all.**
3. **1,568 replays**: every one of the 196 committed `scores` blocks re-graded
   under the amended module across both streams, both levellings and both G2/G3
   forms produced records **byte-identical** to the pre-amendment module (compared
   against `origin/main`'s `gate_grading.py`, i.e. after ABL-434).

The issue's claim is therefore exactly right, and `abl427-t2c-reread` is pinned to
`delta_min` so its published letters stand. The assertion in
`tests/test_abl467_seed_interval.py` is an **equality**, not a bound, so a second
k > 1 read landing anywhere goes red until someone names it.

### 5.5 A red `main`, found and repaired here

**`origin/main` was already failing its own test suite before this branch
touched it**, and the repair is in this PR because leaving it red while stacking
on top of it is worse than fixing it.

ABL-427 (PR #80) and ABL-434 (PR #79) merged back to back on 2026-08-14. #80
landed a fifth `grade_cell` caller — `scripts/abl427_tranche2c_seed_reread.py` —
and #79 landed `MARGIN_ONLY_READERS`, the registry that has to name every such
caller. Each branch was green on the base it was cut from; neither could see the
other. `test_every_ungated_caller_is_registered_with_a_reason` has been failing on
`main` since `ca3c7f8`, on a tree neither author ever ran. **Verified by checking
out `origin/main` detached and running that file alone**, before making any change.

The fix is the registry entry ABL-434 would have written had the script existed:
ABL-427's read is margin-only and **fully covered** — all six cells clear their
registered minimum n (720/684, 720/684, 510/456, and the same three for HR,
checked against the record, not assumed) — so routing it through the coverage gate
would change no letter. This is a repair, not a new decision to publish an ungated
grade.

---

## 6. Order of work

Committed in this order, as ABL-437 and ABL-444 were: **this report and
`experiments/ABL467/config.json` first, the amendment and its tests with them, and
the re-grade in a separate commit afterwards.** The re-grade is a new scope,
`abl467-t2c-regrade`, and a new document; ABL-427's committed record is not edited
or regenerated.

**The input is on `main`.** `reports/abl_427_tranche2c_seed_reread.json` merged as
PR #80 at `dbc37af` while this issue was in flight — it was open when this branch
was cut, and the registration was drafted against the branch copy. The re-grade
therefore reads the committed file and pins its blob hash
(`47e2d9a7fe1073bae84b695c4fbe206490fe6ef3`), so it is reproducible from this tree
alone and cannot silently read a different vintage. The registration and the
amendment in §5 have no dependency on it either way — their tests pin ABL-427's
six cells by raw per-seed WAPE, inline.

---

## 7. Caveats that travel with this

* **Normality is assumed and not testable at k = 12.** §4 is the answer: two
  distribution-free tests agree on every cell. If a future read's draws are
  visibly skewed, Wilcoxon is the registered fallback and amending to it is a new
  registration, not a run-time choice.
* **Bands within a country share a fit** and are not independent (ABL-427's
  caveat, inherited). A pair letter is not a joint 95% statement.
* **`c_B = 0` throughout.** A fitted reference on the ladder voids this.
* **Multiplicity is not corrected.** Each cell is a separate 95% test, exactly as
  `delta_min` was applied per cell; the amendment changes the width, not the
  number of tests. Six cells at 95% is not a 95% family-wise statement, and was
  not before either.
* **k > 31 falls back to `z`**, anti-conservative by at most 3.9% of the correct
  half-width and shrinking. No such read exists.
* **Contamination**, carried from ABL-427 and unchanged by an arithmetic
  amendment: ABL-67 is `net_position`-only; ABL-109 and ABL-111 are
  `energy_load`-only — none of those tables is read here. ABL-71's known
  wrong-write modes are load and net position, which is a provenance caveat on
  this window rather than proof that solar ingest is pristine. ABL-332's
  sub-hourly aggregation fires on both pairs and is the registered behaviour.
* **Window and n**, from ABL-427: fit 2026-01-14 → 2026-07-11, gate 2026-07-11 →
  2026-08-10, out-of-sample by target timestamp, n = 720 / 720 / 510 against
  registered minima 684 / 684 / 456. All six cells clear their minimum n, so
  ABL-434's coverage gate does not touch this verdict.

---

## 8. What this does not do

It promotes nothing. IT stays `U` and is not close. HR resolving `A` makes it
**promotion-eligible under ABL-418, which is necessary and not sufficient** — and
ABL-427 §5 measured the fact that should govern any such conversation: across the
twelve seeds, **1 to 3 fits per cell lose to seasonal-naive D-7 outright**, sd of
skill 4.75–4.98pp against mean margins of 2.69–6.29pp. A single-seed solar model on
these pairs carries roughly a 1-in-6 to 1-in-4 chance of being worse than the
baseline, decided by the seed. That is a serving question, it is not mine, and it
is the reason `draws_losing` is recorded on every amended cell.
