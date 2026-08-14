# ABL-438 — tranche 1b (BG/CH solar) retro-graded: A/A, and what that A is worth

**Evidence pack.** The machine-readable grades are
`reports/abl_438_retro_grade.json`; the generated table is
`reports/abl_438_retro_grade.md`. Both are produced by
`scripts/abl418_retro_grade.py --tranches 1b --issue ABL-438`, which runs
`src/evaluation/gate_grading.py` — the same ladder both gate harnesses call. This
document states what the numbers mean and what they do not.

**No refit, no re-read, no replica read, no write to any dispositioned path.**
Unlike ABL-435, nothing had to be recomputed: tranche 1b's committed record
already carried the full eight-comparator reference suite. The scores were on
disk. ABL-418 retro-graded 2a and 2b only, so 1b simply never had the ladder run
over it.

## 1. Result

Every cell of ABL-381's record grades **A** — G1–G4 hold in all three bands for
both pairs.

| pair | band | n | n ≥ min | gate | skill vs D-7 | vs constant causal | vs climatology causal | slope>0 & corr>0 | grade |
|---|---|---:|:---:|:---:|---:|---:|---:|:---:|:---:|
| BG | 24-36h | 720 | yes | PASS | +22.60% | +74.92% | +55.01% | yes | **A** |
| BG | 36-48h | 720 | yes | PASS | +23.77% | +75.30% | +55.70% | yes | **A** |
| BG | 48-64h | 510 | yes | PASS | +19.86% | +70.62% | +51.54% | yes | **A** |
| CH | 24-36h | 720 | yes | PASS | +35.59% | +91.42% | +78.25% | yes | **A** |
| CH | 36-48h | 720 | yes | PASS | +36.81% | +91.58% | +78.66% | yes | **A** |
| CH | 48-64h | 510 | yes | PASS | +32.98% | +90.24% | +77.04% | yes | **A** |

Pair grades: **BG = A**, **CH = A**.

- **Window:** ABL-348's frozen gate window, 2026-07-11 → 2026-08-10 (exclusive).
  Fit window 2026-01-14 → 2026-07-11. Target series `energy_generation`.
- **n:** 720 per cell in the two 36h bands, 510 in 48-64h, against registered
  minima of 684 and 456. All six clear; the tightest is ×1.05.
- **Out-of-sample.** The gate window is disjoint from the fit window.
- **Baseline:** seasonal-naive D-7, the registered bar, plus the seven other
  comparators ABL-381 recorded.
- **Verdict unchanged.** ABL-381 published PASS 6/6 and it still reads PASS 6/6.
  A grade reads a disposition; it does not replace one.

**This reproduces the reading stated in the ABL-438 description before the run,
cell for cell, including `enough_pairs`.** Nothing disagreed, so nothing needed
naming. It also reproduces the ABL-381 probe numbers to 2 dp on every comparator
— challenger 18.885 / D-7 24.399 / constant causal 75.303 / constant oracle
73.492 / climatology causal 41.980 / climatology oracle 19.155 for BG 24-36h —
so the committed record and the probe that motivated the reference suite are the
same measurement.

## 2. What the A is worth — two qualifiers that travel with it

Grade **A** reads *promotion-eligible, subject to any named data hold*. On these
two pairs there are two qualifiers the ladder cannot see, and neither is
optional.

### 2.1 Both pairs beat the oracle climatology only *inside* the readability floor

Solar's null model is an hour-of-day climatology, not a flat line. The +70 to
+91% margins over `constant_causal` above look decisive and are not: a flat line
cannot represent a diurnal cycle at all, so on solar it measures that the sun
rises. Against the *oracle* hour-of-day climatology — hindsight, therefore not
causally available, therefore gating nothing — both pairs still win in every
band, but by:

| pair | worst band | margin over `climatology_oracle` | absolute | readable at k=1? |
|---|---|---:|---:|:---:|
| BG | 24-36h | **+1.41%** | +0.27pp | **no** |
| CH | 48-64h | **+3.47%** | +0.30pp | **no** |

against a solar readability floor of **10.6482%** at k = 1 (ABL-385's
`delta_min` with `c_B = 0`, correct because every reference here is
deterministic). BG's best band reaches +2.90% and CH's +11.20%; only CH 36-48h
is readable, and a pair takes the worst of its bands.

**This does not move the gate.** An oracle is not causally available and never
gates — G1 is scored against D-7 and holds by +19.9 to +36.8%, far outside the
floor. What it bounds is the *claim*: these pairs are not established to beat a
properly-levelled hindsight climatology by an amount anyone can read at one seed.
Per ABL-417, the floor applies to any margin a reader ranks on, not only to the
one the ladder gates on.

### 2.2 BG carries a live night-contamination hold (ABL-396)

76–85% of BG's geometric-night hours carry above 1 MW, up to 1,097 MW — roughly
5–6% of all energy booked after dark. It is **identical on both actuals tables**,
so it is upstream of this module and not a scoring artifact. ABL-396's screen
found BG an outlier by 3.7× and **the only country whose displacement band is
wide enough to threaten a verdict**.

That band is far wider than BG's +1.41% margin over the oracle climatology in
§2.1. **A grade of A must not be reported for BG solar without this line
attached.** It is registered as data (`HOLDS` in the retro-grade script), carried
into `reports/abl_438_retro_grade.json`, and rendered under BG's table — not left
in a comment, because a hold that lives in a comment is a hold the next reader
does not get.

CH carries no such hold.

### 2.3 The bar-weakness flag reads `False` here, and it survives ABL-437

`bar_weaker_than_a_flat_line` is `False` for all six cells — the registered D-7
bar is *not* clearable by a flat line on either pair. That flag is worth
distrusting on its own, because it is read off `constant_causal`, and ABL-437 (in
progress) is measuring `constant_causal` as inflated by up to 205% where the fit
and gate windows sit at different levels. An inflated constant makes the bar look
*stronger* than it is, so a `False` here could in principle be a false negative.

**It is not, on these cells.** The oracle constant is levelled on the gate
window's own rows by construction, and it is still far worse than the bar:
73.49% vs 24.40% (BG 24-36h) and 94.65% vs 12.67% (CH 24-36h). Even perfectly
levelled, a constant does not come close to clearing D-7 on solar — which is
§2.1's point from the other direction. Whatever ABL-437 concludes about
`constant_causal`, it cannot turn these two bars weak.

**Contamination touching this window:** ABL-396 as above, on BG. ABL-71, ABL-67
and ABL-111/ABL-109 concern net position and actual load and do not touch these
solar cells.

## 3. What was done to the code, and what was deliberately not

**One grader, one more row.** `--tranches` was added to the existing
`scripts/abl418_retro_grade.py` rather than a second grader being written. The
registry gained a row for 1b; the arithmetic gained nothing. The script now also
**refuses** to write ABL-418's output paths under any non-default selection — the
`SCOPE_OUTPUTS` failure in CLAUDE.md is exactly this shape one directory over,
where a scoped run kept a default path, rewrote a dispositioned record under its
own heading, and exited 0.

**Byte-unchanged, verified by blob hash.** All **339** tracked files under
`experiments/` and `reports/` have blob hashes identical to the merge base
(`a838894`) — zero mismatches. That includes
`experiments/ABL348/results_abl381_tranche1b.json`, whose SHA-256
`6ff1629cc4525683de630c72ec04dac1658b045da6cf0847f6d9c6f8f3e6184a` is recorded in
the graded record so a later reader can tell which bytes were graded, and
`reports/abl_418_retro_grade.{md,json}`, which this change does not touch. The
only tracked file modified is the script itself.

**ABL-418's grades did not move.** Regenerating tranches 2a and 2b through the
extended script and stripping the four fields this change adds
(`minimum_n`, `enough_pairs`, `oracle_margin_readable`, `hold`, plus
`issue` / `reported_comparators` / `holds` / `tranche_selection`) gives a JSON
identical to the committed `reports/abl_418_retro_grade.json`. All 48 cell
grades, all 16 pair grades and both denominator-sensitivity lists are unchanged.
Pinned by `test_abl418s_own_selection_still_produces_abl418s_own_grades`.

## 4. One finding that belongs to ABL-418, not to this issue

Regenerating ABL-418's own report through the extended renderer changes three
cells of its **presentation** — no grade, no verdict. Three pairs that its
published table reports as a clean "beats climatology oracle: **yes**" win only
*inside* the floor:

| tranche | pair | worst-band margin over `climatology_oracle` | floor |
|---|---|---:|---:|
| 2a | CH | +8.15% | 10.65% |
| 2a | RO | +5.93% | 10.65% |
| 2b | FI wind_onshore | **+7.48%** | 7.51% |

FI is inside its floor by 0.03pp. This is the same qualifier §2.1 applies to 1b,
and it is the ABL-417 lesson again: apply the floor to any margin you rank on.

**`reports/abl_418_retro_grade.md` is dispositioned evidence and this change does
not rewrite it.** The consequence is a drift worth naming rather than leaving to
be discovered: regenerating that report now produces a file that differs from the
committed one — by the added `n ≥ min` column, the coverage note, and these three
qualifiers. Whether ABL-418's report is regenerated is the CEO's call, filed
separately.

## 5. Boundary

No promotion, no serving-registry change, no ingest change, no dashboard change,
no refit, no replica read, no replica write, no sidecar write. A grade is not a
promotion recommendation and does not become one. **Recommendation: none from
this issue** — the grades are recorded so a promotion discussion can start from a
number rather than from a comment, and BG's ABL-396 hold is live.
