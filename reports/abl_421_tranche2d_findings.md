# ABL-421 — ABL-316 tranche 2d: the six northern solar countries

**The final solar tranche.** EE, FI, LT, LV, NL, SE, read under ABL-348's frozen
pair — fit `2026-01-14 → 2026-07-11`, gate `2026-07-11 → 2026-08-10`, WAPE against
literal seasonal-naive D-7, source `energy_generation`, replica read-only
(9,432,453,120 bytes), 27 features, `exclude_impossible_night: False`, CatBoost at
`random_seed` 42.

Machine record: `experiments/ABL348/results_abl421_tranche2d.json`. Harness report:
`reports/abl_421_solar_tranche2d.md`. Generated tables:
`reports/abl_421_tranche2d_tables.md` / `.json`, regenerable with
`.venv\Scripts\python.exe scripts/abl421_tranche2d_read.py` and derived from the
stored record with no refit.

---

## 1. Headline

**Verdict `FAIL`, 12 of 14 evaluable cells clear the bar — and neither failure is
a loss to D-7.** Both are coverage shortfalls on pairs that beat the baseline
comfortably.

| | |
|---|---|
| grid | 6 countries × 3 bands = **18** |
| declared NOT-EVALUABLE by ABL-348 | **4** (EE, FI on 24-36h and 36-48h) |
| evaluable, and the bar | **14** |
| cells clearing D-7 at the registered minimum n | **12** |
| cells failing | **2** — EE 48-64h, FI 48-64h, **both on n, not on skill** |

Reported pair grades, ABL-418 ladder:

| pair | reported | ladder | why |
|---|:---:|:---:|---|
| LT | **A** | A | 3/3 bands, clears every causal reference |
| LV | **A** | A | 3/3 bands, and the **only pair that beats all four references** |
| SE | **A** | A | 3/3 bands, clears every causal reference |
| NL | **B** | B | fails G3 — loses to hour-of-day climatology on all three bands |
| EE | **—** | A | no band meets the registered minimum n |
| FI | **—** | A | no band meets the registered minimum n |

**No promotion is recommended from this read, for any pair.** This is an evidence
pack; promotion is a CEO/Board decision on a pre-registered gate read, and three
grade-A pairs plus the caveats below are what it has to work with.

---

## 2. The scope is 14 cells, not 18 — and the harness could not previously say so

The issue was scoped as "6 countries × 3 bands = 18 cells". Measured against the
frozen registration it is **14**. ABL-348 `not_evaluable` declares **EE/solar and
FI/solar NOT-EVALUABLE on 24-36h and 36-48h**, before any fit existed, with:

> "A pair listed here is reported NOT-EVALUABLE on the named bands. It is not a
> FAIL and must not be counted as one; a gate read that scores it has misread
> this registration."

**The harness had no way to obey that rule.** `gate_cell` builds a cell for every
country-band that yields rows and marks it `pass: False` when `n` falls under the
registered minimum. So those four cells would each have arrived as an ordinary
*failed* cell and been counted into `passed/18`, rendering `FAIL 10/18` — a
model-quality verdict on a comparison the registration forbids, with nothing in
the exit status to show it. Tranches 2a, 2b and 2c dodged this by excluding both
pairs (`abl316-t2c`'s registration says so in as many words); 2d is the tranche
they belong to.

`SCOPE_NOT_EVALUABLE` is therefore registered in this issue. Declared cells are
subtracted from the bar and routed to a `not_evaluable_cells` list that `passed`,
`disposition` and `attach_grades` never read — still measured and printed, because
a declaration nobody can check is indistinguishable from a challenger quietly
dropped for scoring badly, but carrying no gate outcome and no grade. The table
defaults to empty, so **all five earlier scopes are the identity they always
were**.

What the declared cells measured, for audit only:

| country | band | n | min n | challenger | D-7 | (skill) |
|---|---|---:|---:|---:|---:|---:|
| EE | 24-36h | 543 | 684 | 23.80% | 36.22% | +34.3% |
| EE | 36-48h | 540 | 684 | 24.01% | 36.26% | +33.8% |
| FI | 24-36h | 629 | 684 | 26.29% | 38.12% | +31.0% |
| FI | 36-48h | 628 | 684 | 25.64% | 38.12% | +32.7% |

**Only one of the two shortfalls is ours.** EE's is an ABL-188 bit-identical zero
run (2026-07-21 → 2026-07-22, 44.8h) present identically in **both** source
tables, `source_dependent: false` — reverting ABL-348's source change would not
recover it. **FI's is caused by that source change**: `energy_generation` holds
663 of the 720 gate hours against `energy_renewable`'s 717, the ABL-322 §3.3
phenomenon on a second pair, `source_dependent: true`. That is a finding for
whoever owns the source decision, not a fact about FI's model.

---

## 3. Both failures are coverage, not skill — and FI misses by three rows

ABL-348's `note_48_64h` instructs that the 48-64h band is **not** hard-bounded by
`n_d7_scorable` and that a declared pair "may still clear 456 in that band and
should be reported if it does". So EE's and FI's 48-64h cells were kept on the
bar. They were read, and they did not clear:

| cell | n | min n | challenger | D-7 | skill | `beats_d7` | `enough_pairs` |
|---|---:|---:|---:|---:|---:|:---:|:---:|
| EE 48-64h | **388** | 456 | 25.05% | 35.29% | **+29.0%** | `true` | `false` |
| FI 48-64h | **453** | 456 | 23.99% | 37.98% | **+36.8%** | `true` | `false` |

Both **beat D-7 by far more than the 10.65pp readability floor** and both fall
short of the row minimum — FI **by three rows**. The registered verdict is `FAIL`
and stays `FAIL`: the bar is not re-opened after seeing a result, and the
harness's own recommendation line ("report the losing country/bands as the
finding") is the registered text. But the finding is *not* that EE and FI lost a
race. It is that they were under-covered, and `gate_cell` keeps `beats_d7` and
`enough_pairs` separate precisely so that this distinction survives into the
record.

Pursuing feature work on EE or FI off the back of this `FAIL` would be the exact
mistake `UNREADABLE` exists to prevent, one notch further down.

---

## 4. A gap in the ABL-418 ladder, surfaced for the first time

**EE and FI grade `A`.** ABL-418's `grade_cell` is handed a cell's `scores` and
nothing else — it never sees `gate.enough_pairs` or `gate.n` — so it grades a
*margin*, and both margins are excellent. But a margin the registration does not
consider readable cannot carry a promotion, and `A` means promotion-eligible.

This combination had **never occurred before**: every cell in tranches 2a, 2b and
2c met its minimum n, so the ladder has never been exercised against it.

Handled here by naming a **hold** in the ladder's own vocabulary — `A` is defined
as promotion-eligible *subject to any named data hold* — reporting EE and FI as
`—` with `no band meets the registered minimum n`, and printing the ladder grade
beside it. `gate_grading.py` is deliberately **not** edited: changing the ladder
after seeing a result is the shopping the pre-registration apparatus exists to
prevent, and that module is shared with the wind harness. Whether the ladder
should read `enough_pairs` itself belongs in its own pre-registered issue.

---

## 5. NL: the level is the finding, not the sign

NL was flagged twice in the issue. Both were checked; the second is the one that
matters.

**The signed target is arithmetically negligible, with a number.** NL solar is
negative at every night hour, 1,544 of 1,544 across both windows. `score_predictions`
uses `denom = sum(|actual|)`, so a negative night hour contributes its *magnitude*
rather than cancelling against daylight. The two conventions differ by exactly
NL's absolute night share, **0.040%** of the denominator (the signed share is
−0.040% — same magnitude, opposite sign, which is the tell). Zeroing the night
would raise WAPE by that relative amount: **0.019pp** on NL's 46.53% bar, against
a 10.65pp floor. The numerator is bounded the same way — night actuals average
−0.13 MW. **NL's margin is not a netting artefact.**

**NL's level is.** Gate-window mean **66.4 MW**, window maximum **251.3 MW**, and
the series is **bit-identical in both source tables**, so it is upstream rather
than ours. For scale, over the same 720 hours `energy_generation` books BE at
8,140 MW max, DE at 54,457 MW, and even EE — a country of 1.3 million — at
771.6 MW. NL's published solar series peaks between 139 MW and 425 MW in every
one of the last 18 months, with a clean seasonal shape: it is a small metered
subset behaving like solar, not the Dutch fleet.

Consequences, in order:

1. The gate read is valid **of that series** and internally consistent.
2. It must **not** be quoted as "we can forecast NL solar".
3. NL is the **only** pair below the SK reference line (66.7 MW against SK/solar's
   114.8 MW, the lowest solar fleet already dispositioned). The issue anticipated
   "several"; measured, EE is next lowest at 223.0 MW, nearly twice SK.
4. Any NL promotion recommendation has to carry all of the above — and NL is
   graded `B` regardless, for the independent reason below.

---

## 6. NL loses to the average day; and only LV beats all four references

**NL clears D-7 on all three bands and loses to a causal hour-of-day climatology
on all three** — 36.97 / 37.66 / 36.82% against 34.79 / 34.79 / 34.75%. That is
the ABL-380 defect in its registered form, and G3 caught it: grade **B**. A pair
that cannot beat "what this hour of day usually does" has not earned a promotion
against a 46.53% D-7 bar.

**On solar the constant is a formality.** A flat line scores 80.4–103.2% across
this tranche — NL's causal constant is *above 100%*, worse than predicting zero —
because a constant cannot represent a diurnal cycle and on solar the diurnal cycle
is the signal. So `bar weaker than a flat line? no` is uninformative here for all
six pairs; the climatology is the real test, exactly as CLAUDE.md records.

Against all four model-free references, worst band per pair:

| pair | worst band | clim causal | clim oracle | const causal | beats all four? |
|---|---:|---:|---:|---:|:---:|
| EE | 25.05% | 29.14% | **23.29%** | 80.42% | **no** |
| FI | 23.99% | 45.07% | **22.56%** | 82.52% | **no** |
| LT | 20.94% | 44.80% | **17.47%** | 90.88% | **no** |
| LV | 32.17% | 36.01% | 33.89% | 89.91% | **yes** |
| NL | 37.66% | **34.75%** | **32.57%** | 85.18% | **no** |
| SE | 21.16% | 40.66% | **17.89%** | 87.49% | **no** |

**1 of 6 beats all four (LV).** Every other shortfall is against the **oracle**
climatology, which is causally unavailable and gates nothing — so it qualifies
those reads rather than overturning them. This reproduces ABL-417's finding on a
second stream: an A-graded pair that loses to the average day in hindsight is a
different object from one that does not, and the pass rate alone does not say
which you have.

---

## 7. EE's night floor, bounded

EE carries the third-largest solar night floor in the fleet, `f` = **0.7180%** of
gate-window energy (68 of 86 gate night hours above 1 MW, 12.64 MW mean). The
other five are at or under 0.042%.

| band | status | n | all-hours `A` | implied daylight `W` | `[W(1-f), W(1-f)+f]` | clamped `[A, A+f]` |
|---|:---:|---:|---:|---:|---:|---:|
| 24-36h | NOT-EVALUABLE | 543 | 23.80% | 23.25%–23.97% | 23.08%–24.52% | 23.80%–24.52% |
| 36-48h | NOT-EVALUABLE | 540 | 24.01% | 23.46%–24.19% | 23.30%–24.73% | 24.01%–24.73% |
| 48-64h | gated | 388 | 25.05% | 24.51%–25.23% | 24.33%–25.77% | 25.05%–25.77% |

For a known `W` the band has width exactly `f` = 0.718pp; `W` is bounded rather
than measured here, so the printed envelope is that band across the whole implied
`W` range. **The widest interval is 1.436pp against a 10.65pp floor — a factor of
7 — so EE's night floor cannot move a grade in either direction.** That is now
measured rather than assumed, and it is the only quantitative statement this
tranche makes about EE's two declared bands.

ABL-425 (open, PR #59) independently registers `EE: False` in
`NIGHT_GENERATION_POSSIBLE` — EE's floor is contamination, not real generation —
consistent with bounding it here rather than adjusting for it. That branch's guard
does not bind this scope: it fires only when `exclude_impossible_night` is **on**,
and 2d registers it off.

---

## 8. Contamination touching this window

- **ABL-188** (zero-fill / bit-identical constant runs) — **touches EE directly**
  and is the declared cause of its NOT-EVALUABLE bands. The harness's own
  `constant_runs` screen ran against `energy_generation`, the table fitted and
  scored on.
- **ABL-322 §3.3** (`energy_generation` holding fewer hours than
  `energy_renewable`) — **touches FI**, 663 of 720 gate hours, and is the declared
  cause of its NOT-EVALUABLE bands. Source-dependent.
- **ABL-71 / ABL-67 / ABL-111 / ABL-109** — net-position and actual-load issues;
  none touches a solar target in this window.
- **NL's negative night hours** — our own netting rule, quantified in §5 at
  0.040% of the denominator. Not a defect in this read.

---

## 9. The numbers survive the merge, and here is the proof

These fits were run against the tree at `18301b2` (tranche 2c's head). While they
ran, `main` gained ABL-425, ABL-417 and an ABL-403 follow-up, and ABL-425 edits
`src/solar_geometry.py`, `src/solar_features.py` and `src/db.py` — three files
that between them produce two of this challenger's 27 features, load its target
series, and aggregate it to hourly. A gate record computed on one tree and merged
into another is worth nothing unless that gap is closed.

It is closed by comparison rather than by assertion. Every function and constant
this read depends on was AST-compared between `18301b2` and `origin/main` with
docstrings stripped:

| symbol | verdict |
|---|---|
| `solar_geometry.sun_elevation_deg` | identical |
| `solar_geometry.is_night_hour` | identical (docstring only) |
| `solar_geometry.SOLAR_REPRESENTATIVE_POINTS` | identical |
| `solar_geometry.NIGHT_ELEVATION_THRESHOLD_DEG` | identical |
| `solar_features.night_mask` | identical (docstring only) |
| `solar_features._solar_geometry_features` | identical |
| `solar_features.SOLAR_GEOMETRY_FEATURES` | identical |
| `db.load_renewable_type_data` | identical |
| `db.aggregate_renewable_to_hourly` | identical |
| `db.RENEWABLE_TYPE_COLUMNS` | identical |

ABL-425's additions are a new registry (`NIGHT_GENERATION_POSSIBLE`) and a new
exception, both reached only when `exclude_impossible_night` is **on** — and this
scope registers it off. With `random_seed` pinned at 42 and every input function
unchanged, the fit is deterministic and these numbers are the numbers the merged
tree produces. Full suite on the merged tree: **1128 passed**.

Note that two of the ten symbols moved *textually*. Comparing raw source, or an
`ast.dump` that still carries docstrings, reports them as changed and would have
sent this to an 18-minute re-run for nothing; comparing the code says why they did
not move. That is the cheaper and the stronger statement.

## 10. What this does and does not license

- **No promotion, no serving-registry change, no ingest change.** Evidence only,
  per the issue's boundary. No write to `forecasts`; artifacts land in
  `experiments/ABL421/artifacts/` and are gitignored by `.gitignore:56`.
- **No dispositioned scope's evidence was touched.** This scope's three output
  paths are disjoint from every other scope's, enforced by `check_scope_outputs`.
- **Do not average this tranche's pass rate against 2a's or 2c's.** These are the
  loosest solar bars in the programme (23.92–47.85%) against 2c's 7.11–16.43%.
  ABL-348 registered that reading in advance under
  `reading_caveats_not_band_changes`.
- **Two things are handed on rather than settled here:** whether the ABL-418
  ladder should read `enough_pairs` (§4), and whether FI's source-dependent
  coverage loss is acceptable to the owner of ABL-348's source decision (§2).
