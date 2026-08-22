# ABL-426 — tranche 2a re-read on the registered `energy_generation`

Parent **ABL-316**. Registration **ABL-348** (`experiments/ABL348/config.json`),
**unchanged and not reopened**. Scope `abl316-t2a-generation` = BG, CH, CZ, HU,
PL, RO, SI, SK on `energy_generation`. **8 countries × 3 primary D+2 bands = 24
cells.**

Machine record: `experiments/ABL348/results_abl426_tranche2a_generation.json`.
Harness report: `reports/abl_426_solar_tranche2a_generation.md`.
Arm difference: `reports/abl_426_source_arm_delta.json`
(`scripts/abl426_source_arm_delta.py`).
Replica-vintage screen: `reports/abl_426_vintage_screen.json`
(`scripts/abl426_vintage_screen.py`).

> **No promotion is requested or implied.** No serving-registry change, no write
> to `forecasts`, no ingest change, no dashboard change, no replica write, no
> sidecar write. The replica was opened `mode=ro` throughout. Promotion is a
> CEO-to-Board decision and this pack is evidence for it, not a step in it.

---

## 1. What this read is, and what it is not

ABL-405 read tranche 2a on `energy_renewable`. ABL-348 registers
`energy_generation` for all 37 tranche pairs and lists the source table under
`voids_this_registration`. This is the same eight countries read on the table the
registration names.

**It is not a re-grade of ABL-405, and it does not withdraw ABL-405's
disposition.** That read is published, dispositioned (PERFORMANCE PASS — HOLD FOR
CONTAMINATION ADJUDICATION) and retro-graded by ABL-418 across all 24 cells. It
keeps its scope, its outputs, its machine record and its letters. What ABL-426
adds is a second, compliant read beside it and a measurement of the difference.

**It opens no new registration.** Windows, bands, metric, baseline, minimum n and
the NOT-EVALUABLE list are ABL-348's, unchanged and deliberately not restated
here. A new *scope* is not a new pre-registration: it is how this harness cites an
existing one while writing somewhere new.

**It is a controlled A/B on the source table alone.** That is the whole design,
and §2 is the proof rather than the promise.

---

## 2. The controls, and why they are asserted rather than described

Two gate reads are comparable only if everything registered except the variable
under test is identical. Held identical between `abl316-t2a` and
`abl316-t2a-generation`, each pinned by a test in
`tests/test_abl426_scope_sources.py` rather than by this paragraph:

| registered value | held by |
|---|---|
| countries, and their order | `test_tranche2a_generation_is_tranche2a_on_the_registered_table` |
| `GATE_BASIS` — `challenger`, `seasonal_naive` | `…holds_every_other_registered_value[GATE_BASIS]` |
| `FIT_RULES` — `exclude_impossible_night: False` | `…[FIT_RULES]` |
| `CAUSAL_LEVELLING` — `fit_window` | `…[CAUSAL_LEVELLING]` |
| `G23_READABILITY` — `sign_test` | `…[G23_READABILITY]` |
| `SEED_READABILITY` — `delta_min` | `…[SEED_READABILITY]` |
| feature vector — **both arms absent** from `SCOPE_FEATURES` | `test_neither_arm_pins_a_table_the_other_inherits` |
| output paths — disjoint | `test_the_corrected_read_writes_nowhere_the_published_read_writes` |

The feature vector deserves its own sentence, because the safe-looking move is
the wrong one. Pinning this scope to `FEATURE_COLUMNS` would *look* like a pin and
would not be one — it binds to the same mutable constant the default binds to —
while flipping `meta.feature_set_is_registered_for_scope` to `True` in one arm and
leaving it `False` in the other. That is a difference between the two **records**
that is not a difference between the two **reads**. Both arms therefore inherit
`DEFAULT_SCOPE_FEATURES`, which makes an identical 27-column vector structural
rather than maintained: in the same process, on the same commit, they cannot
disagree. This is ABL-404's argument, applied to the scope ABL-404 was about.

`scripts/abl426_source_arm_delta.py` re-checks all of this from the two machine
records at comparison time (`controlled.all_controls_hold`), so the claim survives
a later edit to either scope's rows.

---

## 3. The defect, reproduced — and audited across the whole programme

Re-read at `origin/main` (`bb29d4b`), not grepped from a working tree.

`main()` resolves the source **once** and hands that one value to the fit, the lag
and rolling features, the D-7 and persistence baselines, the gate actuals and the
ABL-188 screen, then records it as `meta.training_source`. ABL-405's record says
`energy_renewable`, so the run was made without `--renewable-source
energy_generation` and fell through to `db.RENEWABLE_TYPE_SOURCE_TABLE`.

ABL-348 anticipated this failure **in advance**, under `harness_prerequisite`:

> "ABL-345 gives both gate harnesses `--actuals-source`. Until it lands, a harness
> run for these pairs fits on `energy_renewable` through the global default and
> this registration is not satisfied."

ABL-345 had landed. The flag was simply not passed, and nothing tied the flag to
the scope — which is why §6's guard is about the *binding*, not about the flag.

**The audit.** Every committed machine record in the repository, not just 2a:

| scope | `meta.training_source` | registered | |
|---|---|---|---|
| `abl316-t2a` (ABL-405, solar) | **`energy_renewable`** | `energy_generation` | **off-registration** |
| `abl316-t1b` (ABL-381, solar) | `energy_generation` | `energy_generation` | ✔ |
| `abl316-t2c` (ABL-419, solar) | `energy_generation` | `energy_generation` | ✔ |
| `abl316-t2d` (ABL-421, solar) | `energy_generation` | `energy_generation` | ✔ |
| `abl380-tranche1a` (wind) | `energy_generation` | `energy_generation` | ✔ |
| `abl406-tranche2b` (wind) | `energy_generation` | `energy_generation` | ✔ |
| `abl417-tranche2e` (wind) | `energy_generation` | `energy_generation` | ✔ |
| `abl435-tranche2f` (wind) | `energy_generation` | `energy_generation` | ✔ |
| `abl322-pilot` (ABL-436, wind offshore) | `energy_generation` | `energy_generation` | ✔ |
| `abl253` (solar) | `energy_renewable` | `energy_renewable` | ✔ |
| `abl376` (solar) | `energy_renewable` | `energy_renewable` | ✔ |

**One read in the programme is off its registration, and this is it.** Wind is
clean. That bounds the blast radius to eight country-pairs and is why this is an
evidence and registration defect rather than a programme-wide re-read.

---

## 4. A second consequence, which the filing did not have

§3 of `reports/abl_405_tranche2a_findings.md` — the night-floor screen that
motivates the tranche's **contamination hold** — genuinely read
`energy_generation`. Its own machine record confirms it:
`reports/abl_405_night_floor_probe.json` carries `"source": "energy_generation"`
for all eight countries, at `n_rows` 4,272, the full fit window. It carries **no
`energy_renewable` arm at all** — that arm is the one dropped by the probe crash
the same section documents.

So ABL-405's contamination evidence and ABL-405's fit are **on different tables**.
BG's 76.4% of night hours carrying 152 MW, and its 6.37% of energy booked in the
dark — the numbers the hold rests on — describe the registered table, not the
series the challenger was trained on. **Nothing in ABL-405 screens the series it
actually fitted.**

This read is the first of these eight countries where the night-floor screen and
the fitted series are the same series. Both corrections are recorded in place in
ABL-405's pack.

---

## 5. Sizing, re-measured at today's replica

The filing sized the defect against ABL-348's bars, measured 2026-08-12 on a
9,432,453,120-byte replica. Re-measured here on the live replica of 2026-08-22,
using ABL-348's own probe functions so the protocol cannot drift.

**The headline holds: the two tables' D-7 bar is identical to 2 dp on all eight**
(PL differs by 0.01pp). The bar is source-portable, as ABL-348 registered.

Three of the filing's numbers have moved, and one materially:

| | filed (2026-08-12) | measured (2026-08-22) |
|---|---|---|
| gate actuals bit-identical | 7 of 8 (HU only disagrees) | **5 of 8** — CZ 99.3% and PL 99.6% now disagree too (max 54.8 / 48.6 MW) |
| gate row counts | identical, 720/720 | **CZ 706 gen vs 720 ren; PL 704 vs 720** |
| fit-window difference | 27–33 h of 4,272 (0.63–0.77%) | holds for HU/PL/RO/SI/SK; **CZ is 121 h (2.83%)** |

CZ's fit-window gap is four times what the filing claimed. 93 of those 121 hours
are ABL-188 zero-fill runs that `energy_renewable` carries and `energy_generation`
does not — which ABL-348's `contamination` field already named ("CZ solar fit
−93 h … `energy_renewable` only") and which the filing's table-wide 0.63–0.77%
figure averaged away. **CZ is the pair the shipping decision turns on**, so the
one country where the filing understated the difference is the one where it
mattered most.

Every cell still clears the registered minimum n of 684 on both tables, so nothing
here is evaluability.

### The pre-measured bars no longer reproduce, and that is ABL-332 rather than revision

CZ 24.10 → 23.96, HU 18.35 → 18.18, PL 26.11 → 25.99, RO 24.41 → 24.29,
SI 22.28 → 21.65, SK 19.14 → 18.78. **BG and CH are unchanged to the digit.**

That split is the signature of ABL-332's hourly averaging, not of replica
revision: BG and CH are hourly-native, and every country that moved is one the
loader now aggregates from 15-minute rows instead of sub-sampling the `:00`
instant. ABL-405 ran **after** ABL-332 and its published D-7 reproduces today to
the printed digit, so ABL-405 and this read share loader semantics and remain
comparable. What is *not* comparable is either of them against ABL-348's
2026-08-12 bars, and that is a caveat on the bars, not on the reads.

---

<!-- RESULTS -->

---

## 8. Caveats, stated before being asked

- **Window and n.** Fit 2026-01-14 → 2026-07-11 (exclusive); gate 2026-07-11 →
  2026-08-10 (exclusive), out-of-sample by target timestamp. One 30-day summer
  holdout, not year-round evidence. Per-cell n is in the table above and in the
  machine record; registered minimum n is 684 / 684 / 456.
- **Baseline.** Literal seasonal-naive D-7, recomputed on the same table the
  challenger is scored against. No absolute WAPE threshold is registered.
- **Out-of-sample**, by target timestamp, on every number in §6 and §7. Nothing
  here is in-sample.
- **Contamination.** ABL-67 is net-position-only and ABL-109/ABL-111 are
  load-only; neither intersects these targets. ABL-71's known wrong-write modes
  are load and net position, so it is a provenance caveat rather than proof that
  solar ingest is pristine. **ABL-188 does intersect** and is the largest single
  driver of the fit-window difference on CZ (93 h), and the direction is
  favourable to this arm: the zero-fill runs are in `energy_renewable`, the table
  this read does *not* use.
- **Replica vintage.** ABL-405 read a 9,432,453,120-byte replica on 2026-08-13;
  this read is on a later snapshot. The confound is **measured, not assumed** —
  see §6's D-7 control column, which is model-free and which ABL-348 pre-measured
  as identical between the two tables, so any movement in it is vintage rather
  than table.
- **Incumbent.** None. All eight countries hold **zero** solar rows in
  `forecasts` at this replica (BE/FR/AT/DE only, verified not assumed), so the
  incumbent column reads "Not measured" by construction rather than by omission,
  and this scope refits no live pair.
- **TSO** is revision-contaminated context only and is never a gate criterion
  (ABL-470).

---

## 9. Interpretation limits

**A tranche's pass rate must not be averaged against another's.** 2a's bars run
12.67–26.11%; 2c's 7.11–16.43%; 2d's 23.92–47.85%. Same band, different tasks.

**A gate that passes is not a gate that certifies.** The D-7 bar is a floor. Read
the model-free references beside every cell: `constant_causal` says whether the
challenger predicts the *level*, `climatology_causal` whether it predicts the
level *and* the daily shape, and the oracle forms bound what hindsight alone could
have achieved. A cell that clears D-7 while losing to hour-of-day climatology has
not demonstrated skill beyond the average day. **Check each reference's own n
before comparing it to the challenger** — a climatology is 24 levels, and an hour
of day absent from its source window leaves those rows unscored for that column
alone. Nothing is interpolated to close that gap.
