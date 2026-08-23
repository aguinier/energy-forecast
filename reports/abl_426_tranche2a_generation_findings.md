# ABL-426 — tranche 2a re-read on the registered `energy_generation`

**Result: 22 of 24 cells pass — the tranche reads FAIL on the registered table,
against ABL-405's 24/24 PASS on `energy_renewable`. Both failures are HU. CZ, the
one solar pair that clears both oracle references readably, survives the
correction with a wider margin (+13.00% → +15.17% against oracle climatology).**

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
revision. ABL-332 landed on `src/db.py` at **2026-08-12 19:43 UTC** (`70f835e`);
ABL-348's bars were taken earlier that day and ABL-405 ran 2026-08-13 20:18 UTC,
so the bars are pre-ABL-332 and ABL-405 is post. BG and CH are hourly-native, and
every country that moved is one the loader now aggregates from 15-minute rows
instead of sub-sampling the `:00` instant. ABL-322's twin registration was
restated for exactly this reason on the same night (`2ddec62`); ABL-348's was not,
and `experiments/ABL348/config.json` does not mention ABL-332 anywhere. **Filed
separately** — it affects every 15-minute pair in the programme, and no verdict
turns on it, because the harness recomputes D-7 inside the gate window.

### The replica-vintage confound is measured, and it is ~0.04pp

This is the control the whole A/B rests on, so it is taken before the challenger
exists rather than argued afterwards. D-7 is **model-free**: if ABL-405's
published D-7 still reproduces on today's replica, then nothing the two reads are
scored against has moved between them, and a challenger difference is the table.

| | ABL-405's harness D-7, 24–36h (2026-08-13) | this screen on `energy_renewable` (2026-08-22) | ABL-348's registered bar (2026-08-12) |
|---|---:|---:|---:|
| BG | 24.40 | 24.40 | 24.40 |
| CH | 12.67 | 12.67 | 12.67 |
| CZ | 23.96 | 23.96 | 24.10 |
| HU | 18.18 | 18.18 | 18.35 |
| PL | 26.00 | 26.00 | 26.11 |
| RO | 24.29 | 24.29 | 24.41 |
| SI | 21.65 | 21.65 | 22.28 |
| SK | 18.82 | 18.78 | 19.14 |

**Seven of eight reproduce exactly.** SK moves by 0.04pp, which is a real
revision inside a frozen window and is the measured upper bound on the vintage
confound for this comparison. Everything else in the middle column is unmoved
nine days later.

Note that the two columns are not the same estimator — ABL-405's is the harness's
per-band D-7 on the finite intersection, this screen's is a whole-gate-window D-7
on the plain hourly series — so their agreeing to 2 dp on seven pairs is a
stronger statement than a like-for-like match would be.

§7's per-cell D-7 delta between the two arms is the same control taken again,
inside the harness, at full precision. It comes back at **≤0.0074pp on all 24
cells**, which is tighter than this screen's 0.04pp upper bound and settles the
confound rather than bounding it.

### The replica was rebuilt twice during this issue, and both rebuilds were additive

A full replica refresh committed at **2026-08-22 17:06 UTC**, mid-issue: a 10.2 GB
rollback journal against the live file, held for over 40 minutes, during which
*every* read failed and two gate runs died — the first on CZ, 15 minutes in, with
no partial output. The file went from 10,175,365,120 to **10,220,126,208 bytes**.
A second refresh committed overnight, before the gate read of 2026-08-23:
**10,266,849,280 bytes**.

Each is a replica change between the sizing measurement and the gate read, so the
screen was **taken again after each** rather than assumed to still hold. Both
re-takes reproduce the pre-refresh screen on every quantity: D-7 identical to 2 dp
on both tables for all eight, `n_d7_scorable` identical, fit-window hours
identical, bit-identity percentages identical. The second re-take was differenced
leaf by leaf against the first: **243 of 244 leaves byte-identical, the 244th
being `replica_bytes` itself**. **Both refreshes added data outside the frozen
windows and revised nothing inside them.**
`reports/abl_426_vintage_screen.json` is the measurement taken on the snapshot the
gate read below actually ran against, and records `replica_bytes` so which
snapshot it was taken on is not a matter of recollection.

---

## 6. The read: **22 of 24 cells clear the registered bar. The tranche verdict is FAIL.**

Run 2026-08-23 07:36 UTC, replica 10,266,849,280 bytes, `mode=ro`.
`meta.source_is_scope_registered: true` — the guard confirms this read took the
table its scope registers, which is the fact ABL-405's record could not carry.

**ABL-405 read 24/24 PASS on `energy_renewable`. The same eight countries, the
same windows, the same 27 features and the same registration, read on the
`energy_generation` that ABL-348 registers, read 22/24.** ABL-348's strict rule is
all 24 primary cells, so the tranche disposition on the registered table is
**FAIL**, not PASS.

Both failures are HU, at 24-36h and 36-48h. HU was the marginal pair on
`energy_renewable` — +4.6% skill against its D-7 bar, the thinnest margin in the
tranche — and on the registered table the challenger crosses it.

| cell | n | challenger | D-7 | skill vs D-7 | vs oracle flat | vs oracle climatology | grade | gate |
|---|---:|---:|---:|---:|---:|---:|:---:|:---:|
| BG 24-36h | 720 | 19.95% | 24.40% | +18.22% | +72.85% | -4.17% | A | PASS |
| BG 36-48h | 720 | 19.75% | 24.40% | +19.07% | +73.13% | -3.09% | A | PASS |
| BG 48-64h | 510 | 21.40% | 24.99% | +14.37% | +66.47% | -4.98% | A | PASS |
| CH 24-36h | 720 | 7.78% | 12.67% | +38.59% | +91.78% | +13.70% | A | PASS |
| CH 36-48h | 720 | 7.75% | 12.67% | +38.83% | +91.81% | +14.04% | A | PASS |
| CH 48-64h | 510 | 8.32% | 12.53% | +33.59% | +90.54% | +4.34% | A | PASS |
| CZ 24-36h | 706 | 13.02% | 23.96% | +45.67% | +85.82% | +18.31% | A | PASS |
| CZ 36-48h | 706 | 13.07% | 23.96% | +45.45% | +85.77% | +18.00% | A | PASS |
| CZ 48-64h | 498 | 13.62% | 24.02% | +43.31% | +84.33% | +15.17% | A | PASS |
| **HU 24-36h** | 720 | **18.42%** | **18.18%** | **-1.32%** | +80.61% | -29.86% | U(+) | **FAIL** |
| **HU 36-48h** | 720 | **18.43%** | **18.18%** | **-1.37%** | +80.60% | -29.93% | U(+) | **FAIL** |
| HU 48-64h | 510 | 17.18% | 17.89% | +3.97% | +81.20% | -20.31% | U(+) | PASS |
| PL 24-36h | 704 | 17.14% | 25.99% | +34.07% | +81.12% | -11.30% | A | PASS |
| PL 36-48h | 704 | 17.21% | 25.99% | +33.80% | +81.05% | -11.76% | A | PASS |
| PL 48-64h | 498 | 16.09% | 24.51% | +34.36% | +81.19% | -10.09% | A | PASS |
| RO 24-36h | 720 | 18.08% | 24.29% | +25.56% | +81.12% | +9.34% | A | PASS |
| RO 36-48h | 720 | 18.09% | 24.29% | +25.53% | +81.12% | +9.30% | A | PASS |
| RO 48-64h | 510 | 18.74% | 24.99% | +25.03% | +79.86% | +8.36% | A | PASS |
| SI 24-36h | 720 | 18.35% | 21.65% | +15.23% | +80.44% | -41.09% | A | PASS |
| SI 36-48h | 720 | 18.27% | 21.65% | +15.60% | +80.53% | -40.47% | A | PASS |
| SI 48-64h | 510 | 18.99% | 21.22% | +10.50% | +78.92% | -48.83% | U(+) | PASS |
| SK 24-36h | 715 | 17.53% | 18.82% | +6.87% | +81.60% | -33.68% | U(+) | PASS |
| SK 36-48h | 715 | 17.59% | 18.82% | +6.52% | +81.53% | -34.17% | U(+) | PASS |
| SK 48-64h | 507 | 15.90% | 18.34% | +13.33% | +82.92% | -27.03% | A | PASS |

Every cell clears the registered minimum n (684 / 684 / 456). `not_evaluable_cells`
is empty. Grades are computed under this scope's registered `fit_window`
levelling and `sign_test` G2/G3 readability, against the 10.6482% solar k=1 floor.

**The two failing cells are not the interesting number.** The challenger still
beats a hindsight flat line by 66-92% everywhere, and **loses to a hindsight
hour-of-day climatology in 15 of 24 cells** — the same 15 as on
`energy_renewable`, on the same five countries (BG, HU, PL, SI, SK). A gate that
passes is not a gate that certifies, and on this tranche the D-7 bar was never the
demanding reference.

### The contamination screen changes side, and it corroborates §5

ABL-405's screen found a suspect constant run on CZ **in `energy_renewable`**:
92.75 hours of bit-identical `0.0` across 372 rows, 2026-02-11 17:00 →
2026-02-15 13:45, nulled by the builder before the fit. This read's screen finds
**no ≥24-hour bit-identical solar run in `energy_generation`** anywhere in the
registered interval plus its 14-day lookback, for any of the eight.

That is the same ABL-188 zero-fill §5 attributed from the standalone screen, now
shown from the two harness screens themselves rather than inferred. The registered
table does not carry it.

### The fit-coverage gap is larger than the standalone screen measured

The harness records what actually reached the fit, and it is not the same quantity
as raw row availability:

| | fit targets, `energy_renewable` | fit targets, `energy_generation` | delta |
|---|---:|---:|---:|
| BG | 4,173 | 4,272 | +99 |
| CH | 4,188 | 4,272 | +84 |
| **CZ** | **3,881** | **4,272** | **+391 (+10.1%)** |
| HU | 4,191 | 4,272 | +81 |
| PL | 4,191 | 4,272 | +81 |
| RO | 4,191 | 4,272 | +81 |
| SI | 4,188 | 4,272 | +84 |
| SK | 4,173 | 4,262 | +89 |

The filing sized this at 27-33 hours (0.63-0.77%) and §5 re-measured CZ at 121 h
(2.83%). Both are counts of *hours present in each table*. The harness's
`audit.unique_targets` is the count of targets that survived ABL-188 nulling **and
had a complete feature row**, and a 92.75-hour null block also destroys the lag
and rolling features of the days downstream of it. On CZ that compounds 121 raw
hours into **391 fit targets, +10.1% of what ABL-405 actually fitted on**.

**This is the largest single correction in the issue, and it lands on the pair the
shipping decision turns on.** What makes it a finding rather than an alarm is §7:
CZ's challenger moved +0.15 / +0.21 / -0.36pp on a 10% larger fit series.

---

## 7. What reading the wrong table cost, cell by cell

`reports/abl_426_source_arm_delta.json`, from the two committed machine records.
`controlled.all_controls_hold: true` — countries, cells, gate basis, fit rules, the
27 feature names, feature set, both windows and the intended-n table are equal
between the arms. Two record-schema differences are reconciled rather than
counted as control failures, and both checks are in the tool, not in this
paragraph: the three grading registrations ABL-405 predates
(`grading_registration_reconciled: true` — arm B records the pin **both** arms are
graded under), and the two reference columns ABL-437 added after it
(`reported_comparators.reconciled: true` — arm B is a superset, and no added
column is in the gate basis or read by the `fit_window` levelling the letters are
taken under).

**The control that makes this an A/B at all: the D-7 column moves by at most
0.0074pp across all 24 cells.** D-7 is model-free, so the two arms are scored
against the same bar to four decimal places, and the replica-vintage difference
between a 9.43 GB snapshot and a 10.27 GB one — ten days and two full rebuilds
apart — is measured at essentially zero rather than argued away.

| cell | n ren → gen | challenger ren | challenger gen | delta | **D-7 delta (control)** | gate | grade |
|---|---:|---:|---:|---:|---:|:---:|:---:|
| BG 24-36h | 720 | 19.63% | 19.95% | +0.33pp | +0.0000pp | PASS | A |
| BG 36-48h | 720 | 19.51% | 19.75% | +0.24pp | +0.0000pp | PASS | A |
| BG 48-64h | 510 | 20.82% | 21.40% | +0.58pp | +0.0000pp | PASS | A |
| CH 24-36h | 720 | 7.66% | 7.78% | +0.12pp | +0.0000pp | PASS | A |
| CH 36-48h | 720 | 7.51% | 7.75% | +0.24pp | +0.0000pp | PASS | A |
| CH 48-64h | 510 | 7.99% | 8.32% | +0.33pp | +0.0000pp | PASS | A |
| CZ 24-36h | 720 → 706 | 12.87% | 13.02% | +0.15pp | -0.0007pp | PASS | A |
| CZ 36-48h | 720 → 706 | 12.86% | 13.07% | +0.21pp | -0.0007pp | PASS | A |
| CZ 48-64h | 510 → 498 | 13.97% | 13.62% | -0.36pp | -0.0074pp | PASS | A |
| HU 24-36h | 720 | 17.35% | 18.42% | +1.07pp | -0.0028pp | **PASS → FAIL** | U(+) |
| HU 36-48h | 720 | 17.35% | 18.43% | +1.08pp | -0.0028pp | **PASS → FAIL** | U(+) |
| HU 48-64h | 510 | 16.53% | 17.18% | +0.65pp | -0.0043pp | PASS | U(+) |
| PL 24-36h | 720 → 704 | 17.34% | 17.14% | -0.20pp | -0.0065pp | PASS | A |
| PL 36-48h | 720 → 704 | 17.37% | 17.21% | -0.16pp | -0.0065pp | PASS | A |
| PL 48-64h | 510 → 498 | 16.30% | 16.09% | -0.22pp | -0.0062pp | PASS | A |
| RO 24-36h | 720 | 18.76% | 18.08% | -0.68pp | +0.0000pp | PASS | A |
| RO 36-48h | 720 | 18.70% | 18.09% | -0.60pp | +0.0000pp | PASS | A |
| RO 48-64h | 510 | 19.16% | 18.74% | -0.43pp | +0.0000pp | PASS | A |
| SI 24-36h | 720 | 17.91% | 18.35% | +0.44pp | -0.0007pp | PASS | A |
| SI 36-48h | 720 | 18.11% | 18.27% | +0.16pp | -0.0007pp | PASS | A |
| SI 48-64h | 510 | 18.65% | 18.99% | +0.34pp | -0.0008pp | PASS | **A → U(+)** |
| SK 24-36h | 715 | 16.32% | 17.53% | +1.21pp | +0.0000pp | PASS | **A → U(+)** |
| SK 36-48h | 715 | 16.37% | 17.59% | +1.22pp | +0.0000pp | PASS | **A → U(+)** |
| SK 48-64h | 507 | 15.07% | 15.90% | +0.83pp | +0.0000pp | PASS | A |

Challenger WAPE delta across 24 cells: **-0.68pp to +1.22pp, mean +0.27pp**. The
sign is not uniform — RO and PL are *better* on the registered table, CZ 48-64h
too. **Two gate verdicts change and three grades change.** Both verdict changes
are HU; the three grade changes (SI 48-64h, SK 24-36h, SK 36-48h) are A → U(+),
readability losses against the 10.6482% floor with no gate consequence.

**The filing's headline was wrong, and it was wrong in the safe direction.** It
sized the defect at "whatever a 0.7%-shorter fit series moves", from a bar that
did not move and gate actuals that were bit-identical on seven of eight. Those
facts hold — the bar really does not move, to 0.0074pp. What the sizing missed is
that a 0.7% difference in the *input series* is not a 0.7% bound on a fitted
model's *output*: HU's two failing cells move 1.07-1.08pp on a fit-target gap of
81 hours in 4,272 (1.9%), which is enough to cross an 18.18% bar the challenger
cleared by 0.83pp. **A gate margin that thin is not robust to the source table,
and there was no way to know that without running it.**

### The CEO's premise, re-measured on the registered table

The 2026-08-14 comment raising this issue turned on two pairs. Worst band per
country, against the 10.6482% solar k=1 readability floor:

| | vs oracle flat, `energy_renewable` | on `energy_generation` | vs oracle climatology, `energy_renewable` | on `energy_generation` | clears both readably? |
|---|---:|---:|---:|---:|:---:|
| **CZ** | +84.45% | **+84.33%** | +13.00% | **+15.17%** | **yes on both** |
| **RO** | +79.40% | **+79.86%** | +5.93% | **+8.36%** | no on either — inside the floor |
| BG | +67.38% | +66.47% | -2.46% | -4.98% | no |
| CH | +90.91% | +90.54% | +8.15% | +4.34% | no |
| HU | +81.74% | +80.60% | -22.21% | -29.93% | no |
| PL | +81.15% | +81.05% | -12.77% | -11.76% | no |
| SI | +79.30% | +78.92% | -46.16% | -48.83% | no |
| SK | +82.81% | +81.53% | -24.84% | -34.17% | no |

**CZ survives the correction, and its binding margin widens.** It is still the
only solar pair in the tranche that clears both oracle references readably, and
the reference that binds it — oracle climatology, worst band — goes from +13.00%
to **+15.17%**, from 2.35pp above the floor to 4.52pp above it. That holds while
CZ fits on 391 more targets (+10.1%) and while its D-7 bar is unmoved to
0.0007pp. **The one pair the CEO's comment said this read could add is confirmed,
not overturned.**

**RO does not become readable.** It improves on the registered table — +5.93% →
+8.36% against oracle climatology — but the floor is 10.6482% and it stays inside
it. Its position is unchanged in kind: clears both references, neither readably.
Whether that is enough remains the wider-rule question the Board's card already
framed; this read moves the number, not the rule.

**The four provisional exclusions are confirmed, and more firmly than before.**
HU, PL, SI and SK lose to a hindsight hour-of-day climatology by 10-49pp on
*both* tables, and on the registered one HU also loses its gate PASS. Not one of
the four is a source-table artefact. `summary.oracle_reference_moves` is empty:
**no cell changes readability against either oracle reference in either
direction.**

---

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
  this read is on a 10,266,849,280-byte snapshot ten days and two full rebuilds
  later. The confound is **measured, not assumed** — see §7's D-7 control column,
  which is model-free and which ABL-348 pre-measured as identical between the two
  tables, so any movement in it is vintage rather than table. It comes back at
  **≤0.0074pp on all 24 cells.**
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

**A verdict change is not a verdict correction.** ABL-405's 24/24 PASS is a true
statement about `energy_renewable`. This read's 22/24 FAIL is a true statement
about `energy_generation`. The second is the one ABL-348 registered, so it is the
one the registration covers — but the first is not withdrawn, is not wrong about
what it measured, and its cells keep the letters ABL-418 gave them.

---

## 10. Recommendation to the CEO

**No promotion is recommended or implied, and none is requested.** Nothing here
changes what serves. The Board's `ship8` answer of 2026-08-22 covers eight
wind pairs; **this read adds no solar pair to it and removes none.**

1. **Record ABL-405's tranche-2a disposition as source-qualified.** Its
   PERFORMANCE PASS holds on `energy_renewable` and does not hold on the
   registered `energy_generation`, where the same read is 22/24 — a FAIL under
   ABL-348's strict rule. That is a disposition change at tranche level and the
   ledger should carry both readings with their tables named, not replace one
   with the other. The two failing cells are HU 24-36h and HU 36-48h.

2. **CZ solar is confirmed, on the registered table.** Worst-band +84.33% against
   an oracle flat line and **+15.17% against oracle climatology**, both readable
   against the 10.6482% floor, all three bands over minimum n, all three grade A,
   all three PASS. It was held out of the shipping set for the source, and the
   source no longer holds it out. Whether it ships is the Board's call on the
   ABL-316 card, not this pack's.

3. **RO solar is not added by this read.** It improves (+5.93% → +8.36% against
   oracle climatology) and remains inside the readability floor. Under the
   readable rule it does not qualify; under the wider rule its standing is
   unchanged and its margin is better.

4. **HU, PL, SI and SK solar stay excluded, and the provisional label comes off.**
   All four lose to a hindsight hour-of-day climatology on both tables, by 10 to
   49pp. Their exclusion was never a source-table artefact.

5. **The guard is the durable part.** `SCOPE_SOURCES` elects each scope's table in
   the file, before the first fit, and an explicit `--renewable-source` that
   disagrees is recorded as off-registration in the machine record and printed as
   **OFF-REGISTRATION** in the report. The run that produced ABL-405 would now
   fail at import. That is the reason this issue is worth its compute
   independently of which way the numbers went.

**What this cost, and what it bought.** One gate read, eight countries, ~28
minutes of CPU. It bought a corrected tranche disposition, a confirmed CZ, and a
measured answer to the question the filing had to leave open — *does the source
table move a verdict?* It does: two cells, one country, on a margin of 0.83pp.
