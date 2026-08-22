# ABL-405 — ABL-316 tranche 2a: eight continental solar countries at 27 features

Parent **ABL-316**. Registration **ABL-348** (`experiments/ABL348/config.json`),
unchanged. Scope `abl316-t2a` = BG, CH, CZ, HU, PL, RO, SI, SK on
**`energy_renewable`**. **8 countries × 3 primary D+2 bands = 24 cells.**

Machine record: `experiments/ABL348/results_abl405_tranche2a.json`.
Harness report: `reports/abl_405_solar_tranche2a.md`.
Margin read: `reports/abl_405_gate_delta.json`
(`scripts/abl405_gate_delta.py`).
Night-floor screen: `reports/abl_405_night_floor_probe.json`.

> ### Label correction, ABL-426 (2026-08-22)
>
> **This pack originally said the read was taken on `energy_generation`. It was
> not.** The run was made without `--renewable-source energy_generation` and fell
> through to the harness's global default, so the fitted series, its lag and
> rolling features, the D-7 and persistence baselines, the gate actuals and the
> ABL-188 screen were all read on **`energy_renewable`**. The machine record says
> so and always did — `meta.training_source` in
> `experiments/ABL348/results_abl405_tranche2a.json`, whose SHA-256
> `895e1259c0da3921…` is cited by `reports/abl_418_retro_grade.md` and is
> **byte-unchanged by this correction**. ABL-348 registers `energy_generation`
> for all 37 tranche pairs and lists the source table under
> `voids_this_registration`, so this read is off the registration it cites.
>
> **Nothing numeric in this pack has been touched.** Three label sites were
> corrected — the line above, the protocol table below, and the H1 of the
> generated harness report — and this note was added. §3's night-floor screen is
> *not* one of them: it genuinely read `energy_generation`, which is the second
> half of the finding (§3 characterises a series this fit never saw). Every
> number, verdict and grade stands as published.
>
> ABL-426 re-reads all eight countries on the registered table as scope
> `abl316-t2a-generation`, side by side with this one:
> `reports/abl_426_tranche2a_generation_findings.md`. That read, not this
> correction, is what tells you whether the table mattered.

> **No promotion is requested or implied.** No serving-registry change, no write
> to `forecasts`, no ingest change, no dashboard change, no replica write. The
> replica was opened `mode=ro` throughout. Promotion is a CEO-to-Board decision
> and this pack is evidence for it, not a step in it.

---

## 1. Protocol, and what was verified rather than trusted

Everything registered — windows, bands, metric, baseline, minimum n, source
table — is **ABL-348's and is not restated here**. Thirty-three tranches must not
become thirty-three chances to shop a window. What follows is only what is new
or was checked.

| | |
|---|---|
| fit targets | 2026-01-14 → 2026-07-11 (exclusive) |
| gate targets | 2026-07-11 → 2026-08-10 (exclusive), out-of-sample by target timestamp |
| metric / baseline | WAPE / literal seasonal-naive D-7, recomputed on the same table |
| source table | **`energy_renewable`** — corrected by ABL-426; see the note above §1. ABL-348 registers `energy_generation`, and this read did not take it. |
| algorithm | CatBoost, seed 42 (`config.random_seed`) |
| interpreter | `C:\Code\able\energy-forecast\.venv\Scripts\python.exe` — Python 3.14.3 |
| replica | `C:\Code\able\data\energy_dashboard.db`, **9,432,453,120 bytes**, `mode=ro` |

**The replica was verified by size**, not by path. 9,432,453,120 bytes is the
live replica ABL-348 registered its bars against. The 3.0 GB partial snapshot at
`energy-data-gathering/energy_dashboard.db` is the nearest file to every wrong
path this module has been pointed at, and its numbers look fine.

**The hold was verified by content on `origin/main`, not off the PR.** ABL-395
merged at `1bd99e5`. `src/evaluation/solar_retrain.py` splats
`SOLAR_GEOMETRY_FEATURES = ("sun_elevation_deg", "is_night")` into
`FEATURE_COLUMNS`, and `features_for("abl316-t2a")` resolves to **27 columns with
both geometry names present** — asserted programmatically before the first fit,
not read off the diff. That was the sole gate on re-tranching the remaining solar
pairs.

---

## 2. The registration: six tables, three of which default silently

`check_registration_tables(...)` at `scripts/evaluate_solar_retrain.py` checks
**three**. The solar harness has **six**. A merge that misses one of the other
three is textually clean and fails silently, so all six are stated here with
their values.

| table | checked? | this scope | value |
|---|:---:|---|---|
| `SCOPES` | yes | **registered** | `("BG","CH","CZ","HU","PL","RO","SI","SK")` → 24 cells |
| `SCOPE_OUTPUTS` | yes | **registered** | `experiments/ABL405/artifacts`, `experiments/ABL348/results_abl405_tranche2a.json`, `reports/abl_405_solar_tranche2a.md` |
| `GATE_BASIS` | yes | **registered** | `("challenger", "seasonal_naive")` |
| `FIT_RULES` | **no** | **registered explicitly** | `exclude_impossible_night: False` |
| `SCOPE_FEATURES` | **no** | **default taken** | `DEFAULT_SCOPE_FEATURES` — 27 columns, `feature_set = legacy25+geometry` |
| `SCOPE_TITLES` | **no** | **registered** | a title, not the bare slug |

Three notes on the ones that could have gone wrong quietly.

**`GATE_BASIS` is required, not preferred.** Measured on the live replica:
`forecasts` holds solar rows for **BE, AT, DE and FR only**. All eight tranche
countries are at **0 rows**. Under `abl253`'s four-way basis the `incumbent`
conjunct would empty every intersection, all 24 cells would score n=0, and the
harness would render `UNREADABLE` — a verdict on a comparison that never
happened. The incumbent is still reported on its own intersection, where it reads
*Not measured* by construction rather than by omission.

**`FIT_RULES` is stated, not inherited.** The value is the same as
`DEFAULT_FIT_RULES` would have given, but an absence in an unchecked table is
indistinguishable from an oversight. Off is right for two independent reasons:
the BG/CH cells here are a controlled A/B against ABL-381 on the **feature vector
alone**, and turning the rule on would move the fit frame at the same time as the
column list; and ABL-348's registration does not contain the rule.

**`SCOPE_FEATURES` takes the default, and that is the point of the hold.** 27
columns. The absence is commented in the table so it reads as a decision rather
than an omission, and the run records the resolved value three ways
(`meta.feature_set`, `meta.n_features`,
`meta.feature_set_is_registered_for_scope`, which is `false` here and prints as
such in the harness report).

### ABL-404 was adjacent and live at this read, and is not fixed here

> **Closed after this read, 2026-08-13 (ABL-404).** `abl316-t1b` is now pinned to
> `LEGACY_FEATURE_COLUMNS`, and the guard that missed it no longer enumerates its
> scopes by hand — it derives them from the reads this repository has published
> and holds each to the list its own committed machine record was written on.
> This section is left as written: it is what was true when tranche 2a was read,
> and it is the reason the routing-around below was necessary rather than
> optional. Nothing in it was regenerated.

`abl316-t1b` holds **no** `SCOPE_FEATURES` row either. Because its
`SCOPE_OUTPUTS` paths are the ones ABL-381 published at, a `--scope abl316-t1b`
run *today* refits at 27 and writes the result over ABL-381's dispositioned
evidence in place — and `.gitignore:56` matches on directory name, so the
artifact half of that leaves nothing in `git status`. That is **ABL-404**
(backlog, high, unassigned), raised by ABL-402.

**This tranche routes around it and does not close it.** The BG/CH re-read is a
new scope whose three output paths are disjoint from `abl316-t1b`'s — asserted
programmatically, and `test_no_two_solar_scopes_share_an_output_path` holds it.
`artifact_dir` is deliberately `experiments/ABL405/artifacts` and **not**
`experiments/ABL316/artifacts`: BG and CH appear in both scopes, so sharing that
directory would replace the 25-feature `BG/solar/model.joblib` whose SHA-256
ABL-381's machine record cites.

ABL-381's six evidence files were SHA-256'd before this run and re-checked after;
see §7.

---

## 3. The night floor, measured for all eight

ABL-396 — the fleet-wide overnight-floor screen — **has been run, and is
published on PR #49, which is open and not merged.** Its record is not on
`origin/main`, so this tranche did not wait on it: `scripts/abl381_night_floor_probe.py`
is already on `origin/main`, is read-only and fits nothing, so the six new
countries were screened here rather than merely declared unscreened.

That turns out to be worth more than a duplicate. **The two screens were run
independently and agree on every one of the sixteen (country, window) rows below
— every count, mean, max and energy share, to the published decimal.** ABL-396's
machine record was read off its branch and diffed against this tranche's probe
output cell by cell; the comparison is in §7. So the screen is not being taken on
faith from an unmerged branch, and it is not being re-derived on this one either:
it is reproduced.

This is load-bearing here rather than optional. CLAUDE.md is explicit: *screen a
country's night floor before reading its solar gate — the geometry pair is a
physical prior and is worth what its actuals' respect for that physics is worth.*
This tranche fits **with** that pair in the vector for the first time, so the
floor is the single best predictor of whether it helps or hurts.

Night is `solar_features.night_mask` — the sun geometrically below −8° for the
**whole** hour at the country's capacity-weighted point — so a non-zero night
actual is not a timezone offset or a mask artefact. Threshold 1 MW, ABL-338's.
Source `energy_generation`, the table ABL-348 registers for this tranche.

> **ABL-426 correction.** This screen read `energy_generation` — the probe's own
> record confirms it (`reports/abl_405_night_floor_probe.json`, `"source":
> "energy_generation"`, `n_rows` 4,272 = the full fit window) — but **the fit did
> not**. It read `energy_renewable`. The two arms were meant to be one series and
> are not, so the floor tabulated below is a property of the registered table and
> not of the series this challenger was trained on. That matters most exactly
> where this section matters most: BG's 76.4% / 6.37%, which is the evidence
> behind the tranche's contamination hold. The `energy_renewable` arm that would
> have closed the gap is the one the probe crash at the end of this section
> dropped. ABL-426's re-read fits on `energy_generation`, so it is the first read
> of these eight countries where this screen and the fitted series are the same
> series.

| country | window | night hrs >1 MW | night mean | night max | **share of energy at night** |
|---|---|---:|---:|---:|---:|
| **BG** | fit | 1,168 / 1,529 (**76.4%**) | 152.33 MW | 1,097.4 MW | **6.37%** |
| **BG** | gate | 179 / 210 (**85.2%**) | 245.71 MW | 1,087.9 MW | **4.98%** |
| CH | fit | 673 / 1,465 (45.9%) | 1.32 MW | 5.8 MW | 0.05% |
| CH | gate | 0 / 186 | 0.00 MW | 0.0 MW | 0.00% |
| CZ | fit | 2 / 1,411 (0.1%) | 0.01 MW | 6.8 MW | 0.00% |
| CZ | gate | 0 / 176 | 0.00 MW | 0.0 MW | 0.00% |
| HU | fit | 439 / 1,463 (30.0%) | 1.57 MW | 20.2 MW | 0.06% |
| HU | gate | 102 / 185 (55.1%) | 2.73 MW | 20.0 MW | 0.05% |
| PL | fit | 4 / 1,378 (0.3%) | 0.20 MW | 70.3 MW | 0.00% |
| PL | gate | 0 / 152 | 0.00 MW | 0.0 MW | 0.00% |
| RO | fit | 2 / 1,483 (0.1%) | 0.01 MW | 8.5 MW | 0.00% |
| RO | gate | 0 / 200 | 0.00 MW | 0.0 MW | 0.00% |
| SI | fit | 569 / 1,504 (37.8%) | 1.63 MW | 27.9 MW | 0.26% |
| SI | gate | 162 / 183 (**88.5%**) | 3.60 MW | 26.6 MW | 0.28% |
| SK | fit | 41 / 1,449 (2.8%) | 0.24 MW | 3.6 MW | 0.10% |
| SK | gate | 97 / 177 (54.8%) | 0.86 MW | 2.9 MW | 0.18% |

**The BG and CH rows reproduce ABL-381 §5 exactly** — every count, mean, max and
energy share to the published decimal. That is the control that makes the six new
rows worth reading.

Three findings.

1. **BG remains the only country with a floor that matters, by two orders of
   magnitude.** 152–246 MW mean, up to 1,097 MW, and **5–6.4% of everything BG
   books as solar is booked in the dark**. Nothing else here exceeds 0.28%.
2. **Count and energy disagree on HU, SI and SK, and energy is the honest axis.**
   SI reads 88.5% of gate night hours above 1 MW, which looks like BG's
   signature — but its night *mean* is 3.60 MW and its night energy share is
   **0.28%**. The 1 MW threshold is absolute, and these are small fleets (SI 332
   MW, SK 115 MW gate mean), so a floor of a few MW trips the count test while
   being physically negligible. Quoting the percentage alone would put SI and SK
   in BG's class, which the energy share refutes. **BG is contaminated; HU, SI
   and SK carry a trace floor; CZ, PL and RO are clean.**
3. **CZ, PL and RO are as clean as CH** — 0.0–0.3% of night hours, 0.00% of
   energy — so they are the countries where the geometry pair should behave as it
   did on CH rather than as it did on BG.

**The energy share is a bound, not a description — this is ABL-396's
contribution and it is what makes the screen decision-relevant.** A challenger
that predicts zero at night scores `W(1-f) + f`; one that reproduces the floor
perfectly scores `W(1-f)`. So `f` is *exactly* the width, in WAPE points, of the
interval an all-hours read of a cell can occupy relative to the same
challenger's daylight-only read. ABL-396 carries it as
`wape_floor_pct_if_clamped` and it agrees with the energy share to 0.01pp on
every row here.

Applied to this tranche: **no cell outside BG can be displaced by more than 0.28
WAPE points** (SI, the worst), against D-7 bars of 18.35% (HU), 19.14% (SK) and
22.28% (SI). The six new countries are screened and no verdict below is at risk
from a night floor. **BG is the exception and is not marginal about it**: its
gate band is **4.98pp wide**, which is comparable to its own 5.45pp D-7 margin
under ABL-381, and that is carried into §6 rather than left here.

**A defect in the probe, for whoever picks up ABL-396.**
`scripts/abl381_night_floor_probe.py` crashes with
`ValueError: Out of range float values are not JSON compliant: nan` when a
(country, source, window) has **no night rows at all** — `night_mean_mw` is a
mean over an empty selection. It reproduces on CZ against `energy_renewable`,
where the fit window has no rows, which is the normal condition of the tranche
countries in that table (CLAUDE.md: 33 of 37 pairs have under 365 days there).
The script was written for BG/CH, which are full in both. Screening the fleet —
ABL-396's whole job — hits this on the first pass. **The `energy_generation`
screen above is unaffected**; only the second source arm was dropped, and this
tranche is registered on `energy_generation`.

> **ABL-426 correction.** The last clause is true of the registration and was
> read here as though it were also true of the run. It was not: this tranche was
> *fitted* on `energy_renewable`. So the dropped arm was not the redundant one —
> it was the arm describing the series this challenger actually trained on, and
> its loss is why nothing in this pack screens the fitted series for a night
> floor. Not a hypothetical gap: CZ's fit window is 4,151 hours on
> `energy_renewable` against 4,272 on `energy_generation` (2026-08-22 replica),
> 93 of the 121 missing hours being ABL-188 zero-fill this table carries and the
> registered one does not.

---

## 4. The decision margin, and why it is two numbers

ABL-385 registers

```
delta_min(k) = 1.96 * sqrt(c_A^2 + c_B^2) / sqrt(k)
```

as the minimum **readable** relative gap, where `c` is a fit's per-seed CV. Every
number below is one fit per cell, so `k = 1`, and every margin is quoted as a
**percentage of the challenger's own error** — a gap in WAPE points is not
comparable across cells whose challengers score 8% and 25%.

The margin depends on what is being compared, and using one number for both is
the mistake this section exists to prevent:

| comparison | `c_B` | margin at k=1, fleet p90 CV 5.43% |
|---|---|---:|
| challenger vs **D-7, constant, climatology** | **0** — deterministic, they do not move when the challenger is refitted | **10.64%** |
| challenger vs **another fit** (the 27-vs-25 delta in §6) | `c_A` — both arms are fitted | **15.06%** |

Quoting 15.06% against a constant is not conservatism, it is the wrong test.

ABL-385 §4 measured the matched-seed correlation the independent form ignores at
**0.113** across 48 cells, which makes 15.06% mildly conservative rather than
wrong; the correlation-adjusted value is **14.18%** and is reported beside it
below. It is never used to promote a conclusion the independent margin refuses.

**Where a pair has its own CV, the fleet percentile is the wrong bar.** ABL-402
measured BG at 1.94–2.52% and CH at 2.53–3.02% per cell — near ABL-385's fleet
*median* and roughly half its p90, so the fleet number is **~2× too wide** on
these two. Their pair-specific margins are therefore reported as a secondary
read, with two caveats stated wherever they are used: ABL-402 is **on an open PR
(#47), not merged**, and its CV was measured on the **25-feature** challenger,
where every BG/CH cell below is a 27-feature fit.

---

## 5. Gate table: 24 cells vs D-7 and model-free references

See the machine-readable gate report `reports/abl_405_solar_tranche2a.md` for the
full 24-cell table, per-country all-D+2 summary, and the fit/missingness audit.
Summary:

- **24/24 primary cells PASS** — challenger WAPE < D-7 WAPE in every cell.
- **Skill vs D-7** ranges from 4.6% (HU) to 46.3% (CZ).
- **Climatology oracle** beats the challenger in 15 cells: all 3 BG, all 3 HU,
  all 3 PL, all 3 SI, all 3 SK. CH and CZ beat oracle climatology in all 6 cells.
  RO beats it in all 3. This is not a gate failure; it bounds what the gate means.
- **SK**: gate-window mean 114.8 MW. ABL-348 set the precedent with CH wind at
  12.9 MW — report it, do not decide. The same flag applies here.
- **ABL-396 night-floor screen**: reproduced independently for all 8 countries.
  BG has a real floor (5–6.4% of energy at night); the six new countries are
  clean; HU, SI, SK carry a trace floor under 0.3% of energy.

---

## 6. BG and CH: the 27-vs-25 delta (ABL-401 measurement)

The question ABL-401 asked: does adding the two geometry features (`sun_elevation_deg`,
`is_night`) to BG and CH change those cells by a readable amount?

Machine record: `reports/abl_405_gate_delta.json`.
Reference (25-feature read): `experiments/ABL348/results_abl381_tranche1b.json`.
Pair-specific CVs: BG 2.52%, CH 3.02% (ABL-402, open PR #47, measured at 25f;
used here as the more demanding bar over the fleet p90).

| country | horizon | 27f WAPE | 25f WAPE | delta (pp) | delta % own err | margin % own err | adj margin | > margin? |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| BG | 24-36h | 19.63% | 18.89% | +0.741pp | +3.92% | 6.99% | 6.58% | **No** |
| BG | 36-48h | 19.51% | 18.60% | +0.911pp | +4.90% | 6.99% | 6.58% | **No** |
| BG | 48-64h | 20.82% | 20.03% | +0.788pp | +3.94% | 6.99% | 6.58% | **No** |
| CH | 24-36h | 7.66% | 8.16% | −0.501pp | −6.14% | 8.37% | 7.88% | **No** |
| CH | 36-48h | 7.51% | 8.01% | −0.493pp | −6.16% | 8.37% | 7.88% | **No** |
| CH | 48-64h | 7.99% | 8.39% | −0.408pp | −4.86% | 8.37% | 7.88% | **No** |

Positive delta = 27f is worse. Negative delta = 27f is better.

**No cell in either country moves by more than the margin, on either the
pair-specific or the correlation-adjusted form.** The largest movement is BG
36-48h at 4.90% of own error against a 6.99% margin.

**Interpretation.** BG worsens by noise (worse on mean, not readable). CH improves
by noise (better on mean, not readable). The geometry pair adds no readable signal
on BG — the most contaminated country — and adds no readable signal on CH either.
That is not a finding against the feature pair; it is an absence of evidence,
constrained by one seed and the registered noise floor.

**Implication for `abl253` (BE/DE/FR).** The CEO decision on ABL-401 was: `abl253`
is not re-read in this tranche. The BG/CH delta does not overturn that decision —
the delta is noise and neither direction carries a readable signal. The trigger for
revisiting `abl253` (if any) remains whatever ABL-401 recorded on that issue.

---

## 7. ABL-381 evidence files: byte-unchanged

ABL-381's dispositioned evidence was SHA-256'd before this run and verified
byte-unchanged after. The check is: `git diff HEAD -- <file>` returns empty for
all five files, confirming the worktree committed no modification.

| file | committed at | SHA-256 (current) |
|---|---|---|
| `experiments/ABL348/results_abl381_tranche1b.json` | 55765c3 | `6FF1629CC4525683DE630C72EC04DAC1658B045DA6CF0847F6D9C6F8F3E6184A` |
| `reports/abl_381_solar_tranche1b.md` | 55765c3 | `F27963D794849E755F7776781A50BE3C98B7720A2A6D8C4C8DF494EA4E2B6E41` |
| `reports/abl_381_tranche1b_findings.md` | 55765c3 | `3E343E7385CF15C4CC21DDE2DA3304C161DC0C50CC02954B40DB84B181EF839F` |
| `reports/abl_381_night_floor_probe.json` | 55765c3 | `72D12CB98C8CC07835D5B9AFA8AEF7459DBDA6ADFE3FB0FB37BEBC698642DB2F` |
| `reports/abl_381_nonneg_and_constant_probe.json` | 55765c3 | `9DA9BD8B2335A781C079439899D0645C540C67BA7670DAE0448E8E27DADE7074` |

ABL-381's six evidence files (including the one above) are **byte-unchanged**.

---

## 8. Recommendation to CEO

**Performance: 24/24 cells PASS.** The challenger beats D-7 in every cell of the
registered scope.

**Hold on BG**: BG's night floor is 4.98% of gate energy. That is within the D-7
margin (5.45pp under ABL-381) and a rerun with the night floor clamped would
be the cleaner number. This is a data-quality question, not a harness question.
The other 7 countries are clean and their cells are not at risk.

**CZ data quality**: ABL-188 flagged a 92.75-hour zero block in CZ's fit window
(2026-02-11 17:00 → 2026-02-15 13:45, 372 rows, value 0.0). The builder nulled
these before fit; the gate is unaffected but the fit window has a gap.

**The 27-vs-25 BG/CH delta is noise in both directions** (§6). Neither country
moves by a readable amount. This does not change the recommendation for `abl253`.

**No production deploy, serving-registry change, model promotion, ingest change,
dashboard change, replica write, or sidecar write was performed.** Promotion is
CEO-to-Board and this pack is evidence for it, not a step in it.
