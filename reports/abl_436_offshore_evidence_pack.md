# ABL-436 — DE/NL `wind_offshore`: reference suite, grade, and a committed record

**Disposition: the ABL-322 pilot reproduces exactly, and both pairs grade A — but the
two pairs land on opposite sides of the screen that decides what an A is worth.**

- **NL wind_offshore** — grade A, and it beats **both** correctly-levelled oracle
  references by roughly twice the readability floor in every band. It is the fourth pair
  in the whole ABL-316 programme to clear that screen, and the first `wind_offshore` pair
  to do so.
- **DE wind_offshore** — grade A, but it **loses to both oracle references in all three
  bands**, at a near-zero slope, while its own TSO forecast scores 21.1% against the
  model's 66.0%. Not servable on this evidence.
- **Both bars are weak once correctly levelled.** NL's registered
  `bar_weaker_than_a_flat_line = false` is the ABL-435 mis-levelling artifact, not a
  strong bar — a hindsight flat line clears the bar in 6/6 cells across the two pairs
  (§4.1). NL's A survives anyway, because it is earned against the *oracle* references
  the artifact does not touch.

Evidence only. No promotion, no serving-registry change, no write to `forecasts`, no
replica write, no ingest or dashboard change.

Generated 2026-08-14. Machine record: `experiments/ABL322/results_abl436_offshore_reread.json`.
Harness report: `reports/abl_436_offshore_reference_grade.md`.

## 0. The three gaps, and what closed them

| # | gap | closed by |
|---|---|---|
| 1 | no model-free reference | four ABL-389 references beside all 6 cells — §4 |
| 2 | no grade | ABL-418 ladder applied at `attach_grades` — §5 |
| 3 | no machine-readable record | `experiments/ABL322/results_abl436_offshore_reread.json`, committed — §8 |

## 1. Scope: this was a read, not a harness change

The issue asked to settle this first, and to split the work if it turned out to be a
harness change. It did not.

`scripts/evaluate_wind_retrain.py` already carries the pilot as a registered scope.
`abl322-pilot` is present in **all three** registration tables — `SCOPES:77`,
`GATE_BASIS:293`, `SCOPE_OUTPUTS:208` — and `check_registration_tables` /
`check_scope_outputs` pass at import. `ALGORITHMS:53` and `COLUMNS:266` both carry
`wind_offshore`. Nothing needed folding in, and **no file under `src/` or `scripts/`
was modified by this issue**. The diff is three evidence files.

The read was taken as:

```
.venv/Scripts/python.exe scripts/evaluate_wind_retrain.py \
  --scope abl322-pilot --renewable-source energy_generation \
  --replica-db C:\Code\able\data\energy_dashboard.db \
  --artifact-dir experiments/ABL436/artifacts \
  --json-out  experiments/ABL322/results_abl436_offshore_reread.json \
  --report-out reports/abl_436_offshore_reference_grade.md
```

`--renewable-source energy_generation` is **not optional**. The global
`db.RENEWABLE_TYPE_SOURCE_TABLE` is `'energy_renewable'`, and ABL-322's registration
requires `energy_generation` because NL offshore carries 447 provably zero-filled rows
and 668 disagreeing duplicate instants in `energy_renewable` (ABL-318). An unflagged run
fits the wrong table without erroring. The committed record carries
`meta.training_source = "energy_generation"`, so this is checkable rather than asserted.

The three output paths are overridden for the same reason ABL-387 introduced
`SCOPE_OUTPUTS` at all: the scope's registered `report_out` is
`reports/abl_322_pilot_gate.md`, the pilot's **dispositioned** evidence. A default run
overwrites it in place. Both predecessor files are byte-identical to `HEAD` after this
run — `reports/abl_322_pilot_gate.md` at blob `3da919cb`, `experiments/ABL322/config.json`
at `41870ebc`, both matching `git rev-parse HEAD:<path>`.

### 1.1 One harness defect found, and deliberately not fixed here

`SCOPE_OUTPUTS["abl322-pilot"]["json_out"]` is `experiments/ABL322/results.json` — the
exact filename `.gitignore:53` (`experiments/*/results.json`) swallows. **That is the
mechanism behind gap 3**, and it is still live: it is not that the pilot forgot to commit
its record, it is that the scope is registered to write it where Git cannot see it.
`git check-ignore -v` confirms the path resolves to `.gitignore:53`.

Every later tranche routes around this by naming its `json_out` something else
(`results_abl380_tranche1a.json`, `results_abl406_tranche2b.json`,
`results_abl417_tranche2e.json`); `abl322-pilot` is the only scope still pointed at the
swallowed name. This issue routes around it with a flag rather than editing the table,
because changing where a registered scope writes is a registration change and the issue
asked for the split. **Filed separately.** Until it lands, an unflagged
`--scope abl322-pilot` run still (a) writes an uncommittable record and (b) overwrites
the pilot's dispositioned report.

## 2. Reproduction of the pilot

Same registered scope, same frozen windows, same source table, refit from scratch on the
rail interpreter (`.venv`, Python 3.14.3, xgboost 3.3.0). All six cells reproduce the
pilot **to the published precision**, as do MAE, bias, slope, correlation, n, fit rows
(34,176), unique fit targets (4,272), degraded lag-1d rows (23,674) and both per-country
all-D+2 summaries including the TSO columns.

| pair | band | pilot challenger WAPE | this read | pilot D-7 | this read |
|---|---|---:|---:|---:|---:|
| DE | 24-36h | 66.1% | 66.1% | 88.9% | 88.9% |
| DE | 36-48h | 65.7% | 65.7% | 88.9% | 88.9% |
| DE | 48-64h | 66.1% | 66.1% | 87.1% | 87.1% |
| NL | 24-36h | 60.5% | 60.5% | 81.8% | 81.8% |
| NL | 36-48h | 61.3% | 61.3% | 81.8% | 81.8% |
| NL | 48-64h | 63.8% | 63.8% | 88.5% | 88.5% |

Two things this comparison cannot do, stated rather than glossed:

- **It is a 1-decimal-place check, because 1 dp is all the pilot published.** The pilot's
  `results.json` was gitignored, so there is no stored record to diff against. That is
  gap 3 restated as a measurement limit, and it is the last time it applies: from this
  read on, the comparison is `json.load` at full precision.
- **The artifact SHA-256 does not match and cannot.** DE `5900d967…` here against the
  pilot's `c7151e3c…`, NL `d7d65793…` against `5292e38b…`. `Forecaster.save` stamps
  `saved_at`, so a byte hash re-hashes the clock, not the model. The reproduction claim
  above rests on the scores, not on the hashes.

### 2.1 The pilot ran under ABL-332, despite its own registration text

Worth recording, because ABL-322's `config.json` says the opposite in two places. Its
`blocker_state_2026_08_12` lists ABL-332 as "in_review — STILL BLOCKING", and its
`aggregation_convention` registers "score the hourly `:00` instant". ABL-332 in fact
merged at `70f835e`, **2026-08-12 21:43**, roughly eight hours before the pilot's
06:12 UTC run. The loader has emitted hourly means for these pairs ever since, and this
run logs it explicitly:

```
ABL-332: aggregated 21408 sub-hourly rows to 5352 hourly means (0 days 00:15:00 cadence,
0 partial hour(s)) [DE/wind_offshore from energy_generation]
```

So the pilot scored hourly means, not the `:00` instant its own text names. This changes
nothing and is why: ABL-322 pre-measured the sensitivity *before* either model existed,
at 0.04pp (DE) and 0.01pp (NL) on the D-7 bar — §3 re-derives both — and the exact
reproduction in the table above confirms the two reads share one convention. It is
recorded so a later reader does not mistake the registration text for the protocol.

Both DE and NL offshore are 15-minute cadence across the **entire** loaded span
(21,408 sub-hourly rows → 5,352 hourly means, 0 partial hours, for each), so no
mid-window resolution flip touches this window.

## 3. The pilot's integrity check, carried forward

ABL-322's strongest protocol property is that its bar was measured a day before either
model existed, by a separate read-only script. It is carried forward here by
**re-execution**, not by quoting the prose: `scripts/abl322_preregistration_probe.py`
re-run against the live replica on 2026-08-14 reproduces every published figure.

| pair | aggregation | n | mean actual | D-7 WAPE re-derived | ABL-322 registered |
|---|---|---:|---:|---:|---:|
| DE | `:00` instant | 720 | 2,513.24 MW | **88.8173%** | 88.82 |
| DE | hourly mean | 720 | 2,511.41 MW | **88.8562%** | 88.86 |
| NL | `:00` instant | 720 | 1,131.01 MW | **81.7832%** | 81.78 |
| NL | hourly mean | 720 | 1,134.34 MW | **81.7901%** | 81.79 |

Exact to every published digit, including the mean actuals (2513.2 / 1131.0 MW). The
convention sensitivity the registration claimed — 0.04pp on DE, 0.01pp on NL — is
confirmed at 0.0389pp and 0.0069pp. Committed machine-readable at
`experiments/ABL322/abl436_preregistration_recheck.json`.

Replica: `C:\Code\able\data\energy_dashboard.db`, 9,432,453,120 bytes, opened `mode=ro`,
`uri=True` — the same file and the same byte count the pilot recorded.

## 4. Gap 1 — the model-free reference suite

Four ABL-389 references, reported and gating nothing, on the frozen ABL-322 gate window
(2026-07-11 → 2026-08-10, exclusive), out-of-sample by target timestamp. Every reference
scored **n=720 / 720 / 510**, the same rows as the gate basis, and every climatology
covers all 24 hours — so these are like-for-like comparisons, not different measurements.

| pair | band | challenger | D-7 | constant causal | constant oracle | clim. causal | clim. oracle |
|---|---|---:|---:|---:|---:|---:|---:|
| DE | 24-36h | 66.11% | 88.9% | 74.40% | **62.86%** | 73.74% | **62.12%** |
| DE | 36-48h | 65.66% | 88.9% | 74.40% | **62.86%** | 73.74% | **62.12%** |
| DE | 48-64h | 66.15% | 87.1% | 75.79% | **62.32%** | 73.65% | **61.31%** |
| NL | 24-36h | 60.46% | 81.8% | 89.97% | 71.74% | 87.28% | 70.88% |
| NL | 36-48h | 61.26% | 81.8% | 89.97% | 71.74% | 87.28% | 70.88% |
| NL | 48-64h | 63.75% | 88.5% | 92.99% | 73.27% | 88.41% | 72.34% |

Bold marks a reference that beats the challenger.

### 4.1 The ABL-406 bar-weakness question — and why the flag disagrees with itself

On the registered flag the two pairs look opposite. They are not.

| pair | D-7 bar | `constant_causal` | registered flag | `constant_oracle` | correctly levelled |
|---|---:|---:|:---:|---:|:---:|
| DE 24-36h | 88.86% | 74.40% | **weak** | 62.86% | weak |
| DE 36-48h | 88.86% | 74.40% | **weak** | 62.86% | weak |
| DE 48-64h | 87.09% | 75.79% | **weak** | 62.32% | weak |
| NL 24-36h | 81.79% | 89.97% | not weak | 71.74% | **weak** |
| NL 36-48h | 81.79% | 89.97% | not weak | 71.74% | **weak** |
| NL 48-64h | 88.51% | 92.99% | not weak | 73.27% | **weak** |

A causal flat line clears the registered bar in **3 of 6** cells. A correctly-levelled
flat line clears it in **6 of 6**.

**NL's `bar_weaker_than_a_flat_line = false` is an ABL-435 artifact, not a strong bar.**
`constant_causal` is levelled on the fit window and scored on the gate window, and NL's
level moves enough between them to inflate that reference by 25–27% (§6). Corrected for
the levelling, NL's bar is as weak as DE's: a hindsight flat line beats it in every band.
This is the mechanism ABL-435 named and ABL-417 measured — where the oracle constant beat
the bar in 24/24 cells — reproduced here at **6/6**, on the first `wind_offshore` pairs it
has been applied to.

The consequence for ABL-406 is worth stating carefully, because it points the opposite way
to how it first reads. Taken at the registered flag, NL is a *strong bar that passes 3/3*
— the first counterexample to a correspondence that held across eight `wind_onshore` pairs
(5 weak → 5 pass, 3 strong → 3 fail/tie) and was reproduced by ABL-417 and ABL-421.
**Corrected for the levelling, the counterexample disappears**: NL is a weak bar that
passes, exactly as the pattern predicts. ABL-406's finding survives intact. What does not
survive is the flag that tests it — it read `false` on a pair whose bar a flat line clears
in every band. Both pairs here join the weak-bar side, where 9 of the 16 grade-A pairs in
ledger §4.1 already sit outright, with CH and NL `wind_onshore` flagged "no for the wrong
reason" by this same mechanism. DE and NL `wind_offshore` are the third and fourth pairs
to be caught by it.

## 5. Gap 2 — the grade

ABL-418 ladder, wind stream, k=1, readability floor **7.51%**. Applied by the harness at
`attach_grades`, so the markdown and the JSON cannot disagree.

| pair | bands | grade | failed conditions | bar weaker than a flat line? |
|---|---|:---:|---|:---:|
| DE wind_offshore | A / A / A | **A** | — | **yes** |
| NL wind_offshore | A / A / A | **A** | — | no |

All six cells clear G1 (skill vs D-7 +24.0% to +28.0%, against a 7.51% floor), G2, G3 and
G4. Coverage is not vacuous: every cell carries `enough_pairs = true` with n=720 against
a registered minimum of 684 (24-36h, 36-48h) and n=510 against 456 (48-64h) — no cell sits
at its minimum. `beats_d7 = true` in all six.

Both pairs are therefore promotion-*eligible* on the registered ladder. §6 is why that
does not make them servable, and why the two pairs part company there.

## 6. The screen that decides what the A is worth

The ledger's §5.2 finding applies here: `constant_causal` and `climatology_causal` are
levelled on the **fit** window and scored on the **gate** window, so where the fleet level
moves between them, G2/G3 are inflated for free. Measured on these two pairs, worst band:

| pair | `constant_causal` | `constant_oracle` | inflation |
|---|---:|---:|---:|
| DE wind_offshore | 75.79% | 62.32% | **+22%** |
| NL wind_offshore | 92.99% | 73.27% | **+27%** |

Both are affected, both mildly — comparable to PL (22%) and CZ (21%), nowhere near NL
`wind_onshore`'s 205%. But both G2 margins are partly unearned, so the ledger's §5.3
screen is the one that decides. Margins below are on the **challenger's own error**
(ABL-406's denominator, the one ABL-385's CV is measured in), worst band per pair, against
the 7.51% floor. **This is a CEO-side sensitivity and not a change to the registered
ladder**, which stays on causal references only.

| pair | worst band | vs oracle flat | vs oracle climatology | verdict |
|---|---|---:|---:|---|
| **NL** wind_offshore | 48-64h | **+14.93%** | **+13.47%** | clears both **readably** |
| **DE** wind_offshore | 48-64h | **−5.80%** | **−7.31%** | **loses to both** |

- **NL clears both oracle references by roughly twice the floor, in every band**
  (+14.93% to +18.67% vs oracle flat, +13.47% to +17.24% vs oracle climatology). On the
  ledger's §5.3 table it joins EE, GR and SE `wind_onshore` as the **fourth** pair in the
  programme to clear that screen, and the only non-onshore one. Its grade A survives the
  §5.2 critique intact: NL beats the correctly-levelled references, not just the
  mis-levelled ones.
- **DE loses to both oracle references in all three bands** (−4.26% to −5.80% vs oracle
  flat; −5.38% to −7.31% vs oracle climatology). It belongs in the ledger's
  "grade A, not servable" bucket. Stated precisely: **every one of those six losses sits
  inside the 7.51% floor** — the largest, −7.31%, by 0.2pp. They are directional, not
  readable at one seed. The honest disposition for DE is *re-read at k>1 seeds*, not
  *reject* — but nothing in this read supports serving it.

### 6.1 Direction — DE belongs in ledger §5.4

DE clears G4 on sign alone, at slope 0.064–0.073 and correlation 0.132–0.157 across the
three bands. That is the CH `wind_onshore` profile the ledger already lists as "passes
the registered gate with a slope at or near zero, weakly positive" (CH: slope
0.078–0.131, corr 0.138–0.246) — DE offshore is slightly weaker than CH on both. It is a
sixth entry for §5.4, not a new phenomenon.

NL is genuinely directional by comparison — slope 0.215–0.221, correlation 0.432–0.455 —
though a slope near 0.22 with a −12% to −15% bias means a heavily shrunk forecast, which
is a calibration lever rather than a defect.

### 6.2 DE's TSO gap is the real headline

Unchanged from the pilot and worth restating beside the grade: over the same n=1,950,
DE's own TSO forecast scores **21.1% WAPE against the challenger's 66.0%**. The TSO series
is revision-contaminated (a replacement table with no first-seen vintages) and cannot
support a promotion decision in either direction — but a factor-of-three gap is not a
revision artifact. For DE the finding is not "the model passes", it is "the model passes a
bar that a flat line also passes, while a forecast already in the database is three times
better." That is a feature-ingest lever, and it is the strongest recommendation this pack
carries. NL's TSO is 69.0%, worse than the challenger's 61.6%.

## 7. Contamination and limits

- **ABL-188** constant-run screening ran against `energy_generation`, the table these
  pairs are actually fitted from, and found no ≥24-hour bit-identical run in either pair.
  `constant_runs` is empty for both in the committed record, so the verdict is a plain
  PASS and not "PERFORMANCE PASS — HOLD FOR CONTAMINATION ADJUDICATION".
- **ABL-67** is net-position-only; **ABL-109 / ABL-111** are load-only. Neither intersects
  these wind targets.
- **ABL-71**'s known wrong-write modes are load and net position, not wind. That is a
  provenance caveat, not proof that wind ingest is pristine.
- **ABL-318**'s NL offshore defect (447 zero-filled rows, 668 disagreeing duplicate
  instants) lives in `energy_renewable`. These pairs train from `energy_generation`, so it
  does not reach the target — which is exactly why `--renewable-source` is mandatory here.
- The incumbent column reads **Not measured** by construction: DE and NL hold 0 rows in
  `forecasts` for `wind_offshore` (`comparator_n.incumbent = 0`). No incumbent was
  displaced, and no serving pair was refitted by this scope.
- **One 30-day summer holdout.** Out-of-sample by target timestamp, fitted on
  2026-01-14 → 2026-07-11 and scored on 2026-07-11 → 2026-08-10, both frozen at ABL-322's
  registration on 2026-08-12T21:00Z before either model existed. It is not a year-round
  robustness claim, and offshore wind is strongly seasonal.
- **k=1.** One seed per cell. The floor used throughout is ABL-385's `delta_min` with
  `c_B = 0` (7.51% on wind), correct against deterministic references; DE's oracle margins
  sit inside it and NL's do not.

## 8. Gap 3 — what is committed

| file | what it is |
|---|---|
| `experiments/ABL322/results_abl436_offshore_reread.json` | the full machine record — 6 graded cells, 8 comparators each with its own n, reference levels, fit audit, timings |
| `experiments/ABL322/abl436_preregistration_recheck.json` | §3's re-derivation of the pre-challenger D-7 bar |
| `reports/abl_436_offshore_reference_grade.md` | the harness-generated gate report |
| this file | the evidence pack |

Model artifacts stay local under `experiments/ABL436/artifacts/` (`.gitignore:56`), as for
every other tranche. The pilot's own artifact directory was not written to.

## 9. Recommendation to the CEO

1. **NL `wind_offshore` belongs in the ledger's §5.3 top bucket**, as its fourth member
   and only non-onshore one. It beats both correctly-levelled oracle references by roughly
   twice the readability floor in every band — the margin that survives §5.2, since it is
   measured against the references the mis-levelling does not touch. If any wind pair goes
   to the Board, this is the one with the evidence behind it.
2. **DE `wind_offshore` should not be served on this evidence.** Grade A, but on a weak
   bar, losing to both oracle references in all three bands, at a near-zero slope, with a
   TSO forecast in the same database three times better. Not a rejection — the oracle
   losses sit inside the floor — but a *re-read*, and behind ingesting the TSO series as a
   feature.
3. **§4.1 should reach ABL-437 and ABL-435.** NL reads as a counterexample to ABL-406 on
   the registered flag and stops being one the moment the flag is correctly levelled — so
   what this pair actually adds is a fourth witness that the *flag* is unreliable, at 6/6
   cells, on the first `wind_offshore` pairs it has been tested on. ABL-406's finding is
   unharmed; the instrument that reports it is the thing to fix.
4. **The harness gap in §1.1 needs its own change** — `abl322-pilot` is still registered to
   write its record where `.gitignore` swallows it, and to overwrite the pilot's
   dispositioned report on an unflagged run.

Promotion is a CEO-to-Board decision. This issue recommends and does not promote.
