# ABL-381 — ABL-316 tranche 1b: BG and CH solar on `energy_generation`

**Disposition: PASS, 6/6 cells, both pairs.** The pass survives re-scoring on
daylight hours only, so it is not an artefact of the night rows discussed in §5.

Machine record: `experiments/ABL348/results_abl381_tranche1b.json`.
Harness report: `reports/abl_381_solar_tranche1b.md`.
Probes: `reports/abl_381_tranche1b_precheck.json`,
`reports/abl_381_nonneg_and_constant_probe.json`,
`reports/abl_381_night_floor_probe.json`.

Two findings outrank the pass: **BG's solar actuals carry a large overnight
floor** (§5), and **CH's TSO forecast beats the challenger** (§6). Neither
changes the disposition; both change what the disposition is worth.

---

## 1. Protocol, and what was verified rather than trusted

Registration is ABL-348's, frozen at `experiments/ABL348/config.json` and shared
with the wind tranche. Nothing here re-derives it. Fit 2026-01-14 → 2026-07-11,
gate 2026-07-11 → 2026-08-10 (both exclusive), bar seasonal-naive D-7, source
`energy_generation`, metric WAPE, primary bands 24-36h / 36-48h / 48-64h.

**Pre-registration precedes the first fit, checkable in git.** Scope
`abl316-t1b` landed in `SCOPES` and `GATE_BASIS` at commit `776bfe7`, authored
**10:52:36**. The BG artifact was written **10:55** and the CH artifact
**10:59**. The registration commit is the earlier object; the ABL-322 property
holds again.

The ABL-389 re-read (§3) refitted both pairs at **12:40**, so the artifacts on
disk now postdate that first fit. This does not weaken the property: the
registration commit still precedes *every* fit, and the fit is deterministic
under the pinned seed, so the re-read reproduced the first read's challenger and
D-7 WAPE to 1e-12 in all six cells rather than drawing a second result to choose
from. The registered bar, bands, windows and source were never re-derived. See
§7d for why the artifact hash changed anyway and why that is not evidence of a
changed model.

**Gate basis is the two-way `("challenger", "seasonal_naive")`**, not by
preference but by necessity. Measured read-only on the live replica: `forecasts`
holds solar rows for BE/DE/FR/AT only, and **zero** for both BG and CH. Naming
`incumbent` would empty the intersection and return 6 cells at n=0 — a
model-quality verdict on a race that never ran. The incumbent renders
**"Not measured"** in all 6 cells and voids none of them, which is the
acceptance criterion this issue set.

**Provenance, measured not assumed.** The run recorded
`features_match_replica: true` and `ambient_matches_replica: true` against
`C:\Code\able\data\energy_dashboard.db` at **9,432,453,120 bytes** — the live
replica, not the 3.0 GB partial snapshot. Opened `mode=ro`. Sidecar
`C:\Code\able\data\forecasts_local.db` was set explicitly. No write of any kind
was made to the replica.

**Preconditions re-verified rather than carried.** `--renewable-source
energy_generation` passed explicitly. Both artifacts carry
`training_source = energy_generation` in the bundle. Fits ran under
`.venv\Scripts\python.exe` (Python 3.14.3). Both pairs: 720/720 gate hours
present, 0 missing, 4,272 hourly fit rows, no native sub-hourly data in the gate
window, and **720/720 hours bit-identical** against `energy_renewable`
(`max_abs_diff_mw = 0.0`). ABL-188 constant-run screening returned **empty** for
fit, gate and the 14-day feature lookback on both pairs. The issue's table
reproduces exactly: BG mean 1,439.2 MW / D-7 24.40%, CH mean 1,331.0 MW / D-7
12.67%.

**Contamination.** ABL-67 is net-position-only; ABL-109/111 are load-only;
ABL-71's known wrong-write modes are load and net position. None of the four
touches solar in this window. That is a provenance statement, not a clean bill —
§5 documents a solar data defect these four issues do not cover.

**Caveat carried from the harness, not introduced here.** 23,674 of 34,176 fit
rows (69.3%) have a degraded `lag_1d` feature, identically for both countries.
That is the serve-faithful schedule, not a data fault, but it is the dominant
feature-quality limit on these fits.

## 2. The gate read — registered, frozen, and what dispositions

| country | band | n | challenger WAPE | D-7 WAPE | skill vs D-7 | gate |
|---|---|---:|---:|---:|---:|:---:|
| BG | 24-36h | 720 | 18.89% | 24.40% | +22.6% | **PASS** |
| BG | 36-48h | 720 | 18.60% | 24.40% | +23.8% | **PASS** |
| BG | 48-64h | 510 | 20.03% | 24.99% | +19.9% | **PASS** |
| CH | 24-36h | 720 | 8.16% | 12.67% | +35.6% | **PASS** |
| CH | 36-48h | 720 | 8.01% | 12.67% | +36.8% | **PASS** |
| CH | 48-64h | 510 | 8.39% | 12.53% | +33.0% | **PASS** |

All n meet or exceed the registered minimum (48-64h intended 480, delivered 510).
Out-of-sample by target timestamp; gate targets were never fitted. One 30-day
summer holdout — not year-round evidence.

## 3. The reference the CEO authorised, and why a flat line is the wrong one for solar

Reported per cell, on exactly the rows that cell scored. **A reported reference,
not a gate criterion**; the registered bar is unchanged.

All four references are now **ABL-389's**, computed by
`src/evaluation/model_free_reference.py` and attached by
`attach_model_free_references` from the same ABL-188-filtered `builder._actuals`
the gate actuals and the D-7 baseline come from. The local implementation this
read originally carried has been deleted, on the CEO's instruction and for the
reason ABL-389 exists: two implementations of a same-named reference are free to
drift, and a number that differs by provenance is not a measurement anyone can
audit. Each column is scored on **its own intersection** with the gate basis and
carries its own `n`; all four equal the cell `n` on both pairs, because BG and
CH each cover 24/24 hours of the day in both windows.

| country | band | n | challenger | D-7 | causal constant | oracle constant | causal climatology | oracle climatology |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BG | 24-36h | 720 | 18.89% | 24.40% | 75.30% | 73.49% | 41.98% | 19.15% |
| BG | 36-48h | 720 | 18.60% | 24.40% | 75.30% | 73.49% | 41.98% | 19.15% |
| BG | 48-64h | 510 | 20.03% | 24.99% | 68.17% | **63.82%** | 41.33% | 20.38% |
| CH | 24-36h | 720 | 8.16% | 12.67% | 95.08% | 94.65% | 37.53% | 9.02% |
| CH | 36-48h | 720 | 8.01% | 12.67% | 95.08% | 94.65% | 37.53% | 9.02% |
| CH | 48-64h | 510 | 8.39% | 12.53% | 85.99% | **87.91%** | 36.56% | 8.70% |

Causal constant = fit-window mean (BG 855.24 MW, CH 833.37 MW), available at
forecast time. Oracle constant = the **whole-gate-window** median (BG 1087.86 MW,
CH 677.22 MW), hindsight, and the best a single flat line can do over the window.
Causal climatology = fit-window hourly mean; oracle climatology = whole-gate-window
hourly median. All four levels reproduce the Founding Engineer's independently
published values to the decimal, which is what establishes that this read and
ABL-389 loaded the same series.

### What the swap moved, measured rather than assumed

The re-read was expected to change provenance and not value. **It changed one
number in each of two cells, and the exception is worth more than the
confirmation.** Scoring both implementations on the same rows at full precision:

| column | cells 24-36h / 36-48h (n=720) | cells 48-64h (n=510) |
|---|---|---|
| `constant_causal` | identical to 1e-12 | identical to 1e-12 |
| `constant_oracle` | identical to 1e-12 | **BG +1.25pp, CH +8.39pp** |
| `climatology_causal` | identical to 1e-12 | identical to 1e-12 |
| `climatology_oracle` | identical to 1e-12 | identical to 1e-12 |

23 of 24 readings are bit-identical. The one that moved is `constant_oracle` on
the 48-64h band, and the cause is not the missing-hour fallback the hold
anticipated — that fallback never fires here, exactly as predicted. It is that
**the deleted local version took its oracle level on each cell's own rows, and
the canonical module takes one level per pair over the whole gate window.** The
two agree wherever a cell covers the full window and can only differ where it
does not, which is the 48-64h band alone.

**Why the constant moves and the climatology does not, and why that generalises.**
The 48-64h band is a 16-hour-wide lead-time slice, so its rows are not a random
subset of the window — they are a different mix of day and night. CH's night
fraction is 0.204 at 48-64h against 0.258 at 24-36h. An *unconditional* median is
highly sensitive to that mix on a series that is near-zero half the time, so
CH's constant oracle moves 8.39pp; BG's mix barely changes (0.294 vs 0.292) and
its constant moves 1.25pp. An *hour-conditioned* median is invariant to it by
construction, which is why all six climatology readings are unchanged.

Two consequences for the remaining 33:

1. **`constant_oracle` is not comparable across horizon bands on solar.** Its
   level is fitted on the whole window but scored on a band whose hour-of-day
   composition differs, so a band-to-band change in that column is a statement
   about the band's day/night mix and not about the model. CH's column reads
   94.65 / 94.65 / 87.91 across the three bands for that reason alone.
2. **The canonical whole-window level is the right choice anyway**, and the
   deleted per-cell version was the more flattering one — it re-optimised the
   hindsight constant separately for every band, making the reference harder to
   beat and the challenger look worse. Moving to the canonical module weakened
   this reference on 2 of 6 cells. It changes no verdict: the challenger beats
   every reference in every cell either way.

**On solar the constant reference does not do the job it did on wind.** ABL-380's
value came from a constant being *competitive* — CH wind cleared its bar while
scoring 7.1pp worse than an oracle constant. Here every constant is 63–95% WAPE,
because a flat line cannot represent a diurnal cycle at all. A constant that
loses by 60pp tells you the sun rises, not whether the model is any good.

**The honest analogue on solar is an hour-of-day climatology.** Same idea (a
predictor with no model in it), but able to represent the one structure that
dominates the series. It was first measured on this tranche, is now ABL-389's and
lives in the shared module, and since that merge the harness prints it in the
headline gate table for every scope — so the qualification below travels with the
verdict to the remaining 33 without anyone having to re-derive it. It is far more
informative than the constant:

- **CH's oracle climatology is 9.02% against a challenger at 8.16%.** The model
  beats a hindsight hour-of-day median by **0.86pp**. Its 35.6% skill over D-7
  is real but flatters it; against the diurnal cycle it is close to a tie.
- **BG's challenger (18.89%) is marginally better than the oracle climatology
  (19.15%)** — 0.26pp, inside anything I would call a margin.
- Against the *causal* climatology (37.5–42.0%), which is what is actually
  available at forecast time, both models win decisively. That is the fair
  comparison, and both pass it.

The disposition stands. But **"clears seasonal-naive D-7" is a low bar on solar**,
and the D-7 bar being uninformative is now measured on both wind (ABL-380) and
solar. I recommend the climatology reference be reported for the remaining 33.

## 4. ABL-338 non-negativity — the issue's premise is refuted by ABL-338's own evidence

The issue asks me to confirm ABL-338's non-negativity constraint is **active** in
these fits, on the grounds that "new solar fits should inherit it". Measured:

| | BG | CH |
|---|---|---|
| `nonneg_objective` in artifact | `None` | `None` |
| `loss_function` | unset (CatBoost default RMSE) | unset |
| geometry features in fit | none | none |
| feature count | 25 | 25 |

**It is not active — and that is correct, not a defect.** ABL-338's own verdict
is that the constraint *as specified* **degrades daylight accuracy and was
rejected**: the log-link arms cost up to **+15.8%** daylight MAE (Tweedie) and
**+36.8%** (Poisson) against a like-for-like refit. It landed as a reviewed,
**unadopted** capability — which is exactly what `nonneg_objective = None`
records. Inheriting it would have made these models worse. The premise in the
issue text is the thing that does not survive contact with ABL-338.

**The real gap is the other half of ABL-338.** The geometry features
`sun_elevation_deg` / `is_night` *were* adopted — daylight-safe, mean −1.0% and
worst +2.9% across ABL-338's eight country-windows — but
`src/evaluation/solar_retrain.py:FEATURE_COLUMNS` names neither, so the gate
harness cannot pick them up. These artifacts carry 25 features where an
ABL-338-current solar fit would carry 27. Every one of the remaining 33 tranches
inherits that omission.

What it costs, measured on the scored rows:

| | BG | CH |
|---|---:|---:|
| negative predictions | 141 / 2,730 (5.17%) | 701 / 2,730 (25.68%) |
| most negative | −33.4 MW | −32.0 MW |
| night rows predicted negative | 113 / 690 (16.4%) | 515 / 640 (**80.5%**) |
| mean prediction at night | +224.78 MW | −5.48 MW |
| mean **actual** at night | +225.13 MW | 0.00 MW |

CH is the textbook ABL-338 defect: the actuals are exactly zero at night and the
model predicts slightly negative in 80% of those hours. It is small in MW and
does not move the gate, but it is the shape ABL-337's serving clamp exists to
catch. **Recommendation: add the two geometry features to the harness's
`FEATURE_COLUMNS` before the remaining 33** — daylight-safe on ABL-338's
measurement, and the mechanism it identified (nothing distinguishes "0 W/m²
because the sun is down" from "0 W/m² at a dark dawn") applies unchanged here.
That is a Founding Engineer change to a shared harness, not something I should
land inside a tranche read.

## 5. The finding that outranks the pass: BG's solar actuals have a large overnight floor

BG's mean *actual* at night is **225.13 MW**. CH's is **0.00 MW**. Night is
`solar_features.night_mask` — the sun geometrically below −8° for the whole hour
at the country's capacity-weighted point — so this cannot be a timezone offset or
a mask artefact. It is a property of the series.

Measured on both source tables (`reports/abl_381_night_floor_probe.json`),
threshold 1 MW, the same one ABL-338 used on FR:

| pair / source | window | night hours >1 MW | night mean | night max | share of total energy at night |
|---|---|---:|---:|---:|---:|
| BG `energy_generation` | fit | 1,168 / 1,529 (**76.4%**) | 152.33 MW | 1,097.37 MW | **6.37%** |
| BG `energy_generation` | gate | 179 / 210 (**85.2%**) | 245.71 MW | 1,087.86 MW | **4.98%** |
| BG `energy_renewable` | fit | 1,156 / 1,517 (76.2%) | 152.53 MW | 1,097.37 MW | 6.37% |
| BG `energy_renewable` | gate | 179 / 210 (85.2%) | 245.71 MW | 1,087.86 MW | 4.98% |
| CH `energy_generation` | fit | 673 / 1,465 (45.9%) | 1.32 MW | 5.84 MW | 0.05% |
| CH `energy_generation` | gate | 0 / 186 | 0.00 MW | 0.00 MW | 0.00% |
| CH `energy_renewable` | both | 0 | 0.00 MW | 0.00 MW | 0.00% |

Three things follow.

1. **It is far worse than the FR defect ABL-337 filed and ABL-338 measured.** FR
   had 488 of 11,614 night rows above 1 MW (4.2%), max 439.3 MW. BG has **76–85%
   of night rows**, max **1,097 MW** — 2.5× FR's peak, and roughly a twentieth of
   everything BG books as solar is booked in the dark.
2. **It is not source-specific.** `energy_generation` and `energy_renewable`
   agree to the decimal on the max and give identical gate-window counts. So it
   is upstream of the source mapping — an ingest or ENTSO-E feed property, not a
   table-selection defect. Switching source will not fix it.
3. **The model learned it faithfully** — mean night prediction 224.78 MW against
   mean night actual 225.13 MW. BG's model is doing its job on the series it was
   given; the series is the problem.

**This does not manufacture BG's pass.** Re-scoring the same challenger and the
same D-7 on daylight rows only:

| country | band | daylight n | challenger | D-7 | clears |
|---|---|---:|---:|---:|:---:|
| BG | 24-36h | 510 | 18.90% | 24.35% | yes |
| BG | 36-48h | 510 | 18.59% | 24.35% | yes |
| BG | 48-64h | 360 | 20.17% | 25.05% | yes |
| CH | 24-36h | 534 | 7.99% | 12.67% | yes |
| CH | 36-48h | 534 | 7.85% | 12.67% | yes |
| CH | 48-64h | 406 | 8.29% | 12.53% | yes |

**6/6 still clears**, and BG's numbers barely move (18.89 → 18.90). The night
energy inflates the WAPE denominator, but the model tracks it closely enough that
numerator and denominator scale together. The floor is a data-quality defect worth
fixing on its own terms, not a threat to this disposition.

**Escalation: this needs a CEO decision and probably an ingest fix.** ABL-338
found that excluding FR's night rows moved FR's mean night prediction from 22.46
to 0.05 MW *and improved its daylight MAE by 1.5%* — so the floor is not free
even for daylight accuracy. I have not filed against BG's ingest myself because I
do not own ingest; I recommend a defect issue owned by whoever owns ENTSO-E solar
ingest, with BG checked first and **all 35 remaining countries screened for the
same floor before their tranches run** (see §8).

## 6. CH's TSO forecast beats the challenger

All-D+2, per-country, each comparator on its own intersection with the gate basis
(n = 1,950 for every figure below):

| country | challenger | D-7 | persistence | TSO (revision-contaminated) |
|---|---:|---:|---:|---:|
| BG | 19.1% | 24.6% | 73.2% | 33.2% |
| CH | **8.2%** | 12.6% | 87.5% | **7.1%** |

BG's challenger beats TSO comfortably. **CH's does not** — TSO is 1.1pp better.
This is the same shape as ABL-380's finding 5 on the wind side, now reproduced on
solar, and it is the second independent signal (with §3's climatology) that CH's
model is the weaker of the two despite carrying the larger headline skill number.

The caveat is unchanged and is why this cannot gate anything: TSO comes from an
`INSERT OR REPLACE` table with no first-seen vintages, so those values may include
revisions and are not serve-faithful. It is context. But "the TSO forecast we
already ingest is better than the model we just fitted" is the right question to
put in front of a promotion decision for CH, and I am putting it there.

## 7. Harness defects found

**7a. The solar harness hardcoded ABL-253's heading — found here, fixed here.**
`scripts/evaluate_solar_retrain.py` rendered `"# ABL-253 — Serve-faithful solar
retrain gate"` as a literal, not derived from the scope, so a correct BG/CH gate
read came out **headed with another issue's number** — and would have on all 33
remaining tranches. ABL-387 fixed the output paths but not the title. The wind
twin has derived it since ABL-322, so the fix is that line ported:
`f"# Serve-faithful solar retrain gate — registered scope \`{meta['scope']}\`"`.
The first read of this tranche reported the defect and left the artifact
byte-exact; that was the right call while the fix was another issue's, but the
re-read had to regenerate the file anyway, so it is corrected at the source
instead of annotated.

**7b. ABL-387's depth guard tests a proxy, not the property.**
`test_experiment_outputs_stay_one_directory_deep` requires `experiments/<dir>/<name>`
exactly. The run wrote to `experiments/ABL316/artifacts/t1b` to keep 33 tranches
from sharing a directory; `git check-ignore -v` confirms that path *is* ignored by
`.gitignore:56` (`experiments/*/artifacts/` matches on directory name, so
everything beneath it is ignored too). The guard's stated purpose — don't commit a
2 MB binary — was satisfied; the depth proxy rejected it anyway.

I conformed rather than loosen a freshly-landed guard from another issue: on the
first read the two artifacts were **moved** to
`experiments/ABL316/artifacts/{BG,CH}/solar/` and the registration names that.
Founding Engineer's call whether the guard should assert "is git-ignored" instead
of "is 3 path segments" — it matters because per-tranche grouping is the natural
layout for 33 more tranches.

**7d. The artifact SHA-256 cannot witness a refit — `Forecaster.save` stamps
`saved_at`.** The re-read refitted both pairs, and both hashes changed:

| pair | first read | re-read |
|---|---|---|
| BG | `9bbe1e74…aa5e` | `380e5c88…051b` |
| CH | `9ff1a53d…dd5e` | `f79338bb…a270` |

The models are nevertheless **identical**: challenger and D-7 WAPE reproduce to
1e-12 in all six cells (18.885225 / 18.598880 / 20.027198 for BG, 8.161702 /
8.007164 / 8.394333 for CH), which is the check that actually establishes it.
The hashes differ because `src/forecaster.py:save` writes
`"saved_at": datetime.now().isoformat()` into every bundle, so two bit-identical
models are guaranteed different SHA-256 values.

This matters beyond bookkeeping. The first read published those hashes as "the
artifacts' binding identity", and for the *move* they performed that job
correctly — a content-preserving copy leaves the hash alone. But the property
worth having across 33 tranches is **"this fit reproduces"**, and the hash cannot
express it: it will differ on every re-run no matter what, so a mismatch carries
no information and a match is impossible. Anyone who reads a changed SHA as a
changed model will conclude the tranche is irreproducible when it is exactly
reproducible. `random_seed: 42` is fixed in `config.py`, so **prediction equality
is the reproducibility witness** — the SHA identifies a file, not a model. I have
stopped quoting it as one, and the earlier claim above is corrected here rather
than left standing. Founding Engineer: excluding `saved_at` from the hash (or
hashing the estimator alone) would make the artifact hash mean what the reports
have been implying it means.

**7c. The registration-table cross-check earned its keep.** `abl316-t1b` landed in
`SCOPES`/`GATE_BASIS` at `776bfe7`, before `SCOPE_OUTPUTS` existed. Merging ABL-387
was textually clean and GitHub would have reported it mergeable, but import raised
`SCOPE_OUTPUTS is missing 'abl316-t1b'`. Nothing else would have caught it. This is
the second time that guard has fired on exactly this sequence (the wind twin
records the first).

## 8. What makes the remaining 33 different from these two

1. **Screen every country for the §5 night floor before fitting it.** BG passed
   every ABL-348 evaluability check — 720/720 hours, bit-identical across sources,
   no ABL-188 hit — and still carries a defect that none of those checks look for.
   The screen is cheap (`scripts/abl381_night_floor_probe.py`, seconds per country,
   read-only) and it is the difference between a tranche that means something and
   one that certifies a contaminated series.
2. **BG and CH were chosen as the *easy* pairs and one of them is contaminated.**
   ABL-348 picked them for full history, full gate hours and cross-source
   agreement. Whatever fraction of the other 33 carries a night floor, it is not
   lower than it is here.
3. **The geometry-feature gap (§4) is repo-wide**, so all 33 will produce
   25-feature artifacts unless the harness changes first.
4. **Cost is higher than ABL-322's sizing, and these were the cheap case.**
   The first read estimated ~3.5 min/pair from artifact mtimes. The ABL-389
   re-read gives a **directly measured CLI wall-clock: 6 min 06 s for the two
   pairs, ~3.0 min/pair**, `time` on the full `--scope abl316-t1b` invocation.
   That is a ~3× upward correction against the ~60 s/pair ABL-322 measured and
   this issue's expectation of "appreciably less" for hourly countries, and it
   independently confirms ABL-380's finding 6 on the wind side. BG and CH are
   hourly, so the 15-minute countries in the remaining 33 are the expensive case
   and will be worse. 33 pairs is therefore closer to **two hours** of compute
   than to half an hour — still minutes-scale and not a budget problem, but the
   sizing assumption in ABL-316 should be corrected before someone plans around
   it. (Measured on a box running other work concurrently, so read it as an upper
   bound; the mtime estimate and the wall-clock agree to within 15%, which is the
   useful part.)
5. **EE and FI solar remain NOT-EVALUABLE** under the ABL-348 registration —
   EE on both sources, FI broken by the source change. Neither is in this tranche.
   They must not surface later as gate *failures*; they are evaluability failures,
   declared before any model existed.
6. **Do not read `constant_oracle` across horizon bands on solar** (§3). Its level
   is one number per pair over the whole gate window, but the 48-64h band scores
   it on a 16-hour lead-time slice with a different day/night mix, so the column
   moves for reasons that have nothing to do with the model — 8.39pp on CH here.
   `climatology_oracle` is invariant to that by construction and is the column to
   compare across bands. This will recur on all 33 and on every solar pair whose
   bands differ in night fraction, which is all of them.

## 9. Recommendation

- **Disposition both pairs PASS**, 6/6, on the registered bar. Preserve the
  artifacts and the machine record.
- **No promotion from this issue**, and no serving-registry change. Promotion is
  CEO-to-Board. Nothing here was deployed; no replica write was made; no pair
  serving today was refitted.
- **Qualify CH's pass explicitly if it goes to the Board.** It clears D-7 by 33–37%
  while (a) being beaten by the TSO forecast we already ingest, and (b) beating a
  hindsight hour-of-day climatology by 0.86pp. That combination should be in front
  of the decision, not behind it.
- **Three handoffs to the Founding Engineer**, none of which I should land inside a
  tranche read: the geometry features missing from the harness `FEATURE_COLUMNS`
  (§4), the depth-guard proxy (§7b), and `saved_at` making the artifact SHA-256
  useless as a reproducibility witness (§7d). The hardcoded ABL-253 heading (§7a)
  *is* fixed here — it is one line, it is the wind twin's line, and the re-read
  had to regenerate the mislabelled file anyway.
- **One escalation to the CEO**: BG's overnight solar floor (§5) is a data defect
  on both source tables, it is upstream of anything this module controls, and the
  remaining 33 countries should be screened for it before their tranches run.
