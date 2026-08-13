# ABL-376 — excluding physically impossible night rows from the solar fit

**Forecasting Scientist, 2026-08-13.** Evidence pack for the fit-rule change on
branch `fix/abl-376-solar-night-fit-exclusion`. No production deploy, serving
registry change, model promotion, ingest change, dashboard change, replica write
or sidecar write was performed.

## Summary

| | |
|---|---|
| Change | Solar fit drops night rows whose actual the sun says is impossible. Fit side only, never the score. |
| Night predicate | `solar_geometry.is_night_hour` — the serving clamp's own, sun below -8 deg geometric for the whole hour |
| Threshold | 1 MW, ABL-338's, kept rather than re-derived |
| Registered gate | Both arms **PASS 9/9** cells |
| Night, over 8 seeds | FR 43.66 → 43.33 MW — **no collapse**; the issue's 22.46 → 0.05 does not reproduce here (§5) |
| Daylight, over 8 seeds | FR **+0.38%** (worse), inside a 4.40% single-seed null; the issue's −1.5% does not reproduce either (§5) |
| Disposition | Land it as a correctness rule, not as an improvement. Not a promotion recommendation. |

**Read this pack as replacing the issue's two headline numbers, not confirming
them.** They were measured on ABL-338's frame — 27,228 training hours from 2023,
training-time features including `is_night`, a spring holdout. On the registered
frame, at eight seeds, both effects are inside the noise and the daylight one
points the other way. §5 has the numbers and §5's last part has the two
candidate reasons the frames disagree.

What survives unchanged: the rule is principled, it is a provable no-op where
the data is clean (§3), and it does not threaten the gate (§4).

## 1. What I could and could not reproduce

The issue states FR carries **488 of 11,614** night training rows above 1 MW,
max 439.3 MW, on 337 distinct days; DE 4 rows at max 1.7 MW; AT and BE none.

**DE, AT and BE reproduce exactly.** Over the registered fit window the rule
removes **4 DE hours, max 1.75 MW**, and nothing at all for AT and BE.

**FR's figure is frame-dependent and should not be quoted as a property of the
source.** Three different frames give three different counts, all of the same
shape:

| frame | FR night rows | above 1 MW | max |
|---|---:|---:|---:|
| source, `:00` sub-sample, whole history | 12,022 | 531 | 440.0 MW |
| source, `:00` sub-sample, ABL-338's pre-holdout window | 11,250 | 454 | 440.0 MW |
| **the fit frame itself, registered window** | **11,648** | **904 rows / 113 hours** | **285.9 MW** |

Two things drive the spread. The fit frame is **hourly means** of a 15-minute
series — the FR builder aggregates 21,190 sub-hourly rows into 5,299 hourly
means (ABL-332) — so the `:00` sample and the fitted series are different
numbers, not different counts of the same number. And fit rows are per
(target, vintage), eight vintages per target, so 904 rows are 113 distinct
contaminated hours.

The finding itself is not in doubt: FR books solar in the dark, on a scale
nothing else here does, and no physical threshold can honour it.

## 2. Protocol

Two reads of one registered protocol, same replica, same day, differing in
exactly one thing.

| | control | treatment |
|---|---|---|
| scope | `abl253` | `abl376` |
| fit rule | off | on |
| countries, basis, windows | identical | identical |
| outputs | `reports/abl_376_solar_night_fit_control.md` | `reports/abl_376_solar_night_fit.md` |

Replica `C:\Code\able\data\energy_dashboard.db` (9,432,453,120 bytes), source
table `energy_renewable`. Fit targets 2026-01-14 → 2026-07-11; gate targets
2026-07-11 → 2026-08-10, out-of-sample by target timestamp. Baseline is literal
seasonal-naive D-7. CatBoost, `random_seed=42`, 500 iterations.

The control was written to **non-registered paths on purpose**: re-running the
`abl253` scope at its own registered paths would have overwritten ABL-253's
dispositioned gate read in place (ABL-387's failure mode). Verified after the
fact — `experiments/ABL253/` still holds only its pre-existing `config.json`,
and `reports/abl_253_solar_retrain.md` is unmodified.

### Why the control is a fresh run and not the published ABL-253 report

Re-running ABL-253's protocol unchanged, 29 hours after it was published,
does not reproduce its numbers:

| | published 2026-08-12 06:51 | re-read 2026-08-13 11:39 |
|---|---:|---:|
| DE 48-64h MAE | 2,902.0 MW | 2,743.6 MW (**-5.5%**) |
| FR incumbent WAPE (48-64h) | 17.8% | 16.0% |
| FR incumbent WAPE (24-36h) | 20.2% | 16.3% |
| BE, all three bands | 336.9 / 348.3 / 544.3 MW | 336.9 / 348.3 / 544.3 MW (identical) |

**A day of ordinary revision moves these numbers by more than the fit rule
does.** The gate is frozen in protocol, not in data. Any A/B here has to be
same-day against a fresh control; quoting the published table as the baseline
would have attributed a -5.5% data revision to a fit rule that cannot reach DE.

## 3. What the rule removed

From the treatment run's own audit, printed in its scorecard:

| country | night fit rows | excluded rows | excluded hours | max excluded actual |
|---|---:|---:|---:|---:|
| BE | 10,856 | 0 | 0 | n/a |
| DE | 10,952 | 32 | 4 | 1.7 MW |
| FR | 11,648 | 904 | 113 | 285.9 MW |

FR loses 2.7% of its fit rows; DE 0.1%; BE none.

**BE is the control the rule provides for itself.** Its two gate artifacts
predict **bit-identically** — the rule is a provable end-to-end no-op where the
data is clean, and it is stated over countries rather than as an FR special case
precisely so that this is checkable. It also establishes that the harness is
deterministic given (data, seed), which is what makes the seed-paired design in
§5 valid.

## 4. The registered gate: both arms PASS 9/9

| country | band | control WAPE | treatment WAPE | D-7 | gate |
|---|---|---:|---:|---:|:---:|
| BE | 24-36h | 15.56% | 15.56% | 32.94% | PASS |
| BE | 36-48h | 16.09% | 16.09% | 32.94% | PASS |
| BE | 48-64h | 18.89% | 18.89% | 33.17% | PASS |
| DE | 24-36h | 13.37% | **13.28%** | 24.20% | PASS |
| DE | 36-48h | 13.70% | **13.48%** | 24.20% | PASS |
| DE | 48-64h | 12.77% | **12.69%** | 23.34% | PASS |
| FR | 24-36h | 14.16% | 14.29% | 22.46% | PASS |
| FR | 36-48h | 14.75% | 14.90% | 22.46% | PASS |
| FR | 48-64h | 14.72% | 14.97% | 22.21% | PASS |

n = 720 / 720 / 480 per band. **The change does not threaten the gate.**

DE improves in all three bands. FR moves 0.13–0.25pp *against* the challenger,
and that direction is expected rather than alarming:

**FR's gate window is itself contaminated, and we are still scored on it.** Over
the 720 gate hours FR has **24 night hours reading above 1 MW, up to 365.5 MW** —
5,281 MWh booked in the dark, 0.11% of the window's total energy. DE and BE have
**zero**. The treatment model predicts ~0 there by design; the control model
partly chased the contamination and was rewarded for it. The maximum WAPE the
rule can concede on that account is ~0.11pp, which covers most of the 0.13pp and
0.15pp moves outright.

That is the fit/score asymmetry doing exactly what it was built to do. A rule
that also filtered the gate frame would have shown FR improving, and the
improvement would have been the filter marking its own homework.

## 5. Daylight and night, decomposed over a seed spread

**Neither of the two effects this issue was filed on survives a seed spread on
the registered frame.** That is the finding of this section.

### Protocol

Eight seeds — `101, 103, 107, 109, 113, 127, 131, 137` — frozen in
`scripts/abl376_night_seed_spread.py` at commit `b7af17d`, before the first fit,
and deliberately disjoint from the gate's seed 42: a spread anchored on the arm
that produced the headline is not a spread. Two arms per seed differing in
exactly one thing, the fit rule, fitted on frames built **once** per country and
shared by every fit, so at each seed the arms differ by nothing else. The
difference is therefore taken *within* a seed and across-seed variance never
enters it.

Both arms are scored on identical, **unfiltered** gate rows — the registered
window 2026-07-11 → 2026-08-10, 1,950 rows per country after the latest-vintage
selection, split by `solar_features.solar_bands` into daylight / shoulder /
night (FR: 1,243 / 245 / 462). Out-of-sample by target timestamp. Feature set is
the gate's own 25 legacy columns. Replica and source table as §2.

This scores on rows where the actual and the features are finite, not on §4's
four-column gate basis, so its `n` is its own and is quoted above; what matters
is that both arms see the same rows. (The registered JSON predates the
`feature_set` key that later runs carry — the flag that added it is additive and
default-off, and it postdates this read.)

### The night axis — the claimed result does not reproduce

Mean challenger prediction over the gate's night hours, MW:

| country | night rows | control | night-fit | paired change | t (df=7) |
|---|---:|---:|---:|---:|---:|
| FR | 462 | 43.66 ± 10.06 | 43.33 ± 19.59 | **−0.33 MW** | −0.04 |
| DE | 420 | −0.17 ± 43.07 | −10.02 ± 14.68 | −9.85 MW | −0.55 |
| BE | 420 | −6.51 ± 3.19 | −6.51 ± 3.19 | 0.00 MW | — |

The issue reports FR's mean night prediction going **22.46 → 0.05 MW**. Here it
goes 43.66 → 43.33 — a third of a megawatt, against a within-arm spread of 19.6
and a single-seed null whose maximum is 28.81 MW. **The night level does not
collapse on this frame.** It is not made worse either; the rule is simply not
what is holding it up.

### The daylight axis — inside its own null, and pointing the other way

| country | daylight rows | control MAE | paired change | as % | seeds improved | single-seed null (max) |
|---|---:|---:|---:|---:|---:|---:|
| FR | 1,243 | 1,602.1 MW | **+6.03 MW** | **+0.38%** | 2/8 | 70.5 MW (4.40%) |
| DE | 1,278 | 3,784.3 MW | −22.88 MW | −0.60% | 6/8 | 140.0 MW (3.70%) |
| BE | 1,311 | 580.0 MW | 0.00 MW | 0.00% | 0/8 | 31.3 MW (5.39%) |

The issue reports FR daylight MAE **improving 1.5%**. Here it moves +0.38% —
the wrong way — at a paired t of 1.69 on 7 degrees of freedom, six of eight
seeds worse. DE moves −0.60% at t = −1.01. Both are comfortably inside their own
single-seed nulls, which run 3.7–5.4% of MAE. **A one-seed read of this frame
could have reported anything up to a 4.4% FR gap with nothing changed at all**,
which is an order of magnitude more than either the claim or the measurement.

That null is the section's transferable result: it says a single-seed solar A/B
on this harness cannot resolve an effect of the size this issue is about. ABL-338
put its own noise floor at ~1.5% and read its 1.5% daylight gain as "not a
regression" for that reason. On the registered frame the floor is wider still.

### BE is the control the design provides for itself

BE excludes nothing, and its two arms predict **bit-identically at all eight
seeds** — every paired difference is exactly 0.00, on both axes. That is worth
more than a passing metric: it attests that these fits are deterministic given
(data, seed), without which the pairing above would not be valid, and it shows
the rule is a provable end-to-end no-op where the data is clean. Stating the
rule over countries rather than as an FR special case is what makes that
checkable.

### Why this frame and ABL-338's disagree

The two reads are not in contradiction; they are different experiments, and §1
already showed how far frame alone moves a night row count. Two differences are
large enough to matter, and they are separable:

1. **History.** ABL-338 fits FR on 27,228 training hours from 2023-01-01,
   including the flat ~234 MW-through-the-night block the issue quotes. The
   registered gate fits 2026-01-14 → 2026-07-11, where 113 contaminated hours
   sit among 11,648 night fit rows. Removing a sustained multi-month block and
   removing 1% of night rows are not the same intervention.

2. **The model has no way to say "night".** The registered gate fits the 25
   legacy columns. ABL-338 §1 measured what that costs: at every night hour all
   three radiation columns read 0.0 and both target lags read 0.0, so nothing in
   the vector distinguishes "0 W/m² because the sun is down" from "0 W/m² at a
   dark winter dawn", and the model's night output is an incidental country
   constant. ABL-338's own arms — the ones this rule was measured in — carried
   `sun_elevation_deg` and `is_night`. Removing impossible *targets* cannot move
   a level the model has no feature to represent.

**The second of those is testable, and I tested it.** Same eight seeds, same
frames, same rule, FR only, with `sun_elevation_deg` and `is_night` appended to
the gate's 25 columns (`--with-geometry`; exploratory, and its record says so):

| FR night mean prediction | control | night-fit | paired | seeds down | null max |
|---|---:|---:|---:|---:|---:|
| legacy 25 columns (registered) | 43.66 ± 10.06 | 43.33 ± 19.59 | −0.33 MW | 5/8 | 28.81 MW |
| + geometry (27 columns) | 58.39 ± 7.38 | 49.59 ± 19.87 | **−8.81 MW** | **7/8** | 20.40 MW |

**Partly confirmed, and that is the interesting part.** Give the model a way to
say "night" and the same rule moves the night level 27 times further and in a
consistent direction — 7 seeds of 8 down, against 5 of 8 and a third of a
megawatt without it. So the missing feature is real, and it is part of why the
registered frame is inert.

But it is not sufficient. −8.81 MW at t = −1.61 is still inside its own 20.40 MW
null, and 49.59 MW is not 0.05 MW. **The geometry features do not recover
ABL-338's collapse**, so the remaining difference has to sit in explanation 1 —
the fit history. ABL-338 removes a sustained multi-month night block from a
2023-onward fit; the registered window removes 113 scattered hours from six
months of 2026. Those are different interventions and only the first can drive a
night level to zero.

The daylight axis is unmoved by the feature either: +0.19% paired, 5/8 seeds,
inside a 5.19% null.

(DE was requested in the same probe as a near-clean comparator; it excludes 4
hours and carries no information about this mechanism. Whatever it returns does
not bear on the FR read above.)

### What this changes about the recommendation

The rule remains right on its own terms, and §3–§4 still hold: it is
principled, it is a no-op on clean data, and it does not threaten the gate. What
it is *not* is a measured improvement on the registered frame. The issue's two
headline numbers should not be restated as properties of this change — they are
properties of ABL-338's frame, and this pack should be read as replacing them
rather than confirming them.

Three consequences worth stating plainly:

1. **Land it as a correctness rule, not as a win.** Refusing to train on values
   the sun says are impossible is right whether or not it moves a metric, and it
   costs nothing measurable here. Any changelog entry that quotes 22.46 → 0.05
   or "daylight −1.5%" as an outcome of this change would be wrong.
2. **Nothing here rescues FR's night level.** On the registered configuration
   the serving clamp (ABL-337) is what holds that line, and it still is. A
   fit-side fix of the size ABL-338 saw needs the geometry features *and* a fit
   window that actually contains the contamination it is removing — neither of
   which this issue changes.
3. **The single-seed null is the reusable finding.** 3.7–5.4% of daylight MAE on
   this harness, from seed alone. Any future solar A/B on this frame quoting a
   gap smaller than that at one seed is quoting noise, and two are already in
   the record.

## 6. Caveats

- **Contamination.** ABL-67 is net-position-only; ABL-109/111 are load-only;
  ABL-71's known wrong-write modes are load and net position. None is *proven*
  to touch solar, so this is a provenance caveat, not a clean bill. The FR night
  contamination this issue addresses is itself undiagnosed on the ingest side and
  remains open — this change refuses to train on it, and does not fix it.
- ABL-188 constant-run screening found no suspect solar run in the window.
- One 30-day summer holdout. Out-of-sample by target timestamp, not year-round
  evidence, and July/August is when a night-hour rule has the least night to act
  on. §5's night band is 462 FR rows out of 1,950 for that reason.
- **§5's spread is eight seeds of one algorithm on one window.** Eight is enough
  to say the effect is inside the null and not enough to put a tight interval on
  it; the null itself is 28 pairs from those same eight. CatBoost only — the
  gate's algorithm — so none of it transfers to the XGBoost artifacts.
- §5 does not show the rule is harmless *in general*. It shows it is inert on
  this frame. The one place it demonstrably acts is the one the issue found it
  in, and that frame is not this one.
- The rule is **conservative by construction**: `is_night_hour` requires the sun
  below threshold for the whole hour, so shoulder contamination survives it. On
  2026-07-29, of the three FR hours the issue names (03:00, 20:00, 21:00 at
  ~195 MW) it removes only 21:00. It under-removes rather than over-removes,
  which is the right way round — zeroing an hour that really generated would be
  fabricating a number.
- **§4's gate read predates ABL-389.** That change — merged into main while this
  branch was open, and merged into this branch afterwards — reports a constant
  and an hour-of-day climatology reference beside the D-7 bar in both gate
  harnesses. Both §4 arms were fitted before it, so neither carries those
  columns. It does not disturb §4: ABL-389 adds *reported* comparators scored on
  their own intersection with the basis, and the gate basis, the D-7 bar and the
  9/9 verdict are unchanged. But on solar a D-7 bar is close to a formality, and
  the climatology is the reference that would actually test the challenger. A
  re-read of this scope carrying those columns is worth doing and is **not** in
  this pack; it is a follow-up, and it is a question about the solar gate rather
  than about this fit rule.
- Not a promotion recommendation. Promotion is a pre-registered gate read plus a
  Board decision; this is a fit-rule change in the evaluation harness and the
  shared feature module, and it reaches no serving path.

## 7. Reproducing

```
.venv\Scripts\python.exe scripts/evaluate_solar_retrain.py --scope abl376 \
    --replica-db C:\Code\able\data\energy_dashboard.db \
    --sidecar-db C:\Code\able\data\forecasts_local.db
```

The control is the same command with `--scope abl253` **and all three output
flags overridden**; run without them it will overwrite ABL-253's dispositioned
evidence.

The §5 seed spread, which builds each country's frames once and refits 16 times:

```
.venv\Scripts\python.exe scripts/abl376_night_seed_spread.py \
    --replica-db C:\Code\able\data\energy_dashboard.db
```

Roughly 4-5 minutes of frame building and 4-5 seconds per fit, per country.
Adding `--with-geometry` runs the §5 mechanism probe instead; it is exploratory,
it says so in its own record (`feature_set`, `is_registered_read`), and it is
not the registered read.
