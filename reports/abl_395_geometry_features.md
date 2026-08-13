# ABL-395 — the solar gate's feature list, and what the two geometry features actually do

**The list gap is real and is fixed. The defect it was thought to cause is not
measurably fixed by fixing it — and the 80.5% figure that motivated the issue is
not a stable quantity.**

Machine record: `reports/abl_395_geometry_feature_probe.json`.
Probe: `scripts/abl395_geometry_feature_probe.py`.
Guard: `tests/test_gate_feature_list_contract.py`, `tests/feature_list_manifest.json`
(`gate_harness` block).

---

## 1. What was wrong, and what changed

`RenewableFeatureBuilder` has emitted `sun_elevation_deg` and `is_night` for
`forecast_type='solar'` since ABL-338 (`wind_features._solar_geometry_features`).
`src/evaluation/solar_retrain.py:FEATURE_COLUMNS` never asked for them, so every
solar gate read from ABL-253 through ABL-381 fitted **25 features where an
ABL-338-current fit is 27**, and nothing said so: `to_vector` only raises on a
name the builder *cannot* build, and a list that asks for less is exactly right
about what it asks for.

ABL-394's guard did not cover this, because the gate harnesses never call
`get_feature_columns()` — they declare their own list. That is now closed:

- `FEATURE_COLUMNS` splats `solar_features.SOLAR_GEOMETRY_FEATURES` onto the end,
  so the list and the builder cannot name different columns.
- Both harness lists (solar 27, wind 24) are frozen in
  `tests/feature_list_manifest.json` under `gate_harness`, so changing either is
  a diff a reviewer sees.
- `tests/test_gate_feature_list_contract.py` asserts the constant matches the
  manifest, that the builder **produces** every declared name, that the written
  artifact declares the 27 it was fitted on, and that every
  `config.SUPPORTED_COUNTRIES` entry has a `solar_geometry` representative point
  — without one, `to_vector` raises and a tranche dies at its first fit row. All
  24 have one today.

**Not** the other half of ABL-338: the non-negativity constraint was measured and
*rejected* there (+15.8% Tweedie, +36.8% Poisson daylight MAE), and
`nonneg_objective=None` on every gate artifact records that correctly.

**`abl253` and `abl376` do not follow the constant.** They were read at 25;
`SCOPE_FEATURES` pins them there, the way `FIT_RULES` pins what a scope's fit may
see, and for the reason stated over that table. A scope that registers no feature
set gets the 27 — which is what unblocks the remaining ABL-316 tranches without
touching the table. The report and the JSON now name the set (`feature_set`,
`n_features`), because a 25-column and a 27-column artifact are otherwise
indistinguishable after the fact.

## 2. Protocol

One vintage frame per country, built at the 27-name superset and shared by every
fit, so the two arms differ **only** in the column list handed to CatBoost:

| arm | columns |
|---|---|
| `f25` | the list the harness declared through ABL-381 |
| `f27` | those 25 plus `sun_elevation_deg`, `is_night` |

The geometry columns are pure functions of `(country, hour)` and never NaN, so
`finite_training_rows` retains identical rows for both arms — asserted in the
probe, not assumed, because an arm scored on a different row set is not an A/B.
Both arms: **34,176 fit rows, 5,760 gate rows**, 23,674 of the fit rows carrying a
degraded `lag_1d` (69.3%, the serve-faithful schedule, identical for both).

Windows, source, schedule, bands and algorithm are ABL-348's registered ones —
fit 2026-01-14 → 2026-07-11, gate 2026-07-11 → 2026-08-10 (both exclusive),
`energy_generation`, CatBoost, primary bands 24-36h / 36-48h / 48-64h, out of
sample by target timestamp. Replica `C:\Code\able\data\energy_dashboard.db` at
**9,432,453,120 bytes** — the live replica, not the 3.0 GB partial snapshot —
opened read-only. No write of any kind was made to it.

**Two reads, because one seed is not a measurement.** ABL-385 measured the fleet
seed spread: the minimum readable relative gap between two solar fits is **15% at
one seed**. The movements here are ~1-8% relative, so a one-seed quote of them
would be reporting noise. So:

- **`reproduction`** — seed 42, the gate's own seed, against the published
  ABL-381 read.
- **`sweep`** — ABL-376's eight registered seeds `(101, 103, 107, 109, 113, 127,
  131, 137)`, reused verbatim: they were frozen before that issue's first fit and
  are disjoint from 42, so nothing here was selected on them. The comparison is
  taken **paired within each seed**; beside it is the **unpaired null**, every
  control-vs-control seed pair, which is what a single-seed gap looks like when
  nothing changed at all.

**This is not a gate read and dispositions nothing.** It writes under no
registered scope's paths; ABL-381's PASS 6/6 stands as read, on the artifacts it
was read on.

## 3. The control reproduces ABL-381 exactly

At seed 42 the `f25` arm returns ABL-381's §4 and §2 numbers to the decimal —
night-negative counts, mean night prediction, minimum prediction, and all six
gate-band WAPEs. That is what makes the rest of this readable.

| country | arm | features declared by artifact | night rows | night predicted negative | mean pred at night | min pred | 24-36h | 36-48h | 48-64h |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BG | `f25` | 25 | 690 | 113/690 = **16.38%** | 224.78 MW | -33.36 MW | 18.89% | 18.6% | 20.03% |
| BG | `f27` | 27 | 690 | 159/690 = **23.04%** | 218.8 MW | -23.12 MW | 19.95% | 19.75% | 21.4% |
| CH | `f25` | 25 | 640 | 515/640 = **80.47%** | -5.48 MW | -31.97 MW | 8.16% | 8.01% | 8.39% |
| CH | `f27` | 27 | 640 | 410/640 = **64.06%** | -1.91 MW | -21.7 MW | 7.78% | 7.75% | 8.32% |

`features declared by artifact` is read back off the written `model.joblib`, not
counted by hand — with `n_features_produced_by_builder = 27` for both countries,
from a `to_vector` call that raises on any name the builder cannot build. That is
the ABL-394-shaped evidence the issue asked for: declared **and** produced.

**Read at one seed, this table says the fix works on CH (80.47% → 64.06%) and
backfires on BG (16.38% → 23.04%). Both readings are wrong.**

## 4. The eight-seed read

| country | quantity | f25 (mean ± sd) | f27 (mean ± sd) | paired change | t | seeds in that direction | sign p | single-seed null (max) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BG | night rows predicted negative (%) | 20.090 ± 4.109 | 21.630 ± 2.218 | +1.540 (+7.67%) | +0.81 | 5/8 up | 0.7266 | 14.060 (70.0%) |
| BG | mean prediction at night (MW) | 223.044 ± 4.164 | 223.579 ± 2.107 | +0.535 (+0.24%) | +0.30 | 5/8 up | 0.7266 | 14.350 (6.4%) |
| BG | WAPE 24-36h (%) | 19.407 ± 0.471 | 19.869 ± 0.502 | +0.461 (+2.38%) | +1.50 | 6/8 up | 0.2891 | 1.490 (7.7%) |
| BG | WAPE 36-48h (%) | 19.185 ± 0.460 | 19.625 ± 0.506 | +0.440 (+2.29%) | +1.47 | 6/8 up | 0.2891 | 1.480 (7.7%) |
| BG | WAPE 48-64h (%) | 20.675 ± 0.425 | 20.910 ± 0.476 | +0.235 (+1.14%) | +0.89 | 6/8 up | 0.2891 | 1.480 (7.2%) |
| BG | daylight WAPE 24-36h (%) | 19.427 ± 0.469 | 19.869 ± 0.539 | +0.441 (+2.27%) | +1.39 | 6/8 up | 0.2891 | 1.530 (7.9%) |
| BG | daylight WAPE 36-48h (%) | 19.170 ± 0.460 | 19.586 ± 0.550 | +0.416 (+2.17%) | +1.35 | 6/8 up | 0.2891 | 1.510 (7.9%) |
| BG | daylight WAPE 48-64h (%) | 20.826 ± 0.451 | 21.032 ± 0.515 | +0.206 (+0.99%) | +0.74 | 6/8 up | 0.2891 | 1.570 (7.5%) |
| CH | night rows predicted negative (%) | 77.051 ± 10.107 | 73.204 ± 8.156 | -3.847 (-4.99%) | -0.90 | 4/8 down | 1.0000 | 27.340 (35.5%) |
| CH | mean prediction at night (MW) | -5.585 ± 2.198 | -3.704 ± 2.812 | +1.881 (+33.68%) | +1.57 | 6/8 up | 0.2891 | 6.840 (122.5%) |
| CH | WAPE 24-36h (%) | 8.346 ± 0.154 | 8.220 ± 0.158 | -0.126 (-1.51%) | -2.45 | 6/8 down | 0.2891 | 0.460 (5.5%) |
| CH | WAPE 36-48h (%) | 8.299 ± 0.155 | 8.059 ± 0.149 | **-0.240 (-2.89%)** | **-4.57** | **8/8 down** | **0.0078** | 0.470 (5.7%) |
| CH | WAPE 48-64h (%) | 8.704 ± 0.160 | 8.474 ± 0.191 | **-0.230 (-2.64%)** | **-3.01** | **8/8 down** | **0.0078** | 0.450 (5.2%) |
| CH | daylight WAPE 24-36h (%) | 8.188 ± 0.160 | 8.080 ± 0.154 | -0.108 (-1.31%) | -1.93 | 6/8 down | 0.2891 | 0.500 (6.1%) |
| CH | daylight WAPE 36-48h (%) | 8.144 ± 0.144 | 7.914 ± 0.141 | **-0.230 (-2.82%)** | **-4.64** | **8/8 down** | **0.0078** | 0.420 (5.2%) |
| CH | daylight WAPE 48-64h (%) | 8.608 ± 0.161 | 8.384 ± 0.184 | **-0.224 (-2.60%)** | **-2.89** | **8/8 down** | **0.0078** | 0.480 (5.6%) |

The two right-hand columns answer different questions and both are needed. The
**sign test** and **t** are on the paired difference and ask *is there an effect*.
The **single-seed null** is unpaired and asks *could this have been quoted from
one seed* — comparing a paired mean to an unpaired maximum is a much stricter
test than the design requires, and no row here clears it. That is a statement
about one-seed reads, not about the effect.

## 5. What that says

**5a. The 80.5% was one draw, not a measurement.** CH's control night-negative
rate over eight seeds is **77.05% ± 10.11**, and the largest gap between two
*control* fits — same data, same columns, same hyperparameters, one integer
apart — is **27.34 percentage points**. The published 80.47% sits inside that
spread, and so does the 64.06% the treatment returned at seed 42. The paired
change is −3.85pp at 4/8 seeds, sign p = 1.0: **not distinguishable from seed
noise**. The honest answer to "80.5% is the before, report the after" is that
*neither number is a stable quantity*, and the same is true of BG's 16.38% (null
14.06pp on a control mean of 20.09%).

That is not a reason to keep the list at 25. It is a reason to stop quoting a
one-seed night-hour fraction as a defect measurement — including in ABL-381 §4,
where it was quoted, and including in this issue's own premise.

**5b. On CH the geometry pair is a small, clear, daylight-safe accuracy gain.**
−0.23 to −0.24pp WAPE on the two longer bands, **8/8 seeds in the same
direction**, t = −4.57 and −3.01, sign p = 0.0078; the daylight-only re-score
moves identically (−0.230, −0.224, both 8/8), so this is not the night rows
flattering a denominator. Small in absolute terms (~2.7% relative) and on one
30-day summer holdout, but it is the direction ABL-338 predicted and it holds
across every seed.

**5c. On BG it goes the other way, and the mechanism is BG's data, not the
feature.** +0.44pp at 6/8 seeds, sign p = 0.29 — weak evidence, not significant,
and I am not claiming a BG regression. But the direction is consistent across
three bands and matches a specific prediction: **BG's night actuals are not
night.** ABL-381 §5 measured 76-85% of BG's night hours carrying 152-246 MW, max
1,097 MW, agreeing to the decimal across both source tables — an ingest or
ENTSO-E feed defect, escalated there and still open. `is_night` tells the model
the sun is down on hours where the target says 225 MW; CH's night actuals are
exactly **0.00 MW**. So the feature encodes a physical prior, and it is worth
what the actuals' respect for that physics is worth. That is a testable
hypothesis, not a demonstrated cause — §7 names the read that would settle it.

## 6. Does the change still belong? Yes

Not on the strength of §5b alone, which is one country on one holdout. On three
grounds:

1. **Consistency between the two fit paths is a correctness property
   independent of effect size.** `get_feature_columns('solar')` already declares
   both names (31 columns, `feature_list_manifest.json`), so `scripts/train.py`
   and the gate harness were fitting different models for the same pair and
   calling both "the solar challenger".
2. **The one country whose actuals respect the physics improves, consistently.**
3. **It costs nothing that is measurable.** BG's movement does not reach
   significance, and no dispositioned scope moves at all.

## 7. Recommendations

- **The remaining 33 ABL-316 solar tranches are unblocked and should be fitted at
  27.** No new scope has to touch `SCOPE_FEATURES`.
- **Screen a country's night floor before reading its solar gate.** The
  discriminator in §5c is `night_mask` against the actuals, which
  `scripts/abl381_night_floor_probe.py` already measures. ABL-381 §8 recommended
  screening all 35 remaining countries for the same floor; this is a second,
  independent reason to do it before rather than after the fits.
- **Stop quoting single-seed night-hour fractions.** They carry a ~27pp
  single-seed null on CH. Anything that reads as a defect measurement on that axis
  needs the seed spread ABL-385 registered.
- **The read that would settle §5c is a 2×2**: geometry × ABL-376's
  `exclude_impossible_night` fit rule, on BG and CH, on the same eight seeds.
  ABL-376's own mechanism probe found that geometry made the exclusion rule **27×
  more effective** on FR's night level, so the night axis may move only when both
  are present — which is exactly the arm this run did not have (it ran the
  `abl253`/default rule, i.e. no exclusion). Filed rather than run here: it is a
  different question from the feature list. **BG is the informative cell** —
  excluding its impossible night rows removes the very hours that make `is_night`
  a lie.
- **Refit/re-read of the two 25-feature scopes is ABL-401**, not this issue.

## 8. Caveats

- One 30-day summer holdout, two countries, eight seeds. Out of sample by target
  timestamp; gate targets were never fitted. Not year-round evidence.
- 69.3% of fit rows carry a degraded `lag_1d`. That is the serve-faithful
  schedule, identical in both arms, and the dominant feature-quality limit on
  these fits.
- Contamination: ABL-67 is net-position-only, ABL-109/111 are load-only, ABL-71's
  known wrong-write modes are load and net position. None touches solar in this
  window. **BG's night floor (ABL-381 §5) is a solar data defect none of those
  four covers and it is live in this window** — it is the subject of §5c, not a
  confounder that was overlooked.
- No promotion, no serving-registry change, no ingest change, no dashboard
  change, no replica write, no sidecar write.
