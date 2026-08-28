# ABL-583 — ABL-316 ship set: CH solar at 27 features

Forecasting Scientist. One production artifact fitted through the graded gate-harness path,
serving-verified end to end, reproducibility proved by prediction equality, and staged for the
deploy. **Nothing here is scored, graded or promoted.** Membership is the CEO's; this pack is the
evidence behind the one fit that membership named.

Companion machine records, all committed and all outside the `experiments/*/results.json` glob
that `.gitignore` swallows (ABL-440, still open):

| record | what it holds |
|---|---|
| `reports/abl_583_ship_set_training.json` | fit provenance, audits, artifact path and sha256 |
| `reports/abl_583_reproducibility.json` | the refit-and-compare-predictions check |
| `reports/abl_583_contamination_screens.json` | ABL-332 / ABL-200 / ABL-188, the night floor, the ABL-439 vintage screen |
| `reports/abl_583_ch_basis_seam.json` | where the two source tables diverge, resolved to months |
| `reports/abl_583_ch_night_probe.json` | the night question, either side of the ABL-337 serving clamp |
| `reports/abl_583_serving_verification.json` | what `forecast_daily.py` actually served, from its own output |
| `reports/abl_583_weather_archive_start.json` | the archive bound on the fit window, and the one-vintage check |
| `reports/abl_583_scope_value_check.json` | the readmission premise, resolved by value across four revisions |

---

## 1. What authorises this fit, verified rather than accepted

The Board approved a **rule** alongside the `ship8` roster on 2026-08-22 (ABL-316 ledger 14.6,
restated in 15.1): a held pair that later satisfies the same rule joins the shipping set without a
new Board card. CH `solar` was pair 8 of that roster, was **withdrawn** by CEO ruling on 2026-08-27
because tranche 1b graded it at the legacy 25-name solar list while ABL-395 had moved
`solar_retrain.FEATURE_COLUMNS` to 27, and **rejoined** the same day on ABL-581's fresh read at 27
under a new pre-registered scope. The disposition is the CEO's and is **not re-derived here**.

What is checked here is the *premise* of that disposition, because it is the premise this artifact
inherits. Three things, from the machine record and the source tree rather than from the report:

**The read graded the list this artifact is fitted at.** `experiments/ABL348/results_abl581_ch_solar_f27.json`:

```
meta.scope                                abl581-ch-solar-f27
meta.n_features                           27
meta.feature_set                          legacy25+geometry
meta.feature_set_is_registered_for_scope  False    (inherited the default)
meta.registered_source                    energy_generation
meta.source_is_scope_registered           True
meta.feature_columns == src.evaluation.solar_retrain.FEATURE_COLUMNS  ->  True
```

Element for element, not merely by count; the two added names are `sun_elevation_deg` and
`is_night`, exactly ABL-395's move.
`tests/test_abl580_ship_set_batches.py::test_ch_solar_rejoined_at_the_current_27_name_list` holds
this, so a later move of `FEATURE_COLUMNS` to 28 fails the suite rather than silently re-basing
this artifact.

**`feature_set_is_registered_for_scope: False` is the correct configuration, not a gap.**
`SCOPE_FEATURES` is one of the tables `check_registration_tables` deliberately does not enforce,
because inheriting the current list through `DEFAULT_SCOPE_FEATURES` is the intended path for a new
scope (CLAUDE.md, gate-harness section).

This was first checked here by hashing `ast.dump` of each constant's value node across the
revisions. **That check was two-thirds vacuous and the draft table is withdrawn.** The three
constants do not all live where the phrasing implied — `SCOPE_FEATURES`,
`LEGACY_FEATURE_COLUMNS` and `DEFAULT_SCOPE_FEATURES` are in `scripts/evaluate_solar_retrain.py`,
while the list they resolve against is `src/evaluation/solar_retrain.FEATURE_COLUMNS` — and two of
the three are *derived expressions*, not literals:

```
LEGACY_FEATURE_COLUMNS = tuple((c for c in FEATURE_COLUMNS if c not in SOLAR_GEOMETRY_FEATURES))
DEFAULT_SCOPE_FEATURES = FEATURE_COLUMNS
```

`DEFAULT_SCOPE_FEATURES`'s value node is a bare `Name`. Its AST dump is byte-identical at every
revision **whatever `FEATURE_COLUMNS` holds** — which is precisely the move ABL-395 made and
precisely the move that withdrew CH. An AST hash on those two rows would have read "identical"
across the very change it was there to detect.

And `FEATURE_COLUMNS` itself is only *partly* a literal — it ends `*SOLAR_GEOMETRY_FEATURES`,
splatting a tuple out of a third module — so it carries the same blind spot one level up, live on
today's tree. `scripts/abl583_scope_value_check.py --demonstrate-blind-spot` proves this rather
than arguing it, by re-resolving the chain **in memory** with one name appended upstream (nothing
is written to the tree):

| constant | n | AST hash | | value hash | |
|---|---|---|---|---|---|
| `SOLAR_GEOMETRY_FEATURES` | 2 → 3 | `c8d1380d8913777f` | moved | `6924258c061b9d9f` | moved |
| `FEATURE_COLUMNS` | **27 → 28** | `51ea29cc1fefb9f1` | **same** | `5f79bff3e0f262f2` | moved |
| `LEGACY_FEATURE_COLUMNS` | 25 → 25 | `c68a5f7d51d80bbb` | same | `464fdc3eb7e719e9` | same |
| `DEFAULT_SCOPE_FEATURES` | **27 → 28** | `a1d92331f4ddacb8` | **same** | `5f79bff3e0f262f2` | moved |
| `SCOPE_FEATURES` | 3 → 3 | `2cfe6f1126bd2b02` | same | `644aaa700372022e` | same |

`FEATURE_COLUMNS` goes to **28 names** — the exact scenario this pack says must fail the suite
rather than silently re-base the artifact — and the AST instrument reports "identical" for it and
for `DEFAULT_SCOPE_FEATURES`. Those three unchanged AST hashes are, to the digit, three of the
values the withdrawn table published as its evidence: the draft's numbers were right and were
measuring the wrong thing.

Replaced by a **resolved-value** check: at each revision the four constants are lifted out by AST
and evaluated in dependency order, so what is compared is the tuple each name actually holds.
sha256 prefixes of the canonicalised value:

| constant | defined in | value hash | identical across 82e3108 / 49ab9e9 / `origin/main` / HEAD |
|---|---|---|:---:|
| `SOLAR_GEOMETRY_FEATURES` | `src/solar_features.py` | `6924258c061b9d9f` | **yes** |
| `FEATURE_COLUMNS` (27) | `src/evaluation/solar_retrain.py` | `5f79bff3e0f262f2` | **yes** |
| `LEGACY_FEATURE_COLUMNS` (25) | `scripts/evaluate_solar_retrain.py` | `464fdc3eb7e719e9` | **yes** |
| `DEFAULT_SCOPE_FEATURES` | `scripts/evaluate_solar_retrain.py` | `5f79bff3e0f262f2` | **yes** |
| `SCOPE_FEATURES` | `scripts/evaluate_solar_retrain.py` | `644aaa700372022e` | **yes** |

`DEFAULT_SCOPE_FEATURES` and `FEATURE_COLUMNS` carry the *same* hash, which is the "inherits the
current 27" claim as a measurement rather than as a reading of the source. On `origin/main` the
scope table holds exactly `abl253`, `abl376` and `abl316-t1b`, all three bound to the legacy 25,
and `abl581-ch-solar-f27` is **absent**. So tranche 1b's published PASS 6/6 is untouched by the
readmission, and the new scope resolves to the current 27. The CEO's independent check reproduces
— on a stronger instrument than the one this section first used.

**The read is complete on its own terms.** All three bands, from `gate_cells`:

| band | n | registered minimum n | `enough_pairs` | gate | grade | `failed` | `not_readable` |
|---|---:|---:|:---:|:---:|:---:|---|---|
| 24-36h | 720 | 684 | true | PASS | **A** | `[]` | `[]` |
| 36-48h | 720 | 684 | true | PASS | **A** | `[]` | `[]` |
| 48-64h | 510 | 456 | true | PASS | **A** | `[]` | `[]` |

`bar_weaker_than_a_flat_line: false` on all three, `causal_levelling: trailing_28d`,
`g23_readability: floored`, floor 10.648% at k=1. `enough_pairs` is read beside every grade rather
than inferred from the letter -- it nests under `gate`, and a flat lookup for it passes vacuously.

## 2. The check the issue asked to be verified rather than assumed

ABL-580's lesson was that a *restated* constant is how a trainer comes to disagree with the code
that graded a pair: `ALGORITHM = "catboost"` as a module constant would have shipped an NL
`wind_offshore` model no gate ever read, because the wind harness fits offshore with xgboost. The
issue asks for the same failure mode to be closed here, in its next disguise.

It is closed by **removing** a constant rather than by adding one. The CH row's whole content as a
readmission is that it pins nothing:

```
columns_for("CH", "solar")   ->  FEATURE_COLUMNS_BY_TYPE["solar"]
                             ==  src.evaluation.solar_retrain.FEATURE_COLUMNS      (27 names)
algorithm_for("solar")       ->  ALGORITHM_BY_TYPE["solar"]
                             ==  src.evaluation.solar_retrain.ALGORITHM            ("catboost")
```

Both are imported at the top of `scripts/abl525_train_ship_set.py`; neither is restated anywhere in
it. A later move in either arrives at this trainer without an edit, and
`test_the_algorithm_table_is_the_harnesses_own` asserts that as identity of values rather than as a
literal.

**One row per pair, and that is the load-bearing decision.** The obvious way to record a
readmission is to keep the withdrawn row for the history and add a shipping one. That would have
been the ABL-580 defect exactly: `columns_for` matches on `(country, forecast_type)` and returns
the **first** hit, so two CH rows would have decided the shipped artifact's feature list by source
order in the table -- the withdrawn row's legacy-25 pin, silently, with every test still green.
The history goes in an `admission_history` field on the single row instead, which reaches the
committed machine record rather than stopping at a comment, and `test_a_pair_appears_once` is the
guard.

A consequence worth naming rather than leaving to be found: with CH readmitted, **no row in
`SHIP_SET` is held and no row pins a feature list**, so
`test_only_a_withdrawn_row_pins_its_own_feature_list` now passes vacuously.
`test_no_row_pins_a_list_while_none_is_held` asserts that vacuity so it is a stated fact rather
than a silent one, and the pin machinery stays because the ship set shrinks as well as grows --
CH is the worked example of both directions.

> **Addendum, 2026-08-28 (ABL-602).** The paragraph above describes the table as it stood on
> 2026-08-27 and is left as written. It no longer describes today's table: ABL-602 withdrew `HU`
> `wind_onshore` after fitting it, so exactly one row is now held and the vacuity is gone. The
> assertion moved with it -- `test_no_row_pins_a_list_while_none_is_held` is now
> `test_the_only_hold_is_a_disposition_and_no_row_pins_a_list`. **No row pins a feature list**,
> which is the half of the claim that still holds and the half ABL-525 item 2 depends on.

## 3. The fit

Source `energy_generation` (ABL-321/ABL-348 registered, and `meta.source_is_scope_registered` is
`true` on ABL-581's read), 8 pre-registered vintages per target, 14-day lookback for the point lag
and the 168-hour anchors, catboost with `config.get_default_params("catboost")`, no tuning and no
per-country fork of anything.

### 3.1 Why not `scripts/train.py`, measured for solar rather than cited

ABL-583 item 1 says the fit goes through the graded gate-harness path and not `train.py`, because
`train.py`'s feature list carries holiday names the serve-faithful builder cannot build. Measured
on this tree:

```
train.py         features.get_feature_columns("solar")           -> 31 names
gate harness     src.evaluation.solar_retrain.FEATURE_COLUMNS    -> 27 names
in train.py and outside the gate list:
    days_from_holiday, days_to_holiday, is_bridge_day, is_holiday
```

Exactly the four holiday columns and nothing else. `Forecaster.predict_d2` routes `solar` to
`_predict_d2_serve_faithful`, which calls `to_vector(row, artifact.feature_columns)`, and
`to_vector` raises `KeyError` on a column the builder does not produce -- the builder produces no
holiday column (`wind_features.py:179`). So a `train.py` CH solar artifact would load clean, then
raise on its first serving row, `forecast_daily.py` would book a failed result, and CH solar would
serve zero rows. The end-to-end run in section 6 is the direct counterfactual.

### 3.2 The window caveat — measured, not asserted

`FIT_START` / `FIT_END` are the ship set's module constants, unchanged: **2026-01-11 → 2026-08-22**
(223 days).

`FIT_END` stays at the Board's decision date even though the replica now carries actuals past it,
and the reason is stronger with a third batch than it was with two: `abl525_repro_check.py` refits
through `fit_one` on these same module constants, so moving them would make the ABL-525 seven and
the ABL-580 three report a prediction difference that is a window change rather than a drift, and
every artifact in the ship set stays on one window so the deploy is a homogeneous batch.

**ABL-581's grades are in-sample for this artifact.** ABL-583 item 5 asks this to be said plainly,
so it is said with the arithmetic rather than as a warning. Both windows are read out of
`experiments/ABL348/results_abl581_ch_solar_f27.json` — `meta.fit_window` and `meta.gate_window` —
and compared against the trainer's own module constants:

| window | span | days |
|---|---|---:|
| ABL-581's gate-harness **fit** window | 2026-01-14 → 2026-07-11 | 178 |
| ABL-581's **scoring** window (what the grades are computed on) | 2026-07-11 → 2026-08-10 | 30 |
| **this artifact's fit window** | 2026-01-11 → 2026-08-22 | **223** |

The scoring window opens **181 days after** this fit begins and closes **12 days before** it ends,
so it lies wholly inside the training set. ABL-581's read was properly leak-free *on its own terms*
— it scored 30 days it had not fitted — but this artifact is fitted on 45 days more than the gate
harness was, and those 45 days include every row the grades were computed on. **Every A in section 1
is therefore in-sample for the model being shipped, and none of them is this artifact's measured
skill.** This pack scores nothing and grades nothing; the gate read is the authorisation, not the
performance claim.

Worth carrying into that reading: even taken at face value, the A grades rest on which reference is
read beside them. From the same three cells —

| band | skill vs `seasonal_naive` | vs `constant_causal_28d` | vs **`climatology_causal_28d`** |
|---|---:|---:|---:|
| 24–36h | 38.59% | 92.22% | **23.44%** |
| 36–48h | 38.83% | 92.25% | **24.06%** |
| 48–64h | 33.59% | 89.72% | **15.41%** |

The 92% against a causal constant is the number a flat line loses by and certifies nothing for
solar; hour-of-day climatology is the honest null, and against it CH is **+15 to +24pp**. Positive,
real, and a good deal smaller than the headline — and, per the paragraph above, in-sample here
regardless. `bar_weaker_than_a_flat_line` is `false` on all three and the G2/G3 readability floor is
10.648% at k=1, both as ABL-581 published them.

The window is bounded by weather and not by actuals: `energy_generation` reaches back to 2021-01-01
for CH, but a serve-faithful row also needs the weather *forecast* archive, and `weather_data` with
`data_quality='forecast'` starts far later. That start is re-measured for CH on this run's replica
in `reports/abl_583_weather_archive_start.json` rather than inherited from the CZ/RO/NL measurement
the trainer's docstring carries — and it reproduces exactly:

```
CH   first forecast target  2026-01-11 00:00     first run_time  2026-01-11 18:00     70,896 rows
CZ / RO / NL                2026-01-11 00:00                     2026-01-11 18:00
```

So `FIT_START` is not a choice; **2026-01-11 is the first day a serve-faithful CH row can be built
at all**, and the 223-day window is the widest honest one. The archive is common to all four
countries, so CH inherits the same bound as ABL-580's three rather than a country-specific one.

### 3.3 One data vintage across the batch, established by maxima and not by a zero count

A sanctioned replica refresh ran *during* this heartbeat — the file went from 10,632,605,696 to
10,664,824,832 bytes between ABL-580's fit and this one. Since the ship set deploys as one
homogeneous batch, that raises a real question: were CH and ABL-580's three fitted on the same data?

They were. `max(fetched_at)` on `energy_generation` is **2026-08-27 14:27:33** and
`max(created_at)` on `weather_data` is **2026-08-26 15:00:54**, both of which **precede** ABL-580's
fit at 14:53:09Z. No row in either table — inside the fit window or anywhere — was written after
that fit. The refresh rewrote pages without moving the vintage these fits read, so **no refit of
ABL-580's artifacts is implied.**

Worth being explicit about the instrument, because the weaker version of this check is available
and looks the same: `rows_in_fit_window_revised_after_abl580_fit` is 0 for all four countries, but a
zero count alone would *also* be produced by a cut placed past the end of the data, which proves
nothing. The maxima are what carry the claim. Two commits on this branch guard the two places a
concurrent sync would otherwise have misreported.

## 4. Contamination screens — over the window actually fitted on

ABL-583 item 4 asks for these over the window this artifact was **fitted** on, not the one the gate
read. Everything below is over **2026-01-11 → 2026-08-22**, 5,352 hourly CH solar rows on
`energy_generation` — strictly wider than ABL-581's 178-day fit plus 30-day score.

**The three named contamination issues do not touch this pair.** ABL-71 (prod ingest stale) and
ABL-67 (fabricated rows) are both `net_position`; ABL-111 / ABL-109 are zero-as-missing actual
**load** rows. None is a solar-generation issue and none intersects this series. Said explicitly
because the operating brief requires the window to be dispositioned against all three, not because
any of them was a live candidate.

| screen | result over the fit window |
|---|---|
| **ABL-332** sub-hourly aggregation | 5,352 raw rows, **1** distinct minute mark — natively hourly. `rows_the_pre_abl332_builder_would_have_discarded: 0` |
| **ABL-188** constant-run exclusion | **0 rows nulled.** The 432 rows the fit audit drops are missing features or actuals, not guard exclusions |
| **ABL-200** cross-table zero disproof | **`rule_can_fire: false`** — vacuous by construction, see below |
| **ABL-439** fit-to-gate source ratio | fit **1.0424**, gate **1.0000**, discontinuity **0.0424** against a 0.02 threshold — **basis-INCONSISTENT** |

Two of those four need more than a row.

**ABL-200 is vacuous by construction, and that is not a pass.** The rule is wired behind
`if source != RENEWABLE_ZERO_DISPROOF_SOURCE`, and that source *is* `energy_generation`, which is
what this pair reads. So the screen cannot fire — this is already the disproving side. Recorded as
`rule_can_fire: false` rather than as a clean result, because a reader who sees "no zeros disproved"
and infers "no suspect zeros" would have it backwards.

**ABL-332 is verified in the path, not just in the data.** `load_renewable_type_data` calls
`aggregate_renewable_to_hourly`, then `RenewableFeatureBuilder._assert_hourly` raises on an off-hour
index. The same builder object fits and serves, so this is one frame checked twice rather than two
frames that happen to agree — which is what section 6 then exercises end to end.

### 4.1 The basis discontinuity is one date, and it is not the fit/gate boundary

`basis-INCONSISTENT` is the one screen that did not come back clean, so it is resolved rather than
reported. Splitting the two source tables by month (`reports/abl_583_ch_basis_seam.json`):

| month | night hours > 1 MW | night mean | daylight mean | `energy_generation` / `renewable_generation` |
|---|---:|---:|---:|---:|
| 2026-01 | 6.1% | 0.21 MW | 292 MW | 0.721 |
| 2026-02 | 25.3% | 0.74 MW | 555 MW | 1.105 |
| 2026-03 | 59.6% | 1.64 MW | 1,012 MW | 1.172 |
| 2026-04 | 89.0% | 2.49 MW | 1,449 MW | 1.198 |
| 2026-05 | 98.5% | 2.93 MW | 1,526 MW | 0.958 |
| **2026-06** | **0.0%** | **0.00 MW** | 1,665 MW | **1.000** |
| 2026-07 | 0.0% | 0.00 MW | 1,835 MW | 1.000 |
| 2026-08 | 0.0% | 0.00 MW | 1,601 MW | 1.000 |

One event on one date. **From 2026-06-01 the two tables are the same series** and the actuals carry a
hard zero at night; before it they are different series and the actuals carry a small positive night
floor. So the ABL-439 fit-window ratio of 1.0424 is a **blend of two regimes, not a level**, and the
gate-window 1.0000 is not a change — it is the post-seam regime the fit window already contains.

The seam is **2026-06-01**; the fit/gate boundary is **2026-07-11**. They are not the same date, and
conflating them is the easy misreading of a `basis-INCONSISTENT` verdict.

Two consequences, in opposite directions:

- **ABL-581's read is unaffected.** It fits and scores on the registered `energy_generation` at both
  ends, so the cross-table divergence is not in its path at all, and its gate window lies wholly
  post-seam.
- **This artifact is fitted across the seam and serves into the post-seam regime.** That is a
  train/serve mismatch in the *actuals*, and it is a live candidate explanation for the night level
  section 6 measures. Reported, not fixed: `FIT_START`/`FIT_END` are the ship set's shared module
  constants and moving them for one country would fork the batch.

**Magnitude, so this is not over-read.** Pre-seam night means are 0.21–2.93 MW against daylight means
of 292–1,526 MW, and `wape_floor_pct_if_clamped` over the whole fit window is **0.054%**. This is a
basis change, not a BG-scale contamination — the comparison in the record is BG at **6.37%** of total
energy booked at night against CH's **0.04%**, two orders of magnitude apart.

**One thing left unmeasured, named rather than implied.** This pair's gate window is 100%
first-publication, so its gate-side revision is expected-small but **not yet measured**. That is true
of every pair in this gate and is not a blocker; it is recorded so a later reader does not mistake an
unmeasured quantity for a measured-zero one.

## 5. Reproducibility -- by prediction equality, not by artifact hash

`Forecaster.save` stamps `saved_at`, so byte-identical fits give different digests and a hash
comparison reports drift that is not there (ABL-525 item 7, restated as ABL-583 item 3). The pair
was refitted independently into a scratch directory with the same function, window and seed; both
artifacts were loaded through `Forecaster.load` -- the entry point `forecast_daily.py` uses, so this
compares two things that were actually deserialised -- and both predicted on **one shared feature
matrix**, so the comparison cannot hide a builder difference behind a model difference.

One thing this run added to the check, because this run happened alongside a replica refresh: the
refit reads the replica live, so the two arms are only comparable if the replica did not move
between the original fit and the re-check. `able-db-sync` replaces every non-weather table inside a
single transaction, so a sync landing in between would report a **data change as a drift**. The
record now carries `replica.bytes_at_original_fit`, `replica.bytes_now` and
`replica.unchanged_since_original_fit` on both sides rather than assuming it away.

Result, on 168 probe rows from 2026-08-01:

| quantity | value |
|---|---|
| `max_abs_prediction_difference` | **0.0** |
| tolerance | 1e-12 |
| `identical_within_tolerance` | **true** |
| `bit_identical` | **true** |
| `feature_columns_match` / `training_source_match` | true / true (27 names, `energy_generation`) |
| replica unchanged across both arms | **true** (10,664,824,832 bytes on both reads) |
| artifact sha256, original vs refit | `7bcfecd5…` vs `bf8a860b…` — **differ** |

Not merely within tolerance: **bit-identical**, a max difference of exactly zero. And the two
sha256s differ, which is the point of doing it this way — a hash comparison on this same pair of
correct, bit-identical-predicting artifacts would have reported drift. `every_artifact_sha256_differed:
true` is recorded as the positive control for that.

So the recipe is as good as the bytes: the committed `scripts/abl525_train_ship_set.py` regenerates
this artifact's predictions exactly, and whoever deploys it does not have to trust the file it is
handed.

## 6. Serving, verified end to end

`scripts/forecast_daily.py --countries CH --types solar --horizons 1,2`, against a **scratch
sidecar** — `FORECAST_OUTPUT_DB` pointed at a run-scoped temp file, so nothing was written to the
replica or to the real sidecar:

```
Total: 2, Success: 2, Empty: 0, Unreported: 0, Skipped: 0, Failed: 0
Saved 48 forecasts to database
Execution time: 0.5s
```

**48 rows** = 1 pair × 2 horizons (D+1, D+2) × 24 h, targets **2026-08-28 → 2026-08-29**, horizons
3–50 h. **0 failed, 0 empty, 0 unreported, 0 skipped.** Two log lines carry the load-bearing part:

```
Model loaded from C:\Code\able\wt-abl583\models\CH\solar\model.joblib (algorithm: catboost)
Loaded 517 hourly solar records for CH from energy_generation
```

The artifact loaded with the estimator it was fitted with, and its own `training_source` drove the
serve-side read — ABL-331 working end to end on this pair. **No `KeyError` anywhere**, which is the
direct counterfactual to section 3.1: a `scripts/train.py` artifact carrying the four holiday columns
would have loaded just as cleanly and then raised on its first serving row. That is the whole reason
item 1 sends this fit through the gate-harness path, and it is now demonstrated rather than argued.

`reports/abl_583_serving_verification.json` is built from the run's own output tables rather than
from this log.

### 6.1 The night question, on the served series

ABL-583 asks for four specific quantities on the served CH series after the clamp. All four, from
`forecast_clamp_log` and the 48 served rows:

| asked | measured |
|---|---|
| `hours_zeroed` | **16 of 16** night hours |
| `hours_raised_floor` | **1** — and for CH that counter is *daylight-only*, see below |
| any served row negative? | **No.** 48 rows, min **0.0 MW**, 17 exact zeros (16 night + the 1 raised daylight hour) |
| pre-clamp night mean vs night actual | **−2.013 MW** predicted against **0.0000 MW** actual |

The night actual is not a rounding of something small: over the trailing 14 days CH books exactly
**0.0000 MW** across all 98 night hours, and post-seam (§4.1) the fit-window night actual is a hard
zero too. So the comparison is against a true zero.

**CH is a different case from CZ and RO, and the ABL-580 reading does not carry over.** ABL-580 found
those two carrying a *positive* night floor and argued it reads as a global ~2–3% level floor rather
than a night defect. Scaled against each model's own daylight mean:

| pair | pre-clamp night mean | served daylight mean | night level as % of own daylight | min forecast, all hours |
|---|---:|---:|---:|---:|
| CZ `solar` (ABL-580) | +40.12 MW | 1,036.8 MW | **3.87%** | +27.44 MW |
| RO `solar` (ABL-580) | +22.04 MW | 949.4 MW | **2.32%** | +15.87 MW |
| **CH `solar` (this artifact)** | **−2.01 MW** | 1,373.1 MW | **0.15%** | **−17.06 MW** |

(CH's daylight mean is over all 32 daylight hours, one of which the floor raised to 0; over the 31
strictly-positive hours it is 1,417.4 MW. The distinction moves nothing here and is stated so the
basis is not guessed at.)

CH's night level is **an order of magnitude smaller** than the floor ABL-580 described, and it is
*negative* rather than positive. So CH is not the global-level-floor case: its night predictions are
small noise of both signs around zero, 6 of 16 negative. It also goes negative in **daylight** — the
one raised hour, and a whole-window minimum of −17.06 MW, where CZ and RO never predict below +15 MW
anywhere. Both are at 27 features with `sun_elevation_deg` and `is_night` in the vector, so the
difference is the country, not the feature list.

**Two things this is not.** It is not a defect measurement: ABL-395 put CH's night-negative rate over
eight *control* fits at 77.05% ± 10.11 with a 27.34pp single-seed null, so one fit's rate is one
draw. And the 37.5% above is **not comparable** to ABL-581's 64.06% — that was the gate window, this
is 16 night hours of two forecast days. Different denominators; I am not reporting a movement.

**The served series is correct regardless.** Every night row is exactly 0.0, no row is negative, and
the clamp moves the series *toward* an actual of 0.0, not away from it. The gate scores the
*unclamped* challenger — `evaluate_solar_retrain.py` takes `model.predict(...)` raw and the clamp
lives in `save_forecasts()`, which a gate read never calls — so on these hours the served series is
at least as accurate as the one that was graded.

**Reported, not fixed**, as ABL-583 instructs: the fit path, the builder and the class are the ones
that were graded. §4.1 offers the live candidate explanation — this artifact is fitted across the
2026-06-01 basis seam, where the pre-seam actuals carry a small positive night floor, and serves into
the post-seam regime where they are a hard zero.

### 6.2 The night measurement is a snapshot at an as-of, not a property of the artifact

This has to be stated because I hit it: **the first probe run of this issue disagreed with the
serving run on every night field.**

| field | probe @ 16:36Z | probe @ 18:0xZ | serving @ 18:01Z |
|---|---:|---:|---:|
| `hours_zeroed_night` | 9 / 16 | **16 / 16** | **16 / 16** |
| `hours_raised_floor` | 2 | **1** | **1** |
| `mw_removed_night` | −63.6291 | **−32.2087** | **−32.2087** |
| `min_forecast_mw` | −21.4604 | **−17.0620** | **−17.0620** |
| `max_night_forecast_mw` | 2.4043 | **3.4012** | **3.4012** |
| pre-clamp night mean | −3.9768 MW | **−2.0130 MW** | — |

Same artifact, same replica (byte-identical, and both source-table maxima unchanged), same reference
date. The only difference was the wall-clock hour. Re-run in the same hour as the serving run, the
probe reproduces its clamp log on **all 9 shared fields exactly**, which is what
`agreement_with_night_probe.probe_reproduces_the_served_clamp: true` records — a *same-hour*
agreement, labelled as such.

The mechanism is one line, `src/forecaster.py:768`:

```python
observation_as_of = pd.Timestamp(observation_as_of) if observation_as_of is not None else pd.Timestamp(datetime.now())
```

`predict_d2` defaults its as-of to *now*, and `_predict_d2_serve_faithful` passes it to
`builder.row(target_ts, observation_as_of, weather_publication_as_of)`, which gates both the actuals
and the weather run the features may see. **This is the `as_of` / `publication_as_of` split working
correctly** — it is what makes serving leak-free — and the consequence is simply that the served
series legitimately depends on when the runner fires. Night predictions live in a ±20 MW band around
zero against a 3,500 MW daylight peak, so they are the part of the series most easily moved by a
one-cut change in the anchors.

The consequence for anyone reading a clamp-telemetry table: **`hours_zeroed_night` is not a stable
per-artifact property** and two runs of the same model on the same day may report different night
counts. Worth having on record before the weekly forecast-quality review reads that table as a
trend. The graded path is unaffected — the gate harness builds its own frames through
`build_vintage_frame` at explicit registered as-ofs and never calls `predict_d2`.

**One thing that falls out of this, flagged and bounded rather than filed.** `datetime.now()` is
naive **local** time; `target_timestamp_utc` is UTC. On this box (UTC+2) the recorded horizons are
3–50 h, and `2026-08-28 00:00Z − 3.98 h = 2026-08-27 20:01` — the local wall clock, not the 18:01Z it
actually was. So the as-of cut sits one UTC offset later than the true instant. For **live** serving
this is harmless: the extra two hours of rows do not exist yet, so nothing is admitted. It would be a
real leak vector for anything that **replays** this path at a historical as-of with the default, and
it makes `horizon_hours` offset-dependent — ABL-580's run recorded 7–54 h for the same target days
purely because it fired earlier. I have **not** checked the production container's timezone, so I
cannot say whether this reaches prod at all; if it does, it is serving-path code and belongs to the
Founding Engineer, not to me. Raised here rather than filed because I have not measured it beyond
this one run.

### 6.3 Level sanity — a scale check, not a skill claim

Two forecast days against a 14-day actual mean; different windows, so this is not a metric. Its
purpose is the lost-intercept failure mode, where an artifact loaded under the wrong interpreter
keeps its shape and predicts a near-zero-mean series. Compared daylight-only on both sides, because
the clamped night zeros would otherwise drag the served mean down against actuals that are also zero.

| | served, 2026-08-28 → 29 | actual, trailing 14 d | ratio |
|---|---:|---:|---:|
| CH `solar`, daylight | 1,373.1 MW (n=32) | 1,477.5 MW (n=238) | **0.93×** |

Both sides are the mean over *all* daylight hours on the same night mask, so the bases match; on the
31 strictly-positive served hours it is 1,417.4 MW and 0.96×. Within 7% of the recent actual level
either way, comfortably inside the 0.45×–1.53× spread the ABL-525 seven
showed, and nothing resembling a lost intercept. This artifact is **catboost**, which round-trips its
own bias, so it does not carry that hazard in the first place — the check is run anyway because a
passing control is what makes the NL `wind_offshore` warning in §7 meaningful.

### 6.4 Where I stopped

Those 48 rows went to a scratch sidecar under `PAPERCLIP_RUN_SCRATCH_DIR`. **The replica still holds
0 rows for CH `solar` and I never wrote it** — every read in this pack opened `mode=ro`. Production
serves by running `forecast_daily.py` inside the container against a bind-mounted `MODELS_DIR`. The
remaining step is a deploy, which is a hard boundary for me and is routed separately.

## 7. Staged for the deploy -- this PR moves no artifact

`models/` is gitignored with zero tracked files, so **merging this PR ships nothing by itself.**
The artifact is staged at the path ABL-583 item 7 names -- note it is named after the *read*
(`abl581`), not after this issue, which is the CEO's instruction and is left exactly as given so
whoever picks up the deploy finds it where they were told:

```
C:\Code\able\data\abl581_ch_artifacts\CH\solar\model.joblib
```

Same layout as `abl525_ship_set_artifacts\` and `abl580_ship_set_artifacts\`, and all **eight**
machine records are copied beside it as ABL-580 did, so the staging directory is self-describing.

The sha256 was **re-verified after copying**, against `reports/abl_583_ship_set_training.json`:

```
record  7bcfecd590a42c01b06fc4c0584b238db7724ed8fe5807b34989429318239793
staged  7bcfecd590a42c01b06fc4c0584b238db7724ed8fe5807b34989429318239793   match
```

1 of 1. And per §5 the artifact is regenerable by the committed script with bit-identical
predictions, so the recipe is as good as the bytes.

Whoever takes this into the production `MODELS_DIR` should note that it is a **catboost** artifact,
which does not carry the xgboost lost-intercept hazard NL `wind_offshore` carries -- catboost
round-trips its own bias. The interpreter still matters for everything else in the batch.
