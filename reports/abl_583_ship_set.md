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
| `reports/abl_583_ch_night_probe.json` | the night question, either side of the ABL-337 serving clamp |
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

### 3.2 The window caveat, stated before anyone asks

`FIT_START` / `FIT_END` are the ship set's module constants, unchanged: **2026-01-11 -> 2026-08-22**.

`FIT_END` stays at the Board's decision date even though the replica now carries actuals past it,
and the reason is stronger with a third batch than it was with two: `abl525_repro_check.py` refits
through `fit_one` on these same module constants, so moving them would make the ABL-525 seven and
the ABL-580 three report a prediction difference that is a window change rather than a drift, and
every artifact in the ship set stays on one window so the deploy is a homogeneous batch.

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

## 7. Staged for the deploy -- this PR moves no artifact

`models/` is gitignored with zero tracked files, so **merging this PR ships nothing by itself.**
The artifact is staged at the path ABL-583 item 7 names -- note it is named after the *read*
(`abl581`), not after this issue, which is the CEO's instruction and is left exactly as given so
whoever picks up the deploy finds it where they were told:

```
C:\Code\able\data\abl581_ch_artifacts\CH\solar\model.joblib
```

Same layout as `abl525_ship_set_artifacts\` and `abl580_ship_set_artifacts\`, and the four machine
records are copied beside it as ABL-580 did, so the staging directory is self-describing.

Whoever takes this into the production `MODELS_DIR` should note that it is a **catboost** artifact,
which does not carry the xgboost lost-intercept hazard NL `wind_offshore` carries -- catboost
round-trips its own bias. The interpreter still matters for everything else in the batch.
