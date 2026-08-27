# ABL-580 — ABL-316 ship set, second batch: CZ solar, RO solar, NL wind_offshore

Forecasting Scientist. Three production artifacts fitted through the graded gate-harness path,
serving-verified end to end, reproducibility proved by prediction equality, and staged for the
deploy. **Nothing here is scored, graded or promoted.** Membership is the CEO's; this pack is the
evidence behind three fits that membership named.

Companion machine records, all committed and all outside the `experiments/*/results.json` glob
that `.gitignore` swallows (ABL-440, still open):

| record | what it holds |
|---|---|
| `reports/abl_580_ship_set_training.json` | per-pair fit provenance, audits, artifact paths and sha256s |
| `reports/abl_580_reproducibility.json` | the refit-and-compare-predictions check |
| `reports/abl_580_contamination_screens.json` | ABL-332 / ABL-200 / ABL-188, the night floor, the ABL-439 vintage screen |

---

## 1. Why these three, and what authorises them

The Board approved a **rule** alongside the `ship8` roster on 2026-08-22 (ABL-316 ledger 14.6,
restated in 15.1): a held pair that later satisfies the same rule joins the shipping set without a
new Board card. The CEO ran that rule over ABL-426 (tranche 2a re-read on the registered
`energy_generation`) and ABL-471 (the last four vintage screens) and admitted three pairs. The
disposition is theirs and is **not re-derived here** — this pack fits what it names.

The grades and margins behind the admission are on ABL-316. They are reproduced nowhere in this
pack on purpose: re-stating a margin is the first step toward re-deriving one, and this issue
excludes new gate reads.

## 2. The two checks the issue asked to be *verified rather than assumed*

### 2.1 Solar — CZ and RO are the inverse of CH, and it is measured

CH solar was withdrawn from the ship set because tranche 1b graded it at the legacy **25**-name
solar list while ABL-395 moved `solar_retrain.FEATURE_COLUMNS` to **27**. The same question has to
be asked of CZ and RO, and the answer cannot come from a registration table: the scope behind their
disposition, `abl316-t2a-generation`, is **deliberately absent** from
`evaluate_solar_retrain.SCOPE_FEATURES` and resolves through `DEFAULT_SCOPE_FEATURES`. That table's
own comment says the absence is the intended path for a new tranche.

So the authority is the record the fit wrote:

```
experiments/ABL348/results_abl426_tranche2a_generation.json
  meta.n_features                           27
  meta.feature_set                          legacy25+geometry
  meta.feature_set_is_registered_for_scope  False    (inherited the default)
  meta.feature_columns == src.evaluation.solar_retrain.FEATURE_COLUMNS  ->  True
```

Element for element, not merely by count. `abl316-t2a` (ABL-405) reads the same. So CZ and RO solar
are graded on a list the builder still produces today — no pin, no fork, and none of the conditions
that stopped CH. `tests/test_abl580_ship_set_batches.py::test_the_solar_ship_rows_take_the_list_their_read_recorded`
holds this, so a later move of `FEATURE_COLUMNS` to 28 fails the suite rather than silently
re-basing these artifacts.

Also read off the same record: `registered_source` and `training_source` both `energy_generation`;
`fit_rules` `{"exclude_impossible_night": false}`, which is what this fit path does anyway, so the
fit rule matches by construction rather than by configuration.

### 2.2 Wind — the feature list holds; the **algorithm** did not

`wind_retrain.FEATURE_COLUMNS` (lines 16–25) has changed in exactly one commit, `601f10f`
(2026-08-11), its introduction — `git log -L 16,25:src/evaluation/wind_retrain.py` returns that and
nothing else. The offshore re-read is stamped `2026-08-14 00:49 UTC`, so the constant predates it by
three days. And `wind_offshore` takes the same list as `wind_onshore` **by construction**: the wind
harness has no `SCOPE_FEATURES` table at all and applies the one module constant at all three of its
fit/predict sites (`evaluate_wind_retrain.py:917`, `:926`, `:933`) with no per-type branch.

The algorithm is a different story, and reusing ABL-525's script unchanged would have got it wrong:

```
evaluate_wind_retrain.py:55
    ALGORITHMS = {"wind_offshore": "xgboost", "wind_onshore": "catboost"}
```

ABL-525's eight pairs were seven `wind_onshore` plus one `solar` — both catboost — so that script
carried a single `ALGORITHM = "catboost"` module constant, and nothing in it distinguished *the
estimator this pair was graded with* from *the estimator this file happens to name*. The pilot's own
committed record (`experiments/ABL322/results_abl436_offshore_reread.json`) shows DE and NL both
fitted `"algorithm": "xgboost"`. Fitting NL with catboost here would have shipped a model no gate
read — the same class of error ABL-525 item 2 exists to prevent, arriving through a constant instead
of through a feature list.

The fix is to *import* the harnesses' tables rather than restate them, so the trainer cannot come to
disagree with the code that graded the pair. Two properties are held by test rather than by comment:
the resulting xgboost params equal the ABL-436 record's NL params dict exactly, and every ABL-525 row
still resolves to catboost — so **the seven artifacts already fitted are unaffected** and
`abl525_repro_check.py` still reproduces them.

## 3. Per-pair fit

Source `energy_generation` (ABL-321/ABL-348 registered), 8 pre-registered vintages per target,
14-day lookback for the point lag and the 168-hour anchors, one window for all three, no per-country
fork of anything.

| pair | tranche | algorithm | retained / intended rows | targets | features | training_source |
|---|---|---|---:|---:|---:|---|
| CZ `solar` | 2a (ABL-426 re-read) | catboost | 41,736 / 42,816 | 5,247 | 27 | `energy_generation` |
| RO `solar` | 2a (ABL-426 re-read) | catboost | 42,384 / 42,816 | 5,328 | 27 | `energy_generation` |
| NL `wind_offshore` | pilot (`abl322-pilot`, ABL-436 re-read) | **xgboost** | 42,384 / 42,816 | 5,328 | 24 | `energy_generation` |

Requested window **2026-01-11 → 2026-08-22**; retained **2026-01-12 00:00 → 2026-08-21 19:00** (CZ)
and **→ 23:00** (RO, NL). The first day drops because its D-2 vintage predates the weather archive's
first run. Intended rows are 5,352 targets × 8 vintages and the shortfalls are missing actuals: CZ
loses 1,080 rows, of which 840 are 105 targets lost entirely (5,352 → 5,247 unique targets) and 240
are partial vintage losses on targets that survive; RO and NL each lose 432 rows, of which 192 are
24 targets lost entirely (5,352 → 5,328) and 240 are again partial. Every audit count is in the
machine record.

Artifacts, and the sha256 of each as written:

| pair | bytes | sha256 |
|---|---:|---|
| CZ `solar` | 2,112,596 | `2cb5d1e75de1624c3692da22b3e0de2414736af4ccc9694f5e159268c77058b6` |
| RO `solar` | 2,112,308 | `e5a8cb7194d243f2c7971299c41fe25e06a15d79f5697f77266e2373a945ffae` |
| NL `wind_offshore` | 6,383,492 | `3f0ff59bcd31dfcecafbc7f39f64ff5fe6a9753e78dd0d6c915f6a4bd954f915` |

### 3.1 The window caveat, stated before anyone asks

`energy_generation` reaches **2021-01-01** for all three pairs. A *serve-faithful* row also needs the
weather **forecast** archive, and `weather_data` with `data_quality='forecast'` begins **2026-01-11**
for CZ, RO and NL alike — re-measured on today's replica, first run 2026-01-11 18:00 for all three.
An earlier target gets NaN weather and `finite_training_rows` drops it. So the widest honest window
is **223 days, not five years**, against the gate's registered 178.

**It covers the gate window. The figures behind the admission are therefore NOT out-of-sample for
these artifacts.** That is what fitting on full available history asks for and is correct for
production, but a later reader must not mistake those margins for a validation of these three models.
Nothing in this pack was scored or graded.

`FIT_END` stays at the Board's decision date even though the replica now carries actuals to
2026-08-26. Two reasons: every artifact in the ship set is then on one window, so the deploy is a
homogeneous batch; and `abl525_repro_check.py` refits through the same module constants, so moving
them would make the seven ABL-525 artifacts report a prediction difference that is a window change
rather than a drift. Five days is 2.2% of the window.

## 4. Contamination screens

Machine record: `reports/abl_580_contamination_screens.json`. Every screen is taken over the window
these artifacts were **actually fitted on**, not over the window their gate read covered — which
matters, because the fit window is 45 days wider and no existing record covers the extra.

### 4.1 ABL-332 — hourly aggregation, in the path by construction and observed

All three series are **quarter-hourly** on `energy_generation`. `load_renewable_type_data` calls
`aggregate_renewable_to_hourly`, and `RenewableFeatureBuilder._assert_hourly` then *raises* on an
off-hour index rather than subsampling — so fit and serve are the same hourly frame by construction,
not by coincidence. The same builder object does both.

| pair | raw rows in fit window | hourly rows after aggregation | rows the pre-ABL-332 builder would have discarded |
|---|---:|---:|---:|
| CZ `solar` | 20,995 | 5,271 | 15,724 |
| RO `solar` | 21,401 | 5,352 | 16,049 |
| NL `wind_offshore` | 21,408 | 5,352 | 16,056 |

### 4.2 ABL-200 — cannot fire on this batch, and that is a constant not a measurement

`exclude_zeros_disproved_by_sibling` is wired at `load_renewable_type_data` behind
`if source != RENEWABLE_ZERO_DISPROOF_SOURCE`, and that constant **is** `energy_generation`
— read off `src.db` by the screen rather than restated here. All three pairs read that same table,
so the rule never fires: it is one-sided, and `energy_generation` is already the disproving side.

### 4.3 ABL-188 — the constant-run guard ran and found nothing

**0** rows nulled, on all three pairs, over the full fit window. Reported as a count rather than as
a boolean because "the guard ran" and "the guard found nothing" are different facts and only the
second is evidence about these series.

### 4.4 The night floor — neither CZ nor RO shows the BG signature

ABL-405's night probe read `energy_generation` while its *fit* read `energy_renewable`; ABL-426's
re-read is the first where both are one series, and this screen is the first that covers the fit
window of the artifact being shipped. Predicate is `solar_features.night_mask` — the sun
geometrically below −8° for the **whole** hour at the country's capacity-weighted point, bit-identical
to the ABL-337 serving clamp's — so a non-zero night actual is not a timezone offset or a mask
artefact. Threshold 1 MW (ABL-338's).

| pair | window | night hrs > 1 MW | night mean | night max | negative night hrs | energy at night | WAPE floor if clamped |
|---|---|---:|---:|---:|---:|---:|---:|
| CZ | 2026-01-11 → 07-11 | 2 / 1,450 (0.14%) | 0.01 MW | 6.81 MW | 0 | 0.000% | **0.0004%** |
| CZ | 2026-07-11 → 08-22 | 2 / 177 (1.13%) | 0.07 MW | 9.52 MW | 0 | 0.001% | **0.0013%** |
| CZ | **whole fit window** | 4 / 1,627 (0.25%) | 0.01 MW | 9.52 MW | 0 | 0.001% | **0.0006%** |
| RO | 2026-01-11 → 07-11 | 2 / 1,522 (0.13%) | 0.01 MW | 8.50 MW | 0 | 0.001% | **0.0008%** |
| RO | 2026-07-11 → 08-22 | 0 / 284 | — | — | 0 | 0.000% | **0.0000%** |
| RO | **whole fit window** | 2 / 1,806 (0.11%) | 0.01 MW | 8.50 MW | 0 | 0.001% | **0.0005%** |
| *BG, the signature* | *fit / gate* | *76.4% / 85.2%* | *152 / 246 MW* | *~1,090 MW* | — | ***6.37% / 4.98%*** | *~6.4% / 5.0%* |

`wape_floor_pct_if_clamped` is the rankable quantity (ABL-396): the full width, in WAPE points, of
the interval an all-hours read can occupy relative to the same challenger's daylight-only read, and a
hard lower bound on the WAPE of any *served* forecast, since the clamp cannot do better than zero
against a floor. On that axis CZ and RO sit **three to four orders of magnitude** below BG. Neither
shows the BG signature.

Ranking on the count alone would be wrong here and the table shows why: CZ's own gate-side window
reads 1.13% of night hours above threshold, which is a larger *percentage* than several benign rows
elsewhere in the fleet, at a night mean of 0.07 MW against a daylight mean over 1,000 MW. Energy is
the honest axis; BG is the exception and is not marginal about it.

### 4.5 NL `wind_offshore` — the ABL-439 vintage screen, re-derived rather than cited

**The issue's provenance claim is wrong and the figures are right.** It cites this screen as
"ABL-471 (merged, PR #83)". PR #83 is **closed, not merged** — `mergedAt` and `mergeCommit` both
null, closed 2026-08-24T06:41:51Z, two minutes after PR #82 merged; `d6c8408` is not an ancestor of
`origin/main`; none of its four files is tracked. So the record backing the hold-clearing screen for
the one pair whose hold it cleared is untracked.

Re-derived here through `abl439_reporting_basis_probe._hourly` — the same primitive ABL-471 called,
and one that *is* on `origin/main` — over ABL-348's registered windows, on today's replica
(10,632,605,696 bytes), and pinned to ABL-471's published values so a disagreement would be a
failure of this screen rather than a quiet new number:

| window | n hours common | `energy_generation` mean | `energy_renewable` mean | ratio |
|---|---:|---:|---:|---:|
| ABL-348 fit, 2026-01-14 → 07-11 | 4,245 | 1,692.3275 MW | 1,705.5947 MW | **0.9922** |
| ABL-348 gate, 2026-07-11 → 08-10 | 720 | 1,134.3434 MW | 1,144.3739 MW | **0.9912** |

**Discontinuity (fit − gate) = +0.0010 → basis-consistent** against ABL-471's 0.02 cut. Reproduces
ABL-471's published `{fit 0.9922, gate 0.9912, discontinuity 0.0010}` to the decimal, five days on a
newer replica. The discontinuity — not either ratio — is the quantity that voids a gate read: a model
is fitted *and* scored on the registered table, so a steady offset between the two sources voids
nothing and only a change of basis between the two windows can.

For contrast, the pairs still held: LV `solar` +0.1706 and EE `solar` +0.1764, both
basis-INCONSISTENT, and NL `wind_onshore` +1.2659, the pair ABL-439 diagnosed. The gap between
+0.002 and +0.17 is two orders of magnitude wide and empty across all 41 screened pair-records, so
nothing here is a close call.

**Carried caveat, not dropped.** NL `wind_offshore`'s gate window is 100% first-publication, so its
gate-side revision is *expected-small but not yet measured*. That is true of every pair in this gate
and is not a blocker; it is recorded so a later reader does not mistake an unmeasured quantity for a
measured-zero one.

### 4.6 The three named contamination issues

ABL-71 (prod ingest stale) and ABL-67 (fabricated rows) are `net_position`. ABL-111/ABL-109 are
zero-as-missing actual **load** rows. **None of the three touches these series or this window.**

## 5. Reproducibility — by prediction equality, not by artifact hash

`Forecaster.save` stamps `saved_at`, so three byte-identical fits give three different digests and a
hash comparison reports drift that is not there. Each pair was refitted independently into a scratch
directory with the same function, window and seed; both artifacts were loaded through
`Forecaster.load` — the entry point `forecast_daily.py` uses, so this compares two things that were
actually deserialised — and both predicted on **one shared feature matrix**, so the comparison cannot
hide a builder difference behind a model difference.

```
[PASS] CZ/solar         max|a-b| = 0.000e+00 over 168 rows, bit-identical=True, sha256 differs=True
[PASS] RO/solar         max|a-b| = 0.000e+00 over 168 rows, bit-identical=True, sha256 differs=True
[PASS] NL/wind_offshore max|a-b| = 0.000e+00 over 168 rows, bit-identical=True, sha256 differs=True
```

All three **bit-identical**, comfortably inside 1e-12, while **every** artifact sha256 differed —
which is the point. `feature_columns` and `training_source` round-tripped identically on all three
(27 / 27 / 24 names, `energy_generation` throughout), so a matching prediction vector cannot be a
coincidence on a mismatched column order. Probe window 2026-08-01 + 168 h, inside the fit window on
purpose: this measures artifact equality, not generalisation, and an out-of-window hour would only
contribute NaN features.

Record: `reports/abl_580_reproducibility.json`, `all_pairs_reproducible: true`,
`every_artifact_sha256_differed: true`.

## 6. Serving, verified end to end up to a boundary I did not cross

`forecast_daily.py --countries CZ,RO,NL --types solar,wind_offshore --horizons 1,2`:

```
Total: 12, Success: 6, Empty: 0, Unreported: 0, Skipped: 6, Failed: 0
Saved 144 forecasts to database
```

**144 rows** = 3 pairs × 2 horizons (D+1, D+2) × 24 h, targets 2026-08-28 → 2026-08-29, horizons
7–54 h. **0 failed, 0 empty, 0 unreported.** The six skips are the six pairs with no artifact
(CZ/RO `wind_offshore`, NL `solar`) — `skipped` is `Forecaster.load`'s `FileNotFoundError` flag, set
at the one place that knows, which is the correct outcome and not a failure.

Each artifact loaded with the estimator it was fitted with — `catboost` for CZ and RO, **`xgboost`
for NL** — and the log line `Loaded 505 hourly solar records for RO from energy_generation` confirms
the artifact's own `training_source` drove the serve-side read, so ABL-331 works end to end on this
batch too. No `KeyError` anywhere: the direct counterfactual to what a `scripts/train.py` artifact
would have done.

### 6.1 The solar clamp is busy, and that is a finding

CLAUDE.md is explicit that a retrain is verified by the clamp going quiet, not by the retrain having
run. It did not go quiet:

| pair | night hours | zeroed | raised to 0 | MW removed | pre-clamp night mean | pre-clamp night max | min forecast (all hours) |
|---|---:|---:|---:|---:|---:|---:|---:|
| CZ `solar` | 16 | **16 / 16** | 0 | 641.84 MW | **40.12 MW** | 59.76 MW | 27.44 MW |
| RO `solar` | 16 | **16 / 16** | 0 | 352.61 MW | **22.04 MW** | 29.35 MW | 15.87 MW |

Against night actuals whose mean is **0.01 MW**. So both models carry a night floor the series does
not have — at 27 features, with `sun_elevation_deg` and `is_night` in the vector.

Three things make this a finding rather than a blocker, and it is worth being precise about which:

1. **The served series is correct.** All 32 night rows are exactly 0.0 and there are no negative
   rows anywhere. `hours_raised_floor` is 0 for both, so this is not the CH failure mode — these
   models predict *positive* at night, not negative.
2. **The clamp moves the served series toward the truth, not away from it.** The gate scores the
   *unclamped* challenger — `evaluate_solar_retrain.py:1868` takes `model.predict(...)` raw, and the
   clamp lives in `save_forecasts()`, which a gate read never calls. The served model is clamped to
   zero where the actual is 0.01 MW. On those hours the served series is therefore at least as
   accurate as the one that was graded, not less.
3. **It reads more like a global level floor than a night defect.** `min_forecast_mw` across all 48
   hours is 27.44 MW (CZ) and 15.87 MW (RO) — these models never predict below ~2–3% of their own
   daylight mean, anywhere. At night that floor is all there is.

Not fixed here: the fit path, the feature builder and the class are the ones that were graded, and
changing any of them is outside this issue. Recorded so that whoever picks up the solar night
question has the measurement rather than an impression.

### 6.2 Level sanity — a scale check, not a skill claim

Two forecast days against a 14-day actual mean; these are different windows and this is not a metric.
What it is for is the lost-intercept failure mode, where an artifact loaded under the wrong
interpreter keeps its shape and predicts a near-zero-mean series. Solar is compared daylight-only on
both sides, because missing night rows would otherwise bias the actuals upward.

| pair | served mean | actuals mean, 14 d | ratio | basis |
|---|---:|---:|---:|---|
| CZ `solar` | 1,036.8 MW | 1,207.3 MW | 0.86× | daylight |
| RO `solar` | 949.4 MW | 1,410.5 MW | 0.67× | daylight |
| NL `wind_offshore` | 1,336.4 MW | 963.6 MW | 1.39× | all hours |

All the right order of magnitude, inside the 0.45×–1.53× spread the ABL-525 seven showed, and nothing
resembling a lost intercept. RO is the lowest and is two specific days against a fortnight; solar
day-to-day variation spans that easily.

### 6.3 No country gate anywhere — ABL-319 holds

`forecast_daily` iterates `SUPPORTED_COUNTRIES × types` with artifact presence as the only filter,
and the live API is data-driven rather than allow-listed:

```
/api/forecasts/types?country=BE  ->  biomass, hydro_total, load, net_position, price,
                                     renewable, solar, wind_offshore, wind_onshore
/api/forecasts/types?country=CZ  ->  load, net_position, price
/api/forecasts/types?country=RO  ->  load, net_position, price
/api/forecasts/types?country=NL  ->  load, net_position, price
```

BE renders both `solar` and `wind_offshore` because it has rows; CZ, RO and NL do not because they
have none. All three are in `/api/countries`. The moment rows exist, the type appears. **Nothing to
file for the Founding Engineer.**

**A limit on that evidence, stated rather than left implicit.** BE and FR are the positive control
because they are the only one available: the replica read here still holds **0** rows for all seven
ABL-525 `wind_onshore` pairs, and its newest `forecasts` row of any kind is 2026-08-26 19:00 against
a file refreshed 2026-08-27 07:33. So "the type appears the moment rows exist" is demonstrated on
streams that have served for a long time, and **not yet on a newly deployed renewable pair**. That is
not a finding about ABL-579 — the replica mirrors prod on a 07:00 sync, so a deploy completed later
the same day could not appear in it regardless — it is a bound on what this pack can claim. The first
new pair to land will settle it.

### 6.4 Where I stopped

Those 144 rows went to a **scratch sidecar** under `PAPERCLIP_RUN_SCRATCH_DIR`. The replica still
holds **0** rows for CZ `solar`, RO `solar` and NL `wind_offshore`, and I never wrote it — every read
in this pack opened `mode=ro`. The API reads `/data/energy_dashboard.db` inside a container
(`"runtime":"container"` from `/api/health`), and production serves by running `forecast_daily.py`
in that container against a bind-mounted `MODELS_DIR`. The remaining step is a deploy, which is a
hard boundary for me.

## 7. Staged for the deploy — this PR moves no artifact

`models/` is gitignored with zero tracked files, so **merging this PR ships nothing by itself.** The
three artifacts are staged at:

```
C:\Code\able\data\abl580_ship_set_artifacts\CZ\solar\model.joblib
C:\Code\able\data\abl580_ship_set_artifacts\RO\solar\model.joblib
C:\Code\able\data\abl580_ship_set_artifacts\NL\wind_offshore\model.joblib
```

Same layout the ABL-525 seven were staged in at
`C:\Code\able\data\abl525_ship_set_artifacts\`. All three sha256s re-verified against
`reports/abl_580_ship_set_training.json` **after** copying: 3 of 3 match. They are also regenerable
by the committed script and proved bit-identical on refit, so the recipe is as good as the bytes.

Whoever takes these into the production `MODELS_DIR` should note that **NL `wind_offshore` is an
xgboost artifact fitted under `.venv` (Python 3.14.3, xgboost 3.3.0)**. Loading it under the conda
interpreter does not fail — it silently resets the fitted intercept and predicts a near-zero-mean
series with intact shape. The two solar artifacts are catboost and do not carry that hazard.
