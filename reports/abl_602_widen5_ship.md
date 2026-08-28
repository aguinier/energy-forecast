# ABL-602 — the ABL-316 widened set: five artifacts, fitted and serving-verified

**Issue:** ABL-602, under ABL-316.
**Board answer:** `widen7` on ABL-316, 2026-08-28 — adopt the causally-available
standard for the widened set.
**Author:** Forecasting Scientist.
**Read at:** `origin/main` `79433d0b`, branch `ABL-602-widen5-artifacts`.

> **This pack fits and verifies. It scores nothing, grades nothing, re-reads no
> gate, and promotes nothing.** Every letter, margin and grade quoted below is
> read out of a committed record; none is re-derived. Membership of the widened
> set is the CEO's disposition. The production deploy is a separate issue.

---

## 1. What was delivered

Five artifacts, fitted through the **graded** gate-harness path, each with:

- a committed training record carrying its sha256 (`models/` is gitignored, so
  the digest in the record is the only thing that ties a served model to this
  work);
- a reproducibility proof at **1e-12 on the predictions**, not on the artifact
  hash;
- **end-to-end serving verification through `forecast_daily.py` at both
  horizons**, read out of the sidecar that run wrote — not inferred from the
  training log — with each served artifact re-hashed against the training
  record.

| pair | tranche | scope | algorithm | features |
|---|:---:|---|:---:|:---:|
| `LT` `solar` | 2d | `abl316-t2d` | catboost | 27 |
| `SE` `solar` | 2d | `abl316-t2d` | catboost | 27 |
| `HR` `wind_onshore` | 2e | `abl417-tranche2e` | catboost | 24 |
| `HU` `wind_onshore` | 2e | `abl417-tranche2e` | catboost | 24 |
| `PL` `wind_onshore` | 2b | `abl406-tranche2b` | catboost | 24 |

`NO` and `RO` `wind_onshore` are **not** here and are not in `SHIP_SET` at all.
Both clear G1 and fail **G4** — the sign test on the challenger's own slope and
correlation. Re-verified against the records before filing, and the two live in
different places, which the issue's citation blurred:

- `RO`: `experiments/ABL348/results_abl417_tranche2e.json` — grade **B**, `G4`
  false in all three bands, `G1` true in all three.
- `NO`: tranche **2b**, whose stored record predates the G1–G4 ladder, so its
  letters are in `reports/abl_418_retro_grade.json` — grade **B**, `G4` false,
  `G1` true. The 2e record has no `NO` cells at all.

G4 is computed from the challenger's own predictions against the actuals in the
gate window. It is not a reference comparison and carries no hindsight, so it is
admissible under exactly the standard the Board adopted. The exclusion is the
CEO's ruling on ABL-316; this pack only confirms the records it was taken on.

---

## 2. Protocol

**The graded path, not `scripts/train.py`.** `train.py` builds 28 names for wind
and 31 for solar including four holiday columns; the serve-faithful builder
(`src/wind_features.RenewableFeatureBuilder` + `to_vector`) produces 24 and 27
and no holiday column. `Forecaster.predict_d2` routes both types through
`_predict_d2_serve_faithful`, and `to_vector` raises `KeyError` on a column the
builder cannot produce — so a `train.py` artifact for these pairs loads clean and
then raises on its first serving row. These five were fitted with the same
builder, the same `FEATURE_COLUMNS` and the same estimator the tranches used,
through `save_gate_artifact`.

**Algorithm is per forecast type, resolved not restated.** Both types in this
batch are catboost, which is exactly when a batch is tempted to hardcode one.
`algorithm_for` imports `evaluate_wind_retrain.ALGORITHMS` and
`solar_retrain.ALGORITHM`, so an offshore pair added later cannot be silently
fitted with the onshore estimator. (ABL-580 is the standing evidence:
`wind_offshore` is xgboost where onshore and solar are catboost.)

**No per-country fork.** ABL-401 §4 and ABL-525 item 2 forbid it; none of the
five pins a feature list.

**Feature-class check — re-derived, not taken on trust.** This is the check that
withdrew CH `solar` on ABL-525 and cost a re-read on ABL-581, so it was
reproduced rather than accepted:

- **Solar.** `results_abl421_tranche2d.json` records `meta.n_features: 27` and a
  `meta.feature_columns` that is **element-for-element identical** to today's
  `solar_retrain.FEATURE_COLUMNS` and to
  `tests/feature_list_manifest.json → gate_harness.solar.columns`.
  `meta.feature_set_is_registered_for_scope` is `false`, which is the correct
  configuration and not a gap — `SCOPE_FEATURES` is a table whose absence
  encodes a choice, and inheriting the current list is the intended path. The
  manifest's frozen note names only `abl253` and `abl316-t1b` as the scopes
  fitted at the legacy 25. **LT and SE were graded at today's live list.**
- **Wind.** `wind_retrain.FEATURE_COLUMNS` is 24 names, and the sha256 of the
  comma-joined names is `2a034c79a1cd` at **every commit that ever touched the
  file** — `601f10f` (2026-08-11, its introduction), `eaab3e3`, `75adff8`, and
  `origin/main` today. ABL-395's geometry features are solar-only by design
  (`_solar_geometry_features` returns `{}` for any other type, so naming one
  would raise in `to_vector`). **HR, HU and PL sit on today's live list.**

`tests/test_abl602_widened_batch.py` holds both equalities element-for-element,
so a later move of either constant fails the suite rather than silently re-basing
these artifacts.

**Coverage.** `enough_pairs` is `true` on all three gating bands for all five
pairs: n = 720 / 720 / 510 against registered minima 684 / 684 / 456. (Read
beside the grade, not from a flat lookup — it nests under `gate`, so a lookup
that misses it passes vacuously.)

**The fit window covers the gate window.** `FIT_START` / `FIT_END` are
`2026-01-11` → `2026-08-22`, the ship8 batch's window, kept identical so the
deploy is homogeneous and so `abl525_repro_check.py` cannot report a window
change as a drift. The gate window is `2026-07-11` → `2026-08-10`, inside it.
**The tranche figures are therefore not out-of-sample for these artifacts.**
That is what "fit on full available history" asks for and is correct for
production, but it means no number in section 3 is a holdout number for the
model being shipped. `protocol.fitted_on_the_gate_window` is `true` in the
machine record for the same reason.

---

## 3. The letters, and the convention they belong to

**This section is the one to read twice.** Caveat 3 of the issue asks for the
amended letter under the registered defaults and warns against quoting the
published letter bare. Resolving that precisely turns up a correction.

### 3.1 The registered convention *is* the published one

All three scopes in this batch are **published** scopes, and published scopes are
pinned by value on both grading axes:

| scope | `CAUSAL_LEVELLING` | `G23_READABILITY` |
|---|---|---|
| `abl316-t2d` (LT, SE `solar`) | `FIT_WINDOW` | `SIGN_TEST` |
| `abl417-tranche2e` (HR, HU `wind_onshore`) | `FIT_WINDOW` | `SIGN_TEST` |
| `abl406-tranche2b` (PL `wind_onshore`) | `FIT_WINDOW` | `SIGN_TEST` |

ABL-437 and ABL-444 moved the **defaults a new scope inherits** to
`TRAILING_28D` and `FLOORED`. They did not re-grade these scopes, and a pin is
not an oversight — `abl581-ch-solar-f27` is the first scope in the programme
registered *on* the amendments, precisely because it had no letters to protect.

So there are two defensible readings and they are not the same number.
`reports/abl_444_g23_floor_reread.json` reports every pair under all four
combinations. Worst-band pair letters:

| pair | registered `fit_window/sign_test` | amended defaults `trailing_28d/floored` | mixed `fit_window/floored` |
|---|:---:|:---:|:---:|
| `LT` `solar` | **A** | **N** — G3 −6.87%, inside the 10.65% floor | A |
| `SE` `solar` | **A** | **N** — G3 −8.65%, inside the floor | A |
| `PL` `wind_onshore` | **A** | **A** | A |
| `HR` `wind_onshore` | **A** | **N** — G2 +2.80% / G3 +2.02%, inside the 7.51% floor | A |
| `HU` `wind_onshore` | **B** | **B** | N |

### 3.2 Four of the five move, not two

Caveat 3 names HR and HU as the convention-sensitive pairs. The record says
**LT, SE and HR** move — all three A → N — and **only PL** holds a letter under
all four conventions. All three N's are *abstentions*: the margin sits inside the
readability floor, which is "cannot be read", not "lost". That does not block
shipping under the adopted standard. It does mean a reader given only the
published letter would not know which four move, which is what caveat 3 exists to
prevent.

### 3.3 HU is not an abstention under either endpoint

The issue's table reads `HU` as *"G4 true, G2/G3 inside floor — abstention, not a
readable loss"*. That letter is `N`, and `N` appears under exactly one
convention: `fit_window/floored`, which applies **ABL-444's floor without
ABL-437's levelling**. Under both endpoints HU is **B**:

| band | D-7 skill | G2 (`fit_window`) | G3 (`fit_window`) | G2 (`trailing_28d`) | G3 (`trailing_28d`) |
|---|---:|---:|---:|---:|---:|
| 24–36h | +15.52% | −1.75% | −2.13% | **−26.78%** | **−25.51%** |
| 36–48h | +15.13% | −2.21% | −2.60% | **−26.97%** | **−26.06%** |
| 48–64h | +16.53% | −4.90% | −4.17% | **−28.68%** | **−22.83%** |

Under fit-window levelling the margins are signed losses that happen to be
smaller than the 7.51% wind floor — unreadable, hence the mixed convention's
abstention. Under the trailing-28d levelling the same losses are **three to four
times the floor**, and `abl_444_g23_floor_reread.json` records them as `failed`
rather than `not_readable`. That is the definition of a readable loss.

**Stated plainly:** under the causally-available standard the Board adopted, `HU`
`wind_onshore` clears G1 (+15.13% vs seasonal-naive D-7 at its worst band) and
clears G4, and **loses readably to both a flat line and an hour-of-day
climatology** on the trailing-28d reference. G2 and G3 are as causally available
as G4 is — they compare the challenger against references built only from data
available at the forecast issue instant. The argument that excluded NO and RO on
G4 applies to HU on G2/G3 under the amended levelling, and does not apply under
the registered one.

**HU is fitted and verified here anyway**, because membership is the CEO's
disposition and the Board answered `widen7`. This is a decision point, not a
blocker: dropping the HU row leaves the other four untouched.
`tests/test_abl602_widened_batch.py::test_hu_is_a_readable_loss_under_the_amended_defaults`
holds the correction so the abstention reading cannot be quietly restored.

### 3.4 The other two caveats, carried

**Caveat 1 — `SE` `solar` is the thinnest pair in the widened set.** Worst-band
D-7 skill **+11.29%** against a **10.65%** k=1 readability floor: **0.64pp of
headroom**. It now plays the role CZ `wind_onshore` plays for the approved set
(ABL-316 ledger §15.2). **If any future correction moves the k=1 floor, SE
`solar` is the first pair to be withdrawn.** The floor is `c_B = 0` at k = 1
because the references are deterministic; at k > 1 the readability test becomes a
Student-t interval on the cell's own seed draws, and this pair was read at k = 1.

**Caveat 2 — all three wind pairs clear a bar weaker than a flat line.**
`bar_weaker_than_a_flat_line` is true for HR (2 of 3 bands), HU (3 of 3) and PL
(3 of 3) under the registered convention. Seasonal-naive D-7 is an *easier*
reference than a constant on those countries, so a large G1 margin predicts less
than its size suggests (ledger §5.1: a weak bar predicts a pass). Their D-7
margins are real. This does not block shipping under the adopted standard; it has
to be stated, and it is the reason G2/G3 carry the weight they do above.

---

## 4. The artifacts

Machine record: `reports/abl_602_ship_set_training.json`, generated
2026-08-28T12:05:25Z under `.venv\Scripts\python.exe` (Python 3.14.3), replica
`C:\Code\able\data\energy_dashboard.db` at 10,718,515,200 bytes.

| pair | algo | features | fit rows retained / intended | unique targets | build s | fit s | artifact sha256 |
|---|:---:|:---:|---:|---:|---:|---:|---|
| `LT` `solar` | catboost | 27 | 41,724 / 42,816 | 5,247 | 191.8 | 4.0 | `c48696a9bff5f9c0d08e8209e3860e5978ab0feed237645cfac360dbff27e421` |
| `SE` `solar` | catboost | 27 | 42,320 / 42,816 | 5,322 | 198.0 | 6.3 | `8d359c3feb7ab312ea0a9e221180016d011aa979fbd52ac0f047e4177d2b97d3` |
| `HR` `wind_onshore` | catboost | 24 | 42,384 / 42,816 | 5,328 | 62.9 | 4.2 | `d802a9612622ad6eeaef796c6e7ac3efca889219103d81926abd1aee058dc22f` |
| `HU` `wind_onshore` | catboost | 24 | 42,384 / 42,816 | 5,328 | 63.1 | 3.9 | `1a531989bacfd0b8bb8e365e67b4069a2f9a005399a9d36dc98acff74199188b` |
| `PL` `wind_onshore` | catboost | 24 | 42,384 / 42,816 | 5,328 | 63.3 | 4.0 | `26947d53695235caa8ac07565586c3b975424f3eff70b25f66ea7658f28efa69` |

| pair | retained target window | in-sample mean MW | rows excluded (missing actual or feature) |
|---|---|---:|---:|
| `LT` `solar` | 2026-01-12 00:00 → 2026-08-21 18:00 | 345.79 | 1,092 |
| `SE` `solar` | 2026-01-12 00:00 → 2026-08-21 23:00 | 441.72 | 496 |
| `HR` `wind_onshore` | 2026-01-12 00:00 → 2026-08-21 23:00 | 320.89 | 432 |
| `HU` `wind_onshore` | 2026-01-12 00:00 → 2026-08-21 23:00 | 61.10 | 432 |
| `PL` `wind_onshore` | 2026-01-12 00:00 → 2026-08-21 23:00 | 2,103.85 | 432 |

The retained windows start **2026-01-12** for all five, which measures rather than
assumes the weather bound for these countries: ABL-583 re-measured
`weather_data data_quality='forecast'` beginning 2026-01-11 for CZ, RO and NL, and
LT, SE, HR, HU and PL are not in that measurement. Their retained windows land one
target-day after `FIT_START`, which is the same bound. Retention is 97.4–99.0% of
intended rows.

`FIT_START` / `FIT_END` are unchanged at `2026-01-11` / `2026-08-22` — the ship8
window — so this batch is homogeneous with the deployed set and
`abl525_repro_check.py` cannot report a window change as a drift.

**The in-sample mean is in-sample.** It is recorded because it is the level
witness that catches an xgboost-intercept reset on a wrong-interpreter load; it
is not a performance number.

---

## 5. Reproducibility — proved on predictions, not on the hash

Machine record: `reports/abl_602_reproducibility.json`.

`Forecaster.save` stamps `saved_at`, so an artifact digest cannot witness a
refit: three byte-identical fits give three digests. The witness is what the two
artifacts **predict**. Protocol (ABL-580's, unchanged): refit each pair through
the same `fit_one` into a scratch directory; load original and refit through
`Forecaster.load` — the entry point `forecast_daily.py` uses, so both have
actually round-tripped; build **one** shared feature matrix and predict with
both; assert `max |a − b| < 1e-12`.

<!--TABLE:REPRO-->

---

## 6. Serving verification — both horizons, from the sidecar

Machine record: `reports/abl_602_serving_verification.json`.
Script: `scripts/abl602_serving_verification.py`.

**This is read out of the sidecar the serving run wrote, not inferred from the
training log.** Two invocations, both with `FORECAST_OUTPUT_DB` pointed at a
run-scoped scratch file — **neither the replica nor the shared sidecar was
written**:

```
scripts/forecast_daily.py --countries LT,SE    --types solar        --horizons 1,2
scripts/forecast_daily.py --countries HR,HU,PL --types wind_onshore --horizons 1,2
```

Both runs reported `Success: 4/4` and `Success: 6/6`, `Failed: 0`, `Skipped: 0`,
240 rows total. Every model loaded from `C:\Code\able\ef-abl602\models\…`.

| pair | rows | target days | horizon h | min MW | max MW | mean MW | negatives | exact zeros | artifact = training record |
|---|---:|---|---|---:|---:|---:|---:|---:|:---:|
| `LT` `solar` | 48 | 08-29, 08-30 | 9–56 | 0.0 | 923.41 | 295.15 | 0 | 14 | **yes** |
| `SE` `solar` | 48 | 08-29, 08-30 | 9–56 | 0.0 | 1151.60 | 311.35 | 0 | 18 | **yes** |
| `HR` `wind_onshore` | 48 | 08-29, 08-30 | 9–56 | 82.40 | 472.72 | 261.69 | 0 | 0 | **yes** |
| `HU` `wind_onshore` | 48 | 08-29, 08-30 | 9–56 | 7.91 | 51.47 | 28.57 | 0 | 0 | **yes** |
| `PL` `wind_onshore` | 48 | 08-29, 08-30 | 9–56 | 698.21 | 3392.81 | 1794.61 | 0 | 0 | **yes** |

**5 / 5 verified.** "Verified" is the conjunction of four things, not a row
count: both horizons produced a full 24 hours on their own target day; the
artifact on disk re-hashes to the sha256 in the training record **and**
deserialises to the same `feature_columns` and `training_source`; the clamp
behaved as its registration says it should; and every served value is finite.
The digest join is what makes the served model provably the graded one, given
that `models/` is gitignored and no commit protects these files.

### 6.1 The clamp

| pair | clamp row | night hours | zeroed | raised to 0 | MW removed at night | pre-clamp min MW | post-clamp min MW |
|---|:---:|---:|---:|---:|---:|---:|---:|
| `LT` `solar` | yes | 14 | 14 / 14 | 0 | +87.38 | +1.25 | 0.0 |
| `SE` `solar` | yes | 12 | 7 / 12 | 6 | **−80.68** | **−17.26** | 0.0 |
| `HR` `wind_onshore` | none — correct | — | — | — | — | — | — |
| `HU` `wind_onshore` | none — correct | — | — | — | — | — | — |
| `PL` `wind_onshore` | none — correct | — | — | — | — | — | — |

`night_generation_possible` is `false` and `night_mask_applied` is `true` for
both solar pairs, so the mask ran rather than being skipped. A wind pair has no
clamp row because `src/solar_clamp.py` is `renewable_type='solar'` only; the
absence is the correct outcome and is asserted as such.

**`SE` `solar` predicts negative at night.** Its pre-clamp minimum is −17.26 MW,
and the clamp *added* 48.54 MW back across 6 hours while zeroing 7 of 12 night
hours — a net **−80.68 MW "removed"**, i.e. net added. This is the known
27-feature solar night behaviour, not a defect introduced here, and the clamp is
the choke point that stops it reaching a consumer: **no served row is negative
for any of the five pairs.** LT is the clean contrast on the same fit: all 14
night hours positive pre-clamp and zeroed by the mask.

These clamp figures are a **snapshot of the firing hour, not a property of the
artifact.** `predict_d2` anchors its feature build on `datetime.now()` as well as
on the reference date (measured on ABL-583, where two runs 85 minutes apart
disagreed on every night field at identical artifact, replica and reference
date). Compare them only against a same-hour run.

### 6.2 Two honest notes on this run

- **The served horizons are not the gate's horizon bands.** This run fired at
  14:05 UTC, so its `horizon_hours` span 9–56h; the production 08:00 job spans
  roughly 16–63h and the gate scored 24–64h. Serving verification establishes
  that the pipeline serves these five correctly end to end. It does not
  re-establish anything about the 24–64h bands, and is not a measurement of
  accuracy at any horizon.
- **`forecast_runs` is absent from the scratch sidecar**, so run tracking logged
  `Could not start run tracking` and degraded. That is a property of a
  freshly-created scratch file, not of the serving path; forecast rows and clamp
  telemetry both wrote normally.

---

## 7. Contamination

Machine record: `reports/abl_602_contamination_screens.json`. Read-only against
the replica; screens the **fit** window (223 days), which is wider than the
gate's registered 178 and is the window no existing record covers.

| screen | LT `solar` | SE `solar` | HR `wind` | HU `wind` | PL `wind` |
|---|---|---|---|---|---|
| **ABL-332** sub-hourly | quarter-hourly; 20,993 → 5,274 h | quarter-hourly; 21,400 → 5,350 h | 21,404 → 5,352 h | 21,408 → 5,352 h | 21,408 → 5,352 h |
| **ABL-188** constant runs nulled | 0 | 0 | 0 | 0 | 0 |
| **ABL-439** fit−gate ratio discontinuity | 0.0000 | 0.0000 | −0.0007 | +0.0070 | 0.0000 |
| verdict | basis-consistent | basis-consistent | basis-consistent | basis-consistent | basis-consistent |

**All five pairs are quarter-hourly**, and the pre-ABL-332 builder would have
discarded 15,719–16,056 rows per pair by reading only the `:00` sub-sample. The
hourly-mean aggregation is in both the fit and the serve path — the same builder
object — so there is no train/serve skew here.

**ABL-200** cannot fire for this batch: the cross-table zero disproof is wired
behind `source != RENEWABLE_ZERO_DISPROOF_SOURCE`, and every pair here reads
`energy_generation`, which *is* that source.

**ABL-439 vintage:** every discontinuity is at or under 0.007 against a 0.02
threshold. Recorded caveat, carried verbatim: each pair's gate window is 100%
first-publication, so its gate-side revision is expected-small but **not yet
measured** — an unmeasured quantity, not a measured zero.

**The four named contamination issues do not touch this window or these pairs:**
ABL-71 (prod ingest stale) and ABL-67 (fabricated rows) are `net_position`;
ABL-111 / ABL-109 (zero-as-missing) are actual **load**. None is a renewable
generation series.

### 7.1 Night floor — the solar pairs against the BG signature

| | night hours > 1 MW | night mean MW | night max MW | % of total energy at night | WAPE floor if clamped |
|---|---:|---:|---:|---:|---:|
| `LT` `solar` (whole fit window) | 0.88% | 0.16 | 24.29 | **0.014%** | 0.014% |
| `SE` `solar` (whole fit window) | **35.16%** | 0.89 | 4.80 | **0.054%** | 0.054% |
| `SE` `solar` (gate window and after) | **95.18%** | 1.66 | 3.61 | 0.044% | 0.044% |
| `BG` `solar` (ABL-396, for contrast) | 76.4% | 152.33 | 1097.4 | **6.37%** | — |

**`SE`'s hour-count screen fires and its energy screen does not, and the second
is the one that matters.** 95% of SE's night hours in the gate window book more
than 1 MW — on the count alone that is worse than BG. But SE's night *mean* is
1.66 MW against BG's 245.71, its night maximum is 3.61 MW against BG's 1,087.9,
and night energy is **0.044% of the total against BG's 4.98%** — two orders of
magnitude apart. SE carries a small constant metering floor, not BG's
displacement. The clamped-WAPE floor it implies is 0.044%, against SE's 0.64pp
of D-7 headroom; it cannot move the letter. Both numbers are stated because
quoting only the reassuring one is how ABL-396's BG case was nearly missed in
reverse.

LT is clean on both screens. Neither solar pair has a single negative night
actual in the fit window.

---

## 8. What this pack does not establish

1. **Nothing here is out-of-sample.** The fit window covers the gate window, so
   every skill figure in section 3 was measured on rows these artifacts were
   fitted on. That is what "full available history" asks for and is right for
   production, but it means this pack contains **no holdout measurement of the
   models being shipped**.
2. **No gate was re-read and nothing was re-graded.** Every letter and margin is
   read out of a committed record. Under the anti-churn directive, a correction
   is queued on the epic rather than run as a fresh read; section 3.3 is a read
   of `abl_444_g23_floor_reread.json`, not a re-scoring.
3. **Serving verification is not accuracy.** It proves the five serve end to end
   at both horizons from the artifacts this pack fitted. It measures no error.
4. **The artifacts are not protected by a commit.** They live at
   `C:\Code\able\ef-abl602\models\<country>\<type>\model.joblib` in a gitignored
   directory. The deploy issue must copy them to
   `C:\Code\able\energy-forecast\models` — which is where the scheduled job
   resolves `config.MODELS_DIR` — and **re-check each sha256 against section 4
   after copying**. A missing artifact makes the rail log "no trained model",
   write nothing for that pair, and still exit 0.
5. **The HU question is open and is the CEO's.** Section 3.3 is a correction to
   the issue's reading, not a decision. HU is fitted, verified and ready; whether
   it ships is a disposition on ABL-316.
6. **`SE` `solar` is the withdrawal candidate.** 0.64pp of headroom at k = 1. Any
   future move in the readability floor should be checked against it first.

---

## 9. Reproducing this pack

```bash
# All four steps under .venv, from the repo root, with ENERGY_DB_PATH set.
python scripts/abl525_train_ship_set.py       --batch abl602 --replica-db <replica> --models-dir <dir>
python scripts/abl525_repro_check.py          --record reports/abl_602_ship_set_training.json \
                                              --json-out reports/abl_602_reproducibility.json
python scripts/abl580_contamination_screens.py --batch abl602 --replica-db <replica>
# then the two forecast_daily invocations in section 6, into a scratch sidecar, and:
python scripts/abl602_serving_verification.py --sidecar <scratch.db> \
                                              --record reports/abl_602_ship_set_training.json
```

`tests/test_abl602_widened_batch.py` (19 tests) holds the batch membership, both
feature-list equalities, the per-scope convention pins, the HU correction and the
serving verifier's clamp/horizon contracts. It opens no database.
