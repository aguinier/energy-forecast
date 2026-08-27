# ABL-581 — Pre-registration: CH `solar` gate read at 27 features

**Status: registered. Committed before any fit exists.**

Scope id `abl581-ch-solar-f27`, in `scripts/evaluate_solar_retrain.py`. This
document and the harness registration land in the **same commit**, and that commit
is strictly earlier than the one carrying the read. Registering after seeing
numbers is the shape of shopping a registration; the ordering is the only thing
that makes the difference visible to a reviewer, so it is stated here and
checkable in `git log`.

## 1. Why this exists

CH `solar` was pair 8 on the Board's approved `ship8` roster. The CEO withdrew it
on 2026-08-27 (ruling on ABL-525) on two grounds:

- **At 27 features** — the class nobody had graded under the amendments now in
  force. Tranche 1b graded CH `solar` at the legacy **25**-name list; ABL-395 moved
  `solar.FEATURE_COLUMNS` to 27. An artifact pins its own `feature_columns`, so the
  approved class does not reach a 27-feature CH artifact.
- **At the pinned 25** — a per-country serving fork on a list no other artifact
  uses. ABL-525 item 2 forbids it in terms; ABL-401 §4 settles the direction
  (everything from ABL-395 forward is fitted at 27).

The standing rule the Board approved lets a withdrawn pair rejoin by satisfying
the rule again, **with no new Board card**. This read is CH's chance to do that.

**This is a new pre-registration, not an amendment.** ABL-401 §1–§2: a read at a
different feature list sits *beside* the published path and never overwrites it.
`SCOPE_FEATURES['abl316-t1b']` is untouched, and ABL-381's published PASS 6/6
stands.

## 2. What is registered

| item | registered value | where |
|---|---|---|
| scope id | `abl581-ch-solar-f27` | `SCOPES` |
| countries | `("CH",)` — 1 country × 3 primary bands = **3 cells** | `SCOPES` |
| forecast type | `solar` | harness |
| algorithm | `catboost`, `iterations=500, depth=8, learning_rate=0.05, l2_leaf_reg=3` | `config.get_default_params` |
| seeds | **k = 1**, `random_seed = 42` (fixed in `config.py`; this harness has no `--seeds` flag) | `config.py` |
| fit target window | `2026-01-14T00:00:00Z` → `2026-07-11T00:00:00Z` (exclusive) | ABL-348 |
| gate target window | `2026-07-11T00:00:00Z` → `2026-08-10T00:00:00Z` (exclusive) | ABL-348 |
| feature lookback start | `2025-12-31T00:00:00Z` | ABL-348 |
| source table | `energy_generation`, **elected in the file** | `SCOPE_SOURCES` |
| gate basis | `("challenger", "seasonal_naive")` | `GATE_BASIS` |
| fit rule | `exclude_impossible_night: False` | `FIT_RULES` |
| metric | WAPE | ABL-348 |
| baseline (the bar) | literal seasonal-naive D-7, recomputed on the same table the challenger is scored against | ABL-348 |
| primary bands | `24-36h`, `36-48h`, `48-64h` | ABL-348 |
| registered minimum n | `24-36h: 684`, `36-48h: 684`, `48-64h: 456` | ABL-348 |
| registered intended n | `24-36h: 720`, `36-48h: 720`, `48-64h: 480` | ABL-348 |
| causal levelling | `trailing_28d` (**ABL-437's amendment**) | `CAUSAL_LEVELLING` |
| G2/G3 readability | `floored` (**ABL-444's amendment**) | `G23_READABILITY` |
| k>1 readability | `delta_min` — inert at k = 1, registered so the record says which test decided the letter | `SEED_READABILITY` |
| readability floor | 10.6482% (solar, k = 1) | `src/evaluation/gate_grading.py` |
| CH's pre-committed D-7 bar | **12.67%** WAPE (MAE 168.7 MW, mean actual 1331.0 MW, `n_d7_scorable` 720), measured 2026-08-12 before any challenger existed for the pair | ABL-348 `per_pair_bar_measured_before_any_challenger_exists` |

### The feature list — 27 names, registered by absence

`SCOPE_FEATURES` carries **no row** for this scope, deliberately, and that is how
the 27 is inherited (`DEFAULT_SCOPE_FEATURES = FEATURE_COLUMNS`). Consequently
`meta.feature_set_is_registered_for_scope` reads **False**, exactly as it does for
`abl316-t2a`, `abl316-t2c` and `abl316-t2a-generation`. That is correct and
intended, not a defect: the only honest row available is `FEATURE_COLUMNS`, which
binds to the same mutable constant the default already binds to, and the one real
literal pin in reach — `LEGACY_FEATURE_COLUMNS` — is the 25-name list this scope
exists to move off. The witness for the 27 is `meta.feature_columns`, the literal
names the run writes:

```
hour, day_of_week, month, is_weekend, hour_sin, hour_cos, day_sin, day_cos,
month_sin, month_cos, target_value_lag_1d, target_value_lag_7d,
target_value_lag_14d, target_value_roll_24h_mean, target_value_roll_24h_std,
target_value_roll_24h_min, target_value_roll_24h_max,
target_value_roll_168h_mean, target_value_roll_168h_std,
target_value_roll_168h_min, target_value_roll_168h_max,
shortwave_radiation_wm2, direct_radiation_wm2, diffuse_radiation_wm2,
temperature_c, sun_elevation_deg, is_night
```

The last two are ABL-338's geometry features, adopted by ABL-395 — the 25-name
list is this list minus exactly those two.

Note the deliberate asymmetry: the **source** is registered
(`meta.source_is_scope_registered == true`) while the **feature list** is not.
Both are right, for the reasons written beside their own tables.

## 3. Output paths

Registered in `SCOPE_OUTPUTS`, disjoint from every other scope's:

| output | path |
|---|---|
| `artifact_dir` | `experiments/ABL581/artifacts` (gitignored by `.gitignore:56`, which matches the directory name) |
| `json_out` | `experiments/ABL348/results_abl581_ch_solar_f27.json` (**tracked** — one level deep and not named `results.json`) |
| `report_out` | `reports/abl_581_ch_solar_f27.md` |

Nothing here is a path `abl316-t1b`, `abl316-t2a`, `abl316-t2c`, `abl316-t2d` or
`abl316-t2a-generation` writes. This matters more than usual: four of those name CH
too, and `experiments/ABL316/artifacts/CH/solar/model.joblib` is the **25-feature**
artifact whose SHA-256 ABL-381's machine record cites. Sharing that directory would
overwrite a Board-approved record with nothing in `git status` to show it, because
both directories are ignored. `check_scope_outputs` enforces the disjointness at
import.

## 4. References reported beside the read

All four model-free references come from the single implementation in
`src/evaluation/model_free_reference.py` — never a second implementation, because
two reads that compute a named reference differently are not comparable:

- `constant_causal` and `climatology_causal` — levelled on the **trailing 28 days**
  (ABL-437). These are what G2 and G3 read.
- `constant_oracle` and `climatology_oracle` — hindsight forms.
- seasonal-naive D-7 — **the registered bar**, and the only performance criterion.
- TSO — **context only, never a criterion**, under the standing Board directive of
  2026-08-14 item 1.

Every band prints its `n` beside its `minimum_n` and its `enough_pairs` flag
(ABL-434), so no grade can be read as passing vacuously on a coverage-short cell.

## 5. The pass rule, and the bar for rejoining — fixed before the numbers

**Harness pass rule (ABL-348, unchanged):** challenger WAPE < literal seasonal-naive
D-7 WAPE in all three primary bands, each cell meeting its registered minimum n.

**The rejoin bar (CEO, stated on ABL-581 before this read).** CH `solar` rejoins
the shipping set — with no new Board card, reported by comment on ABL-316 — if the
read shows **all four** of:

1. no readable loss to either oracle reference (outside the ABL-444 floor,
   10.6482% for solar at k = 1);
2. clears the registered seasonal-naive D-7 bar readably;
3. every gating band over its `minimum_n`, so the grade is not vacuous (ABL-434);
4. no data-contamination hold.

**If it misses, it misses.** The numbers get reported and the CEO records the
exclusion. No tuning toward the bar, and no second look: the fit is deterministic
at seed 42, so a re-run is a reproduction and not another draw.

## 6. Contamination screens this read must report

- **ABL-188** constant-run screen on the fitted series, on `energy_generation` (the
  table the model is fitted and scored on, per ABL-345) — run by the harness,
  printed in the pack.
- **ABL-200** sibling-disproof of exact zeros, applied inside
  `db.load_renewable_type_data`.
- **ABL-348 §contamination** as it applies to CH: ABL-67 and ABL-109/ABL-111 are
  `net_position` and `load` respectively and do not intersect a solar target;
  ABL-71 is a provenance caveat on ingest generally, not proof that solar ingest is
  pristine. None of ABL-348's five recorded ABL-188 hits inside the registered
  windows is CH.
- **Night behaviour.** CH is `NIGHT_GENERATION_POSSIBLE = False` with a registered
  representative point, so ABL-337's serve-path clamp zeroes every night hour. A
  busy clamp is itself the signal that a fit is wrong, so the night
  negative-prediction rate and night MAE/bias are reported as **corroboration, not
  criteria** — and per `CLAUDE.md` a night-floor change is never dispositioned on
  the negative-prediction rate, whose single-seed null (14.06pp) is wider than the
  effects it is quoted for.

## 7. What voids this registration

The ABL-348 list applies unchanged — a change to the windows, the metric, the
baseline, the minimum n, or the source table. Two additions specific to this scope:

- Reading it with an explicit `--renewable-source` that is not
  `energy_generation`. That is permitted and still runs, but comes back
  `meta.source_is_scope_registered == false` and prints OFF-REGISTRATION, and an
  off-registration read **cannot support a membership change**.
- Any edit to `SCOPES`, `GATE_BASIS`, `SCOPE_SOURCES`, `FIT_RULES`,
  `CAUSAL_LEVELLING`, `G23_READABILITY` or `SEED_READABILITY` for this scope after
  the read commit. The registration commit is the one that counts.

## 8. Prior CH reads this one sits beside, and does not replace

| read | features | source | levelling / G2-G3 | status |
|---|---|---|---|---|
| ABL-381 `abl316-t1b` | 25 | `energy_generation` | `fit_window` / `sign_test` | published PASS 6/6; CH graded **A**. Untouched. |
| ABL-405 `abl316-t2a` | 27 | `energy_renewable` (off-registration; ABL-426 sized it) | `fit_window` / `sign_test` | published; the source defect is recorded, not withdrawn. |
| ABL-426 `abl316-t2a-generation` | 27 | `energy_generation` | `fit_window` / `sign_test` | published. **Holds a 27-feature CH read already.** |
| **ABL-581 `abl581-ch-solar-f27`** | **27** | **`energy_generation`** | **`trailing_28d` / `floored`** | this read. |

ABL-426's CH cells are strong corroboration — same features, same table, same
windows — but they are not the read the rejoin bar asks for, because that scope is
pinned to `fit_window` levelling and a `sign_test` on G2/G3 in order to be a
controlled A/B against `abl316-t2a` on the source table alone, and the bar is
stated over the ABL-437 levelling and the ABL-444 floor. The two are reported side
by side in the evidence pack; the difference between them is two grading
amendments and one later replica snapshot, and nothing else.

## 9. Not in scope

No production artifact, no serving change, no promotion. This produces **evidence
only**. `--include-held` is not run here: training is a follow-up the CEO files,
the same split as ABL-525 and ABL-580. Promotion is a pre-registered gate read plus
a decision that is not mine to take.
