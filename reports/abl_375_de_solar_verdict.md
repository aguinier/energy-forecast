# ABL-375 — DE solar, XGBoost vs the serving CatBoost configuration: verdict

**Verdict on the pre-registered primary: AMBIGUOUS. This read does not license
moving DE solar to XGBoost.** The reason is not that XGBoost lost — it won by
4.5% — but that DE CatBoost's daylight MAE moves 13.8% of its own mean across
three seeds on the registered window, so a 4.5% gap is not separable from the
spread of a single arm against itself.

The finding underneath is bigger than the verdict: **every number this issue was
filed on was a single fit.** ABL-338 ran one seed per arm, and nobody had
measured what a second seed does. On DE it does a lot.

Registration: `experiments/ABL375/config.json`, committed at `b092efe`
(2026-08-13 10:24:37 +02:00) before the first fit. Confirmatory read committed at
`d990fcf` (10:30:38) before the post-hoc seed work existed. Amendments to the
registration are in its `corrections_after_the_fit` block, added rather than
edited in place.

---

## 1. Protocol

| | |
|---|---|
| Registered holdout | **2026-04-30 .. 2026-06-12**, 1,056 hours, inclusive |
| Why not ABL-348's windows | its gate window 2026-07-11 .. 2026-08-10 lies wholly inside ABL-338's summer holdout, on which **both arms of this comparison were already fitted and scored**. Registering a window whose challenger scores are known is not a pre-registration. Its pairs (37 `energy_generation` pairs) and its metric (WAPE on D+2 horizon bands — meaningless at night, where the denominator is ~0) do not transfer either. |
| Why this window | the gap between ABL-338's two committed holdouts. All seven of those windows are enumerated with a non-overlap flag in `reports/abl_375_probe.json`, so "unread" is checkable, not recalled. |
| Fit window | every featured row strictly before the holdout start, per country. DE: 2025-11-26 .. 2026-04-29, n_train 3,751. |
| Control | **a refit on the identically truncated window**, never the live artifact. The live artifacts are fitted through roughly today; scoring them here would be in-sample. |
| Arms | 2 algorithms × {control, geometry} × 3 seeds × 4 countries = 48 fits |
| Geometry | on **both** arms. `src/features.py` on `origin/main` appends it to every solar fit unconditionally, so the geometry arm is what a routine retrain actually produces. |
| Hyperparameters | each algorithm at its own `config.get_default_params()`. **Measured:** all four serving solar artifacts carry hyperparameters field-identical to config's defaults for their algorithm, so the CatBoost arm is the serving configuration refitted rather than a stand-in. Both sides get depth 8, lr 0.05, 500 trees, 50-round early stopping on the same chronological validation split. |
| Bands | `src/solar_features` — night is the serving clamp's own predicate; shoulder is sun ≤ 0° at the hour midpoint; daylight is the rest. Night in MW only. |
| Out-of-sample | yes, every cell. No arm is scored on data it was fitted on. |
| Interpreter | `.venv\Scripts\python.exe`, Python 3.14.3 / xgboost 3.3.0. Replica `C:\Code\able\data\energy_dashboard.db`, 9,432,453,120 bytes, `mode=ro` — the live replica, not the 3.0 GB partial snapshot. |

### What this protocol cannot say

- **Not serve-faithful.** Features come from the training-time pipeline, whose
  lags are anchored at the target hour; at serving they are anchored at the
  generation instant (ABL-183). Every arm carries that identically, so the
  *comparison* holds and the absolute MW are optimistic against the rail. The two
  geometry features are exempt by construction.
- **No horizon bands**, so this cannot substitute for a gate-harness read. It
  answers "which algorithm fits DE solar better", not "does DE solar clear its
  promotion gate".
- **Neither arm is the serving feature set.** `_legacy_feature_columns()` returns
  **29** names, not the 25 the ABL-338 docstring and my own registration claimed:
  four holiday features (`is_holiday`, `days_to_holiday`, `days_from_holiday`,
  `is_bridge_day`) entered `get_feature_columns('solar')` after the serving
  artifacts were fitted. ABL-338's committed runs used the same 29/31, so these
  numbers stay directly comparable to its. But no result here can be phrased as
  "beats the serving artifact" — it beats the serving *configuration* on this
  repo's current feature list.

### Contamination

| issue | touches this window? | handling |
|---|---|---|
| **ABL-337** impossible night actuals | **yes** | dropped from the fit, never from the score, identically on every arm. Fit-window rows above 1 MW: AT 0, BE 0, **DE 4 (max 1.7 MW)**, **FR 464 (max 439.3 MW)**. Holdout: DE **0** — so DE night MW is measured against a true zero. FR **24 rows up to 251 MW**, which no arm can or should predict, so FR night MW carries a contamination floor rather than model error. |
| ABL-188 constant-run screen | applied by `db.load_renewable_type_data` to whatever table is read, so identical on both arms by construction | source is `energy_renewable` for all four countries, read off the artifacts |
| ABL-67 | no — net position only | — |
| ABL-109 / ABL-111 | no — load only | — |
| ABL-71 | provenance caveat | known wrong-write modes are load and net position; not proof solar ingest is clean |

---

## 2. The registered read

Seed mean (min–max) over seeds 42 / 1337 / 2718. Daylight MAE, MW.
Free baseline is literal seasonal-naive D-7 on the same rows.

| country | daylight n | D-7 bar | CatBoost+geom | XGBoost+geom | gap | seed spread cat / xgb | ranges | verdict |
|---|---:|---:|---:|---:|---:|---:|---|---|
| AT | 670 | 781.9 | 618.9 (589.1–637.5) | **496.2** (494.0–499.9) | +19.8% xgb | 7.82% / 1.18% | disjoint | **PASS** |
| BE | 681 | 1,425.5 | **572.2** (561.6–581.9) | 606.1 (597.5–622.1) | −5.9% cat | 3.55% / 4.05% | disjoint | **FAIL** (CatBoost confirmed) |
| **DE** | 686 | 6,509.3 | 4,449.7 (4,224.5–4,838.2) | 4,249.6 (4,161.8–4,362.6) | +4.5% xgb | **13.79%** / 4.73% | **overlapping** | **AMBIGUOUS** |
| FR | 656 | 2,128.1 | 1,001.6 (996.6–1,004.8) | 974.8 (961.1–997.9) | +2.7% xgb | 0.82% / 3.77% | overlapping | **AMBIGUOUS** (reversal) |

The four registered DE conditions, as read:

| condition | result |
|---|---|
| seed-mean favours xgboost | **pass** (4,249.6 < 4,449.7) |
| strict seed non-overlap, max(xgb) < min(cat) | **fail** — 4,362.6 > 4,224.5 |
| gap ≥ 3.0% | pass (4.5%) — but the threshold is below CatBoost's own 13.79% spread, which is what condition 2 exists to catch |
| night guardrail, xgb night mean ≤ cat's | **pass as registered** (−44.5 ≤ 230.7 MW) — and see §5: it passes on a *negative* value, which is a defect in how I wrote the guardrail, reported rather than rewritten |

Every arm beats the D-7 floor comfortably; DE's best cell (xgboost+control, 3,914.6 MW) sits 39.9% below it.

### Both counter-cases, as the registration required

- **BE confirms the expectation.** CatBoost wins by 5.9%, ranges disjoint. BE
  stays on CatBoost.
- **FR reverses it.** The registration predicted CatBoost for FR, on the strength
  of ABL-338's summer single-seed read (CatBoost 1,285.1 vs XGBoost 1,468.6 MW,
  −14.3% on the convention used here). On the registered window FR favours
  **XGBoost** by 2.7%, and on the control arm by 3.7%. The gap is inside the
  noise, so the verdict is AMBIGUOUS, not a flip — but the *direction* moved 17
  points between windows. FR's algorithm preference is window-dependent, which is
  a finding about the per-country policy rather than about FR.
- **AT is decisive and consistent** with it already serving XGBoost (+19.8%,
  disjoint), which is a useful sanity check on the whole protocol: the one
  country whose answer we already know comes out right.

---

## 3. The seed spread is the actual result

Nothing in the ABL-338 lineage had measured it. Once measured on DE:

| window | status | CatBoost+geom spread | XGBoost+geom spread | gap | ranges |
|---|---|---:|---:|---:|---|
| registered, 2026-04-30 .. 06-12 | **CONFIRMATORY** | **13.79%** | 4.73% | +4.5% | overlapping |
| summer, 2026-06-13 .. 08-11 | already observed, post-hoc seeded | 6.98% | 4.03% | **+13.4%** | disjoint |
| spring, 2026-03-01 .. 04-29 | already observed, post-hoc seeded | 4.62% | 4.12% | **+14.3%** | disjoint |

The 3.0% materiality threshold I registered — twice ABL-338's ~1.5% perturbation
estimate — is **too small on every one of the three windows.** The real floor for
DE CatBoost is 4.6–13.8%. ABL-338's 1.5% was estimated by perturbation on a
different question and does not survive contact with a reseeded fit.

Concretely, on the number this issue was filed on. **DE loses 10.7% daylight MAE
to XGBoost** is `3,614.6 → 3,227.6 MW` — and locating it exactly matters: it is
the **control** arm (no geometry), on the **uncleaned** fit (ABL-337 rows left
in), summer window, one seed per side. Its two nearest seeded equivalents, both on
that same summer window:

| arm set | CatBoost | XGBoost | gap | ranges |
|---|---:|---:|---:|---|
| control, cleaned, 3 seeds | 3,633.0 (3,504.5–3,852.2) | 3,199.0 (3,173.3–3,238.6) | **+11.9%** | disjoint |
| geometry, cleaned, 3 seeds | 3,602.1 (3,443.5–3,694.8) | 3,120.9 (3,076.3–3,202.2) | **+13.4%** | disjoint |

So on summer the effect is *larger* than the 10.7% reported and it does survive
its own noise, on both feature sets. It is on the registered window that it does
not.

**This is a genuine three-window disagreement, not a refutation.** DE's XGBoost
advantage is 13.4% (summer) and 14.3% (spring), both seed-disjoint, and 4.5%
(registered late spring), overlapping. The registered window is also where DE
CatBoost is most unstable, and DE's 156-day fit window is the shortest of the four
countries — its `energy_renewable` solar history begins 2025-09-08. A 13.8%
seed spread on a ~3,700-row fit is what an underdetermined fit looks like.

---

## 4. Bands, and the footgun this issue was meant to defuse

DE, all three windows, seed means. Night and shoulder in MW.

| window | arm | daylight | shoulder | night mean pred |
|---|---|---:|---:|---:|
| registered | catboost+control | 4,463.5 | 308.5 | 212.8 |
| registered | catboost+geometry | 4,449.7 | 317.2 | 230.7 |
| registered | xgboost+control | **3,914.6** | 54.0 | −38.2 |
| registered | xgboost+geometry | 4,249.6 | 59.4 | −44.5 |
| summer | catboost+control | 3,633.0 | 215.2 | 90.2 |
| summer | catboost+geometry | 3,602.1 | 379.9 | **306.2** |
| summer | xgboost+control | 3,199.0 | 182.5 | 57.0 |
| summer | xgboost+geometry | **3,120.9** | **77.3** | **19.1** |
| spring | catboost+control | 10,367.8 | 345.8 | 213.3 |
| spring | catboost+geometry | 9,069.0 | 341.9 | 212.7 |
| spring | xgboost+control | 7,903.9 | 185.6 | 63.8 |
| spring | xgboost+geometry | **7,770.6** | 158.2 | 104.9 |

**The recorded footgun, re-measured.** The issue cites geometry making DE worse
under CatBoost at night, 220.1 → 453.8 MW. Both of those are seed 42. At three
seeds the summer figure is **90.2 → 306.2 MW** — the direction holds and is
substantial (3.4×), but the control end especially was a single unlucky draw: two
of its three seeds read ~25 MW, one reads 220 MW. On spring (213.3 → 212.7) and
on the registered window (212.8 → 230.7) the geometry delta is **flat**. So the
geometry-specific hazard is a summer phenomenon, as I reported on this issue
before the seeds existed, and its magnitude was overstated by the single draw.

**What is consistent across all three windows is the level, not the delta.**
DE-on-CatBoost sits at a 213–306 MW night floor and a 308–380 MW shoulder MAE;
DE-on-XGBoost sits at −45 to +105 MW night and 54–182 MW shoulder. That gap is
present on every window, under both feature sets, and it is the stronger argument
for the hold than the geometry delta ever was.

**Geometry's effect on XGBoost DE also disagrees by window**: −2.4% daylight on
summer, −1.7% on spring, **+8.6% worse** on the registered window. ABL-338's
shoulder halving reproduces as a seed mean on summer (182.5 → 77.3 MW, −58%) and
weakens elsewhere (spring −15%, registered +10%).

---

## 5. Two defects in my own registration, reported not repaired

1. **`feature_sets` said 25/27 names; the fitted arms carry 29/31.** Cause and
   consequence in §1. It does not change any arm that was fitted — the arm was
   defined by construction and that construction is what ran — but it does mean
   no claim here is against the serving *feature list*.
2. **The night guardrail is one-sided.** I registered "xgboost night mean ≤
   catboost night mean", which a large **negative** value satisfies. DE
   xgboost+geometry reads −44.5 MW with up to 267 negative predictions, and
   passes a guardrail it should not obviously pass. Rewriting it after seeing the
   number would void the registration, so it is reported as registered with the
   negative level beside it. A future registration should use `|night mean|` or
   the serving clamp's own counters. Night safety in production comes from
   `src/solar_clamp.py` (ABL-337) on either algorithm, not from the fit.

---

## 6. Recommendation

**Do not promote DE solar to XGBoost on this read.** The pre-registered primary
is AMBIGUOUS and that is the answer the gate gives. Promotion is the CEO's call
and it should not rest on a 4.5% gap inside a 13.8% seed spread.

**Keep the hold on refitting DE solar on CatBoost with geometry.** Not because of
the geometry delta — that is a summer-only effect and its headline magnitude was a
single draw — but because DE-on-CatBoost carries a 213–306 MW night floor and a
308–380 MW shoulder MAE on every window measured, against XGBoost's much lower
levels. The footgun in this issue's description is real in direction and
overstated in size; the underlying reason to move DE is unchanged.

**What would settle it.** The obstacle is variance, not effect size, and it is
cheap to remove: DE's fits take ~1 second each. A registered read with 10+ seeds
per arm and a rolling-origin sequence of 4–6 holdout windows would separate "DE
prefers XGBoost" from "DE CatBoost is unstable on a 156-day fit" — which are
different findings with different fixes, and the second one would also apply to
BE and FR. I can propose that registration if you want it; it is a separate
issue, not a widening of this one.

**Two things found here that outlive this issue** (filed separately rather than
folded in):

- Every solar number in ABL-338 and ABL-253 is single-seed, and DE's spread is
  4.6–13.8%. Any solar gate read whose decision margin is under ~10% is currently
  unreadable.
- Four holiday features are in `get_feature_columns('solar')` and in **no**
  serving solar artifact. The next routine retrain of any solar country silently
  picks up four never-evaluated features — the same shape of footgun as the
  geometry one, and nobody has read it.

---

## Reproduce

```
# pre-registration probe (reads only, no fits)
.venv\Scripts\python.exe scripts/abl375_preregistration_probe.py --out reports/abl_375_probe.json

# the registered read, one invocation per algorithm
.venv\Scripts\python.exe scripts/abl338_solar_holdout.py --countries AT,BE,DE,FR \
    --holdout 2026-04-30:2026-06-12 --drop-impossible-night \
    --arms control,geometry --seeds 42,1337,2718 \
    --force-algorithm {catboost|xgboost} --out reports/abl_375_solar --tag registered

# post-hoc DE seed characterisation on the already-observed windows
.venv\Scripts\python.exe scripts/abl338_solar_holdout.py --countries DE \
    --holdout {2026-06-13:2026-08-11|2026-03-01:2026-04-29} --drop-impossible-night \
    --arms control,geometry --seeds 42,1337,2718 \
    --force-algorithm {catboost|xgboost} --out reports/abl_375_solar --tag noisefloor_{summer|spring}

# read the bar (never fits)
.venv\Scripts\python.exe scripts/abl375_read_gate.py --out reports/abl_375_de_solar_algorithm
```

`ENERGY_DB_PATH` must be passed explicitly from a worktree — `.env` is gitignored
and `config.DATABASE_PATH` otherwise degrades to a bare `\data\energy_dashboard.db`.

Full tables: `reports/abl_375_de_solar_algorithm.md`.
Raw: `reports/abl_375_de_solar_algorithm.json`, `reports/abl_375_solar/*.json`.
No model was written to `models/`. Nothing here touches the serving registry.
