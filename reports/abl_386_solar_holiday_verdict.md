# ABL-386 — the four holiday features on a solar target: evidence pack

**Registered verdict: MIXED on the primary contrast, NO_EFFECT on the
replicate.** Nothing measured here supports keeping the four holiday features in
the solar feature list on merit. The evidence does not rise to a clean "they
hurt" either: every effect measured is smaller than the seed spread of the arms
being compared, and the pooled paired difference over 24 fits is +1.22% with a
standard deviation of 4.45%.

**The finding that does not depend on the verdict** is scope item 4's, and it is
bigger than solar: **no serving artifact of any forecast type carries any holiday
feature.** 66 of the 66 artifacts that have a feature list at all are missing all
four — including all 24 serving `load` artifacts, for the type `src/features.py`
documents them as "high impact" for.

Registration: `experiments/ABL386/config.json`, committed at `adfdb2a`
(2026-08-13 12:50 +02:00) **before any arm was fitted** on the registered
holdout, together with both pre-fit probes. The verdict below is produced by
`scripts/abl386_holiday_verdict.py`, which is arithmetic over the registered
rule and contains no judgement.

---

## 1. Protocol

| | |
|---|---|
| Registered holdout | **2026-04-30 .. 2026-06-12**, inclusive, **1,056 hours per country** |
| Countries | AT, BE, DE, FR (every solar country) |
| Fit window | every featured row strictly before the holdout start, per country. n_train after cleaning: AT 3,647 · BE 20,277 · DE 3,751 · FR 28,636 |
| Arms | 2×2 of {holidays} × {ABL-338 geometry} — 25 / 27 / 29 / 31 names |
| Cells | 4 countries × 2 algorithms × 3 seeds × 4 arms = **96 fits** |
| Seeds | 42 / 1337 / 2718 |
| Metric | daylight MAE in MW, seed-mean. Shoulder and night in MW only |
| Out-of-sample | yes, every cell. No arm is scored on data it was fitted on |
| Interpreter | `.venv\Scripts\python.exe`, Python 3.14.3 / xgboost 3.3.0 |
| Replica | `C:\Code\able\data\energy_dashboard.db`, **9,432,453,120 bytes** — asserted equal to ABL-375's reading of the same file, so this is the live replica and not the 3.0 GB `energy-data-gathering` partial snapshot |

### The arms

| arm | holidays | geometry | n | what it is |
|---|---|---|---:|---|
| `control_noholiday` | no | no | **25** | **exactly the serving solar feature set** |
| `geometry_noholiday` | no | yes | **27** | what a retrain would produce if holidays were excluded — *the proposal* |
| `control` | yes | no | **29** | ABL-338 / ABL-375's `control` |
| `geometry` | yes | yes | **31** | what a retrain produces on `origin/main` today — *the incumbent* |

The primary contrast is **31 vs 27**: both carry geometry, which is
unconditional in `src/features.py`, so this is the decision as it would really be
taken. **29 vs 25** is the replicate — the same question with geometry off.
Between the paired arms, the *only* thing that varies is the presence of the four
holiday names. Same rows, same truncation, same hyperparameters, same
early-stopping split, same seed.

### The reproducibility check passed exactly

Registered as a check on the run, not as a metric. `control` and `geometry` were
refitted here rather than quoted from ABL-375, because this issue modified the
script between them. All four countries reproduce ABL-375's committed CatBoost
`geometry` seed means **and ranges to the decimal**:

| | ABL-375 committed | ABL-386 refit | Δ |
|---|---:|---:|---:|
| AT | 618.9 (589.1–637.5) | 618.9 (589.1–637.5) | +0.0 |
| BE | 572.2 (561.6–581.9) | 572.2 (561.6–581.9) | −0.0 |
| DE | 4,449.7 (4,224.5–4,838.2) | 4,449.7 (4,224.5–4,838.2) | +0.0 |
| FR | 1,001.6 (996.6–1,004.8) | 1,001.6 (996.6–1,004.8) | +0.0 |

So the `ARM_HOLIDAYS` change provably does not touch the pre-existing arms, and
the numbers below are directly comparable to ABL-375's and ABL-338's.

### Power was measured before the fit, not argued after it

A null on a window with no holidays in it would mean nothing. The registered
holdout carries **3–4 holiday days per country — 6.8–9.1% of rows, and the same
share of *daylight* rows**, where the primary metric lives (Labour Day,
Ascension, Whit Monday, plus FR's 8 May and AT's Corpus Christi). It is 2–3×
holiday-denser than the fit window before it. No holiday feature is constant in
either window in any country, and `days_to_holiday` takes all 7 distinct values
everywhere.

**A null here cannot be explained away as "no holidays in the window".** That is
why the probe ran first.

### Contamination

| issue | touches this window? | handling |
|---|---|---|
| **ABL-337** impossible night actuals | **yes** | `--drop-impossible-night`: dropped from the fit, never from the score, identically on every arm. Fit rows dropped: AT 0 · BE 0 · **DE 4** · **FR 464**. The n_train figures above reconcile exactly (DE 3,755−4, FR 29,100−464). Holdout: **DE 0**, so DE night MW is measured against a true zero. **FR 24 rows up to 251 MW**, which no arm can or should predict |
| **ABL-188** constant-run screen | yes, DE | excludes 6,408 DE rows (2025-09-08 .. 2025-11-14) held at exactly 0. Applied by `src/db` to whatever table is read, so identical on every arm |
| ABL-67 | no — net position only | — |
| ABL-109 / ABL-111 | no — load only | they **would** touch the load follow-up in §7, and are flagged there |
| ABL-71 | provenance caveat | known wrong-write modes are load and net position; not proof solar ingest is clean |

### What this protocol cannot say

- **Not serve-faithful.** Training-time features are anchored at the target hour;
  at serving they are anchored at the generation instant (ABL-183). Every arm
  carries that identically, so the *contrast* holds and the absolute MW are
  optimistic against the rail. The four holiday features and the geometry pair
  are exempt by construction — all six are calendar or astronomical functions of
  the target timestamp, not lagged actuals, so they are identical in both paths.
- **No horizon bands**, so this cannot substitute for a gate-harness read.
- **One window.** A null here is a null on late spring. It is not a claim about
  December, whose holiday cluster is the densest of the year and whose solar
  output is smallest.
- **Half-read window, stated rather than hidden.** The 29- and 31-name arms were
  already scored on this window by ABL-375, so the incumbent side of the contrast
  was known to me and the challenger side was not. The window is *inherited* on a
  non-overlap criterion that has nothing to do with holidays, not selected —
  choosing a different one now, having seen ABL-375's numbers, would have been
  the suspicious act.

---

## 2. The registered read

Seed mean (min–max) of **daylight MAE in MW** over seeds 42 / 1337 / 2718.
`effect = 100 × (holidays − no_holidays) / no_holidays`; **positive means the
holiday features make it worse**. A cell counts only if the two arms' seed ranges
are **disjoint** — `max(better) < min(worse)`.

### Primary: `geometry` (31) vs `geometry_noholiday` (27)

| cell | 31, holidays | 27, no holidays | effect | seed spread hol / nohol | ranges | d |
|---|---:|---:|---:|---:|---|---:|
| catboost/AT | 618.9 (589.1–637.5) | 596.9 (568.9–620.0) | +3.70% | 7.82% / 8.56% | overlapping | 0 |
| catboost/BE | 572.2 (561.6–581.9) | 552.0 (549.7–556.4) | +3.66% | 3.55% / 1.23% | **disjoint** | **−1** |
| catboost/DE | 4,449.7 (4,224.5–4,838.2) | 4,330.5 (3,999.7–4,524.6) | +2.75% | **13.79%** / 12.12% | overlapping | 0 |
| catboost/FR | 1,001.6 (996.6–1,004.8) | 983.5 (977.6–989.1) | +1.84% | 0.82% / 1.17% | **disjoint** | **−1** |
| xgboost/AT | 496.2 (494.0–499.9) | 505.0 (502.5–510.1) | −1.75% | 1.18% / 1.50% | **disjoint** | **+1** |
| xgboost/BE | 606.1 (597.5–622.1) | 604.3 (587.8–619.2) | +0.31% | 4.05% / 5.20% | overlapping | 0 |
| xgboost/DE | 4,249.6 (4,161.8–4,362.6) | 4,307.8 (4,209.5–4,381.6) | −1.35% | 4.73% / 3.99% | overlapping | 0 |
| xgboost/FR | 974.8 (961.1–997.9) | 974.3 (959.8–987.0) | +0.05% | 3.77% / 2.78% | overlapping | 0 |

**sum(d) = −1 over 8 cells; 3 disjoint — 1 favours keeping, 2 favour excluding.**
Registered thresholds are HELP at ≥ +4 and HARM at ≤ −4, NO_EFFECT at ≤ 2 disjoint
cells. → **MIXED**.

### Replicate: `control` (29) vs `control_noholiday` (25)

| cell | 29, holidays | 25, no holidays | effect | ranges | d |
|---|---:|---:|---:|---|---:|
| catboost/AT | 653.9 (647.2–660.0) | 679.5 (627.2–720.1) | −3.77% | overlapping | 0 |
| catboost/BE | 563.0 (552.0–576.9) | 558.2 (546.9–564.4) | +0.86% | overlapping | 0 |
| catboost/DE | 4,463.5 (4,131.5–4,815.6) | 4,060.5 (3,758.1–4,278.8) | +9.93% | overlapping | 0 |
| catboost/FR | 1,018.7 (1,000.5–1,034.1) | 1,017.1 (1,002.5–1,025.1) | +0.16% | overlapping | 0 |
| xgboost/AT | 517.8 (514.2–524.7) | 527.0 (523.0–530.2) | −1.74% | overlapping | 0 |
| xgboost/BE | 635.2 (627.2–643.1) | 613.5 (604.3–624.2) | +3.54% | **disjoint** | **−1** |
| xgboost/DE | 3,914.6 (3,856.8–3,958.1) | 3,969.6 (3,909.1–4,032.2) | −1.39% | overlapping | 0 |
| xgboost/FR | 980.7 (979.8–982.4) | 988.3 (975.9–1,010.7) | −0.76% | overlapping | 0 |

**sum(d) = −1, only 1 of 8 cells disjoint → NO_EFFECT.** The replicate does not
disagree in *direction* with the primary, so no downgrade is triggered; both lean
very slightly against the holiday features.

Note `catboost/DE` at **+9.93%** — the largest single effect in either table, and
still overlapping, because that cell's own seed spread is 15.33% / 12.82%. It is
the clearest illustration of why no fixed percentage threshold was registered.

### Every arm clears the free baseline

Literal seasonal-naive D-7 daylight MAE on the same rows: AT 781.9 · BE 1,425.5 ·
DE 6,509.3 · FR 2,128.1 MW (daylight n = 670 / 681 / 686 / 656). The worst arm in
either table beats it comfortably; DE's best arm sits 39.9% below it. D-7 is the
sanity floor here, not the decision bar — a feature-list question is a
within-model question.

---

## 3. Bands: shoulder and night (scope item 3)

Seed means, MW. Night in MW only — its denominator is ~0, so a percentage there
would report the denominator rather than the model.

| cell | arm | daylight | shoulder | night mean pred | neg. preds |
|---|---|---:|---:|---:|---:|
| catboost/DE | 25 serving | 4,060.5 | 277.8 | 171.2 | 58 |
| catboost/DE | 27 no-holiday | 4,330.5 | 392.1 | **287.5** | 2 |
| catboost/DE | 29 | 4,463.5 | 308.5 | 212.8 | 35 |
| catboost/DE | 31 incumbent | 4,449.7 | 317.2 | 230.7 | 26 |
| xgboost/DE | 25 serving | 3,969.6 | 71.1 | −44.2 | 221 |
| xgboost/DE | 27 no-holiday | 4,307.8 | 50.0 | −39.8 | 188 |
| xgboost/DE | 29 | 3,914.6 | 54.0 | −38.2 | 229 |
| xgboost/DE | 31 incumbent | 4,249.6 | 59.4 | −44.5 | 203 |

The full 32-row table is in `reports/abl_386_solar/`. The pattern ABL-375 reported
holds and is **not** a holiday effect: DE-on-CatBoost sits at a 171–288 MW night
floor and a 278–392 MW shoulder MAE at every holiday setting; DE-on-XGBoost sits
at −44 to −38 MW night and 50–71 MW shoulder at every holiday setting. The
algorithm gap dwarfs the holiday gap in both bands.

### Night guardrail — registered as `|night mean|`, and it fails on two CatBoost cells

ABL-375 reported, as a defect in its own registration, that its one-sided `≤`
guardrail was satisfied by a large *negative* night mean. This registration used
the absolute value, fixed before any number was seen.

| cell | holidays | no holidays | change from excluding | nohol seed spread | pass |
|---|---:|---:|---:|---:|---|
| catboost/AT | 26.1 | 28.4 | +2.3 | 4.8 | PASS |
| catboost/BE | 11.6 | 12.1 | +0.5 | 1.7 | PASS |
| **catboost/DE** | 230.7 | 287.5 | **+56.7** | 39.6 | **FAIL** |
| **catboost/FR** | 28.9 | 36.4 | **+7.5** | 4.0 | **FAIL** |
| xgboost/AT | 4.1 | 3.6 | −0.5 | 2.3 | PASS |
| xgboost/BE | 0.8 | 0.8 | −0.0 | 1.7 | PASS |
| xgboost/DE | 44.5 | 39.8 | −4.8 | 18.8 | PASS |
| xgboost/FR | 37.2 | 32.5 | −4.7 | 21.8 | PASS |

**This is the one result that argues against exclusion, and it is reported as
prominently as the ones that argue for it.** Under CatBoost, removing the holiday
features buys ~3% of daylight MAE and costs night level: DE's mean night
prediction rises from 231 to 288 MW against a true zero (DE has no contaminated
night rows in this holdout, so that is model error, not bad actuals). FR's
failure is smaller and carries the ABL-337 caveat only indirectly — the guardrail
is on *predictions*, so contamination reaches it through the fit, and the fit was
cleaned of all 464 rows.

Under XGBoost the guardrail passes in all four countries, with the sign reversed.

---

## 4. Scope item 4: the drift is repo-wide, not solar-only

`scripts/abl386_feature_drift_probe.py`, run before any fit. Every serving
`model.joblib` under the live models directory, compared to
`get_feature_columns()` for its own type:

| type | serving artifacts | missing from **every one** of them |
|---|---:|---|
| load | 24 | the 4 holiday names |
| price | 24 | the 4 holiday names |
| renewable | 4 | the 4 holiday names |
| **solar** | 4 | the 4 holiday names **+ `sun_elevation_deg`, `is_night`** |
| wind_onshore | 4 | the 4 holiday names |
| wind_offshore | 2 | the 4 holiday names |
| biomass | 2 | the 4 holiday names |
| hydro_total | 2 | the 4 holiday names |

**66 of the 66 artifacts that carry a feature list are missing all four holiday
features.** Solar is not distinctive in having the holiday gap — it is
distinctive only in having a *second* gap on top of it, ABL-338's geometry, which
was deliberate and documented.

The sharpest version: `src/features.py` comments these four as *"high impact for
load forecasting"*, and **none of the 24 serving load artifacts has them**. No
served forecast of any type has ever used a holiday feature.

One artifact — BE `price_cascade` lightgbm `20260221_201435` — carries an *empty*
`feature_columns` list. That is a different model class, not holiday drift; it is
excluded from the counts above and reported separately in the probe JSON.

### Why the gap exists: a provenance gap, not a regression

Added after review. ABL-394 proposed a mechanism — that the fit-time intersection
`[c for c in get_feature_columns(t) if c in df.columns]` dropped the four names
because `create_holiday_features` never ran on a training frame, and that
ABL-338 (`5cf2296`) made them live by threading `country_code` into
`create_all_features`. **That does not reproduce.** Measured three ways:

1. `git show 5cf2296 --stat -- scripts/train.py` is **empty**. ABL-338 did not
   touch the training script.
2. At `5cf2296^`, `create_all_features` already carried
   `country_code: Optional[str] = None` and already called
   `create_holiday_features(df, country_code)`; `scripts/train.py:488` already
   passed `country_code=country_code`.
3. Executing the pre-ABL-338 tree directly (detached worktree at `5cf2296^`,
   `.venv` Python 3.14.3) produces **all four** holiday columns on both `solar`
   and `load`, and the fit-time intersection **keeps** all four.

Both the four holiday names and the `country_code` threading trace to `996c45a`
*Initial commit (migrated from energy-dashboard monorepo)*, dated **2026-03-05**.
Every serving solar artifact is stamped `20260112`–`20260223` — **all four
predate the migration**. So the honest reading is that no serving artifact was
produced by this repo's training path, and the declared/served divergence has
been latent since migration rather than introduced by any commit in it.

This does not weaken the issue's claim; it strengthens the basis for it. "The
next retrain picks up four never-evaluated features" is here confirmed by
*execution* rather than inferred from list arithmetic: a fit today produces the
four columns and keeps them, on both solar and load.

#### The refuted mechanism has since been written into `main` as fact

Recorded 2026-08-13, when `origin/main` (`1bd99e5`) was merged into this branch.
ABL-394 landed while this branch waited, and it states the mechanism above —
the one that does not reproduce — as settled, in two places that a future reader
will treat as authoritative:

- `CLAUDE.md`: "All 66 artifacts ... were fitted before ABL-338 (`5cf2296`)
  threaded `country_code` into `create_all_features`, so `create_holiday_features`
  never ran on a training frame".
- `tests/test_feature_list_contract.py` module docstring, under the heading
  **"The mechanism, measured rather than assumed"**: "the training sites called
  `create_all_features(df, forecast_type)` with no `country_code`".

Both are wrong on the same point, and it is checkable in one command:

    git show 5cf2296^:scripts/train.py | grep -n "create_all_features("

At `5cf2296^` that returns two call sites, not one. The **training** site inside
the per-country fit loop already reads
`create_all_features(df, forecast_type, country_code=country_code)` — it passed
`country_code` *before* ABL-338. The site that omits it is the one in
`evaluate_against_baselines`, the **validation** path, which is the separate
zero-skill defect in the next section and never writes an artifact's
`feature_columns`. (Line numbers deliberately omitted: both files have moved
since. On the merged tree the same grep on `scripts/train.py` shows the training
site passing `country_code` and the `evaluate_against_baselines` site still not.)

ABL-394's *tests* are sound and none of this makes them red: they assert that
omitting `country_code` drops the four names, which is true. What is unsupported
is the antecedent — that the training sites omitted it. The test proves the
conditional; the docstring asserts the premise.

Nothing in this PR changes `CLAUDE.md` or ABL-394's docstring: the correction
belongs to whoever owns that text, and rewriting a merged issue's doctrine inside
this evidence-only branch is how a `CLAUDE.md` cascade starts. Raised as a
follow-up instead. The merge is otherwise clean and this branch's numbers are
unaffected — the contradiction is textual, not arithmetic.

### A live consequence found while checking that: skill scores silently go to zero

`scripts/train.py:715`, inside `evaluate_against_baselines`, builds the
validation frame with `create_all_features(val_df, forecast_type)` — **without
`country_code`, which is already a parameter of that same function**. It then
indexes `val_df[forecaster.feature_columns]`.

For a model fitted today the fitted list contains the four holiday names and the
validation frame does not, so that indexing raises `KeyError`. The whole body is
wrapped in `except Exception as e: logger.warning(...)`, and `skill_scores` is
pre-initialised to `{'skill_vs_persistence': 0.0, 'skill_vs_seasonal_naive':
0.0}` — so the failure is swallowed and **zero skill is recorded as if measured**.

Confirmed on `load`, `price`, `renewable`, `wind_onshore` (all four missing
columns are exactly the holiday names). `solar` is protected: ABL-338 added a
loud `ValueError` for a missing `country_code` on solar. Reported separately;
this issue is evidence-only and does not carry the fix.

### The 25-name claim is now a checkable predicate

`get_feature_columns('solar', include_holidays=False)` minus
`SOLAR_GEOMETRY_FEATURES` reproduces the `feature_columns` of all four serving
solar artifacts **exactly — name for name and in order**, n = 25. So
`control_noholiday` is the first arm in the ABL-338 / 375 / 386 lineage that
genuinely *is* the serving feature set, and this is the first time its number can
be quoted at all (as a refit on the truncated window — never as the live
artifact, which is fitted through roughly today and would score in-sample here).

Worth recording: on `catboost/DE` the **serving 25-name list is the best of the
four arms** at 4,060.5 MW, against 4,449.7 for the 31-name list a retrain would
produce today. That is inside DE CatBoost's 12–15% seed spread and is not a
claim — but it is the opposite of what "the current list is an improvement on the
serving one" would predict, and it is the first time anyone could check.

---

## 5. Exploratory, not registered: the algorithm interaction

Pairing by seed is far more powerful than the unpaired range test that was
registered, because the two arms in a pair share their seed. It was **not**
registered, so it cannot move the verdict; it is the sharpest hypothesis this run
generated and it belongs to the follow-up.

Paired effect per seed, primary contrast, positive = holidays worse:

| cell | s42 | s1337 | s2718 | mean | all 3 seeds agree? |
|---|---:|---:|---:|---:|---|
| catboost/AT | +4.72 | +12.07 | −4.97 | +3.94 | no |
| catboost/BE | +4.27 | +2.11 | +4.57 | +3.65 | **yes, worse** |
| catboost/DE | +7.17 | −6.63 | +8.30 | +2.95 | no |
| catboost/FR | +1.44 | +1.30 | +2.78 | +1.84 | **yes, worse** |
| xgboost/AT | −3.15 | −0.53 | −1.54 | −1.74 | **yes, better** |
| xgboost/BE | −1.15 | −3.50 | +5.83 | +0.40 | no |
| xgboost/DE | −0.43 | −1.13 | −2.50 | −1.35 | **yes, better** |
| xgboost/FR | −2.62 | +0.57 | +2.22 | +0.06 | no |

| pool | mean | sd | n positive |
|---|---:|---:|---:|
| all 24 paired fits | **+1.22%** | **4.45%** | 13/24 |
| CatBoost only (12) | +3.09% | 5.20% | 10/12 |
| XGBoost only (12) | −0.66% | 2.61% | 3/12 |

Two things to take from this, and no more:

1. **The pooled effect is a coin flip.** +1.22% mean against a 4.45% standard
   deviation, 13 of 24 fits positive. Even *paired by seed*, the seed-to-seed
   variation in the difference is larger than the difference. `catboost/AT` swings
   from +12.07% to −4.97% between two seeds **on the paired difference itself**.
2. **The algorithm split is the real pattern:** CatBoost leans harmed (+3.09%,
   10/12 fits positive, all four country means positive), XGBoost leans
   indifferent-to-helped (−0.66%, 3/12 positive). A mechanism is available —
   `is_holiday` and `is_bridge_day` are near-constant binary columns at 3–9%
   density, and the two libraries treat low-cardinality integer features very
   differently — but **this run does not test it.**

No significance test is quoted, because none was registered and the 24 fits are
not independent (4 countries × 3 seeds on one shared window, with the country as
the real unit of replication). Treating these as 24 samples would overstate the
evidence, and the direction is not established either way.

---

## 6. Recommendation

**The registered mapping for MIXED is: report the disagreement, and recommend
exclusion on parsimony only if no cell shows disjoint HELP; otherwise recommend a
further read and name what it must measure.** `xgboost/AT` shows disjoint HELP
(−1.75%), so the registered recommendation is **a further read**, not exclusion.
I am following the mapping I registered rather than the conclusion I expected.

That said, the read is not empty, and three things are settled well enough to act
on:

1. **There is no case for keeping these four on merit.** Across 8 cells the net
   score is −1, both contrasts lean against them, and the single disjoint cell
   favouring them is one country-algorithm pair at −1.75% against two disjoint
   cells at +3.66% and +1.84% the other way. The prior — solar output is driven
   by irradiance, not by whether it is a public holiday — survives contact with
   the data.
2. **Nor is there a case for excluding them on daylight accuracy.** The effect is
   inside the noise on 5 of 8 cells and the pooled paired difference is not
   distinguishable from zero.
3. **The night guardrail is the one live argument for keeping them**, on CatBoost
   only, and it is a two-country result (DE +56.7 MW, FR +7.5 MW) that XGBoost
   reverses.

**So the decision should not be made on these numbers.** It should be made on the
footgun, which is established independently of the verdict and is repo-wide: four
features that no serving artifact of any type carries will enter the next retrain
of any country, on any type, without ever having been evaluated. That is a
process defect, not a modelling result, and fixing it does not require knowing
whether the features help.

### What I am handing over

- **To the CEO, for a decision:** whether to make the feature list *deliberate* —
  i.e. pin each type's list to what its artifacts carry, or gate a retrain on an
  explicit list change — rather than letting `get_feature_columns()` drift into
  the next fit. This is the ABL-375 recommendation generalised from geometry to
  every feature, and it is a Founding Engineer change, not mine.
- **Not recommended:** changing `get_feature_columns('solar')` on the strength of
  this read alone. Scope item 5 is respected either way — the feature list and
  the serving registry are not touched in the same change, and nothing here is
  promoted.

### What a further read must measure

1. **Seeds, not windows.** 3 seeds and an unpaired range test is a coarse
   instrument — by construction better at ruling an effect out than sizing one.
   A follow-up should register the **paired-by-seed** difference as its primary
   statistic and enough seeds to separate a ~3% effect from a 4.5% paired sd.
2. **December.** This window is late spring. The densest holiday cluster of the
   year sits in the window with the least solar output, which is both where an
   effect is most likely and where it matters least in MW. Registering it is
   cheap and it is the obvious counter-case.
3. **The algorithm interaction as the hypothesis**, pre-registered rather than
   discovered: *CatBoost is harmed by near-constant binary features on a solar
   target; XGBoost is not.* If that holds it is a statement about the feature
   list per algorithm, which is a different recommendation from the one this
   issue was scoped to make.

---

## 7. Follow-up this read does not cover

The load and price question is larger than solar and has the **opposite prior**.
These features were added for load — 24 serving load artifacts and 24 price
artifacts lack all four, and `src/features.py` calls them "high impact" for
exactly that type. If they do help load, the finding is not "drift" but
"24 countries are serving load forecasts without a feature the repo believes in".

That read is a separate issue: different target, different bands (no solar
geometry, no night clamp), and it is touched by **ABL-109 / ABL-111**
(zero-as-missing actual-load rows) and **ABL-71**, none of which intersect this
solar window. It should not be folded into this one.

---

## 8. Files

| | |
|---|---|
| registration | `experiments/ABL386/config.json` (committed `adfdb2a`, pre-fit) |
| drift probe | `scripts/abl386_feature_drift_probe.py` → `reports/abl_386_feature_drift.json` |
| power probe | `scripts/abl386_holiday_density_probe.py` → `reports/abl_386_holiday_density.json` |
| the 96 fits | `scripts/abl338_solar_holdout.py` → `reports/abl_386_solar/holdout_abl386_{catboost,xgboost}_cleaned.json` |
| verdict arithmetic | `scripts/abl386_holiday_verdict.py` → `reports/abl_386_holiday_verdict_tables.{json,md}` (generated; this file is hand-written and the script does not overwrite it) |
