# ABL-407: where the 66-artifact holiday gap actually comes from

**Measured 2026-08-13** against `origin/main` `1bd99e5` and the live `models/` tree.
Supersedes the causal story ABL-394 landed in `CLAUDE.md`, `src/features.py` and
`tests/test_feature_list_contract.py`. **No arithmetic in ABL-394 changes** — the
66-artifact count, the 23/23/26/25/27/25/24/24 lengths, `feature_list_manifest.json`
and its `serving_gap` block are all unaffected, and every ABL-394 test stays green.
Only the explanation changes.

## 1. The claim that was in `main`

> All 66 artifacts that carry a `feature_columns` list at all were fitted before
> ABL-338 (5cf2296) threaded `country_code` into `create_all_features`, so
> `create_holiday_features` never ran on a training frame and the fit-site
> narrowing dropped these four names in silence.

It appeared in three places, not the two ABL-407 named: `CLAUDE.md` (holiday-features
note), `tests/test_feature_list_contract.py` (module docstring), and
`src/features.py` (`select_feature_columns` docstring, lines 547-555 as landed).

## 2. Why it is wrong — four checks

**(a) ABL-338 never touched the training script.**

```
$ git show 5cf2296 --stat -- scripts/train.py
(empty)
```

`5cf2296` touched `src/features.py`, `src/forecaster.py`, `src/solar_features.py`,
`src/wind_features.py`, `scripts/abl338_solar_holdout.py` and eight report files.
Not `scripts/train.py`.

**(b) The training site already passed `country_code` before it.**

```
$ git show 5cf2296^:scripts/train.py | grep -n "create_all_features("
488:            df = create_all_features(df, forecast_type, country_code=country_code)
653:        val_df = create_all_features(val_df, forecast_type)
736:        val_df = create_all_features(val_df, "price", country_code=country_code)
750:        load_val = create_all_features(load_val, "load", country_code=country_code)
751:        ren_val = create_all_features(ren_val, "renewable", country_code=country_code)
```

Line 488 is inside `train_model` (defined at line 349) — the per-country fit loop,
the site whose narrowing decides an artifact's `feature_columns`. It passed
`country_code`. Line 653 is inside `evaluate_against_baselines` (line 616) — the
**validation** frame, which writes no artifact. That single omission is the separate
defect **ABL-397**, and conflating the two is what produced the wrong story.

**(c) The builder already accepted and used it.** At `5cf2296^`,
`src/features.py:363` declares `create_all_features(..., country_code: Optional[str]
= None)` and line 397 calls `create_holiday_features(df, country_code)`.

**(d) Executing the pre-ABL-338 tree, rather than reading it.** Detached worktree at
`5cf2296^` (`981e4d6`), run under the repo `.venv` (Python 3.14.3), with the same
synthetic best-case frame `tests/test_feature_list_contract.py::_frame` uses:

| tree `981e4d6` = `5cf2296^` | holiday cols produced | fit-time intersection keeps | fit width |
|---|---|---|---|
| `solar`, **with** `country_code` (training shape, line 488) | 4/4 | **4/4** | 29/29 |
| `solar`, without `country_code` (validation shape, line 653) | 0/4 | 0/4 | — |
| `load`, **with** `country_code` (training shape) | 4/4 | **4/4** | 30/30 |
| `load`, without `country_code` (validation shape) | 0/4 | 0/4 | — |

A fit on the pre-ABL-338 training path produces all four holiday columns and keeps
all four. The mechanism cannot be what put the gap in the artifacts.

**Provenance of both halves.** `git log -S` on the training call and on
`def create_holiday_features` each return exactly one commit: `996c45a`
*Initial commit (migrated from energy-dashboard monorepo)*, **2026-03-05 21:44:33 +0100**.
Neither the names nor the threading was introduced by any later commit in this repo.

## 3. What the artifacts say

`models/` is gitignored, so this is measured on the live tree in the primary
checkout. The "66" is exactly the top-level set `models/{CC}/{type}/model.joblib`;
per-type counts reproduce `feature_list_manifest.json`'s `n_artifacts` exactly
(biomass 2, hydro_total 2, load 24, price 24, renewable 4, solar 4, wind_offshore 2,
wind_onshore 4 = 66). 198 `model.joblib` files exist in total; the other variants
(`production/` 27, `candidate/` 66, `previous/` 17, `centroid/` 9, `multipoint/` 9)
are not the serving set and are not in scope here.

- **66 of 66 carry none of the four holiday names.** Unchanged from ABL-394.
- **60 of 66 were saved `2025-12-26T13:43:28` .. `2026-02-23T19:38:22`** — before
  `996c45a` (2026-03-05). These predate this repository. Nothing in this repo's
  history produced them, so no commit here can be their cause.
- **6 of 66 were saved `2026-04-04T18:55:21` .. `2026-04-04T18:55:35`** — a month
  *after* the migration — and still carry none of the four. They are
  **BE, DE, FR × {load, price}**.

Both stamps agree on that split: the artifact's self-reported `saved_at` and the
file's mtime select the same 6 files. (`saved_at` alone is weak evidence — it is
`datetime.now()` taken at write time, `src/forecaster.py:988`, so it dates the
*save*, not the fit — which is why the mtime is quoted as an independent second
witness rather than relied on alone.)

### The honest reading

For **60 of 66**, the gap is a provenance gap: the artifacts were not produced by
this repo's training path at all, and the declared/served divergence has been latent
since migration rather than introduced by any commit in it.

For the remaining **6**, that explanation does not hold and **no cause is
established**. They postdate the migration, every training site on `main` and at
`5cf2296^` passes `country_code`, and a fit on either tree keeps all four names —
yet these six carry none. Do not paper this over with either story. Candidates not
yet discriminated: produced outside this repo and copied in; produced by a script no
longer in the tree; or produced by a path that reached `Forecaster.save` without the
training-site narrowing. Establishing which is not in ABL-407's scope.

## 4. What this does and does not change

- **ABL-394's tests stay green and are correct.** They assert the conditional —
  omit `country_code`, lose the four names — which is true and is re-confirmed in
  section 2(d) above. What was unsupported was the antecedent, that the training
  sites omitted it. The tests never asserted that; only the prose did.
- **No number moves.** The 66 count, the per-type served lengths, the manifest and
  its `serving_gap` block are untouched.
- **The four names are still live for the next fit**, and still never evaluated on
  any target. ABL-386's MIXED read on solar stands.
- **ABL-397 is the validation-path site** (`scripts/train.py:717` on `main`,
  line 653 at `5cf2296^`). It is real, it is separate, and it writes no artifact.

## 5. Found while measuring — not fixed here

`Forecaster.predict_d2` (`src/forecaster.py:715`) builds its serving frame with
`create_all_features(df, self.forecast_type)` — **no `country_code`**
(`src/forecaster.py:801`) — and then hard-indexes the artifact's own list,
`same_hour_data.iloc[-1:][self.feature_columns]` (`src/forecaster.py:824`, again at
`:874`). This is the live daily serving path (`scripts/forecast_daily.py:176`).

It is inert **only because** no artifact currently declares the four holiday names.
The moment any retrain produces one that does — which is precisely what ABL-394
says the next fit will do — `predict_d2` raises `KeyError` on the first serve. That
is a different site from ABL-397's (serving, not validation), so it is filed
separately rather than folded in here.
