# ABL-597 — what actually loads the serving artifacts, and what must be pinned

**Date:** 2026-08-28 · **Author:** Forecasting Scientist · **Status:** evidence pack

Scope: `requirements.txt` in `energy-forecast` used `>=` floors, so an unrelated
container rebuild could float `xgboost` / `catboost` / `numpy` under the serving
artifacts. `models/` is gitignored, so no commit protects those bytes. This pack
determines *which* packages are on the load path, *which versions* production
runs, and *whether drift has already happened*.

Everything below is read-only. No database was opened, no artifact rewritten,
nothing deployed.

## 1. Which packages the artifacts need — determined, not guessed

`scripts/abl597_artifact_load_path.py` replays the pickle opcodes of every
`models/<CC>/<type>/model.joblib` and resolves each `GLOBAL` / `STACK_GLOBAL` to
a `(module, qualname)` pair. Those names are literally in the artifact bytes:
they are what the unpickler will import.

**67 artifacts, 67 parsed, 0 errors.**

| symbol the unpickler imports | artifacts |
|---|---|
| `catboost.core.CatBoostRegressor` | 48 |
| `numpy.core.multiarray.scalar` | 49 |
| `numpy.dtype` | 49 |
| `xgboost.core.Booster` | 18 |
| `xgboost.sklearn.XGBRegressor` | 18 |
| `builtins.bytearray` | 18 |
| `collections.OrderedDict` | 10 |
| `collections.defaultdict` | 1 |
| `lightgbm.basic.Booster` | 1 |
| `lightgbm.sklearn.LGBMRegressor` | 1 |

Re-derive with:

```
.venv\Scripts\python.exe scripts/abl597_artifact_load_path.py --check-intercept
```

**Three corrections to the issue's candidate list.**

1. **`numpy` is not "likely", it is the most fragile item in the set.** 49 of 67
   artifacts carry `numpy.core.multiarray.scalar` — the **pre-2.0** path. Under
   numpy 2.x that name resolves only through the `numpy.core` compatibility
   shim. A rebuild that floats numpy to a release which drops the shim stops 49
   artifacts loading. Production is on numpy 2.4.6, i.e. the shim is doing real
   work today.

2. **`scikit-learn` names no symbol in any artifact** — but it is still on the
   load path, because `xgboost.sklearn` and `lightgbm.sklearn` cannot import
   without it, and sklearn cannot import without `scipy`. Same for `joblib`,
   which is the reader itself. Pinning these on *import-path* grounds is correct;
   claiming they are in the pickle would not have been.

3. **`pandas` is not on the unpickle path at all.** It is on the *prediction*
   path (the feature frame). Still pinned, but for that reason, not this one.

`lightgbm` was not in the issue's candidate set and does appear —
`BE/price_cascade` is a LightGBM cascade.

## 2. What production runs

`docker exec energy-forecast python3 -m pip freeze`, captured 2026-08-28 by the
Deployment Engineer under **ABL-598**. Not a fresh resolve. All 13 values now in
`requirements.txt` were diffed programmatically against that capture — zero
transcription drift.

```
xgboost==3.2.0   catboost==1.2.10   lightgbm==4.7.0   scikit-learn==1.9.0
numpy==2.4.6     pandas==3.0.5      scipy==1.17.1     joblib==1.5.3
optuna==4.9.0    holidays==0.103    python-dotenv==1.2.3
pytz==2026.3.post1                  tqdm==4.70.0
```

## 3. Drift has already happened — but it has not damaged anything

This is the part the issue did not anticipate, and it cuts both ways.

**All 18 xgboost artifacts were written by `xgboost 2.1.4`.** Production loads
them under **3.2.0**; the local rail loads them under **3.3.0**. That is a major
version apart, in both environments, right now. The version is decoded from the
UBJSON `version` field in the *stored* blob.

> **Methodological note, because it cost a pass.** Asking a loaded booster for
> its version reports the **current** library, not the writer's —
> `save_raw` / `save_config` re-serialise with whatever is installed. My first
> read did exactly that and returned "3.3.0" for all 18, which reads as *artifact
> matches environment* and is meaningless. Only the untouched bytes are evidence.

**The intercept survived.** `base_score` as the 2.1.4 writer stored it is
identical to what the booster reports after `joblib.load` under 3.3.0 on
**18 of 18** artifacts (relative error ≤ 1e-9). So the failure mode CLAUDE.md
documents — trees kept, fitted intercept silently reset, shape without level —
**is not occurring**. This is a negative result and it is reported as one: the
predictions being served are level-correct.

Two caveats on that reassurance:

- It was checked under **3.3.0** (the rail), not under production's **3.2.0**.
  A load under 3.2.0 is the check that would settle it for production, and that
  needs a container. It is in the handover below.
- **The in-artifact guard cannot fire on any of these.** `Forecaster.save`
  writes the ABL-183 witness (`xgboost_version`, `base_score`) for xgboost
  artifacts, but **none of the 67 artifacts on disk carry those keys** — nor
  `training_source`. Every serving artifact predates that machinery. So
  `xgboost_artifact_guard` has nothing to compare against, and `requirements.txt`
  is the *only* thing pinning the library under them.

`catboost` writer version is not recoverable by this method; its blob carries a
git commit, not a version triple. Since prod and rail are both on 1.2.10, there
is no catboost skew to measure — but this is unverified rather than confirmed.

One artifact (`BE/price_cascade`) matches the byte string `xgboost` in a metadata
field while containing no xgboost model; it is reported as writer `None` and is
benign.

## 4. What changed

- `requirements.txt` — every entry moved from `>=` to `==` at the ABL-598
  production values. `scipy` added explicitly: it is on the load path via
  sklearn and floated transitively before. No version was *moved*; this is a
  freeze at today's known-good, per the issue's "not in scope: upgrading
  anything".
- `tests/test_abl597_requirement_pins.py` — five tests. Every load-path package
  is `==`; the whole file is `==`; every load-path package is declared; the pins
  equal the recorded production set; every requirement has a recorded production
  version behind it. Editing a pin without editing the recorded set goes red, so
  a version move is a reviewed decision rather than a resolver outcome.
- `scripts/abl597_artifact_load_path.py` — the derivation above, re-runnable.

## 5. Limits of this change — stated, not papered over

1. **This is not a lockfile.** It pins the direct requirements only. The
   transitive closure (`narwhals`, `six`, `threadpoolctl`, `python-dateutil`,
   `nvidia-nccl-cu12`, …) still floats on rebuild. Closing that needs a
   constraints file and a `Dockerfile` change — deploy-path territory, and a
   separate reviewable change.
2. **The rail and the container are not on the same versions.** Training runs
   under `.venv` at xgboost **3.3.0** / numpy **2.5.1**; the container serves at
   **3.2.0** / **2.4.6**, and `docker-compose.yml` mounts the host `models/`
   straight into `/app/models`. So the *next* retrain writes a 3.3.0 pickle that
   production reads under 3.2.0 — a newer pickle into an older library, which is
   the unsupported direction. Today's artifacts are 2.1.4 and unaffected. This
   pack does not fix it and it is not what ABL-597 asked for; it is the sharper
   version of the same risk and needs its own issue.
3. **Scope item 3 (rebuild and prove no drift) is not done here** and cannot be:
   it needs the container. Handover below.
4. `requirements-chronos.txt` still uses floors. Deliberate, out of scope — it
   is the workstation GPU venv and is explicitly not installed in the prod image.

## 6. Handover — Deployment Engineer

To close scope item 3:

1. Rebuild the image and confirm `pip freeze` reproduces §2 for the 13 pinned
   packages.
2. In the rebuilt container, run
   `python3 scripts/abl597_artifact_load_path.py --check-intercept`. Under
   production's xgboost 3.2.0 this is the check §3 could not make; it exits
   non-zero if any intercept moved.
3. Scoped `--dry-run` across the full matrix against the ABL-585 pre-rebuild
   baseline (`Total: 392, Success: 148`).

Per the issue: if a pin turns out to be unsatisfiable, stop and report rather
than resolving it by moving a version.
