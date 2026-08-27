> **Archived from `CLAUDE.md` on 2026-08-27** (companion to the ABL-536
> energy-dashboard-frontend trim). Historical narrative, incident forensics
> and dated measurements; `file:line` references are frozen as of the archive
> date. The durable rules distilled from this material live in the repo-root
> `CLAUDE.md`; where they conflict, the root file wins.
# The interpreter is part of the configuration (ABL-69)

## The interpreter is part of the configuration (ABL-69)

**This box has two Pythons, and a model artifact is only valid under the one
that wrote it.** The bare `python` on `PATH` is *not* the one the pipeline uses.

| role | interpreter | Python | xgboost |
|---|---|---|---|
| **the rail** — trains, serves, evaluates | `C:\Code\able\energy-forecast\.venv\Scripts\python.exe` | 3.14.3 | **3.3.0** |
| whatever `python` resolves to | `C:\Users\guill\miniconda3\python.exe` | 3.11.4 | 2.1.4 |

`scripts/workstation/run-net-position.ps1` invokes `$Repo\.venv\Scripts\python.exe`
explicitly for every step, so the scheduled job is consistent. **An interactive
run is not**, and that is where this bites.

An xgboost-3.3.0 pickle loaded under 2.1.4 does not fail. It keeps its trees and
**silently resets the fitted intercept to the 0.5 default** — FR's is 6,585.93 MW
— then predicts a near-zero-mean series. Measured on FR W12, 2026-08-08:

| interpreter | FR W12 MAE | SMAPE |
|---|---:|---:|
| `.venv` (3.3.0) | **1,688 MW** | 28% |
| conda (2.1.4) | 5,824 MW | 189% |

Predictions came back at mean −6 MW / std 575 against actuals at mean 5,818,
while correlation held at 0.615 — a model with shape and no level, which reads
as a bad model rather than a bad load. The only signal is a `UserWarning` about
serialized models. Nothing crashes and no test fails; the backtest simply reports
that the challenger lost.

`src/challengers/v014.py` now refuses this rather than trusting it.
`save_model` writes the xgboost version and the fitted intercept into the
artifact; `load_model` reads the intercept back out of the booster's own config
and raises `ModelArtifactError` when it has moved, naming the interpreter to use.
It checks the **symptom**, not version equality — so it stays silent across
upgrades that are actually fine, and fires whenever predictions would be wrong.
An artifact written before the guard carries no witness and still loads: absent
evidence is "cannot check", not "corrupt".

Run anything that loads a model — `train_v014.py`, `backtest_v014.py`,
`forecast_challengers.py`, `evaluate_net_position.py` — under `.venv`, and note
that `.env` is gitignored, so a **git worktree has no `.env`** and
`config.DATABASE_PATH` degrades to a bare `\data\energy_dashboard.db`. Pass
`ENERGY_DB_PATH` explicitly from a worktree.

One configured exception, measured rather than assumed (ABL-354): the
`tso-correction` runner is pinned to the conda interpreter at `config.py:490`,
not the rail. Its artifacts are **LightGBM**, not xgboost, and LightGBM
round-trips a booster as text. The three BE models
(`models/tso_correction/BE/*/model.joblib`, trained 2026-04-01) load with no
warning and predict identically to 6 dp under lightgbm 4.6.0 (conda) and 4.7.0
(`.venv`), and a full `-m src.tso_correction_forecaster --country BE --horizon 2`
gives the same `tso_raw` (1191.096365 MW mean) and `tso_corrected` (1254.571424)
under both. The ABL-69 failure does not reach this runner. That is a fact about
the artifact format, not a general licence — anything holding an xgboost pickle
still belongs on `.venv`.
