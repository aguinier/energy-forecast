> **Archived from `CLAUDE.md` on 2026-08-27** (companion to the ABL-536
> energy-dashboard-frontend trim). Historical narrative, incident forensics
> and dated measurements; `file:line` references are frozen as of the archive
> date. The durable rules distilled from this material live in the repo-root
> `CLAUDE.md`; where they conflict, the root file wins.
# Architecture

## Architecture

```
energy_forecast/
├── config.py           # Configuration (paths, countries, model params)
├── requirements.txt    # Python dependencies
├── src/
│   ├── db.py               # Database operations
│   ├── data_quality.py     # Training-data invariants (ABL-188: rejects
│   │                       # suspect constant-value runs from energy_renewable;
│   │                       # ABL-200: rejects zeros energy_generation disproves
│   │                       # at the same instant, at any run length)
│   ├── features.py         # Feature engineering (incl. holiday features)
│   ├── solar_geometry.py   # Sun elevation per (country, timestamp) — one
│   │                       # capacity-weighted point per country (ABL-337);
│   │                       # also NIGHT_GENERATION_POSSIBLE, the per-country
│   │                       # night-generation fact, no default (ABL-425)
│   ├── solar_clamp.py      # Serving-path night mask + non-negativity floor
│   │                       # for solar, with per-run telemetry (ABL-337).
│   │                       # ES is exempt from the night mask (ABL-425)
│   ├── metrics.py          # Evaluation metrics
│   ├── forecaster.py       # Forecaster class (XGBoost/LightGBM/CatBoost)
│   ├── hyperopt.py         # Optuna Bayesian hyperparameter optimization
│   ├── feature_selection.py # Automated feature selection
│   ├── validation.py       # Walk-forward validation
│   ├── baselines.py        # Baseline models (persistence, seasonal naive)
│   ├── model_registry.py   # Model versioning and registry
│   ├── deployment.py       # Model deployment management
│   └── chronos2/           # Chronos-2 foundation model (ported from netpredict2)
│       ├── engine.py           # Chronos-2 pipeline wrapper (forecast, batch)
│       ├── input_builder.py    # DB loading + covariate alignment
│       ├── finetuner.py        # Fine-tuning pipeline (5000 steps, cosine LR)
│       └── covariate_mapper.py # Country→covariate mapping (ENTSO-E + weather)
├── scripts/
│   ├── train.py              # Training script (enhanced)
│   ├── train_chronos2.py     # Chronos-2 fine-tuning script
│   ├── forecast_daily.py     # Daily forecast job
│   ├── abl335_solar_night_probe.py # Solar forecasts/actuals vs sun geometry
│   ├── forecast_chronos2.py  # Chronos-2 forecast generation
│   ├── compare_experiments.py # Cross-experiment backtest comparison
│   └── scheduler_setup.sh    # Cron setup
├── experiments/        # Versioned experiment configs (V001-Vnnn)
│   ├── registry.json       # Master experiment index
│   └── V00N/config.json    # Per-experiment configuration
├── models/             # Saved model artifacts
└── logs/               # Execution logs
```

### Importing this repo

`src/` is a package and is always imported as one. There is exactly one shape,
and every entry point and test uses it:

```python
sys.path.insert(0, str(Path(__file__).parent.parent))   # repo root, NOT src/
import config                                            # top-level, at the root
from src.db import load_training_data                    # package-qualified
```

Inside `src/`, siblings are imported **relatively** — `from .db import ...`,
`from ..features import ...`. Never `import db`.

This is not style. Putting `src/` on `sys.path` and importing flat gives a module
no parent package, so any relative import inside it raises `ImportError:
attempted relative import with no known parent package` — and where it does not
raise, it silently loads a *second* copy of the module under a second name, with
its own module-level state. `scripts/train.py` was dead by the first mechanism
from ABL-188 (`574eb80`, which added `src/db.py`'s `from .data_quality import
...`) until ABL-340 fixed it: seven months of a documented CLI that could not
run. Nine of 34 scripts were affected; five of them were the `test_*.py` probes
in `scripts/` that also broke bare `pytest` collection (ABL-336).

`tests/test_script_imports.py` holds the line — it executes the module-level
import block of every entry point in `scripts/` and the repo root, rejects any
flat sibling import inside `src/`, and (ABL-354) launches every
`config.MODEL_RUNNERS` entry with `--help` to prove it starts. A new script that
copies an old `sys.path.insert(..., 'src')` preamble fails there rather than
seven months later.

Two consequences worth knowing:

- Anything inside `src/` with a `__main__` — a demo like `src/features.py`, or a
  real entry point like `src/tso_correction_forecaster.py` — needs a parent
  package for its relative imports, so it runs as `python -m src.features`,
  never `python src/features.py`.

  **`config.MODEL_RUNNERS` launches two entry points that live inside `src/`,
  as subprocesses.** `scripts/forecast_daily.py:189` (`build_runner_command`) is
  the one place that builds that argv: a `script` under `src/` becomes
  `-m src.<module>` with `cwd` at the repo root; anything else stays a path.
  Before ABL-354 it was always a path, so `src/tso_correction_forecaster.py` —
  moved to relative imports by ABL-340, as the rule above requires — died at its
  import line on every run, and every BE solar / wind_onshore / wind_offshore
  forecast from the `tso-correction` runner was lost. **The job still exited
  `[DONE]`**: `run_external_model` records a dead subprocess as one failed
  *result*, and the summary line (`Total: 10, Success: 8, Failed: 2`) is the
  only trace. A runner that cannot start reads like a run that went fine, which
  is why the guard now launches them instead of trusting the summary. What the
  summary itself can now say is in "What a runner reports" below (ABL-370).
- `src/evaluation.py` is dead code. `src/evaluation/` is a package and shadows
  it — `src.evaluation` always resolves to the directory. `src/__init__.py:44`
  already has its re-export commented out.

### Help text is ASCII; report bodies are not

`--help` output must be plain ASCII. `scripts/train.py:262` held a literal `→`,
and `python scripts/train.py --help > /dev/null` exited **1** with
`UnicodeEncodeError` — CPython takes stdout's encoding from what stdout *is*, so
a console writes through `WriteConsoleW` and survives, while a pipe or a file
falls back to the locale codepage (cp1252 here) and `argparse` raises inside the
`--help` action itself (ABL-364). Interactively it looked fine; every harness,
CI step and agent that captures stdout saw a traceback instead of usage.

**A module docstring is help text.** 18 of the 38 entry points pass
`description=__doc__`, so an em dash in a docstring is the same defect one
codepage over (cp1252 encodes `—`, cp850 does not). Nine scripts beyond
`train.py` were carrying one.

`tests/test_help_text_encoding.py` holds the line: it reads every entry point's
parser out of the AST — `help=`, `description=`, `epilog=`, and `__doc__` where
it is passed as one — and rejects any character above U+007F. Write `->` and
`--`. It is a static sweep because `--help` on an arbitrary script means
executing that script to module scope, and it runs `scripts/train.py --help`
under `PYTHONIOENCODING=ascii` as the one end-to-end case, so the assertion does
not depend on the codepage of the box.

"Entry point" there means the same set as `test_script_imports.py`: every
`scripts/*.py`, the repo-root runners, **and every `config.MODEL_RUNNERS`
script**. That last group is not decoration — `test_model_runner_launches`
starts each runner with `--help` through a pipe, and two of them
(`src/chronos_forecaster.py`, `src/tso_correction_forecaster.py`) live under
`src/`, outside the `scripts/` glob. Both are ASCII-clean today. Note that
neither passes `description=__doc__` — they use short literals
(`tso_correction_forecaster.py:375`) — so unlike the `scripts/` entry points
above, **their module docstrings are not help text** and an em dash in one is
harmless. What is swept, and what matters, is their `help=`/`description=`
literals: a non-ASCII character there stops the runner from starting, and per
the bullet above `forecast_daily` books a runner that cannot start as a failed
*result* and still prints `[DONE]`.

Report bodies are the deliberate exception and keep `Δ`, `→`, `·`. They are
printed from one known place, so they re-encode the stream there
(`evaluate_net_position.py:125-132`, `compare_challenger.py:127-133`) — which is
not available to `--help`, since argparse prints before any line of `main()`
runs. Do not "fix" this the other way by forcing UTF-8 on stdout at import time:
that is a runtime change in 38 scripts, forgettable in the 39th, and it
re-encodes the log files the `scripts/workstation/*.ps1` jobs capture.
