"""ABL-341: `scripts/train.py`'s `--optuna` / `--feature-selection` prep block
must ask for the same table the artifact will be fitted on.

ABL-339 (`1a133d6`) threaded `renewable_source` into that block. Before it, the
block probed freshness through the global `RENEWABLE_TYPE_SOURCE_TABLE` and
loaded the target series with **no `source` at all**, while the `Forecaster`
constructed ~50 lines later already got `training_source=renewable_source`. With
`--renewable-source energy_generation` that tuned the hyperparameters and
selected the feature set on `energy_renewable`, then fitted the shipped model on
`energy_generation` — a train/train skew inside a single run, in the new flag's
own code path. Nothing in the logs distinguishes it: the run reports a trained
model, the model *is* fitted on the table the operator asked for, and only the
hyperparameters and the feature set come from the other one.

That hunk shipped with no test. ABL-339's 280-test suite covers `src/db.py` and
`src/forecaster.py` but cannot reach `scripts/train.py`, which does not import
at all under a plain `import`: flat `import db` (`scripts/train.py:37`) against
`src/db.py:17`'s `from .data_quality import ...` (ABL-340; root cause ABL-188,
`574eb80`). `_load_train_script` below reaches it anyway, by aliasing the four
`src/` modules that only fail *flat* under the flat names `train.py` asks for —
the same technique `test_the_walk_forward_window_asks_the_same_question` in
`tests/test_per_artifact_training_source.py` already uses for `validation`.

The shim is a test fixture, not a fix: `python scripts/train.py` is still dead
until ABL-340 lands, and this file does not make it less dead. What it does buy
is that the argument threading is pinned now rather than after, and that ABL-340
cannot rebind these names to something else without this failing.

**Why the assertions are on the arguments and not on the result.** The two
tables carry the same series for most country/type pairs, so a test that fitted
a model each way and compared outputs would pass under the bug for nearly every
pair. The defect is visible only at the call site: which table each of the three
consumers was *asked* for.
"""
import importlib
import importlib.util
import logging
import sys
import types
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

#: The `src/` modules that import fine as `src.X` but not as a bare `X`, because
#: they use package-relative imports. `scripts/train.py` asks for all four under
#: their flat names. Measured on 2026-08-12 against `87edd50`: the other seven
#: (`config`, `metrics`, `baselines`, `model_registry`, `hyperopt`,
#: `feature_selection`, `features`) import flat unaided.
_FLAT_ALIASES = ("db", "forecaster", "deployment", "validation")


def _load_train_script():
    """Import `scripts/train.py` as a module object.

    Loaded under the name `scripts_train`, so the file's `if __name__ ==
    '__main__'` guard stays shut and nothing here can be confused with the real
    `train` module ABL-336 is about.

    Aliasing rather than stubbing is deliberate: `train.db` ends up being the
    very `src.db` this test suite imports elsewhere, so what the spies below
    replace is the real function object, and a signature the real `db` does not
    accept would still be an error.

    The aliases are installed unconditionally and restored afterwards rather
    than left in place with `setdefault`. Once ABL-340 lands, a bare `import db`
    may start succeeding on its own, and it would then produce a module object
    *distinct* from `src.db` — a `setdefault` would keep whichever one some
    earlier test file happened to import first, and the identity this file
    patches through would depend on collection order. Restoring also keeps the
    rest of the session's `sys.modules` exactly as it was found.
    """
    saved = {flat: sys.modules.get(flat) for flat in _FLAT_ALIASES}
    try:
        for flat in _FLAT_ALIASES:
            sys.modules[flat] = importlib.import_module("src." + flat)
        spec = importlib.util.spec_from_file_location(
            "scripts_train", ROOT / "scripts" / "train.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for flat, previous in saved.items():
            if previous is None:
                sys.modules.pop(flat, None)
            else:
                sys.modules[flat] = previous
    return module


train = _load_train_script()

COUNTRY = "XX"
TYPE = "wind_onshore"

#: The freshness probe's answer, so the window the loader is handed is a
#: readout of it rather than of `datetime.now()`.
LATEST = datetime(2026, 8, 11, 12, 0)
EXPECTED_END = "2026-08-12"  # LATEST + 1 day, per train.py:485


class _Recorder:
    """Records every call's keyword arguments. `calls` rather than a single slot
    so a call site that fires twice cannot be mistaken for one that fired once.
    """

    def __init__(self, result=None):
        self.calls = []
        self._result = result

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self._result

    @property
    def only(self):
        assert len(self.calls) == 1, f"expected exactly one call, got {self.calls}"
        return self.calls[0]


@pytest.fixture
def spies(monkeypatch):
    """Replaces the three consumers of `renewable_source` in `train_model` with
    recorders, and stops the run at the first one whose return value the code
    checks.

    `load_training_data` returns an empty frame, which `train.py:493` turns into
    `ValueError("No training data for ...")`. `train_model` catches everything
    into `result['error']`, so that string is also the evidence that execution
    reached the loader instead of falling over somewhere earlier — a spy that
    was never called would otherwise read the same as one whose assertion
    passed vacuously.
    """
    probe = _Recorder(result=LATEST)
    loader = _Recorder(result=pd.DataFrame())
    forecaster = _Recorder()

    monkeypatch.setattr(train, "get_latest_data_timestamp", probe)
    monkeypatch.setattr(train.db, "load_training_data", loader)
    monkeypatch.setattr(train, "Forecaster", forecaster)
    # Outside `train_model`'s try/except, and it touches the real models/
    # directory. Nothing under test reads it.
    monkeypatch.setattr(train, "get_registry", lambda: types.SimpleNamespace())

    return types.SimpleNamespace(probe=probe, loader=loader, forecaster=forecaster)


def _train_model(renewable_source, **overrides):
    kwargs = dict(
        country_code=COUNTRY,
        forecast_type=TYPE,
        start_date="2026-01-01",
        end_date=None,  # the open window, so the freshness probe is consulted
        algorithm="catboost",
        hyperparams=None,
        grid_search=False,
        grid_params=None,
        logger=logging.getLogger("test_abl341"),
        run_evaluation=False,
        use_optuna=True,
        feature_selection=False,
        renewable_source=renewable_source,
    )
    kwargs.update(overrides)
    return train.train_model(**kwargs)


# ---------------------------------------------------------------------------
# The two call sites in the prep block.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["--optuna", "--feature-selection"])
def test_the_prep_block_probes_the_table_the_run_will_train_on(spies, flag):
    """`--renewable-source energy_generation` reaches the freshness probe.

    Both flags open the same block (`train.py:470`), and both are reachable
    alone, so both are exercised: a fix applied to one entry condition only
    would leave the other loading from the wrong table.
    """
    result = _train_model(
        "energy_generation",
        use_optuna=(flag == "--optuna"),
        feature_selection=(flag == "--feature-selection"),
    )

    assert result["error"] == f"No training data for {COUNTRY} {TYPE}", (
        f"the {flag} prep block did not reach the loader; it failed earlier "
        f"with {result['error']!r}, so the assertions below would prove nothing"
    )
    assert spies.probe.only[1].get("source") == "energy_generation", (
        "the prep block asked for the freshness of a table other than the one "
        "the run was told to train on. An open window then closes on the wrong "
        "table's last instant — truncated where it lags, and left to "
        "datetime.now() where the pair has no rows in it at all."
    )


@pytest.mark.parametrize("flag", ["--optuna", "--feature-selection"])
def test_the_prep_block_loads_the_target_series_from_that_same_table(spies, flag):
    """The half that was worse than a truncated window: pre-ABL-339 this call
    passed no `source` at all, so the tuning and the feature selection ran on
    `energy_renewable` while the shipped model was fitted on
    `energy_generation`.
    """
    _train_model(
        "energy_generation",
        use_optuna=(flag == "--optuna"),
        feature_selection=(flag == "--feature-selection"),
    )

    args, kwargs = spies.loader.only
    assert kwargs.get("source") == "energy_generation", (
        "the Optuna / feature-selection prep loaded the target series from a "
        "table other than the one the artifact will be fitted on. The run still "
        "reports a trained model and the model is still fitted on the requested "
        "table; only the hyperparameters and the selected feature set come from "
        "the other one, and nothing in the logs says so."
    )


def test_the_window_the_loader_gets_is_the_one_the_probe_resolved(spies):
    """Ties the two call sites together: the probe's answer is what closes the
    open window. If the loader were reading its end date from anywhere else,
    asserting each call's `source` separately would still pass while the two
    described different windows.
    """
    _train_model("energy_generation")

    args, kwargs = spies.loader.only
    # `end_date` is the fourth positional at train.py:488; accept either form so
    # this does not fail on a purely cosmetic change to the call.
    end_date = kwargs["end_date"] if "end_date" in kwargs else args[3]
    assert end_date == EXPECTED_END, (
        f"the loader was given {args!r} / {kwargs!r}; expected the window to "
        f"close at {EXPECTED_END}, one day past the probe's {LATEST}"
    )


# ---------------------------------------------------------------------------
# The three-way agreement — this is the property, not the two call sites alone.
# ---------------------------------------------------------------------------


def test_probe_loader_and_forecaster_are_all_asked_for_the_same_table(spies):
    """The defect was never one call site being wrong in isolation; it was two
    of the three disagreeing with the third. The `Forecaster` leg is what the
    artifact is actually fitted on, so it is the one the other two have to
    match.

    Reached with the prep block switched off, because with it on the empty-frame
    stop above fires first — by design, since letting the prep block run to
    completion would mean running a real feature build and a real Optuna study.
    """
    _train_model("energy_generation", use_optuna=False, feature_selection=False)
    fitted_on = spies.forecaster.only[1].get("training_source")

    _train_model("energy_generation")
    probed = spies.probe.only[1].get("source")
    loaded = spies.loader.only[1].get("source")

    assert probed == loaded == fitted_on == "energy_generation", (
        f"probe asked for {probed!r}, loader asked for {loaded!r}, Forecaster "
        f"was constructed on {fitted_on!r}. All three have to name one table or "
        "the run tunes on one and ships a model fitted on another."
    )


# ---------------------------------------------------------------------------
# Back-compatibility — a legacy invocation is byte-for-byte unchanged.
# ---------------------------------------------------------------------------


def test_no_renewable_source_flag_passes_the_pre_change_default(spies):
    """`--renewable-source` defaults to `None` (`train.py:248`), and `None` is
    exactly what the pre-ABL-339 call sites effectively passed: the probe took
    no `source` argument and the loader took none either, both of which
    `src/db.py` resolves through `RENEWABLE_TYPE_SOURCE_TABLE`.

    This case cannot fail under the bug, and that is the point of it — it is the
    assertion that the fix did not move the invocation every existing training
    run uses. `test_the_freshness_probe_default_is_the_pre_change_table` in
    `tests/test_per_artifact_training_source.py` pins what `db` then resolves
    that `None` to; here it is only that nothing else is substituted for it.
    """
    _train_model(None)

    assert spies.probe.only[1].get("source", "absent") in (None, "absent")
    assert spies.loader.only[1].get("source", "absent") in (None, "absent")


def test_the_pre_change_default_resolves_to_energy_renewable(spies):
    """The `None` above is only back-compatible because `db` turns it into
    `energy_renewable`. Stated here so the previous test is a statement about a
    table and not about a sentinel: if the global default were ever flipped, the
    legacy path would move and this fails rather than the `None` assertion
    silently continuing to pass.
    """
    from src import db

    assert db.RENEWABLE_TYPE_SOURCE_TABLE == "energy_renewable"


# ---------------------------------------------------------------------------
# Guard on the fixture itself.
# ---------------------------------------------------------------------------


def test_a_closed_window_does_not_consult_the_probe_at_all(spies):
    """`end_date` is only resolved when it is `None` (`train.py:479`). Pins that
    the probe assertions above are measuring the open-window branch, and that a
    caller who named its own end date still gets it.
    """
    _train_model("energy_generation", end_date="2026-03-01")

    assert spies.probe.calls == []
    assert spies.loader.only[1].get("source") == "energy_generation"


def test_the_spies_replaced_the_real_functions(spies):
    """A recorder bound to a name nothing calls records nothing, and every
    assertion above would then fail on `only` rather than pass vacuously — but
    it would fail for the wrong reason. Check the wiring directly."""
    from src import db

    assert train.db is db, (
        "train.db is not the src.db this suite patches; the alias shim in "
        "_load_train_script no longer matches how scripts/train.py imports"
    )
    assert sys.modules.get("db") is not train.db, (
        "the alias shim leaked into sys.modules; _load_train_script is supposed "
        "to restore it, so a later test file's own `db` import is unaffected"
    )
    assert train.get_latest_data_timestamp is spies.probe
    assert train.db.load_training_data is spies.loader
