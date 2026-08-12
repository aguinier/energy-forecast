"""ABL-346: `scripts/train.py` must refuse to start when no sidecar resolves.

`src/db.py:48` picks a write target with

    target = getattr(config, 'FORECAST_OUTPUT_DB', None) or config.DATABASE_PATH

and `config.py:23` is `os.getenv('FORECAST_OUTPUT_DB')` -- `None` unless the
environment supplies it. There is no default and no assertion, so an unset
variable does not fail: the `or` falls through and **every** write connection
targets the replica. `scripts/train.py` opened with `initialize_all_tables()`
(DDL, on a write connection) before any training happened, and contained no
occurrence of `FORECAST_OUTPUT_DB` at all.

That made "tranche runs will not touch the shared database" true only
*conditionally*, on an environment variable nothing checked, in a runner whose
environment has already been wrong once. ABL-316 §4 routes 37 country/stream
pairs through this entry point.

Its sibling already refuses the same case (`forecast_challengers.py:322-325`);
this ports that guard.

**Placement is the property, not the message.** A check that runs after the
first write connection is decorative, so the tests below are written to fail if
the guard is deleted *or* moved after `initialize_all_tables()` -- not merely if
its wording changes:

* `test_no_table_is_created_when_no_sidecar_resolves` spies on
  `initialize_all_tables` and asserts it is never reached. Moving the guard below
  it turns `[]` into `['called']`.
* `test_the_first_ddl_lands_on_the_sidecar_not_the_replica` opens a real write
  connection *inside* that spy, so what it measures is where `src/db.py` actually
  resolves at the instant of the first DDL -- not a copy of db.py's expression
  re-asserted in the test.
* `test_a_refused_run_leaves_the_replica_untouched` is the end-to-end case from
  the issue's acceptance, in a subprocess, against a fixture replica seeded so
  that `config.validate_config()` genuinely **passes**. That seeding is what
  makes the assertion non-vacuous: measured 2026-08-12, with the guard removed
  the same fixture gains `forecasts`, `model_evaluations`, `deployed_models` and
  `forecast_runs`. Without the seed, `validate_config()` would exit first and the
  replica would stay clean for a reason that has nothing to do with the guard.
"""
import importlib.util
import logging
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src import db  # noqa: E402

TRAIN_PY = ROOT / "scripts" / "train.py"

#: The tables `initialize_all_tables()` creates (`src/db.py:1752-1760`). Any of
#: them appearing in the replica after a refused run means the guard did not stop
#: the run before the first write connection.
DDL_TABLES = {"forecasts", "model_evaluations", "deployed_models", "forecast_runs"}


def _load_train():
    """Import `scripts/train.py` under a name that keeps its `__main__` guard
    shut, so `main()` can be called directly instead of only through a
    subprocess."""
    spec = importlib.util.spec_from_file_location("scripts_train_abl346", TRAIN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train = _load_train()


def _seed_plausible_replica(path):
    """Write the minimum that `config.validate_config()` accepts as the live
    replica: a `net_position` table holding a recent row for each of
    `DB_CURRENCY_PROBE_COUNTRIES` (`config.py:306`, `config.py:372-410`).

    Recent rather than fixed, because the currency check is relative to
    `datetime.now()` with a 48h bound (`config.py:311`) -- a hardcoded date would
    make this file start failing two days after it was written.
    """
    recent = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE net_position (country_code TEXT, timestamp_utc TEXT)")
    con.executemany(
        "INSERT INTO net_position VALUES (?, ?)",
        [(c, recent) for c in config.DB_CURRENCY_PROBE_COUNTRIES],
    )
    con.commit()
    con.close()
    return path


def _tables(path):
    con = sqlite3.connect(path)
    try:
        return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        con.close()


# ---------------------------------------------------------------------------
# The pure helper.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("supplied", [None, "", "   "])
def test_nothing_configured_resolves_to_none(supplied):
    """The three ways of supplying nothing. The two blank forms are not
    hypothetical: `FORECAST_OUTPUT_DB=` in a `.env`, and
    `--sidecar-db "$env:FORECAST_OUTPUT_DB"` with the variable unset, both reach
    argparse as a string that is *present* and means the opposite."""
    assert train.resolve_sidecar_db(supplied) is None


def test_a_configured_path_survives_unchanged():
    assert train.resolve_sidecar_db(r"C:\sidecar\forecasts.db") == r"C:\sidecar\forecasts.db"


def test_a_sidecar_that_does_not_exist_yet_is_still_a_sidecar():
    """`None` must mean "no target resolved", not "the file is missing". A
    sidecar is created on first write, so requiring it to pre-exist would refuse
    the first run on a clean workstation."""
    assert train.resolve_sidecar_db("/nonexistent/dir/side.db") == "/nonexistent/dir/side.db"


# ---------------------------------------------------------------------------
# Placement: the guard runs before the first write connection.
# ---------------------------------------------------------------------------


@pytest.fixture
def guard_probe(monkeypatch, tmp_path):
    """Runs `main()` far enough to observe `initialize_all_tables`, and no
    further.

    `validate_config` is stubbed out deliberately. Under the bug this test has
    to reach `initialize_all_tables()` for the assertion to mean anything, and
    the real `validate_config` would exit first against a tmp replica -- leaving
    "no tables were created" true for the wrong reason. What it checks is covered
    by `tests/test_db_currency.py`.
    """
    calls = []

    class _Stop(Exception):
        """Ends the run at the first DDL; nothing past it is under test."""

    def spy():
        calls.append("called")
        raise _Stop

    monkeypatch.setattr(train, "setup_logging", lambda: logging.getLogger("abl346"))
    monkeypatch.setattr(train.config, "validate_config", lambda: True)
    monkeypatch.setattr(train, "initialize_all_tables", spy)
    monkeypatch.setattr(config, "DATABASE_PATH", tmp_path / "replica.db")

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["train.py", *argv])
        try:
            return train.main(), calls
        except _Stop:
            return None, calls

    return run


def test_no_table_is_created_when_no_sidecar_resolves(guard_probe, monkeypatch):
    """The core placement assertion. Delete the guard, or move it below
    `initialize_all_tables()`, and `calls` becomes non-empty."""
    monkeypatch.setattr(config, "FORECAST_OUTPUT_DB", None)

    rc, calls = guard_probe("--countries", "DE", "--types", "renewable")

    assert calls == [], (
        "initialize_all_tables() ran with no sidecar configured. Its DDL goes to "
        "a write connection, and src/db.py resolves that to DATABASE_PATH when "
        "FORECAST_OUTPUT_DB is unset - so this is the replica being written to. "
        "The guard is missing, or it sits below the call it is supposed to "
        "precede."
    )
    assert rc == train.SIDECAR_REQUIRED_EXIT


def test_the_first_ddl_lands_on_the_sidecar_not_the_replica(monkeypatch, tmp_path):
    """The positive half, measured rather than asserted from a copy of db.py's
    expression: open a real write connection at the moment of the first DDL and
    see which file it created.

    This is what makes `--sidecar-db` more than decoration. Training writes
    through `src/db.py`'s module-level helpers, which read
    `config.FORECAST_OUTPUT_DB` per connection rather than taking a path, so a
    flag parsed into `args` and left there would satisfy the guard while every
    write still went to the replica.
    """
    replica = tmp_path / "replica.db"
    sidecar = tmp_path / "sidecar.db"

    class _Stop(Exception):
        pass

    def spy():
        with db.get_connection(readonly=False) as conn:
            conn.execute("CREATE TABLE ddl_probe (x INTEGER)")
        raise _Stop

    monkeypatch.setattr(train, "setup_logging", lambda: logging.getLogger("abl346"))
    monkeypatch.setattr(train.config, "validate_config", lambda: True)
    monkeypatch.setattr(train, "initialize_all_tables", spy)
    monkeypatch.setattr(config, "DATABASE_PATH", replica)
    monkeypatch.setattr(config, "FORECAST_OUTPUT_DB", None)  # flag is the only source
    monkeypatch.setattr(
        sys, "argv",
        ["train.py", "--countries", "DE", "--types", "renewable", "--sidecar-db", str(sidecar)],
    )

    with pytest.raises(_Stop):
        train.main()

    assert sidecar.exists() and "ddl_probe" in _tables(sidecar), (
        "--sidecar-db was accepted but the write did not land there; the flag is "
        "decorative unless main() also assigns config.FORECAST_OUTPUT_DB"
    )
    assert not replica.exists(), f"the replica was written to at {replica}"


def test_the_environment_variable_alone_is_enough(guard_probe, monkeypatch, tmp_path):
    """No flag, variable set: the run proceeds exactly as it does today. This
    case cannot fail under the bug, and that is the point of it -- it pins that
    the guard did not change the invocation every existing training run uses."""
    monkeypatch.setattr(config, "FORECAST_OUTPUT_DB", str(tmp_path / "sidecar.db"))

    rc, calls = guard_probe("--countries", "DE", "--types", "renewable")

    assert calls == ["called"], "a configured sidecar was refused; the guard is too strict"
    assert rc is None  # stopped inside the spy, not returned from the guard


# ---------------------------------------------------------------------------
# End to end: the issue's acceptance criteria, through the real CLI.
# ---------------------------------------------------------------------------


def _run_train(tmp_path, argv, sidecar_env):
    """`scripts/train.py` in a subprocess against a fixture replica.

    `sidecar_env` of `None` removes FORECAST_OUTPUT_DB from the child's
    environment. `config.py:11` calls `load_dotenv()`, so a developer's
    gitignored `.env` could in principle put it back -- `load_dotenv` does not
    override keys already present, but it does supply absent ones. That would
    make these tests *fail* loudly on exit code, not pass vacuously, which is the
    right failure mode. Measured 2026-08-12: the checkout's `.env` does not
    define it.
    """
    import os

    replica = _seed_plausible_replica(tmp_path / "replica.db")
    env = os.environ.copy()
    env["ENERGY_DB_PATH"] = str(replica)
    env.pop("ALLOW_STALE_DB", None)  # must not mask the currency check
    if sidecar_env is None:
        env.pop("FORECAST_OUTPUT_DB", None)
    else:
        env["FORECAST_OUTPUT_DB"] = sidecar_env

    proc = subprocess.run(
        [sys.executable, str(TRAIN_PY), *argv],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600,
    )
    return proc, replica


def test_a_refused_run_leaves_the_replica_untouched(tmp_path):
    """The acceptance case: variable unset, no flag, real CLI.

    Asserts the exit code rather than merely "non-zero" because several
    unrelated startup failures also exit non-zero -- a configuration error exits
    1. Without this, a guard accidentally moved below `validate_config()` would
    still look like it passed.
    """
    proc, replica = _run_train(tmp_path, ["--countries", "DE", "--types", "renewable"], None)

    assert proc.returncode == train.SIDECAR_REQUIRED_EXIT, (
        f"expected exit {train.SIDECAR_REQUIRED_EXIT}, got {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    assert "FORECAST_OUTPUT_DB" in proc.stdout + proc.stderr, (
        "refused without naming the variable the operator has to set"
    )
    leaked = DDL_TABLES & _tables(replica)
    assert not leaked, (
        f"the refused run created {sorted(leaked)} in the replica at {replica}. "
        "validate_config() passes against this fixture, so execution reached "
        "initialize_all_tables() - the guard is below it."
    )


def test_an_empty_variable_is_refused_like_an_absent_one(tmp_path):
    """`FORECAST_OUTPUT_DB=` reads as configured to a truthiness check on
    presence, and resolves to the replica in `src/db.py`'s `or`, exactly like
    unset."""
    proc, replica = _run_train(tmp_path, ["--countries", "DE", "--types", "renewable"], "")

    assert proc.returncode == train.SIDECAR_REQUIRED_EXIT, proc.stdout + proc.stderr
    assert not DDL_TABLES & _tables(replica)


def test_help_still_exits_zero(tmp_path):
    """ABL-340 restored `--help` after seven months of `ImportError`. A guard
    that refuses before argparse can print usage would undo it: an operator
    trying to find out *how* to set the sidecar would be refused for not having
    set it."""
    proc, _ = _run_train(tmp_path, ["--help"], None)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "--sidecar-db" in proc.stdout, "the new flag is not documented in --help"
