"""ABL-355: one gate run, one database file.

Both gate harnesses take `--replica-db`, and until this issue it governed
**only** the reads that go through `src/evaluation/scorecard.py` — the incumbent
forecasts, the TSO series and the contamination screen. The fitted series did
not come from it: `RenewableFeatureBuilder` → `_load_actuals_series` →
`db.load_renewable_type_data` → `db.get_connection()`, which opens
`config.DATABASE_PATH` (`ENERGY_DB_PATH`). The weather archive was on the same
path. So one run could fit a challenger on one file, score it against an
incumbent from another, and print a single path under `Replica:` as if it were
the source of everything.

The crash the issue measured — `sqlite3.OperationalError: unable to open
database file` with `ENERGY_DB_PATH` unset — was the *benign* case, and only
because the bare `\\data\\energy_dashboard.db` default does not exist. Point
`ENERGY_DB_PATH` at any real database that is not the replica and there was no
error at all, just a silently cross-sourced gate read. That is what
`test_the_builder_reads_the_file_it_was_handed` pins: two databases that both
open cleanly and disagree, so reading the wrong one is a wrong number rather
than an exception.

This is the ABL-321/331 shape one level down. Those threaded the *table* through
the same chain; the file was still whatever the ambient environment named. A
caller that can select the table but not the database can still be wrong about
where its numbers came from.
"""
import importlib.util
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src import db  # noqa: E402
from src.evaluation.scorecard import (  # noqa: E402
    ScorecardConfig, describe_opened_databases, opened_databases,
)
from src.wind_features import RenewableFeatureBuilder  # noqa: E402

COUNTRY = "XX"
START = pd.Timestamp("2026-01-01")
END = pd.Timestamp("2026-01-08")

#: The two files carry the same instants with different values, so a read of the
#: wrong one returns a full, plausible series rather than an empty frame.
REPLICA_SOLAR_BASE = 100.0
AMBIENT_SOLAR_BASE = 900.0
REPLICA_TEMPERATURE_K = 300.0
AMBIENT_TEMPERATURE_K = 111.0


def _write_db(path: Path, solar_base: float, temperature_k: float) -> Path:
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE energy_renewable (country_code TEXT, timestamp_utc TIMESTAMP,
            data_quality TEXT, solar_mw REAL, wind_onshore_mw REAL,
            wind_offshore_mw REAL);
        CREATE TABLE energy_generation (country_code TEXT, timestamp_utc TIMESTAMP,
            data_quality TEXT, solar_mw REAL, wind_onshore_mw REAL,
            wind_offshore_mw REAL);
        CREATE TABLE weather_data (country_code TEXT, timestamp_utc TIMESTAMP,
            forecast_run_time TIMESTAMP, data_quality TEXT, temperature_2m_k REAL,
            wind_speed_10m_ms REAL, wind_speed_100m_ms REAL,
            shortwave_radiation_wm2 REAL, direct_radiation_wm2 REAL,
            diffuse_radiation_wm2 REAL);
        CREATE TABLE forecasts (country_code TEXT, forecast_type TEXT,
            target_timestamp_utc TIMESTAMP, generated_at TIMESTAMP,
            horizon_hours REAL, forecast_value REAL, model_name TEXT);
        CREATE TABLE energy_generation_forecast (country_code TEXT,
            target_timestamp_utc TIMESTAMP, forecast_type TEXT, solar_mw REAL,
            wind_onshore_mw REAL, wind_offshore_mw REAL);
        """
    )
    hours = pd.date_range(START, END, freq="h")
    for table in db._RENEWABLE_TYPE_SOURCES:
        con.executemany(
            f"INSERT INTO {table} (country_code, timestamp_utc, data_quality, "
            "solar_mw, wind_onshore_mw, wind_offshore_mw) VALUES (?,?,'actual',?,?,?)",
            [(COUNTRY, str(ts), solar_base + i, solar_base + i, solar_base + i)
             for i, ts in enumerate(hours)],
        )
    con.executemany(
        "INSERT INTO weather_data (country_code, timestamp_utc, forecast_run_time, "
        "data_quality, temperature_2m_k, wind_speed_10m_ms, wind_speed_100m_ms, "
        "shortwave_radiation_wm2, direct_radiation_wm2, diffuse_radiation_wm2) "
        "VALUES (?,?,?,'forecast',?,?,?,?,?,?)",
        [(COUNTRY, str(ts), str(START), temperature_k, 5.0, 7.0, 100.0, 60.0, 40.0)
         for ts in hours],
    )
    con.commit()
    con.close()
    return path


@pytest.fixture
def replica(tmp_path):
    return _write_db(tmp_path / "replica.db", REPLICA_SOLAR_BASE, REPLICA_TEMPERATURE_K)


@pytest.fixture
def ambient(tmp_path, monkeypatch):
    """A second, equally valid database installed as `config.DATABASE_PATH`.

    `db.get_connection` reads `config.DATABASE_PATH` at call time, so patching
    the attribute is what an `ENERGY_DB_PATH` pointing somewhere else looks like
    from inside the process.
    """
    path = _write_db(tmp_path / "ambient.db", AMBIENT_SOLAR_BASE, AMBIENT_TEMPERATURE_K)
    monkeypatch.setattr(config, "DATABASE_PATH", path)
    return path


# ---------------------------------------------------------------------------
# The builder — the read site the harnesses could not reach.
# ---------------------------------------------------------------------------


def test_the_builder_reads_the_file_it_was_handed(replica, ambient):
    """The defect, stated as a number rather than as an exception.

    Both files open, both carry the whole window, and their solar series differ
    by 800 MW at every instant. Before `db_path` the builder read `ambient`
    whatever the harness had resolved, so the gate scored a challenger fitted on
    one file against an incumbent loaded from the other.
    """
    builder = RenewableFeatureBuilder(COUNTRY, "solar", START, END, db_path=str(replica))

    assert not builder._actuals.empty
    assert builder._actuals.iloc[0] == REPLICA_SOLAR_BASE
    assert builder.db_path == str(replica)


def test_the_weather_archive_follows_the_same_file(replica, ambient):
    """The second read site, and the one easy to thread halfway.

    `_load_weather_archive` goes through `db.get_connection()` too. A builder
    whose actuals came from the replica and whose weather came from
    `ENERGY_DB_PATH` is still a cross-sourced fit — every weather feature in the
    model would be from the other file.
    """
    builder = RenewableFeatureBuilder(COUNTRY, "solar", START, END, db_path=str(replica))

    assert not builder._weather.empty
    assert set(builder._weather["temperature_2m_k"]) == {REPLICA_TEMPERATURE_K}


def test_an_unhanded_builder_still_reads_the_ambient_path(ambient):
    """Serving passes no `db_path` and must not move.

    `forecast_daily` and every other live caller construct the builder with no
    database argument and depend on `config.DATABASE_PATH`. The fix is an
    override for callers that have already resolved a file, not a new default.
    """
    builder = RenewableFeatureBuilder(COUNTRY, "solar", START, END)

    assert builder.db_path is None
    assert builder._actuals.iloc[0] == AMBIENT_SOLAR_BASE
    assert set(builder._weather["temperature_2m_k"]) == {AMBIENT_TEMPERATURE_K}


def test_a_handed_builder_needs_no_ambient_database_at_all(replica, monkeypatch):
    """The issue's measured crash, inverted into the guarantee that replaces it.

    With `ENERGY_DB_PATH` unset, `config.DATABASE_PATH` degrades to a bare
    `\\data\\energy_dashboard.db` that does not exist (CLAUDE.md records this for
    worktrees), and the run died at `src/db.py` `get_connection` after
    `--replica-db` had already been validated.

    This also fails if some future read site is added to the builder without
    threading `db_path`: pointing the ambient path at a file that cannot be
    opened means any unthreaded read raises here rather than passing unnoticed.
    """
    monkeypatch.setattr(config, "DATABASE_PATH", Path(r"\data\energy_dashboard.db"))

    builder = RenewableFeatureBuilder(COUNTRY, "solar", START, END, db_path=str(replica))

    assert builder._actuals.iloc[0] == REPLICA_SOLAR_BASE
    assert set(builder._weather["temperature_2m_k"]) == {REPLICA_TEMPERATURE_K}


def test_the_loader_reads_the_file_it_was_handed(replica, ambient):
    """`load_renewable_type_data` is public and called directly elsewhere."""
    frame = db.load_renewable_type_data(
        COUNTRY, "solar", "2026-01-01", "2026-01-09", db_path=str(replica))

    assert not frame.empty
    assert frame["target_value"].iloc[0] == REPLICA_SOLAR_BASE


def test_the_table_and_the_file_are_independent(replica, ambient):
    """`source` names the table, `db_path` names the file, and neither implies
    the other. ABL-321/331 threaded the first through this same chain; a caller
    that can select the table but not the database can still be wrong about
    where its numbers came from."""
    for source in db._RENEWABLE_TYPE_SOURCES:
        frame = db.load_renewable_type_data(
            COUNTRY, "solar", "2026-01-01", "2026-01-09",
            source=source, db_path=str(replica))
        assert frame["target_value"].iloc[0] == REPLICA_SOLAR_BASE


def test_a_write_connection_refuses_a_database_argument(tmp_path):
    """Writes keep the single rule `FORECAST_OUTPUT_DB or DATABASE_PATH`.

    Honouring `db_path` for a write would hand any caller a way around the
    replica-purity guard (`tests/test_train_sidecar_guard.py`); silently
    ignoring it would be this issue again with the sign flipped — a caller
    naming a target that is not the one written. Refusing is the only option
    that cannot produce a wrong file quietly.
    """
    with pytest.raises(ValueError, match="read-only override"):
        with db.get_connection(readonly=False, db_path=str(tmp_path / "anything.db")):
            pass


# ---------------------------------------------------------------------------
# The record and the report — what the run says about its own sources.
# ---------------------------------------------------------------------------


def _config(replica, sidecar=None):
    return ScorecardConfig(str(replica), str(sidecar) if sidecar else None,
                           START, END, models={"solar": "catboost"})


def test_the_record_names_the_feature_database_separately(replica, tmp_path):
    """`features` is a separate field even though both harnesses now make it
    equal to `replica`. What ABL-355 cost was the absence of the record: a run
    that had read two files was indistinguishable from one that had read one."""
    record = opened_databases(_config(replica), str(replica), Path("/somewhere/else.db"))

    assert record["replica"] == str(replica.resolve())
    assert record["features"] == str(replica.resolve())
    assert record["features_match_replica"] is True
    assert record["ambient_energy_db_path"] == str(Path("/somewhere/else.db"))


def test_the_record_does_not_claim_a_sidecar_that_was_never_opened(replica, tmp_path):
    """`_load_forecasts` reads the sidecar only when it exists — and the
    harnesses default `--sidecar-db` to `str(config.FORECAST_OUTPUT_DB)`, which
    is the literal string `'None'` when that variable is unset. A record that
    named it anyway would report a file no run ever opened."""
    missing = _config(replica, tmp_path / "no-sidecar.db")
    assert opened_databases(missing, str(replica), replica)["sidecar"] is None

    unset = _config(replica, "None")
    assert opened_databases(unset, str(replica), replica)["sidecar"] is None

    present = _write_db(tmp_path / "sidecar.db", 1.0, 1.0)
    record = opened_databases(_config(replica, present), str(replica), replica)
    assert record["sidecar"] == str(present.resolve())


def test_the_report_names_every_file_the_run_opened(replica, tmp_path):
    sidecar = _write_db(tmp_path / "sidecar.db", 1.0, 1.0)
    record = opened_databases(_config(replica, sidecar), str(replica), replica)

    text = "\n".join(describe_opened_databases(record, 4096))

    assert str(replica.resolve()) in text
    assert str(sidecar.resolve()) in text
    assert "4,096 bytes" in text


def test_the_report_does_not_claim_one_file_when_a_sidecar_was_opened(replica, tmp_path):
    """The incumbent is the one read the replica does not hold alone.

    `_load_forecasts` opens the sidecar too when it exists, and a sidecar row
    wins an exact vintage match. A report that printed "every read in this run
    comes from that one file" over an opened sidecar would be ABL-355 itself,
    reprinted inside its own fix: one path named for reads that came from two.
    """
    sidecar = _write_db(tmp_path / "sidecar.db", 1.0, 1.0)

    with_sidecar = "\n".join(describe_opened_databases(
        opened_databases(_config(replica, sidecar), str(replica), replica), 1))
    without = "\n".join(describe_opened_databases(
        opened_databases(_config(replica), str(replica), replica), 1))

    assert "Every read in this run comes from that one file" not in with_sidecar
    assert "the only read it does not hold alone" in with_sidecar
    assert "the sidecar's is the one scored" in with_sidecar
    assert "Every read in this run comes from that one file" in without


def test_the_ambient_path_is_compared_as_a_file_not_as_a_string(replica, tmp_path):
    """`--replica-db` defaults to `str(config.DATABASE_PATH)`, so the usual run
    has one file under two spellings. Comparing the strings would print "not
    read by this run" about the very file every read came from — a false line
    in the report this issue exists to make true."""
    (tmp_path / "sub").mkdir()
    same_file_other_spelling = tmp_path / "sub" / ".." / replica.name

    record = opened_databases(_config(replica), str(replica), same_file_other_spelling)

    assert str(same_file_other_spelling) != record["replica"]
    assert record["ambient_matches_replica"] is True
    assert "was **not** read by this run" not in "\n".join(
        describe_opened_databases(record, 1))


def test_the_report_names_an_unread_ambient_path_when_it_differs(replica):
    """A reader comparing this report to one written before ABL-355 needs to see
    whether the two would have diverged. When `ENERGY_DB_PATH` is the replica,
    saying so twice is noise; when it is not, silence hides the whole finding."""
    elsewhere = opened_databases(_config(replica), str(replica), Path("/other/energy.db"))
    same = opened_databases(_config(replica), str(replica), replica.resolve())

    assert "was **not** read by this run" in "\n".join(describe_opened_databases(elsewhere, 1))
    assert "was **not** read by this run" not in "\n".join(describe_opened_databases(same, 1))


def test_the_report_calls_a_cross_sourced_run_unpublishable(replica, tmp_path):
    """Unreachable from either harness, and kept for the reason the harnesses
    are not the only possible caller: if some future one does split them, the
    report has to say so rather than print one path for two files."""
    split = opened_databases(_config(replica), str(tmp_path / "elsewhere.db"), replica)

    text = "\n".join(describe_opened_databases(split, 1))

    assert split["features_match_replica"] is False
    assert "Cross-sourced run" in text
    assert "unpublishable" in text
    assert str((tmp_path / "elsewhere.db").resolve()) in text


# ---------------------------------------------------------------------------
# The harnesses — that they pass the resolved replica through at all.
# ---------------------------------------------------------------------------


def _load_harness(name):
    """Import a `scripts/` harness as a module object, `__main__` guard shut."""
    spec = importlib.util.spec_from_file_location(
        f"scripts_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _BuilderSpy(Exception):
    """Captures the builder's construction arguments and stops the run there.

    Everything past the first `RenewableFeatureBuilder` needs a real fit; none
    of it can change the argument already captured.
    """

    def __init__(self, kwargs):
        super().__init__("builder constructed")
        self.kwargs = kwargs


@pytest.mark.parametrize("name", ["evaluate_solar_retrain", "evaluate_wind_retrain"])
def test_the_harness_hands_the_builder_the_resolved_replica(name, replica, ambient,
                                                            monkeypatch, tmp_path):
    """Both harnesses are affected identically, so both are pinned identically.

    Asserted against the *resolved* replica, not against `args.replica_db`: the
    harnesses resolve once and record that string in `meta`, so a relative path
    reaching the builder would mean the report could name a file the run did not
    open.
    """
    harness = _load_harness(name)

    def spy(*args, **kwargs):
        raise _BuilderSpy(kwargs)

    monkeypatch.setattr(harness, "RenewableFeatureBuilder", spy)
    monkeypatch.setattr(sys, "argv", [
        f"{name}.py",
        "--replica-db", str(replica),
        # Does not exist, so `_load_forecasts` reads the replica's empty
        # `forecasts` table only and never touches a real sidecar.
        "--sidecar-db", str(tmp_path / "no-sidecar.db"),
    ])

    with pytest.raises(_BuilderSpy) as caught:
        harness.main()

    assert caught.value.kwargs["db_path"] == str(replica.resolve())
    assert caught.value.kwargs["db_path"] != str(ambient)
