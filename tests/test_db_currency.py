"""Startup validation must reject a *stale but present* database (ABL-73).

`DATABASE_PATH.exists()` passed against a 3.0 GB partial snapshot whose
net_position ends 2024-01-15 and holds zero rows for AT and DE. A training run
against it produces a 19-country program with the priority majors silently
missing and a backtest whose numbers look fine.
"""
import importlib
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

NOW = datetime(2026, 8, 7, 21, 0, tzinfo=timezone.utc)


def _config(monkeypatch, tmp_path, db_name="replica.db"):
    monkeypatch.setenv("ENERGY_DB_PATH", str(tmp_path / db_name))
    import config
    importlib.reload(config)
    return config


def _make_db(path, rows):
    """rows: list of (country_code, timestamp_utc)."""
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE net_position ("
        "  country_code TEXT NOT NULL,"
        "  timestamp_utc TIMESTAMP NOT NULL,"
        "  net_position_mw REAL,"
        "  PRIMARY KEY (country_code, timestamp_utc))"
    )
    con.executemany(
        "INSERT INTO net_position (country_code, timestamp_utc, net_position_mw) "
        "VALUES (?, ?, 100.0)",
        rows,
    )
    con.commit()
    con.close()


# ---------------------------------------------------------------------------
# The pure classifier
# ---------------------------------------------------------------------------

def test_current_replica_has_no_problems(monkeypatch, tmp_path):
    config = _config(monkeypatch, tmp_path)
    latest = {c: "2026-08-07 21:00:00" for c in config.DB_CURRENCY_PROBE_COUNTRIES}
    assert config.classify_db_currency(latest, NOW) == []


def test_zero_rows_is_reported_per_country(monkeypatch, tmp_path):
    """The decoy's actual shape for AT and DE."""
    config = _config(monkeypatch, tmp_path)
    latest = {"BE": "2026-08-07 21:00:00", "AT": None, "DE": None}
    problems = config.classify_db_currency(latest, NOW)
    assert len(problems) == 2
    assert any(p.startswith("AT: no net_position rows at all") for p in problems)
    assert any(p.startswith("DE: no net_position rows at all") for p in problems)


def test_ancient_rows_are_reported_even_though_present(monkeypatch, tmp_path):
    """The decoy's actual shape for BE/NL/FR: rows exist, but stop in 2023-24.

    This is the case `DATABASE_PATH.exists()` could never catch.
    """
    config = _config(monkeypatch, tmp_path)
    latest = {"BE": "2023-02-28 23:00:00", "NL": "2023-01-31 23:00:00",
              "FR": "2024-01-15 23:00:00"}
    problems = config.classify_db_currency(latest, NOW)
    assert len(problems) == 3
    assert all("old, limit 48h" in p for p in problems)


def test_future_timestamps_are_not_stale(monkeypatch, tmp_path):
    """net_position is day-ahead — a healthy replica reaches tomorrow's market
    day. Only staleness is disqualifying, never freshness."""
    config = _config(monkeypatch, tmp_path)
    ahead = (NOW + timedelta(hours=30)).strftime("%Y-%m-%d %H:%M:%S")
    assert config.classify_db_currency({"BE": ahead}, NOW) == []


def test_both_timestamp_separators_are_accepted(monkeypatch, tmp_path):
    """The column holds both `2026-07-20T00:00:00` and `2026-07-20 00:00:00`."""
    config = _config(monkeypatch, tmp_path)
    assert config.classify_db_currency({"BE": "2026-08-07T21:00:00"}, NOW) == []
    assert config.classify_db_currency({"BE": "2026-08-07 21:00:00"}, NOW) == []


def test_trailing_offset_is_converted_not_rejected(monkeypatch, tmp_path):
    """A minority of rows carry `+02:00`; it must be honoured as an offset."""
    config = _config(monkeypatch, tmp_path)
    assert config.classify_db_currency({"BE": "2026-08-07 23:00:00+02:00"}, NOW) == []


def test_unparseable_timestamp_is_a_problem_not_a_crash(monkeypatch, tmp_path):
    config = _config(monkeypatch, tmp_path)
    problems = config.classify_db_currency({"BE": "not-a-timestamp"}, NOW)
    assert len(problems) == 1
    assert "unparseable" in problems[0]


def test_threshold_is_not_a_tuned_edge(monkeypatch, tmp_path):
    """Any bound from ~2 days to ~1 year separates the replica from the decoy."""
    config = _config(monkeypatch, tmp_path)
    replica = {"BE": "2026-08-07 21:00:00"}
    decoy = {"BE": "2024-01-15 23:00:00"}
    for hours in (48, 24 * 7, 24 * 30, 24 * 365):
        assert config.classify_db_currency(replica, NOW, max_age_hours=hours) == []
        assert config.classify_db_currency(decoy, NOW, max_age_hours=hours) != []


# ---------------------------------------------------------------------------
# The database probe
# ---------------------------------------------------------------------------

def test_probe_reports_none_for_a_country_with_no_rows(monkeypatch, tmp_path):
    config = _config(monkeypatch, tmp_path)
    db = tmp_path / "replica.db"
    _make_db(db, [("BE", "2026-08-07 21:00:00"), ("NL", "2026-08-07 21:00:00")])
    latest = config.probe_net_position_latest(db)
    assert set(latest) == set(config.DB_CURRENCY_PROBE_COUNTRIES)
    assert latest["BE"] == "2026-08-07 21:00:00"
    assert latest["AT"] is None and latest["DE"] is None


def test_probe_normalises_separator_before_taking_max(monkeypatch, tmp_path):
    """'T' (84) sorts above ' ' (32), so a raw MAX() picks the older row."""
    config = _config(monkeypatch, tmp_path)
    db = tmp_path / "replica.db"
    _make_db(db, [("BE", "2026-08-07T00:00:00"), ("BE", "2026-08-07 21:00:00")])
    assert config.probe_net_position_latest(db)["BE"] == "2026-08-07 21:00:00"


def test_missing_net_position_table_is_a_readable_error(monkeypatch, tmp_path):
    config = _config(monkeypatch, tmp_path)
    db = tmp_path / "replica.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE something_else (x INTEGER)")
    con.commit(); con.close()
    with pytest.raises(RuntimeError, match="cannot read net_position"):
        config.probe_net_position_latest(db)
    # ...and check_database_currency turns it into a problem, never an exception.
    assert any("cannot read net_position" in p
               for p in config.check_database_currency(db, NOW))


# ---------------------------------------------------------------------------
# validate_config() end to end
# ---------------------------------------------------------------------------

def _write_replica(path, newest):
    _make_db(path, [(c, newest) for c in ["BE", "NL", "AT", "FR", "DE"]])


def test_validate_config_passes_on_a_current_database(monkeypatch, tmp_path):
    config = _config(monkeypatch, tmp_path)
    monkeypatch.delenv(config.ALLOW_STALE_DB_ENV, raising=False)
    newest = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
    _write_replica(tmp_path / "replica.db", newest)
    assert config.validate_config() is True


def test_validate_config_rejects_the_stale_copy(monkeypatch, tmp_path):
    """The regression: present on disk, so the old exists() check passed."""
    config = _config(monkeypatch, tmp_path)
    monkeypatch.delenv(config.ALLOW_STALE_DB_ENV, raising=False)
    db = tmp_path / "replica.db"
    # The decoy's real shape: BE/NL/FR ancient, AT/DE absent entirely.
    _make_db(db, [("BE", "2023-02-28 23:00:00"), ("NL", "2023-01-31 23:00:00"),
                  ("FR", "2024-01-15 23:00:00")])
    assert db.exists(), "the point of this test is that the file is present"
    with pytest.raises(ValueError) as excinfo:
        config.validate_config()
    message = str(excinfo.value)
    assert "does not look like the live replica" in message
    assert "AT: no net_position rows at all" in message
    assert "DE: no net_position rows at all" in message


def test_allow_stale_db_downgrades_the_failure_to_a_warning(monkeypatch, tmp_path, capsys):
    config = _config(monkeypatch, tmp_path)
    monkeypatch.setenv(config.ALLOW_STALE_DB_ENV, "1")
    _write_replica(tmp_path / "replica.db", "2023-01-31 23:00:00")
    assert config.validate_config() is True
    assert "WARNING" in capsys.readouterr().out


def test_missing_database_still_reported_as_before(monkeypatch, tmp_path):
    config = _config(monkeypatch, tmp_path, db_name="does_not_exist.db")
    monkeypatch.delenv(config.ALLOW_STALE_DB_ENV, raising=False)
    with pytest.raises(ValueError, match="Database not found"):
        config.validate_config()
