"""Only the champion reaches production (ABL-68).

Since the shadow rail writes challenger vintages into the same sidecar, the
push has to name the model it ships. Before this, it took the newest
`generated_at` for `forecast_type='net_position'` regardless of model — and the
challengers run *after* the champion in the same job, so the newest vintage in
the sidecar is a challenger's. These tests fail against that version.
"""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

push = importlib.import_module("push_net_position_forecast")

CHAMPION = "chronos-2-V010"
CHALLENGER = "chronos-2-V016"


def _sidecar(tmp_path):
    """Champion at 06:00, challenger at 06:05 — the real daily ordering."""
    path = tmp_path / "sidecar.db"
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE forecasts (
        id INTEGER PRIMARY KEY AUTOINCREMENT, country_code TEXT, forecast_type TEXT,
        target_timestamp_utc TIMESTAMP, generated_at TIMESTAMP, horizon_hours INTEGER,
        forecast_value REAL, model_name TEXT, model_version TEXT)""")
    con.execute("""CREATE TABLE forecast_quantiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT, country_code TEXT, forecast_type TEXT,
        target_timestamp_utc TIMESTAMP, generated_at TIMESTAMP, quantile REAL,
        forecast_value REAL, model_name TEXT)""")
    rows = [
        ("FR", "net_position", "2026-08-09 00:00:00", "2026-08-07 06:00:00", 42, 1000.0, CHAMPION, "20260807_060000"),
        ("FR", "net_position", "2026-08-09 01:00:00", "2026-08-07 06:00:00", 43, 1100.0, CHAMPION, "20260807_060000"),
        ("FR", "net_position", "2026-08-09 00:00:00", "2026-08-07 06:05:00", 42, 2500.0, CHALLENGER, "20260807_060500"),
        ("FR", "net_position", "2026-08-09 01:00:00", "2026-08-07 06:05:00", 43, 2600.0, CHALLENGER, "20260807_060500"),
    ]
    con.executemany("""INSERT INTO forecasts (country_code, forecast_type,
        target_timestamp_utc, generated_at, horizon_hours, forecast_value,
        model_name, model_version) VALUES (?,?,?,?,?,?,?,?)""", rows)
    con.executemany("""INSERT INTO forecast_quantiles (country_code, forecast_type,
        target_timestamp_utc, generated_at, quantile, forecast_value, model_name)
        VALUES (?,?,?,?,?,?,?)""", [
        ("FR", "net_position", "2026-08-09 00:00:00", "2026-08-07 06:00:00", 0.5, 1000.0, CHAMPION),
        ("FR", "net_position", "2026-08-09 00:00:00", "2026-08-07 06:05:00", 0.5, 2500.0, CHALLENGER),
    ])
    con.commit()
    con.close()
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def test_latest_vintage_ignores_a_newer_challenger(tmp_path):
    con = _sidecar(tmp_path)
    generated_at, model_name, _ = push.latest_vintage(con, CHAMPION)
    assert model_name == CHAMPION
    assert generated_at == "2026-08-07 06:00:00"


def test_payload_carries_only_champion_rows(tmp_path):
    con = _sidecar(tmp_path)
    generated_at, model_name, version = push.latest_vintage(con, CHAMPION)
    payload = push.build_payload(con, generated_at, model_name, version)
    assert payload["model"]["name"] == CHAMPION
    assert [r["forecast_value"] for r in payload["rows"]] == [1000.0, 1100.0]
    # One row per (country, target hour): a model mix would duplicate them.
    keys = [(r["country_code"], r["target_timestamp_utc"]) for r in payload["rows"]]
    assert len(keys) == len(set(keys))
    assert payload["rows"][0]["quantiles"] == {"0.5": 1000.0}


def test_shared_generated_at_still_separates_models(tmp_path):
    """Belt and braces: if a challenger ever shares the champion's timestamp,
    the row query must still not mix them into one payload."""
    path = tmp_path / "same_ts.db"
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE forecasts (
        id INTEGER PRIMARY KEY AUTOINCREMENT, country_code TEXT, forecast_type TEXT,
        target_timestamp_utc TIMESTAMP, generated_at TIMESTAMP, horizon_hours INTEGER,
        forecast_value REAL, model_name TEXT, model_version TEXT)""")
    con.executemany("""INSERT INTO forecasts (country_code, forecast_type,
        target_timestamp_utc, generated_at, horizon_hours, forecast_value,
        model_name, model_version) VALUES (?,?,?,?,?,?,?,?)""", [
        ("FR", "net_position", "2026-08-09 00:00:00", "2026-08-07 06:00:00", 42, 1000.0, CHAMPION, "v"),
        ("FR", "net_position", "2026-08-09 00:00:00", "2026-08-07 06:00:00", 42, 9999.0, CHALLENGER, "v"),
    ])
    con.commit(); con.close()
    ro = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    payload = push.build_payload(ro, "2026-08-07 06:00:00", CHAMPION, "v")
    assert [r["forecast_value"] for r in payload["rows"]] == [1000.0]


def test_nothing_to_push_when_champion_absent(tmp_path):
    """A sidecar holding only challengers must push nothing, not the challenger."""
    path = tmp_path / "challenger_only.db"
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE forecasts (
        id INTEGER PRIMARY KEY AUTOINCREMENT, country_code TEXT, forecast_type TEXT,
        target_timestamp_utc TIMESTAMP, generated_at TIMESTAMP, horizon_hours INTEGER,
        forecast_value REAL, model_name TEXT, model_version TEXT)""")
    con.execute("""INSERT INTO forecasts (country_code, forecast_type,
        target_timestamp_utc, generated_at, horizon_hours, forecast_value,
        model_name, model_version) VALUES
        ('FR','net_position','2026-08-09 00:00:00','2026-08-07 06:05:00',42,2500.0,?,'v')""",
        (CHALLENGER,))
    con.commit(); con.close()
    ro = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    assert push.latest_vintage(ro, CHAMPION) is None


@pytest.mark.parametrize("env,expected", [(None, CHAMPION), ("chronos-2-V016", "chronos-2-V016")])
def test_champion_is_configured_not_inferred(monkeypatch, env, expected):
    if env is None:
        monkeypatch.delenv("CHAMPION_MODEL_NAME", raising=False)
    else:
        monkeypatch.setenv("CHAMPION_MODEL_NAME", env)
    assert push.champion_model_name() == expected
