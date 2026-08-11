"""Every registered model is pushed, independently (ABL-175).

Before this, only CHAMPION_MODEL_NAME was ever shipped to the dashboard, so
challenger accrual in prod was permanently zero and the ABL-70 promotion gate
could only ever see ABL-137's one-shot backfill. These tests pin: all four
registered models get their own push attempt, one model's failure or absence
does not touch another's, and quantiles ride along only for the model that
actually stored them.
"""
import importlib
import io
import json
import sqlite3
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

push = importlib.import_module("push_net_position_forecast")
from src.challengers.registry import CHALLENGERS, CHAMPION_MODEL_NAME

CHAMPION = CHAMPION_MODEL_NAME
V012 = CHALLENGERS["V012"].model_name
V014 = CHALLENGERS["V014"].model_name
V016 = CHALLENGERS["V016"].model_name


def _sidecar(tmp_path, name="sidecar.db"):
    """Champion plus all three challengers, on the same target hour - the real
    daily shape. Only V016 carries a quantile band, matching the correction
    layer being the sole quantile-emitting challenger."""
    path = tmp_path / name
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
        ("FR", "net_position", "2026-08-09 00:00:00", "2026-08-07 06:05:00", 42, 1100.0, V012, "20260807_060500"),
        ("FR", "net_position", "2026-08-09 00:00:00", "2026-08-07 06:07:00", 42, 1200.0, V014, "20260807_060700"),
        ("FR", "net_position", "2026-08-09 00:00:00", "2026-08-07 06:09:00", 42, 1300.0, V016, "20260807_060900"),
    ]
    con.executemany("""INSERT INTO forecasts (country_code, forecast_type,
        target_timestamp_utc, generated_at, horizon_hours, forecast_value,
        model_name, model_version) VALUES (?,?,?,?,?,?,?,?)""", rows)
    con.executemany("""INSERT INTO forecast_quantiles (country_code, forecast_type,
        target_timestamp_utc, generated_at, quantile, forecast_value, model_name)
        VALUES (?,?,?,?,?,?,?)""", [
        ("FR", "net_position", "2026-08-09 00:00:00", "2026-08-07 06:09:00", 0.1, 1250.0, V016),
        ("FR", "net_position", "2026-08-09 00:00:00", "2026-08-07 06:09:00", 0.9, 1350.0, V016),
    ])
    con.commit()
    con.close()
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


class _FakeResponse:
    def __init__(self, body):
        self._body = json.dumps(body).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_models_to_push_is_champion_then_every_registered_challenger():
    names = push.models_to_push()
    assert names[0] == CHAMPION
    assert set(names[1:]) == {V012, V014, V016}
    assert len(names) == len(set(names))


def test_models_to_push_dedupes_when_champion_overridden_to_a_challenger(monkeypatch):
    monkeypatch.setenv("CHAMPION_MODEL_NAME", V016)
    names = push.models_to_push()
    # V016 appears once even though it is both the (overridden) champion and a
    # registered challenger; the real default champion name drops out entirely.
    assert names.count(V016) == 1
    assert set(names) == {V012, V014, V016}


def test_all_four_models_push_independently_with_correct_bodies(tmp_path, monkeypatch):
    con = _sidecar(tmp_path)
    sent = []

    def fake_urlopen(request, timeout=None):
        sent.append(json.loads(request.data.decode("utf-8")))
        return _FakeResponse({"data": {"points": 1, "quantiles": 0, "replaced": False}})

    monkeypatch.setattr(push.urllib.request, "urlopen", fake_urlopen)

    results = {}
    for model_name in push.models_to_push():
        status, detail = push.push_model(con, "http://dashboard.example", "tok", model_name)
        results[model_name] = (status, detail)

    assert {name: status for name, (status, _) in results.items()} == {
        CHAMPION: "ok", V012: "ok", V014: "ok", V016: "ok",
    }
    # Each request named its own model and carried only that model's row.
    assert {p["model"]["name"] for p in sent} == {CHAMPION, V012, V014, V016}
    for p in sent:
        assert len(p["rows"]) == 1
    # Only V016's request carries quantiles; V012/V014 are median-only.
    by_model = {p["model"]["name"]: p for p in sent}
    assert by_model[V016]["rows"][0]["quantiles"] == {"0.1": 1250.0, "0.9": 1350.0}
    assert "quantiles" not in by_model[V012]["rows"][0]
    assert "quantiles" not in by_model[V014]["rows"][0]


def test_one_model_missing_does_not_block_the_others(tmp_path, monkeypatch):
    """Sidecar holds only the champion and V016 - V012/V014 never ran today
    (e.g. forecast_challengers.py partially failed). Both present models must
    still push."""
    path = tmp_path / "partial.db"
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE forecasts (
        id INTEGER PRIMARY KEY AUTOINCREMENT, country_code TEXT, forecast_type TEXT,
        target_timestamp_utc TIMESTAMP, generated_at TIMESTAMP, horizon_hours INTEGER,
        forecast_value REAL, model_name TEXT, model_version TEXT)""")
    con.executemany("""INSERT INTO forecasts (country_code, forecast_type,
        target_timestamp_utc, generated_at, horizon_hours, forecast_value,
        model_name, model_version) VALUES (?,?,?,?,?,?,?,?)""", [
        ("FR", "net_position", "2026-08-09 00:00:00", "2026-08-07 06:00:00", 42, 1000.0, CHAMPION, "v"),
        ("FR", "net_position", "2026-08-09 00:00:00", "2026-08-07 06:09:00", 42, 1300.0, V016, "v"),
    ])
    con.commit()
    con.close()
    ro = sqlite3.connect(f"file:{path}?mode=ro", uri=True)

    monkeypatch.setattr(
        push.urllib.request, "urlopen",
        lambda request, timeout=None: _FakeResponse({"data": {"points": 1, "quantiles": 0, "replaced": False}}),
    )

    results = {m: push.push_model(ro, "http://dashboard.example", "tok", m)[0]
               for m in push.models_to_push()}
    assert results[CHAMPION] == "ok"
    assert results[V016] == "ok"
    assert results[V012] == "no_data"
    assert results[V014] == "no_data"


def test_one_models_http_failure_does_not_abort_or_corrupt_the_others(tmp_path, monkeypatch):
    con = _sidecar(tmp_path)
    calls = []

    def flaky_urlopen(request, timeout=None):
        body = json.loads(request.data.decode("utf-8"))
        calls.append(body["model"]["name"])
        if body["model"]["name"] == V014:
            raise urllib.error.HTTPError(
                request.full_url, 500, "boom", hdrs=None, fp=io.BytesIO(b"server error"))
        return _FakeResponse({"data": {"points": 1, "quantiles": 0, "replaced": False}})

    monkeypatch.setattr(push.urllib.request, "urlopen", flaky_urlopen)

    results = {m: push.push_model(con, "http://dashboard.example", "tok", m)[0]
               for m in push.models_to_push()}

    # All four were attempted - V014's failure did not stop the loop.
    assert set(calls) == {CHAMPION, V012, V014, V016}
    assert results[V014] == "failed"
    assert results[CHAMPION] == "ok"
    assert results[V012] == "ok"
    assert results[V016] == "ok"


def test_main_exit_code_reflects_worst_status(tmp_path, monkeypatch):
    con_path = tmp_path / "sidecar.db"
    _sidecar(tmp_path, name="sidecar.db").close()

    monkeypatch.setenv("FORECAST_OUTPUT_DB", str(con_path))
    monkeypatch.setenv("DASHBOARD_API_URL", "http://dashboard.example")
    monkeypatch.setenv("DASHBOARD_WRITE_TOKEN", "tok")

    monkeypatch.setattr(
        push.urllib.request, "urlopen",
        lambda request, timeout=None: _FakeResponse({"data": {"points": 1, "quantiles": 0, "replaced": False}}),
    )
    assert push.main() == 0

    def failing_urlopen(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(push.urllib.request, "urlopen", failing_urlopen)
    assert push.main() == 1
