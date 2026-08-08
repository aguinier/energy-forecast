"""V014 in the shadow rail: what it writes, and what it declines to (ABL-69).

`tests/test_v014_model.py` pins the model's refusals; this pins the runner's.
The runner is where a plausible wrong vintage would actually reach the sidecar
and be scored, so the three things it must never do are: serve a country with no
trained model, write a refused hour as 0.0, and quietly become better-informed
than the schedule promises when the job fires late.
"""
import importlib
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from src.challengers.registry import CHAMPION_MODEL_NAME, spec_for
from src.challengers.v014 import V014Model, save_model
from src.challengers.v014_features import DAY_AHEAD_CUTOFF_HOUR, ServeWindow

TARGET_DATE = "2026-08-08"
RUN_DAY = pd.Timestamp("2026-08-06")
GENERATED_AT = pd.Timestamp("2026-08-06 06:00:00")
SPEC = spec_for("V014")


class _ConstantBooster:
    def predict(self, X):
        return np.full(len(X), 1234.0)


@pytest.fixture
def rail(monkeypatch, tmp_path):
    monkeypatch.setenv("ENERGY_DB_PATH", str(tmp_path / "replica.db"))
    monkeypatch.setenv("FORECAST_OUTPUT_DB", str(tmp_path / "sidecar.db"))
    import config
    importlib.reload(config)
    mod = importlib.import_module("forecast_challengers")
    return importlib.reload(mod)


@pytest.fixture
def replica(tmp_path):
    """A replica holding net position for XX only. YY is registered nowhere and
    has no rows — it stands for a country the fleet has but V014 has not fitted."""
    path = tmp_path / "replica.db"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE net_position (country_code TEXT, timestamp_utc TIMESTAMP,
                                   net_position_mw REAL);
        CREATE TABLE energy_price (country_code TEXT, timestamp_utc TIMESTAMP,
                                   price_eur_mwh REAL);
        CREATE TABLE energy_load_forecast (country_code TEXT, forecast_type TEXT,
                                           target_timestamp_utc TIMESTAMP,
                                           forecast_value_mw REAL);
        CREATE TABLE energy_generation_forecast (country_code TEXT, forecast_type TEXT,
                                                 target_timestamp_utc TIMESTAMP,
                                                 solar_mw REAL, wind_onshore_mw REAL,
                                                 wind_offshore_mw REAL,
                                                 total_forecast_mw REAL);
        CREATE TABLE crossborder_flows (country_from TEXT, country_to TEXT,
                                        timestamp_utc TIMESTAMP, flow_mw REAL);
        CREATE TABLE weather_data (country_code TEXT, timestamp_utc TIMESTAMP,
                                   forecast_run_time TIMESTAMP, data_quality TEXT,
                                   temperature_2m_k REAL, wind_speed_10m_ms REAL,
                                   wind_speed_100m_ms REAL,
                                   shortwave_radiation_wm2 REAL);
    """)
    cutoff = RUN_DAY + pd.Timedelta(hours=DAY_AHEAD_CUTOFF_HOUR)
    for ts in pd.date_range(RUN_DAY - pd.Timedelta(days=35), cutoff, freq="h"):
        con.execute("INSERT INTO net_position VALUES ('XX', ?, ?)",
                    (str(ts), 1000 + ts.hour * 10))
    con.commit()
    con.close()
    return str(path)


def _trained(models_dir, country, columns):
    save_model(V014Model(country=country, booster=_ConstantBooster(),
                         feature_columns=list(columns), neighbours=[]), models_dir)


def _feature_columns(replica_db):
    from src.challengers.v014_features import build_cache, build_features
    con = sqlite3.connect(f"file:{replica_db}?mode=ro", uri=True)
    try:
        window = ServeWindow.for_target_day(TARGET_DATE)
        cache = build_cache(con, "XX", RUN_DAY - pd.Timedelta(days=35),
                            window.target_index.max())
        return list(build_features(cache, window, neighbours=[]).columns)
    finally:
        con.close()


def test_v014_writes_its_own_model_name_for_every_served_hour(rail, replica, tmp_path):
    _trained(tmp_path / "models", "XX", _feature_columns(replica))
    rows, qrows = rail.run_v014(SPEC, ["XX"], TARGET_DATE, GENERATED_AT, {},
                                replica_db=replica, models_dir=tmp_path / "models")
    assert len(rows) == 24
    assert {r["model_name"] for r in rows} == {"xgboost-V014"}
    assert CHAMPION_MODEL_NAME not in {r["model_name"] for r in rows}
    # No quantiles: V014 is a point forecast. Emitting a fabricated band would
    # feed the gate's coverage_10_90 criterion a number nothing measured.
    assert qrows == []


def test_a_country_with_no_trained_model_is_skipped_not_improvised(rail, replica, tmp_path):
    _trained(tmp_path / "models", "XX", _feature_columns(replica))
    rows, _ = rail.run_v014(SPEC, ["XX", "YY"], TARGET_DATE, GENERATED_AT, {},
                            replica_db=replica, models_dir=tmp_path / "models")
    assert {r["country_code"] for r in rows} == {"XX"}


def test_a_country_with_no_observations_is_refused_never_written_as_zero(
        rail, replica, tmp_path):
    """XX's rows are deleted, so every anchor feature is NaN — GR's condition.
    The run must produce nothing for it rather than a confident number, and in
    particular must not store 0.0, which is a real balanced-border reading."""
    _trained(tmp_path / "models", "XX", _feature_columns(replica))
    con = sqlite3.connect(replica)
    con.execute("DELETE FROM net_position")
    con.commit()
    con.close()
    rows, _ = rail.run_v014(SPEC, ["XX"], TARGET_DATE, GENERATED_AT, {},
                            replica_db=replica, models_dir=tmp_path / "models")
    assert rows == []


def test_a_late_run_does_not_become_a_better_informed_run(rail, replica, tmp_path):
    """The window comes from the target date, so a job that fires at 18:00 gets
    the same 06:00Z cutoffs as one that fires on time. The opposite — a run
    whose target implies a run instant in the future — is refused outright,
    because its features would reach past what exists."""
    _trained(tmp_path / "models", "XX", _feature_columns(replica))
    late = pd.Timestamp("2026-08-06 18:00:00")
    on_time, _ = rail.run_v014(SPEC, ["XX"], TARGET_DATE, GENERATED_AT, {},
                               replica_db=replica, models_dir=tmp_path / "models")
    late_rows, _ = rail.run_v014(SPEC, ["XX"], TARGET_DATE, late, {},
                                 replica_db=replica, models_dir=tmp_path / "models")
    assert [r["forecast_value"] for r in late_rows] == [r["forecast_value"] for r in on_time]

    too_early = pd.Timestamp("2026-08-05 06:00:00")   # target implies a run tomorrow
    assert rail.run_v014(SPEC, ["XX"], TARGET_DATE, too_early, {},
                         replica_db=replica, models_dir=tmp_path / "models") == ([], [])


def test_v014_runs_by_default_rather_than_only_when_asked_for(rail, monkeypatch):
    """Registering a challenger and then leaving it off the default list is how
    a model ends up with no vintages on the day the gate window opens — and the
    comparison window is the *intersection* of the models' spans, so a late
    starter shortens it for everyone (ABL-72 G4)."""
    assert "V014" in rail.RUNNERS
    monkeypatch.setattr(sys, "argv", ["forecast_challengers.py"])
    import argparse

    captured = {}
    real_parse = argparse.ArgumentParser.parse_args

    def spy(self, *a, **kw):
        captured["defaults"] = {act.dest: act.default for act in self._actions}
        raise SystemExit(0)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", spy)
    try:
        rail.main()
    except SystemExit:
        pass
    finally:
        monkeypatch.setattr(argparse.ArgumentParser, "parse_args", real_parse)
    assert "V014" in captured["defaults"]["experiments"].split(",")
