"""Observation bound and publication bound are separate instants (ABL-68).

`build_for_country` used one `as_of` for both: how far observations reach, and
which covariate runs had been issued. For a day-ahead-published target those are
16 hours apart, so no single value is right --

  * `D 22:00` (the correct observation bound for net_position, which is
    published ~12:45 CET on D-1) also admits weather runs issued at 12:00Z on D,
    which the 06:00Z run could not have seen;
  * `D 06:00` (the correct publication bound) truncates the target context 16h
    short, which is what `compare_experiments.py:178` does to every net_position
    backtest week.

Measured against the as-served 2026-08-06 06:00 vintage: passing `D 22:00` for
both put the worst country 1,881 MW from what production served; splitting the
bounds brought 16 of 19 countries under 0.3% of mean |forecast|.
"""
import importlib
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

TARGET_DATE = "2026-06-12"
RUN_INSTANT = "2026-06-10 06:00:00"        # when the job fires
OBSERVATION_BOUND = "2026-06-10 22:00:00"  # day-ahead target reaches D 21:00

# Two weather runs: one the 06:00 job could see, one issued after it.
EARLY_RUN = "2026-06-10 00:00:00"
LATE_RUN = "2026-06-10 12:00:00"
EARLY_TEMP, LATE_TEMP = 280.0, 300.0

LEVEL = 8000.0


def _seed_db(path):
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE net_position "
                "(country_code TEXT, timestamp_utc TEXT, net_position_mw REAL)")
    con.execute("CREATE TABLE weather_data (country_code TEXT, timestamp_utc TEXT, "
                "temperature_2m_k REAL, wind_speed_100m_ms REAL, "
                "shortwave_radiation_wm2 REAL, data_quality TEXT, forecast_run_time TEXT)")
    con.execute("CREATE TABLE energy_price (country_code TEXT, timestamp_utc TEXT, "
                "price_eur_mwh REAL, data_quality TEXT)")
    con.execute("CREATE TABLE energy_load (country_code TEXT, timestamp_utc TEXT, "
                "load_mw REAL, data_quality TEXT)")
    con.execute("CREATE TABLE energy_load_forecast (country_code TEXT, "
                "target_timestamp_utc TEXT, forecast_value_mw REAL)")
    con.execute("CREATE TABLE crossborder_flows (country_from TEXT, country_to TEXT, "
                "timestamp_utc TEXT, flow_mw REAL)")

    # Net position observed right up to the day-ahead bound (D 21:00).
    last_obs = pd.Timestamp(OBSERVATION_BOUND) - pd.Timedelta(hours=1)
    idx = pd.date_range(last_obs - pd.Timedelta(days=60), last_obs, freq="h")
    con.executemany("INSERT INTO net_position VALUES (?,?,?)",
                    [("FR", ts.strftime("%Y-%m-%d %H:%M:%S"), LEVEL) for ts in idx])

    # Both weather runs cover the whole forecast window; they differ only in
    # value, so which one the builder picks is directly readable.
    fc = pd.date_range(last_obs, pd.Timestamp(TARGET_DATE) + pd.Timedelta(hours=23),
                       freq="h")
    rows = []
    for run_time, temp in ((EARLY_RUN, EARLY_TEMP), (LATE_RUN, LATE_TEMP)):
        rows += [("FR", ts.strftime("%Y-%m-%d %H:%M:%S"), temp, 5.0, 100.0,
                  "forecast", run_time) for ts in fc]
    con.executemany("INSERT INTO weather_data VALUES (?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()


@pytest.fixture
def build(monkeypatch, tmp_path):
    db = tmp_path / "t.db"
    _seed_db(db)
    monkeypatch.setenv("ENERGY_DB_PATH", str(db))
    import config
    importlib.reload(config)
    from src.chronos2 import input_builder
    importlib.reload(input_builder)
    return input_builder.InputBuilder()


def _future_temp(inp):
    for name, arr in inp["future_covariates"].items():
        if "temperature" in name:
            return float(np.asarray(arr, dtype=float)[-1])
    raise AssertionError(f"no temperature covariate in {list(inp['future_covariates'])}")


def test_single_as_of_admits_a_weather_run_the_job_could_not_see(build):
    """The leak, stated as a fact about the old signature: bounding publication
    at the observation bound picks up the 12:00 run."""
    inp = build.build_for_country("FR", "net_position", TARGET_DATE,
                                  as_of=OBSERVATION_BOUND)
    assert _future_temp(inp) == pytest.approx(LATE_TEMP)


def test_publication_bound_excludes_it(build):
    """Serve-faithful: observations to D 22:00, weather runs only to D 06:00."""
    inp = build.build_for_country("FR", "net_position", TARGET_DATE,
                                  as_of=OBSERVATION_BOUND,
                                  publication_as_of=RUN_INSTANT)
    assert _future_temp(inp) == pytest.approx(EARLY_TEMP)


def test_publication_bound_does_not_shorten_the_context(build):
    """The whole point of splitting: the narrower publication bound must not
    cost the context the 16h of actuals a day-ahead target really had."""
    split = build.build_for_country("FR", "net_position", TARGET_DATE,
                                    as_of=OBSERVATION_BOUND,
                                    publication_as_of=RUN_INSTANT)
    both_early = build.build_for_country("FR", "net_position", TARGET_DATE,
                                         as_of=RUN_INSTANT,
                                         publication_as_of=RUN_INSTANT)
    gap = both_early["prediction_length"] - split["prediction_length"]
    assert gap == 16, f"expected the documented 16h difference, got {gap}h"
    assert split["future_index"][0] == pd.Timestamp(OBSERVATION_BOUND)


def test_publication_bound_defaults_to_as_of(build):
    """Back-compat: existing callers passing only as_of keep exact behaviour."""
    old = build.build_for_country("FR", "net_position", TARGET_DATE,
                                  as_of=OBSERVATION_BOUND)
    explicit = build.build_for_country("FR", "net_position", TARGET_DATE,
                                       as_of=OBSERVATION_BOUND,
                                       publication_as_of=OBSERVATION_BOUND)
    assert _future_temp(old) == pytest.approx(_future_temp(explicit))
    assert old["prediction_length"] == explicit["prediction_length"]


def test_live_path_bounds_nothing(build):
    """Live runs pass neither bound: the database holds only what has been
    ingested, so the freshest run is already in the past."""
    inp = build.build_for_country("FR", "net_position", TARGET_DATE)
    assert _future_temp(inp) == pytest.approx(LATE_TEMP)
