"""V014 sees only what a 06:00Z run held, and never invents what it did not (ABL-69).

A tabular model evaluates every feature at the *target* timestamp, so unlike the
Chronos-2 champion it has no single context cutoff to lean on — each column has
to justify its own availability. These tests pin the three ways that goes wrong:
a lag that reaches past the cutoff, a source row that arrived after the run, and
a gap answered with a fabricated zero.

The fixture database is deliberately tiny and hand-built. What matters is not
volume but that each table carries a row *past* the serve cutoff, so a test can
fail if that row reaches a feature.
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.challengers.v014_features import (CROSSBORDER_SERVE_LAG_HOURS,
                                           DAY_AHEAD_CUTOFF_HOUR,
                                           SAME_HOUR_LAGS, ServeFaithfulnessError,
                                           ServeWindow, assert_lag_is_serve_safe,
                                           build_cache, build_features)

RUN_DAY = pd.Timestamp("2026-08-06")
TARGET_DAY = pd.Timestamp("2026-08-08")


def _hours(start, end):
    return pd.date_range(start, end, freq="h")


@pytest.fixture
def db(tmp_path):
    """A replica-shaped database whose every source runs past the serve cutoff."""
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
    # Everything from 40 days before the run through the end of the target day.
    # The hours past the cutoff (D 21:00) are the trap: 1e6 is unmistakable if
    # it ever reaches a feature.
    span = _hours(RUN_DAY - pd.Timedelta(days=40), TARGET_DAY + pd.Timedelta(hours=23))
    cutoff = RUN_DAY + pd.Timedelta(hours=DAY_AHEAD_CUTOFF_HOUR)
    for ts in span:
        past_cutoff = ts > cutoff
        value = 1e6 if past_cutoff else 1000 + ts.hour * 10
        con.execute("INSERT INTO net_position VALUES ('XX', ?, ?)", (str(ts), value))
        con.execute("INSERT INTO energy_price VALUES ('XX', ?, ?)",
                    (str(ts), 1e6 if past_cutoff else 50 + ts.hour))
        con.execute("INSERT INTO energy_load_forecast VALUES ('XX','day_ahead',?,?)",
                    (str(ts), 1e6 if past_cutoff else 40000 + ts.hour * 100))
        con.execute("INSERT INTO energy_generation_forecast "
                    "VALUES ('XX','day_ahead',?,?,?,NULL,NULL)",
                    (str(ts), 1e6 if past_cutoff else ts.hour * 5,
                     1e6 if past_cutoff else 2000))
        con.execute("INSERT INTO crossborder_flows VALUES ('XX','YY',?,?)",
                    (str(ts), 1e6 if past_cutoff else 300))
    con.commit()
    yield con
    con.close()


def _features(db, **kwargs):
    window = ServeWindow.for_run_day(RUN_DAY)
    cache = build_cache(db, "XX", RUN_DAY - pd.Timedelta(days=40),
                        TARGET_DAY + pd.Timedelta(days=1))
    return build_features(cache, window, **kwargs), window


# --- the serve window -------------------------------------------------------

def test_run_day_and_target_day_are_two_sides_of_one_window():
    assert ServeWindow.for_target_day(TARGET_DAY).run_ts == \
        ServeWindow.for_run_day(RUN_DAY).run_ts


def test_cutoff_is_the_measured_day_ahead_bound_not_the_run_instant():
    """A 06:00Z run legitimately holds day-ahead data through D 21:00 (ABL-28).
    Bounding at the run instant instead would throw away 15 real hours."""
    w = ServeWindow.for_run_day(RUN_DAY)
    assert w.run_ts == RUN_DAY + pd.Timedelta(hours=6)
    assert w.day_ahead_cutoff == RUN_DAY + pd.Timedelta(hours=21)
    assert (w.target_index[0], w.target_index[-1]) == \
        (TARGET_DAY, TARGET_DAY + pd.Timedelta(hours=23))


# --- the lag guard, which is the one check that can actually fire ------------

def test_a_48h_lag_is_refused_and_50h_is_the_boundary():
    """The binding target hour is D+2 23:00, exactly 50h past the D 21:00 cutoff.
    A 48h same-hour lag reaches D 22:00 and D 23:00 — hours that do not exist at
    a 06:00Z run — and is the mistake this guard exists to catch."""
    w = ServeWindow.for_run_day(RUN_DAY)
    for unsafe in (24, 47, 48, 49):
        with pytest.raises(ServeFaithfulnessError):
            assert_lag_is_serve_safe(w, unsafe, "probe")
    for safe in (50, 51, 72, 168):
        assert_lag_is_serve_safe(w, safe, "probe")


def test_every_configured_lag_is_serve_safe():
    w = ServeWindow.for_run_day(RUN_DAY)
    for lag in SAME_HOUR_LAGS:
        assert_lag_is_serve_safe(w, lag, "configured")
    assert min(SAME_HOUR_LAGS) >= 50


# --- nothing reaches past the cutoff ----------------------------------------

def test_no_feature_carries_a_value_stored_after_the_serve_cutoff(db):
    """The fixture writes 1e6 into every source past D 21:00. If any of it lands
    in a feature, the backtest is a claim about information nobody had."""
    features, _ = _features(db)
    numeric = features.select_dtypes(include=[np.number])
    worst = numeric.abs().max().max()
    assert worst < 1e5, \
        f"a post-cutoff value reached the feature frame (max |value| {worst:.0f})"


def test_weather_admits_only_issued_runs_at_or_before_the_run_instant(db):
    """Three runs cover the target day: one issued before the 06:00Z run, one
    after it, and a reanalysis row. Only the first may be used — the second is
    information the run did not have, and the third is *observed* weather
    wearing a forecast's clothes."""
    target = TARGET_DAY + pd.Timedelta(hours=12)
    rows = [
        ("2026-08-05 12:00:00", "forecast", 290.0),   # issued before the run
        ("2026-08-06 12:00:00", "forecast", 999.0),   # issued after the run
        (str(target), "actual", 888.0),               # reanalysis nowcast
    ]
    for run_time, quality, temp in rows:
        db.execute("INSERT INTO weather_data VALUES ('XX',?,?,?,?,5.0,9.0,300.0)",
                   (str(target), run_time, quality, temp))
    db.commit()
    features, _ = _features(db)
    assert features.loc[target, "wx_temperature_2m_k"] == 290.0
    assert features.loc[target, "weather_available"] == 1


def test_weather_absent_is_nan_and_flagged_never_backfilled_from_reanalysis(db):
    """W01-W10 have no issued-weather archive at all (it begins 2026-01-11).
    Falling back to the reanalysis there would hand the model observed weather
    and flatter every weather-driven backtest score."""
    target = TARGET_DAY + pd.Timedelta(hours=12)
    db.execute("INSERT INTO weather_data VALUES ('XX',?,?, 'actual', 300.0,5.0,9.0,400.0)",
               (str(target), str(target)))
    db.commit()
    features, _ = _features(db)
    assert features["wx_temperature_2m_k"].isna().all()
    assert (features["weather_available"] == 0).all()


# --- a gap stays a gap ------------------------------------------------------

def test_a_missing_crossborder_hour_is_nan_and_flagged_not_zero(db):
    """The champion's aligner answers a flow gap with `fillna(0.0)`, i.e. a
    fabricated 'no flow across this border' presented as a measurement (ABL-74).
    Here the gap stays NaN and `xb_missing` names it."""
    missing_hour = TARGET_DAY + pd.Timedelta(hours=6) - pd.Timedelta(
        hours=CROSSBORDER_SERVE_LAG_HOURS)
    db.execute("DELETE FROM crossborder_flows WHERE timestamp_utc = ?",
               (str(missing_hour),))
    db.commit()
    features, _ = _features(db)
    row = features.loc[TARGET_DAY + pd.Timedelta(hours=6)]
    assert np.isnan(row[f"xb_net_lag{CROSSBORDER_SERVE_LAG_HOURS}h"])
    assert row["xb_missing"] == 1
    assert features["xb_missing"].sum() == 1


def test_crossborder_is_net_of_both_directions(db):
    """Net position is a balance; a features frame that only counted exports
    would be systematically wrong in one direction for every importer."""
    hour = TARGET_DAY - pd.Timedelta(hours=CROSSBORDER_SERVE_LAG_HOURS)
    db.execute("INSERT INTO crossborder_flows VALUES ('ZZ','XX',?,120.0)", (str(hour),))
    db.commit()
    features, _ = _features(db)
    assert features.loc[TARGET_DAY, f"xb_net_lag{CROSSBORDER_SERVE_LAG_HOURS}h"] == 180.0


# --- shape ------------------------------------------------------------------

def test_one_row_per_target_hour_with_no_duplicate_columns(db):
    features, window = _features(db)
    assert list(features.index) == list(window.target_index)
    assert len(features) == 24
    assert len(set(features.columns)) == len(features.columns)


def test_the_frame_is_numeric_so_xgboost_can_consume_the_nans_natively(db):
    features, _ = _features(db)
    non_numeric = [c for c in features.columns
                   if not pd.api.types.is_numeric_dtype(features[c])]
    assert non_numeric == []
