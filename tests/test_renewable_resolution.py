"""ABL-332: one resolution leaves the renewable read, and subsampling is loud.

The defect these tests pin is not "the builder was hourly" -- being hourly is
correct and intended. It is that the builder was hourly while its *input* was
not, and nothing said so. `series.loc[ts.floor("h")]` finds the `:00` row of a
quarter-hourly country, returns a scalar, and builds every lag from a quarter of
the data with no exception and no log line.

Measured on the replica 2026-08-12, that was 22 of the 24
`config.SUPPORTED_COUNTRIES` in `energy_renewable` -- the table serving reads
today -- not a DE/NL quirk. `reports/abl_332_renewable_resolution.md` has the
per-country table.

So the properties are:

  1. `load_renewable_type_data` returns hourly instants, whatever the source
     stores, and the value is the hourly *mean* -- not the `:00` sub-sample.
  2. Aggregating cannot invent a measurement: an hour with no live sub-sample
     stays NaN rather than becoming 0.0.
  3. The builder raises rather than subsampling if it is ever handed a
     sub-hourly series again.
  4. Training and serving read the same numbers -- the whole point of doing
     this at the shared read rather than inside the builder.
"""
import importlib
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src import db
from src.wind_features import (
    RenewableFeatureBuilder,
    SubHourlyResolutionError,
    _assert_hourly,
)

COUNTRY = "XQ"
START, END = "2026-01-01", "2026-01-05"

#: Four days of quarter-hourly instants -- the DE/NL/AT/... shape.
_QUARTERS = pd.date_range("2026-01-01", "2026-01-04 23:45", freq="15min")

#: An hour whose four quarter-hours are 100/200/300/400. Mean 250; the `:00`
#: sub-sample alone says 100. Any test that reads 100 is reading the bug.
_SHAPED_HOUR = pd.Timestamp("2026-01-02 09:00")


def _solar_at(ts: pd.Timestamp) -> float:
    """A deliberately intra-hour-varying series: the `:00` sample is never the
    hour's mean, so a subsampling read is arithmetically detectable."""
    if ts.floor("h") == _SHAPED_HOUR:
        return {0: 100.0, 15: 200.0, 30: 300.0, 45: 400.0}[ts.minute]
    return 500.0 + ts.dayofyear * 24 + ts.hour * 10 + ts.minute


@pytest.fixture
def replica(tmp_path, monkeypatch):
    """A quarter-hourly country in both tables, plus an hourly one for contrast."""
    path = tmp_path / "replica.db"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE energy_generation (country_code TEXT, timestamp_utc TIMESTAMP,
            solar_mw REAL, wind_onshore_mw REAL, wind_offshore_mw REAL,
            hydro_run_mw REAL, hydro_reservoir_mw REAL, biomass_mw REAL,
            data_quality TEXT DEFAULT 'actual');
        CREATE TABLE energy_renewable (country_code TEXT, timestamp_utc TIMESTAMP,
            solar_mw REAL DEFAULT 0, wind_onshore_mw REAL DEFAULT 0,
            wind_offshore_mw REAL DEFAULT 0, hydro_run_mw REAL DEFAULT 0,
            hydro_reservoir_mw REAL DEFAULT 0, biomass_mw REAL DEFAULT 0,
            data_quality TEXT DEFAULT 'actual');
        CREATE TABLE weather_data (country_code TEXT, timestamp_utc TIMESTAMP,
            forecast_run_time TIMESTAMP, data_quality TEXT,
            temperature_2m_k REAL, relative_humidity_2m_frac REAL,
            wind_speed_10m_ms REAL, wind_speed_100m_ms REAL,
            shortwave_radiation_wm2 REAL, direct_radiation_wm2 REAL,
            diffuse_radiation_wm2 REAL);
        """
    )
    for ts in _QUARTERS:
        solar = _solar_at(ts)
        onshore = 1000.0 + ts.hour * 5 + ts.minute
        for table in ("energy_generation", "energy_renewable"):
            con.execute(
                f"INSERT INTO {table} VALUES (?, ?, ?, ?, NULL, ?, NULL, NULL, 'actual')",
                (COUNTRY, str(ts), solar, onshore, 42.0),
            )
    # An hourly country, to prove the aggregation is a no-op there.
    for ts in pd.date_range("2026-01-01", "2026-01-04 23:00", freq="h"):
        con.execute(
            "INSERT INTO energy_generation VALUES (?, ?, ?, ?, NULL, ?, NULL, NULL, 'actual')",
            ("HH", str(ts), 700.0 + ts.hour, 1500.0 + ts.hour, 42.0),
        )
    con.commit()
    con.close()

    monkeypatch.setenv("ENERGY_DB_PATH", str(path))
    importlib.reload(config)
    importlib.reload(db)
    yield path
    monkeypatch.undo()
    importlib.reload(config)
    importlib.reload(db)


# ---------------------------------------------------------------------------
# 1. One resolution leaves the read
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source", ["energy_generation", "energy_renewable"])
def test_the_loader_returns_hourly_instants_whatever_the_source_stores(replica, source):
    frame = db.load_renewable_type_data(COUNTRY, "solar", START, END, source=source)
    stamps = pd.DatetimeIndex(frame["timestamp_utc"])
    off_hour = stamps[stamps != stamps.floor("h")]
    assert len(off_hour) == 0, (
        f"{len(off_hour)} of {len(stamps)} instants are off the hour (first "
        f"{off_hour[0] if len(off_hour) else None}). Every lookup in "
        "src/wind_features.py floors to the hour, so these do not reach a "
        "feature -- they are discarded in silence. That is ABL-332."
    )
    assert len(stamps) == len(set(stamps)), "one instant, one row"


@pytest.mark.parametrize("source", ["energy_generation", "energy_renewable"])
def test_the_hourly_value_is_the_mean_not_the_00_subsample(replica, source):
    """The arithmetic that separates a fix from a relabelling.

    The shaped hour holds 100/200/300/400. Mean 250. A builder still reading the
    `:00` sub-sample sees 100 -- a real measurement, 60% low, wearing the name of
    the hour.
    """
    frame = db.load_renewable_type_data(COUNTRY, "solar", START, END, source=source)
    value = frame.loc[frame["timestamp_utc"] == _SHAPED_HOUR, "target_value"]
    assert len(value) == 1
    assert value.iloc[0] == pytest.approx(250.0), (
        f"expected the mean of 100/200/300/400 = 250.0, got {value.iloc[0]}. "
        "100.0 means the :00 sub-sample is still what reaches the model."
    )


def test_an_already_hourly_country_is_returned_untouched(replica):
    """The aggregation must not be a second thing that can go wrong for the five
    countries (BE, BG, CH, LV, PT) that were never affected."""
    frame = db.load_renewable_type_data("HH", "solar", START, END, source="energy_generation")
    assert len(frame) == 96, f"4 days x 24 h, got {len(frame)}"
    assert frame["target_value"].tolist() == [700.0 + ts.hour for ts in frame["timestamp_utc"]]


# ---------------------------------------------------------------------------
# 2. Aggregating cannot invent a measurement
# ---------------------------------------------------------------------------


def test_an_all_null_hour_stays_nan_and_does_not_become_zero():
    """`sum` without `min_count=1` collapses an all-NaN group to 0.0 -- the exact
    trap top-level CLAUDE.md names. `mean` does not, and this pins that it is
    `mean` that is used, in the presence of a live neighbour hour so a passing
    test cannot just be an empty frame."""
    stamps = pd.date_range("2026-01-01 00:00", periods=8, freq="15min")
    values = [np.nan] * 4 + [10.0, 20.0, 30.0, 40.0]
    frame = pd.DataFrame({"timestamp_utc": stamps, "target_value": values})

    hourly = db.aggregate_renewable_to_hourly(frame)

    dead = hourly.loc[hourly["timestamp_utc"] == pd.Timestamp("2026-01-01 00:00"), "target_value"]
    assert len(dead) == 1
    assert pd.isna(dead.iloc[0]), (
        f"an hour with no live sub-sample read as {dead.iloc[0]}, not NaN. "
        "NULL is not 0: a country that reported nothing must not train as a "
        "measured zero."
    )
    live = hourly.loc[hourly["timestamp_utc"] == pd.Timestamp("2026-01-01 01:00"), "target_value"]
    assert live.iloc[0] == pytest.approx(25.0)


def test_a_partial_hour_averages_the_samples_it_has_and_says_so(caplog):
    """Three of four quarter-hours present -> the mean of the three, which is
    what `resample('h').mean()` has always given training. It is logged rather
    than silently dropped or filled: the value is real, its support is not
    full, and the reader gets to know which."""
    stamps = pd.DatetimeIndex([
        "2026-01-01 00:00", "2026-01-01 00:15", "2026-01-01 00:45",
        "2026-01-01 01:00", "2026-01-01 01:15", "2026-01-01 01:30", "2026-01-01 01:45",
    ])
    frame = pd.DataFrame({"timestamp_utc": stamps,
                          "target_value": [10.0, 20.0, 60.0, 1.0, 2.0, 3.0, 4.0]})

    with caplog.at_level("WARNING", logger="energy_forecast"):
        hourly = db.aggregate_renewable_to_hourly(frame, context="XQ/solar")

    partial = hourly.loc[hourly["timestamp_utc"] == pd.Timestamp("2026-01-01 00:00"), "target_value"]
    assert partial.iloc[0] == pytest.approx(30.0)
    assert "ABL-332" in caplog.text and "partial hour" in caplog.text


def test_the_suspect_constant_guard_still_sees_the_native_resolution(replica, caplog):
    """Order matters. `exclude_suspect_constant_runs` infers cadence from the
    series' own median step and measures runs in hours (ABL-188's real instance
    is 6,408 *quarter*-hours). Averaging first would blur a run's edges into
    non-constant values; so the guard runs first and this pins that it still
    fires end-to-end through the hourly read."""
    path = replica
    con = sqlite3.connect(path)
    con.execute("DELETE FROM energy_generation WHERE country_code = 'ZZ'")
    for ts in pd.date_range("2026-01-01", "2026-01-04 23:45", freq="15min"):
        # A flat 0.0 for the whole span: unambiguously a zero-fill, not a night.
        con.execute(
            "INSERT INTO energy_generation VALUES (?, ?, ?, ?, NULL, ?, NULL, NULL, 'actual')",
            ("ZZ", str(ts), 0.0, 1.0, 42.0),
        )
    con.commit()
    con.close()

    with caplog.at_level("WARNING", logger="energy_forecast"):
        frame = db.load_renewable_type_data("ZZ", "solar", START, END, source="energy_generation")

    assert "ABL-188" in caplog.text, "the zero-fill guard did not fire"
    assert frame["target_value"].notna().sum() == 0, (
        "a zero-filled run survived aggregation as a measured value"
    )


# ---------------------------------------------------------------------------
# 3. Subsampling is loud
# ---------------------------------------------------------------------------


def test_the_builder_refuses_a_sub_hourly_series_instead_of_subsampling():
    series = pd.Series(
        [100.0, 200.0, 300.0, 400.0],
        index=pd.DatetimeIndex(["2026-01-02 09:00", "2026-01-02 09:15",
                                "2026-01-02 09:30", "2026-01-02 09:45"]),
    )
    with pytest.raises(SubHourlyResolutionError) as excinfo:
        _assert_hourly(series, "XQ/solar")
    message = str(excinfo.value)
    assert "3 of 4" in message, message
    assert "2026-01-02 09:15" in message, message


def test_an_hourly_series_passes_the_guard_unchanged():
    series = pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2026-01-02 09:00", "2026-01-02 10:00"]))
    assert _assert_hourly(series, "XQ/solar") is series


def test_an_empty_series_passes_the_guard():
    """A country the TSO does not report reaches the builder as an empty frame
    (ABL-321) -- that is not a resolution failure and must not raise here."""
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    assert _assert_hourly(empty, "XQ/wind_offshore").empty


# ---------------------------------------------------------------------------
# 4. Training and serving read the same numbers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source", ["energy_generation", "energy_renewable"])
def test_the_builders_lag_is_the_hourly_mean_not_the_00_subsample(replica, source):
    """End to end, through the real builder: the lag that lands in the model.

    Target 2026-01-03 09:00, so `lag_1d` resolves to 2026-01-02 09:00 -- the
    shaped hour. Before ABL-332 this read 100.0.
    """
    builder = RenewableFeatureBuilder(
        COUNTRY, "solar", "2026-01-01", "2026-01-04", actuals_source=source
    )
    row = builder.row("2026-01-03 09:00", observation_as_of="2026-01-02 18:00")
    lag = row["target_value_lag_1d"]
    assert lag.source_timestamp == _SHAPED_HOUR
    assert lag.value == pytest.approx(250.0), (
        f"lag_1d = {lag.value}; 100.0 is the :00 sub-sample, 250.0 is the hour."
    )


def test_training_and_serving_agree_on_every_hour(replica):
    """The reason the fix is at the shared read and not inside the builder.

    `load_training_data` resamples to the hourly mean and `features.py` shifts by
    `days * 24` rows -- both only correct on an hourly frame. Serving floors to
    the hour. With the aggregation at the read, the two arms are the same series
    rather than two definitions that happen to agree on hourly countries.
    """
    training = db.load_training_data(COUNTRY, "solar", START, END)
    serving = db.load_renewable_type_data(COUNTRY, "solar", START, END)

    merged = training.merge(serving, on="timestamp_utc", suffixes=("_train", "_serve"))
    assert len(merged) == len(training) > 0
    pd.testing.assert_series_equal(
        merged["target_value_train"], merged["target_value_serve"],
        check_names=False,
        obj="training frame vs serving frame",
    )


def test_aggregation_raises_the_non_zero_fraction_the_train_screen_thresholds_on():
    """The one training-path consumer ABL-332 does move, pinned deliberately.

    `scripts/train.py:354` decides whether a pair is worth training at all from
    `(target_value > 0).sum() / len(df)` against a 0.30/0.50 threshold, and it
    reads `load_renewable_type_data` *without* resampling. An hourly mean is
    non-zero whenever any sub-sample in the hour is, so aggregation can only
    raise that fraction -- never lower it.

    This is not a defect to be fixed by reverting the aggregation: the screen
    now measures the same hourly frame the model is fitted on. It is pinned
    because it flips a real verdict (IT/wind_offshore, 0.4865 -> 0.5764 on the
    2026-08-12 replica) and the next reader should find that intended, not
    discover it in a training sweep.
    """
    # One non-zero quarter per hour: the classic low-capacity offshore shape.
    stamps = pd.date_range("2026-01-01", periods=4 * 24, freq="15min")
    df = pd.DataFrame({
        "timestamp_utc": stamps,
        "target_value": [7.0 if ts.minute == 30 else 0.0 for ts in stamps],
    })

    before = (df["target_value"] > 0).sum() / len(df)
    hourly = db.aggregate_renewable_to_hourly(df)
    after = (hourly["target_value"] > 0).sum() / len(hourly)

    assert before == pytest.approx(0.25)
    assert after == pytest.approx(1.0)
    assert after > before, "aggregation must never lower the screen's fraction"

    # And the direction holds for an all-zero hour: a measured zero stays zero,
    # so the screen is not simply inflated everywhere.
    zeros = pd.DataFrame({
        "timestamp_utc": stamps,
        "target_value": [0.0] * len(stamps),
    })
    zero_hourly = db.aggregate_renewable_to_hourly(zeros)
    assert (zero_hourly["target_value"] > 0).sum() == 0
