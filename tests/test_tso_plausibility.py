"""ABL-431: the TSO day-ahead plausibility guard, and the ways it must not bite.

The guard exists because HU's published `wind_onshore_mw` reached 140,996 MW
on 2026-02-04 against a 283 MW fleet, and ABL-247 proposes feeding that series
to a model. The easy half is catching it. The half these tests are mostly about
is the constraint the issue put on the fix: **it must not become a blanket
filter.** This repo has twice shipped a deliberately narrow guard to avoid
discarding legitimately published values -- ABL-71 keeps published zeros,
ABL-109's is not a blanket `> 0` because DE solar has 56 real overnight zeros --
and a guard that widens into those is worse than the defect it replaces,
because deleting real MW is silent and a 140 GW outlier is not.
"""

import ast
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tso_plausibility import (  # noqa: E402
    PLAUSIBILITY_TOLERANCE,
    REFERENCE_QUANTILE,
    TSO_FORECAST_SOURCES,
    VINTAGE_ARCHIVE_DAY_AHEAD_MODEL,
    VINTAGE_ARCHIVE_TABLE,
    UnknownTsoSourceError,
    clear_reference_cache,
    forecast_read,
    guard_series,
    guard_tso_frame,
    guard_tso_series,
    implausible_mask,
    reference_scale,
)

REPO_ROOT = Path(__file__).parent.parent

GEN_FORECAST = "energy_generation_forecast"
LOAD_FORECAST = "energy_load_forecast"
ARCHIVE = VINTAGE_ARCHIVE_TABLE


@pytest.fixture(autouse=True)
def _no_cross_test_cache():
    """The reference cache is process-wide; two fixtures must not share one."""
    clear_reference_cache()
    yield
    clear_reference_cache()


def _db(gen_forecast=(), generation=(), load_forecast=(), load=(), archive=()):
    """A throwaway replica-shaped database, in memory.

    Deliberately not the real replica: these tests pin the guard's *rules*, and
    a rule pinned against live data stops being a test the day the data moves.
    The live extent is measured by `scripts/abl431_tso_plausibility_census.py`.
    """
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE energy_generation_forecast (
            country_code TEXT, target_timestamp_utc TIMESTAMP,
            solar_mw REAL, wind_onshore_mw REAL, wind_offshore_mw REAL,
            total_forecast_mw REAL, forecast_type TEXT DEFAULT 'day_ahead');
        CREATE TABLE energy_generation (
            country_code TEXT, timestamp_utc TIMESTAMP,
            solar_mw REAL, wind_onshore_mw REAL, wind_offshore_mw REAL);
        CREATE TABLE energy_load_forecast (
            country_code TEXT, target_timestamp_utc TIMESTAMP,
            forecast_value_mw REAL, forecast_type TEXT DEFAULT 'day_ahead');
        CREATE TABLE energy_load (
            country_code TEXT, timestamp_utc TIMESTAMP, load_mw REAL);
        CREATE TABLE forecast_vintage_archive (
            source TEXT NOT NULL CHECK (source IN ('ml', 'tso')),
            forecast_type TEXT NOT NULL, country_code TEXT NOT NULL,
            target_timestamp_utc TEXT NOT NULL, model_name TEXT NOT NULL,
            run_timestamp_utc TEXT NOT NULL, horizon_hours INTEGER,
            forecast_value REAL NOT NULL, first_seen_at TEXT NOT NULL);
    """)
    conn.executemany(
        "INSERT INTO forecast_vintage_archive "
        "(source, forecast_type, country_code, target_timestamp_utc, model_name, "
        " run_timestamp_utc, forecast_value, first_seen_at) "
        "VALUES (?, ?, ?, ?, ?, '2026-02-01T00:00:00Z', ?, '2026-02-01T00:00:00Z')",
        archive)
    conn.executemany(
        "INSERT INTO energy_generation_forecast "
        "(country_code, target_timestamp_utc, wind_onshore_mw, forecast_type) "
        "VALUES (?, ?, ?, 'day_ahead')", gen_forecast)
    conn.executemany(
        "INSERT INTO energy_generation (country_code, timestamp_utc, wind_onshore_mw) "
        "VALUES (?, ?, ?)", generation)
    conn.executemany(
        "INSERT INTO energy_load_forecast "
        "(country_code, target_timestamp_utc, forecast_value_mw, forecast_type) "
        "VALUES (?, ?, ?, 'day_ahead')", load_forecast)
    conn.executemany(
        "INSERT INTO energy_load (country_code, timestamp_utc, load_mw) VALUES (?, ?, ?)",
        load)
    conn.commit()
    return conn


def _hours(n, start="2026-02-01"):
    return pd.date_range(start, periods=n, freq="h")


def _rows(country, values, start="2026-02-01"):
    return [(country, str(t), float(v)) for t, v in zip(_hours(len(values), start), values)]


def _archive_rows(country, values, forecast_type="wind_onshore",
                  model_name="tso-day_ahead", source="tso", start="2026-02-01"):
    """The same series, shaped for the tall archive."""
    return [(source, forecast_type, country, str(t), model_name, float(v))
            for t, v in zip(_hours(len(values), start), values)]


# --------------------------------------------------------------------------
# It catches the thing it was built for
# --------------------------------------------------------------------------

def test_the_hu_signature_is_refused_and_its_neighbours_are_not():
    """A x1000 scale error goes; the same day's ordinary values stay.

    Shaped like the real incident: a long healthy history, then one day whose
    values are three orders of magnitude out. The assertion that matters is the
    second one -- a guard that also took the healthy tail would be useless.
    """
    healthy = _rows("HU", [50.0 + i % 200 for i in range(2000)])
    conn = _db(gen_forecast=healthy, generation=healthy)

    series = pd.Series([120.0, 140996.2, 68399.2, 305.0, 0.0],
                       index=_hours(5, "2026-02-04"))
    guarded = guard_tso_series(series, conn, "HU", GEN_FORECAST, "wind_onshore_mw")

    assert pd.isna(guarded.iloc[1]) and pd.isna(guarded.iloc[2])
    assert guarded.iloc[0] == 120.0
    assert guarded.iloc[3] == 305.0
    assert guarded.iloc[4] == 0.0


def test_the_stored_rows_are_not_mutated():
    """Read-site guard, not a repair. The issue forbids touching the table.

    "A value that looks impossible is sometimes just not published yet" -- so
    the row has to survive for whoever re-reads it after an upstream fix.
    """
    rows = _rows("HU", [50.0] * 500 + [140996.2])
    conn = _db(gen_forecast=rows, generation=rows)
    before = conn.execute(
        f"SELECT COUNT(*), MAX(wind_onshore_mw) FROM {GEN_FORECAST}").fetchone()

    guard_tso_series(pd.Series([140996.2], index=_hours(1)), conn, "HU",
                     GEN_FORECAST, "wind_onshore_mw")

    assert conn.execute(
        f"SELECT COUNT(*), MAX(wind_onshore_mw) FROM {GEN_FORECAST}").fetchone() == before


def test_the_input_series_is_not_mutated_in_place():
    rows = _rows("HU", [50.0] * 500)
    conn = _db(gen_forecast=rows, generation=rows)
    series = pd.Series([9999.0, 40.0], index=_hours(2))

    guard_tso_series(series, conn, "HU", GEN_FORECAST, "wind_onshore_mw")

    assert series.iloc[0] == 9999.0


# --------------------------------------------------------------------------
# It is not a blanket filter -- the constraint the issue put on the fix
# --------------------------------------------------------------------------

def test_a_published_zero_is_never_refused():
    """ABL-71 keeps published zeros; ABL-109 refused a blanket `> 0`.

    The guard is one-sided by construction, so this holds at any tolerance and
    for any reference. Pinned anyway: it is the property those two issues were
    filed to protect, and a later "also drop obvious nonsense at the bottom"
    would break it without breaking anything else here.
    """
    rows = _rows("DE", [1000.0 + i % 500 for i in range(1000)])
    conn = _db(gen_forecast=rows, generation=rows)

    series = pd.Series([0.0] * 56, index=_hours(56))
    guarded = guard_tso_series(series, conn, "DE", GEN_FORECAST, "wind_onshore_mw")

    assert (guarded == 0.0).all()
    assert guarded.notna().all()


def test_a_zero_reference_refuses_to_evaluate_rather_than_rejecting_everything():
    """A landlocked country's `wind_offshore_mw` is 0.0 forever.

    Its reference is 0.0, so `value > tolerance * 0` would flag every non-zero
    value such a country ever published -- the blanket filter in its purest
    form. 28 of the replica's 174 pairs are in this state, so this is the
    common case, not an edge one. The guard must fail open and say why.
    """
    rows = _rows("AT", [0.0] * 1000)
    conn = _db(gen_forecast=rows, generation=rows)

    scale = reference_scale(conn, "AT", GEN_FORECAST, "wind_onshore_mw")
    assert not scale.evaluable
    assert scale.threshold_mw is None
    assert "no fleet" in scale.reason or "no scale" in scale.reason

    series = pd.Series([0.0, 5.0, 900.0], index=_hours(3))
    guarded = guard_tso_series(series, conn, "AT", GEN_FORECAST, "wind_onshore_mw")
    pd.testing.assert_series_equal(guarded, series)


def test_a_brand_new_fleets_first_output_is_not_refused():
    """The same mechanism, in the case that would actually cost us something.

    A country standing up its first wind farm has a history of zeros, so there
    is no reference and the guard passes its first real MW through. Failing
    open here is deliberate: an unguarded new fleet is a bounded cost, and a
    guard that rejects a country's first real generation is not.
    """
    rows = _rows("EE", [0.0] * 900)
    conn = _db(gen_forecast=rows, generation=rows)

    first_output = pd.Series([0.0, 12.0, 45.0, 88.0], index=_hours(4))
    guarded = guard_tso_series(first_output, conn, "EE", GEN_FORECAST, "wind_onshore_mw")

    assert guarded.notna().all()
    assert guarded.tolist() == first_output.tolist()


def test_a_growing_fleet_is_scaled_by_its_own_recent_level_not_its_early_one():
    """NL solar went from nothing to 7.9 GW inside this history.

    The reference is a high quantile of the *whole* series, so it tracks the
    grown fleet. A frozen capacity table -- the other way to read "scaled to
    installed capacity" -- would still be holding this country to its first year
    and would reject today's real output.
    """
    ramp = [10.0] * 800 + [float(v) for v in range(100, 900)]
    rows = _rows("NL", ramp)
    conn = _db(gen_forecast=rows, generation=rows)

    scale = reference_scale(conn, "NL", GEN_FORECAST, "wind_onshore_mw")
    assert scale.reference_mw > 800.0

    today = pd.Series([880.0], index=_hours(1))
    assert guard_tso_series(today, conn, "NL", GEN_FORECAST,
                            "wind_onshore_mw").notna().all()


def test_the_actuals_side_rescues_a_forecast_table_that_underreports():
    """And the forecast side rescues an actuals table that does.

    Neither anchor is sound alone. NL's `energy_generation.solar_mw` tops out at
    428.8 MW while NL's own published solar forecast reaches 7,871 MW -- an
    actuals-only reference would reject 18x of legitimate NL solar. The reverse
    case is the defect's own home, so the max of the two is the only form that
    is safe in both directions.
    """
    big_forecast = _rows("NL", [5000.0 + i % 1000 for i in range(1000)])
    tiny_actuals = _rows("NL", [30.0 + i % 50 for i in range(1000)])
    conn = _db(gen_forecast=big_forecast, generation=tiny_actuals)
    forecast_led = reference_scale(conn, "NL", GEN_FORECAST, "wind_onshore_mw")
    assert forecast_led.reference_mw > 5000.0

    clear_reference_cache()
    conn2 = _db(gen_forecast=tiny_actuals, generation=big_forecast)
    actual_led = reference_scale(conn2, "NL", GEN_FORECAST, "wind_onshore_mw")
    assert actual_led.reference_mw > 5000.0


def test_a_contaminated_cluster_smaller_than_the_quantile_tail_cannot_raise_its_own_bar():
    """The reference is a quantile, not a maximum, for exactly this reason.

    HU's incident is 96 rows in 196,984 (0.0487%), an order of magnitude inside
    the 1 - p99.5 = 0.5% the reference tolerates. A maximum-anchored reference
    would have been set *by* the defect and would have passed it.
    """
    rows = _rows("HU", [50.0] * 2000 + [140996.2] * 5)
    conn = _db(gen_forecast=rows, generation=rows)

    scale = reference_scale(conn, "HU", GEN_FORECAST, "wind_onshore_mw")
    assert scale.reference_mw < 100.0
    assert guard_tso_series(pd.Series([140996.2], index=_hours(1)), conn, "HU",
                            GEN_FORECAST, "wind_onshore_mw").isna().all()


def test_a_missing_actuals_table_falls_back_to_the_forecast_side_not_a_crash():
    """The guard sits on a read path, so it may not raise on a partial database.

    A fixture or a partial snapshot with no `energy_generation` still has a
    forecast side to anchor on -- and that side alone is what catches HU, at
    497x its own p99.5. Failing open on the missing side keeps the guard useful
    without letting it take down a read it was only supposed to watch.
    """
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE energy_generation_forecast (
            country_code TEXT, target_timestamp_utc TIMESTAMP,
            wind_onshore_mw REAL, forecast_type TEXT DEFAULT 'day_ahead');
    """)
    conn.executemany(
        "INSERT INTO energy_generation_forecast "
        "(country_code, target_timestamp_utc, wind_onshore_mw, forecast_type) "
        "VALUES (?, ?, ?, 'day_ahead')", _rows("HU", [50.0] * 500))
    conn.commit()

    scale = reference_scale(conn, "HU", GEN_FORECAST, "wind_onshore_mw")
    assert scale.evaluable
    assert scale.actual_quantile_mw is None and scale.n_actual == 0
    assert scale.forecast_quantile_mw == pytest.approx(50.0)

    guarded = guard_tso_series(pd.Series([40.0, 140996.2], index=_hours(2)),
                               conn, "HU", GEN_FORECAST, "wind_onshore_mw")
    assert guarded.iloc[0] == 40.0 and pd.isna(guarded.iloc[1])


def test_a_database_with_neither_side_is_not_evaluable_rather_than_an_error():
    conn = sqlite3.connect(":memory:")
    scale = reference_scale(conn, "HU", GEN_FORECAST, "wind_onshore_mw")
    assert not scale.evaluable
    series = pd.Series([140996.2], index=_hours(1))
    pd.testing.assert_series_equal(
        guard_tso_series(series, conn, "HU", GEN_FORECAST, "wind_onshore_mw"), series)


def test_nan_is_left_as_nan_and_never_counted_as_refused():
    rows = _rows("HU", [50.0] * 500)
    conn = _db(gen_forecast=rows, generation=rows)
    series = pd.Series([float("nan"), 40.0, 9999.0], index=_hours(3))

    scale = reference_scale(conn, "HU", GEN_FORECAST, "wind_onshore_mw")
    guarded, outcome = guard_series(series, scale)

    assert pd.isna(guarded.iloc[0])
    assert outcome.n_flagged == 1
    assert outcome.n_observed == 2


# --------------------------------------------------------------------------
# Registration: no defaults, no guessing
# --------------------------------------------------------------------------

def test_an_unregistered_column_raises_rather_than_guessing_a_scale():
    """No default, for the same reason NIGHT_GENERATION_POSSIBLE has none.

    A guard that guesses which fleet it is scaling against can null real MW and
    log it as a correction, which is the failure this module exists to avoid
    causing rather than to cause somewhere else.
    """
    conn = _db()
    with pytest.raises(UnknownTsoSourceError):
        reference_scale(conn, "HU", GEN_FORECAST, "biomass_mw")


def test_every_column_the_read_sites_pass_is_registered():
    """The registry has to cover what the wired read sites actually ask for.

    Both lists are imported rather than restated: a new forecast type added to
    the scorecard, or a new generation column added to V014, must land in
    TSO_FORECAST_SOURCES in the same change or fail here rather than at the
    first read.
    """
    from src.challengers.v014_features import GENERATION_FORECAST_COLUMNS
    from src.evaluation.scorecard import TSO_SPECS

    for table, _timestamp_col, value_col in TSO_SPECS.values():
        assert (table, value_col) in TSO_FORECAST_SOURCES, \
            f"scorecard reads {table}.{value_col} with no registered scale"

    for column in GENERATION_FORECAST_COLUMNS:
        assert (GEN_FORECAST, column) in TSO_FORECAST_SOURCES, \
            f"V014 reads {GEN_FORECAST}.{column} with no registered scale"


def test_the_registered_constants_are_the_ones_the_measurement_named():
    """The tolerance sits in a measured empty band; drifting it silently
    re-opens the question ABL-431 answered. 3.0 is between the highest healthy
    pair (PT solar, 1.82x) and the lowest anomalous one (MK total, 4.12x)."""
    assert PLAUSIBILITY_TOLERANCE == 3.0
    assert 1.82 < PLAUSIBILITY_TOLERANCE < 4.12
    assert REFERENCE_QUANTILE == 0.995


# --------------------------------------------------------------------------
# Serve-faithfulness and cache identity
# --------------------------------------------------------------------------

def test_as_of_bounds_the_reference_so_a_backtest_can_stay_serve_faithful():
    """The default is the whole history -- correct for serving, where the whole
    history *is* everything available. A backtest reconstructing a past vintage
    passes its observation cutoff, and must then get a smaller reference."""
    ramp = _rows("NL", [10.0] * 500 + [900.0] * 500)
    conn = _db(gen_forecast=ramp, generation=ramp)

    full = reference_scale(conn, "NL", GEN_FORECAST, "wind_onshore_mw")
    early = reference_scale(conn, "NL", GEN_FORECAST, "wind_onshore_mw",
                            as_of=str(_hours(1000)[400]))

    assert early.reference_mw < full.reference_mw
    assert early.as_of is not None and full.as_of is None


def test_the_cache_does_not_serve_one_databases_reference_to_another(tmp_path):
    """This box carries a 3.0 GB stale partial snapshot beside the live replica
    (CLAUDE.md), and it is the nearest file to every wrong path this module has
    been pointed at. Its HU wind history stops in 2023, so a reference cached
    from it and served to a read of the replica would scale today's fleet by a
    three-year-old one -- silently, and in the direction that deletes MW.

    Two real files, because the property is about the file the connection is
    attached to and an in-memory pair cannot distinguish them.
    """
    def _write(path, values):
        conn = _db(gen_forecast=_rows("HU", values), generation=_rows("HU", values))
        disk = sqlite3.connect(path)
        conn.backup(disk)
        conn.close()
        return disk

    stale = _write(str(tmp_path / "stale_snapshot.db"), [10.0] * 500)
    live = _write(str(tmp_path / "replica.db"), [5000.0] * 500)
    try:
        stale_scale = reference_scale(stale, "HU", GEN_FORECAST, "wind_onshore_mw")
        live_scale = reference_scale(live, "HU", GEN_FORECAST, "wind_onshore_mw")

        assert stale_scale.reference_mw == pytest.approx(10.0)
        assert live_scale.reference_mw == pytest.approx(5000.0)

        # And the live file's own 4,900 MW read survives, which it would not
        # have if the stale reference had been served to it from the cache.
        guarded = guard_tso_series(pd.Series([4900.0], index=_hours(1)), live,
                                   "HU", GEN_FORECAST, "wind_onshore_mw")
        assert guarded.notna().all()
    finally:
        stale.close()
        live.close()


def test_reference_scale_is_memoised_per_pair():
    rows = _rows("HU", [50.0] * 500)
    conn = _db(gen_forecast=rows, generation=rows)
    assert reference_scale(conn, "HU", GEN_FORECAST, "wind_onshore_mw") is \
        reference_scale(conn, "HU", GEN_FORECAST, "wind_onshore_mw")


# --------------------------------------------------------------------------
# Frame helper and the load table
# --------------------------------------------------------------------------

def test_guard_tso_frame_handles_an_aliased_column():
    rows = _rows("BE", [100.0] * 500)
    conn = _db(gen_forecast=rows, generation=rows)
    df = pd.DataFrame({"timestamp_utc": _hours(2),
                       "tso_forecast_mw": [95.0, 99999.0]})

    out = guard_tso_frame(df, conn, "BE", GEN_FORECAST, "wind_onshore_mw",
                          frame_column="tso_forecast_mw")

    assert out["tso_forecast_mw"].tolist()[0] == 95.0
    assert pd.isna(out["tso_forecast_mw"].iloc[1])
    assert df["tso_forecast_mw"].iloc[1] == 99999.0  # caller's frame untouched


def test_the_load_forecast_table_is_guarded_against_measured_load():
    rows = _rows("DE", [60000.0 + i % 10000 for i in range(1000)])
    conn = _db(load_forecast=rows, load=rows)

    series = pd.Series([70000.0, 7000000.0], index=_hours(2))
    guarded = guard_tso_series(series, conn, "DE", LOAD_FORECAST, "forecast_value_mw")

    assert guarded.iloc[0] == 70000.0
    assert pd.isna(guarded.iloc[1])


def test_the_guard_reports_what_it_did():
    rows = _rows("HU", [50.0] * 500)
    conn = _db(gen_forecast=rows, generation=rows)
    series = pd.Series([40.0, 140996.2], index=_hours(2))

    scale = reference_scale(conn, "HU", GEN_FORECAST, "wind_onshore_mw")
    _, outcome = guard_series(series, scale, context="unit-test")

    assert outcome.applied
    assert outcome.n_flagged == 1
    assert outcome.max_ratio > 100
    assert outcome.first_flagged == outcome.last_flagged == _hours(2)[1]


# --------------------------------------------------------------------------
# The vintage archive -- the table ABL-247 actually fits on (ABL-458)
# --------------------------------------------------------------------------

def test_the_hu_signature_is_refused_in_the_vintage_archive_too():
    """The same 96 rows live in the archive, and that is where ABL-247 reads.

    ABL-431 wired the two live tables. The archive is a *tall* table -- one
    `forecast_value` column discriminated by `source`/`forecast_type` -- so it
    needed no new rule, only a read shape. It is the same poison: on the live
    replica the archive's HU `wind_onshore` maxes at the identical 140,996.245
    over the identical 2026-02-03 23:00 .. 2026-02-04 22:45 window.
    """
    conn = _db(archive=_archive_rows("HU", [50.0] * 500),
               generation=_rows("HU", [50.0] * 500))

    series = pd.Series([40.0, 140996.245], index=_hours(2))
    guarded = guard_tso_series(series, conn, "HU", ARCHIVE, "wind_onshore")

    assert guarded.iloc[0] == 40.0
    assert pd.isna(guarded.iloc[1])


def test_the_archive_is_held_to_the_same_fleet_as_the_live_table():
    """A variable does not change fleet by being archived.

    Pinned as an equality between the two registrations rather than by
    restating the actuals counterpart, so the two cannot drift apart: whatever
    scale the live column is held to, the archived one is held to as well.
    """
    twins = {
        "solar": (GEN_FORECAST, "solar_mw"),
        "wind_onshore": (GEN_FORECAST, "wind_onshore_mw"),
        "wind_offshore": (GEN_FORECAST, "wind_offshore_mw"),
        "load": (LOAD_FORECAST, "forecast_value_mw"),
    }
    for variable, live_key in twins.items():
        assert (ARCHIVE, variable) in TSO_FORECAST_SOURCES, \
            f"the archive stores forecast_type='{variable}' with no registered scale"
        assert TSO_FORECAST_SOURCES[(ARCHIVE, variable)] == \
            TSO_FORECAST_SOURCES[live_key], \
            f"archived {variable} is scaled against a different fleet than {live_key}"

    # And nothing else: the archive holds no aggregate row, so registering a
    # `total` would invent a series the table does not have.
    archived = {c for t, c in TSO_FORECAST_SOURCES if t == ARCHIVE}
    assert archived == set(twins)


def test_a_week_ahead_unit_error_cannot_set_the_bar_that_catches_it():
    """The reference is the day-ahead slice alone, and this is why.

    `tso-week_ahead` reaches 4.76% of a pair's archived rows (DK load, measured
    2026-08-14) -- an order of magnitude past the 1 - q = 0.5% contaminated
    cluster the quantile tolerates. Were it pooled in, a week-ahead scale error
    would sit in its own p99.5 tail and lift the threshold above itself. Here
    30 poisoned week-ahead rows against 500 clean day-ahead ones would carry
    the reference from 50 MW to 140,996 MW and let the value through.
    """
    conn = _db(archive=_archive_rows("HU", [50.0] * 500) +
                       _archive_rows("HU", [140996.245] * 30,
                                     model_name="tso-week_ahead",
                                     start="2027-01-01"))

    scale = reference_scale(conn, "HU", ARCHIVE, "wind_onshore")

    assert scale.reference_mw == pytest.approx(50.0)
    assert scale.n_forecast == 500  # the week-ahead rows are not in the sample
    guarded = guard_tso_series(pd.Series([140996.245], index=_hours(1)),
                               conn, "HU", ARCHIVE, "wind_onshore")
    assert pd.isna(guarded.iloc[0])


def test_our_own_model_output_does_not_set_the_tso_reference():
    """The archive interleaves `source='ml'` rows -- our forecasts, not a TSO's.

    They measure the same fleet but are not a published TSO series, and a
    challenger that overshot would otherwise raise the bar the TSO read is held
    to. Excluded by the same clause that picks the day-ahead product.
    """
    conn = _db(archive=_archive_rows("HU", [50.0] * 500) +
                       _archive_rows("HU", [900000.0] * 200, source="ml",
                                     model_name="chronos2", start="2027-01-01"))

    scale = reference_scale(conn, "HU", ARCHIVE, "wind_onshore")

    assert scale.reference_mw == pytest.approx(50.0)
    assert scale.n_forecast == 500


def test_the_archive_discriminators_are_bound_not_interpolated():
    """`forecast_read` is the one definition of the read shape, for both
    callers, and it parameterises the values it filters on rather than pasting
    them into SQL."""
    expression, where, params = forecast_read(ARCHIVE, "wind_onshore")
    assert expression == "forecast_value"
    assert where.count("?") == len(params) == 3
    assert params == ["tso", VINTAGE_ARCHIVE_DAY_AHEAD_MODEL, "wind_onshore"]
    assert "wind_onshore" not in where

    # The wide tables keep their own shape: the variable is the column, and
    # `forecast_type` names the horizon rather than the variable.
    expression, where, params = forecast_read(GEN_FORECAST, "wind_onshore_mw")
    assert expression == "wind_onshore_mw"
    assert where.count("?") == len(params) == 1
    assert params == ["day_ahead"]


def test_an_unregistered_archive_variable_raises_rather_than_guessing():
    """No default on the tall table either -- `forecast_type` is free text in
    the schema, so a new variable arriving in the archive must be registered
    rather than scaled against whatever the read happens to select."""
    conn = _db()
    with pytest.raises(UnknownTsoSourceError):
        reference_scale(conn, "HU", ARCHIVE, "hydro_total")


def test_a_zero_reference_in_the_archive_still_refuses_to_evaluate():
    """The all-zero landlocked-offshore case, on the tall table. 56 of the 318
    census pairs have no fleet to scale against; the guard must pass them
    through rather than flag every non-zero value a new fleet ever publishes."""
    conn = _db(archive=_archive_rows("HU", [0.0] * 500,
                                     forecast_type="wind_offshore"))

    scale = reference_scale(conn, "HU", ARCHIVE, "wind_offshore")

    assert not scale.evaluable
    guarded = guard_tso_series(pd.Series([12.0], index=_hours(1)),
                               conn, "HU", ARCHIVE, "wind_offshore")
    assert guarded.iloc[0] == 12.0


def test_the_archive_bounds_its_reference_on_target_timestamp_for_a_backtest():
    """`as_of` has to reach the tall table too, or a backtest reconstructing a
    past vintage would silently take its scale from the whole archive."""
    ramp = _archive_rows("NL", [10.0] * 500 + [900.0] * 500)
    conn = _db(archive=ramp)

    full = reference_scale(conn, "NL", ARCHIVE, "wind_onshore")
    early = reference_scale(conn, "NL", ARCHIVE, "wind_onshore",
                            as_of=str(_hours(1000)[400]))

    assert early.reference_mw < full.reference_mw


# --------------------------------------------------------------------------
# The read sites are wired, checked statically
# --------------------------------------------------------------------------

#: Every module that reads a TSO day-ahead forecast column, and therefore must
#: route it through the guard. A new read site added without the guard is the
#: failure mode this pins -- it would pass every other test in the suite and
#: only show up as a model that fitted on a 140 GW row.
GUARDED_READ_SITES = (
    "scripts/abl247_vintage_availability_probe.py",
    "scripts/abl430_ro_country_diagnosis.py",
    "scripts/abl439_reporting_basis_probe.py",
    # ABL-607. Reads `source = 'ml'` rows only -- our own forecasts, not an
    # ingested TSO series -- so this is not the read class ABL-431 was filed
    # about, and its sibling pack ABL-246 guards its TSO arm while leaving its
    # ML arm raw. Registered guarded anyway because the script's published
    # claim is a per-country WAPE *ranking* and WAPE is unbounded above: one
    # row three orders of magnitude out would decide a country's cell on its
    # own. The reference is set from the `tso-day_ahead` slice and the actuals,
    # so our own arm cannot raise the bar it is held to.
    "scripts/abl607_d2_load_diagnosis.py",
    "src/challengers/v014_features.py",
    "src/chronos2/input_builder.py",
    "src/evaluation/scorecard.py",
    "src/evaluation/tso_correction.py",
    "src/tso_correction_forecaster.py",
)

GUARD_CALLS = {"guard_tso_series", "guard_tso_frame"}


def _called_names(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


@pytest.mark.parametrize("relative_path", GUARDED_READ_SITES)
def test_every_wired_read_site_calls_the_guard(relative_path):
    source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert _called_names(tree) & GUARD_CALLS, (
        f"{relative_path} reads a TSO day-ahead forecast column but never calls "
        f"the ABL-431 guard")


TSO_TABLES = ("energy_generation_forecast", "energy_load_forecast",
              VINTAGE_ARCHIVE_TABLE)

#: Directories the sweep does not walk, and why.  **Everything else under the
#: repo root is in scope**, which is the ABL-462 fix: the sweep shipped walking
#: `src/` alone, so byte-identical unguarded readers were caught under `src/`
#: and passed silently under `scripts/` -- the directory ABL-247's gated
#: backtest actually occupies.  A denylist of non-source directories cannot
#: acquire a new blind spot when someone adds a directory; an allowlist of
#: roots did, and nothing in the suite noticed for two weeks.
SWEEP_SKIP_DIRS = frozenset({
    ".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "node_modules", "build", "dist",
    "site-packages", ".ipynb_checkpoints",
    # The suite itself is deliberately out of scope: these tests CREATE the
    # three tables in fixtures and assert on their names, so a guard here would
    # be circular -- a test proving the guard nulls a 140 GW row has to be able
    # to write one.  Pinned by
    # `test_the_suite_is_out_of_scope_deliberately_not_incidentally`.
    "tests",
})


def _swept_python_files(root: Path):
    """Every `*.py` under ``root`` that is repo source, not tooling or fixture."""
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if set(relative.parts[:-1]) & SWEEP_SKIP_DIRS:
            continue
        yield path


def unguarded_tso_readers(root: Path, guarded=GUARDED_READ_SITES, exempt=()):
    """Files under ``root`` naming a TSO forecast table with no guard or exemption.

    Factored out of the assertion so the sweep can be pointed at a synthetic
    tree and *proved to fire* -- see the positive controls below.  A scope check
    that has never been observed to fail is exactly the shape of defect ABL-462
    was filed about.
    """
    found = []
    for path in _swept_python_files(root):
        relative = path.relative_to(root).as_posix()
        if relative in exempt or relative in guarded:
            continue
        if relative.endswith("tso_plausibility.py"):
            continue
        text = path.read_text(encoding="utf-8")
        if any(table in text for table in TSO_TABLES):
            found.append(relative)
    return found


#: Files that name a TSO forecast table but do not read one, with the reason.
#: Kept separate from the deliberate-raw-read exemptions because the claim is
#: different and is independently pinned by
#: `test_mention_only_exemptions_execute_no_query_against_a_tso_table`.
MENTION_ONLY_EXEMPT = (
    # ABL-462: names both live tables only inside the `source_cutoffs` block of
    # the attestation manifest it emits -- a description of where V014's
    # features come from.  The read itself is `src/challengers/v014_features.py`,
    # which is on the guarded list.  Measured: no FROM/JOIN against either table
    # anywhere in the file.
    "scripts/attest_net_position_serve_faithfulness.py",
)

#: Files that read a TSO forecast column raw, on purpose, with the reason.
EXEMPT_READS = (
    # Column->covariate mapping only; the read itself is input_builder's.
    "src/chronos2/covariate_mapper.py",
    # `TSOBaseline` queries a `timestamp_utc` column that neither TSO table
    # has (they use `target_timestamp_utc`), so every call raises
    # OperationalError. Wiring a guard into a read that cannot execute would
    # assert a coverage this module does not have. Left as found: it is a
    # separate defect and fixing it is not this issue.
    "src/baselines.py",
    # chronos-bolt legacy path; its runner points at a venv that does not
    # exist on this box (CLAUDE.md, "Skipped is a flag"), so it is unrunnable
    # here and cannot be verified end-to-end under this change.
    "src/chronos_forecaster.py",
    "src/chronos_train.py",
)

SWEEP_EXEMPT = MENTION_ONLY_EXEMPT + EXEMPT_READS


def test_no_unguarded_module_reads_a_tso_forecast_table():
    """The inverse, so the guarded list above cannot quietly go stale.

    Any repo module naming one of the three TSO forecast tables is either on the
    guarded list or on an acknowledged-exempt list, with a reason.  This is the
    check that fires when ABL-247 adds its feature read -- and ABL-462 is why it
    now walks `scripts/`, which is where that read will be written.

    `forecast_vintage_archive` is in the list because ABL-247 reads *that*
    table, not the two live ones -- it needs issued vintages, and the archive is
    the only place they exist. Without it here this test would have passed while
    ABL-247 fitted straight through the guard on the same 96 HU rows (ABL-458).
    """
    unguarded = unguarded_tso_readers(REPO_ROOT, exempt=SWEEP_EXEMPT)
    assert not unguarded, (
        f"these modules read a TSO forecast table without the ABL-431 guard and "
        f"are not on the exempt list: {unguarded}")


# --------------------------------------------------------------------------
# ABL-462: the sweep's scope, proved rather than declared
# --------------------------------------------------------------------------

#: An unguarded reader, byte-identical wherever it is planted. This is the
#: control ABL-462 ran by hand against `origin/main` = e0ec351: dropped in
#: `src/` the sweep named it, dropped in `scripts/` the suite returned 35
#: passed.  Keeping it as a fixture is what stops that from recurring.
_UNGUARDED_PROBE = '''"""Positive control: an unguarded TSO archive read."""
import pandas as pd


def read(conn):
    return pd.read_sql("SELECT * FROM forecast_vintage_archive", conn)
'''

#: Every directory of the repo the sweep must cover, derived from the tree
#: rather than listed, so a new source directory arrives already controlled.
SWEPT_DIRECTORIES = tuple(sorted({
    path.relative_to(REPO_ROOT).parts[0]
    if len(path.relative_to(REPO_ROOT).parts) > 1 else "."
    for path in _swept_python_files(REPO_ROOT)}))


@pytest.mark.parametrize("directory", SWEPT_DIRECTORIES)
def test_sweep_catches_an_unguarded_read_in_every_swept_directory(
        tmp_path, directory):
    """The positive control, one per directory the repo actually keeps code in.

    Without this, a passing sweep proves nothing -- which was the defect: the
    `src/`-only walk returned clean over an unguarded `scripts/` read.
    """
    target = tmp_path if directory == "." else tmp_path / directory
    target.mkdir(parents=True, exist_ok=True)
    probe = target / "_control_probe.py"
    probe.write_text(_UNGUARDED_PROBE, encoding="utf-8")

    found = unguarded_tso_readers(tmp_path, exempt=SWEEP_EXEMPT)

    assert found == [probe.relative_to(tmp_path).as_posix()], (
        f"an unguarded archive read under {directory!r} was not detected; the "
        f"sweep does not cover that directory")


def test_the_directories_abl247_will_write_in_are_swept():
    """Named, not inferred: ABL-247's gated backtest lands here, not in `src/`.

    `abl247-prereg` requires every archive read to go through `guard_tso_series`.
    That requirement is only enforceable if the sweep can see the directory the
    work occupies.
    """
    assert {"scripts", "experiments", "src"} <= set(SWEPT_DIRECTORIES), (
        f"sweep covers {SWEPT_DIRECTORIES}")


def test_the_suite_is_out_of_scope_deliberately_not_incidentally(tmp_path):
    """`tests/` is excluded by a named rule, so the exclusion stays reviewed.

    The fixtures in this file create all three tables and write a 140,996 MW row
    on purpose. Sweeping them would make the guard's own negative controls
    unwritable.
    """
    assert "tests" in SWEEP_SKIP_DIRS
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_probe.py").write_text(_UNGUARDED_PROBE,
                                                      encoding="utf-8")

    assert unguarded_tso_readers(tmp_path, exempt=SWEEP_EXEMPT) == []


_QUERY_CONTEXT = re.compile(r"\b(?:FROM|JOIN|INTO|UPDATE)\s+([A-Za-z_][\w]*)",
                            re.IGNORECASE)


@pytest.mark.parametrize("relative_path", MENTION_ONLY_EXEMPT)
def test_mention_only_exemptions_execute_no_query_against_a_tso_table(
        relative_path):
    """A `names it but never reads it` exemption must not rot into a real read.

    The other exemptions are claims about intent and can only be reviewed. This
    one is a claim about the file's contents, so it is checked: if a
    `FROM energy_load_forecast` is ever added here, the exemption fails rather
    than covering it.
    """
    text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    queried = {name.lower() for name in _QUERY_CONTEXT.findall(text)}
    assert not queried & set(TSO_TABLES), (
        f"{relative_path} is exempt as mention-only but now queries "
        f"{sorted(queried & set(TSO_TABLES))}; guard the read or move it to "
        f"EXEMPT_READS with a reason")
