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
    UnknownTsoSourceError,
    clear_reference_cache,
    guard_series,
    guard_tso_frame,
    guard_tso_series,
    implausible_mask,
    reference_scale,
)

REPO_ROOT = Path(__file__).parent.parent

GEN_FORECAST = "energy_generation_forecast"
LOAD_FORECAST = "energy_load_forecast"


@pytest.fixture(autouse=True)
def _no_cross_test_cache():
    """The reference cache is process-wide; two fixtures must not share one."""
    clear_reference_cache()
    yield
    clear_reference_cache()


def _db(gen_forecast=(), generation=(), load_forecast=(), load=()):
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
    """)
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
# The read sites are wired, checked statically
# --------------------------------------------------------------------------

#: Every module that reads a TSO day-ahead forecast column, and therefore must
#: route it through the guard. A new read site added without the guard is the
#: failure mode this pins -- it would pass every other test in the suite and
#: only show up as a model that fitted on a 140 GW row.
GUARDED_READ_SITES = (
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


def test_no_unguarded_module_reads_a_tso_forecast_table():
    """The inverse, so the list above cannot quietly go stale.

    Any `src/` module naming one of the two TSO forecast tables is either on the
    guarded list or on the acknowledged-exempt list below, with a reason. This
    is the check that fires when ABL-247 adds its feature read.
    """
    exempt = {
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
    }
    tables = ("energy_generation_forecast", "energy_load_forecast")
    unguarded = []
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative in exempt or relative in GUARDED_READ_SITES:
            continue
        if relative.endswith("tso_plausibility.py"):
            continue
        text = path.read_text(encoding="utf-8")
        if any(t in text for t in tables):
            unguarded.append(relative)
    assert not unguarded, (
        f"these modules read a TSO forecast table without the ABL-431 guard and "
        f"are not on the exempt list: {unguarded}")
