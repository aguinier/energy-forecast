"""ABL-607: the diagnosis' archive read is guarded, and the guard actually fires.

ABL-462 widened the plausibility sweep past `src/`, and it caught this script's
`forecast_vintage_archive` read. Wiring the call satisfies the static sweep --
but a static sweep only proves the *name* `guard_tso_frame` appears in the file.
The pack's published answer is "**0 rows refused over the window**", and that
sentence is worthless if the wiring could not have refused anything. So the
refusal is exercised here on a synthetic replica, at the scale of the incident
the guard was written for (HU's 140,996 MW against a 283 MW fleet, ABL-431).

Not run against the live replica on purpose: a rule pinned against live data
stops being a test the day the data moves. The live refusal count is measured by
the script itself and reported in section 0 of
`reports/abl_607_d2_load_diagnosis.json`.
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.abl607_d2_load_diagnosis import guard_ml_vintages  # noqa: E402
from src.tso_plausibility import clear_reference_cache  # noqa: E402


@pytest.fixture(autouse=True)
def _no_cross_test_cache():
    """The reference cache is process-wide and keyed on the database file --
    every in-memory fixture reports the same key, so they must not share one."""
    clear_reference_cache()
    yield
    clear_reference_cache()


def _hours(n, start="2026-08-13"):
    return pd.date_range(start, periods=n, freq="h")


def _replica(load_actuals, tso_day_ahead=()):
    """A replica-shaped database holding only what the reference is built from."""
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE energy_load (
            country_code TEXT, timestamp_utc TIMESTAMP, load_mw REAL);
        CREATE TABLE forecast_vintage_archive (
            source TEXT NOT NULL, forecast_type TEXT NOT NULL,
            country_code TEXT NOT NULL, target_timestamp_utc TEXT NOT NULL,
            model_name TEXT NOT NULL, run_timestamp_utc TEXT NOT NULL,
            horizon_hours INTEGER, forecast_value REAL NOT NULL,
            first_seen_at TEXT NOT NULL);
    """)
    conn.executemany(
        "INSERT INTO energy_load (country_code, timestamp_utc, load_mw) "
        "VALUES (?, ?, ?)",
        [("HU", str(t), float(v)) for t, v in
         zip(_hours(len(load_actuals), "2026-01-01"), load_actuals)])
    conn.executemany(
        "INSERT INTO forecast_vintage_archive "
        "(source, forecast_type, country_code, target_timestamp_utc, "
        " model_name, run_timestamp_utc, horizon_hours, forecast_value, "
        " first_seen_at) VALUES ('tso', 'load', 'HU', ?, 'tso-day_ahead', "
        " '2026-01-01T00:00:00Z', 24, ?, '2026-01-01T00:00:00Z')",
        [(str(t), float(v)) for t, v in
         zip(_hours(len(tso_day_ahead), "2026-01-01"), tso_day_ahead)])
    conn.commit()
    return conn


def _ml_frame(values):
    """The frame `load_archive` hands the guard, with only the columns it uses."""
    targets = _hours(len(values))
    return pd.DataFrame({
        "country_code": "HU",
        "target_timestamp_utc": [str(t) for t in targets],
        "model_name": "load-catboost",
        "horizon_hours": 30,
        "forecast_value": [float(v) for v in values],
        "first_seen_at": "2026-08-12T06:00:00Z",
        "target": targets,
    })


def test_the_read_refuses_a_kw_published_as_mw_row():
    """The ABL-431 incident's own scale, arriving on our arm instead of theirs."""
    conn = _replica(load_actuals=[280.0] * 500)
    df = _ml_frame([275.0] * 47 + [140996.245])

    out = guard_ml_vintages(df, conn)

    assert out.attrs["guard_refusals"] == 1
    assert len(out) == 47
    assert 140996.245 not in set(out["forecast_value"])
    census = {row["country"]: row for row in out.attrs["guard_census"]}
    assert census["HU"]["n_refused"] == 1
    assert census["HU"]["max_over_threshold"] > 100


def test_a_plausible_arm_is_returned_unchanged_and_reports_its_headroom():
    """The other half: no row is refused, and the pass carries the ratio it
    cleared by. A bare `0 refused` is not evidence; this is what section 0 of
    the record reports so the number can be read."""
    conn = _replica(load_actuals=[280.0] * 500)
    values = [200.0, 275.0, 310.0, 250.0]
    df = _ml_frame(values)

    out = guard_ml_vintages(df, conn)

    assert out.attrs["guard_refusals"] == 0
    assert list(out["forecast_value"]) == values
    census = out.attrs["guard_census"][0]
    assert census["threshold_mw"] == pytest.approx(3.0 * 280.0)
    assert census["max_over_threshold"] == pytest.approx(310.0 / 840.0)


def test_the_reference_is_not_set_by_our_own_arm():
    """The registration's load-bearing exclusion, exercised end to end.

    Our `source = 'ml'` rows are excluded from the reference by
    `forecast_read`, so an ML arm that overshot cannot lift the bar it is held
    to. If it could, the guard would certify whatever we happened to publish.
    """
    conn = _replica(load_actuals=[280.0] * 500)
    df = _ml_frame([140996.245] * 400)

    out = guard_ml_vintages(df, conn)

    # All 400 refused: the bar stays at 3 x 280 MW however many rows we publish
    # above it.
    assert out.attrs["guard_refusals"] == 400
    assert out.empty
    assert out.attrs["guard_census"][0]["threshold_mw"] == pytest.approx(840.0)


def test_the_test_is_one_sided_so_an_under_forecast_survives():
    """A low outlier is never refused. It must not be: an under-forecast is a
    real error of the arm this pack is measuring, and a guard that removed it
    would flatter the arm it is checking."""
    conn = _replica(load_actuals=[280.0] * 500)
    df = _ml_frame([0.0, 1.0, 275.0])

    out = guard_ml_vintages(df, conn)

    assert out.attrs["guard_refusals"] == 0
    assert list(out["forecast_value"]) == [0.0, 1.0, 275.0]


def test_row_order_survives_the_per_country_split():
    """`build_ml_arms` breaks ties with `sort_values(...).groupby(...).last()`
    and pandas' default sort is quicksort, which is not stable. The guard has
    to hand back the rows in the order it got them, or a zero-refusal run
    could still move a vintage selection."""
    conn = _replica(load_actuals=[280.0] * 500)
    df = _ml_frame([200.0, 210.0, 220.0, 230.0])
    df.loc[[0, 2], "country_code"] = "SK"
    conn.executemany(
        "INSERT INTO energy_load (country_code, timestamp_utc, load_mw) "
        "VALUES ('SK', ?, 280.0)",
        [(str(t),) for t in _hours(500, "2026-01-01")])
    conn.commit()

    out = guard_ml_vintages(df, conn)

    assert list(out.index) == [0, 1, 2, 3]
    assert list(out["country_code"]) == ["SK", "HU", "SK", "HU"]
    assert list(out["forecast_value"]) == [200.0, 210.0, 220.0, 230.0]


def test_a_non_evaluable_reference_fails_open_and_says_so():
    """A country with no fleet history is passed through unguarded rather than
    emptied. The `evaluable` flag is in the census for exactly this reason: a
    zero-refusal cell that was never evaluated is not a cell that passed."""
    conn = _replica(load_actuals=[0.0] * 500)
    df = _ml_frame([200.0, 5000.0])

    out = guard_ml_vintages(df, conn)

    assert out.attrs["guard_refusals"] == 0
    assert len(out) == 2
    census = out.attrs["guard_census"][0]
    assert census["evaluable"] is False
    assert census["max_over_threshold"] is None
