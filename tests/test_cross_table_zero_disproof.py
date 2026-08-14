"""ABL-200: a zero in `energy_renewable` is adjudicated against
`energy_generation`, not against a duration threshold.

The three classes the CEO registered on the issue, and which this file exists
to keep apart:

  1. **Disprovable** -- BE `wind_offshore` 2025-11-14/15, where the twin table
     reports 5.8-424 MW at the identical instants. Excluded, at any run length.
  2. **Not disprovable, absent twin** -- FR `wind_offshore` 2023-01-01/05-31,
     3,619 quarter-hours of 0.0 whose `energy_generation` sibling is NULL for
     the identical span, consistent with the offshore category genuinely not
     being reported that early. Must NOT be excluded by this rule; it stays
     governed by the 24h duration rule, which already reaches it.
  3. **Not disprovable, corroborated zero** -- a real overnight solar zero, or
     BE `wind_offshore` 2026-03-08/10 where the twin reads -11 to -30 MW (A75
     netting: an idle farm drawing house load, which is what a gross 0.0 looks
     like from the signed side). Must NOT be excluded.

Everything here is synthetic. The replica numbers these shapes are taken from
are in `reports/abl_200_cross_table_zero_disproof.md`, regenerable with
`scripts/abl200_cross_table_zero_census.py`.
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
from src.data_quality import (
    SIBLING_DISPROOF_MIN_CALIBRATION_ROWS,
    SIBLING_DISPROOF_QUANTILE,
    adjudicate_zeros_against_sibling,
    exclude_suspect_constant_runs,
    exclude_zeros_disproved_by_sibling,
)

#: Long enough that the floor is estimable -- the guard refuses to adjudicate
#: below `SIBLING_DISPROOF_MIN_CALIBRATION_ROWS` positive-value instants.
N_HOURS = 3000
START = "2025-01-01"


def _frame(values, start=START):
    return pd.DataFrame({
        "timestamp_utc": pd.date_range(start, periods=len(values), freq="h"),
        "target_value": np.asarray(values, dtype=float),
    })


def _rows(start, stop):
    """Half-open, like every other range in this file. `.loc[a:b]` is
    label-INCLUSIVE on both ends, which is one row more than it reads as."""
    return list(range(start, stop))


def _agreeing_pair(n=N_HOURS, seed=0):
    """A pair whose two tables agree, i.e. a floor of ~0 -- the common case:
    32 of 100 real pairs agree bit-for-bit at least 99% of the time."""
    rng = np.random.default_rng(seed)
    values = np.abs(rng.normal(500.0, 200.0, n)) + 1.0
    return _frame(values), _frame(values.copy())


# ---------------------------------------------------------------------------
# Class 1: disprovable at any run length
# ---------------------------------------------------------------------------


def test_a_six_hour_zero_run_is_excluded_when_the_twin_reports_generation():
    """The case the 24h threshold cannot reach. BE `wind_offshore` carries 105
    flat-zero runs of 6h+ and only 9 of them are 24h+."""
    series, sibling = _agreeing_pair()
    run = _rows(100, 106)
    series.loc[run, "target_value"] = 0.0
    sibling.loc[run, "target_value"] = [424.5, 289.9, 181.4, 154.9, 102.9, 150.6]

    assert exclude_suspect_constant_runs(series, "target_value").loc[
        run, "target_value"
    ].eq(0.0).all(), "a 6h run must be invisible to the duration guard, or this test proves nothing"

    out = exclude_zeros_disproved_by_sibling(series, sibling)
    assert out.loc[run, "target_value"].isna().all()


def test_a_single_isolated_zero_is_excluded_too():
    """`at any run length` includes one row. LV solar's 374 zero-filled rows
    form no qualifying run at all."""
    series, sibling = _agreeing_pair()
    series.loc[500, "target_value"] = 0.0
    sibling.loc[500, "target_value"] = 900.0

    out = exclude_zeros_disproved_by_sibling(series, sibling)
    assert np.isnan(out.loc[500, "target_value"])
    assert out["target_value"].isna().sum() == 1


def test_an_excluded_row_is_nan_not_the_siblings_value():
    """Repair beats delete, but this is neither: the twin holds a different
    vintage and a possibly different netting convention, so it is good enough
    to refute a zero and not good enough to become the target. No stored row is
    touched either -- this is a read-site exclusion."""
    series, sibling = _agreeing_pair()
    series.loc[500, "target_value"] = 0.0
    sibling.loc[500, "target_value"] = 900.0

    out = exclude_zeros_disproved_by_sibling(series, sibling)
    assert np.isnan(out.loc[500, "target_value"])
    assert series.loc[500, "target_value"] == 0.0, "the caller's frame must not be mutated"


# ---------------------------------------------------------------------------
# Class 2: absent twin -- the FR 2023 negative case
# ---------------------------------------------------------------------------


def test_fr_2023_shape_is_not_excluded_by_this_rule():
    """FR `wind_offshore` 2023-01-01 -> 2023-05-31: 3,619 quarter-hours of 0.0
    whose sibling is NULL for the identical span. A NULL sibling disproves
    nothing -- it agrees that nothing was reported."""
    series, sibling = _agreeing_pair()
    outage = _rows(200, 700)
    series.loc[outage, "target_value"] = 0.0
    # `load_renewable_type_data` drops NULL rows, so an absent sibling is an
    # absent *instant*, not a NaN value. Model it the way the loader delivers it.
    sibling = sibling.drop(index=outage).reset_index(drop=True)

    out = exclude_zeros_disproved_by_sibling(series, sibling)
    assert out.loc[outage, "target_value"].eq(0.0).all()
    assert out["target_value"].isna().sum() == 0


def test_the_duration_rule_still_reaches_the_fr_shape():
    """...and the reason that is safe: the existing 24h rule already catches
    it. The new rule loosens nothing for the class it declines to judge."""
    series, sibling = _agreeing_pair()
    outage = _rows(200, 700)
    series.loc[outage, "target_value"] = 0.0
    sibling = sibling.drop(index=outage).reset_index(drop=True)

    guarded = exclude_suspect_constant_runs(series, "target_value")
    guarded = exclude_zeros_disproved_by_sibling(guarded, sibling)
    assert guarded.loc[outage, "target_value"].isna().all()


# ---------------------------------------------------------------------------
# Class 3: corroborated zeros -- night, and A75 netting
# ---------------------------------------------------------------------------


def test_a_genuine_overnight_solar_zero_is_not_excluded():
    """Both tables say zero. There is no disagreement to adjudicate, and
    ABL-109's 56 legitimate DE overnight solar zeros must survive."""
    n = N_HOURS
    hours = pd.date_range(START, periods=n, freq="h")
    daylight = ((hours.hour >= 6) & (hours.hour < 20)).astype(float)
    rng = np.random.default_rng(1)
    values = daylight * (rng.random(n) * 800.0 + 50.0)
    series, sibling = _frame(values), _frame(values.copy())

    out = exclude_zeros_disproved_by_sibling(series, sibling)
    assert out["target_value"].isna().sum() == 0
    assert (out["target_value"] == 0.0).sum() == int((daylight == 0).sum())


def test_a_negative_sibling_does_not_disprove_a_zero():
    """BE `wind_offshore` 2026-03-08/10, the window ABL-198 adjudicated: the
    twin reads -11 to -30 MW on all 40 rows. `energy_generation` is signed
    net-of-consumption (A75) and `energy_renewable` holds no negative value in
    any of the 120 pairs, so that is an idle farm drawing house load, not 25 MW
    of hidden generation. 2,158 of BE `wind_offshore`'s 2,214 exact zeros are
    this shape; a test on |sibling| would null every one of them."""
    series, sibling = _agreeing_pair()
    run = _rows(100, 140)
    series.loc[run, "target_value"] = 0.0
    sibling.loc[run, "target_value"] = -25.0

    out = exclude_zeros_disproved_by_sibling(series, sibling)
    assert out.loc[run, "target_value"].eq(0.0).all()


# ---------------------------------------------------------------------------
# The floor: per pair, calibrated, and refused rather than guessed
# ---------------------------------------------------------------------------


def test_a_bit_identical_pair_gets_a_zero_floor():
    series, sibling = _agreeing_pair()
    verdict = adjudicate_zeros_against_sibling(series, sibling)
    assert verdict.evaluable
    assert verdict.floor == pytest.approx(0.0)


def test_a_vintage_divergent_pair_sets_its_own_high_bar_and_falls_quiet():
    """NL `wind_onshore`'s two tables disagree by a median 311.8 MW (the ABL-439
    revision seam). A disproof smaller than the disagreement the pair shows
    routinely is not evidence, and the rule must not fire on it -- without
    anyone choosing a number for NL."""
    series, sibling = _agreeing_pair()
    sibling["target_value"] += 400.0          # a standing vintage offset
    series.loc[500, "target_value"] = 0.0
    sibling.loc[500, "target_value"] = 150.0  # inside that pair's normal disagreement

    verdict = adjudicate_zeros_against_sibling(series, sibling)
    assert verdict.evaluable
    assert verdict.floor > 150.0
    assert verdict.n_disproved == 0

    series.loc[600, "target_value"] = 0.0
    sibling.loc[600, "target_value"] = 5000.0  # far outside it
    assert adjudicate_zeros_against_sibling(series, sibling).n_disproved == 1


def test_an_all_zero_series_is_refused_rather_than_adjudicated():
    """The 20 pairs with no calibration population at all -- landlocked
    countries whose `wind_offshore_mw` is 0.0 forever. With no positive instant
    there is nothing to estimate a floor from, and a floor of 0.0 would let any
    sibling value delete a new fleet's first output. ABL-431's `evaluable`
    pattern, for ABL-431's reason."""
    series = _frame(np.zeros(N_HOURS))
    sibling = _frame(np.full(N_HOURS, 12.0))

    verdict = adjudicate_zeros_against_sibling(series, sibling)
    assert not verdict.evaluable
    assert verdict.n_disproved == 0
    assert "refusing to adjudicate" in verdict.reason
    assert exclude_zeros_disproved_by_sibling(series, sibling)["target_value"].notna().all()


def test_the_calibration_minimum_sits_in_a_measured_empty_band():
    """20 of the 120 pairs have a calibration population of exactly 0 and the
    smallest non-zero one is 2,559, so anything in (0, 2559) is the same rule.
    If this constant is ever moved out of that band it stops being free."""
    assert 0 < SIBLING_DISPROOF_MIN_CALIBRATION_ROWS < 2559


def test_the_floor_is_not_estimated_from_the_defect_it_must_catch():
    """The calibration population is instants where the renewable side is
    strictly positive. A long zero-fill must not raise the bar that would catch
    it -- which is what including the zeros would do."""
    series, sibling = _agreeing_pair()
    fill = _rows(100, 1100)
    series.loc[fill, "target_value"] = 0.0
    sibling.loc[fill, "target_value"] = 900.0

    verdict = adjudicate_zeros_against_sibling(series, sibling)
    assert verdict.calibration_n == N_HOURS - 1000
    assert verdict.floor == pytest.approx(0.0)
    assert verdict.n_disproved == 1000


# ---------------------------------------------------------------------------
# Ordering, alignment, direction
# ---------------------------------------------------------------------------


def test_the_new_rule_runs_after_the_duration_rule_and_cannot_weaken_it():
    """A 48h zero-run with the twin reporting generation on only its middle
    hours. Adjudicating first would null those rows, split the run at the new
    gap, and drop both 20-ish-hour halves under `min_run_hours` -- so rows the
    24h guard excludes today would start entering training. In the registered
    order the two are strictly additive."""
    series, sibling = _agreeing_pair()
    run = _rows(100, 148)
    series.loc[run, "target_value"] = 0.0
    sibling.loc[run, "target_value"] = 0.0
    sibling.loc[_rows(120, 128), "target_value"] = 900.0

    wrong_order = exclude_suspect_constant_runs(
        exclude_zeros_disproved_by_sibling(series, sibling), "target_value"
    )
    registered_order = exclude_zeros_disproved_by_sibling(
        exclude_suspect_constant_runs(series, "target_value"), sibling
    )

    assert registered_order.loc[run, "target_value"].isna().all()
    assert wrong_order.loc[run, "target_value"].notna().any(), (
        "if this stops being true the ordering hazard is gone and the comment "
        "in db.py should be restated, not deleted"
    )


def test_alignment_is_on_parsed_instants_not_stored_spellings():
    """`energy_renewable` stores BE's 2025-11-09 -> 2025-11-25 rows in the ISO
    `2025-11-14T16:00:00` form while `energy_generation` stores every row in the
    `2025-11-14 16:00:00` form. A SQL join on `timestamp_utc` returns NULL for
    all 540 of them -- including every row of the worked example this rule was
    written for."""
    series, sibling = _agreeing_pair()
    series.loc[500, "target_value"] = 0.0
    sibling.loc[500, "target_value"] = 900.0
    # The sibling arrives having been parsed out of the other spelling; if the
    # implementation compared strings it would find nothing here.
    sibling["timestamp_utc"] = pd.to_datetime(
        sibling["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S"), format="mixed"
    )
    assert adjudicate_zeros_against_sibling(series, sibling).n_disproved == 1


def test_a_contradictory_duplicate_instant_in_the_twin_disproves_nothing():
    """`energy_generation` has no duplicate instants today, but a disprover
    that picked one of two contradictory spellings would decide a training row
    on row order."""
    series, sibling = _agreeing_pair()
    series.loc[500, "target_value"] = 0.0
    contradiction = pd.DataFrame({
        "timestamp_utc": [sibling.loc[500, "timestamp_utc"]] * 2,
        "target_value": [0.0, 900.0],
    })
    sibling = pd.concat([sibling.drop(index=500), contradiction], ignore_index=True)
    assert adjudicate_zeros_against_sibling(series, sibling).n_disproved == 0


def test_the_registered_quantile_is_not_a_knife_edge():
    """Over all pairs the rule nulls 896 / 739 / 564 / 416 rows at q = 0.90 /
    0.95 / 0.99 / 1.00 and no acceptance case changes verdict in that range.
    Held here on the disprovable and the netting cases, which are the two that
    must not swap."""
    series, sibling = _agreeing_pair()
    series.loc[_rows(100, 106), "target_value"] = 0.0
    sibling.loc[_rows(100, 106), "target_value"] = 424.5
    series.loc[_rows(200, 206), "target_value"] = 0.0
    sibling.loc[_rows(200, 206), "target_value"] = -25.0

    for q in (0.90, 0.95, 0.99, 1.00):
        verdict = adjudicate_zeros_against_sibling(series, sibling, quantile=q)
        assert verdict.n_disproved == 6, f"verdict moved at q={q}"
    assert SIBLING_DISPROOF_QUANTILE in (0.90, 0.95, 0.99, 1.00)


# ---------------------------------------------------------------------------
# The read site
# ---------------------------------------------------------------------------

COUNTRY = "XX"
DB_START, DB_END = "2025-01-01", "2025-05-01"
_DB_HOURS = pd.date_range("2025-01-01", periods=2400, freq="h")
#: A 6-hour zero-fill: a quarter of `min_run_hours`, invisible to ABL-188.
DB_ZERO_FILL = range(1000, 1006)
#: A 6-hour zero run whose twin is negative -- A75 netting, a real zero.
DB_NETTING_ZEROS = range(1200, 1206)


@pytest.fixture
def replica(tmp_path, monkeypatch):
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
        """
    )
    for i, ts in enumerate(_DB_HOURS):
        onshore = 1000.0 + (i % 700)
        offshore = 300.0 + (i % 250)
        gen_offshore = offshore
        ren_offshore = offshore
        if i in DB_ZERO_FILL:
            ren_offshore = 0.0                      # class 1: disprovable
        if i in DB_NETTING_ZEROS:
            ren_offshore, gen_offshore = 0.0, -25.0  # class 3: corroborated zero
        con.execute(
            "INSERT INTO energy_generation VALUES (?, ?, NULL, ?, ?, NULL, NULL, NULL, 'actual')",
            (COUNTRY, str(ts), onshore, gen_offshore),
        )
        # Stored in the other spelling, as BE's real rows are.
        con.execute(
            "INSERT INTO energy_renewable VALUES (?, ?, 0.0, ?, ?, 0.0, 0.0, 0.0, 'actual')",
            (COUNTRY, ts.isoformat(), onshore, ren_offshore),
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


def _hours_of(frame, offsets):
    wanted = {_DB_HOURS[i] for i in offsets}
    return frame[frame["timestamp_utc"].isin(wanted)]["target_value"]


def test_load_renewable_type_data_excludes_the_disprovable_zeros(replica):
    frame = db.load_renewable_type_data(
        COUNTRY, "wind_offshore", DB_START, DB_END, source="energy_renewable"
    )
    assert len(_hours_of(frame, DB_ZERO_FILL)) == len(DB_ZERO_FILL)
    assert _hours_of(frame, DB_ZERO_FILL).isna().all()


def test_load_renewable_type_data_keeps_the_netting_zeros(replica):
    frame = db.load_renewable_type_data(
        COUNTRY, "wind_offshore", DB_START, DB_END, source="energy_renewable"
    )
    assert _hours_of(frame, DB_NETTING_ZEROS).eq(0.0).all()


def test_the_guard_does_not_run_when_reading_the_honest_table(replica):
    """Directional. `energy_generation` is already the NaN-preserving side and
    has nothing to adjudicate against -- disproving it with `energy_renewable`,
    which cannot encode "not reported", would be unsound."""
    frame = db.load_renewable_type_data(
        COUNTRY, "wind_offshore", DB_START, DB_END, source="energy_generation"
    )
    assert frame["target_value"].notna().all()
    assert _hours_of(frame, DB_NETTING_ZEROS).eq(-25.0).all()


def test_a_stream_the_twin_never_reports_is_untouched(replica):
    """`solar_mw` is 0.0 for every row in `energy_renewable` and NULL for every
    row in `energy_generation` -- the FR 2023 shape at full length. Nothing to
    disprove with, and nothing disproved."""
    frame = db.load_renewable_type_data(
        COUNTRY, "solar", DB_START, DB_END, source="energy_renewable"
    )
    assert not frame.empty
    # The 24h duration rule owns this one, and does reach it.
    assert frame["target_value"].isna().all()


def test_the_disproof_source_is_the_nan_preserving_table(replica):
    assert db.RENEWABLE_ZERO_DISPROOF_SOURCE == "energy_generation"
