"""Pure correctness checks for the ABL-188 training-data invariant."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_quality import exclude_suspect_constant_runs, find_suspect_constant_runs


def _hourly_series(values, start="2025-09-08T22:00:00"):
    timestamps = pd.date_range(start, periods=len(values), freq="h")
    return pd.DataFrame({"timestamp_utc": timestamps, "target_value": values})


def test_no_runs_for_normal_diurnal_solar():
    # A plausible solar day: zero at night, a midday peak, never repeating.
    values = [0.0, 0.0, 5.0, 400.0, 900.0, 400.0, 5.0, 0.0, 0.0]
    df = _hourly_series(values)
    assert find_suspect_constant_runs(df, "target_value") == []
    assert exclude_suspect_constant_runs(df, "target_value").equals(df)


def test_flags_long_bit_identical_zero_run_like_de_solar():
    # Reproduces the ABL-188 shape: an exact-zero run far longer than any
    # single night (energy_renewable.solar_mw was 0.0 for 6,408 straight
    # quarter-hours / 1,602 hours for DE, 2025-09-08 to 2025-11-14).
    values = [0.0] * 48
    df = _hourly_series(values)

    runs = find_suspect_constant_runs(df, "target_value", min_run_hours=24.0)
    assert len(runs) == 1
    assert runs[0].value == 0.0
    assert runs[0].n_rows == 48
    assert runs[0].duration_hours == pytest.approx(47.0)

    cleaned = exclude_suspect_constant_runs(df, "target_value", min_run_hours=24.0)
    assert cleaned["target_value"].isna().all()


def test_does_not_flag_a_single_ordinary_night():
    # A dozen consecutive zero hours (one long winter night) is normal and
    # must not be excluded -- only min_run_hours+ counts as suspect.
    values = [0.0] * 12
    df = _hourly_series(values)
    assert find_suspect_constant_runs(df, "target_value", min_run_hours=24.0) == []


def test_missing_day_splits_identical_solar_nights():
    # ABL-253 found FR nighttime zeros on either side of a wholly missing day.
    # Adjacency after sorting is not continuity: the missing interval must
    # split the runs or two legitimate nights become a false contamination hit.
    first_night = pd.date_range("2025-12-31 17:00", periods=28, freq="15min")
    second_night = pd.date_range("2026-01-02 01:45", periods=23, freq="15min")
    df = pd.DataFrame({
        "timestamp_utc": first_night.append(second_night),
        "target_value": 0.0,
    })
    assert find_suspect_constant_runs(df, "target_value", min_run_hours=24.0) == []


def test_short_runs_below_threshold_pass_through_untouched():
    values = [0.0, 0.0, 0.0, 5.0, 0.0, 0.0]
    df = _hourly_series(values)
    cleaned = exclude_suspect_constant_runs(df, "target_value", min_run_hours=24.0)
    assert cleaned["target_value"].tolist() == values


def test_preexisting_nan_is_not_treated_as_a_run():
    values = [np.nan] * 30 + [1.0]
    df = _hourly_series(values)
    assert find_suspect_constant_runs(df, "target_value", min_run_hours=24.0) == []


def test_flags_a_nonzero_constant_run_too():
    # The invariant is not solar/zero-specific -- any implausibly long
    # bit-identical run is a missingness signature, whatever the value.
    values = [123.456] * 30
    df = _hourly_series(values)
    runs = find_suspect_constant_runs(df, "target_value", min_run_hours=24.0)
    assert len(runs) == 1
    assert runs[0].value == 123.456


def test_only_rows_inside_the_run_are_nulled():
    values = [10.0] + [0.0] * 30 + [20.0]
    df = _hourly_series(values)
    cleaned = exclude_suspect_constant_runs(df, "target_value", min_run_hours=24.0)
    assert cleaned["target_value"].iloc[0] == 10.0
    assert cleaned["target_value"].iloc[-1] == 20.0
    assert cleaned["target_value"].iloc[1:-1].isna().all()
