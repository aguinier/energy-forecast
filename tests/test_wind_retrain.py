"""Protocol checks for the pre-registered ABL-195 wind gate."""

import pandas as pd

from src.evaluation.wind_retrain import INTENDED_N, SCHEDULE_N, gate_cell, schedule_vintages
from src.evaluation.scorecard import horizon_band


def test_measured_schedule_reproduces_registered_band_counts():
    counts = {name: 0 for name in INTENDED_N}
    for target in pd.date_range("2026-07-11", "2026-08-10", freq="h", inclusive="left"):
        for generated in schedule_vintages(target):
            band = horizon_band((target - generated).total_seconds() / 3600)
            if band is not None:
                counts[band] += 1
    # The gate selects the latest vintage per target within each band.
    selected = {name: 0 for name in INTENDED_N}
    for target in pd.date_range("2026-07-11", "2026-08-10", freq="h", inclusive="left"):
        bands = {horizon_band((target - generated).total_seconds() / 3600)
                 for generated in schedule_vintages(target)}
        for band in bands - {None}:
            selected[band] += 1
    assert selected == SCHEDULE_N
    assert selected != INTENDED_N  # pre-registration count arithmetic did not reproduce
    assert sum(counts.values()) == 30 * 24 * 8


def test_gate_is_strict_and_requires_95_percent_of_pairs():
    assert gate_cell(9.9, 10.0, 684, 720)["pass"] is True
    assert gate_cell(10.0, 10.0, 684, 720)["pass"] is False
    assert gate_cell(9.9, 10.0, 683, 720)["pass"] is False
