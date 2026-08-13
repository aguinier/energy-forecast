"""Pure correctness checks for the all-type forecast scorecard (ABL-129)."""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db import RENEWABLE_TYPE_COLUMNS
from src.baselines import aligned_point_baselines
from src.evaluation.scorecard import (
    ACTUAL_SPECS, GENERATION_RENEWABLE_COLUMNS, RETIRED_RENEWABLE_ACTUAL_SPECS,
    filter_measured_actuals, horizon_band, mean_scored_actual, normalize_timestamps,
    null_aware_sum, score_against_baseline, render_markdown, score_predictions,
    select_latest_per_band,
)


def test_normalize_timestamps_accepts_both_separators():
    parsed = normalize_timestamps(["2026-08-01T03:00:00", "2026-08-01 03:00:00"])
    assert parsed.iloc[0] == parsed.iloc[1] == pd.Timestamp("2026-08-01 03:00:00")


def test_score_predictions_empty_and_zero_denominator_are_not_measured():
    assert score_predictions([], []) == {
        "n": 0, "wape_pct": None, "mae": None, "bias_pct": None,
        "slope": None, "correlation": None,
    }
    measured_zero = score_predictions([0.0, 0.0], [1.0, 2.0])
    assert measured_zero["n"] == 2
    assert measured_zero["wape_pct"] is None
    assert measured_zero["bias_pct"] is None


def test_score_recovers_signed_bias_slope_and_skill_on_same_pairs():
    score = score_predictions([10.0, 20.0, 30.0], [5.0, 10.0, 15.0])
    assert score["wape_pct"] == 50.0
    assert score["bias_pct"] == -50.0
    assert score["slope"] == pytest.approx(0.5)
    assert score["correlation"] == pytest.approx(1.0)

    comparison = score_against_baseline(
        [10.0, 20.0, 30.0], [5.0, 10.0, 15.0], [0.0, np.nan, 0.0])
    assert comparison["n"] == 2
    assert comparison["model_on_same_pairs"]["wape_pct"] == 50.0
    assert comparison["baseline"]["wape_pct"] == 100.0
    assert comparison["skill_pct"] == 50.0


def test_aligned_baselines_use_literal_d7_and_no_post_generation_actual():
    index = pd.date_range("2026-07-01", "2026-08-02", freq="h")
    history = pd.Series(np.arange(len(index), dtype=float), index=index)
    targets = pd.DatetimeIndex(["2026-08-01 00:00:00"])
    generated = pd.DatetimeIndex(["2026-07-31 19:00:24"])
    result = aligned_point_baselines(history, targets, generated)
    assert result.iloc[0]["seasonal_naive"] == history.loc["2026-07-25 00:00:00"]
    # Lead is ceil(4h59m36s) = 5h, so persistence reads 19:00, not 20:00.
    assert result.iloc[0]["persistence"] == history.loc["2026-07-31 19:00:00"]


def test_latest_selection_is_within_each_horizon_band():
    rows = pd.DataFrame({
        "forecast_type": ["load"] * 4,
        "model_name": ["catboost"] * 4,
        "country_code": ["DE"] * 4,
        "target_ts": pd.to_datetime(["2026-08-01 00:00:00"] * 4),
        "generated_at": pd.to_datetime([
            "2026-07-31 19:00:00", "2026-07-31 18:00:00",
            "2026-07-30 19:00:00", "2026-07-30 18:00:00",
        ]),
        "horizon_hours": [4, 5, 28, 29],
        "forecast_value": [1.0, 2.0, 3.0, 4.0],
        "source_rank": [0, 0, 0, 0],
    })
    selected = select_latest_per_band(rows)
    assert set(selected["horizon_band"]) == {"2-12h", "24-36h"}
    assert set(selected["forecast_value"]) == {1.0, 3.0}
    assert horizon_band(0) is None
    assert horizon_band(64) == "48-64h"


def test_actual_rules_drop_only_impossible_load_zero_and_named_gr():
    base = pd.DataFrame({
        "country_code": ["DE", "DE", "GR", "FR"],
        "ts": pd.to_datetime(["2026-08-01 00:00:00"] * 4),
        "actual": [0.0, 1.0, 0.0, 0.0],
    })
    assert filter_measured_actuals(base, "load")["actual"].tolist() == [1.0]
    assert filter_measured_actuals(base, "solar")["actual"].tolist() == [0.0, 1.0, 0.0, 0.0]
    net = filter_measured_actuals(base, "net_position")
    assert set(net["country_code"]) == {"DE", "FR"}


def test_report_renders_empty_comparison_as_not_measured():
    empty = score_predictions([], [])
    comparison = score_against_baseline([], [], [])
    results = {
        "meta": {
            "window": {"start": "a", "end_exclusive": "b"},
            "selected_forecast_rows": 0, "paired_actual_rows": 0,
            "selection": "test", "load_actual_rule": "load only",
            "net_position_gate": "separate", "vintage_counts": {},
            "abl128_reproduction": {},
            "excluded": {"net_position": {"GR": "fabricated zeros"}},
            "scoring_truth": {"load": {"table": "energy_load",
                                       "expression": "load_mw"}},
        },
        "pooled": [{"forecast_type": "load", "model_name": "catboost",
                    "horizon_band": "all", "model": empty,
                    "mean_actual": None,
                    "baselines": {name: comparison for name in
                                  ("seasonal_naive", "persistence", "tso")}}],
        "by_country_horizon": [],
    }
    report = render_markdown(results, "now")
    assert "| load | catboost | all | 0 | Not measured | Not measured" in report
    assert "| load | `energy_load` | `load_mw` |" in report


# --------------------------------------------------------------------------
# ABL-410: one statement of the actual, and it is not the frozen table.
# --------------------------------------------------------------------------

def test_hydro_scoring_truth_is_the_training_side_definition_itself():
    """Not "the same rule" — the same object. A copy is what drifted.

    `scorecard` scored `hydro_run_mw + hydro_reservoir_mw` strictly while
    `db.py` used the null-aware form on the measurement that 9 of the 24
    supported countries report exactly one component. Two definitions of one
    quantity in one repo; this pins them to one.
    """
    table, expression = ACTUAL_SPECS["hydro_total"]
    assert table == "energy_generation"
    assert expression is RENEWABLE_TYPE_COLUMNS["hydro_total"]


def test_no_renewable_family_type_is_scored_against_the_frozen_table():
    family = ("renewable", "solar", "wind_onshore", "wind_offshore", "biomass",
              "hydro_total")
    assert {ACTUAL_SPECS[t][0] for t in family} == {"energy_generation"}
    # The retired mapping is a record, not a fallback: it must stay the frozen
    # one, and nothing may quietly score against it again.
    assert {t[0] for t in RETIRED_RENEWABLE_ACTUAL_SPECS.values()} == {"energy_renewable"}
    assert set(RETIRED_RENEWABLE_ACTUAL_SPECS) == set(family)


def test_renewable_total_excludes_pumped_storage():
    """The store is not a primary source, and folding it in is the ABL-410 bug.

    `energy_renewable` has no `hydro_pumped_mw` column at all — it folds pumping
    into `hydro_reservoir_mw`, which is why BE's frozen "hydro" actual was
    84.7% pumped storage across the 22,641 hours both tables carry, and 99.3%
    of it over the last published window. The list must not reintroduce it.
    """
    assert "hydro_pumped_mw" not in GENERATION_RENEWABLE_COLUMNS
    assert "energy_storage_mw" not in GENERATION_RENEWABLE_COLUMNS
    assert GENERATION_RENEWABLE_COLUMNS == (
        "solar_mw", "wind_onshore_mw", "wind_offshore_mw",
        "hydro_run_mw", "hydro_reservoir_mw", "biomass_mw",
        "geothermal_mw", "marine_mw", "other_renewable_mw",
    )


def test_null_aware_sum_keeps_a_partial_report_and_a_measured_zero():
    """One unreported component must not erase the ones beside it, and a
    country reporting none of them must not read as generating zero."""
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE t (label TEXT, a REAL, b REAL, c REAL)")
    con.executemany("INSERT INTO t VALUES (?, ?, ?, ?)", [
        ("all reported", 1.0, 2.0, 3.0),
        ("one missing", 1.0, None, 3.0),
        ("measured zero", 0.0, 0.0, 0.0),
        ("none reported", None, None, None),
    ])
    expression = null_aware_sum(("a", "b", "c"))
    rows = dict(con.execute(f"SELECT label, {expression} FROM t").fetchall())
    con.close()
    assert rows["all reported"] == 6.0
    assert rows["one missing"] == 4.0        # strict `+` would give NULL
    assert rows["measured zero"] == 0.0      # a reading, not an absence
    assert rows["none reported"] is None     # COALESCE-only would give 0.0


def test_mean_actual_covers_exactly_the_scored_pairs():
    """WAPE's denominator level, on the same mask `score_predictions` uses."""
    group = pd.DataFrame({
        "actual": [10.0, 20.0, np.nan, 30.0],
        "forecast_value": [1.0, 2.0, 3.0, np.nan],
    })
    assert mean_scored_actual(group) == pytest.approx(15.0)
    assert score_predictions(group["actual"], group["forecast_value"])["n"] == 2
    assert mean_scored_actual(pd.DataFrame({"actual": [], "forecast_value": []})) is None
