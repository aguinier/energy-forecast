"""Pure correctness checks for the all-type forecast scorecard (ABL-129)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baselines import aligned_point_baselines
from src.evaluation.scorecard import (
    filter_measured_actuals, horizon_band, normalize_timestamps, score_against_baseline,
    render_markdown, score_predictions, select_latest_per_band,
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
        },
        "pooled": [{"forecast_type": "load", "model_name": "catboost",
                    "horizon_band": "all", "model": empty,
                    "baselines": {name: comparison for name in
                                  ("seasonal_naive", "persistence", "tso")}}],
        "by_country_horizon": [],
    }
    report = render_markdown(results, "now")
    assert "| load | catboost | all | 0 | Not measured" in report
