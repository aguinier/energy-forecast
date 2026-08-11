"""Pure checks for the ABL-187 correction experiment."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.price_correction import (
    evaluate_country,
    fit_bias_only,
    fit_slope_intercept,
)


def test_bias_only_recovers_training_mean_residual():
    correction = fit_bias_only(pd.Series([12.0, 22.0]), pd.Series([10.0, 20.0]))
    assert correction.slope == 1.0
    assert correction.intercept == 2.0


def test_affine_recovers_slope_and_intercept():
    correction = fit_slope_intercept(
        pd.Series([5.0, 8.0, 11.0]), pd.Series([1.0, 2.0, 3.0]))
    assert correction.slope == pytest.approx(3.0)
    assert correction.intercept == pytest.approx(2.0)


def test_evaluation_uses_common_finite_holdout_intersection():
    train = pd.DataFrame({"actual": [12.0, 22.0, 32.0], "forecast_value": [10.0, 20.0, 30.0]})
    holdout = pd.DataFrame({
        "actual": [42.0, 52.0],
        "forecast_value": [40.0, 50.0],
        "seasonal_naive": [41.0, float("nan")],
    })
    scores, rows = evaluate_country(train, holdout)
    assert scores["n_fit"] == 3
    assert scores["n"] == len(rows) == 1
    assert scores["bias_only"]["wape_pct"] == pytest.approx(0.0)
    assert scores["raw"]["wape_pct"] == pytest.approx(100 * 2 / 42)
