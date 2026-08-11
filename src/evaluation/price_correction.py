"""Leak-free per-country price correction experiment helpers (ABL-187).

This module only fits and scores correction parameters. It has no serving or
registry integration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.evaluation.scorecard import score_predictions


@dataclass(frozen=True)
class AffineCorrection:
    slope: float
    intercept: float

    def predict(self, values: pd.Series) -> pd.Series:
        return self.slope * values + self.intercept

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _finite_pairs(actual: pd.Series, forecast: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    actual_values = actual.to_numpy(dtype=float)
    forecast_values = forecast.to_numpy(dtype=float)
    valid = np.isfinite(actual_values) & np.isfinite(forecast_values)
    return actual_values[valid], forecast_values[valid]


def fit_bias_only(actual: pd.Series, forecast: pd.Series) -> AffineCorrection:
    """Fit an additive mean-residual correction on training rows only."""
    actual_values, forecast_values = _finite_pairs(actual, forecast)
    if len(actual_values) == 0:
        raise ValueError("bias correction requires at least one finite training pair")
    return AffineCorrection(
        slope=1.0,
        intercept=float(np.mean(actual_values - forecast_values)),
    )


def fit_slope_intercept(actual: pd.Series, forecast: pd.Series) -> AffineCorrection:
    """Fit OLS actual = slope * forecast + intercept on training rows only."""
    actual_values, forecast_values = _finite_pairs(actual, forecast)
    if len(actual_values) < 2 or float(np.var(forecast_values)) == 0.0:
        raise ValueError("affine correction requires two varying finite training forecasts")
    design = np.column_stack([forecast_values, np.ones(len(forecast_values))])
    slope, intercept = np.linalg.lstsq(design, actual_values, rcond=None)[0]
    return AffineCorrection(slope=float(slope), intercept=float(intercept))


def evaluate_country(train: pd.DataFrame, holdout: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Fit both variants and score one country's untouched holdout intersection."""
    bias = fit_bias_only(train["actual"], train["forecast_value"])
    affine = fit_slope_intercept(train["actual"], train["forecast_value"])

    scored = holdout[["actual", "forecast_value", "seasonal_naive"]].copy()
    finite = np.isfinite(scored.to_numpy(dtype=float)).all(axis=1)
    scored = scored.loc[finite].copy()
    scored["bias_only"] = bias.predict(scored["forecast_value"])
    scored["affine"] = affine.predict(scored["forecast_value"])

    scores = {
        "n_fit": int(np.isfinite(train[["actual", "forecast_value"]].to_numpy(dtype=float)).all(axis=1).sum()),
        "n": int(len(scored)),
        "fit": {"bias_only": bias.to_dict(), "affine": affine.to_dict()},
        "raw": score_predictions(scored["actual"], scored["forecast_value"]),
        "bias_only": score_predictions(scored["actual"], scored["bias_only"]),
        "affine": score_predictions(scored["actual"], scored["affine"]),
        "seasonal_naive": score_predictions(scored["actual"], scored["seasonal_naive"]),
    }
    raw_wape = scores["raw"]["wape_pct"]
    naive_wape = scores["seasonal_naive"]["wape_pct"]
    for variant in ("bias_only", "affine"):
        corrected_wape = scores[variant]["wape_pct"]
        scores[variant]["delta_vs_raw_points"] = corrected_wape - raw_wape
        scores[variant]["beats_raw"] = corrected_wape < raw_wape
        scores[variant]["beats_naive"] = corrected_wape < naive_wape
    return scores, scored
