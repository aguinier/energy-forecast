"""Pre-registered protocol constants for the ABL-253 solar retrain gate."""

from src.evaluation.wind_retrain import (
    INTENDED_N,
    PRIMARY_BANDS,
    RUN_TIMES,
    SCHEDULE_N,
    attach_baselines,
    build_vintage_frame,
    common_scores,
    finite_training_rows,
    gate_cell,
    schedule_vintages,
    scored_with_comparators,
    select_latest_challenger_per_band,
)
from src.solar_features import SOLAR_GEOMETRY_FEATURES


# ABL-395. The last two names are ABL-338's geometry features, and they arrive
# here by splat from `solar_features.SOLAR_GEOMETRY_FEATURES` rather than spelled
# out, so this list and the builder that has to produce them cannot name
# different columns. `RenewableFeatureBuilder` has emitted both for
# `forecast_type='solar'` since ABL-338 (`wind_features._solar_geometry_features`);
# only this list never asked for them, so every gate fit ran at 25 features where
# an ABL-338-current solar fit is 27.
#
# **This is the half of ABL-338 that was adopted.** The non-negativity constraint
# was measured and *rejected* there (+15.8% Tweedie, +36.8% Poisson daylight MAE
# against a like-for-like refit), and `nonneg_objective=None` on every gate
# artifact correctly records that; nothing here brings it back. The geometry
# features were the half that passed — daylight-safe at mean -1.0% and worst
# +2.9% across ABL-338's eight country-windows — for the mechanism ABL-338
# identified: at every night hour all three radiation columns and both same-hour
# lags read exactly 0.0, so nothing in the 25-name vector distinguished "0 W/m2
# because the sun is down" from "0 W/m2 at a dark winter dawn", and the ensemble's
# value near that origin was wherever its residual happened to settle.
#
# Two consequences, both deliberate and neither cosmetic:
#
# - **A re-run of `abl253` or `abl316-t1b` no longer reproduces its published
#   read.** Those artifacts were fitted at 25 features; a fit from this commit
#   forward is a different challenger, and the two are not comparable cell by
#   cell. The published reads stand as read — their `feature_columns` is in each
#   artifact and their disposition is dispositioned — and whether they are refit
#   is ABL-401, not this list. Every other registered property (countries, bands,
#   bar, windows, metric, basis, outputs) is untouched.
# - **Every country in `config.SUPPORTED_COUNTRIES` must have a solar
#   representative point**, because `to_vector` raises on a column the builder
#   did not produce and `_solar_geometry_features` contributes nothing for a
#   country absent from `solar_geometry.SOLAR_REPRESENTATIVE_POINTS`. All 24 do
#   today; `tests/test_gate_feature_list_contract.py` holds that, so a new
#   ABL-316 tranche fails in the test suite rather than at its first fit.
FEATURE_COLUMNS = (
    "hour", "day_of_week", "month", "is_weekend", "hour_sin", "hour_cos",
    "day_sin", "day_cos", "month_sin", "month_cos",
    "target_value_lag_1d", "target_value_lag_7d", "target_value_lag_14d",
    "target_value_roll_24h_mean", "target_value_roll_24h_std",
    "target_value_roll_24h_min", "target_value_roll_24h_max",
    "target_value_roll_168h_mean", "target_value_roll_168h_std",
    "target_value_roll_168h_min", "target_value_roll_168h_max",
    "shortwave_radiation_wm2", "direct_radiation_wm2",
    "diffuse_radiation_wm2", "temperature_c",
    *SOLAR_GEOMETRY_FEATURES,
)

COUNTRIES = ("BE", "DE", "FR")
ALGORITHM = "catboost"

# ABL-389's `attach_model_free_references` and the two level functions behind it
# are deliberately *not* re-exported here. They live in
# `src.evaluation.model_free_reference`, which both harnesses import directly, so
# that the two gates cannot end up computing the same named reference by two
# routes. This module re-exports the shared *protocol*; the model-free reference
# is shared code, not a solar protocol constant.
__all__ = (
    "ALGORITHM", "COUNTRIES", "FEATURE_COLUMNS", "INTENDED_N", "PRIMARY_BANDS",
    "RUN_TIMES", "SCHEDULE_N", "SOLAR_GEOMETRY_FEATURES", "attach_baselines",
    "build_vintage_frame", "common_scores", "finite_training_rows", "gate_cell",
    "schedule_vintages", "scored_with_comparators",
    "select_latest_challenger_per_band",
)
