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
)

COUNTRIES = ("BE", "DE", "FR")
ALGORITHM = "catboost"

# ABL-389's `attach_constant_references` and `constant_reference_levels` are
# deliberately *not* re-exported here. They live in
# `src.evaluation.constant_reference`, which both harnesses import directly, so
# that the two gates cannot end up computing the same named reference by two
# routes. This module re-exports the shared *protocol*; the constant reference
# is shared code, not a solar protocol constant.
__all__ = (
    "ALGORITHM", "COUNTRIES", "FEATURE_COLUMNS", "INTENDED_N", "PRIMARY_BANDS",
    "RUN_TIMES", "SCHEDULE_N", "attach_baselines", "build_vintage_frame",
    "common_scores", "finite_training_rows", "gate_cell", "schedule_vintages",
    "scored_with_comparators", "select_latest_challenger_per_band",
)
