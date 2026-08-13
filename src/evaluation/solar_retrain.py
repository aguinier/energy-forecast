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
    gate_verdict,
    schedule_vintages,
    scores_with_comparators,
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

ALGORITHM = "catboost"

# ABL-379: `COUNTRIES = ("BE", "DE", "FR")` used to live here and was iterated
# directly by the harness, which made reading a gate for any other country an
# edit to this file. The registered pair set is now `SCOPES` in
# `scripts/evaluate_solar_retrain.py`, alongside the gate basis and cell count
# that have to move with it -- exactly as the wind harness carries its own.
# A second "registered countries" constant next to that table would be a second
# source of truth for the same fact, so it is gone rather than left dangling.

__all__ = (
    "ALGORITHM", "FEATURE_COLUMNS", "INTENDED_N", "PRIMARY_BANDS",
    "RUN_TIMES", "SCHEDULE_N", "attach_baselines", "build_vintage_frame",
    "common_scores", "finite_training_rows", "gate_cell", "gate_verdict",
    "schedule_vintages", "scores_with_comparators",
    "select_latest_challenger_per_band",
)
