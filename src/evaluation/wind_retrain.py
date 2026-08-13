"""Serve-faithful wind retraining primitives for the ABL-195 gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.baselines import aligned_point_baselines
from src.evaluation.scorecard import horizon_band, score_predictions
from src.wind_features import RenewableFeatureBuilder, to_vector


FEATURE_COLUMNS = (
    "hour", "day_of_week", "month", "is_weekend", "hour_sin", "hour_cos",
    "day_sin", "day_cos", "month_sin", "month_cos",
    "target_value_lag_1d", "target_value_lag_7d", "target_value_lag_14d",
    "target_value_roll_24h_mean", "target_value_roll_24h_std",
    "target_value_roll_24h_min", "target_value_roll_24h_max",
    "target_value_roll_168h_mean", "target_value_roll_168h_std",
    "target_value_roll_168h_min", "target_value_roll_168h_max",
    "wind_speed_100m_ms", "wind_speed_10m_ms", "temperature_c",
)

RUN_TIMES = ((2, "07:00"), (2, "14:00"), (2, "15:30"), (2, "19:00"),
             (1, "07:00"), (1, "14:00"), (1, "15:30"), (1, "19:00"))
PRIMARY_BANDS = ("24-36h", "36-48h", "48-64h")
INTENDED_N = {"2-12h": 240, "12-24h": 600, "24-36h": 720,
              "36-48h": 720, "48-64h": 480}
# Exact counts implied by the eight pre-registered run instants.  These differ
# from three intended counts written in the issue; the discrepancy was found
# by a protocol test before fitting/scoring and must remain visible.
SCHEDULE_N = {"2-12h": 210, "12-24h": 570, "24-36h": 720,
              "36-48h": 720, "48-64h": 510}


def schedule_vintages(target_timestamp: pd.Timestamp) -> list[pd.Timestamp]:
    """The eight renewable run instants pre-registered for each target day."""
    day = pd.Timestamp(target_timestamp).normalize()
    return [pd.Timestamp(f"{(day - pd.Timedelta(days=days)).date()} {clock}")
            for days, clock in RUN_TIMES]


def build_vintage_frame(
    builder: RenewableFeatureBuilder,
    start,
    end,
    feature_columns: Iterable[str] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Call the shared builder once per target/vintage, retaining provenance counts."""
    rows = []
    actuals = builder._actuals  # the builder's ABL-188-filtered exact-hour series
    for target in pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="h", inclusive="left"):
        actual = actuals.get(target, np.nan)
        for generated_at in schedule_vintages(target):
            features = builder.row(target, generated_at, generated_at)
            vector = to_vector(features, feature_columns)
            rows.append({
                "target_ts": target,
                "generated_at": generated_at,
                "horizon_hours": (target - generated_at).total_seconds() / 3600.0,
                "horizon_band": horizon_band((target - generated_at).total_seconds() / 3600.0),
                "actual": float(actual) if pd.notna(actual) else np.nan,
                "degraded_lag_1d": bool(features["target_value_lag_1d"].degraded),
                **vector,
            })
    return pd.DataFrame(rows)


def finite_training_rows(
    frame: pd.DataFrame,
    feature_columns: Iterable[str] = FEATURE_COLUMNS,
) -> tuple[pd.DataFrame, dict]:
    required = ["actual", *feature_columns]
    valid = np.isfinite(frame[required].to_numpy(dtype=float)).all(axis=1)
    kept = frame.loc[valid].reset_index(drop=True)
    return kept, {
        "intended_rows": int(len(frame)),
        "retained_rows": int(valid.sum()),
        "excluded_missing_actual_or_feature": int((~valid).sum()),
        "unique_targets": int(kept["target_ts"].nunique()),
        "degraded_lag_1d_rows": int(kept["degraded_lag_1d"].sum()),
    }


def select_latest_challenger_per_band(frame: pd.DataFrame) -> pd.DataFrame:
    usable = frame.dropna(subset=["horizon_band"]).copy()
    return (usable.sort_values(["target_ts", "horizon_band", "generated_at"])
                  .drop_duplicates(["target_ts", "horizon_band"], keep="last")
                  .reset_index(drop=True))


def attach_baselines(frame: pd.DataFrame, history: pd.Series) -> pd.DataFrame:
    result = frame.copy()
    baselines = aligned_point_baselines(
        history, pd.DatetimeIndex(result["target_ts"]),
        pd.DatetimeIndex(result["generated_at"]),
    )
    result["seasonal_naive"] = baselines["seasonal_naive"].to_numpy()
    result["persistence"] = baselines["persistence"].to_numpy()
    return result


def common_scores(frame: pd.DataFrame, columns: Iterable[str]) -> tuple[dict, pd.DataFrame]:
    names = ["actual", *columns]
    valid = np.isfinite(frame[names].to_numpy(dtype=float)).all(axis=1)
    common = frame.loc[valid].copy()
    return ({name: score_predictions(common["actual"], common[name]) for name in columns}, common)


def gate_cell(challenger_wape: float | None, naive_wape: float | None,
              n: int, intended_n: int) -> dict:
    min_n = int(np.ceil(0.95 * intended_n))
    enough = n >= min_n
    beats = (challenger_wape is not None and naive_wape is not None
             and challenger_wape < naive_wape)
    return {"pass": bool(enough and beats), "n": n, "intended_n": intended_n,
            "minimum_n": min_n, "beats_d7": bool(beats), "enough_pairs": bool(enough)}


def scores_with_comparators(frame: pd.DataFrame, gate_basis: Iterable[str],
                            reported: Iterable[str]) -> tuple[dict, pd.DataFrame, dict]:
    """Score on the scope's registered gate basis; report the rest beside it.

    The gate basis is the set of columns that must be *simultaneously finite*
    for a row to be scored. Every comparator outside it is scored on its own
    intersection *with* the basis, so a comparator that does not exist for a
    pair costs its own row and nothing else -- it reads "Not measured" instead
    of emptying the cell.

    That distinction is the whole point. With `incumbent` inside the basis, a
    country with zero rows in `forecasts` has NaN on every row, the intersection
    is empty, and the cell scores n=0 with every score None -- which renders as
    a plausible FAIL on a comparison that never happened. Both new-country
    tranches are in exactly that position (ABL-322 for wind, ABL-379 for solar).

    Returns the basis scores, the basis intersection, and each comparator's own n.
    """
    gate_basis, reported = tuple(gate_basis), tuple(reported)
    scores, common = common_scores(frame, gate_basis)
    comparator_n = {name: len(common) for name in gate_basis}
    for name in reported:
        if name in scores:
            continue
        sub_scores, sub_common = common_scores(frame, (*gate_basis, name))
        scores[name], comparator_n[name] = sub_scores[name], len(sub_common)
    return scores, common, comparator_n


def gate_verdict(gate_cells: Iterable[dict], registered_cells: int,
                 contaminated: bool) -> dict:
    """Classify a gate read against its scope's *registered* cell count.

    Shared by both harnesses because the classification is a property of the
    pre-registration, not of the stream, and ABL-322 fixed it in the wind
    harness only -- which is the defect ABL-379 then had to fix again in solar.
    Each harness still owns its own recommendation prose; only the disposition
    is decided here.

    `registered_cells` is the scope's `len(pairs) x len(PRIMARY_BANDS)`, fixed
    in the file before the run. It is deliberately not `len(gate_cells)`: a pair
    that silently yields no rows must fall short of the count rather than
    quietly leave the denominator.

    The fourth outcome is the one ABL-322 added and this issue ports. A cell
    that scored zero rows did not lose a race -- it never ran one. Reporting
    that as FAIL reads as a model-quality verdict on a comparison that never
    happened, and the correct response to the two is opposite.
    """
    cells = list(gate_cells)
    passed = sum(bool(cell["gate"]["pass"]) for cell in cells)
    unreadable = sum(1 for cell in cells if cell["gate"]["n"] == 0)
    performance_pass = len(cells) == registered_cells and passed == registered_cells
    if performance_pass:
        verdict = ("PERFORMANCE PASS — HOLD FOR CONTAMINATION ADJUDICATION"
                   if contaminated else "PASS")
    elif unreadable:
        verdict = "UNREADABLE"
    else:
        verdict = "FAIL"
    return {"verdict": verdict, "passed": passed, "unreadable": unreadable,
            "scored_cells": len(cells), "registered_cells": registered_cells,
            "performance_pass": performance_pass}
