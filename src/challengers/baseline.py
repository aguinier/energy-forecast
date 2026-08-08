"""V012 — the floor (ABL-68).

Persistence (same hour, last available day) averaged with 28-day hour-of-day
climatology, using only actuals the run could have seen. Not a promotion
candidate: it exists so that "the challenger beat the baseline" is a claim about
a stored, auditable series rather than a number recomputed at reporting time.

**It calls the evaluation's `baseline_predictions` rather than reimplementing
it.** The gate reads "beats V012 baselines on MAE in >= 80% of countries", and
the eval computes its own `baseline_ensemble` for that comparison. Two
definitions of the same baseline is how renewable share ended up computed three
different ways that disagreed. One definition, imported.
"""

from __future__ import annotations

import pandas as pd

from src.evaluation.net_position import baseline_predictions


def forecast_baseline_ensemble(actuals: pd.Series, as_of: pd.Timestamp,
                               targets: pd.DatetimeIndex,
                               climatology_days: int = 28) -> pd.Series:
    """Mean of persistence and climatology over `targets`, indexed by target.

    Returns NaN — never 0.0 — for a target hour whose components cannot be
    formed, which happens when a country has no actuals before `as_of`. A
    net position of 0 MW is a real, meaningful value (balanced border), so
    filling a gap with it would publish a measurement we never made. The
    caller drops NaN rows instead of storing them.
    """
    preds = baseline_predictions(actuals, as_of, targets, climatology_days)
    return preds[["persistence", "climatology"]].mean(axis=1, skipna=False)
