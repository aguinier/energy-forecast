"""V012 is the same baseline the gate scores against (ABL-68).

The promotion gate says a challenger must "beat V012 baselines on MAE in >= 80%
of countries", and `evaluate_net_position` computes its own `baseline_ensemble`
to check that. If the served V012 and the eval's baseline were two separate
implementations they would eventually disagree, and the gate would be comparing
a challenger against a floor nobody actually served. So V012 calls the eval's
function; these tests pin that it stays that way.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.challengers.baseline import forecast_baseline_ensemble
from src.evaluation.net_position import baseline_predictions

AS_OF = pd.Timestamp("2026-08-06 22:00")
TARGETS = pd.date_range("2026-08-08", periods=24, freq="h")


def _actuals(days=40, level=2000.0):
    idx = pd.date_range(AS_OF - pd.Timedelta(days=days), AS_OF - pd.Timedelta(hours=1),
                        freq="h")
    return pd.Series(level + 500 * np.sin(np.arange(len(idx)) * 2 * np.pi / 24),
                     index=idx)


def test_matches_the_evaluation_baseline_ensemble_exactly():
    actuals = _actuals()
    served = forecast_baseline_ensemble(actuals, AS_OF, TARGETS)
    reference = baseline_predictions(actuals, AS_OF, TARGETS)[
        ["persistence", "climatology"]].mean(axis=1)
    assert np.allclose(served.to_numpy(), reference.to_numpy())


def test_uses_no_actual_at_or_after_as_of():
    """Serve-faithful: an actual published after the run must not move it."""
    actuals = _actuals()
    before = forecast_baseline_ensemble(actuals, AS_OF, TARGETS)
    future = pd.Series(
        [99999.0] * 24,
        index=pd.date_range(AS_OF, periods=24, freq="h"))
    after = forecast_baseline_ensemble(pd.concat([actuals, future]), AS_OF, TARGETS)
    assert np.allclose(before.to_numpy(), after.to_numpy())


def test_no_history_yields_nan_not_zero():
    """A country with nothing to average must produce no number. Net position
    of 0 MW is a real value (balanced border), so filling with it would publish
    a measurement never made."""
    out = forecast_baseline_ensemble(pd.Series(dtype=float), AS_OF, TARGETS)
    assert out.isna().all()
    assert not (out == 0.0).any()


def test_a_missing_component_does_not_silently_halve_the_other():
    """skipna=False: if only one of persistence/climatology exists, the mean of
    the two is unknown, not the one that happens to be present."""
    # One observation, 40 days back: outside the 28-day climatology window, so
    # persistence exists for that hour and climatology does not.
    lone = pd.Series([1500.0], index=[AS_OF - pd.Timedelta(days=40)])
    out = forecast_baseline_ensemble(lone, AS_OF, TARGETS)
    hour = (AS_OF - pd.Timedelta(days=40)).hour
    matched = out[out.index.hour == hour]
    assert matched.isna().all(), "climatology was empty; the mean must be NaN"


def test_reproduces_a_flat_series():
    flat = pd.Series(1234.0, index=pd.date_range(
        AS_OF - pd.Timedelta(days=30), AS_OF - pd.Timedelta(hours=1), freq="h"))
    out = forecast_baseline_ensemble(flat, AS_OF, TARGETS)
    assert np.allclose(out.to_numpy(), 1234.0)
