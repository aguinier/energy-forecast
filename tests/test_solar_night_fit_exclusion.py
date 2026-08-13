"""ABL-376: night rows the sun forbids leave the fit, and never the score.

The asymmetry is the point of the issue and is the thing these tests exist to
hold: we refuse to *train* on values the sun says are impossible, and we still
*score* against whatever the source reports. A filter that leaked into the gate
frame would let the challenger mark its own homework, and it would do so while
every number in the report still rendered and every other test still passed.
"""

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.solar_features import (  # noqa: E402
    IMPOSSIBLE_NIGHT_THRESHOLD_MW,
    exclude_impossible_night_rows,
    impossible_night_mask,
    night_mask,
)
from src.solar_geometry import is_night_hour  # noqa: E402

#: A midwinter day, so every country here has a wide unambiguous night band.
WINTER_DAY = pd.date_range("2026-01-15", periods=24, freq="h")
#: A midsummer day: the night band is narrow, which is where a whole-hour
#: predicate and a midpoint one disagree.
SUMMER_DAY = pd.date_range("2026-07-29", periods=24, freq="h")


def _frame(hours, actuals, extra=None):
    data = {"target_ts": hours, "actual": np.asarray(actuals, dtype=float)}
    if extra:
        data.update(extra)
    return pd.DataFrame(data)


def test_the_predicate_is_the_serving_clamps_not_a_second_copy():
    """`solar_geometry.is_night_hour` decides, or the fit and the clamp drift.

    ABL-337's whole reason for one geometry module is that a training filter and
    a serving clamp disagreeing about which hours are night makes both wrong.
    """
    for country in ("FR", "DE", "BE", "AT"):
        clamp_view = np.asarray(is_night_hour(country, WINTER_DAY), dtype=bool)
        # A huge actual everywhere, so the mask is decided by geometry alone.
        mask = impossible_night_mask(country, WINTER_DAY, np.full(24, 9999.0))
        np.testing.assert_array_equal(mask, clamp_view)


def test_excluded_rows_are_exactly_night_and_above_threshold():
    actual = np.zeros(24)
    actual[3] = 195.0      # deep night in January at FR
    actual[12] = 8000.0    # midday, huge, must never be touched
    frame = _frame(WINTER_DAY, actual)

    kept, audit = exclude_impossible_night_rows(frame, "FR")

    night = night_mask("FR", WINTER_DAY)
    assert night[3] and not night[12], "fixture assumes 03:00 night, 12:00 day in January"
    assert audit["excluded_rows"] == 1
    assert audit["max_excluded_mw"] == pytest.approx(195.0)
    assert 3 not in kept["target_ts"].dt.hour.tolist()
    assert 12 in kept["target_ts"].dt.hour.tolist()


def test_a_daylight_row_is_never_excluded_however_large():
    """The rule is about physical impossibility, not about outliers."""
    day_hours = ~night_mask("FR", WINTER_DAY)
    actual = np.where(day_hours, 1e6, 0.0)
    kept, audit = exclude_impossible_night_rows(_frame(WINTER_DAY, actual), "FR")
    assert audit["excluded_rows"] == 0
    assert len(kept) == 24


def test_threshold_is_strict_so_exactly_one_megawatt_survives():
    night_hour = WINTER_DAY[np.argmax(night_mask("FR", WINTER_DAY))]
    at_threshold = _frame([night_hour], [IMPOSSIBLE_NIGHT_THRESHOLD_MW])
    above = _frame([night_hour], [IMPOSSIBLE_NIGHT_THRESHOLD_MW + 0.01])

    assert exclude_impossible_night_rows(at_threshold, "FR")[1]["excluded_rows"] == 0
    assert exclude_impossible_night_rows(above, "FR")[1]["excluded_rows"] == 1


def test_missing_actuals_are_left_to_the_missingness_audit():
    """A NaN night actual is not "impossible", it is absent.

    Flagging it here would make this audit and `finite_training_rows` sum past
    the rows actually dropped, so the report would overstate the removal.
    """
    actual = np.zeros(24)
    actual[night_mask("FR", WINTER_DAY)] = np.nan
    kept, audit = exclude_impossible_night_rows(_frame(WINTER_DAY, actual), "FR")
    assert audit["excluded_rows"] == 0
    assert len(kept) == 24


def test_a_country_whose_data_is_clean_loses_nothing():
    """Stated over countries, not for FR. BE and AT carry no such rows at all."""
    actual = np.where(night_mask("BE", WINTER_DAY), 0.0, 3000.0)
    for country in ("BE", "AT"):
        _, audit = exclude_impossible_night_rows(_frame(WINTER_DAY, actual), country)
        assert audit["excluded_rows"] == 0
        assert audit["night_rows"] > 0, "the rule must have been evaluated, not skipped"


def test_audit_partitions_the_frame_and_counts_hours_under_rows():
    """Rows are per (target, vintage); hours are the distinct contaminated targets."""
    night_hour = SUMMER_DAY[np.argmax(night_mask("FR", SUMMER_DAY))]
    # One contaminated hour seen at three vintages.
    frame = _frame([night_hour] * 3, [195.0] * 3,
                   extra={"generated_at": pd.date_range("2026-07-27", periods=3, freq="h")})

    kept, audit = exclude_impossible_night_rows(frame, "FR")

    assert audit["excluded_rows"] == 3
    assert audit["excluded_targets"] == 1
    assert audit["retained_rows"] + audit["excluded_rows"] == len(frame)
    assert len(kept) == 0


def test_the_frame_is_not_mutated_and_other_columns_survive():
    actual = np.zeros(24)
    actual[3] = 195.0
    frame = _frame(WINTER_DAY, actual, extra={"hour": np.arange(24), "flag": list("abcdefghijklmnopqrstuvwx")})
    before = frame.copy(deep=True)

    kept, _ = exclude_impossible_night_rows(frame, "FR")

    pd.testing.assert_frame_equal(frame, before)
    assert list(kept.columns) == list(frame.columns)
    assert kept["flag"].tolist() == [c for i, c in enumerate("abcdefghijklmnopqrstuvwx") if i != 3]


def test_empty_frame_returns_a_readable_audit_rather_than_raising():
    kept, audit = exclude_impossible_night_rows(_frame([], []), "FR")
    assert len(kept) == 0
    assert audit["excluded_rows"] == 0
    assert audit["max_excluded_mw"] is None
    assert audit["threshold_mw"] == IMPOSSIBLE_NIGHT_THRESHOLD_MW


def test_length_mismatch_raises_rather_than_masking_by_position():
    with pytest.raises(ValueError, match="disagree in length"):
        impossible_night_mask("FR", WINTER_DAY, [0.0, 1.0])


# --------------------------------------------------------------------------
# The fit/score asymmetry, pinned structurally.
# --------------------------------------------------------------------------

HARNESS = Path(__file__).parent.parent / "scripts" / "evaluate_solar_retrain.py"


def _calls_to(tree, function_name):
    return [node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == function_name]


def test_the_exclusion_is_applied_to_the_fit_frame_and_to_nothing_else():
    """AST, not text: the filter's first argument must be the fit frame.

    A run that passed `gate_finite` here would still fit, still score, still
    render every number and still pass every other test in this file -- while
    quietly deleting the rows the challenger is supposed to be held to account
    on. There is no output that shows it, so it is pinned at the call site.
    """
    tree = ast.parse(HARNESS.read_text(encoding="utf-8"))
    calls = _calls_to(tree, "exclude_impossible_night_rows")

    assert len(calls) == 1, "expected exactly one exclusion site in the harness"
    first_arg = calls[0].args[0]
    assert isinstance(first_arg, ast.Name) and first_arg.id == "fit", (
        "the exclusion must be applied to the fit frame; applying it to the gate "
        "frame would let the challenger delete its own scoring rows"
    )


def test_registration_tables_cover_every_scope_and_pin_abl253_unchanged():
    """ABL-253 stays reproducible: same title, and the rule off."""
    sys.path.insert(0, str(HARNESS.parent))
    import importlib.util

    spec = importlib.util.spec_from_file_location("_solar_harness", HARNESS)
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)

    scopes = set(harness.SCOPES)
    assert set(harness.FIT_RULES) == scopes
    assert set(harness.SCOPE_TITLES) == scopes
    for scope in scopes:
        assert "exclude_impossible_night" in harness.FIT_RULES[scope]

    # The dispositioned read must keep both its rule and its heading.
    assert harness.FIT_RULES["abl253"]["exclude_impossible_night"] is False
    assert harness.SCOPE_TITLES["abl253"] == "ABL-253 — Serve-faithful solar retrain gate"
    assert harness.FIT_RULES["abl376"]["exclude_impossible_night"] is True
    # The A/B is on the rule alone.
    assert harness.SCOPES["abl376"] == harness.SCOPES["abl253"]
    assert harness.GATE_BASIS["abl376"] == harness.GATE_BASIS["abl253"]
