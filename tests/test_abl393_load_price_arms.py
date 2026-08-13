"""ABL-393: the load/price extension of the ABL-386 harness, and its two claims.

`tests/test_abl386_holiday_arms.py` pins the solar half. This file pins what
ABL-393 added, and both of its claims are the kind that stay true-looking after
they stop being true:

    1. `control_noholiday` is EXACTLY the serving feature list on load and on
       price. There is no geometry on these types, so unlike solar the arm is not
       merely close to serving — the contrast IS "what a retrain produces" against
       "what is served today", and every sentence in the evidence pack rests on
       that.

    2. The holiday-subset predicate is defined once. The pre-fit density probe
       reports how many holiday-affected rows a candidate window holds and the
       A/B scores the arms over them; if those two drift apart, a window is
       registered under one definition and read under another, and nothing in
       either output would show it.

Both are checked against `tests/feature_list_manifest.json` rather than against
`models/`, which is gitignored and absent in CI.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402
from scripts.abl338_solar_holdout import (  # noqa: E402
    AGGREGATE_TYPES,
    FITTABLE_TYPES,
    SOLAR_ONLY_ARMS,
    _legacy_feature_columns,
)
from src.features import (  # noqa: E402
    HOLIDAY_FEATURES,
    HOLIDAY_SUBSETS,
    get_feature_columns,
    holiday_subset_masks,
)

MANIFEST = json.loads(
    (Path(__file__).parent / "feature_list_manifest.json").read_text(encoding="utf-8")
)


@pytest.mark.parametrize("forecast_type", AGGREGATE_TYPES)
def test_control_noholiday_reconstructs_the_served_list_exactly(forecast_type):
    """Order included. `Forecaster.to_vector` rebuilds a serving row from
    `feature_columns` in order, so a same-set-different-order list is a different
    model input rather than a cosmetic difference."""
    gap = MANIFEST["serving_gap"][forecast_type]
    served = [
        c
        for c in get_feature_columns(forecast_type)
        if c not in gap["declared_but_missing_from_every_serving_artifact"]
    ]
    assert _legacy_feature_columns(forecast_type, include_holidays=False) == served
    assert len(served) == gap["n_served"]


@pytest.mark.parametrize("forecast_type", AGGREGATE_TYPES)
def test_the_gap_on_these_types_is_the_four_holiday_names_and_nothing_else(forecast_type):
    """Solar is distinctive in having a second gap on top (ABL-338's geometry).
    Load and price are not, which is why one factor is enough here and ABL-386's
    2x2 collapses to a single contrast. If a fifth name ever goes missing, the
    single contrast stops isolating the question and this goes red."""
    gap = MANIFEST["serving_gap"][forecast_type]
    assert set(gap["declared_but_missing_from_every_serving_artifact"]) == set(HOLIDAY_FEATURES)
    dropped = set(_legacy_feature_columns(forecast_type, include_holidays=True)) - set(
        _legacy_feature_columns(forecast_type, include_holidays=False)
    )
    assert dropped == set(HOLIDAY_FEATURES)


def test_load_and_price_are_fittable_and_carry_no_solar_only_arm():
    """`--type` used to be `config.RENEWABLE_TYPES` alone. The two solar-only
    refusals must keep applying to the new types: a `geometry` arm on load would
    fit the identical list as `control` and report a spurious null."""
    for forecast_type in AGGREGATE_TYPES:
        assert forecast_type in FITTABLE_TYPES
        assert forecast_type not in config.RENEWABLE_TYPES
    assert set(config.RENEWABLE_TYPES) <= set(FITTABLE_TYPES)
    # net_position is Chronos-2's, is not fitted by `Forecaster`, and has no
    # artifact under models/<CC>/net_position/ for the harness to read.
    assert "net_position" not in FITTABLE_TYPES
    assert "control_noholiday" not in SOLAR_ONLY_ARMS


def _frame(dates, holiday, bridge, to_h, from_h):
    return pd.DataFrame({
        "timestamp_utc": pd.to_datetime(dates),
        "is_holiday": holiday,
        "is_bridge_day": bridge,
        "days_to_holiday": to_h,
        "days_from_holiday": from_h,
    })


def test_holiday_subsets_partition_the_frame():
    """`holiday_affected` and `ordinary` must cover every row exactly once, or the
    two subset MAEs in the report are computed on overlapping or incomplete sets
    and cannot be read against each other."""
    frame = _frame(
        ["2025-12-24", "2025-12-25", "2025-12-26", "2026-03-10", "2026-03-11"],
        holiday=[0, 1, 0, 0, 0],
        bridge=[0, 0, 1, 0, 0],
        to_h=[1, 0, 7, 5, 4],
        from_h=[7, 0, 1, 4, 5],
    )
    masks = holiday_subset_masks(frame)
    assert set(masks) == set(HOLIDAY_SUBSETS)
    assert (masks["holiday_affected"] | masks["ordinary"]).all()
    assert not (masks["holiday_affected"] & masks["ordinary"]).any()
    # A holiday is always holiday-affected; the converse is the point of the set.
    assert (masks["holiday"] <= masks["holiday_affected"]).all()
    assert list(masks["holiday_affected"]) == [True, True, True, False, False]


def test_holiday_subsets_are_empty_rather_than_raising_without_the_columns():
    """`create_all_features` only builds the four columns when given a country. A
    caller that did not ask for the subsets should not have to care."""
    assert holiday_subset_masks(pd.DataFrame({"timestamp_utc": []})) == {}


def test_the_subset_predicate_has_exactly_one_definition():
    """The pre-fit density probe reports holiday-affected rows and the A/B scores
    them. Two copies of this predicate would let a window be registered under one
    and read under another, with nothing in either output showing it.

    Checked textually because the failure is a *second* implementation appearing,
    which no import-level assertion can see.
    """
    root = Path(__file__).parent.parent
    callers = [
        root / "scripts" / "abl338_solar_holdout.py",
        root / "scripts" / "abl386_holiday_density_probe.py",
    ]
    for path in callers:
        text = path.read_text(encoding="utf-8")
        assert "holiday_subset_masks" in text, f"{path.name} should use the shared predicate"
        assert "def holiday_subset_masks" not in text, f"{path.name} defines a second copy"
        assert "is_bridge_day\"] == 1" not in text, f"{path.name} inlines the predicate"
