"""ABL-386: the holdout harness's arm composition, tied to the reviewed manifest.

ABL-394 guards the *declared* list — change `get_feature_columns()` without
changing `feature_list_manifest.json` and it goes red. What it deliberately does
not guard is this harness's arms, and those carry a claim of their own that is
easy to break and expensive to notice:

    `control_noholiday` is the only arm in the ABL-338/375/386 lineage whose
    feature list *is* the serving solar feature set.

Every committed run before this issue used 29 or 31 names, so no result from this
script could be phrased as "beats the serving artifact" — only as beating the
serving *configuration* on this repo's current list. That distinction is the
finding, and it survives only while `control_noholiday` keeps reconstructing the
served list exactly. If someone adds a fifth non-geometry, non-holiday name to
the solar list, ABL-394 goes red (good) and so should this (also good, and for a
different reason: the arm stops being a stand-in for serving).

The reconstruction is checked against the manifest's `serving_gap` block rather
than against `models/`, which is gitignored and absent in CI. `serving_gap` is
the dated read of the four live solar artifacts; ABL-394's own
`test_dropping_the_recorded_gap_reproduces_the_served_list_length` is what keeps
it honest against the declared list.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.abl338_solar_holdout import (  # noqa: E402
    ARM_HOLIDAYS,
    ARM_SPECS,
    SOLAR_ONLY_ARMS,
    _legacy_feature_columns,
)
from src.features import HOLIDAY_FEATURES, get_feature_columns  # noqa: E402
from src.solar_features import SOLAR_GEOMETRY_FEATURES  # noqa: E402

MANIFEST = json.loads(
    (Path(__file__).parent / "feature_list_manifest.json").read_text(encoding="utf-8")
)
SOLAR_GAP = MANIFEST["serving_gap"]["solar"]


def _arm_columns(arm: str) -> list:
    """Rebuild one arm's feature list the way the run loop does — geometry read
    off `ARM_SPECS` rather than inferred from the arm name, so a rename of an arm
    cannot quietly change what this test is measuring."""
    use_geometry = ARM_SPECS[arm][0]
    columns = _legacy_feature_columns("solar", include_holidays=ARM_HOLIDAYS.get(arm, True))
    if use_geometry:
        columns = columns + list(SOLAR_GEOMETRY_FEATURES)
    return columns


#: The 2x2 the issue needs, and the counts the harness docstring publishes.
EXPECTED_COUNTS = {
    "control_noholiday": 25,
    "control": 29,
    "geometry_noholiday": 27,
    "geometry": 31,
}


@pytest.mark.parametrize("arm,expected", sorted(EXPECTED_COUNTS.items()))
def test_arm_feature_counts_are_what_the_evidence_pack_published(arm, expected):
    """The verdict table names these four numbers. A wrong one misleads the next
    reader about what "serving" means, which is the whole failure ABL-386 is
    about."""
    assert len(_arm_columns(arm)) == expected


def test_control_noholiday_reconstructs_the_served_solar_list_exactly():
    """The load-bearing claim: order included, not just length.

    `Forecaster.to_vector` rebuilds a serving row from `feature_columns` in
    order, so a same-set-different-order list is a different model input, not a
    cosmetic difference.
    """
    served = [
        c
        for c in get_feature_columns("solar")
        if c not in SOLAR_GAP["declared_but_missing_from_every_serving_artifact"]
    ]
    assert _arm_columns("control_noholiday") == served
    assert len(served) == SOLAR_GAP["n_served"]


def test_the_holiday_arms_differ_by_exactly_the_four_holiday_names():
    """Neither more nor fewer. `include_holidays=False` must not also move
    something else, or the arm stops isolating the question."""
    dropped = set(_arm_columns("control")) - set(_arm_columns("control_noholiday"))
    assert dropped == set(HOLIDAY_FEATURES)
    assert len(HOLIDAY_FEATURES) == 4


def test_holiday_arms_are_not_solar_only_unlike_the_geometry_arms():
    """`SOLAR_ONLY_ARMS` exists because a geometry arm on a non-solar type is
    byte-identical to `control` — an arm that measured nothing while reporting
    "no effect". The holiday arms do not have that failure mode:
    `create_holiday_features` runs for every forecast type, so the four names are
    in the frame whatever the type and dropping them is always a real change.
    Pinned so nobody "fixes" the asymmetry by adding it to the refusal list.
    """
    assert "geometry_noholiday" in SOLAR_ONLY_ARMS
    assert "control_noholiday" not in SOLAR_ONLY_ARMS

    for forecast_type in ("load", "wind_onshore"):
        with_holidays = _legacy_feature_columns(forecast_type, include_holidays=True)
        without = _legacy_feature_columns(forecast_type, include_holidays=False)
        assert set(with_holidays) - set(without) == set(HOLIDAY_FEATURES)


def test_geometry_subtraction_is_a_no_op_off_solar():
    """The other half of the generalisation ABL-385 made: `_legacy_feature_columns`
    subtracts the geometry pair unconditionally, which is only meaningful for
    solar because only solar declares it."""
    for forecast_type in ("load", "wind_onshore"):
        assert _legacy_feature_columns(forecast_type) == get_feature_columns(forecast_type)
        assert not set(SOLAR_GEOMETRY_FEATURES) & set(get_feature_columns(forecast_type))
