"""ABL-394: make the training feature list a decision instead of an accident.

`get_feature_columns()` is evaluated at fit time, and `Forecaster.load` reads
`feature_columns` off the artifact. So a serving model keeps its own list
forever, and a change to the list reaches production only at the next retrain —
silently, whenever that happens to be. That is how the four holiday names
(`is_holiday`, `days_to_holiday`, `days_from_holiday`, `is_bridge_day`) came to
be absent from all 66 serving artifacts that carry a list at all, and how
ABL-375 and ABL-386 both ended up spending an evidence pack rediscovering it.

**What these tests prove is a conditional** (`_effective_columns` below is the
same intersection `Forecaster.train` computes):

    call `create_all_features(df, forecast_type)` with no `country_code` and
    `create_holiday_features` does not run, so the four holiday names are
    declared but never produced — and the `if col in df.columns` narrowing drops
    them without a word.

Dropping exactly those four reproduces the served list length for every one of
the eight forecast types with an artifact: 23/23/26/25/27/25/24/24. Nothing else
is needed to explain the 66-artifact gap. It is one plumbing gap, not eight
independent drifts.

**The conditional is not a history.** An earlier version of this docstring
asserted the antecedent too — that "before ABL-338 (5cf2296), the training sites
called `create_all_features(df, forecast_type)` with no `country_code`". ABL-407
refuted that and it is removed: `git show 5cf2296 --stat -- scripts/train.py` is
empty, and at `5cf2296^` the training site already read
`create_all_features(df, forecast_type, country_code=country_code)`. The
pre-ABL-338 site that *did* omit it built the validation frame in
`evaluate_against_baselines`, which never writes an artifact's `feature_columns`
— that is **ABL-397**, and keeping the two apart is what this paragraph is for.
Where the 66 artifacts actually came from is measured in
`reports/abl_407_holiday_gap_provenance.md`. The tests below are unaffected:
they assert the conditional and never the antecedent.

So there are two properties worth holding, and they are different:

1. **The declared list is reviewed.** `feature_list_manifest.json` is the frozen
   copy; changing `get_feature_columns()` without changing it goes red, which
   puts the diff in front of a reviewer. This is the "something decides" part.
2. **A declared name is actually produced.** A name the feature builder cannot
   produce is not a feature — it is a silent no-op that shrinks the fit. Nobody
   had this test until ABL-394. Note it would have passed on the training path
   at `5cf2296^` too (ABL-407 ran it there): what it guards is the *next* fit,
   not a historical regression.

What these tests deliberately do *not* assert is that the declared list equals
what serving artifacts carry. It does not, on all 8 types, and cannot be made to
without a retrain nobody has approved. The gap is recorded in the manifest's
`serving_gap` block instead, so it is explicit and dated rather than silent.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402
from src.features import (  # noqa: E402
    create_all_features,
    get_feature_columns,
    select_feature_columns,
)

MANIFEST_PATH = Path(__file__).parent / "feature_list_manifest.json"
MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
DECLARED = MANIFEST["declared"]
SERVING_GAP = MANIFEST["serving_gap"]

#: Every forecast type the repo knows how to build a weather block for. Driving
#: the parametrisation off config rather than off a hand-written list is what
#: makes a *new* type fail here instead of slipping in unrecorded.
FORECAST_TYPES = sorted(config.WEATHER_FEATURES)

#: Every raw weather column any type asks for, so the synthetic frame below is a
#: best case: whatever is missing from the fit is missing because the *feature
#: builder* did not produce it, not because the fixture forgot a column.
RAW_WEATHER_COLUMNS = sorted({c for cols in config.WEATHER_FEATURES.values() for c in cols})


def _frame(days: int = 40) -> pd.DataFrame:
    """A training frame carrying every raw input `create_all_features` can use."""
    hours = pd.date_range("2025-01-01", periods=24 * days, freq="h")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "timestamp_utc": hours,
        "target_value": rng.normal(1000.0, 50.0, len(hours)),
    })
    for column in RAW_WEATHER_COLUMNS:
        centre = 280.0 if column == "temperature_2m_k" else 100.0
        frame[column] = rng.normal(centre, 5.0, len(hours))
    return frame


def _effective_columns(forecast_type: str, country_code: str = "DE") -> list:
    """What a fit of this type would actually train on — the same intersection
    `Forecaster.train` takes, through the same helper it now calls."""
    featured = create_all_features(_frame(), forecast_type, country_code=country_code)
    return select_feature_columns(forecast_type, featured.columns)


# ---------------------------------------------------------------------------
# 1. The declared list is reviewed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("forecast_type", FORECAST_TYPES)
def test_declared_list_still_matches_the_reviewed_manifest(forecast_type):
    """Changing `get_feature_columns()` must change this file in the same commit.

    That is the entire point of ABL-394. If this fails, do not regenerate the
    manifest to get green — decide, in the commit message, that the new list is
    what every country should train on at its next fit, and say what evaluated
    it. Every artifact retrained after that commit serves the new list.
    """
    assert get_feature_columns(forecast_type) == DECLARED[forecast_type]


def test_manifest_covers_every_type_the_repo_can_fit():
    """A new forecast type arrives with a recorded feature list or not at all."""
    assert sorted(DECLARED) == FORECAST_TYPES


@pytest.mark.parametrize("forecast_type", FORECAST_TYPES)
def test_declared_list_has_no_duplicates(forecast_type):
    """A duplicated name would pass the length checks and hand the model the
    same column twice."""
    declared = get_feature_columns(forecast_type)
    assert len(declared) == len(set(declared))


# ---------------------------------------------------------------------------
# 2. A declared name is actually produced
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("forecast_type", FORECAST_TYPES)
def test_every_declared_feature_is_actually_produced(forecast_type):
    """A name the builder cannot produce is a silent no-op, not a feature.

    This is the assertion that was red from the initial commit until ABL-338 and
    that nobody was running: the four holiday names sat in every type's declared
    list while `create_all_features` was called without a `country_code`, so
    they were dropped at the fit site with no log line, no error, and no trace in
    the artifact beyond a list four names shorter than expected.
    """
    effective = _effective_columns(forecast_type)
    missing = [c for c in get_feature_columns(forecast_type) if c not in effective]
    assert not missing, (
        f"{forecast_type}: declared but never produced by create_all_features, so "
        f"silently dropped at fit time: {missing}"
    )


def test_holiday_features_need_a_country_and_are_dropped_without_one():
    """The exact mechanism behind the 66-artifact gap, pinned.

    `load_training_data` does not carry a `country_code` column, so a caller that
    omits the argument gets a frame with no holiday columns. Before ABL-338 that
    was every training site. Solar is excluded because it now refuses outright
    rather than fitting without geometry, which is that issue's own guard.
    """
    holiday = [c for c in DECLARED["load"] if "holiday" in c or c == "is_bridge_day"]
    assert len(holiday) == 4

    without_country = create_all_features(_frame(), "load")
    for name in holiday:
        assert name not in without_country.columns
    assert select_feature_columns("load", without_country.columns) == [
        c for c in DECLARED["load"] if c not in holiday
    ]

    with_country = create_all_features(_frame(), "load", country_code="DE")
    for name in holiday:
        assert name in with_country.columns


@pytest.mark.parametrize("forecast_type", sorted(SERVING_GAP))
def test_dropping_the_recorded_gap_reproduces_the_served_list_length(forecast_type):
    """The measurement that says this is one plumbing gap, not eight drifts.

    `serving_gap` is ABL-386's read of the live models directory on 2026-08-13.
    If today's declared list minus the recorded gap no longer has the length the
    artifacts carry, then something *else* has changed the list too, and the
    "one cause" story in this module's docstring has stopped being true.
    """
    gap = SERVING_GAP[forecast_type]
    declared = get_feature_columns(forecast_type)
    missing = gap["declared_but_missing_from_every_serving_artifact"]

    assert set(missing) <= set(declared), (
        f"{forecast_type}: the recorded gap names features that are no longer "
        f"declared: {sorted(set(missing) - set(declared))}"
    )
    assert gap["served_but_not_declared"] == []
    assert len(declared) - len(missing) == gap["n_served"]


# ---------------------------------------------------------------------------
# 3. The narrowing says what it drops
# ---------------------------------------------------------------------------


def test_select_feature_columns_names_every_column_it_drops(caplog):
    declared = get_feature_columns("load")
    available = [c for c in declared if c != "temperature_c"]

    with caplog.at_level(logging.WARNING, logger="energy_forecast"):
        selected = select_feature_columns("load", available, "DE train")

    assert selected == available
    assert "temperature_c" in caplog.text
    assert "DE train" in caplog.text
    assert f"{len(available)} of {len(declared)}" in caplog.text


def test_select_feature_columns_is_silent_when_nothing_is_dropped(caplog):
    declared = get_feature_columns("price")
    with caplog.at_level(logging.WARNING, logger="energy_forecast"):
        selected = select_feature_columns("price", declared + ["an_unused_column"])
    assert selected == declared
    assert caplog.text == ""


def test_select_feature_columns_returns_declared_order_not_frame_order():
    """The artifact's `feature_columns` is the column order the model was fitted
    on, and `to_vector` rebuilds a serving row from it. Frame order must not leak
    into it."""
    declared = get_feature_columns("wind_onshore")
    selected = select_feature_columns("wind_onshore", list(reversed(declared)))
    assert selected == declared


# ---------------------------------------------------------------------------
# 4. Characterisation: what the list builder does with a name it does not know
# ---------------------------------------------------------------------------


def test_an_unrecognised_forecast_type_returns_a_weatherless_list():
    """Not an endorsement — a record. `get_feature_columns` has no notion of a
    valid forecast type: a typo returns a plausible list with the weather block
    silently empty, and the fit that used it would look ordinary. Changing this
    to raise touches every caller, so it is recorded here and left to a decision
    rather than fixed in passing.
    """
    typo = get_feature_columns("solr")
    assert "shortwave_radiation_wm2" not in typo
    assert typo == [c for c in get_feature_columns("biomass") if c != "temperature_2m_k"]
