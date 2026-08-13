"""Protocol checks for the pre-registered ABL-253 solar gate."""

from src.evaluation.solar_retrain import (
    ALGORITHM, COUNTRIES, FEATURE_COLUMNS, SOLAR_GEOMETRY_FEATURES,
)


def test_solar_gate_scope_is_frozen_to_currently_served_pairs():
    assert COUNTRIES == ("BE", "DE", "FR")
    assert ALGORITHM == "catboost"


def test_solar_feature_vector_is_the_abl191_shape_plus_abl338_geometry():
    """25 (ABL-191's artifact shape) + 2 (ABL-338's adopted half) = 27.

    The 25 below are what every gate fit up to and including ABL-381 ran on.
    ABL-395 appends the geometry pair: `RenewableFeatureBuilder` had emitted both
    for solar since ABL-338, and only this list never asked for them, so the
    artifacts were built two features short of an ABL-338-current fit while
    declaring nothing was missing.

    Appended last, and the weather block is asserted at its own offset rather
    than at `[-4:]`, so the radiation columns keep a fixed position in the
    vector — the artifact's `feature_columns` *is* the column order `to_vector`
    rebuilds a serving row from.
    """
    assert len(FEATURE_COLUMNS) == 27
    assert FEATURE_COLUMNS[21:25] == (
        "shortwave_radiation_wm2", "direct_radiation_wm2",
        "diffuse_radiation_wm2", "temperature_c",
    )
    assert FEATURE_COLUMNS[25:] == SOLAR_GEOMETRY_FEATURES
    assert SOLAR_GEOMETRY_FEATURES == ("sun_elevation_deg", "is_night")


def test_no_duplicate_feature_names():
    """A splatted tuple is the one way a name could arrive twice, and a duplicate
    would pass every length check while handing the model the same column twice."""
    assert len(set(FEATURE_COLUMNS)) == len(FEATURE_COLUMNS)
