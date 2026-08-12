"""Protocol checks for the pre-registered ABL-253 solar gate."""

from src.evaluation.solar_retrain import ALGORITHM, COUNTRIES, FEATURE_COLUMNS


def test_solar_gate_scope_is_frozen_to_currently_served_pairs():
    assert COUNTRIES == ("BE", "DE", "FR")
    assert ALGORITHM == "catboost"


def test_solar_feature_vector_matches_abl191_artifact_shape():
    assert len(FEATURE_COLUMNS) == 25
    assert FEATURE_COLUMNS[-4:] == (
        "shortwave_radiation_wm2", "direct_radiation_wm2",
        "diffuse_radiation_wm2", "temperature_c",
    )
