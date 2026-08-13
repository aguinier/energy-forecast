"""Protocol checks for the pre-registered ABL-253 solar gate."""

from src.evaluation.solar_retrain import ALGORITHM, FEATURE_COLUMNS


def test_solar_gate_algorithm_is_frozen():
    # ABL-379: the pair set moved out of this module. `COUNTRIES` was iterated
    # directly by the harness, so reading a gate for any other country meant
    # editing this file; the registered pair set is now `SCOPES` in
    # `scripts/evaluate_solar_retrain.py`, where the gate basis, cell count and
    # output paths that have to move with it also live.
    # `tests/test_solar_gate_scope_registration.py` pins that the default scope
    # still reproduces exactly the ("BE", "DE", "FR") set this asserted.
    assert ALGORITHM == "catboost"


def test_solar_feature_vector_matches_abl191_artifact_shape():
    assert len(FEATURE_COLUMNS) == 25
    assert FEATURE_COLUMNS[-4:] == (
        "shortwave_radiation_wm2", "direct_radiation_wm2",
        "diffuse_radiation_wm2", "temperature_c",
    )
