"""ABL-338: the solar-geometry training features, and the fit-side constraint.

The load-bearing property here is that a training row and the serving row for
the same (country, hour) carry **identical** geometry values. ABL-337's handover
put it plainly: "the serving clamp and your training feature must be the same
number, or the feature says 'sun is up' at an hour the clamp zeroes." These
tests check that against the real serve path rather than against a second copy
of the arithmetic.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import create_all_features, get_feature_columns  # noqa: E402
from src.forecaster import NONNEG_OBJECTIVES, Forecaster  # noqa: E402
from src.solar_features import (  # noqa: E402
    SOLAR_GEOMETRY_FEATURES,
    night_mask,
    solar_geometry_frame,
)
from src.solar_geometry import NIGHT_ELEVATION_THRESHOLD_DEG, is_night_hour  # noqa: E402
from src.wind_features import _solar_geometry_features, FeatureRequest, to_vector  # noqa: E402


AUGUST_DAY = pd.date_range("2026-08-14 00:00", periods=24, freq="h")


def test_is_night_is_bit_identical_to_the_clamps_own_predicate():
    """The feature and the serving clamp must agree on every hour, exactly.

    Not "agree to a tolerance": `is_night` is the same call the clamp makes, so
    a disagreement anywhere means the two have been allowed to drift apart.
    """
    for country in ("AT", "BE", "DE", "FR"):
        frame = solar_geometry_frame(country, AUGUST_DAY)
        clamp_view = np.asarray(is_night_hour(country, AUGUST_DAY), dtype=int)
        assert np.array_equal(frame["is_night"].to_numpy(), clamp_view)


def test_training_and_serving_produce_the_same_geometry_for_the_same_hour():
    """The anti-skew property, checked through the real serve-path builder.

    `_solar_geometry_features` is what `RenewableFeatureBuilder.row` calls; the
    frame is what `create_all_features` embeds. Same country, same hour, so the
    two must be equal to the bit.
    """
    country = "DE"
    training_frame = solar_geometry_frame(country, AUGUST_DAY)
    for position, target in enumerate(AUGUST_DAY):
        request = FeatureRequest.build(
            country, "solar", target,
            observation_as_of=pd.Timestamp("2026-08-12 06:00"),
        )
        served = _solar_geometry_features(request)
        for name in SOLAR_GEOMETRY_FEATURES:
            assert served[name].value == float(training_frame[name].iloc[position]), (
                f"{name} differs between training and serving at {target}"
            )


def test_geometry_features_are_only_added_for_solar():
    """A wind or load artifact must not silently acquire two new columns."""
    solar_columns = get_feature_columns("solar")
    for name in SOLAR_GEOMETRY_FEATURES:
        assert name in solar_columns
    for forecast_type in ("load", "price", "wind_onshore", "wind_offshore", "renewable"):
        for name in SOLAR_GEOMETRY_FEATURES:
            assert name not in get_feature_columns(forecast_type)


def test_serving_omits_geometry_for_a_country_with_no_representative_point():
    """A legacy artifact for an unmapped country must still be servable.

    Contributing nothing lets a pre-ABL-338 artifact (which names none of these
    columns) serve exactly as before, while `to_vector` still refuses for an
    artifact that does name them — the error belongs where the model expects the
    feature, not where the country lookup fails.
    """
    request = FeatureRequest.build(
        "XX", "solar", pd.Timestamp("2026-08-14 12:00"),
        observation_as_of=pd.Timestamp("2026-08-12 06:00"),
    )
    assert _solar_geometry_features(request) == {}

    with pytest.raises(KeyError):
        to_vector({}, ["sun_elevation_deg"])


def test_night_hours_are_flagged_and_midday_is_not():
    """Sanity, on a country and a day whose answer is not in dispute."""
    frame = solar_geometry_frame("DE", AUGUST_DAY)
    assert frame["is_night"].iloc[0] == 1          # 00:00 UTC
    assert frame["is_night"].iloc[12] == 0         # 12:00 UTC
    assert frame["sun_elevation_deg"].iloc[12] > 40
    assert frame["sun_elevation_deg"].iloc[0] < NIGHT_ELEVATION_THRESHOLD_DEG


def test_elevation_is_the_hour_midpoint_not_the_hour_start():
    """`sun_elevation_deg` describes the hour, so it is offset by 30 minutes.

    This is what keeps it from being a monotone restatement of `is_night` (which
    is a threshold on the hour's *maximum*), and therefore what makes the second
    feature carry information at the shoulder hours rather than just repeating
    the first.
    """
    from src.solar_geometry import sun_elevation_deg

    frame = solar_geometry_frame("DE", AUGUST_DAY)
    at_start = np.asarray(sun_elevation_deg("DE", AUGUST_DAY))
    at_midpoint = np.asarray(sun_elevation_deg("DE", AUGUST_DAY + pd.Timedelta(minutes=30)))
    assert np.allclose(frame["sun_elevation_deg"].to_numpy(), at_midpoint)
    assert not np.allclose(frame["sun_elevation_deg"].to_numpy(), at_start)


def test_night_mask_matches_the_frames_is_night_column():
    """One definition of night for the fit, the score and the clamp."""
    for country in ("AT", "FR"):
        frame = solar_geometry_frame(country, AUGUST_DAY)
        assert np.array_equal(
            night_mask(country, AUGUST_DAY), frame["is_night"].to_numpy().astype(bool)
        )


def test_solar_features_reach_create_all_features():
    country = "FR"
    hours = pd.date_range("2026-01-01", periods=24 * 40, freq="h")
    rng = np.random.default_rng(0)
    raw = pd.DataFrame({
        "timestamp_utc": hours,
        "target_value": np.clip(rng.normal(500, 200, len(hours)), 0, None),
        "shortwave_radiation_wm2": rng.uniform(0, 800, len(hours)),
        "direct_radiation_wm2": rng.uniform(0, 600, len(hours)),
        "diffuse_radiation_wm2": rng.uniform(0, 200, len(hours)),
        "temperature_2m_k": rng.uniform(270, 300, len(hours)),
    })
    featured = create_all_features(raw, "solar", country_code=country)
    for name in SOLAR_GEOMETRY_FEATURES:
        assert name in featured.columns
    expected = solar_geometry_frame(country, featured["timestamp_utc"])
    assert np.allclose(
        featured["sun_elevation_deg"].to_numpy(), expected["sun_elevation_deg"].to_numpy()
    )


def test_create_all_features_refuses_solar_without_a_country():
    """Silently training a solar model without the daylight variable is the
    exact failure ABL-338 exists to remove, so it raises rather than warns."""
    raw = pd.DataFrame({
        "timestamp_utc": pd.date_range("2026-01-01", periods=24 * 40, freq="h"),
        "target_value": np.arange(24 * 40, dtype=float),
    })
    with pytest.raises(ValueError, match="country_code"):
        create_all_features(raw, "solar")


# ---------------------------------------------------------------------------
# The fit-side non-negativity constraint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("objective", sorted(NONNEG_OBJECTIVES))
def test_nonneg_objective_sets_a_log_link_loss_per_algorithm(objective):
    for algorithm in ("xgboost", "lightgbm", "catboost"):
        forecaster = Forecaster("DE", "solar", algorithm=algorithm, nonneg_objective=objective)
        loss = forecaster.hyperparams.get("objective") or forecaster.hyperparams.get("loss_function")
        assert loss == next(iter(NONNEG_OBJECTIVES[objective][algorithm].values()))
        # CatBoost would ignore a stray `objective` beside `loss_function`, which
        # reads as if it applied. Exactly one of the two names may survive.
        assert ("objective" in forecaster.hyperparams) != ("loss_function" in forecaster.hyperparams)


def test_a_log_link_fit_cannot_emit_a_negative_prediction():
    """The whole point of choosing a link over a post-hoc clip."""
    rng = np.random.default_rng(1)
    n = 2000
    elevation = rng.uniform(-60, 60, n)
    X = pd.DataFrame({
        "sun_elevation_deg": elevation,
        "is_night": (elevation < NIGHT_ELEVATION_THRESHOLD_DEG).astype(int),
        "shortwave_radiation_wm2": np.clip(elevation, 0, None) * 12,
    })
    y = pd.Series(np.clip(elevation, 0, None) * 300 + rng.normal(0, 50, n)).clip(lower=0)

    forecaster = Forecaster("DE", "solar", algorithm="xgboost", nonneg_objective="tweedie")
    forecaster.feature_columns = list(X.columns)
    # The default hyperparameters carry `early_stopping_rounds`, so the real fit
    # path always has a validation set; give it one rather than special-casing.
    split = int(n * 0.8)
    forecaster._train_simple(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])

    predictions = forecaster.model.predict(X)
    assert (predictions >= 0).all()
    # Including where the incumbents went most wrong: fully dark hours.
    assert (predictions[X["is_night"] == 1] >= 0).all()


def test_nonneg_fit_refuses_a_negative_target_and_names_what_it_found():
    forecaster = Forecaster("BE", "solar", nonneg_objective="tweedie")
    with pytest.raises(ValueError, match="below zero"):
        forecaster._assert_nonneg_target(pd.Series([1.0, 2.0, -3.0]))


def test_no_nonneg_objective_leaves_the_default_loss_untouched():
    """An artifact that predates the constraint must fit exactly as before."""
    forecaster = Forecaster("DE", "solar", algorithm="xgboost")
    assert forecaster.hyperparams["objective"] == "reg:squarederror"
    assert forecaster.nonneg_objective is None
    forecaster._assert_nonneg_target(pd.Series([-5.0, 1.0]))  # no constraint, no refusal


def test_unknown_nonneg_objective_is_refused():
    with pytest.raises(ValueError, match="Unknown nonneg_objective"):
        Forecaster("DE", "solar", nonneg_objective="logistic")
