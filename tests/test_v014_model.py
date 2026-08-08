"""V014 refuses rather than improvises, and its held-out split is real (ABL-69).

The failure this program keeps hitting is a confident wrong number, not a crash.
A tabular model is unusually good at producing one: XGBoost returns a value for
any row you hand it, including a row where every informative feature is NaN — it
just composes its default split directions and answers. These tests pin the
three refusals that stop that, plus the two places the backtest's honesty is
decided (which target days the fit saw, and how validation is split).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.challengers.registry import CHAMPION_MODEL_NAME, spec_for
from src.challengers.v014 import (ANCHOR_FEATURES, MIN_ANCHOR_FEATURES,
                                  ModelArtifactError, V014Model, _base_score,
                                  _split_by_run_day, backtest_target_days,
                                  load_model, model_path, run_days_for_span,
                                  save_model)


class _ConstantBooster:
    """Stands in for a fitted XGBRegressor: always answers, never refuses.

    That is the point — the refusal has to live in `predict_frame`, because the
    booster itself will happily return a number for an all-NaN row.
    """

    def predict(self, X):
        return np.full(len(X), 4321.0)


def _model(columns):
    return V014Model(country="XX", booster=_ConstantBooster(),
                     feature_columns=list(columns), neighbours=["YY"])


def _frame(index, **overrides):
    data = {c: np.arange(len(index), dtype=float) for c in ANCHOR_FEATURES}
    data["other"] = np.ones(len(index))
    frame = pd.DataFrame(data, index=index)
    for col, values in overrides.items():
        frame[col] = values
    return frame


# --- refusals ---------------------------------------------------------------

def test_a_row_with_no_anchor_observation_is_nan_never_zero():
    """This is GR's condition: no net-position actual at the serve cutoff, so
    every lag and trailing aggregate is NaN. The champion published a ~1e-7 MW
    flat line from exactly that input and the dashboard had to withhold it at
    render time (ABL-25/ABL-35). A 0.0 would be worse still — a 0 MW net
    position is a real, balanced-border reading."""
    index = pd.date_range("2026-08-08", periods=24, freq="h")
    frame = _frame(index)
    for col in ANCHOR_FEATURES:
        frame[col] = np.nan
    out = _model(frame.columns).predict_frame(frame)
    assert out.isna().all()
    assert not (out == 0).any()


def test_one_missing_anchor_is_tolerated_two_are_not():
    """Ordinary ingest jitter loses a single hour 72h back; that must not cost
    the country its whole day. A series that has actually stopped loses all of
    them."""
    index = pd.date_range("2026-08-08", periods=4, freq="h")
    assert MIN_ANCHOR_FEATURES == 2 and len(ANCHOR_FEATURES) == 3

    one_gone = _frame(index, **{ANCHOR_FEATURES[1]: np.nan})
    assert _model(one_gone.columns).predict_frame(one_gone).notna().all()

    two_gone = _frame(index, **{ANCHOR_FEATURES[1]: np.nan, ANCHOR_FEATURES[2]: np.nan})
    assert _model(two_gone.columns).predict_frame(two_gone).isna().all()


def test_refusal_is_per_target_hour_not_per_country():
    """A country that loses three hours keeps the other twenty-one. Dropping the
    whole day would trade a wrong number for a missing one at 8x the cost."""
    index = pd.date_range("2026-08-08", periods=24, freq="h")
    frame = _frame(index)
    for col in ANCHOR_FEATURES:
        frame.iloc[5:8, frame.columns.get_loc(col)] = np.nan
    out = _model(frame.columns).predict_frame(frame)
    assert out.isna().sum() == 3
    assert out.notna().sum() == 21


def test_no_model_for_a_country_raises_and_names_the_fix(tmp_path):
    """Substituting another country's model would return a plausible number in
    the wrong order of magnitude — DE swings +/-20 GW, EE +/-1 GW — and nothing
    downstream would flag it."""
    with pytest.raises(FileNotFoundError) as exc:
        load_model(tmp_path, "ZZ")
    message = str(exc.value)
    assert "train_v014.py" in message
    assert "Refusing rather than substituting" in message


def test_a_saved_model_round_trips_with_its_feature_list_and_neighbours(tmp_path):
    """The feature list travels with the artifact because column *order* is part
    of the model: a frame built with a different neighbour set would silently
    feed the booster the wrong columns."""
    model = _model(["a", "b", "c"])
    path = save_model(model, tmp_path)
    assert path == model_path(tmp_path, "XX")
    restored = load_model(tmp_path, "XX")
    assert restored.feature_columns == ["a", "b", "c"]
    assert restored.neighbours == ["YY"]


def test_a_model_whose_intercept_did_not_survive_loading_is_refused(tmp_path):
    """The cross-environment corruption that cost a whole backtest (ABL-69).

    This box runs two Pythons: the rail's `energy-forecast/.venv` (xgboost
    3.3.0), which trains and serves, and a conda 3.11 (xgboost 2.1.4) owning the
    bare `python`. An xgboost-3.3.0 pickle read under 2.1.4 keeps its trees but
    resets the fitted intercept to the 0.5 default, and then predicts a
    near-zero-mean series while raising nothing but a `UserWarning`. Measured on
    FR W12: MAE 1,688 MW under the right interpreter and 5,824 MW under the
    wrong one — a challenger reported as bad for a reason that was not the
    model. Here that corruption is simulated by rewriting the witness, because
    the real one needs a second interpreter.
    """
    import joblib

    model = _model(["a", "b", "c"])
    path = save_model(model, tmp_path)
    blob = joblib.load(path)
    assert blob["base_score"] is None or isinstance(blob["base_score"], float)
    blob["base_score"] = 6585.93          # what a real fit stores for FR
    blob["xgboost_version"] = "3.3.0"
    joblib.dump(blob, path)               # the booster still reads back 0.5-ish

    with pytest.raises(ModelArtifactError) as exc:
        load_model(tmp_path, "XX")
    message = str(exc.value)
    assert "6,585.93" in message                 # what it should have been
    assert ".venv" in message                    # names the fix, not just the fault
    assert "3.3.0" in message                    # names who wrote it


@pytest.mark.parametrize("serialised, expected", [
    ("[4.8775327E3]", 4877.5327),   # xgboost 3.x: one entry per target
    ("5E-1", 0.5),                  # xgboost 2.x: a bare scalar
    ("[]", None),                   # no target, nothing to witness
])
def test_the_intercept_is_read_from_both_xgboost_spellings(serialised, expected):
    """The guard against version skew must itself survive version skew.

    3.x writes `base_score` as a JSON array string and 2.x as a scalar. Parsing
    only the scalar form yields None on every 3.x artifact — and a None witness
    disables the guard *silently*, which is how the first cut of this shipped 19
    models carrying no witness at all.
    """
    class _Cfg:
        def save_config(self):
            import json
            return json.dumps({"learner": {"learner_model_param":
                                           {"base_score": serialised}}})

    class _Booster:
        def get_booster(self):
            return _Cfg()

    got = _base_score(_Booster())
    assert got is None if expected is None else got == pytest.approx(expected)


def test_an_unreadable_intercept_does_not_break_saving():
    """A booster that cannot report a config yields no witness rather than
    raising — saving a model must not depend on introspecting it."""
    assert _base_score(_ConstantBooster()) is None


def test_an_intact_model_loads_and_carries_the_version_that_wrote_it(tmp_path):
    """The guard must not fire on the ordinary case, or it gets disabled."""
    import joblib
    import xgboost

    save_model(_model(["a", "b"]), tmp_path)
    assert load_model(tmp_path, "XX").feature_columns == ["a", "b"]
    assert joblib.load(model_path(tmp_path, "XX"))["xgboost_version"] == xgboost.__version__


def test_an_artifact_written_before_the_guard_still_loads(tmp_path):
    """Absent witness means "cannot check", not "corrupt". A guard that refused
    every pre-existing artifact would force a retrain to read this one."""
    import joblib

    path = save_model(_model(["a", "b"]), tmp_path)
    blob = joblib.load(path)
    del blob["base_score"], blob["xgboost_version"]
    joblib.dump(blob, path)
    assert load_model(tmp_path, "XX").feature_columns == ["a", "b"]


def test_predict_aligns_to_the_trained_column_list_not_the_frames_order():
    """A frame whose columns arrive in a different order must not be silently
    re-interpreted position by position."""
    index = pd.date_range("2026-08-08", periods=3, freq="h")
    frame = _frame(index)
    model = _model(list(frame.columns))
    shuffled = frame[list(reversed(frame.columns))]
    assert model.predict_frame(shuffled).equals(model.predict_frame(frame))


# --- what the fit was allowed to see ----------------------------------------

def test_backtest_target_days_are_dropped_from_the_training_run_days():
    """A run day D produces target D+2, so the exclusion is on the *target*,
    two days later. Excluding the run day itself would hold out the wrong days
    and the W01-W12 backtest would be scoring days the fit had seen."""
    weeks = [("W01", "2024-01-15", "2024-01-21")]
    held_out = backtest_target_days(weeks)
    days = run_days_for_span("2024-01-10", "2024-01-25", held_out)
    assert pd.Timestamp("2024-01-13") not in days   # target 2024-01-15
    assert pd.Timestamp("2024-01-19") not in days   # target 2024-01-21
    assert pd.Timestamp("2024-01-12") in days       # target 2024-01-14
    assert pd.Timestamp("2024-01-20") in days       # target 2024-01-22
    assert len(days) == 16 - 7


def test_the_twelve_registered_weeks_are_all_held_out():
    held_out = backtest_target_days(config.BACKTEST_WEEKS)
    assert len(config.BACKTEST_WEEKS) == 12
    assert len(held_out) == 12 * 7
    for _, start, end in config.BACKTEST_WEEKS:
        assert pd.Timestamp(start) in held_out and pd.Timestamp(end) in held_out


def test_validation_splits_on_run_day_never_on_row():
    """The 24 hours of one run share every run-anchored feature. Splitting on
    rows would put 06:00 and 07:00 of the same target day on opposite sides, and
    the validation score would be measuring memorisation of its own run."""
    days = pd.date_range("2026-01-01", periods=100, freq="D")
    frame = pd.concat([
        pd.DataFrame({"run_day": d, "x": range(24)},
                     index=pd.date_range(d + pd.Timedelta(days=2), periods=24, freq="h"))
        for d in days])
    train, val = _split_by_run_day(frame, 0.12)
    assert not set(train["run_day"]) & set(val["run_day"])
    assert train["run_day"].max() < val["run_day"].min()
    assert len(val) % 24 == 0


def test_a_short_history_yields_no_validation_rather_than_a_meaningless_one():
    """Early stopping on three run days would stop on noise. Returning an empty
    validation set makes the caller skip early stopping instead."""
    days = pd.date_range("2026-01-01", periods=5, freq="D")
    frame = pd.DataFrame({"run_day": days, "x": 1.0})
    train, val = _split_by_run_day(frame, 0.12)
    assert len(train) == 5 and val.empty


# --- registration -----------------------------------------------------------

def test_v014_is_registered_as_a_promotion_candidate_with_its_own_model_name():
    spec = spec_for("V014")
    assert spec.model_name == "xgboost-V014"
    assert spec.promotion_candidate is True
    assert spec.model_name != CHAMPION_MODEL_NAME, \
        "sharing the champion's model_name would let the push ship a shadow model"


def test_v014_has_an_experiment_config_declaring_the_same_model_name():
    import json

    cfg = json.loads((config.EXPERIMENTS_DIR / "V014" / "config.json").read_text())
    assert cfg["model"]["model_name"] == spec_for("V014").model_name
    assert cfg["forecast_types"] == ["net_position"]
