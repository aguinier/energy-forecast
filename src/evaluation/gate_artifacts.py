"""One writer for the pre-registered gate harnesses' model artifacts.

ABL-342. `scripts/evaluate_wind_retrain.py` and `scripts/evaluate_solar_retrain.py`
each fitted a bare estimator and `joblib.dump`ed seven keys of their own
choosing, bypassing `Forecaster.save`. Post-ABL-331 that shape is a silent
train/serve skew: `Forecaster.load` resolves an **absent** `training_source` to
`LEGACY_RENEWABLE_TRAINING_SOURCE` ('energy_renewable') for every
`config.RENEWABLE_TYPES` artifact, and every key in `load` is read with
`.get(..., default)`. So a pair fitted on `energy_generation` and written in the
bare shape loads clean, raises nothing, and serves every lag and rolling feature
from the other table for the rest of the artifact's life — the exact state the
ABL-321 verdict rejected as unmeasured. The same shape omits `base_score` and
`xgboost_version`, which makes ABL-183's intercept witness a no-op, so an
xgboost booster that lost its fitted intercept to a version mismatch would also
load unchallenged.

Both facts are derived by `Forecaster.save`. The fix is therefore to stop having
a second writer rather than to teach the second writer the same keys: the
harnesses hand over the fitted model and the builder that produced its training
rows, and the provenance comes along by construction.

The **builder** is the argument, not a source string, on purpose. ABL-331's rule
is that an artifact records what training actually read, not what it meant to
read; `RenewableFeatureBuilder.actuals_source` is the value the target-series
loader was handed, so passing it through cannot drift from the series the model
was fitted on. `None` there means "db's default", and `_resolved_training_source`
resolves the same `None` through the same constant — so a harness that names no
source still records the table it read rather than the absent key that would
later be guessed at.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

from ..forecaster import Forecaster
from ..wind_features import RenewableFeatureBuilder


def save_gate_artifact(
    path: Path,
    *,
    model: Any,
    builder: RenewableFeatureBuilder,
    algorithm: str,
    params: Dict[str, Any],
    feature_columns: Sequence[str],
    fit_window: Sequence[Any],
) -> Path:
    """Write a gate-harness artifact through `Forecaster.save`.

    Args:
        path: Destination `model.joblib`; parents are created.
        model: The estimator the harness fitted.
        builder: The `RenewableFeatureBuilder` that produced the training rows.
            Supplies the country, the forecast type and — the point of this
            module — the table the target series was actually read from.
        algorithm: 'xgboost', 'lightgbm' or 'catboost'.
        params: The hyperparameters the model was fitted with.
        feature_columns: The fitted feature order.
        fit_window: [start, end_exclusive] of the fit, stringified into the
            artifact.

    Returns:
        The path written.
    """
    forecaster = Forecaster(
        builder.country_code,
        builder.forecast_type,
        algorithm=algorithm,
        training_source=builder.actuals_source,
    )
    # `__init__` merges hyperparams *over the algorithm defaults*, which would
    # put back a key the harness deliberately removed — the wind gate pops
    # `early_stopping_rounds` because its final fit has no validation set. The
    # artifact has to record the params the model was fitted with, so assign
    # them rather than let the merge restate the defaults.
    forecaster.hyperparams = dict(params)
    forecaster.model = model
    forecaster.feature_columns = list(feature_columns)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    forecaster.save(
        str(path),
        # Kept for the harnesses' own readers, which predate this module's key
        # set. `params` is `hyperparams` under the name those readers use; it is
        # written under both names rather than migrating readers in this diff.
        extra_metadata={
            "params": dict(params),
            "fit_window": [str(bound) for bound in fit_window],
        },
    )
    return path
