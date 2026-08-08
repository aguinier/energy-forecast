"""V014 — per-country XGBoost net-position challenger (ABL-69).

One model per country, fitted on the serve-faithful features in
`v014_features.py`. Per-country rather than pooled because net position is a
*balance*: DE swings +/-20 GW and EE +/-1 GW, they have different borders, and a
pooled model would spend its capacity learning the country dummy.

Three refusals are load-bearing, and all three exist because the failure mode
this program keeps hitting is a plausible wrong number rather than a crash:

- **No model for a country: no forecast.** `predict_country` raises rather than
  falling back to another country's model or to a global one. A DE-shaped model
  applied to EE would return numbers in the wrong order of magnitude and nothing
  downstream would flag it.
- **No anchor observation: no forecast.** If the run held no net-position actual
  at all inside the trailing window (`np_at_cutoff` NaN), every lag and every
  trailing aggregate is NaN and the tree returns whatever its default split
  directions happen to compose to — a confident number derived from nothing.
  That is GR's exact condition (ABL-25/ABL-35), and it is the condition under
  which the champion published its ~1e-7 MW flat line. `MIN_ANCHOR_FEATURES`
  states how much of the frame has to be real before a row is served.
- **Nothing is filled.** A NaN feature reaches XGBoost as NaN, which it handles
  natively by learning a default direction at fit time. Imputing a mean would
  turn "we do not know this border's flow" into "the flow was average".

Training holds out the twelve backtest weeks by **target day**, matching
`scripts/train.py --exclude-backtest`. One residual is worth naming rather than
implying away: a *retained* run day can still read a held-out week through its
own lag and trailing-mean features (a 28-day trailing mean stays contaminated
for four weeks after each week). Removing that would cost 12 x 28 = 336 of ~1,300
run days, and it would buy nothing measurable — a 28-day mean carries no
information about any individual held-out hour, and the model never sees a
held-out hour as a *label*. The backtest therefore tests what it claims to test:
predictions for target days the fit never scored against.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .v014_features import (ServeWindow, SourceCache, build_features,
                            build_training_frame, feature_columns)

logger = logging.getLogger("energy_forecast.v014")

EXPERIMENT_ID = "V014"
MODEL_NAME = "xgboost-V014"
FORECAST_TYPE = "net_position"

#: Feature columns that must be non-NaN for a target hour to be served. These
#: are the ones carrying the country's own recent level; without them the model
#: is extrapolating from calendar features alone.
ANCHOR_FEATURES = ("np_at_cutoff", "np_lag72h", "np_last7d_mean")

#: How many of `ANCHOR_FEATURES` have to be present. Two of three tolerates a
#: single missing hour 72h back (ordinary ingest jitter) while still refusing a
#: country whose series has stopped.
MIN_ANCHOR_FEATURES = 2

DEFAULT_PARAMS = {
    "n_estimators": 900,
    "max_depth": 6,
    "learning_rate": 0.04,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "reg_lambda": 2.0,
    "reg_alpha": 0.0,
    "tree_method": "hist",
    "objective": "reg:squarederror",
    "n_jobs": 4,
    "random_state": 42,
}

#: Fraction of run days (most recent, chronologically) used for early stopping.
VALIDATION_FRACTION = 0.12
EARLY_STOPPING_ROUNDS = 60


@dataclass
class V014Model:
    country: str
    booster: object
    feature_columns: list[str]
    neighbours: list[str]
    metadata: dict = field(default_factory=dict)

    def predict_frame(self, features: pd.DataFrame) -> pd.Series:
        """Predict for a feature frame, refusing rows with no anchor.

        Returns NaN — never 0.0 — for a refused hour. A net position of 0 MW is
        a real, balanced-border reading, so a 0 stand-in would publish a
        measurement nobody made. The caller drops NaN rows rather than storing
        them.
        """
        aligned = features.reindex(columns=self.feature_columns)
        missing = [c for c in self.feature_columns if c not in features.columns]
        if missing:
            logger.warning("V014 %s: %d feature(s) absent at predict time, left NaN: %s",
                           self.country, len(missing), ", ".join(missing[:6]))
        raw = self.booster.predict(aligned.to_numpy(dtype=np.float32))
        out = pd.Series(np.asarray(raw, dtype=float), index=features.index)
        present = sum(features[c].notna() if c in features.columns
                      else pd.Series(False, index=features.index)
                      for c in ANCHOR_FEATURES)
        refused = present < MIN_ANCHOR_FEATURES
        if refused.any():
            logger.warning("V014 %s: refusing %d of %d target hours - fewer than "
                           "%d anchor features present", self.country,
                           int(refused.sum()), len(out), MIN_ANCHOR_FEATURES)
            out[refused] = np.nan
        return out


def model_path(models_dir: Path, country: str) -> Path:
    return Path(models_dir) / FORECAST_TYPE / EXPERIMENT_ID / f"{country}.joblib"


def _base_score(booster) -> Optional[float]:
    """The booster's intercept, read back out of its own saved config.

    This is the one number that betrays a cross-version unpickle: XGBoost fits
    it to the mean of the target, and a booster deserialised by an incompatible
    version silently falls back to the 0.5 default. Measured on FR's artifact,
    2026-08-08: **4,877.53 read under xgboost 3.3.0, 0.5 under 2.1.4** — the
    same file.

    The two versions do not spell it the same way, which is fitting for the
    check that exists to catch version skew. 3.x serialises a JSON array string
    (`'[4.8775327E3]'`, one entry per target), 2.x a bare scalar (`'5E-1'`).
    Parsing only the scalar form returns None on every 3.x artifact, and a None
    witness disables the guard silently — which is how the first cut of this
    shipped 19 models carrying no witness at all.
    """
    import json

    try:
        cfg = json.loads(booster.get_booster().save_config())
        raw = cfg["learner"]["learner_model_param"]["base_score"]
        if isinstance(raw, str) and raw.lstrip().startswith("["):
            values = json.loads(raw)
            return float(values[0]) if values else None
        return float(raw)
    except Exception:  # noqa: BLE001 - an unreadable config must not break saving
        return None


class ModelArtifactError(RuntimeError):
    """Raised when a stored model did not survive deserialisation intact."""


def save_model(model: V014Model, models_dir: Path) -> Path:
    import joblib
    import xgboost

    path = model_path(models_dir, model.country)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"country": model.country, "booster": model.booster,
                 "feature_columns": model.feature_columns,
                 "neighbours": model.neighbours, "metadata": model.metadata,
                 # Integrity witnesses — see `load_model`. Written here so the
                 # artifact carries the evidence of its own soundness.
                 "xgboost_version": xgboost.__version__,
                 "base_score": _base_score(model.booster)}, path)
    return path


def load_model(models_dir: Path, country: str) -> V014Model:
    import joblib

    path = model_path(models_dir, country)
    if not path.exists():
        raise FileNotFoundError(
            f"V014 has no model for {country} ({path}). Refusing rather than "
            f"substituting another country's model: net position is a per-country "
            f"balance and a substituted model would return a plausible number in "
            f"the wrong order of magnitude. Train it first: "
            f"python scripts/train_v014.py --countries {country}")
    blob = joblib.load(path)
    booster = blob["booster"]
    _assert_artifact_survived_load(path, blob, booster)
    return V014Model(country=blob["country"], booster=booster,
                     feature_columns=list(blob["feature_columns"]),
                     neighbours=list(blob.get("neighbours", [])),
                     metadata=dict(blob.get("metadata", {})))


#: How far the reloaded intercept may drift from the stored one before the
#: artifact is refused. The corruption this catches is not a rounding
#: difference — it resets the intercept to 0.5 from thousands of MW — so the
#: tolerance only has to absorb float32 round-tripping.
BASE_SCORE_TOLERANCE_MW = 1.0


def _assert_artifact_survived_load(path: Path, blob: dict, booster) -> None:
    """Refuse a model whose intercept did not survive deserialisation.

    **This exists because it happened, silently, and cost a whole backtest**
    (ABL-69). This box has two Python environments — the scheduled rail's
    `energy-forecast/.venv` (3.14 / xgboost 3.3.0), which trains and serves, and
    a conda 3.11 / xgboost 2.1.4 that owns the bare `python` on PATH. An
    xgboost-3.3.0 pickle loaded under 2.1.4 does not fail: it drops the fitted
    intercept to the 0.5 default and predicts a near-zero-mean series, emitting
    nothing louder than a `UserWarning` about serialized models. Measured on FR
    W12: **MAE 1,688 MW under the right interpreter, 5,824 MW under the wrong
    one**, with SMAPE at 189% against a 200% ceiling. Nothing crashed and no
    test failed — the backtest simply reported that the challenger was bad.

    A version-equality check would be the obvious guard and is the wrong one:
    it forbids upgrades that are actually fine and says nothing about the case
    where the same version deserialises badly. This checks the **symptom** —
    the number the corruption destroys — so it is version-agnostic and fires
    exactly when predictions would be wrong.
    """
    stored = blob.get("base_score")
    if stored is None:
        return  # Written before this guard existed; nothing to compare against.
    actual = _base_score(booster)
    if actual is not None and abs(actual - stored) <= BASE_SCORE_TOLERANCE_MW:
        return
    import sys

    import xgboost

    # An unreadable intercept is a failure, not a pass. Storing a witness means
    # a real fitted booster went in, so being unable to read one back out is
    # itself evidence the artifact did not survive.
    read = "unreadable" if actual is None else f"{actual:,.4f}"
    raise ModelArtifactError(
        f"V014 model {path.name} did not survive loading: its intercept read "
        f"back as {read} but was saved as {stored:,.4f}. This model was "
        f"written by xgboost {blob.get('xgboost_version', '?')} and is being "
        f"loaded by xgboost {xgboost.__version__} on {sys.executable}. Loading "
        f"it anyway would predict a near-zero-mean series and report the "
        f"challenger as bad rather than raising. Run this under the same "
        f"interpreter the scheduled rail uses "
        f"(energy-forecast/.venv/Scripts/python.exe), or retrain."
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def backtest_target_days(weeks: Iterable[tuple[str, str, str]]) -> set[pd.Timestamp]:
    """Every calendar day inside the held-out backtest weeks."""
    days: set[pd.Timestamp] = set()
    for _, start, end in weeks:
        days.update(pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D"))
    return days


def run_days_for_span(start, end, exclude_target_days: Optional[set] = None
                      ) -> list[pd.Timestamp]:
    """Run days over `[start, end]`, dropping those whose target day is held out."""
    excluded = exclude_target_days or set()
    days = []
    for day in pd.date_range(pd.Timestamp(start).normalize(),
                             pd.Timestamp(end).normalize(), freq="D"):
        if (day + pd.Timedelta(days=2)) in excluded:
            continue
        days.append(day)
    return days


def _split_by_run_day(frame: pd.DataFrame, validation_fraction: float
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split on run day, never on row.

    Splitting on rows would put 06:00 and 07:00 of the same target day on
    opposite sides, and the 24 hours of one run share every run-anchored
    feature — the validation score would then be measuring memorisation of the
    training half of the same day.
    """
    days = np.sort(frame["run_day"].unique())
    if len(days) < 10:
        return frame, frame.iloc[0:0]
    n_val = max(1, int(round(len(days) * validation_fraction)))
    cut = days[-n_val]
    return frame[frame["run_day"] < cut], frame[frame["run_day"] >= cut]


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """MAE, RMSE and the regression slope the promotion gate reads.

    `slope` is actual-on-forecast, the same orientation as
    `src/evaluation/net_position.py` — below 1 means the forecast is flatter
    than reality, which is the defect this whole program exists to fix.
    """
    mask = np.isfinite(actual) & np.isfinite(predicted)
    if mask.sum() < 2:
        return {"mae": None, "rmse": None, "slope": None, "n": int(mask.sum())}
    a, p = actual[mask], predicted[mask]
    var = float(np.var(p))
    slope = float(np.cov(a, p, bias=True)[0, 1] / var) if var > 0 else None
    return {"mae": float(np.mean(np.abs(a - p))),
            "rmse": float(np.sqrt(np.mean((a - p) ** 2))),
            "slope": slope, "n": int(mask.sum())}


def train_country(conn, country: str, run_days: list[pd.Timestamp],
                  neighbours: Iterable[str], params: Optional[dict] = None,
                  cache: Optional[SourceCache] = None) -> tuple[V014Model, dict]:
    """Fit one country's model. Raises if there is not enough paired data."""
    from xgboost import XGBRegressor

    neighbours = list(neighbours)
    frame = build_training_frame(conn, country, run_days, neighbours=neighbours, cache=cache)
    if frame.empty:
        raise ValueError(f"{country}: no feature rows over the requested run days")
    frame = frame[frame["target_net_position_mw"].notna()]
    if len(frame) < 24 * 90:
        raise ValueError(
            f"{country}: only {len(frame)} paired rows (< 90 target days). "
            f"Refusing to fit — a model this thin would be scored as if it were "
            f"comparable to the champion's three years.")

    cols = feature_columns(frame)
    train, val = _split_by_run_day(frame, VALIDATION_FRACTION)
    cfg = {**DEFAULT_PARAMS, **(params or {})}

    X_tr = train[cols].to_numpy(dtype=np.float32)
    y_tr = train["target_net_position_mw"].to_numpy(dtype=float)
    fit_kwargs = {}
    if not val.empty:
        cfg = {**cfg, "early_stopping_rounds": EARLY_STOPPING_ROUNDS}
        fit_kwargs["eval_set"] = [(val[cols].to_numpy(dtype=np.float32),
                                   val["target_net_position_mw"].to_numpy(dtype=float))]
        fit_kwargs["verbose"] = False

    booster = XGBRegressor(**cfg)
    booster.fit(X_tr, y_tr, **fit_kwargs)

    model = V014Model(country=country, booster=booster, feature_columns=cols,
                      neighbours=neighbours)
    report = {
        "country": country,
        "rows_total": int(len(frame)),
        "rows_train": int(len(train)),
        "rows_validation": int(len(val)),
        "run_day_first": str(frame["run_day"].min().date()),
        "run_day_last": str(frame["run_day"].max().date()),
        "n_features": len(cols),
        "neighbours": neighbours,
        "best_iteration": int(getattr(booster, "best_iteration", 0) or 0),
        # What fraction of training rows had a real issued weather forecast. The
        # archive begins 2026-01-11, so on a full-span fit this is small and the
        # weather columns are mostly NaN — stated, so a feature-importance read
        # is not mistaken for "the model ignores weather".
        "weather_available_frac": float(frame["weather_available"].mean()),
        "xb_missing_frac": float(frame["xb_missing"].mean()),
        "train": _metrics(y_tr, booster.predict(X_tr)),
    }
    if not val.empty:
        report["validation"] = _metrics(
            val["target_net_position_mw"].to_numpy(dtype=float),
            model.predict_frame(val[cols]).to_numpy())
    model.metadata = report
    return model, report


# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------

def predict_country(conn, models_dir: Path, country: str, window: ServeWindow,
                    cache: Optional[SourceCache] = None) -> pd.Series:
    """One country's 24 target hours for `window`. NaN where the run is refused."""
    from .v014_features import build_cache

    model = load_model(models_dir, country)
    if cache is None:
        cache = build_cache(conn, country,
                            window.day_ahead_cutoff - pd.Timedelta(days=35),
                            window.target_index.max())
    features = build_features(cache, window, neighbours=model.neighbours)
    return model.predict_frame(features)
