"""Split-conformal recalibration of a stored quantile band (ABL-650).

The Chronos-2 net-position head emits nine levels, 0.10 .. 0.90, and the
dashboard draws the outer pair as a band captioned "p10-p90". Measured over the
ABL-595 gate window that band is too narrow: the champion's actuals fall inside
it in 9 of 19 zones at the pre-registered [75, 85]% bar, and as low as 67.1%
(PT). The emitted levels are the right levels -- the PIT is smooth, monotone and
crossing-free, and all four nested central intervals under-cover together, which
is a scale defect and not a labelling one.

This module fixes the scale. Two multipliers per zone, anchored at the stored
median:

    q'_t = q50 - s_lo * (q50 - q_t)      for t < 0.5
    q'_t = q50 + s_hi * (q_t - q50)      for t > 0.5
    q'_50 = q50

Three properties this shape is chosen for:

- **The median cannot move.** `q50` is a fixed point of the map by construction,
  and the served point forecast in `forecasts` is a different row entirely,
  which this module never touches. `verify_median_invariant` re-checks it
  numerically rather than trusting the algebra.
- **It cannot create a crossing.** Each side is an increasing affine map of the
  emitted quantile with a non-negative multiplier, so the stored order survives.
- **The two sides are fitted separately.** The champion's PIT is shifted as well
  as narrow (its q50 sits at empirical 51.8%), so one symmetric multiplier would
  reach 80% total coverage with unbalanced tails. Two one-sided fits put 10% on
  each side.

`s_lo` is the split-conformal quantile of the normalised lower deviation, so on
exchangeable data the calibrated p10 has finite-sample coverage at least
1 - alpha. A calibration is only ever fitted on vintages strictly earlier than
the vintages it is scored on.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# The registered multipliers. One file, keyed by `model_name`; a model that is
# not in it is served exactly as its head emits, which is the safe default -- an
# unregistered model must not inherit somebody else's spread.
REGISTRY_PATH = (Path(__file__).parents[1] / "experiments"
                 / "net_position_quantile_calibration.json")

QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
QCOLS = tuple(f"q{int(round(q * 100))}" for q in QUANTILE_LEVELS)
MEDIAN_COL = "q50"

# A half-width at or below this many MW carries no information about spread and
# would make the normalised score explode. Those rows are dropped from the fit
# (the transform leaves them unchanged anyway -- scaling zero is zero).
MIN_HALFWIDTH_MW = 1e-9


class InsufficientCalibrationDataError(RuntimeError):
    """Fewer fit rows than the conformal level can resolve.

    Refusing is the point. A multiplier estimated from a handful of rows would
    still produce a band, and the band would still be labelled p10-p90.
    """


@dataclass(frozen=True)
class ZoneCalibration:
    """The two multipliers for one zone, with the evidence behind them."""

    s_lo: float
    s_hi: float
    n_fit: int
    alpha_lo: float
    alpha_hi: float

    def as_dict(self) -> dict:
        return {"s_lo": round(float(self.s_lo), 4),
                "s_hi": round(float(self.s_hi), 4),
                "n_fit": int(self.n_fit),
                "alpha_lo": self.alpha_lo, "alpha_hi": self.alpha_hi}

    @classmethod
    def identity(cls) -> "ZoneCalibration":
        return cls(1.0, 1.0, 0, float("nan"), float("nan"))


def _conformal_quantile(scores: np.ndarray, level: float) -> float:
    """The split-conformal (n+1)-corrected empirical quantile.

    `level` is the mass that must fall at or below the returned value, so a
    two-tailed 10-90 band asks for `level = 1 - alpha` on each side.
    """
    n = len(scores)
    k = int(np.ceil((n + 1) * level))
    if k > n:
        raise InsufficientCalibrationDataError(
            f"{n} fit rows cannot resolve a conformal quantile at level "
            f"{level:.3f}; need at least {int(np.ceil(level / (1 - level)))}")
    return float(np.sort(scores)[k - 1])


def _scores(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Normalised deviations below and above the median.

    Both are signed: a row above the median contributes a negative lower score.
    Keeping every row on both sides is deliberate -- the lower multiplier has to
    be the quantile of the *whole* deviation distribution, not of its lower half,
    or the tail probability it targets is the wrong one.
    """
    med = df[MEDIAN_COL].to_numpy(dtype=float)
    lo_hw = med - df["q10"].to_numpy(dtype=float)
    hi_hw = df["q90"].to_numpy(dtype=float) - med
    a = df["actual"].to_numpy(dtype=float)
    ok = (lo_hw > MIN_HALFWIDTH_MW) & (hi_hw > MIN_HALFWIDTH_MW) & np.isfinite(a)
    return (med[ok] - a[ok]) / lo_hw[ok], (a[ok] - med[ok]) / hi_hw[ok]


def fit_zone_calibration(fit_df: pd.DataFrame, zones, alpha_lo: float = 0.10,
                         alpha_hi: float = 0.10, pooled: bool = False
                         ) -> dict[str, ZoneCalibration]:
    """Fit one `ZoneCalibration` per zone on `fit_df`.

    `fit_df` must hold `country_code`, `actual` and the nine `q*` columns, and
    must come from vintages strictly earlier than anything it will be applied
    to. `pooled=True` fits one calibration on every row and files it under the
    single key in `zones` -- the shrunk alternative to 19 independent fits, kept
    so the two can be compared out-of-sample instead of assumed.
    """
    out: dict[str, ZoneCalibration] = {}
    for zone in zones:
        sub = fit_df if pooled else fit_df[fit_df["country_code"] == zone]
        lo, hi = _scores(sub)
        if len(lo) == 0:
            raise InsufficientCalibrationDataError(f"{zone}: no usable fit rows")
        out[zone] = ZoneCalibration(
            s_lo=max(_conformal_quantile(lo, 1 - alpha_lo), 0.0),
            s_hi=max(_conformal_quantile(hi, 1 - alpha_hi), 0.0),
            n_fit=int(len(lo)), alpha_lo=alpha_lo, alpha_hi=alpha_hi)
    return out


def apply_zone_calibration(df: pd.DataFrame,
                           calibration: dict[str, ZoneCalibration],
                           qcols=QCOLS) -> pd.DataFrame:
    """Return a copy of `df` with the quantile columns recalibrated.

    A zone with no calibration passes through untouched: an unregistered zone
    must keep emitting the band it was fitted to emit, not a band scaled by
    somebody else's multipliers.
    """
    out = df.copy()
    med = out[MEDIAN_COL].to_numpy(dtype=float)
    s_lo = np.ones(len(out))
    s_hi = np.ones(len(out))
    codes = out["country_code"].to_numpy()
    for zone, cal in calibration.items():
        m = codes == zone
        s_lo[m] = cal.s_lo
        s_hi[m] = cal.s_hi
    for col in qcols:
        if col == MEDIAN_COL:
            continue
        v = out[col].to_numpy(dtype=float)
        level = int(col[1:]) / 100.0
        out[col] = np.where(level < 0.5, med - s_lo * (med - v),
                            med + s_hi * (v - med))
    return out


def calibrate_quantile_array(quantiles: np.ndarray, levels, cal: ZoneCalibration
                             ) -> np.ndarray:
    """The serving-side form: recalibrate a `(n_levels, horizon)` block.

    `levels[i]` is the nominal level of `quantiles[i]`, and exactly one of them
    must be 0.5 -- the anchor. Without a median row there is nothing to hold
    fixed, and a band recentred on something else is a change to what we
    forecast, which this fix is explicitly not allowed to make.
    """
    levels = list(levels)
    if 0.5 not in levels:
        raise ValueError(
            "calibrate_quantile_array needs the 0.5 level as the anchor; "
            f"got levels={levels}")
    med = np.asarray(quantiles, dtype=float)[levels.index(0.5)]
    out = np.array(quantiles, dtype=float, copy=True)
    for i, level in enumerate(levels):
        if level < 0.5:
            out[i] = med - cal.s_lo * (med - out[i])
        elif level > 0.5:
            out[i] = med + cal.s_hi * (out[i] - med)
    return out


class CalibrationRegistrationError(RuntimeError):
    """The registry is present but does not describe a usable calibration."""


def load_registry(path: Path | str | None = None) -> dict:
    """Parse the registration file. Missing file -> no calibrations at all."""
    p = Path(path or REGISTRY_PATH)
    if not p.exists():
        logger.warning("no quantile-calibration registration at %s; every band "
                       "is served as emitted", p)
        return {}
    doc = json.loads(p.read_text(encoding="utf-8"))
    models = doc.get("models") or {}
    for name, spec in models.items():
        for key in ("s_lo_applied", "s_hi_applied", "alpha", "fit_window",
                    "fit_vintages", "mode"):
            if key not in spec:
                raise CalibrationRegistrationError(
                    f"{name}: registration is missing '{key}'")
        if spec["mode"] != "pooled":
            raise CalibrationRegistrationError(
                f"{name}: mode '{spec['mode']}' is not registered. Per-zone "
                f"multipliers were measured on ABL-650 and lose to no "
                f"calibration at all out-of-sample; only 'pooled' ships.")
        # An upstream model's calibration is already inside this model's input
        # (V016 is an affine map of the champion's band, so it inherits the
        # champion's widening exactly). The applied factor is the increment;
        # the total is what the band needs end to end. Cross-checking them here
        # means a change to one registration cannot silently double-count in
        # the other.
        up = spec.get("upstream")
        if up:
            if up not in models:
                raise CalibrationRegistrationError(
                    f"{name}: upstream '{up}' is not registered")
            for side in ("lo", "hi"):
                composed = (spec[f"s_{side}_applied"]
                            * models[up][f"s_{side}_applied"])
                if abs(composed - spec[f"s_{side}_total"]) > 1e-3:
                    raise CalibrationRegistrationError(
                        f"{name}: s_{side}_applied x {up} = {composed:.4f} but "
                        f"s_{side}_total is {spec[f's_{side}_total']}")
    return models


def registered_calibration(model_name: str, zones,
                           path: Path | str | None = None
                           ) -> dict[str, ZoneCalibration]:
    """The `{zone: ZoneCalibration}` map to apply to `model_name`'s output.

    Empty when the model is unregistered -- callers treat that as "serve the
    band the head emitted", never as an error.
    """
    spec = load_registry(path).get(model_name)
    if spec is None:
        return {}
    cal = ZoneCalibration(s_lo=float(spec["s_lo_applied"]),
                          s_hi=float(spec["s_hi_applied"]),
                          n_fit=int(spec.get("fit_rows", 0)),
                          alpha_lo=float(spec["alpha"]),
                          alpha_hi=float(spec["alpha"]))
    return {z: cal for z in zones}


def calibrate_quantile_dict(quantile_dict: dict, model_name: str,
                            country_code: str, path: Path | str | None = None
                            ) -> tuple[dict, ZoneCalibration | None]:
    """The `forecast_chronos2` shape: `{level: array}` in, calibrated out.

    Returns the map plus the calibration that was applied (None when the model
    is unregistered), so the caller can log which multipliers a vintage carries
    instead of the reader having to infer it from the numbers.
    """
    cal = registered_calibration(model_name, [country_code], path).get(country_code)
    if cal is None:
        return quantile_dict, None
    levels = sorted(quantile_dict)
    block = np.vstack([np.asarray(quantile_dict[level], dtype=float)
                       for level in levels])
    out = calibrate_quantile_array(block, levels, cal)
    return {level: out[i] for i, level in enumerate(levels)}, cal


def verify_median_invariant(before: pd.DataFrame, after: pd.DataFrame) -> float:
    """Max |delta| on the median column, in MW. Must be exactly 0.0.

    Called by the harness and by the tests rather than left as a comment: the
    constraint on this change is that the point forecast is bit-identical, and
    "the algebra says so" is not a measurement.
    """
    if len(before) != len(after):
        raise ValueError("median check needs the same rows on both sides")
    return float((after[MEDIAN_COL].to_numpy() - before[MEDIAN_COL].to_numpy())
                 .__abs__().max())
