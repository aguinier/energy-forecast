"""V016 — statistical correction layer on the champion (ABL-68).

Two mechanisms, both fitted per country on serve-faithful champion forecasts:

1. **Affine recalibration** `g = a + b*f`, fitted per country.
2. **AR(1) error correction** on the recalibrated residual, applied at the
   *true serve lead*.

Both were specified on ABL-68 to attack measured defects, and both turn out to
be bounded by the D+2 horizon and by the champion's per-country correlation in
ways the headline numbers hide. Measured, not assumed:

**Recalibration does not fix the shrinkage.** ABL-68 targets a median
forecast-on-actual slope of 0.676 (BE 0.27, NL 0.37, FR 0.42). But for an
affine map, `slope(corrected on actual) = b * slope(f on a)`, and with `b` from
OLS that product is exactly `rho**2`. Fitted on 2026-01-19..2026-06-15, this
*lowers* the slope in 15 of 16 countries (FR 0.480 -> 0.398, BE 0.298 -> 0.201,
NL 0.299 -> 0.171) while lowering MAE. The reason is that `b < 1` for 15 of 16:
the champion is not merely shrunk, it is shrunk *and* noisy, and the
error-minimising move is to shrink it further. Unit slope requires inflating
variance, which raises error. You get minimum error or faithful amplitude, not
both — `method` chooses which, and neither reaches the gate's [0.8, 1.2]
because that needs rho >= 0.894 (ols) or rho >= 0.8 (variance) and measured
per-country rho is 0.41-0.88.

**And a note on what mechanism 2 can deliver**, because that headline number is
misleading too. Residual lag-1 autocorrelation is 0.85-0.96 in every country, which
sounds like a large exploitable signal. But this is a D+2 product: the run fires
at 06:00Z on day D, net position is day-ahead published so actuals reach D 21:00,
and the target hours are D+2 00:00-23:00. The most recent residual the model can
observe is therefore **27 to 51 hours** before the hours it is correcting. An
AR(1) carried that far decays to `phi**27 .. phi**51` — 0.058..0.005 at
phi=0.90, and only 0.33..0.12 at phi=0.96. So the correction is small for most
countries and negligible for many, and that is a property of the horizon, not a
bug in the fit. `fit_country_correction` records `ar1_carry_at_min_lead` so the
report can state it rather than implying lag-1 autocorrelation transfers to D+2.

Guards, in the spirit of "when the data cannot support a number, render nothing":
a country whose fit is too thin, degenerate, or implausible is **passed through
uncorrected** with a stated reason. V016 is then simply V010 for that country —
honest, and visible in the report — rather than V010 distorted by coefficients
fitted on noise.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

# A per-country affine plus an AR coefficient is 3 parameters. Fitting them on
# one target day's 24 hours would be fitting the weather that day. Require a
# month of distinct target days and a real sample.
MIN_PAIRS = 480
MIN_TARGET_DAYS = 30
# Recalibration that has to more than triple a forecast is not recalibration;
# it means the champion carries little signal for that country and the affine
# term would mostly amplify its error.
MAX_SLOPE = 3.0
MIN_SLOPE = 0.1
# Below this the champion and the actual barely move together, so `b` is fitted
# on noise regardless of how many pairs there are.
MIN_CORR = 0.3


@dataclass
class CountryCorrection:
    """Fitted coefficients for one country, plus why they are or are not used."""
    country: str
    n_pairs: int
    n_target_days: int
    intercept_mw: float
    slope: float
    ar1_phi: float
    applied: bool
    reason: str
    corr: float | None = None
    slope_forecast_on_actual: float | None = None   # the shrinkage being fixed
    ar1_carry_at_min_lead: float | None = None      # phi**27, the honest ceiling
    serve_parity_verified: bool = True
    method: str = "ols"

    @property
    def is_identity(self) -> bool:
        return not self.applied

    def to_dict(self) -> dict:
        return asdict(self)


def _identity(country: str, reason: str, n_pairs: int = 0,
              n_days: int = 0, **kw) -> CountryCorrection:
    return CountryCorrection(
        country=country, n_pairs=n_pairs, n_target_days=n_days,
        intercept_mw=0.0, slope=1.0, ar1_phi=0.0, applied=False,
        reason=reason, **kw)


def fit_country_correction(pairs: pd.DataFrame, country: str,
                           min_pairs: int = MIN_PAIRS,
                           min_target_days: int = MIN_TARGET_DAYS,
                           serve_parity_verified: bool = True,
                           method: str = "ols") -> CountryCorrection:
    """Fit `a`, `b`, `phi` for one country.

    `pairs` needs columns `target_ts`, `forecast_value`, `actual`. Rows are
    (vintage, target hour) pairs from serve-faithful champion forecasts.

    `method` picks what the affine term optimises, and the two cannot be had at
    once:

    * `ols` — regression of actual on forecast. Minimises squared error, so it
      minimises the error of the number the dashboard shows. Its arithmetic
      consequence is `slope(corrected on actual) = b * slope(f on a) = rho**2`,
      so the amplitude diagnostic gets *worse* wherever rho < 1. Measured here:
      rho is 0.41-0.88 per country, so rho**2 is 0.17-0.77.
    * `variance` — match the forecast's standard deviation to the actual's.
      Forces `sd_ratio` to 1 and lifts the slope to `rho`, but it is not the
      error-minimising map, so MAE is higher than under `ols`.

    This is why the ABL-24 gate's `slope in [0.8, 1.2]` cannot be reached by
    *any* affine recalibration of V010: it needs rho >= 0.894 under `ols` or
    rho >= 0.8 under `variance`, and no country clears the first, few the
    second. Amplitude fidelity at this correlation is a better-model problem
    (V014/V015), not a correction-layer one.

    Returns a correction that is either applied, or an explicit identity with a
    reason. It never returns coefficients it does not trust.
    """
    df = pairs.dropna(subset=["forecast_value", "actual"])
    n = len(df)
    days = int(df["target_ts"].dt.normalize().nunique()) if n else 0

    if not serve_parity_verified:
        return _identity(country, "serve-parity unverified for this country: a "
                                  "fit on a reconstruction that does not match "
                                  "what production served would miscalibrate it",
                         n, days, serve_parity_verified=False)
    if n < min_pairs or days < min_target_days:
        return _identity(country, f"insufficient fitting data ({n} pairs over "
                                  f"{days} target days; need {min_pairs}/"
                                  f"{min_target_days})", n, days)

    f = df["forecast_value"].to_numpy(dtype=float)
    a = df["actual"].to_numpy(dtype=float)
    if np.std(f) == 0 or np.std(a) == 0:
        return _identity(country, "degenerate: forecast or actual has zero "
                                  "variance", n, days)

    corr = float(np.corrcoef(a, f)[0, 1])
    if not np.isfinite(corr) or abs(corr) < MIN_CORR:
        return _identity(country, f"champion carries too little signal "
                                  f"(corr {corr:.3f} < {MIN_CORR})", n, days,
                         corr=corr)

    if method == "ols":
        slope = float(np.cov(f, a, bias=True)[0, 1] / np.var(f))
    elif method == "variance":
        # Sign from the correlation: matching magnitudes must not flip a
        # negatively-correlated forecast into looking right.
        slope = float(np.sign(corr) * np.std(a) / np.std(f))
    else:
        raise ValueError(f"unknown method {method!r}; use 'ols' or 'variance'")
    intercept = float(np.mean(a) - slope * np.mean(f))
    # The shrinkage in the ABL-24 convention (forecast on actual).
    slope_f_on_a = float(np.cov(a, f, bias=True)[0, 1] / np.var(a))

    if not (MIN_SLOPE <= slope <= MAX_SLOPE):
        return _identity(country, f"implausible recalibration slope {slope:.3f} "
                                  f"outside [{MIN_SLOPE}, {MAX_SLOPE}]", n, days,
                         corr=corr, slope_forecast_on_actual=slope_f_on_a)

    # AR(1) on the recalibrated residual, on an unduplicated hourly chain.
    resid = (df.assign(r=a - (intercept + slope * f))
               .sort_values("target_ts")
               .drop_duplicates("target_ts", keep="last")
               .set_index("target_ts")["r"].asfreq("h"))
    phi = _lag1_autocorr(resid)
    # A negative lag-1 coefficient carried 27-51 hours would alternate sign with
    # the lead; that is not a stable correction to extrapolate. Clip to "no
    # AR term" rather than invent an oscillation.
    phi = float(np.clip(phi, 0.0, 0.999)) if np.isfinite(phi) else 0.0

    return CountryCorrection(
        country=country, n_pairs=n, n_target_days=days,
        intercept_mw=intercept, slope=slope, ar1_phi=phi, applied=True,
        reason="fitted", corr=corr, slope_forecast_on_actual=slope_f_on_a,
        ar1_carry_at_min_lead=float(phi ** MIN_SERVE_LEAD_H),
        serve_parity_verified=True, method=method)


def _lag1_autocorr(series: pd.Series) -> float:
    pairs = pd.concat([series, series.shift(1)], axis=1).dropna()
    if len(pairs) < 48:
        return 0.0
    value = pairs.corr().iloc[0, 1]
    return float(value) if np.isfinite(value) else 0.0


# A 06:00Z run on D sees actuals through D 21:00; the target day starts
# D+2 00:00. The nearest hour it corrects is 27 hours after the last residual.
MIN_SERVE_LEAD_H = 27


def apply_correction(forecast: np.ndarray, targets: pd.DatetimeIndex,
                     correction: CountryCorrection,
                     last_residual: float | None = None,
                     last_residual_ts: pd.Timestamp | None = None) -> np.ndarray:
    """Corrected forecast for `targets`.

    An unapplied correction returns the champion's values unchanged — V016 is
    V010 for that country, which is the honest answer when the fit is not
    trustworthy.

    The AR(1) term decays by the real distance from the last observed residual
    to each target hour, so a stale residual contributes almost nothing on its
    own. Nothing is carried when no residual is available.
    """
    forecast = np.asarray(forecast, dtype=float)
    if correction.is_identity:
        return forecast.copy()

    corrected = correction.intercept_mw + correction.slope * forecast

    if last_residual is not None and last_residual_ts is not None \
            and correction.ar1_phi > 0 and np.isfinite(last_residual):
        leads = ((pd.DatetimeIndex(targets) - pd.Timestamp(last_residual_ts))
                 / pd.Timedelta(hours=1)).to_numpy(dtype=float)
        # A residual from *after* the target would be information the run did
        # not have; refuse rather than carry it backwards.
        carry = np.where(leads > 0, correction.ar1_phi ** leads, 0.0)
        corrected = corrected + carry * float(last_residual)

    return corrected


def latest_residual(history: pd.DataFrame, as_of: pd.Timestamp,
                    correction: CountryCorrection
                    ) -> tuple[float | None, pd.Timestamp | None]:
    """Most recent recalibrated residual observable at `as_of`.

    `history` needs `target_ts`, `forecast_value`, `actual` — champion forecasts
    for hours that have since been measured. The residual is taken against the
    *recalibrated* forecast, matching what `apply_correction` adds it to.
    """
    if correction.is_identity:
        return None, None
    usable = history.dropna(subset=["forecast_value", "actual"])
    usable = usable[usable["target_ts"] < pd.Timestamp(as_of)]
    if usable.empty:
        return None, None
    row = usable.sort_values("target_ts").iloc[-1]
    recal = correction.intercept_mw + correction.slope * float(row["forecast_value"])
    return float(row["actual"]) - recal, pd.Timestamp(row["target_ts"])
