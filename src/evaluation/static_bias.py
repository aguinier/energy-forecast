"""ABL-651 -- intercept-only static bias correction for the net-position champion.

One parameter per zone, subtracted from the forecast:

    corrected = forecast - intercept_mw

That is the whole model. It is deliberately the *only* shape here, because the
restriction is the reason the Board authorised this and not a general correction
layer: an added constant leaves `cov(actual, forecast)` untouched, so the gate's
`slope` (OLS of forecast on actual), `corr` and `sd_ratio` are invariant by
construction. V016's affine layer moved slope *away* from the gate band in 15 of
19 zones (`docs/claude/09-model-details.md`); an intercept cannot, and
`tests/test_static_bias.py` pins that numerically rather than asserting it.

**What decides whether a zone is corrected at all.** ABL-65 measured a rolling
per-zone offset on the 198-day reconstruction and found nothing to estimate
(oracle in-sample gain 0.01% MAE, holdout -0.13%). That verdict was taken on a
cohort that predates the context-cutoff fix, so it does not settle the live
post-fix question -- but it does set the prior, and it is why every zone here has
to earn its correction against an explicit stability test instead of inheriting
one from a list. The tests are in `Thresholds`; each one exists because a
specific failure mode would otherwise pass:

* **materiality** -- a bias smaller than the gate's own 5%-of-mean-|net position|
  bar is not the defect this issue was opened about.
* **sign agreement across split halves** -- a zone whose bias changes sign
  between the two halves of the window has no static offset to remove; fitting
  one on the pooled mean bakes in whichever half was longer.
* **magnitude agreement across split halves** -- both halves must carry the bias
  on their own. This is the test DE fails: a large pooled bias assembled from
  one quiet half and one extreme half is a level excursion, not an offset.
* **separation from zero** -- the correction lowers expected out-of-sample MSE by
  `se**2 * (t**2 - 1)`, so the break-even is `|t| = 1`, not `|t| = 1.96`.
  `MIN_ABS_T = 2.0` is deliberately conservative of that break-even: it keeps
  ~75% of the available gain while refusing anything a coin could produce. The
  break-even is reported per zone so the cost of that conservatism is visible.

**The independence unit is the target day, not the hour.** Hourly net-position
residuals are autocorrelated at 0.75-0.97 lag-1, so a t-statistic built on ~500
hourly errors overstates its own evidence by an order of magnitude -- the mistake
ABL-65 §2 named (`1/n_eff`). Day-level bias autocorrelation past the serve gap
is ~0 (ABL-65 §5: median -0.059 at lag 2 days), which is what makes the day mean
usable as a unit; `day_acf1`/`day_acf2` are measured on the scored window and
reported so that assumption is checked rather than inherited.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd

# The gate's own per-country bias bar (ABL-24 plan Rev 3 §4, `GATE_BIAS_FRAC`).
# Reused rather than re-chosen: "material" should mean the same thing here as it
# means in the criterion this correction exists to move.
MIN_BIAS_FRAC = 0.05

# Each half must carry at least this share of the pooled bias, and must clear
# MIN_BIAS_FRAC on its own half. Together these bound the half-to-half magnitude
# ratio to roughly [1/3, 3] and refuse a bias that lives in one half only.
MIN_HALF_SHARE = 0.5

# Expected out-of-sample MSE change from subtracting a fitted mean is
# `se**2 - bias**2 = se**2 * (1 - t**2)`, so any |t| > 1 helps in expectation.
# Requiring 2 gives a margin against that break-even.
MIN_ABS_T = 2.0
BREAK_EVEN_ABS_T = 1.0

# The split-half agreement tests are the load-bearing ones, and a half needs
# enough target days to be able to disagree: at 7 days per half the test has
# almost no power, which is not a hypothetical. Fitted on the 14 post-fix target
# days available up to 2026-08-20, the test admitted RO -- and RO's frozen
# -279 MW intercept then cost **+32.6% MAE** on the held-out 11 days, because
# RO's bias had swung sign (ABL-65 section 5 measured RO's day-bias
# autocorrelation at -0.171 across exactly the gap a correction must bridge).
# The same test over 27 days rejects RO on sign disagreement. So the floor is
# ten target days per half, and the number is a measurement, not a convention.
MIN_TARGET_DAYS = 20
MIN_PAIRS = 400

# ABL-31/ABL-35: a whole vintage inside 1 MW is a zero-filled context, not a
# balanced zone. Never dress one in a fitted level (the guard ABL-65 carries).
DEGENERATE_MAX_ABS_MW = 1.0


@dataclass(frozen=True)
class Thresholds:
    """The qualification test, as data, so a report can print what it ran."""
    min_bias_frac: float = MIN_BIAS_FRAC
    min_half_share: float = MIN_HALF_SHARE
    min_abs_t: float = MIN_ABS_T
    min_target_days: int = MIN_TARGET_DAYS
    min_pairs: int = MIN_PAIRS

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class WindowStats:
    """Bias and shape statistics for one zone over one window."""
    n: int
    n_target_days: int
    bias_mw: float | None = None
    mae_mw: float | None = None
    mean_abs_actual_mw: float | None = None
    bias_frac_pct: float | None = None       # |bias| as % of mean |net position|
    day_se_mw: float | None = None           # s.e. of the mean of day biases
    t_stat: float | None = None
    ci_lo_mw: float | None = None
    ci_hi_mw: float | None = None
    nw_se_mw: float | None = None            # Newey-West s.e., Bartlett, lag 3
    nw_t_stat: float | None = None
    day_sign_frac: float | None = None       # share of days sharing the pooled sign
    day_acf1: float | None = None
    day_acf2: float | None = None
    mean_actual_mw: float | None = None
    slope: float | None = None               # OLS of forecast on actual (ABL-24)
    corr: float | None = None
    sd_ratio: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ZoneDecision:
    """Whether a zone is corrected, by how much, and why."""
    country: str
    applied: bool
    intercept_mw: float
    reason: str
    tests: dict = field(default_factory=dict)
    window: WindowStats | None = None
    half1: WindowStats | None = None
    half2: WindowStats | None = None

    @property
    def is_identity(self) -> bool:
        return not self.applied

    def to_dict(self) -> dict:
        return {
            "country": self.country,
            "applied": self.applied,
            "intercept_mw": self.intercept_mw,
            "reason": self.reason,
            "tests": self.tests,
            "window": self.window.to_dict() if self.window else None,
            "half1": self.half1.to_dict() if self.half1 else None,
            "half2": self.half2.to_dict() if self.half2 else None,
        }


def _identity(country: str, reason: str, **kw) -> ZoneDecision:
    return ZoneDecision(country=country, applied=False, intercept_mw=0.0,
                        reason=reason, **kw)


def _t_critical(dof: int) -> float:
    """Two-sided 95% Student-t critical value.

    Table-driven so the module does not take a scipy dependency the rest of the
    evaluation package does not have; above 30 the normal value is within 2%.
    """
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
             13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
             19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
             25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}
    if dof < 1:
        return float("nan")
    return table.get(dof, 1.96 + 2.4 / max(dof, 1))


def _newey_west_se(day_bias: np.ndarray, max_lag: int = 3) -> float | None:
    """Autocorrelation-robust s.e. of the mean of a day-bias series.

    The plain s.e. treats target days as independent. Day-level bias
    autocorrelation is small past the serve gap but not identically zero, and a
    positive lag-1 makes the plain t **overstate** its evidence -- which matters
    only for a zone the test lets through, so this is reported beside every
    verdict rather than substituted for it. Bartlett weights, truncation 3
    (~n**(1/3) at the window lengths this runs on).
    """
    x = np.asarray(day_bias, dtype=float)
    n = len(x)
    if n < 6:
        return None
    d = x - x.mean()
    gamma0 = float(np.dot(d, d) / n)
    total = gamma0
    for k in range(1, min(max_lag, n - 1) + 1):
        gamma_k = float(np.dot(d[:-k], d[k:]) / n)
        total += 2.0 * (1.0 - k / (max_lag + 1.0)) * gamma_k
    # A truncated estimator can go non-positive on a short, strongly
    # negatively-autocorrelated series; fall back rather than emit a fake number.
    if total <= 0:
        return None
    return float(np.sqrt(total / n))


def measure(pairs: pd.DataFrame) -> WindowStats:
    """Bias, its day-level uncertainty, and the shape statistics, for one zone.

    `pairs` needs `target_ts`, `forecast_value`, `actual`. Bias is
    `mean(forecast - actual)` and slope is the OLS of forecast on actual, both
    exactly as `src/evaluation/net_position.point_metrics` defines them -- this
    module must not be able to disagree with the gate about what a zone's bias
    is.
    """
    df = pairs.dropna(subset=["forecast_value", "actual"])
    n = int(len(df))
    days = df["target_ts"].dt.normalize()
    n_days = int(days.nunique()) if n else 0
    if n == 0:
        return WindowStats(n=0, n_target_days=0)

    a = df["actual"].to_numpy(dtype=float)
    f = df["forecast_value"].to_numpy(dtype=float)
    err = f - a
    bias = float(np.mean(err))
    mean_abs_actual = float(np.mean(np.abs(a)))
    var_a = float(np.var(a))

    day_bias = pd.Series(err, index=days.to_numpy()).groupby(level=0).mean().sort_index()
    if len(day_bias) > 1:
        sd = float(day_bias.std(ddof=1))
        se = sd / np.sqrt(len(day_bias))
        t = bias / se if se > 0 else None
        half = _t_critical(len(day_bias) - 1) * se
        ci = (bias - half, bias + half)
    else:
        se, t, ci = None, None, (None, None)

    def _acf(lag: int) -> float | None:
        if len(day_bias) <= lag + 2:
            return None
        x, y = day_bias.to_numpy()[:-lag], day_bias.to_numpy()[lag:]
        if np.std(x) == 0 or np.std(y) == 0:
            return None
        return float(np.corrcoef(x, y)[0, 1])

    nw_se = _newey_west_se(day_bias.to_numpy())
    nw_t = bias / nw_se if nw_se else None

    return WindowStats(
        n=n, n_target_days=n_days,
        bias_mw=bias,
        mae_mw=float(np.mean(np.abs(err))),
        mean_abs_actual_mw=mean_abs_actual,
        bias_frac_pct=(100.0 * abs(bias) / mean_abs_actual
                       if mean_abs_actual > 0 else None),
        day_se_mw=se, t_stat=t, ci_lo_mw=ci[0], ci_hi_mw=ci[1],
        nw_se_mw=nw_se, nw_t_stat=nw_t,
        day_sign_frac=float(np.mean(np.sign(day_bias.to_numpy()) == np.sign(bias)))
        if len(day_bias) else None,
        day_acf1=_acf(1), day_acf2=_acf(2),
        mean_actual_mw=float(np.mean(a)),
        slope=(float(np.cov(a, f, bias=True)[0, 1] / var_a)
               if var_a > 0 and n > 1 else None),
        corr=(float(np.corrcoef(a, f)[0, 1])
              if var_a > 0 and np.std(f) > 0 and n > 1 else None),
        sd_ratio=float(np.std(f) / np.std(a)) if var_a > 0 else None,
    )


def split_halves(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a zone's pairs into two halves by *target day*, earliest first.

    By day rather than by row so an unevenly covered day cannot put the same
    day on both sides of the split, which would make the two halves share
    information and flatter the stability test.
    """
    days = sorted(pairs["target_ts"].dt.normalize().unique())
    if len(days) < 2:
        return pairs.iloc[:0], pairs.iloc[:0]
    cut = days[len(days) // 2]
    d = pairs["target_ts"].dt.normalize()
    return pairs[d < cut], pairs[d >= cut]


def qualify(country: str, pairs: pd.DataFrame,
            thresholds: Thresholds | None = None) -> ZoneDecision:
    """Decide whether this zone gets an intercept, and what it is.

    Every refusal carries the number that produced it, because "left alone" and
    "measured and left alone" are different claims and only the second one is
    evidence.
    """
    th = thresholds or Thresholds()
    df = pairs.dropna(subset=["forecast_value", "actual"])
    if df.empty:
        return _identity(country, "no scored pairs in the fitting window")

    stats = measure(df)
    if stats.n < th.min_pairs or stats.n_target_days < th.min_target_days:
        return _identity(country,
                         f"insufficient fitting data ({stats.n} pairs over "
                         f"{stats.n_target_days} target days; need "
                         f"{th.min_pairs}/{th.min_target_days})", window=stats)

    # ABL-31: a zone whose whole forecast series sits inside 1 MW had a
    # zero-filled context. An offset would manufacture a level out of nothing.
    if float(np.nanmax(np.abs(df["forecast_value"].to_numpy(dtype=float)))) \
            <= DEGENERATE_MAX_ABS_MW:
        return _identity(country,
                         f"degenerate forecast series (all within "
                         f"{DEGENERATE_MAX_ABS_MW:g} MW -- ABL-31 zero-filled "
                         f"context)", window=stats)

    h1_df, h2_df = split_halves(df)
    h1, h2 = measure(h1_df), measure(h2_df)

    bias = stats.bias_mw
    tests: dict = {}

    tests["material"] = {
        "pass": bool(stats.bias_frac_pct is not None
                     and stats.bias_frac_pct >= 100.0 * th.min_bias_frac),
        "detail": (f"|bias| {abs(bias):,.0f} MW = {stats.bias_frac_pct:.1f}% of "
                   f"mean |net position| {stats.mean_abs_actual_mw:,.0f} MW "
                   f"(need >= {100 * th.min_bias_frac:.0f}%)"),
    }

    same_sign = bool(h1.bias_mw is not None and h2.bias_mw is not None
                     and np.sign(h1.bias_mw) == np.sign(bias)
                     and np.sign(h2.bias_mw) == np.sign(bias))
    tests["sign_agrees_across_halves"] = {
        "pass": same_sign,
        "detail": (f"half 1 {h1.bias_mw:+,.0f} MW / half 2 {h2.bias_mw:+,.0f} MW "
                   f"vs pooled {bias:+,.0f} MW"
                   if h1.bias_mw is not None and h2.bias_mw is not None
                   else "a half has no scored pairs"),
    }

    half_ok = bool(
        h1.bias_mw is not None and h2.bias_mw is not None
        and min(abs(h1.bias_mw), abs(h2.bias_mw)) >= th.min_half_share * abs(bias)
        and h1.bias_frac_pct is not None and h2.bias_frac_pct is not None
        and h1.bias_frac_pct >= 100.0 * th.min_bias_frac
        and h2.bias_frac_pct >= 100.0 * th.min_bias_frac)
    tests["magnitude_agrees_across_halves"] = {
        "pass": half_ok,
        "detail": (f"halves {abs(h1.bias_mw):,.0f} / {abs(h2.bias_mw):,.0f} MW "
                   f"({h1.bias_frac_pct:.1f}% / {h2.bias_frac_pct:.1f}% of their own "
                   f"mean |net position|); need both >= "
                   f"{th.min_half_share:.0%} of {abs(bias):,.0f} MW and >= "
                   f"{100 * th.min_bias_frac:.0f}%"
                   if h1.bias_frac_pct is not None and h2.bias_frac_pct is not None
                   else "a half has no scored pairs"),
    }

    t = stats.t_stat
    tests["separated_from_zero"] = {
        "pass": bool(t is not None and abs(t) >= th.min_abs_t),
        "detail": (f"t = {t:+.2f} on {stats.n_target_days} target-day means "
                   f"(s.e. {stats.day_se_mw:,.0f} MW; 95% CI "
                   f"[{stats.ci_lo_mw:,.0f}, {stats.ci_hi_mw:,.0f}] MW); "
                   f"need |t| >= {th.min_abs_t:.1f}, break-even |t| = "
                   f"{BREAK_EVEN_ABS_T:.0f}"
                   if t is not None else "day-level s.e. not estimable"),
    }

    failed = sorted(k for k, v in tests.items() if not v["pass"])
    if failed:
        return _identity(
            country,
            "failed " + ", ".join(failed) + " -- "
            + "; ".join(tests[k]["detail"] for k in failed),
            tests=tests, window=stats, half1=h1, half2=h2)

    return ZoneDecision(country=country, applied=True, intercept_mw=float(bias),
                        reason=(f"static bias {bias:+,.0f} MW = "
                                f"{stats.bias_frac_pct:.1f}% of mean |net position|, "
                                f"one-signed across both halves "
                                f"({h1.bias_mw:+,.0f} / {h2.bias_mw:+,.0f} MW), "
                                f"t = {t:+.2f} on {stats.n_target_days} target days"),
                        tests=tests, window=stats, half1=h1, half2=h2)


def fit_static_bias(pairs: pd.DataFrame, countries=None,
                    thresholds: Thresholds | None = None) -> dict[str, ZoneDecision]:
    """Qualify and fit every zone. `pairs` must hold only fitting-window rows.

    Serve-faithfulness is the caller's window, not a flag here: this function
    has no access to anything outside the frame it is handed, so it cannot read
    a row the fit was not entitled to see.
    """
    out: dict[str, ZoneDecision] = {}
    names = sorted(countries) if countries is not None \
        else sorted(pairs["country_code"].unique())
    for cc in names:
        out[cc] = qualify(cc, pairs[pairs["country_code"] == cc], thresholds)
    return out


def apply_static_bias(forecast, decision: ZoneDecision):
    """`forecast - intercept`, or the forecast unchanged for an unqualified zone.

    Intercept-only is enforced here structurally: there is no coefficient to
    multiply by. A zone left alone gets a copy, not a scaled series, so the
    corrected model is literally the champion wherever the test did not fire.
    """
    f = np.asarray(forecast, dtype=float)
    if decision.is_identity:
        return f.copy()
    return f - decision.intercept_mw


def apply_to_frame(pairs: pd.DataFrame,
                   decisions: dict[str, ZoneDecision]) -> pd.DataFrame:
    """Add a `corrected` column to a pairs frame, zone by zone.

    A zone with no decision is passed through uncorrected rather than dropped:
    a correction layer must never be able to silently lose a zone the champion
    forecast.
    """
    out = pairs.copy()
    corrected = np.empty(len(out), dtype=float)
    corrected[:] = np.nan
    values = out["forecast_value"].to_numpy(dtype=float)
    for cc, idx in out.groupby("country_code").indices.items():
        dec = decisions.get(cc)
        corrected[idx] = (values[idx] if dec is None
                          else apply_static_bias(values[idx], dec))
    out["corrected"] = corrected
    return out


def level_drift_diagnostic(h1: WindowStats, h2: WindowStats,
                           window: WindowStats) -> dict:
    """How much of the half-to-half bias change is amplitude shrinkage.

    With a forecast that regresses on the actual as `f = a0 + slope*a`, the mean
    error over any window is `a0 - (1 - slope) * mean(actual)`. So a zone whose
    slope sits well below 1 -- every net-position zone (ABL-595: 16 of 19 below
    0.8, several below 0.3) -- produces a *window-dependent* bias whenever the
    mean level moves, with no static offset present at all.

    This is the discriminator between "this zone has an offset" and "this zone
    is shrunk and the level moved", and the second is not correctable by an
    intercept: the intercept fitted on one window is the negative of the level
    drift into the next.
    """
    if (h1.bias_mw is None or h2.bias_mw is None or window.slope is None
            or h1.mean_actual_mw is None or h2.mean_actual_mw is None):
        return {"measurable": False}
    observed = h2.bias_mw - h1.bias_mw
    level_shift = h2.mean_actual_mw - h1.mean_actual_mw
    predicted = -(1.0 - window.slope) * level_shift
    return {
        "measurable": True,
        "observed_bias_change_mw": observed,
        "mean_actual_shift_mw": level_shift,
        "predicted_from_shrinkage_mw": predicted,
        "explained_frac": (predicted / observed) if observed != 0 else None,
        "slope": window.slope,
    }
