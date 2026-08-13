"""ABL-65 — residual/bias correction shapes for the net-position champion.

This module exists to answer one question with a measurement rather than an
argument: **can anything computable from past residuals lower the champion's
error at the lead it actually serves?**

The whole design hinges on one number the issue's premise does not contain.
The champion is a D+2 product: the run fires at 06:00Z on day D, net position
is day-ahead published so actuals reach D 21:00, and the target hours are
D+2 00:00-23:00. So the freshest residual any correction may read is
**27 to 50 hours older than the hour it is correcting** (`SERVE_LEADS_H`).
The reported residual AR lag-1 of 0.75-0.97 is measured between *adjacent*
hours and does not survive that gap — see `reports/abl_65_net_position_correction.md`
for the measured decay. Every shape here is therefore built from quantities
that are still informative across a two-day gap, or is included precisely so
that its failure is on the record.

Serve-faithfulness is structural, not a convention someone has to remember.
`estimate_error` never receives a history frame; the driver
(`backtest_corrections`) slices history to `target_ts < as_of` and to vintages
generated strictly before the one being corrected, and `test_residual_correction.py`
asserts that a future-dated row changes nothing. A correction that peeks is
worthless (ABL-65), so the peek is made impossible rather than audited.

Two guards, both in the spirit of "when the data cannot support a number,
render nothing":

* **Thin history is an identity, not a small correction.** Below
  `MIN_HISTORY_DAYS` distinct target days the shape returns zeros with a
  reason. A cold-started rail must serve the champion unchanged.
* **A degenerate vintage is never corrected** (ABL-31). If a vintage's whole
  forecast series sits inside `DEGENERATE_MAX_ABS_MW`, its context was
  zero-filled and the series carries no signal; adding a fitted offset to it
  manufactures a plausible-looking series out of a known-empty one. That is
  the exact defect ABL-31/ABL-35 cost a dedicated fix to remove, and this
  layer must not reintroduce it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

# A 06:00Z run on day D sees actuals through D 21:00; its targets are
# D+2 00:00-23:00. Both ends are load-bearing and are asserted in the tests.
SERVE_LEADS_H = (27, 50)

# Below this many distinct target days of observed residual, every shape is an
# identity. Three days is deliberately permissive — the point is to exclude a
# literal cold start, not to impose the 30-day floor V016's affine fit needs,
# because a rolling mean is one parameter, not three.
MIN_HISTORY_DAYS = 3

# Per hour-of-day bucket, for the diurnal shape. 24 buckets fitted on one week
# is 7 observations each; below 3 the bucket falls back to the grand mean.
MIN_OBS_PER_HOUR = 3

# ABL-31/ABL-35: a whole series inside 1 MW is a zero-filled context, not a
# balanced zone. Same floor the dashboard's `classifyActualSeries` uses.
DEGENERATE_MAX_ABS_MW = 1.0

# Every shape this module knows. Checked before anything else runs, so a typo is
# a crash rather than a silent identity carrying a nonsense reason.
KINDS = frozenset({"identity", "offset", "diurnal", "level_ar", "lead_ar"})


@dataclass(frozen=True)
class CorrectionSpec:
    """One correction shape. `kind` selects the estimator; the rest are its knobs."""
    name: str
    kind: str                     # identity | offset | diurnal | level_ar | lead_ar
    window_days: int = 7          # rolling history window (offset, diurnal, lead_ar)
    shrink: float = 1.0           # diurnal: weight on the per-hour mean vs grand mean
    damping: float | None = None  # level_ar: fixed phi; None = fit from history
    min_history_days: int = MIN_HISTORY_DAYS

    def describe(self) -> str:
        if self.kind == "identity":
            return "champion unchanged"
        if self.kind == "offset":
            return f"rolling constant offset over the last {self.window_days} available days"
        if self.kind == "diurnal":
            return (f"rolling hour-of-day offset table over {self.window_days} days"
                    + (f", shrunk {self.shrink:.2f} toward the grand mean"
                       if self.shrink < 1.0 else ""))
        if self.kind == "level_ar":
            return ("damped level update from the most recent fully observed day"
                    + (f" (phi={self.damping:.2f} fixed)" if self.damping is not None
                       else " (phi fitted on history)"))
        if self.kind == "lead_ar":
            return (f"lead-aware AR: the last observed residual hour scaled by the "
                    f"residual autocorrelation measured at each target's real lead, "
                    f"over {self.window_days} days")
        raise ValueError(f"unknown correction kind {self.kind!r}")


@dataclass(frozen=True)
class CorrectionResult:
    """The estimated error to subtract, plus why it is what it is."""
    estimate_mw: np.ndarray
    applied: bool
    reason: str

    @property
    def is_identity(self) -> bool:
        return not self.applied


def _identity(n: int, reason: str) -> CorrectionResult:
    return CorrectionResult(np.zeros(n, dtype=float), False, reason)


def _window(history: pd.DataFrame, as_of: pd.Timestamp, days: int) -> pd.DataFrame:
    """History rows inside the rolling window. `history` is already < as_of."""
    if days <= 0:
        return history
    return history[history["target_ts"] >= pd.Timestamp(as_of) - pd.Timedelta(days=days)]


def estimate_error(history: pd.DataFrame, targets: pd.DatetimeIndex,
                   as_of: pd.Timestamp, spec: CorrectionSpec,
                   forecast: np.ndarray | None = None) -> CorrectionResult:
    """Estimated error (forecast - actual) for `targets`, from `history` only.

    `history` needs columns `target_ts` and `err` (= forecast - actual) and MUST
    already be restricted to hours the run could observe. This function does not
    re-check that, on purpose: the restriction belongs to the driver, where the
    vintage's `as_of` is known, and duplicating it here would let the two drift.

    The returned estimate is *subtracted* from the champion, so a positive
    estimate means "this forecast has been running high".
    """
    targets = pd.DatetimeIndex(targets)
    n = len(targets)

    # Validate the shape before any early return. A misspelled `kind` that fell
    # through to an identity would report "no correction applied" for a reason
    # that is a typo — the same silently-not-a-failure shape ABL-72 found in the
    # gate's missing criteria.
    if spec.kind not in KINDS:
        raise ValueError(f"unknown correction kind {spec.kind!r}; expected one of "
                         f"{', '.join(sorted(KINDS))}")

    if spec.kind == "identity":
        return _identity(n, "identity shape")

    # ABL-31: never dress a zero-filled context in a fitted level.
    if forecast is not None and len(forecast):
        if float(np.nanmax(np.abs(np.asarray(forecast, dtype=float)))) <= DEGENERATE_MAX_ABS_MW:
            return _identity(n, f"degenerate vintage: whole forecast series within "
                                f"{DEGENERATE_MAX_ABS_MW:g} MW (ABL-31 zero-filled context)")

    hist = history.dropna(subset=["err"])
    if hist.empty:
        return _identity(n, "no observed residual history")
    n_days = int(hist["target_ts"].dt.normalize().nunique())
    if n_days < spec.min_history_days:
        return _identity(n, f"insufficient history ({n_days} target days; "
                            f"need {spec.min_history_days})")

    if spec.kind == "offset":
        w = _window(hist, as_of, spec.window_days)
        if w.empty:
            return _identity(n, "rolling window holds no observed residual")
        return CorrectionResult(np.full(n, float(w["err"].mean())), True,
                                f"offset from {len(w)} hours over "
                                f"{w['target_ts'].dt.normalize().nunique()} days")

    if spec.kind == "diurnal":
        w = _window(hist, as_of, spec.window_days)
        if w.empty:
            return _identity(n, "rolling window holds no observed residual")
        grand = float(w["err"].mean())
        by_hour = w.groupby(w["target_ts"].dt.hour)["err"].agg(["mean", "size"])
        usable = by_hour[by_hour["size"] >= MIN_OBS_PER_HOUR]["mean"]
        # Shrinkage toward the grand mean is what keeps 24 parameters honest on a
        # week of data; an hour with too few observations falls back entirely.
        profile = spec.shrink * usable + (1.0 - spec.shrink) * grand
        est = profile.reindex(targets.hour).to_numpy(dtype=float)
        est = np.where(np.isnan(est), grand, est)
        return CorrectionResult(est, True,
                                f"hour-of-day table from {len(w)} hours, "
                                f"{len(usable)}/24 hours estimated")

    if spec.kind == "level_ar":
        # The most recent *fully observed* day. Day D is complete through 21:00
        # at as_of = D 22:00, which is what `last_day` picks up.
        last_day = hist["target_ts"].dt.normalize().max()
        recent = hist[hist["target_ts"].dt.normalize() == last_day]
        if recent.empty:
            return _identity(n, "no fully observed recent day")
        level = float(recent["err"].mean())
        phi = spec.damping if spec.damping is not None else _fit_day_level_phi(hist)
        if phi is None:
            return _identity(n, "day-level persistence not estimable from history")
        return CorrectionResult(np.full(n, phi * level), True,
                                f"level {level:,.0f} MW from {last_day.date()} "
                                f"damped by phi={phi:.3f}")

    if spec.kind == "lead_ar":
        w = _window(hist, as_of, spec.window_days)
        chain = (w.sort_values("target_ts").drop_duplicates("target_ts", keep="last")
                  .set_index("target_ts")["err"].asfreq("h"))
        if chain.dropna().empty:
            return _identity(n, "rolling window holds no observed residual")
        last_ts = chain.dropna().index.max()
        last_err = float(chain.loc[last_ts])
        leads = ((targets - last_ts) / pd.Timedelta(hours=1)).to_numpy(dtype=float)
        rho = _autocorr_at(chain, leads)
        # A residual dated after the target would be information the run did not
        # have. Refuse rather than carry it backwards (mirrors V016).
        est = np.where(leads > 0, rho * last_err, 0.0)
        return CorrectionResult(est, True,
                                f"last residual {last_err:,.0f} MW at {last_ts}, "
                                f"leads {leads.min():.0f}-{leads.max():.0f}h, "
                                f"rho {np.nanmin(rho):.3f}..{np.nanmax(rho):.3f}")

    raise ValueError(f"unknown correction kind {spec.kind!r}")


def _fit_day_level_phi(hist: pd.DataFrame, max_days: int = 60) -> float | None:
    """OLS slope of a target day's mean residual on the day-two-days-earlier's.

    Two days, not one, because that is the real gap: the freshest fully observed
    day at as_of is the target day minus two. Fitting lag-1 and serving lag-2
    would overstate what the term can carry, which is the same mistake as
    quoting hourly AR lag-1 for a D+2 product.
    """
    daily = hist.groupby(hist["target_ts"].dt.normalize())["err"].mean()
    daily = daily.tail(max_days)
    pairs = pd.concat([daily, daily.shift(2)], axis=1).dropna()
    if len(pairs) < 10:
        return None
    y, x = pairs.iloc[:, 0].to_numpy(), pairs.iloc[:, 1].to_numpy()
    var = float(np.var(x))
    if var <= 0:
        return None
    phi = float(np.cov(x, y, bias=True)[0, 1] / var)
    # A negative or explosive coefficient extrapolated two days out is not a
    # stable level update; clip to the range a damping term can defend.
    return float(np.clip(phi, 0.0, 1.0))


def _autocorr_at(chain: pd.Series, leads: np.ndarray) -> np.ndarray:
    """Residual autocorrelation measured at each requested lead.

    This is the honest replacement for `phi ** lead`. An AR(1) carried 27-50
    hours assumes a decay the process does not follow: measured on 198 target
    days, `phi ** 27` and the real lag-27 correlation disagree by up to 3.4x in
    both directions, and at lag 48 the AR(1) prediction is positive where the
    measurement is zero or negative.
    """
    out = np.zeros(len(leads), dtype=float)
    cache: dict[int, float] = {}
    for i, lead in enumerate(leads):
        if not np.isfinite(lead) or lead <= 0:
            continue
        k = int(round(lead))
        if k not in cache:
            pair = pd.concat([chain, chain.shift(k)], axis=1).dropna()
            v = pair.corr().iloc[0, 1] if len(pair) >= 48 else np.nan
            cache[k] = float(v) if np.isfinite(v) else 0.0
        out[i] = cache[k]
    return out


# ---------------------------------------------------------------------------
# Serve-faithful backtest driver
# ---------------------------------------------------------------------------

def _as_of(generated_at: pd.Timestamp) -> pd.Timestamp:
    """Publication cutoff for a vintage — imported behaviour, restated locally.

    Kept identical to `net_position.as_of_for_vintage`; the test asserts the two
    agree, so this module cannot drift into a more generous cutoff than the one
    the report's baselines are built on.
    """
    from .net_position import as_of_for_vintage
    return as_of_for_vintage(generated_at)


def backtest_corrections(pairs: pd.DataFrame, specs: list[CorrectionSpec],
                         score_from: pd.Timestamp | None = None) -> pd.DataFrame:
    """Apply every spec to every vintage, reading only pre-`as_of` residuals.

    `pairs` needs `country_code`, `generated_at`, `target_ts`, `forecast_value`,
    `actual`. Returns one row per (country, vintage, target hour, spec) with the
    corrected forecast, so the caller scores whatever it likes downstream.

    `score_from` restricts which vintages are *scored*; history before it is
    still read. That split is the point — a correction deployed on day X has the
    history of day X-1, and pretending otherwise either invents a cold start or
    hides one.
    """
    required = {"country_code", "generated_at", "target_ts", "forecast_value", "actual"}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"pairs is missing {sorted(missing)}")

    df = pairs.dropna(subset=["actual"]).copy()
    df["err"] = df["forecast_value"] - df["actual"]
    out = []

    for country, cg in df.groupby("country_code", sort=True):
        cg = cg.sort_values(["generated_at", "target_ts"])
        for gen, vg in cg.groupby("generated_at", sort=True):
            if score_from is not None and gen < pd.Timestamp(score_from):
                continue
            as_of = _as_of(pd.Timestamp(gen))
            # Serve-faithful slice: hours already published AND produced by a
            # vintage that had already run. Both conditions, not either.
            hist = cg[(cg["target_ts"] < as_of) & (cg["generated_at"] < gen)]
            # When several past vintages covered one hour, the operational view
            # is the most recent one that had run by now.
            hist = (hist.sort_values("generated_at")
                        .drop_duplicates("target_ts", keep="last")
                        .sort_values("target_ts")[["target_ts", "err"]])
            targets = pd.DatetimeIndex(vg["target_ts"])
            fc = vg["forecast_value"].to_numpy(dtype=float)
            for spec in specs:
                res = estimate_error(hist, targets, as_of, spec, forecast=fc)
                out.append(pd.DataFrame({
                    "country_code": country,
                    "generated_at": gen,
                    "target_ts": targets,
                    "spec": spec.name,
                    "actual": vg["actual"].to_numpy(dtype=float),
                    "forecast_value": fc,
                    "corrected": fc - res.estimate_mw,
                    "estimate_mw": res.estimate_mw,
                    "applied": res.applied,
                    "reason": res.reason,
                    "history_hours": len(hist),
                }))
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


def score_corrections(scored: pd.DataFrame, baselines: pd.DataFrame | None = None
                      ) -> pd.DataFrame:
    """Per (country, spec) MAE/RMSE/bias, and skill against the ensemble.

    `baselines` optionally carries `country_code`, `generated_at`, `target_ts`
    and `baseline_ensemble` — the serve-faithful persistence+climatology mean
    the gate reads. Skill is computed only over rows where the ensemble exists,
    so a country whose baseline is missing reads as unmeasured rather than as a
    win.
    """
    df = scored
    if baselines is not None:
        df = df.merge(baselines, on=["country_code", "generated_at", "target_ts"],
                      how="left")
    rows = []
    for (country, spec), g in df.groupby(["country_code", "spec"]):
        err = g["corrected"] - g["actual"]
        base_err = g["forecast_value"] - g["actual"]
        row = {
            "country_code": country, "spec": spec, "n": int(len(g)),
            "mae_mw": float(np.mean(np.abs(err))),
            "rmse_mw": float(np.sqrt(np.mean(err ** 2))),
            "bias_mw": float(np.mean(err)),
            "uncorrected_mae_mw": float(np.mean(np.abs(base_err))),
            "applied_frac": float(g["applied"].mean()),
            "mean_abs_estimate_mw": float(np.mean(np.abs(g["estimate_mw"]))),
        }
        row["mae_delta_pct"] = (100.0 * (1.0 - row["mae_mw"] / row["uncorrected_mae_mw"])
                                if row["uncorrected_mae_mw"] > 0 else None)
        if "baseline_ensemble" in g.columns:
            sub = g.dropna(subset=["baseline_ensemble"])
            if len(sub):
                bm = float(np.mean(np.abs(sub["baseline_ensemble"] - sub["actual"])))
                row["ensemble_mae_mw"] = bm
                row["n_vs_ensemble"] = int(len(sub))
                cm = float(np.mean(np.abs(sub["corrected"] - sub["actual"])))
                um = float(np.mean(np.abs(sub["forecast_value"] - sub["actual"])))
                row["skill_vs_ensemble_pct"] = 100.0 * (1.0 - cm / bm) if bm > 0 else None
                row["uncorrected_skill_vs_ensemble_pct"] = (
                    100.0 * (1.0 - um / bm) if bm > 0 else None)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["country_code", "spec"]).reset_index(drop=True)


def default_specs() -> list[CorrectionSpec]:
    """The shapes ABL-65 weighs, each present for a stated reason.

    `lead_ar_7d` is included even though the measured lead-27..50 autocorrelation
    predicts it can do almost nothing: it is this issue's headline candidate, and
    a candidate refuted by measurement has to be measured, not argued away.
    """
    return [
        CorrectionSpec("uncorrected", "identity"),
        CorrectionSpec("offset_3d", "offset", window_days=3),
        CorrectionSpec("offset_7d", "offset", window_days=7),
        CorrectionSpec("offset_14d", "offset", window_days=14),
        CorrectionSpec("offset_28d", "offset", window_days=28),
        CorrectionSpec("diurnal_7d", "diurnal", window_days=7),
        CorrectionSpec("diurnal_14d", "diurnal", window_days=14),
        CorrectionSpec("diurnal_28d_shrunk", "diurnal", window_days=28, shrink=0.5),
        CorrectionSpec("level_ar_fitted", "level_ar"),
        CorrectionSpec("level_ar_phi05", "level_ar", damping=0.5),
        CorrectionSpec("lead_ar_7d", "lead_ar", window_days=7),
        CorrectionSpec("lead_ar_28d", "lead_ar", window_days=28),
    ]


def with_window(spec: CorrectionSpec, days: int) -> CorrectionSpec:
    return replace(spec, name=f"{spec.name}_{days}d", window_days=days)
