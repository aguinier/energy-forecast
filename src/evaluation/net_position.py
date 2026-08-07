"""Durable evaluation for the net-position forecast (ABL-30, Phase B).

Every future net-position model claim goes through this module: it joins the
sidecar's as-served vintages (plus the prod-pushed copies in the replica) to
`net_position` actuals and scores them per country, with the amplitude checks
(regression slope, sd-ratio) that caught ABL-24's shrinkage, serve-faithful
baselines, quantile calibration, an error decomposition, and the pre-registered
promotion gate from the ABL-24 plan (C3).

Conventions this module commits to — change them only with a matching change
in the report footer, because published numbers cite them:

- **bias = mean(forecast - actual)** (positive = over-forecast), matching
  `src/evaluation/metrics.py`.
- **slope** is the OLS slope of *forecast on actual* — the convention of the
  ABL-24/ABL-28 measurements (a calibrated forecast has slope ~1; shrinkage
  toward zero reads as slope < 1). Pooled slope mixes country means and is
  inflated by between-country variance; the per-country numbers are the ones
  the gate reads.
- **Every vintage-target pair counts.** When two vintages cover the same
  target hour, both pairs score (matching ABL-24/ABL-28's 6,274-pair method).
  The AR diagnostics use the latest vintage only, which is the one view that
  needs an unduplicated hourly chain.
- **Serve-faithful as_of**: `net_position` for day X is published day-ahead
  (~12:45 CET on X-1), so a run at hour G on day D could see actuals through
  D 21:00 UTC if G < 11:00 UTC, else through D+1 21:00. Baselines only read
  actuals at or before that cutoff. (See ABL-28: `as_of = D 22:00`.)
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

# When the context-cutoff fix (energy-forecast 1c5a24f) started producing the
# live vintages: committed 2026-08-04 16:29 +0200 = 14:29 UTC, between the
# 08-04 and 08-05 scheduled runs. Vintages generated before this ran the
# zero-padded context (ABL-28's 0b defect).
FIX_DEPLOYED_UTC = pd.Timestamp("2026-08-04 14:29:00")

# Promotion gate (pre-registered in the ABL-24 plan, C3).
GATE_SLOPE_RANGE = (0.8, 1.2)
GATE_COVERAGE_RANGE = (75.0, 85.0)  # % of actuals inside the 10-90 band
GATE_BIAS_FRAC = 0.05               # |bias| < 5% of mean |net position|
GATE_BASELINE_COUNTRY_FRAC = 0.80   # beat ensemble MAE in >= 80% of countries
GATE_MIN_LIVE_VINTAGES = 14         # plan Rev 3:54 — live shadow vintages in the window

# Zones the gate is pre-registered to exclude (plan Rev 3:55), each with its
# reason. Excluding by *name* rather than by symptom is the whole point: GR is
# excluded today only as a side-effect of having zero paired actuals, so the
# moment GR actuals partially resume it silently re-enters the gate and fails
# it on thin data. The reasons are reported, so an exclusion is never silent.
GATE_EXCLUDED_COUNTRIES = {
    "LU": "not an independent bidding zone in the A25 day-ahead net position — "
          "the row duplicates DE, so scoring it double-counts DE",
    "GR": "actuals are fabricated exact zeros, not measurements (ABL-35: every "
          "published row since 2025-10-01 is 0.0 while GR moved a median "
          "1,142 MW across its borders); row deletion pending on ABL-67",
}

# The eight criteria pre-registered in the ABL-24 plan Rev 3 §4. The gate emits
# exactly these names and checks itself against this tuple, so a criterion
# cannot go missing the way `min_live_shadow_vintages` and `excluded_zones_LU_GR`
# did — silently absent, and therefore silently not a failure (ABL-72).
PRE_REGISTERED_CHECKS = (
    "min_live_shadow_vintages",
    "excluded_zones_LU_GR",
    "beat_baseline_ensemble_80pct",
    "bias_under_5pct_per_country",
    "slope_in_range_per_country",
    "coverage_10_90_in_band_per_country",
    "no_regression_W01_W12",
    "serve_faithful_inputs_verified",
)


@dataclass
class EvalConfig:
    replica_db: str
    sidecar_db: str | None = None
    model_name: str = "chronos-2-V010"
    start: str | None = None            # target-timestamp window (UTC)
    end: str | None = None
    cohort_split: pd.Timestamp = FIX_DEPLOYED_UTC
    climatology_days: int = 28
    top_misses: int = 10
    candidate_backtest: str | None = None   # candidate's W01-W12 JSON
    reference_backtest: str | None = None   # reference W01-W12 JSON (V010 serve-faithful)
    serve_faithful_verified: bool = False   # manual attestation, never inferred
    quantiles: tuple = field(default=QUANTILE_LEVELS)
    # Vintage window the *gate* scores over (generated_at, UTC). `None` start
    # means `cohort_split`, which is the honest default: scoring the champion
    # over every stored vintage measures 94% pre-fix data and hands a challenger
    # a 2.60x MAE handicap in its favour (ABL-72 G1). The full report still
    # covers every vintage — only the gate is restricted, and the restriction
    # must be identical across the models of one comparison.
    gate_vintage_start: pd.Timestamp | None = None
    gate_vintage_end: pd.Timestamp | None = None


def resolve_gate_vintage_window(cfg: EvalConfig) -> tuple[pd.Timestamp, pd.Timestamp | None]:
    """The [start, end) generated_at window the gate scores over."""
    start = cfg.gate_vintage_start
    return (cfg.cohort_split if start is None else pd.Timestamp(start),
            None if cfg.gate_vintage_end is None else pd.Timestamp(cfg.gate_vintage_end))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _ro_connect(path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _parse_ts(series: pd.Series) -> pd.Series:
    # Both separator forms exist in these columns elsewhere in the database;
    # parse rather than string-compare so this module never inherits ABL-21.
    return pd.to_datetime(series, format="mixed")


def load_forecasts(cfg: EvalConfig) -> pd.DataFrame:
    """As-served vintages: sidecar plus prod-pushed rows in the replica.

    The sidecar is the authoritative as-served record (the push mirrors it),
    so on overlap the sidecar row wins; the replica fills anything the push
    created that the sidecar no longer holds. Returns one row per
    (country, target_ts, generated_at) with a `source` tag, plus quantile
    columns q10..q90 where stored.
    """
    frames = []
    sources = [("replica", cfg.replica_db)]
    if cfg.sidecar_db:
        sources.insert(0, ("sidecar", cfg.sidecar_db))
    for source, path in sources:
        con = _ro_connect(path)
        try:
            df = pd.read_sql_query(
                """SELECT country_code, target_timestamp_utc, generated_at,
                          horizon_hours, forecast_value
                   FROM forecasts
                   WHERE forecast_type = 'net_position' AND model_name = ?""",
                con, params=(cfg.model_name,))
            qf = pd.read_sql_query(
                """SELECT country_code, target_timestamp_utc, generated_at,
                          quantile, forecast_value
                   FROM forecast_quantiles
                   WHERE forecast_type = 'net_position' AND model_name = ?""",
                con, params=(cfg.model_name,))
        finally:
            con.close()
        if df.empty:
            continue
        df["target_ts"] = _parse_ts(df["target_timestamp_utc"])
        df["generated_at"] = _parse_ts(df["generated_at"])
        df["source"] = source
        df = df.drop(columns=["target_timestamp_utc"])
        if not qf.empty:
            qf["target_ts"] = _parse_ts(qf["target_timestamp_utc"])
            qf["generated_at"] = _parse_ts(qf["generated_at"])
            wide = qf.pivot_table(index=["country_code", "target_ts", "generated_at"],
                                  columns="quantile", values="forecast_value")
            wide.columns = [f"q{int(round(q * 100))}" for q in wide.columns]
            df = df.merge(wide.reset_index(),
                          on=["country_code", "target_ts", "generated_at"], how="left")
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    all_rows = pd.concat(frames, ignore_index=True)

    # Integrity: where both sources carry the same vintage row, the pushed
    # value must equal the sidecar value. Report, don't fix.
    key = ["country_code", "target_ts", "generated_at"]
    if len(frames) == 2:
        merged = frames[0].merge(frames[1], on=key, suffixes=("_side", "_repl"))
        max_diff = float((merged["forecast_value_side"]
                          - merged["forecast_value_repl"]).abs().max()) if len(merged) else 0.0
    else:
        max_diff = float("nan")
    deduped = all_rows.drop_duplicates(subset=key, keep="first")  # sidecar first
    deduped.attrs["source_counts"] = all_rows["source"].value_counts().to_dict()
    deduped.attrs["overlap_max_abs_diff_mw"] = max_diff

    if cfg.start:
        deduped = deduped[deduped["target_ts"] >= pd.Timestamp(cfg.start)]
    if cfg.end:
        deduped = deduped[deduped["target_ts"] < pd.Timestamp(cfg.end)]
    return deduped.reset_index(drop=True)


def load_actuals(cfg: EvalConfig, pad_days: int = 45) -> pd.DataFrame:
    """Hourly net_position actuals, padded back far enough for baselines."""
    con = _ro_connect(cfg.replica_db)
    try:
        df = pd.read_sql_query(
            """SELECT country_code, timestamp_utc, net_position_mw
               FROM net_position WHERE net_position_mw IS NOT NULL""", con)
    finally:
        con.close()
    df["ts"] = _parse_ts(df["timestamp_utc"])
    df = df.drop(columns=["timestamp_utc"]).rename(columns={"net_position_mw": "actual"})
    if cfg.start:
        df = df[df["ts"] >= pd.Timestamp(cfg.start) - pd.Timedelta(days=pad_days)]
    if cfg.end:
        df = df[df["ts"] < pd.Timestamp(cfg.end)]
    # Duplicate publications: keep the last row per (country, hour).
    df = (df.sort_values("ts").groupby(["country_code", df["ts"].dt.floor("h")])
            .tail(1).reset_index(drop=True))
    df["ts"] = df["ts"].dt.floor("h")
    return df


# ---------------------------------------------------------------------------
# Serve-faithful baselines
# ---------------------------------------------------------------------------

def as_of_for_vintage(generated_at: pd.Timestamp) -> pd.Timestamp:
    """Last actual hour (exclusive bound) available when this vintage ran."""
    day = generated_at.normalize()
    if generated_at.hour >= 11:  # day-ahead publication (~10:45 UTC) done
        day = day + pd.Timedelta(days=1)
    return day + pd.Timedelta(hours=22)


def baseline_predictions(actuals: pd.Series, as_of: pd.Timestamp,
                         targets: pd.DatetimeIndex, climatology_days: int = 28
                         ) -> pd.DataFrame:
    """Persistence (same hour, last available day) and hour-of-day climatology,
    using only actuals strictly before `as_of`. Index = targets."""
    hist = actuals[actuals.index < as_of].dropna()
    out = pd.DataFrame(index=targets, columns=["persistence", "climatology"],
                       dtype=float)
    if hist.empty:
        return out
    clim_hist = hist[hist.index >= as_of - pd.Timedelta(days=climatology_days)]
    by_hour_mean = clim_hist.groupby(clim_hist.index.hour).mean()
    by_hour_last = hist.groupby(hist.index.hour).last()
    hours = pd.Index(targets.hour)
    out["persistence"] = by_hour_last.reindex(hours).to_numpy()
    out["climatology"] = by_hour_mean.reindex(hours).to_numpy()
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def point_metrics(actual: np.ndarray, forecast: np.ndarray) -> dict:
    n = len(actual)
    if n == 0:
        return {"n": 0}
    err = forecast - actual
    sum_abs_a = float(np.sum(np.abs(actual)))
    var_a = float(np.var(actual))
    out = {
        "n": int(n),
        "bias_mw": float(np.mean(err)),
        "mae_mw": float(np.mean(np.abs(err))),
        "rmse_mw": float(np.sqrt(np.mean(err ** 2))),
        "mean_abs_actual_mw": sum_abs_a / n,
        "wape_pct": 100.0 * float(np.sum(np.abs(err))) / sum_abs_a if sum_abs_a > 0 else None,
        "slope": float(np.cov(actual, forecast, bias=True)[0, 1] / var_a) if var_a > 0 and n > 1 else None,
        "sd_ratio": float(np.std(forecast) / np.std(actual)) if var_a > 0 else None,
        "corr": float(np.corrcoef(actual, forecast)[0, 1]) if var_a > 0 and np.std(forecast) > 0 and n > 1 else None,
    }
    out["nmae"] = out["mae_mw"] / out["mean_abs_actual_mw"] if out["mean_abs_actual_mw"] > 0 else None
    return out


def quantile_metrics(df: pd.DataFrame, quantiles=QUANTILE_LEVELS) -> dict:
    """Pinball loss per stored quantile + 10-90 band coverage."""
    out = {"pinball_mw": {}, "coverage_10_90_pct": None, "n_with_band": 0}
    a = df["actual"].to_numpy()
    for q in quantiles:
        col = f"q{int(round(q * 100))}"
        if col not in df.columns:
            continue
        p = df[col].to_numpy()
        mask = ~np.isnan(p)
        if mask.sum() == 0:
            continue
        diff = a[mask] - p[mask]
        out["pinball_mw"][col] = float(np.mean(np.where(diff >= 0, q * diff, (q - 1) * diff)))
    if "q10" in df.columns and "q90" in df.columns:
        band = df.dropna(subset=["q10", "q90"])
        out["n_with_band"] = int(len(band))
        if len(band):
            inside = (band["actual"] >= band["q10"]) & (band["actual"] <= band["q90"])
            out["coverage_10_90_pct"] = 100.0 * float(inside.mean())
    return out


def decompose_error(actual: np.ndarray, forecast: np.ndarray,
                    hour_of_day: np.ndarray) -> dict:
    """Sequential MSE decomposition: static bias -> affine recalibration ->
    diurnal-systematic -> residual. Each stage's removal is the extra MSE a
    correction of that shape would recover; the fractions sum to 1."""
    n = len(actual)
    if n < 48:
        return {"n": int(n), "note": "too few pairs to decompose"}
    e0 = forecast - actual
    mse0 = float(np.mean(e0 ** 2))
    if mse0 == 0:
        return {"n": int(n), "mse_total": 0.0}
    # 1. static bias
    e1 = e0 - np.mean(e0)
    mse1 = float(np.mean(e1 ** 2))
    # 2. affine recalibration a' = alpha*f + beta (OLS of actual on forecast)
    var_f = float(np.var(forecast))
    if var_f > 0:
        alpha = float(np.cov(forecast, actual, bias=True)[0, 1] / var_f)
        beta = float(np.mean(actual) - alpha * np.mean(forecast))
        e2 = (alpha * forecast + beta) - actual
    else:
        e2 = e1
    mse2 = float(np.mean(e2 ** 2))
    # 3. hour-of-day systematic (profile of the affine-corrected error)
    prof = pd.Series(e2).groupby(pd.Series(hour_of_day)).transform("mean").to_numpy()
    e3 = e2 - prof
    mse3 = float(np.mean(e3 ** 2))
    return {
        "n": int(n),
        "mse_total": mse0,
        "frac_static_bias": (mse0 - mse1) / mse0,
        "frac_affine": max(0.0, mse1 - mse2) / mse0,
        "frac_diurnal": max(0.0, mse2 - mse3) / mse0,
        "frac_residual": mse3 / mse0,
        "affine_alpha": alpha if var_f > 0 else None,
        "affine_beta": beta if var_f > 0 else None,
    }


def residual_autocorr(chain: pd.DataFrame) -> dict:
    """Lag-1/lag-24 autocorrelation of the error on an hourly chain
    (latest-vintage-only rows, so no duplicated target hours)."""
    s = (chain.sort_values("target_ts")
              .drop_duplicates("target_ts", keep="last")
              .set_index("target_ts"))
    err = (s["forecast_value"] - s["actual"]).asfreq("h")
    out = {}
    for lag in (1, 24):
        pairs = pd.concat([err, err.shift(lag)], axis=1).dropna()
        out[f"lag{lag}"] = (float(pairs.corr().iloc[0, 1]) if len(pairs) >= 48 else None)
    return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _cut_metrics(df: pd.DataFrame, keys: list[str]) -> list[dict]:
    rows = []
    for vals, g in df.groupby(keys):
        vals = vals if isinstance(vals, tuple) else (vals,)
        m = point_metrics(g["actual"].to_numpy(), g["forecast_value"].to_numpy())
        rows.append({**dict(zip(keys, [str(v) for v in vals])),
                     "n": m["n"], "bias_mw": m.get("bias_mw"), "mae_mw": m.get("mae_mw")})
    return rows


def per_country_metrics(scored: pd.DataFrame, paired: pd.DataFrame,
                        cfg: EvalConfig) -> dict:
    """Per-country point metrics, quantiles, baseline skill, decomposition, AR.

    Called twice: once over every scored pair (the report) and once over the
    gate's vintage window (what actually gates). Keeping one implementation is
    the point — the gate and the report must not be able to disagree about
    what a country's MAE is.
    """
    out: dict = {}
    for country, g in scored.groupby("country_code"):
        a, f = g["actual"].to_numpy(), g["forecast_value"].to_numpy()
        m = point_metrics(a, f)
        m.update(quantile_metrics(g, cfg.quantiles))
        for name in ("persistence", "climatology", "baseline_ensemble"):
            sub = g.dropna(subset=[name])
            if len(sub):
                bm = float(np.mean(np.abs(sub[name] - sub["actual"])))
                m[f"{name}_mae_mw"] = bm
                m[f"skill_vs_{name}_pct"] = 100.0 * (1.0 - m["mae_mw"] / bm) if bm > 0 else None
        m["decomposition"] = decompose_error(a, f, g["hour"].to_numpy())
        m["residual_autocorr"] = residual_autocorr(g)
        out[country] = m
    # Countries with vintages but zero paired actuals (GR-shaped).
    for country in sorted(set(paired["country_code"]) - set(scored["country_code"])):
        n = int((paired["country_code"] == country).sum())
        out[country] = {"n": 0, "rows_unpaired": n, "coverage": "no_paired_actuals"}
    return out


def build_gate_scope(scored: pd.DataFrame, paired: pd.DataFrame,
                     cfg: EvalConfig) -> dict:
    """The restricted view the promotion gate scores (ABL-72 G1).

    `evaluate` scores every stored vintage, which is right for the report and
    wrong for the gate: 94% of the champion's stored vintages predate the
    context-cutoff fix (1c5a24f), so gating on them measures MAE 1,439 MW /
    slope 0.26 where the model actually serving is 553 MW / 0.90. A challenger
    compared against that would clear a bar 2.60x easier than the real one.

    So the gate reads its own window, defaulting to vintages at or after
    `cohort_split`, and the window plus its vintage count go in the report
    header — a restriction nobody can see is not a restriction.
    """
    start, end = resolve_gate_vintage_window(cfg)
    in_window = paired["generated_at"] >= start
    if end is not None:
        in_window &= paired["generated_at"] < end
    gate_paired = paired[in_window]
    gate_scored = scored[scored["generated_at"] >= start]
    if end is not None:
        gate_scored = gate_scored[gate_scored["generated_at"] < end]

    vintages = sorted(gate_paired["generated_at"].unique())
    return {
        "vintage_start": str(start),
        "vintage_end": str(end) if end is not None else None,
        "vintages": int(len(vintages)),
        "vintage_list": [str(v) for v in vintages],
        "pairs_scored": int(len(gate_scored)),
        "countries_measured": int(gate_scored["country_code"].nunique()),
        "per_country": per_country_metrics(gate_scored, gate_paired, cfg),
        "excluded_countries": dict(GATE_EXCLUDED_COUNTRIES),
    }


def evaluate(cfg: EvalConfig) -> dict:
    forecasts = load_forecasts(cfg)
    if forecasts.empty:
        return {"error": f"no '{cfg.model_name}' net_position forecasts found"}
    actuals = load_actuals(cfg)

    act_lookup = {c: g.set_index("ts")["actual"].sort_index()
                  for c, g in actuals.groupby("country_code")}

    # Pair every vintage row with its actual, and attach serve-faithful baselines.
    paired = forecasts.merge(
        actuals.rename(columns={"ts": "target_ts"}),
        on=["country_code", "target_ts"], how="left")
    base_cols = {"persistence": [], "climatology": []}
    empty_hist = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    for (country, gen), g in paired.groupby(["country_code", "generated_at"]):
        hist = act_lookup.get(country, empty_hist)
        preds = baseline_predictions(hist, as_of_for_vintage(gen),
                                     pd.DatetimeIndex(g["target_ts"]),
                                     cfg.climatology_days)
        for k in base_cols:
            base_cols[k].append(pd.Series(preds[k].to_numpy(), index=g.index))
    for k, chunks in base_cols.items():
        paired[k] = pd.concat(chunks).sort_index()
    paired["baseline_ensemble"] = paired[["persistence", "climatology"]].mean(axis=1)

    scored = paired.dropna(subset=["actual"]).copy()
    scored["hour"] = scored["target_ts"].dt.hour
    scored["weekday"] = scored["target_ts"].dt.dayofweek
    scored["horizon_bucket"] = (scored["horizon_hours"] // 12 * 12).astype(int).map(
        lambda h: f"{h:02d}-{h + 11:02d}h")
    scored["cohort"] = np.where(scored["generated_at"] < cfg.cohort_split,
                                "pre_fix", "post_fix")

    results: dict = {
        "meta": {
            "model": cfg.model_name,
            "replica_db": str(cfg.replica_db),
            "sidecar_db": str(cfg.sidecar_db),
            "target_window": [str(scored["target_ts"].min()), str(scored["target_ts"].max())] if len(scored) else None,
            "vintages": int(forecasts["generated_at"].nunique()),
            "vintage_span": [str(forecasts["generated_at"].min()), str(forecasts["generated_at"].max())],
            "pairs_scored": int(len(scored)),
            "rows_unpaired": int(len(paired) - len(scored)),
            "cohort_split_utc": str(cfg.cohort_split),
            "source_counts": forecasts.attrs.get("source_counts"),
            "sidecar_vs_pushed_max_abs_diff_mw": forecasts.attrs.get("overlap_max_abs_diff_mw"),
            "actuals_max_ts": str(actuals["ts"].max()) if len(actuals) else None,
        },
        "per_country": {}, "pooled": {}, "cohorts": {}, "per_vintage": [],
        "cuts": {}, "case_studies": [], "gate_scope": {}, "gate": {},
    }

    results["per_country"] = per_country_metrics(scored, paired, cfg)

    results["pooled"] = point_metrics(scored["actual"].to_numpy(),
                                      scored["forecast_value"].to_numpy())
    for cohort, g in scored.groupby("cohort"):
        results["cohorts"][cohort] = point_metrics(g["actual"].to_numpy(),
                                                   g["forecast_value"].to_numpy())
    for cohort in ("pre_fix", "post_fix"):
        results["cohorts"].setdefault(cohort, {"n": 0})

    per_vintage = []
    for gen, g in paired.groupby("generated_at"):
        sub = g.dropna(subset=["actual"])
        row = {"generated_at": str(gen),
               "cohort": "pre_fix" if gen < cfg.cohort_split else "post_fix",
               "rows": int(len(g)), "pairs": int(len(sub))}
        if len(sub):
            row.update({k: v for k, v in point_metrics(
                sub["actual"].to_numpy(), sub["forecast_value"].to_numpy()).items()
                if k in ("bias_mw", "mae_mw", "wape_pct")})
        per_vintage.append(row)
    results["per_vintage"] = sorted(per_vintage, key=lambda r: r["generated_at"])

    results["cuts"] = {
        "country_x_horizon": _cut_metrics(scored, ["country_code", "horizon_bucket"]),
        "country_x_hour": _cut_metrics(scored, ["country_code", "hour"]),
        "country_x_weekday": _cut_metrics(scored, ["country_code", "weekday"]),
    }

    worst = scored.assign(abs_err=(scored["forecast_value"] - scored["actual"]).abs()) \
                  .nlargest(cfg.top_misses, "abs_err")
    results["case_studies"] = [
        {"country": r.country_code, "target_ts": str(r.target_ts),
         "generated_at": str(r.generated_at), "forecast_mw": round(r.forecast_value, 1),
         "actual_mw": round(r.actual, 1), "error_mw": round(r.forecast_value - r.actual, 1)}
        for r in worst.itertuples()]

    results["backtest_vs_live"] = _backtest_vs_live(results, scored, cfg)
    results["gate_scope"] = build_gate_scope(scored, paired, cfg)
    results["gate"] = promotion_gate(results, cfg)
    return results


def _backtest_vs_live(results: dict, scored: pd.DataFrame, cfg: EvalConfig) -> list[dict]:
    """The B3 credibility check, kept in every report: live MAE against the
    W01-W12 serve-faithful backtest, per backtest country. A model whose live
    ratio sits far above 1 is claiming skill its serving does not deliver —
    the ABL-24 signature. The post-fix column accrues as actuals arrive."""
    if not cfg.reference_backtest or not Path(cfg.reference_backtest).exists():
        return []
    try:
        ref = _backtest_mae(cfg.reference_backtest)
    except (OSError, ValueError, KeyError):
        return []
    rows = []
    for country in sorted(ref):
        m = results["per_country"].get(country, {})
        if not m.get("n"):
            continue
        row = {"country": country, "backtest_mae_mw": ref[country],
               "live_mae_mw": m["mae_mw"],
               "live_over_backtest": m["mae_mw"] / ref[country] if ref[country] else None}
        post = scored[(scored["country_code"] == country) & (scored["cohort"] == "post_fix")]
        if len(post):
            pm = float(np.mean(np.abs(post["forecast_value"] - post["actual"])))
            row["post_fix_mae_mw"] = pm
            row["post_fix_over_backtest"] = pm / ref[country] if ref[country] else None
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Promotion gate (ABL-24 plan, C3 — pre-registered)
# ---------------------------------------------------------------------------

def promotion_gate(results: dict, cfg: EvalConfig) -> dict:
    """Score the eight pre-registered criteria of the ABL-24 plan Rev 3 §4.

    Reads `results["gate_scope"]` — the vintage-windowed view — never the full
    `per_country` table, which mixes in pre-fix vintages (ABL-72 G1).
    """
    scope = results.get("gate_scope") or {}
    per_country = scope.get("per_country", {})
    checks: dict = {}

    # G2 (plan Rev 3:54). Fails closed: no scope, or a scope with too few
    # vintages, is a FAIL, not an un-evaluable check. `meta.vintages` was
    # reported and never gated on, which is how a 6-vintage read could have
    # produced a PASS.
    n_vint = scope.get("vintages")
    checks["min_live_shadow_vintages"] = {
        "pass": n_vint is not None and n_vint >= GATE_MIN_LIVE_VINTAGES,
        "detail": (f"{n_vint} live shadow vintages in the gate window "
                   f"(need >= {GATE_MIN_LIVE_VINTAGES})"
                   if n_vint is not None else
                   "no gate scope computed — cannot count vintages")}

    # G3 (plan Rev 3:55). Exclude by name and say so. A zone that is excluded
    # only because it happens to have no paired actuals silently re-enters the
    # moment it publishes one, and then fails the gate on thin data.
    present = {c: per_country.get(c, {}) for c in GATE_EXCLUDED_COUNTRIES}
    had_data = sorted(c for c, m in present.items() if m.get("n", 0) > 0)
    measured = {c: m for c, m in per_country.items()
                if m.get("n", 0) > 0 and c not in GATE_EXCLUDED_COUNTRIES}
    checks["excluded_zones_LU_GR"] = {
        "pass": not (set(measured) & set(GATE_EXCLUDED_COUNTRIES)),
        "detail": (f"excluded by name: {', '.join(sorted(GATE_EXCLUDED_COUNTRIES))}"
                   + (f" — {', '.join(had_data)} had scored pairs in this window "
                      f"and {'was' if len(had_data) == 1 else 'were'} still excluded"
                      if had_data else " — neither had scored pairs in this window")
                   + f"; {len(measured)} zones gated"),
        "excluded": {c: GATE_EXCLUDED_COUNTRIES[c] for c in sorted(GATE_EXCLUDED_COUNTRIES)},
        "excluded_with_data": had_data}

    beat = [c for c, m in measured.items()
            if m.get("baseline_ensemble_mae_mw") and m["mae_mw"] < m["baseline_ensemble_mae_mw"]]
    evaluable = [c for c, m in measured.items() if m.get("baseline_ensemble_mae_mw")]
    frac = len(beat) / len(evaluable) if evaluable else None
    checks["beat_baseline_ensemble_80pct"] = {
        "pass": frac is not None and frac >= GATE_BASELINE_COUNTRY_FRAC,
        "detail": f"beats ensemble MAE in {len(beat)}/{len(evaluable)} countries"
                  + (f" ({100 * frac:.0f}%)" if frac is not None else ""),
        "countries_failing": sorted(set(evaluable) - set(beat)),
    }

    bias_fail = [c for c, m in measured.items()
                 if m["mean_abs_actual_mw"] > 0
                 and abs(m["bias_mw"]) >= GATE_BIAS_FRAC * m["mean_abs_actual_mw"]]
    checks["bias_under_5pct_per_country"] = {
        "pass": not bias_fail, "countries_failing": sorted(bias_fail),
        "detail": f"{len(measured) - len(bias_fail)}/{len(measured)} countries within "
                  f"|bias| < {GATE_BIAS_FRAC:.0%} of mean |net position|"}

    lo, hi = GATE_SLOPE_RANGE
    slope_fail = [c for c, m in measured.items()
                  if m.get("slope") is None or not (lo <= m["slope"] <= hi)]
    checks["slope_in_range_per_country"] = {
        "pass": not slope_fail, "countries_failing": sorted(slope_fail),
        "detail": f"{len(measured) - len(slope_fail)}/{len(measured)} countries with "
                  f"slope in [{lo}, {hi}]"}

    clo, chi = GATE_COVERAGE_RANGE
    cov_vals = {c: m.get("coverage_10_90_pct") for c, m in measured.items()}
    cov_fail = [c for c, v in cov_vals.items() if v is None or not (clo <= v <= chi)]
    checks["coverage_10_90_in_band_per_country"] = {
        "pass": not cov_fail, "countries_failing": sorted(cov_fail),
        "detail": f"{len(measured) - len(cov_fail)}/{len(measured)} countries with "
                  f"10-90 coverage in [{clo:.0f}, {chi:.0f}]%"}

    checks["no_regression_W01_W12"] = _backtest_regression_check(cfg)

    checks["serve_faithful_inputs_verified"] = {
        "pass": bool(cfg.serve_faithful_verified),
        "detail": "attested via --serve-faithful-verified" if cfg.serve_faithful_verified
                  else "not attested — requires a manual serve-parity check "
                       "(bit-reproduce a live vintage, as ABL-28 did)"}

    # The verdict must not be able to say PASS while a pre-registered criterion
    # is absent or un-evaluable. It used to span "only evaluable checks", so the
    # two criteria that were never implemented could not fail — they simply were
    # not there (ABL-72 G2). PASS now requires all eight present and true.
    missing = [n for n in PRE_REGISTERED_CHECKS if n not in checks]
    failed = sorted(n for n, c in checks.items() if c.get("pass") is False)
    unevaluable = sorted(n for n, c in checks.items() if c.get("pass") is None)
    if failed:
        verdict = "FAIL"
    elif missing or unevaluable:
        verdict = "INCOMPLETE"
    else:
        verdict = "PASS"

    note = (f"{len(checks)}/{len(PRE_REGISTERED_CHECKS)} pre-registered criteria scored; "
            f"PASS requires all {len(PRE_REGISTERED_CHECKS)} present and passing")
    if missing:
        note += f". MISSING: {', '.join(missing)}"
    if unevaluable:
        note += f". Not evaluable: {', '.join(unevaluable)}"
    return {"checks": checks, "verdict": verdict, "note": note,
            "criteria_missing": missing, "criteria_failed": failed,
            "criteria_unevaluable": unevaluable,
            "gate_vintage_start": (results.get("gate_scope") or {}).get("vintage_start"),
            "gate_vintage_end": (results.get("gate_scope") or {}).get("vintage_end"),
            "gate_vintages": (results.get("gate_scope") or {}).get("vintages")}


def _backtest_regression_check(cfg: EvalConfig) -> dict:
    if not cfg.candidate_backtest:
        return {"pass": None,
                "detail": "not evaluable — no candidate W01-W12 backtest supplied "
                          "(--candidate-backtest); required before any promotion"}
    try:
        cand = _backtest_mae(cfg.candidate_backtest)
        ref = _backtest_mae(cfg.reference_backtest) if cfg.reference_backtest else {}
    except (OSError, ValueError, KeyError) as e:
        return {"pass": False, "detail": f"backtest JSON unreadable: {e}"}
    if not ref:
        return {"pass": None, "detail": "no reference backtest to compare against"}
    worse = {c: (cand[c], ref[c]) for c in ref if c in cand and cand[c] > ref[c] * 1.0}
    return {"pass": not worse,
            "detail": ("no country regresses vs reference W01-W12 MAE" if not worse
                       else f"regressions: {', '.join(f'{c} {v[1]:.0f}->{v[0]:.0f}' for c, v in sorted(worse.items()))}")}


def _backtest_mae(path: str) -> dict:
    """Mean W01-W12 MAE per country from a compare_experiments-style JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    model = next(iter(data.values()))  # single-model files: {'V010': {...}}
    out = {}
    for country, types in model.items():
        weeks = types.get("net_position", {})
        maes = [w["mae"] for w in weeks.values() if isinstance(w, dict) and "mae" in w]
        if maes:
            out[country] = float(np.mean(maes))
    return out


# ---------------------------------------------------------------------------
# Multi-model comparison (ABL-72 G4)
# ---------------------------------------------------------------------------

def load_vintage_span(cfg: EvalConfig, model_name: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """(earliest, latest) generated_at stored for one model, across both DBs."""
    stamps = []
    for path in ([cfg.sidecar_db] if cfg.sidecar_db else []) + [cfg.replica_db]:
        con = _ro_connect(path)
        try:
            row = con.execute(
                """SELECT MIN(generated_at), MAX(generated_at) FROM forecasts
                   WHERE forecast_type = 'net_position' AND model_name = ?""",
                (model_name,)).fetchone()
        finally:
            con.close()
        if row and row[0]:
            stamps += [pd.Timestamp(row[0]), pd.Timestamp(row[1])]
    return (min(stamps), max(stamps)) if stamps else None


def common_vintage_window(cfg: EvalConfig, model_names: list[str]
                          ) -> tuple[pd.Timestamp | None, pd.Timestamp | None, dict]:
    """The widest [start, end) every listed model actually covers.

    A challenger that only started shadowing last week must not be compared to
    a champion over the champion's longer history — the overlap is the only
    window where the comparison means anything. Floored at `cohort_split` so the
    champion's pre-fix vintages can never re-enter through this path.
    """
    spans = {m: load_vintage_span(cfg, m) for m in model_names}
    found = {m: s for m, s in spans.items() if s}
    missing = sorted(set(model_names) - set(found))
    if not found:
        return None, None, {"spans": {}, "models_with_no_vintages": missing}
    start = max([s[0] for s in found.values()] + [cfg.cohort_split])
    end = min(s[1] for s in found.values()) + pd.Timedelta(seconds=1)  # inclusive of the last
    detail = {"spans": {m: [str(s[0]), str(s[1])] for m, s in found.items()},
              "models_with_no_vintages": missing}
    return start, end, detail


def compare_models(cfg: EvalConfig, model_names: list[str]) -> dict:
    """Score several models over one identical vintage window (ABL-72 G4).

    The C2c deliverable is a single table with a column per candidate. The
    load-bearing property is that every column is measured over the *same*
    vintages: an unequal window is exactly the contamination G1 fixes, moved
    from one model's history into the comparison between models. So the window
    is resolved once, stamped into every child config, and asserted afterwards.
    """
    if cfg.gate_vintage_start is not None or cfg.gate_vintage_end is not None:
        start, end = resolve_gate_vintage_window(cfg)
        window_detail = {"source": "explicit (--gate-vintage-start/--gate-vintage-end)"}
    else:
        start, end, window_detail = common_vintage_window(cfg, model_names)
        window_detail["source"] = "intersection of the models' stored vintage spans"
        if start is None:
            return {"error": "no stored net_position vintages for any of "
                             + ", ".join(model_names)}

    per_model, errors = {}, {}
    for name in model_names:
        child = replace(cfg, model_name=name,
                        gate_vintage_start=start, gate_vintage_end=end)
        res = evaluate(child)
        if "error" in res:
            errors[name] = res["error"]
        else:
            per_model[name] = res

    # The property this whole function exists for: one window, every column.
    windows = {n: (r["gate_scope"]["vintage_start"], r["gate_scope"]["vintage_end"])
               for n, r in per_model.items()}
    if len(set(windows.values())) > 1:
        raise AssertionError(f"models were scored over different windows: {windows}")

    countries = sorted({c for r in per_model.values()
                        for c, m in r["gate_scope"]["per_country"].items()
                        if m.get("n", 0) > 0 and c not in GATE_EXCLUDED_COUNTRIES})
    return {
        "models": model_names,
        "window": {"vintage_start": str(start),
                   "vintage_end": str(end) if end is not None else None,
                   **window_detail},
        "vintages_per_model": {n: r["gate_scope"]["vintages"] for n, r in per_model.items()},
        "pairs_per_model": {n: r["gate_scope"]["pairs_scored"] for n, r in per_model.items()},
        "verdict_per_model": {n: r["gate"]["verdict"] for n, r in per_model.items()},
        "countries": countries,
        "per_model": per_model,
        "errors": errors,
    }


def render_comparison_markdown(cmp: dict, generated_at: str) -> str:
    if "error" in cmp:
        return f"# Net-position model comparison\n\n**{cmp['error']}**\n"
    models, w = cmp["models"], cmp["window"]
    scored = [m for m in models if m in cmp["per_model"]]
    lines = [
        "# Net-position model comparison",
        "",
        f"**Generated:** {generated_at} · **Models:** " + ", ".join(f"`{m}`" for m in models),
        "",
        f"**Identical vintage window:** {w['vintage_start'][:16]} → "
        f"{(w['vintage_end'] or 'open')[:16]} — {w['source']}.",
        "",
        "Every column below is measured over this one window. That is the whole "
        "point of the table: comparing a challenger's recent vintages against a "
        "champion's longer, partly pre-fix history is the ABL-72 G1 defect moved "
        "between models instead of within one.",
        ""]
    if cmp["errors"]:
        lines += ["**Not scored:** "
                  + "; ".join(f"`{m}` — {e}" for m, e in sorted(cmp["errors"].items())), ""]

    lines += ["## Coverage and gate verdict", "",
              "| model | vintages in window | pairs | gate verdict |",
              "|---|---:|---:|---|"]
    for m in scored:
        lines.append(f"| `{m}` | {cmp['vintages_per_model'][m]} | "
                     f"{cmp['pairs_per_model'][m]:,} | {cmp['verdict_per_model'][m]} |")
    lines += ["", "Unequal vintage counts inside one window are real and are printed "
                  "rather than smoothed: a model that shadowed fewer days is measured "
                  "on fewer days, and the reader needs to see that before the MAEs.", ""]

    for label, key, nd in [("MAE (MW)", "mae_mw", 0), ("slope", "slope", 2),
                           ("bias (MW)", "bias_mw", 0)]:
        lines += [f"## {label} by country", "",
                  "| country | " + " | ".join(f"`{m}`" for m in scored) + " |",
                  "|---" * (len(scored) + 1) + "|"]
        for c in cmp["countries"]:
            cells = []
            for m in scored:
                metrics = cmp["per_model"][m]["gate_scope"]["per_country"].get(c, {})
                cells.append(_fmt(metrics.get(key), nd) if metrics.get("n") else "—")
            lines.append(f"| {c} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["---", "",
              "**Reading this.** `—` means that model scored no pairs for that country "
              "in this window — not a zero error. Excluded by name from every column: "
              + ", ".join(f"{c} ({r.split('—')[0].strip()})"
                          for c, r in sorted(GATE_EXCLUDED_COUNTRIES.items())) + ".", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _fmt(v, nd=0):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:,.{nd}f}"


def render_markdown(results: dict, generated_at: str) -> str:
    if "error" in results:
        return f"# Net-position evaluation\n\n**{results['error']}**\n"
    meta, lines = results["meta"], []
    lines += [
        "# Net-position forecast evaluation",
        "",
        f"**Model:** `{meta['model']}` · **Generated:** {generated_at} · "
        f"**Pairs:** {meta['pairs_scored']:,} over {meta['vintages']} vintages "
        f"({meta['vintage_span'][0][:16]} → {meta['vintage_span'][1][:16]})",
        f"**Targets:** {meta['target_window'][0] if meta['target_window'] else '—'} → "
        f"{meta['target_window'][1] if meta['target_window'] else '—'} · "
        f"**Actuals through:** {meta['actuals_max_ts']} · "
        f"**Unpaired rows (no actual yet):** {meta['rows_unpaired']:,}",
        "",
        f"Sidecar vs prod-pushed overlap max |Δ|: "
        f"{_fmt(meta['sidecar_vs_pushed_max_abs_diff_mw'], 3)} MW",
        "",
        "## Promotion gate (pre-registered, ABL-24 C3)",
        "",
        f"**Verdict: {results['gate']['verdict']}**",
        ""]
    scope = results.get("gate_scope") or {}
    if scope:
        lines += [
            f"**Gate vintage window:** {str(scope['vintage_start'])[:16]} → "
            f"{str(scope['vintage_end'])[:16] if scope['vintage_end'] else 'open'} · "
            f"**{scope['vintages']} vintages** · {scope['pairs_scored']:,} pairs · "
            f"{scope['countries_measured']} zones with pairs",
            "",
            "The gate scores this window only. The tables below cover every stored "
            "vintage, including the pre-fix ones — scoring the gate on those measured "
            "the champion at MAE 1,439 MW / slope 0.26 where the serving model is "
            "553 MW / 0.90, a 2.60x handicap in a challenger's favour (ABL-72).",
            ""]
    lines += ["| check | pass | detail |", "|---|---|---|"]
    for name in PRE_REGISTERED_CHECKS:
        c = results["gate"]["checks"].get(name)
        if c is None:
            lines.append(f"| {name} | ❌ | **NOT IMPLEMENTED** — pre-registered "
                         f"criterion absent from this gate |")
            continue
        mark = "—" if c["pass"] is None else ("✅" if c["pass"] else "❌")
        fails = c.get("countries_failing")
        extra = f" (failing: {', '.join(fails)})" if fails else ""
        lines.append(f"| {name} | {mark} | {c['detail']}{extra} |")
    for name, c in results["gate"]["checks"].items():
        if name not in PRE_REGISTERED_CHECKS:   # an added, non-pre-registered check
            mark = "—" if c["pass"] is None else ("✅" if c["pass"] else "❌")
            lines.append(f"| {name} (not pre-registered) | {mark} | {c['detail']} |")
    lines += ["", f"_{results['gate']['note']}._"]

    lines += ["", "## Pooled and cohort view", "",
              "| cohort | n | bias MW | MAE MW | RMSE MW | WAPE | slope | sd ratio | corr |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, m in [("all", results["pooled"]),
                     ("pre-fix (zero-padded context)", results["cohorts"].get("pre_fix", {})),
                     ("post-fix", results["cohorts"].get("post_fix", {}))]:
        if m.get("n"):
            lines.append(f"| {label} | {m['n']:,} | {_fmt(m['bias_mw'])} | {_fmt(m['mae_mw'])} "
                         f"| {_fmt(m['rmse_mw'])} | {_fmt(m['wape_pct'], 1)}% "
                         f"| {_fmt(m['slope'], 2)} | {_fmt(m['sd_ratio'], 2)} | {_fmt(m['corr'], 2)} |")
        else:
            lines.append(f"| {label} | 0 | — | — | — | — | — | — | — | (no scored pairs)")
    lines += ["", f"Cohort split: vintages generated before {meta['cohort_split_utc']} UTC "
                  "ran the pre-1c5a24f zero-padded context (ABL-28)."]

    lines += ["", "## Per country", "",
              "| country | n | bias MW | MAE MW | WAPE | slope | sd ratio | corr | "
              "10-90 cov | pers. MAE | clim. MAE | ens. MAE | skill vs ens. |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for country in sorted(results["per_country"]):
        m = results["per_country"][country]
        if not m.get("n"):
            lines.append(f"| {country} | 0 | — | — | — | — | — | — | — | — | — | — | "
                         f"no paired actuals ({m.get('rows_unpaired', 0)} rows unpaired) |")
            continue
        lines.append(
            f"| {country} | {m['n']:,} | {_fmt(m['bias_mw'])} | {_fmt(m['mae_mw'])} "
            f"| {_fmt(m['wape_pct'], 1)}% | {_fmt(m.get('slope'), 2)} | {_fmt(m.get('sd_ratio'), 2)} "
            f"| {_fmt(m.get('corr'), 2)} | {_fmt(m.get('coverage_10_90_pct'), 1)}% "
            f"| {_fmt(m.get('persistence_mae_mw'))} | {_fmt(m.get('climatology_mae_mw'))} "
            f"| {_fmt(m.get('baseline_ensemble_mae_mw'))} "
            f"| {_fmt(m.get('skill_vs_baseline_ensemble_pct'), 1)}% |")

    lines += ["", "## Error decomposition (share of MSE a correction of each shape would recover)",
              "", "| country | static bias | affine (scale) | diurnal | residual | resid AR lag-1 | lag-24 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for country in sorted(results["per_country"]):
        m = results["per_country"][country]
        d = m.get("decomposition", {})
        if not d or "frac_static_bias" not in d:
            continue
        ar = m.get("residual_autocorr", {})
        lines.append(f"| {country} | {d['frac_static_bias']:.1%} | {d['frac_affine']:.1%} "
                     f"| {d['frac_diurnal']:.1%} | {d['frac_residual']:.1%} "
                     f"| {_fmt(ar.get('lag1'), 2)} | {_fmt(ar.get('lag24'), 2)} |")

    lines += ["", "## Per vintage (accumulating; unpaired = actuals not yet published)", "",
              "| generated_at | cohort | rows | pairs | bias MW | MAE MW | WAPE |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for r in results["per_vintage"]:
        lines.append(f"| {r['generated_at'][:16]} | {r['cohort']} | {r['rows']} | {r['pairs']} "
                     f"| {_fmt(r.get('bias_mw'))} | {_fmt(r.get('mae_mw'))} "
                     f"| {_fmt(r.get('wape_pct'), 1)}{'%' if r.get('wape_pct') is not None else ''} |")

    if results.get("backtest_vs_live"):
        lines += ["", "## Backtest (W01-W12 serve-faithful) vs live — credibility check", "",
                  "| country | backtest MAE | live MAE (all) | ratio | post-fix MAE | post-fix ratio |",
                  "|---|---:|---:|---:|---:|---:|"]
        for r in results["backtest_vs_live"]:
            lines.append(f"| {r['country']} | {_fmt(r['backtest_mae_mw'])} | {_fmt(r['live_mae_mw'])} "
                         f"| {_fmt(r['live_over_backtest'], 2)} | {_fmt(r.get('post_fix_mae_mw'))} "
                         f"| {_fmt(r.get('post_fix_over_backtest'), 2)} |")
        lines += ["", "A live/backtest ratio well above 1 means the serving path is not "
                      "delivering the skill the backtest claimed (ABL-24's gap); the "
                      "post-fix columns accumulate as actuals arrive."]

    lines += ["", f"## Largest misses (top {len(results['case_studies'])})", "",
              "| country | target | vintage | forecast MW | actual MW | error MW |",
              "|---|---|---|---:|---:|---:|"]
    for cs in results["case_studies"]:
        lines.append(f"| {cs['country']} | {cs['target_ts'][:16]} | {cs['generated_at'][:16]} "
                     f"| {_fmt(cs['forecast_mw'])} | {_fmt(cs['actual_mw'])} | {_fmt(cs['error_mw'])} |")

    lines += ["", "---", "",
              "**Conventions.** bias = mean(forecast − actual). slope = OLS of forecast on actual "
              "(calibrated ≈ 1; < 1 = shrinkage toward zero — the ABL-24 signature). "
              "Every vintage-target pair scores; AR diagnostics use the latest vintage only. "
              "Baselines are serve-faithful: they read only actuals available at each vintage's "
              "day-ahead publication cutoff (as_of = run-day 22:00 UTC for the 06:00Z schedule). "
              "Ensemble = pointwise mean of persistence and climatology (V012 will formalize). "
              "Pooled rows mix country means — per-country numbers are the ones that gate.", ""]
    return "\n".join(lines)
