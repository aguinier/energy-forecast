"""ABL-650 -- diagnose and recalibrate the net-position p10-p90 band.

Two jobs, in order.

1. **Which defect is this?** The issue could be (a) genuinely mis-calibrated
   quantiles or (b) a correct band drawn or labelled wrongly. This script emits
   the marginal PIT over all nine stored levels plus the four nested central
   intervals; a labelling defect moves one band, a scale defect moves all four
   monotonically.

2. **Fit a calibration out-of-sample.** Split-conformal, two multipliers per
   zone, anchored at the stored q50 so the served point forecast cannot move:

       q'_t = q50 - s_lo * (q50 - q_t)      for t < 0.5
       q'_t = q50 + s_hi * (q_t - q50)      for t > 0.5
       q'_50 = q50                          exactly, by construction

   `s_lo` is the conformal quantile of the normalised lower deviation
   (q50 - a) / (q50 - q10) at level 1 - alpha_lo, so a fraction alpha_lo of the
   fit rows fall below the calibrated p10; `s_hi` mirrors it above p90. Two
   one-sided fits rather than one symmetric one, because the champion's PIT is
   shifted as well as narrow -- a symmetric widening would hit 80% total while
   leaving the tails unbalanced.

   The map is piecewise linear and increasing on each side of a fixed q50, so it
   cannot create a quantile crossing that was not already there.

Everything here is read-only against both databases. Nothing is written to the
replica, the sidecar, or any forecast table -- the output is a JSON evidence
record and a markdown pack.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.net_position import (  # noqa: E402
    EvalConfig,
    GATE_COUNTRIES,
    QUANTILE_LEVELS,
    load_actuals,
    load_forecasts,
)
from src.quantile_calibration import (  # noqa: E402
    QCOLS,
    ZoneCalibration,
    apply_zone_calibration,
    fit_zone_calibration,
)

CENTRAL_BANDS = ((0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6))

# Whose calibration is already inside this model's input. V016 reads the
# champion's stored band and maps it affinely, so it inherits whatever widening
# the champion carries.
UPSTREAM = {"chronos-2-V016": "chronos-2-V010"}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def paired_rows(model: str, replica_db: str, sidecar_db: str) -> pd.DataFrame:
    """Every (vintage, zone, target hour) with an actual, gated zones only."""
    cfg = EvalConfig(replica_db=replica_db, sidecar_db=sidecar_db, model_name=model)
    fc = load_forecasts(cfg)
    if fc.empty:
        return fc
    ac = load_actuals(cfg).rename(columns={"ts": "target_ts"})
    fc = fc[fc["country_code"].isin(GATE_COUNTRIES)]
    df = fc.merge(ac, on=["country_code", "target_ts"], how="inner")
    missing = [c for c in QCOLS if c not in df.columns]
    for c in missing:
        df[c] = np.nan
    return df.dropna(subset=list(QCOLS)).reset_index(drop=True)


def vintage_slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Rows whose *vintage* falls in [start, end). The window is on
    `generated_at`, never on the target hour: an out-of-sample split has to
    separate the runs, not the hours they happen to cover."""
    return df[(df["generated_at"] >= pd.Timestamp(start))
              & (df["generated_at"] < pd.Timestamp(end))]


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------

def marginal_pit(df: pd.DataFrame) -> list[dict]:
    out = []
    for q, col in zip(QUANTILE_LEVELS, QCOLS):
        emp = float((df["actual"] <= df[col]).mean())
        out.append({"nominal": q, "empirical_below": round(emp, 4),
                    "gap_pp": round(100 * (emp - q), 2)})
    return out


def central_coverage(df: pd.DataFrame) -> list[dict]:
    out = []
    for lo, hi in CENTRAL_BANDS:
        lc, hc = f"q{int(lo * 100)}", f"q{int(hi * 100)}"
        inside = (df["actual"] >= df[lc]) & (df["actual"] <= df[hc])
        out.append({"band": f"{lc}-{hc}", "nominal_pct": round(100 * (hi - lo), 1),
                    "measured_pct": round(100 * float(inside.mean()), 2),
                    "mean_width_mw": round(float((df[hc] - df[lc]).mean()), 1)})
    return out


def per_zone_band(df: pd.DataFrame) -> pd.DataFrame:
    g = df.copy()
    g["inside"] = (g["actual"] >= g["q10"]) & (g["actual"] <= g["q90"])
    g["below"] = g["actual"] < g["q10"]
    g["above"] = g["actual"] > g["q90"]
    g["width"] = g["q90"] - g["q10"]
    g["abserr"] = (g["forecast_value"] - g["actual"]).abs()
    out = g.groupby("country_code").agg(
        n=("inside", "size"),
        coverage_pct=("inside", lambda s: round(100 * float(s.mean()), 1)),
        below_pct=("below", lambda s: round(100 * float(s.mean()), 1)),
        above_pct=("above", lambda s: round(100 * float(s.mean()), 1)),
        mean_width_mw=("width", lambda s: round(float(s.mean()), 1)),
        mae_mw=("abserr", lambda s: round(float(s.mean()), 1)),
        mean_abs_np=("actual", lambda s: round(float(s.abs().mean()), 1)),
    )
    out["width_over_mae"] = (out["mean_width_mw"] / out["mae_mw"]).round(2)
    return out


def vintage_block_ci(df: pd.DataFrame, n_boot: int, seed: int,
                     lo_col: str = "q10", hi_col: str = "q90") -> dict:
    """Percentile CI on 10-90 coverage, resampling whole *vintages*.

    An hour is not an independent draw: one vintage forecasts 24 consecutive
    hours from one context, and a run that misreads the day misses most of it.
    Treating 480 hours as 480 trials makes the binomial interval about five
    times too tight and turns ordinary run-to-run variation into a defect.
    The block is the vintage, so the sample size that matters is the number of
    runs, not the number of rows.
    """
    rng = np.random.default_rng(seed)
    keys = df["generated_at"].to_numpy()
    uniq = np.unique(keys)
    groups = [np.flatnonzero(keys == v) for v in uniq]
    inside = ((df["actual"].to_numpy() >= df[lo_col].to_numpy())
              & (df["actual"].to_numpy() <= df[hi_col].to_numpy()))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        draws[b] = inside[idx].mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"coverage_pct": round(100 * float(inside.mean()), 2),
            "ci95_pct": [round(100 * float(lo), 2), round(100 * float(hi), 2)],
            "blocks": int(len(groups)), "rows": int(len(df)),
            "outside_80": not (lo <= 0.80 <= hi)}


def pinball(df: pd.DataFrame) -> float:
    """Mean pinball loss over the nine stored levels -- the proper score the
    band is not allowed to wreck while chasing coverage."""
    losses = []
    a = df["actual"].to_numpy()
    for q, col in zip(QUANTILE_LEVELS, QCOLS):
        d = a - df[col].to_numpy()
        losses.append(np.mean(np.where(d >= 0, q * d, (q - 1) * d)))
    return float(np.mean(losses))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def build_variants(fit_df: pd.DataFrame, zones, alpha: float, shrink) -> dict:
    """The candidate calibrations, all fitted on `fit_df` alone.

    `none` is the status quo and the thing every other variant has to beat.
    `pooled` fits one multiplier pair on every zone's rows at once; `per_zone`
    fits nineteen independent pairs; `shrink_w` interpolates between them. The
    three are carried side by side rather than one being chosen up front,
    because which one generalises is the question, not an assumption.
    """
    pooled_fit = fit_zone_calibration(fit_df, ["__pooled__"], alpha_lo=alpha,
                                      alpha_hi=alpha, pooled=True)["__pooled__"]
    per_zone = fit_zone_calibration(fit_df, zones, alpha_lo=alpha, alpha_hi=alpha)
    out = {"none": None,
           "pooled": {z: pooled_fit for z in zones},
           "per_zone": per_zone}
    for w in shrink:
        out[f"shrink_{w:g}"] = {
            z: ZoneCalibration(
                s_lo=w * c.s_lo + (1 - w) * pooled_fit.s_lo,
                s_hi=w * c.s_hi + (1 - w) * pooled_fit.s_hi,
                n_fit=c.n_fit, alpha_lo=alpha, alpha_hi=alpha)
            for z, c in per_zone.items()}
    return out, pooled_fit, per_zone


def score_variants(ev_df: pd.DataFrame, variants: dict) -> dict:
    scored = {}
    for name, cal in variants.items():
        out = ev_df if cal is None else apply_zone_calibration(ev_df, cal)
        pz = per_zone_band(out)
        scored[name] = {
            "pooled_coverage_pct": round(
                100 * float(((out["actual"] >= out["q10"])
                             & (out["actual"] <= out["q90"])).mean()), 2),
            "zones_in_gate_band_75_85": int(
                ((pz["coverage_pct"] >= 75) & (pz["coverage_pct"] <= 85)).sum()),
            "zones_total": len(pz),
            "mean_abs_coverage_error_pp": round(
                float((pz["coverage_pct"] - 80.0).abs().mean()), 2),
            "mean_width_mw": round(float((out["q90"] - out["q10"]).mean()), 1),
            "median_width_over_mae": round(float(pz["width_over_mae"].median()), 2),
            "mean_pinball_mw": round(pinball(out), 2),
            "median_max_abs_delta_mw": float(
                (out["q50"].to_numpy() - ev_df["q50"].to_numpy()).__abs__().max()),
            "point_max_abs_delta_mw": float(
                (out["forecast_value"].to_numpy()
                 - ev_df["forecast_value"].to_numpy()).__abs__().max()),
            "crossings": int((np.diff(out[list(QCOLS)].to_numpy(), axis=1) < 0)
                             .any(axis=1).sum()),
        }
    return scored


def rolling_origin(model: str, df: pd.DataFrame, args) -> list[dict]:
    """Expanding-window folds: every fold fits on vintages strictly earlier
    than the ones it scores.

    One split is one draw. With fourteen vintages a side, a variant can win a
    single split on noise; the folds are here so "pooled beats none" has to hold
    more than once before it is reported as a result.
    """
    cuts = [c.strip() for c in args.fold_cuts.split(",") if c.strip()]
    folds = []
    for i in range(len(cuts) - 1):
        fit_df = vintage_slice(df, args.fit_start, cuts[i])
        ev_df = vintage_slice(df, cuts[i], cuts[i + 1])
        if fit_df.empty or ev_df.empty:
            continue
        zones = sorted(set(fit_df["country_code"]) & set(ev_df["country_code"]))
        variants, pooled_fit, per_zone = build_variants(
            fit_df, zones, args.alpha, args.shrink)
        fold = {
            "fold": i + 1,
            "fit_window_vintages": [args.fit_start, cuts[i]],
            "eval_window_vintages": [cuts[i], cuts[i + 1]],
            "fit_rows": int(len(fit_df)), "eval_rows": int(len(ev_df)),
            "fit_vintages": int(fit_df["generated_at"].nunique()),
            "eval_vintages": int(ev_df["generated_at"].nunique()),
            "pooled_multipliers": pooled_fit.as_dict(),
            "scores": score_variants(ev_df, variants),
        }
        # Does a zone's coverage in the fit window predict its coverage in the
        # eval window at all? If it does not, no per-zone calibration fitted on
        # this much data can transfer, whatever its form.
        f_cov = per_zone_band(fit_df)["coverage_pct"]
        e_cov = per_zone_band(ev_df)["coverage_pct"]
        common = f_cov.index.intersection(e_cov.index)
        fold["zone_coverage_persistence"] = {
            "pearson_r": round(float(np.corrcoef(f_cov[common], e_cov[common])[0, 1]), 3),
            "spearman_r": round(float(f_cov[common].corr(e_cov[common], method="spearman")), 3),
            "zones": len(common),
            "fit_range_pct": [float(f_cov[common].min()), float(f_cov[common].max())],
            "eval_range_pct": [float(e_cov[common].min()), float(e_cov[common].max())],
            "max_abs_shift_pp": round(float((e_cov[common] - f_cov[common]).abs().max()), 1),
        }
        folds.append(fold)
    return folds


def evaluate(model: str, df: pd.DataFrame, args) -> dict:
    fit_df = vintage_slice(df, args.fit_start, args.fit_end)
    ev_df = vintage_slice(df, args.eval_start, args.eval_end)
    if fit_df.empty or ev_df.empty:
        raise SystemExit(f"{model}: fit or eval window is empty "
                         f"({len(fit_df)} / {len(ev_df)} rows)")

    zones = sorted(set(fit_df["country_code"]) & set(ev_df["country_code"]))
    variant_map, pooled_fit, per_zone = build_variants(
        fit_df, zones, args.alpha, args.shrink)
    result = {
        "model": model,
        "fit_window_vintages": [args.fit_start, args.fit_end],
        "eval_window_vintages": [args.eval_start, args.eval_end],
        "fit_rows": int(len(fit_df)),
        "fit_vintages": int(fit_df["generated_at"].nunique()),
        "eval_rows": int(len(ev_df)),
        "eval_vintages": int(ev_df["generated_at"].nunique()),
        "zones": zones,
        "target_coverage_pct": round(100 * (1 - 2 * args.alpha), 1),
        "multipliers": {
            "pooled": pooled_fit.as_dict(),
            "per_zone": {z: c.as_dict() for z, c in per_zone.items()},
        },
        "rolling_origin_folds": rolling_origin(model, df, args),
        "variants": {},
    }

    for name, cal in variant_map.items():
        out = ev_df if cal is None else apply_zone_calibration(ev_df, cal)
        pz = per_zone_band(out)
        in_band = int(((pz["coverage_pct"] >= 75) & (pz["coverage_pct"] <= 85)).sum())
        result["variants"][name] = {
            "pooled_coverage_pct": round(
                100 * float(((out["actual"] >= out["q10"])
                             & (out["actual"] <= out["q90"])).mean()), 2),
            "zones_in_gate_band_75_85": in_band,
            "zones_total": len(pz),
            "mean_abs_coverage_error_pp": round(
                float((pz["coverage_pct"] - 80.0).abs().mean()), 2),
            "mean_width_mw": round(float((out["q90"] - out["q10"]).mean()), 1),
            "median_width_over_mae": round(float(pz["width_over_mae"].median()), 2),
            "mean_pinball_mw": round(pinball(out), 2),
            "marginal_pit": marginal_pit(out),
            "central_coverage": central_coverage(out),
            "per_zone": pz.reset_index().to_dict("records"),
            # The whole point of the constraint: q50 and the served point row
            # are compared numerically, not asserted.
            "median_max_abs_delta_mw": float(
                (out["q50"] - ev_df["q50"]).abs().max()),
            "point_max_abs_delta_mw": float(
                (out["forecast_value"] - ev_df["forecast_value"]).abs().max()),
            "q50_equals_served_point_rows": int(
                (out["q50"] == out["forecast_value"]).sum()),
            "crossings": int((np.diff(out[list(QCOLS)].to_numpy(), axis=1) < 0)
                             .any(axis=1).sum()),
        }

    # The published gate window, uncalibrated -- the "before" the issue quotes,
    # each zone carrying the vintage-block interval that says whether its miss
    # is a defect or a fortnight of weather.
    gate = vintage_slice(df, args.gate_start, args.gate_end)
    if not gate.empty:
        pz = per_zone_band(gate).reset_index().to_dict("records")
        for rec in pz:
            rec["ci"] = vintage_block_ci(
                gate[gate["country_code"] == rec["country_code"]],
                args.n_boot, args.seed)
        result["gate_window_before"] = {
            "window_vintages": [args.gate_start, args.gate_end],
            "rows": int(len(gate)),
            "vintages": int(gate["generated_at"].nunique()),
            "pooled": vintage_block_ci(gate, args.n_boot, args.seed),
            "marginal_pit": marginal_pit(gate),
            "central_coverage": central_coverage(gate),
            "per_zone": pz,
            "zones_outside_null": sorted(
                r["country_code"] for r in pz if r["ci"]["outside_80"]),
            "band_75_85_attainability": attainability(gate, args),
        }

    # The multipliers to register: the method is validated by the folds above,
    # so the shipped fit uses every post-fix vintage rather than half of them.
    full = vintage_slice(df, args.fit_start, args.eval_end)
    full_pooled = fit_zone_calibration(full, ["__pooled__"], alpha_lo=args.alpha,
                                       alpha_hi=args.alpha, pooled=True)["__pooled__"]
    result["registration_fit"] = {
        "window_vintages": [args.fit_start, args.eval_end],
        "rows": int(len(full)), "vintages": int(full["generated_at"].nunique()),
        "pooled": full_pooled.as_dict(),
    }
    return result


def attainability(gate: pd.DataFrame, args) -> dict:
    """Could a *perfectly calibrated* band pass a per-zone [75, 85]% screen
    on this many vintages?

    Each zone's bootstrap draws are recentred on 80% and the fraction landing
    inside [75, 85] is read off. That fraction is the best a zone could do with
    a true coverage of exactly 80% and this window's run-to-run variability -- a
    ceiling, not an estimate of the current model. Reported because a screen
    whose ceiling is far below 19/19 is measuring window length, not calibration.
    """
    rng = np.random.default_rng(args.seed)
    per_zone_p, ceiling = {}, 1.0
    for zone, sub in gate.groupby("country_code"):
        keys = sub["generated_at"].to_numpy()
        groups = [np.flatnonzero(keys == v) for v in np.unique(keys)]
        inside = ((sub["actual"].to_numpy() >= sub["q10"].to_numpy())
                  & (sub["actual"].to_numpy() <= sub["q90"].to_numpy()))
        draws = np.array([inside[np.concatenate(
            [groups[i] for i in rng.integers(0, len(groups), len(groups))])].mean()
            for _ in range(args.n_boot)])
        centred = draws - draws.mean() + 0.80
        p = float(((centred >= 0.75) & (centred <= 0.85)).mean())
        per_zone_p[zone] = round(p, 3)
        ceiling *= p
    return {"per_zone_p_in_band_if_perfectly_calibrated": per_zone_p,
            "expected_zones_in_band_of_19": round(sum(per_zone_p.values()), 1),
            "p_all_19_in_band": float(f"{ceiling:.3g}"),
            "note": "ceiling under a true coverage of exactly 80%, from each "
                    "zone's own vintage-block variability"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--replica-db", required=True)
    p.add_argument("--sidecar-db", required=True)
    p.add_argument("--models", default="chronos-2-V010,chronos-2-V016",
                   help="comma-separated model_name list (only models that "
                        "emit quantiles)")
    p.add_argument("--fit-start", default="2026-08-05")
    p.add_argument("--fit-end", default="2026-08-19")
    p.add_argument("--eval-start", default="2026-08-19")
    p.add_argument("--eval-end", default="2026-09-02")
    p.add_argument("--gate-start", default="2026-08-07")
    p.add_argument("--gate-end", default="2026-08-27")
    p.add_argument("--alpha", type=float, default=0.10,
                   help="per-tail miscoverage; 0.10 -> a 10-90 band")
    p.add_argument("--fold-cuts",
                   default="2026-08-15,2026-08-21,2026-08-27,2026-09-02",
                   help="expanding-window fold boundaries on generated_at; "
                        "fold i fits [fit-start, cut_i) and scores [cut_i, cut_i+1)")
    p.add_argument("--shrink", default="0.5",
                   help="comma-separated shrinkage weights toward the pooled "
                        "multiplier (1 = per-zone, 0 = pooled)")
    p.add_argument("--write-registration",
                   help="also emit the experiments/ registration file")
    p.add_argument("--n-boot", type=int, default=2000,
                   help="vintage-block bootstrap draws for the coverage CI")
    p.add_argument("--seed", type=int, default=650)
    p.add_argument("--json-out", required=True)
    args = p.parse_args()
    args.shrink = [float(w) for w in args.shrink.split(",") if w.strip()]

    if pd.Timestamp(args.eval_start) < pd.Timestamp(args.fit_end):
        raise SystemExit("eval window must start at or after the fit window ends "
                         "- an overlapping split is not out-of-sample")

    doc = {"issue": "ABL-650", "alpha": args.alpha, "models": {}}
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        df = paired_rows(model, args.replica_db, args.sidecar_db)
        if df.empty:
            doc["models"][model] = {"error": "no paired rows with quantiles"}
            continue
        doc["models"][model] = evaluate(model, df, args)
        v = doc["models"][model]["variants"]
        print(f"{model}: eval n={doc['models'][model]['eval_rows']} "
              f"| coverage none={v['none']['pooled_coverage_pct']}% "
              f"pooled={v['pooled']['pooled_coverage_pct']}% "
              f"per_zone={v['per_zone']['pooled_coverage_pct']}% "
              f"| zones in [75,85] {v['none']['zones_in_gate_band_75_85']} -> "
              f"{v['per_zone']['zones_in_gate_band_75_85']} of {v['none']['zones_total']}")

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out}")

    if args.write_registration:
        write_registration(doc, Path(args.write_registration), args)
        print(f"wrote {args.write_registration}")
    return 0


def write_registration(doc: dict, path: Path, args) -> None:
    """Emit the registration file from the measured fit.

    Derived, never typed: the multipliers in `experiments/` are the ones this
    run measured, and re-running the script is how they are re-derived.

    `upstream` handles the one composition in the field. V016's band is an
    affine map of the champion's, so its half-widths are exactly proportional to
    V010's -- once the champion's band is widened, V016's input is already
    widened by the same factor. The registration therefore carries both the
    total the band needs and the increment this model applies, and the loader
    checks that increment x upstream reproduces the total.
    """
    models = {}
    for name, m in doc["models"].items():
        if "registration_fit" not in m:
            continue
        fit = m["registration_fit"]
        spec = {
            "mode": "pooled",
            "alpha": args.alpha,
            "s_lo_total": fit["pooled"]["s_lo"],
            "s_hi_total": fit["pooled"]["s_hi"],
            "s_lo_applied": fit["pooled"]["s_lo"],
            "s_hi_applied": fit["pooled"]["s_hi"],
            "fit_window": fit["window_vintages"],
            "fit_vintages": fit["vintages"],
            "fit_rows": fit["rows"],
        }
        if name in UPSTREAM:
            spec["upstream"] = UPSTREAM[name]
        models[name] = spec

    for name, up in UPSTREAM.items():
        if name in models and up in models:
            for side in ("lo", "hi"):
                models[name][f"s_{side}_applied"] = round(
                    models[name][f"s_{side}_total"]
                    / models[up][f"s_{side}_applied"], 4)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "issue": "ABL-650",
        "generated_by": "scripts/abl650_band_calibration.py",
        "evidence": args.json_out,
        "protocol": (
            "Split-conformal, two multipliers per model, anchored at the stored "
            "q50. Fitted pooled over every gated zone: ABL-650 measured per-zone "
            "multipliers losing to no calibration at all on held-out vintages "
            "(champion 9 -> 4 zones inside [75, 85]%), because a zone's measured "
            "coverage does not persist window to window (fit-to-eval Pearson r "
            "0.27 / -0.20 / 0.06 over three folds). The pooled form is the only "
            "one that survives out-of-sample."),
        "invariants": [
            "q50 is a fixed point: the served point forecast cannot move",
            "the map is increasing on each side of q50: no new crossings",
            "new rows only; stored quantile history is never rewritten",
        ],
        "models": models,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
